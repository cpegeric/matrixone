// Copyright 2026 Matrix Origin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fileservice

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/util/toml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type countingStorage struct {
	ObjectStorage
	reads atomic.Int64
	bytes atomic.Int64

	mu     sync.Mutex
	starts []int64
}

func (c *countingStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	c.reads.Add(1)
	if min != nil && max != nil {
		c.bytes.Add(*max - *min)
	}
	if min != nil && (max != nil || *min > 0) {
		c.mu.Lock()
		c.starts = append(c.starts, *min)
		c.mu.Unlock()
	}
	return c.ObjectStorage.Read(ctx, key, min, max)
}

func newPrefetchRangeFS(t *testing.T, unit int64, size int) (*S3FS, *countingStorage, []byte) {
	ctx := context.Background()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "s3", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{
		DiskPath: ptrTo(t.TempDir()), DiskCapacity: ptrTo[toml.ByteSize](64 << 20),
		MemoryCapacity: ptrTo[toml.ByteSize](0),
	}, nil, false, false)
	require.NoError(t, err)
	fs.diskCache.prefetchRange = prefetchRangeConfig{on: unit > 0, fixed: unit}
	t.Cleanup(func() { fs.Close(ctx) })
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i*7 + i/1021)
	}
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: int64(len(data)), Data: data}}, Policy: SkipDiskCache | SkipMemoryCache}))
	storage := &countingStorage{ObjectStorage: fs.server.raw}
	fs.server.raw = storage
	return fs, storage, data
}

// readRangeOf reads the range and waits until the prefetch ranges it read are cached.
func readRangeOf(t *testing.T, fs *S3FS, policy Policy, offset, size int64) []byte {
	vec := &IOVector{FilePath: "object", Policy: policy | SkipFullFilePreloads | SkipMemoryCache,
		Entries: []IOEntry{{Offset: offset, Size: size}}}
	require.NoError(t, fs.Read(context.Background(), vec))
	waitObjectsEnd(fs.diskCache, "object")
	flushPublishers(fs)
	data := slices.Clone(vec.Entries[0].Data)
	vec.Release()
	return data
}

// A short range read caches the aligned prefetch range around it, which serves later reads
// within it without object storage: at the object's end the range holds the bytes up to
// the end, a read crossing a unit boundary reads the units it spans as one range, and a
// read wider than the prefetch window reads its range alone.
func TestS3FSPrefetchRangeServesLaterReads(t *testing.T) {
	// reads and bytes are the storage counts after the read, bytes unchecked for -1.
	type step struct{ off, size, reads, bytes int64 }
	for _, tc := range []struct {
		name  string
		size  int
		steps []step
	}{
		{"unit", 64 << 10, []step{{5000, 2, 1, 4096}, {6000, 100, 1, 4096}, {4096, 1, 1, 4096}, {9000, 10, 2, -1}}},
		{"object end", 10000, []step{{9990, 10, 1, -1}, {8200, 100, 1, -1}}},
		{"across units", 64 << 10, []step{{4000, 200, 1, 8192}, {0, 8192, 1, 8192}}},
		{"wider than the window", (prefetchWindowBlocks + 2) * 4096,
			[]step{{100, prefetchWindowBlocks*4096 + 1, 1, prefetchWindowBlocks*4096 + 1}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 4096, tc.size)
			for _, s := range tc.steps {
				require.Equal(t, data[s.off:s.off+s.size], readRangeOf(t, fs, 0, s.off, s.size), s.off)
				require.Equal(t, s.reads, storage.reads.Load(), s.off)
				if s.bytes >= 0 {
					require.Equal(t, s.bytes, storage.bytes.Load(), s.off)
				}
			}
		})
	}
}

// A read near the end of a unit reads from its own block, and the blocks of the unit
// before it in a range of their own.
func TestS3FSPrefetchRangeStartsAtTheNeededBlock(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 256<<10, 512<<10)
	require.Equal(t, data[200<<10:200<<10+10], readRangeOf(t, fs, 0, 200<<10, 10))
	storage.mu.Lock()
	starts := slices.Clone(storage.starts)
	storage.mu.Unlock()
	require.ElementsMatch(t, []int64{192 << 10, 0}, starts)
	require.Equal(t, int64(256<<10), storage.bytes.Load())
	require.Equal(t, data[10:20], readRangeOf(t, fs, 0, 10, 10))
	require.Equal(t, int64(2), storage.reads.Load())
}

// A read whose context is done starts no prefetch range.
func TestS3FSPrefetchRangeSkipsCanceledRead(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 4096, 64<<10)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var opens atomic.Int64
	_, reads, ok, _ := fs.diskCache.prefetchRead(ctx, "object", 0, []IOEntry{{Offset: 5000, Size: 2}}, 5000, 5002, 4096,
		openerSource(func(context.Context, int64, int64) (io.ReadCloser, error) {
			opens.Add(1)
			return nil, context.Canceled
		}))
	fs.diskCache.prefetchObjects.reads.Wait()
	require.False(t, ok)
	require.Zero(t, reads)
	require.Zero(t, opens.Load())
}

// A run of a FilePrefetchObject past the object's end caches nothing and learns the
// object's size; with the size known, a run ends there and is cached.
func TestS3FSFileRunPastObjectEnd(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 4096, 10000)
	fs.diskCache.memoryPrefetchMax = 4096
	require.Equal(t, data[5000:9990], readRangeOf(t, fs, 0, 5000, 4990))
	_, err := os.Stat(fs.diskCache.pathForIOEntry("object", IOEntry{Offset: 4096, Size: 8192}))
	require.True(t, os.IsNotExist(err), "a run past the end publishes no file")
	size, ok := fs.diskCache.prefetched.size(fs.diskCache.objectKey("object"))
	require.True(t, ok)
	require.Equal(t, int64(10000), size)

	fs, storage, data := newPrefetchRangeFS(t, 4096, 10000)
	fs.diskCache.memoryPrefetchMax = 4096
	fs.diskCache.prefetched.store(fs.diskCache.objectKey("object"), 10000)
	require.Equal(t, data[5000:9990], readRangeOf(t, fs, 0, 5000, 4990))
	require.Equal(t, int64(1), storage.reads.Load())
	require.Equal(t, data[4096:10000], readRangeOf(t, fs, 0, 4096, 10000-4096))
	require.Equal(t, int64(1), storage.reads.Load(), "the run ending at the object's end is cached")
}

type blockingStorage struct {
	ObjectStorage
	reads   atomic.Int64
	started chan struct{}
	release chan struct{}
}

func (b *blockingStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	if b.reads.Add(1) == 1 {
		close(b.started)
		<-b.release
	}
	return b.ObjectStorage.Read(ctx, key, min, max)
}

// A closed cache starts no prefetch reads.
func TestS3FSPrefetchRangeAfterClose(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 4096, 64<<10)
	fs.diskCache.closePrefetches(context.Background())
	require.Equal(t, data[5000:5002], readRangeOf(t, fs, 0, 5000, 2))
	require.Equal(t, int64(2), storage.bytes.Load())
}

// A read spanning cached units reads only the units between them, and takes the rest
// from the cache.
func TestS3FSPrefetchRangeMergesCachedUnits(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 4096, 64<<10)
	readRangeOf(t, fs, 0, 4100, 1)
	readRangeOf(t, fs, 0, 12300, 1)
	require.Equal(t, int64(8192), storage.bytes.Load())
	require.Equal(t, data[4096:16384], readRangeOf(t, fs, 0, 4096, 12288))
	require.Equal(t, int64(3), storage.reads.Load())
	require.Equal(t, int64(3*4096), storage.bytes.Load(), "only the middle unit")
	require.Equal(t, data[4000:16000], readRangeOf(t, fs, 0, 4000, 12000))
	require.Equal(t, int64(4), storage.reads.Load(), "only the first unit")
	require.Equal(t, int64(4*4096), storage.bytes.Load())
}

type trackedReader struct {
	ctx   context.Context
	r     io.ReadCloser
	eof   bool
	gate  chan struct{} // closed to let reads past the first chunk through
	first bool
	ended *atomic.Int64 // readers closed after their end
}

func (t *trackedReader) Read(p []byte) (int, error) {
	if t.first && t.gate != nil {
		<-t.gate
		if err := t.ctx.Err(); err != nil {
			return 0, err
		}
	}
	t.first = true
	n, err := t.r.Read(p)
	if err == io.EOF {
		t.eof = true
	}
	return n, err
}

func (t *trackedReader) Close() error {
	if t.eof {
		t.ended.Add(1)
	}
	return t.r.Close()
}

// A prefetch read reads each range to its end, and finishes after the read that started
// it returns and its context ends.
func TestPrefetchReadOutlivesItsReader(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 128<<10, 256<<10)
	d := fs.diskCache
	gate := make(chan struct{})
	var opened, ended atomic.Int64
	open := func(ctx context.Context, start, end int64) (io.ReadCloser, error) {
		opened.Add(1)
		r, err := fs.server.raw.Read(ctx, "object", &start, &end)
		if err != nil {
			return nil, err
		}
		return &trackedReader{ctx: ctx, r: r, gate: gate, ended: &ended}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	entries := []IOEntry{{Offset: 100, Size: 10}}
	content, reads, ok, _ := d.prefetchRead(ctx, "object", SkipFullFilePreloads, entries, 100, 110, 128<<10, openerSource(open))
	require.True(t, ok)
	require.Equal(t, 1, reads)
	require.Equal(t, data[100:110], content)
	cancel()
	close(gate)
	d.prefetchObjects.reads.Wait()
	require.Equal(t, int64(1), opened.Load())
	require.Equal(t, int64(1), ended.Load(), "read to its end")
	r, ok := d.ranges.containing(d.objectKey("object"), 0, 128<<10, nil)
	require.True(t, ok, "cached after its reader's context ended")
	require.Equal(t, int64(128<<10), r.size)
}

// Deleting an object deletes its cached ranges: a read of another offset in a range
// cached before the delete reads the object written after it.
func TestDeleteRemovesCachedRanges(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 4096, 64<<10)
	ctx := context.Background()
	require.Equal(t, data[5000:5002], readRangeOf(t, fs, 0, 5000, 2))
	require.Equal(t, int64(1), storage.reads.Load())
	require.NoError(t, fs.Delete(ctx, "object"))
	require.Empty(t, fs.diskCache.ranges.of(fs.diskCache.objectKey("object")))
	_, err := fs.diskCache.ranges.covering(fs.diskCache.objectKey("object"), 6000, 100)
	require.False(t, err, "the deleted object's ranges are not indexed")

	recreated := slices.Clone(data)
	for i := range recreated {
		recreated[i] ^= 0xff
	}
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: int64(len(recreated)), Data: recreated}}, Policy: SkipDiskCache | SkipMemoryCache}))
	require.Equal(t, recreated[6000:6100], readRangeOf(t, fs, 0, 6000, 100))
	require.Equal(t, int64(2), storage.reads.Load(), "the read goes to object storage")
}

// stallingStorage serves the first after bytes of the first read, then holds it until
// its context ends or release is closed.
type stallingStorage struct {
	ObjectStorage
	after    int
	reads    atomic.Int64
	stalled  chan struct{}
	release  chan struct{}
	canceled atomic.Bool
}

func (s *stallingStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	r, err := s.ObjectStorage.Read(ctx, key, min, max)
	if err != nil || s.reads.Add(1) > 1 {
		return r, err
	}
	return &stallingReader{r: r, s: s, ctx: ctx, left: s.after}, nil
}

type stallingReader struct {
	r    io.ReadCloser
	s    *stallingStorage
	ctx  context.Context
	left int
}

func (r *stallingReader) Read(p []byte) (int, error) {
	if r.left == 0 {
		r.left = -1
		close(r.s.stalled)
		select {
		case <-r.ctx.Done():
			r.s.canceled.Store(true)
			return 0, r.ctx.Err()
		case <-r.s.release:
		}
	}
	if r.left > 0 && len(p) > r.left {
		p = p[:r.left]
	}
	n, err := r.r.Read(p)
	if r.left > 0 {
		r.left -= n
	}
	return n, err
}

func (r *stallingReader) Close() error { return r.r.Close() }

// A flight's buffer is kept for reuse only once the flight is forgotten and no read copies
// from it.
func TestPrefetchFlightBufferReusedAfterLastHolder(t *testing.T) {
	o := newBufferDownload(getFlightBuf(100<<10), nil, 0, foregroundRequest)
	require.True(t, o.hold())
	o.retire()
	require.NotNil(t, o.buf, "a reader still holds it")
	o.release()
	require.Nil(t, o.buf)
}

// The memory budget of a server is a share of the memory available, measured at most once
// per refresh, the default when it cannot be measured, or the configured budget.
func TestServerMemoryFollowsAvailableMemory(t *testing.T) {
	prev := memoryAvailable
	t.Cleanup(func() { memoryAvailable = prev })
	var calls int
	avail := uint64(4 << 30)
	memoryAvailable = func() (uint64, bool) { calls++; return avail, true }
	d := &DiskCache{memoryPrefetchMax: 20 << 20, server: newObjectStorageServer(nil, minRequestRoom)}
	share := int64(avail / 100 * serverMemoryPercent)
	require.Equal(t, share, d.server.memoryBudget())
	avail = 1 << 20
	require.Equal(t, share, d.server.memoryBudget(), "measured once per refresh")
	require.Equal(t, 1, calls)
	d.server.measuredAt.Store(0)
	require.Equal(t, int64(avail/100*serverMemoryPercent), d.server.memoryBudget(), "a small cgroup holds its share")
	avail = 0
	d.server.measuredAt.Store(0)
	require.Zero(t, d.server.memoryBudget(), "exhausted memory holds none")
	memoryAvailable = func() (uint64, bool) { return 0, false }
	d.server.measuredAt.Store(0)
	require.Equal(t, int64(serverMemoryDefault), d.server.memoryBudget())
	d.server.memoryLimit = 7
	require.Equal(t, int64(7), d.server.memoryBudget())
}

// A range read takes its bytes from a prefetch object that caches nothing: a read that
// caches its entries caches them, and one that caches whole objects caches no byte.
func TestS3FSRangeReadCachesItsEntriesNotItsPrefetchObject(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 4096, 64<<10)
	read := func(policy Policy, rangeRead bool) {
		vec := &IOVector{FilePath: "object", Policy: policy | SkipMemoryCache,
			Entries: []IOEntry{{Offset: 5000, Size: 10}, {Offset: 5010, Size: 10}}}
		if rangeRead {
			// A read that caches whole objects reads ranges when it does not download one.
			require.NoError(t, fs.read(context.Background(), vec, true, false))
		} else {
			require.NoError(t, fs.Read(context.Background(), vec))
		}
		fs.diskCache.prefetchObjects.reads.Wait()
		flushPublishers(fs)
		require.Equal(t, data[5000:5010], vec.Entries[0].Data)
		require.Equal(t, data[5010:5020], vec.Entries[1].Data)
		vec.Release()
	}
	cacheFiles := func() []string {
		var files []string
		require.NoError(t, filepath.Walk(fs.diskCache.path, func(p string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() && strings.HasSuffix(p, cacheFileSuffix) {
				files = append(files, filepath.Base(p))
			}
			return err
		}))
		slices.Sort(files)
		return files
	}

	read(0, true)
	require.Equal(t, int64(1), storage.reads.Load())
	require.Empty(t, cacheFiles())

	read(SkipFullFilePreloads|ExactRead, false)
	require.Equal(t, int64(2), storage.reads.Load())
	require.Equal(t, []string{
		filepath.Base(fs.diskCache.pathForIOEntry("object", IOEntry{Offset: 5000, Size: 10})),
		filepath.Base(fs.diskCache.pathForIOEntry("object", IOEntry{Offset: 5010, Size: 10})),
	}, cacheFiles(), "the entries, not the prefetch object's range")
}

// A read of a cached range whose file is short or removed returns storage's bytes, and
// drops the range: the reads after it are served from the range cached again.
func TestDiskCacheBadRangeFileIsDropped(t *testing.T) {
	for _, mode := range []string{"truncate", "remove"} {
		t.Run(mode, func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 4096, 64<<10)
			require.Equal(t, data[5000:5002], readRangeOf(t, fs, 0, 5000, 2))
			rs := fs.diskCache.ranges.of(fs.diskCache.objectKey("object"))
			require.Len(t, rs, 1)
			if mode == "truncate" {
				require.NoError(t, os.Truncate(rs[0].diskPath, 100))
			} else {
				require.NoError(t, os.Remove(rs[0].diskPath))
			}
			fs.diskCache.files.closeAll()
			require.Equal(t, data[6000:6010], readRangeOf(t, fs, SkipAllCache, 6000, 10), "control")
			before := storage.reads.Load()
			require.Equal(t, data[6000:6010], readRangeOf(t, fs, 0, 6000, 10))
			require.Equal(t, before+1, storage.reads.Load())
			for i := range 4 {
				off := int64(6100 + 10*i)
				require.Equal(t, data[off:off+10], readRangeOf(t, fs, 0, off, 10))
			}
			require.LessOrEqual(t, storage.reads.Load(), before+2, "the bad range is dropped and cached again")
		})
	}
}

type failingRangeStorage struct {
	ObjectStorage
	fail atomic.Bool
}

func (f *failingRangeStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	if f.fail.Load() && min != nil && *min > 0 {
		return nil, errors.New("injected range read failure")
	}
	return f.ObjectStorage.Read(ctx, key, min, max)
}

type statCountingStorage struct {
	ObjectStorage
	stats atomic.Int64
}

func (s *statCountingStorage) Stat(ctx context.Context, key string) (int64, error) {
	s.stats.Add(1)
	return s.ObjectStorage.Stat(ctx, key)
}

func newObjectSplitFS(t *testing.T, split objectSplitConfig, size int) (*S3FS, *countingStorage, []byte) {
	fs, storage, data := newPrefetchRangeFS(t, 0, size)
	fs.diskCache.objectSplit = split
	fs.diskCache.prefetched.store(fs.diskCache.objectKey("object"), int64(size))
	return fs, storage, data
}

func readFullObjectOf(fs *S3FS, offset, size int64) ([]byte, error) {
	vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache, Entries: []IOEntry{{Offset: offset, Size: size}}}
	err := fs.Read(context.Background(), vec)
	data := slices.Clone(vec.Entries[0].Data)
	vec.Release()
	flushPublishers(fs)
	return data, err
}

// flushPublishers waits for the publishers of the caches on disk of fs.
func flushPublishers(fs *S3FS) {
	ctx := context.Background()
	if fs.diskCache != nil {
		fs.diskCache.Flush(ctx)
	}
	if fs.indexFileCache != nil {
		fs.indexFileCache.disk.Flush(ctx)
	}
}

// Every whole-object download of a large object splits, however many downloads are in
// flight: nine held at once read two parts each. The control downloads one.
func TestS3FSObjectSplitWithManyDownloads(t *testing.T) {
	for _, n := range []int{9, 1} {
		t.Run(map[int]string{9: "many", 1: "control"}[n], func(t *testing.T) {
			const size = 300 << 10
			ctx := context.Background()
			fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, size)
			d := fs.diskCache
			<-d.load.done
			names := make([]string, n)
			for i := range names {
				names[i] = fmt.Sprintf("o%d", i)
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: names[i], Policy: SkipAllCache,
					Entries: []IOEntry{{Size: size, Data: data}}}))
				d.prefetched.store(d.objectKey(names[i]), size)
			}
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
			fs.server.raw = storage
			done := make(chan error, n)
			for _, name := range names {
				go func() { done <- fs.PrefetchFile(ctx, name) }()
			}
			opens := func() int {
				total := 0
				for _, name := range names {
					_, _, c := storage.counts(fs.pathToKey(name))
					total += c
				}
				return total
			}
			require.Eventually(t, func() bool { return opens() == 2*n }, diskCacheLifecycleTestTimeout, time.Millisecond)
			close(storage.gate)
			for range names {
				require.NoError(t, <-done)
			}
			require.Equal(t, 2*n, opens())
		})
	}
}

// A failed part fails the download, which publishes no whole object; the read has its
// bytes from the part that did not fail.
func TestS3FSObjectSplitFailedPartPublishesNothing(t *testing.T) {
	fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
	failing := &failingRangeStorage{ObjectStorage: fs.server.raw}
	failing.fail.Store(true)
	fs.server.raw = failing
	got, err := readFullObjectOf(fs, 10, 10)
	require.NoError(t, err)
	require.Equal(t, data[10:20], got)
	require.False(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))))

	failing.fail.Store(false)
	got, err = readFullObjectOf(fs, 10, 10)
	require.NoError(t, err)
	require.Equal(t, data[10:20], got)
}

// disk-prefetch-memory-max defaults to 32 MiB and is at least 8 MiB.
func TestDiskPrefetchMemoryMaxConfig(t *testing.T) {
	require.Equal(t, int64(32<<20), (&CacheConfig{}).diskPrefetchMemoryMax())
	require.Equal(t, int64(64<<20), (&CacheConfig{DiskPrefetchMemoryMax: ptrTo[toml.ByteSize](64 << 20)}).diskPrefetchMemoryMax())
	require.Equal(t, int64(8<<20), (&CacheConfig{DiskPrefetchMemoryMax: ptrTo[toml.ByteSize](1)}).diskPrefetchMemoryMax())
	require.Equal(t, int64(32<<20), (&CacheConfig{DiskPrefetchMemoryMax: ptrTo[toml.ByteSize](0)}).diskPrefetchMemoryMax())
}

// A download with no room free waits for room without failing, and downloads once it is
// freed: the whole object of a read, and the file-backed fill of a column read past
// disk-prefetch-memory-max.
func TestS3FSDownloadWaitsForConn(t *testing.T) {
	for _, whole := range []bool{true, false} {
		t.Run(map[bool]string{true: "whole object", false: "file-backed column read"}[whole], func(t *testing.T) {
			fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
			d := fs.diskCache
			d.memoryPrefetchMax = 1
			limitRoom(t, fs.server, 1, 1)
			free := occupy(t, fs.server, foregroundRequest, 1)
			done := make(chan []byte, 1)
			go func() {
				read := readFullObjectOf
				if !whole {
					read = func(fs *S3FS, offset, size int64) ([]byte, error) {
						return readColumnRange(context.Background(), fs, PrefetchRange{Offset: offset, Size: size})
					}
				}
				got, err := read(fs, 100, 10)
				assert.NoError(t, err)
				done <- got
			}()
			require.Eventually(t, func() bool { _, waiting := fs.server.counts(); return waiting[foregroundRequest] > 0 },
				10*time.Second, time.Millisecond, "the download waits for room")
			require.Never(t, func() bool { return len(done) > 0 }, 100*time.Millisecond, time.Millisecond)
			require.False(t, d.cache.Contains(d.wholePath("object", int64(len(data)))))
			free()
			require.Equal(t, data[100:110], <-done)
			require.Equal(t, whole, d.cache.Contains(d.wholePath("object", int64(len(data)))))
			require.Zero(t, heldRoom(fs.server))
		})
	}
}

// waitingForConn reports whether a whole-object download of the object at diskPath is
// registered and none of its parts began: its requests wait for room.
func waitingForConn(d *DiskCache, diskPath string) bool {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, p := range f.m[diskPath] {
		if !p.whole {
			continue
		}
		if download := p.joinableDownload(); download != nil {
			download.mu.Lock()
			began := slices.ContainsFunc(download.parts, func(part downloadPart) bool { return part.begun })
			download.mu.Unlock()
			if !began {
				return true
			}
		}
	}
	return false
}

// A read whose whole-object download waits for a connection when its object is deleted,
// or the cache closes, reads its range and caches nothing.
func TestS3FSReadWaitingForConnectionEndsOnDeleteOrClose(t *testing.T) {
	for _, end := range []string{"delete", "close"} {
		t.Run(end, func(t *testing.T) {
			fs, _, data := newObjectSplitFS(t, objectSplitConfig{}, 300<<10)
			fs.server.raw = keepOnDeleteStorage{fs.server.raw}
			d := fs.diskCache
			limitRoom(t, fs.server, 2, 1)
			bystanders := occupy(t, fs.server, foregroundRequest, 2)
			diskPath := d.objectKey("object")
			done := make(chan []byte, 1)
			go func() {
				got, err := readFullObjectOf(fs, 100, 10)
				require.NoError(t, err)
				done <- got
			}()
			require.Eventually(t, func() bool { return waitingForConn(d, diskPath) }, 10*time.Second, time.Millisecond)
			if end == "delete" {
				// The delete of the cached object; its request waits for the room the
				// bystanders hold.
				require.NoError(t, d.DeletePaths(context.Background(), []string{"object"}))
			} else {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				d.closePrefetches(ctx)
				require.NoError(t, ctx.Err(), "the close waits for no download")
			}
			require.Eventually(t, func() bool { return d.wholeFill(diskPath, false) == nil }, 5*time.Second, time.Millisecond,
				"the download ends")
			require.Zero(t, heldRoom(fs.server), "it took no room")
			// The read's own range read waits for room of its own.
			bystanders()
			require.Equal(t, data[100:110], <-done)
			require.Empty(t, d.objectFiles("object"))
		})
	}
}

// A range fill, file or memory backed, whose publication finds the publisher queue full
// waits for room holding no room of the server, so reads go on; the fills publish once
// room frees. A delete of the object ends the waits of its fills, and nothing of the object
// is cached. The control is the fill that finds room in the queue, which a read does not
// wait for.
func TestS3FSRangeFillWaitsForPublisherRoom(t *testing.T) {
	for _, tc := range []struct {
		name   string
		memory bool
		ev     string
	}{
		{"file/room freed", false, "room freed"},
		{"file/delete", false, "delete"},
		{"memory/room freed", true, "room freed"},
		{"memory/delete", true, "delete"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ev := tc.ev
			fs, storage, data := newPrefetchRangeFS(t, 4096, 64<<10)
			fs.server.raw = keepOnDeleteStorage{storage}
			d := fs.diskCache
			if tc.memory {
				// The first fill holds the last of the memory: the others and the reads
				// use files.
				fs.server.memoryLimit = flightBufSize(4096)
			} else {
				d.memoryPrefetchMax = 1
			}
			shrinkPublisher(d, 1)
			key := d.objectKey("object")
			registered := func() int {
				d.prefetchObjects.mu.Lock()
				defer d.prefetchObjects.mu.Unlock()
				return len(d.prefetchObjects.m[key])
			}
			syncStarted, unblock := installBlockedDiskCacheFileSync(d)
			t.Cleanup(unblock)
			ctx := context.Background()
			read := func(policy Policy, offset int64) []byte {
				vec := &IOVector{FilePath: "object", Policy: policy | SkipFullFilePreloads | SkipMemoryCache,
					Entries: []IOEntry{{Offset: offset, Size: 10}}}
				require.NoError(t, fs.Read(ctx, vec))
				b := slices.Clone(vec.Entries[0].Data)
				vec.Release()
				return b
			}
			require.Equal(t, data[0:10], read(0, 0))
			select {
			case <-syncStarted:
			case <-time.After(diskCacheLifecycleTestTimeout):
				t.Fatal("the publisher did not take the first fill")
			}
			require.Equal(t, data[4096:4106], read(0, 4096))
			require.Eventually(t, func() bool { return len(d.pub.jobs) == cap(d.pub.jobs) },
				diskCacheLifecycleTestTimeout, time.Millisecond, "the second fill is queued")
			require.Equal(t, data[8192:8202], read(0, 8192))
			require.Equal(t, 3, registered(), "the fills wait for their publication")
			ended := make(chan error, 1)
			go func() {
				for registered() > 0 {
					time.Sleep(time.Millisecond)
				}
				ended <- nil
			}()
			requireSubmitterWaits(t, d, ended)
			require.Zero(t, heldRoom(fs.server), "a waiting fill holds no room")
			require.Equal(t, data[20000:20010], read(SkipDiskCacheWrites, 20000), "a read goes on")

			reads := storage.reads.Load()
			// The delete ends the wait of the fill, and waits for the publisher.
			deleted := make(chan error, 1)
			if ev == "delete" {
				go func() { deleted <- fs.Delete(ctx, "object") }()
				select {
				case <-ended:
				case <-time.After(diskCacheLifecycleTestTimeout):
					t.Fatal("the delete did not end the wait of the fill")
				}
			}
			unblock()
			if ev == "delete" {
				require.NoError(t, <-deleted)
			}
			waitObjectsEnd(d, "object")
			flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
			defer cancel()
			d.Flush(flushCtx)
			require.NoError(t, flushCtx.Err())
			if ev == "delete" {
				require.Empty(t, d.objectFiles("object"))
			} else {
				for _, off := range []int64{0, 4096, 8192} {
					require.Equal(t, data[off+1:off+11], read(0, off+1))
				}
				require.Equal(t, reads, storage.reads.Load(), "the fills are cached")
			}
			requireNoPublications(t, d)
			requireNoTempFiles(t, d)
		})
	}
}

// Deleting an object forgets the size a prefetch told of: after the path is written again
// at another length, a read to the end of it reads the new object whole.
func TestDeleteForgetsPrefetchedSize(t *testing.T) {
	fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
	ctx := context.Background()
	require.NoError(t, fs.Delete(ctx, "object"))
	_, known := fs.diskCache.prefetched.size(fs.diskCache.objectKey("object"))
	require.False(t, known)

	longer := make([]byte, 2*len(data))
	for i := range longer {
		longer[i] = byte(i*11 + i/313)
	}
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: int64(len(longer)), Data: longer}}, Policy: SkipAllCache}))
	got, err := readFullObjectOf(fs, 0, -1)
	require.NoError(t, err)
	require.Equal(t, longer, got)
}

// A read of bytes in a part of a split download that waits for room joins the download and
// waits for the part, making no request of its own.
func TestS3FSReadOfPartWaitingForConnectionJoinsIt(t *testing.T) {
	fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
	limitRoom(t, fs.server, 2, 1)
	bystander := occupy(t, fs.server, foregroundRequest, 1)
	key := fs.pathToKey("object")
	g := newBlackBoxStorage(fs.server.raw, func(k string, off int64) bool { return k == key && off == 0 })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	defer release()
	whole := make(chan error, 1)
	go func() {
		_, err := readFullObjectOf(fs, 100, 10)
		whole <- err
	}()
	require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 1 }, 5*time.Second, time.Millisecond,
		"the first part holds the only free room")
	require.Eventually(t, waitingFor(fs.server, foregroundRequest, 1), 5*time.Second, time.Millisecond,
		"the second part waits for room")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: []IOEntry{{Offset: 250 << 10, Size: 10}}}
	read := make(chan error, 1)
	go func() { read <- fs.Read(ctx, vec) }()
	require.Never(t, waitingFor(fs.server, foregroundRequest, 2), 100*time.Millisecond, time.Millisecond,
		"the read makes no request of its own")
	bystander()
	require.NoError(t, <-read)
	require.Equal(t, data[250<<10:250<<10+10], vec.Entries[0].Data)
	vec.Release()
	release()
	require.NoError(t, <-whole)
	_, _, total := g.counts(key)
	require.Equal(t, 2, total, "the object is read once, in two parts")
}

// A read whose entry write finds the publisher queue full waits for room: it returns once
// room frees, with its entry cached, or once its context ends, with its entry not cached.
// The control is the read whose write finds room, which returns at once. An asynchronous
// update whose context ended before it is submitted fails and caches nothing.
func TestS3FSReadEntryWriteWaitsForPublisherRoom(t *testing.T) {
	for _, ev := range []string{"room freed", "context ends"} {
		t.Run(ev, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
			d := fs.diskCache
			shrinkPublisher(d, 1)
			syncStarted, unblock := installBlockedDiskCacheFileSync(d)
			t.Cleanup(unblock)
			type res struct {
				b   []byte
				err error
			}
			read := func(ctx context.Context, offset int64) res {
				vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache | SkipFullFilePreloads | ExactRead,
					Entries: []IOEntry{{Offset: offset, Size: 10}}}
				err := fs.Read(ctx, vec)
				b := append([]byte(nil), vec.Entries[0].Data...)
				vec.Release()
				return res{b, err}
			}
			ctx := context.Background()
			canceled, cancelNow := context.WithCancel(ctx)
			cancelNow()
			require.ErrorIs(t, d.Update(canceled, &IOVector{FilePath: "object",
				Entries: []IOEntry{{Offset: 400, Size: 10, Data: data[400:410]}}}, true), context.Canceled)
			require.Equal(t, res{data[100:110], nil}, read(ctx, 100))
			select {
			case <-syncStarted:
			case <-time.After(diskCacheLifecycleTestTimeout):
				t.Fatal("the publisher did not take the first entry")
			}
			require.Equal(t, res{data[200:210], nil}, read(ctx, 200))

			readCtx, cancel := context.WithCancel(ctx)
			defer cancel()
			done := make(chan res, 1)
			go func() { done <- read(readCtx, 300) }()
			returned := make(chan error, 1)
			go func() { r := <-done; done <- r; returned <- r.err }()
			requireSubmitterWaits(t, d, returned)
			if ev == "context ends" {
				cancel()
			} else {
				unblock()
			}
			select {
			case <-returned:
				r := <-done
				if r.err == nil {
					require.Equal(t, data[300:310], r.b)
				} else {
					require.ErrorIs(t, r.err, context.Canceled)
				}
			case <-time.After(diskCacheLifecycleTestTimeout):
				t.Fatalf("%s: the read still waits for room", ev)
			}
			unblock()
			flushCtx, cancelFlush := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
			defer cancelFlush()
			d.Flush(flushCtx)
			require.NoError(t, flushCtx.Err())
			for off, cached := range map[int64]bool{100: true, 200: true, 300: ev == "room freed", 400: false} {
				vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: off, Size: 10}}}
				require.NoError(t, d.Read(ctx, vec))
				require.Equal(t, cached, vec.Entries[0].done, off)
				vec.Release()
			}
			requireNoPublications(t, d)
			requireNoTempFiles(t, d)
		})
	}
}

// onPublisher reports whether the calling goroutine is a worker of a cache publisher.
func onPublisher() bool {
	stack := string(debug.Stack())
	return strings.Contains(stack, "(*publisher).run(")
}

// Every kind of file and entry directory a cache publishes is published by a worker of the
// publisher of its cache: range fills, file and memory backed, an entry a read caches, a
// whole object a read or a prefetch fills, the content of a write, an index object and an
// index-file cache entry. The control is the count of publications the hooks see.
func TestEveryPublicationIsThePublishers(t *testing.T) {
	ctx := context.Background()
	read := func(fs *S3FS, policy Policy, offset, size int64) {
		vec := &IOVector{FilePath: "object", Policy: policy | SkipMemoryCache, Entries: []IOEntry{{Offset: offset, Size: size}}}
		require.NoError(t, fs.Read(ctx, vec))
		vec.Release()
	}
	for _, tc := range []struct {
		name    string
		newFS   func(t *testing.T) *S3FS
		publish func(t *testing.T, fs *S3FS)
	}{
		{"file range fill", func(t *testing.T) *S3FS {
			fs, _, _ := newPrefetchRangeFS(t, 4096, 64<<10)
			fs.diskCache.memoryPrefetchMax = 1
			return fs
		}, func(t *testing.T, fs *S3FS) { read(fs, SkipFullFilePreloads, 100, 10) }},
		{"memory range fill", func(t *testing.T) *S3FS {
			fs, _, _ := newPrefetchRangeFS(t, 4096, 64<<10)
			return fs
		}, func(t *testing.T, fs *S3FS) { read(fs, SkipFullFilePreloads, 100, 10) }},
		{"entry", func(t *testing.T) *S3FS {
			fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
			return fs
		}, func(t *testing.T, fs *S3FS) { read(fs, SkipFullFilePreloads|ExactRead, 100, 10) }},
		{"whole object of a read", func(t *testing.T) *S3FS {
			fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
			return fs
		}, func(t *testing.T, fs *S3FS) { read(fs, 0, 100, 10) }},
		{"whole object of a prefetch", func(t *testing.T) *S3FS {
			fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
			return fs
		}, func(t *testing.T, fs *S3FS) { require.NoError(t, fs.PrefetchFile(ctx, "object")) }},
		{"content of a write", func(t *testing.T) *S3FS {
			fs, _, _ := newPrefetchRangeFS(t, 0, 1)
			return fs
		}, func(t *testing.T, fs *S3FS) {
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "written", Entries: []IOEntry{{Size: 10, Data: make([]byte, 10)}}}))
		}},
		{"index object", func(t *testing.T) *S3FS {
			fs, _, _ := newIndexObjectFS(t, true)
			return fs
		}, func(t *testing.T, fs *S3FS) {
			require.NoError(t, fs.PrefetchFile(WithFileServicePolicy(ctx, IndexFileObject), "object"))
		}},
		{"index-file cache entry", func(t *testing.T) *S3FS {
			fs, _, _ := newIndexObjectFS(t, true)
			return fs
		}, func(t *testing.T, fs *S3FS) {
			_, err := fs.indexFileCache.Put(ctx, "entry", []IndexCacheFile{{Name: "data",
				Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader("x")), nil }}})
			require.NoError(t, err)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs := tc.newFS(t)
			var mu sync.Mutex
			var publications, inline int
			hook := func() {
				mu.Lock()
				defer mu.Unlock()
				publications++
				if !onPublisher() {
					inline++
				}
			}
			fs.diskCache.hooks.beforeFilePublication = hook
			if fs.indexFileCache != nil {
				fs.indexFileCache.disk.hooks.beforeFilePublication = hook
			}
			tc.publish(t, fs)
			waitObjectsEnd(fs.diskCache, "object")
			fs.diskCache.Flush(ctx)
			if fs.indexFileCache != nil {
				fs.indexFileCache.disk.Flush(ctx)
			}
			mu.Lock()
			defer mu.Unlock()
			require.Positive(t, publications, "the control: the kind publishes")
			require.Zero(t, inline, "publications outside the publisher")
		})
	}
}

// A read during a publication reads the bytes of the object whatever the publication's
// phase: while the file is made durable, between the claim and the rename, and once it is
// published. A file at the published path is always whole. The control is the read before
// any publication.
func TestReadDuringPublicationSeesOldOrNewState(t *testing.T) {
	for _, kind := range []string{"range", "whole object"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			unit, diskPath := int64(4096), ""
			var policy Policy = SkipFullFilePreloads
			if kind == "whole object" {
				unit, policy = 0, 0
			}
			fs, storage, data := newPrefetchRangeFS(t, unit, 300<<10)
			d := fs.diskCache
			d.memoryPrefetchMax = 1
			if kind == "range" {
				diskPath = d.pathForIOEntry("object", IOEntry{Offset: 0, Size: unit})
			} else {
				diskPath = d.wholePath("object", int64(len(data)))
			}
			readAll := func(phase string) {
				var wg sync.WaitGroup
				for i := range 8 {
					wg.Add(1)
					go func() {
						defer wg.Done()
						off := int64(100 + 7*i)
						vec := &IOVector{FilePath: "object", Policy: policy | SkipMemoryCache | SkipDiskCacheWrites,
							Entries: []IOEntry{{Offset: off, Size: 10}}}
						assert.NoError(t, fs.Read(ctx, vec), phase)
						assert.Equal(t, data[off:off+10], vec.Entries[0].Data, phase)
						vec.Release()
					}()
				}
				wg.Wait()
				if info, err := os.Stat(diskPath); err == nil {
					want := unit
					if kind == "whole object" {
						want = int64(len(data))
					}
					require.Equal(t, want, info.Size(), "%s: a published file is whole", phase)
				}
			}
			readAll("control")
			type gate struct{ at, resume chan struct{} }
			newGate := func() gate { return gate{make(chan struct{}), make(chan struct{})} }
			durable, claimed := newGate(), newGate()
			var durableOnce, claimedOnce sync.Once
			d.hooks.beforeFilePublication = func() {
				durableOnce.Do(func() { close(durable.at); <-durable.resume })
			}
			d.hooks.afterPublicationClaim = func() {
				claimedOnce.Do(func() { close(claimed.at); <-claimed.resume })
			}
			resume := func(g gate) { defer func() { _ = recover() }(); close(g.resume) }
			t.Cleanup(func() { resume(durable); resume(claimed) })
			vec := &IOVector{FilePath: "object", Policy: policy | SkipMemoryCache, Entries: []IOEntry{{Offset: 10, Size: 10}}}
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[10:20], vec.Entries[0].Data)
			vec.Release()
			for _, step := range []struct {
				phase string
				g     gate
			}{{"made durable", durable}, {"claimed", claimed}} {
				select {
				case <-step.g.at:
				case <-time.After(diskCacheLifecycleTestTimeout):
					t.Fatalf("the publication did not reach %s", step.phase)
				}
				readAll(step.phase)
				resume(step.g)
			}
			waitObjectsEnd(d, "object")
			d.Flush(ctx)
			reads := storage.reads.Load()
			readAll("published")
			require.Equal(t, reads, storage.reads.Load(), "the published file serves the reads")
		})
	}
}

// A publisher never waits for a submitter waiting for room in its queue: with the queue
// full, an index object fill and a Put of the same entry both wait for room, holding
// nothing the publisher needs, and every publication completes once room frees.
func TestIndexPublisherDoesNotWaitOnWaitingSubmitters(t *testing.T) {
	ctx := context.Background()
	fs, _, data := newIndexObjectFS(t, true)
	c := fs.indexFileCache
	shrinkPublisher(c.disk, 1)
	syncStarted, unblock := installBlockedDiskCacheFileSync(c.disk)
	t.Cleanup(unblock)
	put := func(name string) chan error {
		done := make(chan error, 1)
		go func() {
			_, err := c.Put(ctx, name, []IndexCacheFile{{Name: IndexObjectFile,
				Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }}})
			done <- err
		}()
		return done
	}
	first := put("first")
	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the publisher did not take the first entry")
	}
	queued := put("queued")
	require.Eventually(t, func() bool { return len(c.disk.pub.jobs) == cap(c.disk.pub.jobs) },
		diskCacheLifecycleTestTimeout, time.Millisecond)
	fill := make(chan error, 1)
	go func() { fill <- fs.PrefetchFile(WithFileServicePolicy(ctx, IndexFileObject), "object") }()
	requireSubmitterWaits(t, c.disk, fill)
	same := put(IndexObjectEntry("object"))
	requireSubmitterWaits(t, c.disk, same)
	require.Zero(t, heldRoom(fs.server), "the waiting fill holds no room")
	unblock()
	for name, done := range map[string]chan error{"first": first, "queued": queued, "fill": fill, "same entry": same} {
		select {
		case err := <-done:
			require.NoError(t, err, name)
		case <-time.After(diskCacheLifecycleTestTimeout):
			t.Fatalf("%s did not complete", name)
		}
	}
	flushPublishers(fs)
	diskPath, cached := fs.indexObjectCached("object")
	require.True(t, cached)
	got, err := os.ReadFile(filepath.Join(diskPath, IndexObjectFile))
	require.NoError(t, err)
	require.Equal(t, data, got)
	requireNoPublications(t, c.disk)
	requireNoTempFiles(t, c.disk)
}

// jpRead is one range read of object storage.
type jpRead struct {
	key        string
	start, end int64 // end is -1 for a read to the end
}

// jpStall holds a range read after it delivered after bytes until release is closed, or
// ends it there, short, once short is set, or with an error once fail is set.
type jpStall struct {
	after   int64
	short   atomic.Bool
	fail    atomic.Bool
	release chan struct{}
	sent    atomic.Int64
}

// jpStorage records the range reads of object storage and holds those stall returns a
// stall for.
type jpStorage struct {
	ObjectStorage
	stall func(r jpRead) *jpStall

	mu    sync.Mutex
	reads []jpRead
}

func (s *jpStorage) Read(ctx context.Context, key string, from, to *int64) (io.ReadCloser, error) {
	r := jpRead{key: key, end: -1}
	if from != nil {
		r.start = *from
	}
	if to != nil {
		r.end = *to
	}
	s.mu.Lock()
	s.reads = append(s.reads, r)
	s.mu.Unlock()
	rc, err := s.ObjectStorage.Read(ctx, key, from, to)
	if err != nil {
		return nil, err
	}
	if s.stall == nil {
		return rc, nil
	}
	st := s.stall(r)
	if st == nil {
		return rc, nil
	}
	return &jpReader{r: rc, st: st, done: ctx.Done(), ctx: ctx}, nil
}

// readsOf returns the range reads of the object at key.
func (s *jpStorage) readsOf(key string) []jpRead {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []jpRead
	for _, r := range s.reads {
		if r.key == key {
			out = append(out, r)
		}
	}
	return out
}

// overlapping counts the range reads of key holding bytes of [lo, hi).
func (s *jpStorage) overlapping(key string, lo, hi int64) int {
	n := 0
	for _, r := range s.readsOf(key) {
		if r.start < hi && (r.end < 0 || lo < r.end) {
			n++
		}
	}
	return n
}

type jpReader struct {
	r    io.ReadCloser
	st   *jpStall
	done <-chan struct{}
	ctx  context.Context
}

func (g *jpReader) Read(p []byte) (int, error) {
	sent := g.st.sent.Load()
	if sent >= g.st.after {
		select {
		case <-g.st.release:
		case <-g.done:
			return 0, g.ctx.Err()
		}
		if g.st.short.Load() {
			return 0, io.EOF
		}
		if g.st.fail.Load() {
			return 0, errors.New("injected source failure")
		}
	} else if rest := g.st.after - sent; int64(len(p)) > rest {
		p = p[:rest]
	}
	n, err := g.r.Read(p)
	g.st.sent.Add(int64(n))
	return n, err
}

func (g *jpReader) Close() error { return g.r.Close() }

func newJPStall(after int64) *jpStall {
	return &jpStall{after: after, release: make(chan struct{})}
}

// jpFS returns a file service whose object "object" holds size bytes and "other" holds
// 64 KiB, read through a jpStorage.
func jpFS(t *testing.T, size int) (*S3FS, *jpStorage, []byte, []byte) {
	return jpUnitFS(t, 0, size)
}

// jpUnitFS is jpFS with prefetch ranges of unit bytes, none for 0.
func jpUnitFS(t *testing.T, unit int64, size int) (*S3FS, *jpStorage, []byte, []byte) {
	fs, _, data := newPrefetchRangeFS(t, unit, size)
	other := make([]byte, 64<<10)
	for i := range other {
		other[i] = byte(i * 13)
	}
	require.NoError(t, fs.Write(context.Background(), IOVector{FilePath: "other",
		Entries: []IOEntry{{Size: int64(len(other)), Data: other}}, Policy: SkipDiskCache | SkipMemoryCache}))
	s := &jpStorage{ObjectStorage: fs.server.raw}
	fs.server.raw = s
	return fs, s, data, other
}

type jpResult struct {
	data []byte
	err  error
}

// jpColumnRead reads r of the object at filePath as a column read under a deadline of
// timeout, in the background.
func jpColumnRead(fs *S3FS, filePath string, r PrefetchRange, timeout time.Duration) <-chan jpResult {
	return jpReadOf(fs, filePath, ColumnPrefetch, r, timeout)
}

// jpReadOf reads r of the object at filePath with policy under a deadline of timeout, in
// the background.
func jpReadOf(fs *S3FS, filePath string, policy Policy, r PrefetchRange, timeout time.Duration) <-chan jpResult {
	done := make(chan jpResult, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		vec := &IOVector{FilePath: filePath, Policy: policy | SkipFullFilePreloads | SkipMemoryCache,
			Entries: []IOEntry{{Offset: r.Offset, Size: r.Size}}}
		if err := fs.Read(ctx, vec); err != nil {
			done <- jpResult{err: err}
			return
		}
		got := slices.Clone(vec.Entries[0].Data)
		vec.Release()
		done <- jpResult{data: got}
	}()
	return done
}

func jpWait(t *testing.T, done <-chan jpResult, within time.Duration, what string) jpResult {
	t.Helper()
	select {
	case r := <-done:
		return r
	case <-time.After(within):
		t.Fatalf("%s did not return within %v", what, within)
		return jpResult{}
	}
}

// jpBystander reads the object "other" with a deadline of 2 s while a probe holds its
// stall, and requires its bytes.
func jpBystander(t *testing.T, fs *S3FS, other []byte) {
	t.Helper()
	r := PrefetchRange{Offset: 1000, Size: 100}
	got := jpWait(t, jpColumnRead(fs, "other", r, 2*time.Second), 3*time.Second, "the bystander read")
	require.NoError(t, got.err)
	require.Equal(t, other[r.Offset:r.Offset+r.Size], got.data)
}

// (a) A read of bytes a running background fill of the whole object writes late, behind
// a slow download, joins the fill: the object is read once. So does a file-backed range
// read through prefetch ranges while the download has no byte yet. Control: a read of
// bytes the fill has written is served by them at once.
func TestJoinProbeReadJoinsSlowRunningWholeFill(t *testing.T) {
	const size = 4 << 20
	for _, tc := range []struct {
		name   string
		unit   int64
		after  int64
		policy Policy
		r      PrefetchRange
	}{
		{"column read", 0, 512 << 10, ColumnPrefetch, PrefetchRange{Offset: 3 << 20, Size: 1 << 10}},
		{"file-backed range read before the first byte", 1 << 20, 0, 0, PrefetchRange{Offset: 3 << 20, Size: 64 << 10}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, s, data, other := jpUnitFS(t, tc.unit, size)
			if tc.unit > 0 {
				fs.diskCache.memoryPrefetchMax = 256 << 10
			}
			key := fs.pathToKey("object")
			st := newJPStall(tc.after)
			releaseStall := sync.OnceFunc(func() { close(st.release) })
			defer releaseStall()
			var once sync.Once
			s.stall = func(r jpRead) *jpStall {
				var got *jpStall
				if r.key == key && r.start == 0 {
					once.Do(func() { got = st })
				}
				return got
			}
			require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: size}))
			require.Eventually(t, func() bool { return len(s.readsOf(key)) == 1 && st.sent.Load() >= st.after },
				5*time.Second, time.Millisecond)
			// The download's rate so far is its bytes over this wait.
			time.Sleep(200 * time.Millisecond)

			if tc.after > 0 {
				control := PrefetchRange{Offset: 1000, Size: 1 << 10}
				got := jpWait(t, jpColumnRead(fs, "object", control, 2*time.Second), 3*time.Second, "the control read")
				require.NoError(t, got.err)
				require.Equal(t, data[control.Offset:control.Offset+control.Size], got.data)
				require.Len(t, s.readsOf(key), 1, "the control read is served by the written bytes")
			}

			done := jpReadOf(fs, "object", tc.policy, tc.r, 10*time.Second)
			require.Never(t, func() bool { return len(done) > 0 }, 200*time.Millisecond, time.Millisecond,
				"the read waits for the fill it joined")
			require.Len(t, s.readsOf(key), 1, "the read does not read the bytes again")
			jpBystander(t, fs, other)
			releaseStall()
			got := jpWait(t, done, 5*time.Second, "the joined read")
			require.NoError(t, got.err)
			require.Equal(t, data[tc.r.Offset:tc.r.Offset+tc.r.Size], got.data)
			probeWaitIdle(t, fs)
			require.Equal(t, 1, s.overlapping(key, tc.r.Offset, tc.r.Offset+tc.r.Size), "the bytes are read once")
		})
	}
}

// (a) A read of bytes a running background fill of a range writes late, file or memory
// backed, joins it, as a column read or an exact read of the whole range; a read skipping
// disk-cache reads reads its bytes itself. Control: a read of bytes the fill has written is
// served by them at once, and a read of bytes outside the fill reads them itself, at once.
// Once the fill ends, its range serves later reads.
func TestJoinProbeReadJoinsSlowRunningRangeFileFill(t *testing.T) {
	fill := PrefetchRange{Offset: 0, Size: 4 << 20}
	late := PrefetchRange{Offset: 3584 << 10, Size: 1 << 10}
	for _, tc := range []struct {
		name   string
		memory bool
		policy Policy
		r      PrefetchRange
		joins  bool
	}{
		{"file/column", false, ColumnPrefetch, late, true},
		{"memory/column", true, ColumnPrefetch, late, true},
		{"file/exact whole range", false, ExactColumnsRead, fill, true},
		{"file/exact skipping disk-cache reads", false, ExactColumnsRead | SkipDiskCacheReads | SkipDiskCacheWrites, late, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, s, data, other := jpFS(t, 6<<20)
			if !tc.memory {
				fs.diskCache.memoryPrefetchMax = 64 << 10
			}
			key := fs.pathToKey("object")
			st := newJPStall(512 << 10)
			releaseStall := sync.OnceFunc(func() { close(st.release) })
			defer releaseStall()
			s.stall = func(r jpRead) *jpStall {
				if r.key == key && r.start == fill.Offset && r.end == fill.Offset+fill.Size {
					return st
				}
				return nil
			}
			require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: fill.Size,
				Ranges: []PrefetchRange{fill}, Scan: &PrefetchScan{}}))
			require.Eventually(t, func() bool { return st.sent.Load() >= st.after }, 5*time.Second, time.Millisecond)
			time.Sleep(200 * time.Millisecond)
			f := &fs.diskCache.prefetchObjects
			f.mu.Lock()
			for _, fl := range f.m[fs.diskCache.objectKey("object")] {
				require.Equal(t, tc.memory, fl.joinableDownload().memory)
			}
			f.mu.Unlock()

			for _, control := range []PrefetchRange{{Offset: 1000, Size: 1 << 10}, {Offset: 5 << 20, Size: 1 << 10}} {
				got := jpWait(t, jpColumnRead(fs, "object", control, 2*time.Second), 3*time.Second, "the control read")
				require.NoError(t, got.err)
				require.Equal(t, data[control.Offset:control.Offset+control.Size], got.data)
				require.Equal(t, 1, s.overlapping(key, control.Offset, control.Offset+control.Size))
			}

			done := jpReadOf(fs, "object", tc.policy, tc.r, 10*time.Second)
			if tc.joins {
				require.Never(t, func() bool { return len(done) > 0 }, 200*time.Millisecond, time.Millisecond,
					"the read waits for the fill it joined")
				jpBystander(t, fs, other)
				releaseStall()
			}
			got := jpWait(t, done, 5*time.Second, "the read")
			require.NoError(t, got.err)
			require.Equal(t, data[tc.r.Offset:tc.r.Offset+tc.r.Size], got.data)
			releaseStall()
			probeWaitIdle(t, fs)
			reads := map[bool]int{true: 1, false: 2}[tc.joins]
			require.Equal(t, reads, s.overlapping(key, tc.r.Offset, tc.r.Offset+tc.r.Size), "the bytes are read once by the fill")
			later := PrefetchRange{Offset: 2 << 20, Size: 1 << 10}
			got = jpWait(t, jpColumnRead(fs, "object", later, 2*time.Second), 3*time.Second, "the later read")
			require.NoError(t, got.err)
			require.Equal(t, data[later.Offset:later.Offset+later.Size], got.data)
			require.Equal(t, 1, s.overlapping(key, later.Offset, later.Offset+later.Size), "the cached range serves it")
		})
	}
}

// (b) A read of bytes two running fills and a gap between them hold joins both fills and
// reads only the gap. Control: a read of bytes neither fill holds reads them itself.
func TestJoinProbeReadJoinsTwoFillsAndFillsTheGap(t *testing.T) {
	fs, s, data, other := jpFS(t, 512<<10)
	key := fs.pathToKey("object")
	a, c := PrefetchRange{Offset: 0, Size: 64 << 10}, PrefetchRange{Offset: 128 << 10, Size: 64 << 10}
	stalls := map[int64]*jpStall{a.Offset: newJPStall(0), c.Offset: newJPStall(0)}
	release := sync.OnceFunc(func() {
		for _, st := range stalls {
			close(st.release)
		}
	})
	defer release()
	s.stall = func(r jpRead) *jpStall {
		for _, f := range []PrefetchRange{a, c} {
			if r.key == key && r.start == f.Offset && r.end == f.Offset+f.Size {
				return stalls[f.Offset]
			}
		}
		return nil
	}
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: a.Size + c.Size,
		Ranges: []PrefetchRange{a, c}, Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { return len(s.readsOf(key)) == 2 }, 5*time.Second, time.Millisecond)

	control := PrefetchRange{Offset: 300 << 10, Size: 1 << 10}
	got := jpWait(t, jpColumnRead(fs, "object", control, 2*time.Second), 3*time.Second, "the control read")
	require.NoError(t, got.err)
	require.Equal(t, data[control.Offset:control.Offset+control.Size], got.data)

	r := PrefetchRange{Offset: 0, Size: c.Offset + c.Size}
	done := jpColumnRead(fs, "object", r, 10*time.Second)
	require.Never(t, func() bool { return len(done) > 0 }, 200*time.Millisecond, time.Millisecond,
		"the read waits for the fills it joined")
	jpBystander(t, fs, other)
	release()
	got = jpWait(t, done, 5*time.Second, "the joined read")
	require.NoError(t, got.err)
	require.Equal(t, data[r.Offset:r.Offset+r.Size], got.data)
	probeWaitIdle(t, fs)
	require.Equal(t, 1, s.overlapping(key, a.Offset, a.Offset+a.Size), "the first fill's bytes are read once")
	require.Equal(t, 1, s.overlapping(key, c.Offset, c.Offset+c.Size), "the second fill's bytes are read once")
	require.Equal(t, 1, s.overlapping(key, a.Offset+a.Size, c.Offset), "the gap is read once")
	require.Contains(t, s.readsOf(key), jpRead{key: key, start: a.Offset + a.Size, end: c.Offset}, "only the gap is read")
}

// (b) A range read through prefetch ranges of bytes a running fill holds only part of a
// block of joins the fill for those bytes and reads only the rest of the unit. Control: a
// read of a unit no fill holds reads the whole unit.
func TestJoinProbePrefetchRangeReadSplitsBlockAtFill(t *testing.T) {
	const unit = 64 << 10
	fs, s, data, other := jpUnitFS(t, unit, 512<<10)
	key := fs.pathToKey("object")
	fill := PrefetchRange{Offset: 10 << 10, Size: 10 << 10}
	st := newJPStall(0)
	releaseStall := sync.OnceFunc(func() { close(st.release) })
	defer releaseStall()
	s.stall = func(r jpRead) *jpStall {
		if r.key == key && r.start == fill.Offset && r.end == fill.Offset+fill.Size {
			return st
		}
		return nil
	}
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: fill.Size,
		Ranges: []PrefetchRange{fill}, Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { return len(s.readsOf(key)) == 1 }, 5*time.Second, time.Millisecond)

	read := func(r PrefetchRange, timeout time.Duration) <-chan jpResult {
		return jpReadOf(fs, "object", 0, r, timeout)
	}
	control := PrefetchRange{Offset: 4*unit + 100, Size: 100}
	got := jpWait(t, read(control, 2*time.Second), 3*time.Second, "the control read")
	require.NoError(t, got.err)
	require.Equal(t, data[control.Offset:control.Offset+control.Size], got.data)
	require.Contains(t, s.readsOf(key), jpRead{key: key, start: 4 * unit, end: 5 * unit}, "the control read reads its unit")

	r := PrefetchRange{Offset: 0, Size: 30 << 10}
	done := read(r, 10*time.Second)
	require.Never(t, func() bool { return len(done) > 0 }, 200*time.Millisecond, time.Millisecond,
		"the read waits for the fill it joined")
	jpBystander(t, fs, other)
	releaseStall()
	got = jpWait(t, done, 5*time.Second, "the joined read")
	require.NoError(t, got.err)
	require.Equal(t, data[r.Offset:r.Offset+r.Size], got.data)
	probeWaitIdle(t, fs)
	require.Equal(t, 1, s.overlapping(key, fill.Offset, fill.Offset+fill.Size), "the fill's bytes are read once")
	require.Equal(t, 1, s.overlapping(key, 0, fill.Offset), "the bytes before the fill are read once")
	require.Equal(t, 1, s.overlapping(key, fill.Offset+fill.Size, r.Offset+r.Size), "the bytes after the fill are read once")
}

// (c) A joined fill that ends short of its range leaves the read to read exactly its own
// bytes: it returns them. A joined fill whose source fails after a prefix fails the read
// with the source's error. Control: the same read with the fill not short is served by the
// fill.
func TestJoinProbeJoinedFillEndsShort(t *testing.T) {
	for _, tc := range []struct {
		name        string
		after       int64
		short, fail bool
	}{{name: "control"}, {name: "short", short: true}, {name: "failed after a prefix", after: 512, fail: true}} {
		t.Run(tc.name, func(t *testing.T) {
			fs, s, data, other := jpFS(t, 300<<10)
			key := fs.pathToKey("object")
			fill := PrefetchRange{Offset: 40 << 10, Size: 1 << 10}
			st := newJPStall(tc.after)
			releaseStall := sync.OnceFunc(func() { close(st.release) })
			defer releaseStall()
			var once sync.Once
			s.stall = func(r jpRead) *jpStall {
				var got *jpStall
				if r.key == key && r.start == fill.Offset {
					once.Do(func() { got = st })
				}
				return got
			}
			require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: fill.Size,
				Ranges: []PrefetchRange{fill}, Scan: &PrefetchScan{}}))
			require.Eventually(t, func() bool { return st.sent.Load() >= tc.after && len(s.readsOf(key)) == 1 },
				5*time.Second, time.Millisecond)
			done := jpColumnRead(fs, "object", fill, 10*time.Second)
			require.Never(t, func() bool { return len(done) > 0 }, 100*time.Millisecond, time.Millisecond,
				"the read waits for the fill it joined")
			require.Len(t, s.readsOf(key), 1)
			jpBystander(t, fs, other)
			st.short.Store(tc.short)
			st.fail.Store(tc.fail)
			releaseStall()
			got := jpWait(t, done, 5*time.Second, "the joined read")
			probeWaitIdle(t, fs)
			if tc.fail {
				require.ErrorContains(t, got.err, "injected source failure")
				return
			}
			require.NoError(t, got.err)
			require.Equal(t, data[fill.Offset:fill.Offset+fill.Size], got.data)
			want := []jpRead{{key: key, start: fill.Offset, end: fill.Offset + fill.Size}}
			if tc.short {
				want = append(want, want[0])
			}
			require.Equal(t, want, s.readsOf(key), "the fill serves the read, or it reads its own bytes")
		})
	}
}

// (c) A delete of the object stops the fill a read joined: the read returns at once, and
// a read of the object written again gets its new bytes. Control: a read of another object
// is not affected.
func TestJoinProbeJoinedFillDeleted(t *testing.T) {
	fs, s, _, other := jpFS(t, 300<<10)
	key := fs.pathToKey("object")
	fill := PrefetchRange{Offset: 40 << 10, Size: 1 << 10}
	st := newJPStall(0)
	releaseStall := sync.OnceFunc(func() { close(st.release) })
	defer releaseStall()
	var once sync.Once
	s.stall = func(r jpRead) *jpStall {
		var got *jpStall
		if r.key == key && r.start == fill.Offset {
			once.Do(func() { got = st })
		}
		return got
	}
	ctx := context.Background()
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: fill.Size,
		Ranges: []PrefetchRange{fill}, Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { return len(s.readsOf(key)) == 1 }, 5*time.Second, time.Millisecond)
	done := jpColumnRead(fs, "object", fill, 10*time.Second)
	require.Never(t, func() bool { return len(done) > 0 }, 100*time.Millisecond, time.Millisecond)
	require.NoError(t, fs.Delete(ctx, "object"))
	jpWait(t, done, 2*time.Second, "the read of the deleted object")
	jpBystander(t, fs, other)

	fresh := make([]byte, 300<<10)
	for i := range fresh {
		fresh[i] = byte(i*3 + 1)
	}
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: int64(len(fresh)), Data: fresh}},
		Policy: SkipDiskCache | SkipMemoryCache}))
	got := jpWait(t, jpColumnRead(fs, "object", fill, 2*time.Second), 3*time.Second, "the read of the new object")
	require.NoError(t, got.err)
	require.Equal(t, fresh[fill.Offset:fill.Offset+fill.Size], got.data)
}

// (d) A read that joins a waiting background fill makes it foreground: it runs while every
// background fill and all background room are held, before an earlier waiting background
// fill. Control: that earlier fill runs only once the background fills are free.
func TestJoinProbeJoinedWaitingFillRunsFirst(t *testing.T) {
	fs, s, data, _ := jpFS(t, 300<<10)
	releaseBusy := probeBusy(t, fs)
	freeRoom := occupy(t, fs.server, backgroundRequest, fs.server.backgroundMax-1)
	defer freeRoom()
	key, otherKey := fs.pathToKey("object"), fs.pathToKey("other")
	ctx := context.Background()
	early := PrefetchRange{Offset: 0, Size: 1 << 10}
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "other", Size: early.Size, Due: 1,
		Ranges: []PrefetchRange{early}, Scan: &PrefetchScan{}}))
	late := PrefetchRange{Offset: 100 << 10, Size: 1 << 10}
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: late.Size, Due: 100,
		Ranges: []PrefetchRange{late}, Scan: &PrefetchScan{}}))
	require.True(t, probeQueued(fs, "other"))
	require.True(t, probeQueued(fs, "object"))

	got := jpWait(t, jpColumnRead(fs, "object", late, 2*time.Second), 3*time.Second, "the joined read")
	require.NoError(t, got.err)
	require.Equal(t, data[late.Offset:late.Offset+late.Size], got.data)
	require.Len(t, s.readsOf(key), 1, "the joined fill serves the read")
	require.True(t, probeQueued(fs, "other"), "the earlier background fill still waits")
	require.Empty(t, s.readsOf(otherKey))
	freeRoom()
	releaseBusy()
	probeWaitIdle(t, fs)
	require.Len(t, s.readsOf(otherKey), 1)
}

// (e) A running fill of the object into the index-file cache, held, is not joined by a read
// of the disk cache, which reads its bytes itself; a running fill into the disk cache, held,
// is not joined by a read of the index-file cache. Control: each held fill still ends with
// its object cached.
// A prefetch of the object into one cache after or during a prefetch into the other, held
// at its download or at its publication, fills its own cache: each cache is filled, and
// the object is read once for each.
func TestJoinProbeCachesNeverJoinEachOther(t *testing.T) {
	type sequence struct{ first, second, hold string }
	sequences := make([]sequence, 0, 9)
	for _, sq := range []sequence{
		{"disk object", "index object", "download"},
		{"index object", "disk object", "download"},
		{"disk file", "index object", "publication"},
		{"disk object", "index object", "publication"},
	} {
		sequences = append(sequences, sq, sequence{sq.first, sq.second, sq.hold + " overlapped"})
	}
	sequences = append(sequences, sequence{"disk file", "index file", "none"})
	for _, sq := range sequences {
		t.Run(sq.first+" then "+sq.second+"/"+sq.hold, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				fs, storage, data := newIndexObjectFS(t, true)
				hold, overlap := strings.CutSuffix(sq.hold, " overlapped")
				key := fs.pathToKey("object")
				g := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return hold == "download" })
				fs.server.raw = g
				held := make(chan struct{})
				release := sync.OnceFunc(func() { close(g.gate); close(held) })
				defer release()
				publishing := make(chan struct{})
				var first atomic.Bool
				fs.diskCache.hooks.beforeFilePublication = func() {
					if hold == "publication" && first.CompareAndSwap(false, true) {
						close(publishing)
						<-held
					}
				}
				prefetch := func(kind string) error {
					ctx := context.Background()
					if strings.HasPrefix(kind, "index") {
						ctx = WithFileServicePolicy(ctx, IndexFileObject)
					}
					if strings.HasSuffix(kind, "file") {
						return fs.PrefetchFile(ctx, "object")
					}
					return fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: int64(len(data))})
				}
				firstDone := make(chan error, 1)
				go func() { firstDone <- prefetch(sq.first) }()
				switch hold {
				case "download":
					require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 1 }, 5*time.Second, time.Millisecond)
				case "publication":
					<-publishing
				}
				if !overlap {
					release()
					require.NoError(t, <-firstDone)
					waitPrefetcherIdle(t, fs)
					flushPublishers(fs)
				}
				require.NoError(t, prefetch(sq.second))
				if overlap {
					synctest.Wait()
					release()
					require.NoError(t, <-firstDone)
				}
				waitPrefetcherIdle(t, fs)
				flushPublishers(fs)
				require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))))
				require.True(t, fs.indexObjectHeld("object"))
				require.EqualValues(t, 2, storage.reads.Load())
			})
		})
	}
	for _, held := range []string{"index", "disk"} {
		t.Run(held, func(t *testing.T) {
			fs, _, data := newIndexObjectFS(t, true)
			s := &jpStorage{ObjectStorage: fs.server.raw}
			fs.server.raw = s
			key := fs.pathToKey("object")
			st := newJPStall(0)
			releaseStall := sync.OnceFunc(func() { close(st.release) })
			defer releaseStall()
			var once sync.Once
			s.stall = func(r jpRead) *jpStall {
				var got *jpStall
				if r.key == key {
					once.Do(func() { got = st })
				}
				return got
			}
			ctx := context.Background()
			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
			readPolicy := Policy(IndexFileObject)
			if held == "index" {
				require.NoError(t, fs.PrefetchObject(WithFileServicePolicy(ctx, IndexFileObject), req))
				readPolicy = 0
			} else {
				require.NoError(t, fs.PrefetchObject(ctx, req))
			}
			require.Eventually(t, func() bool { return len(s.readsOf(key)) == 1 }, 5*time.Second, time.Millisecond)

			done := make(chan jpResult, 1)
			go func() {
				ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
				defer cancel()
				vec := &IOVector{FilePath: "object", Policy: readPolicy | SkipFullFilePreloads | SkipMemoryCache, Entries: []IOEntry{{Offset: 100, Size: 10}}}
				if err := fs.Read(ctx, vec); err != nil {
					done <- jpResult{err: err}
					return
				}
				got := slices.Clone(vec.Entries[0].Data)
				vec.Release()
				done <- jpResult{data: got}
			}()
			got := jpWait(t, done, 3*time.Second, "the read of the other cache")
			require.NoError(t, got.err)
			require.Equal(t, data[100:110], got.data)
			require.GreaterOrEqual(t, len(s.readsOf(key)), 2, "the read reads its bytes itself")

			releaseStall()
			probeWaitIdle(t, fs)
			if held == "index" {
				require.Eventually(t, func() bool { return fs.indexObjectHeld("object") }, 5*time.Second, time.Millisecond)
			} else {
				require.Eventually(t, func() bool {
					return fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data))))
				}, 5*time.Second, time.Millisecond)
			}
		})
	}
}

// probeRead reads r of the object as a column read within a deadline of 2 s.
func probeRead(fs *S3FS, r PrefetchRange) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return readColumnRange(ctx, fs, r)
}

// A prefetch of bytes a running fill does not hold gets a fill of its own at once, read
// while the running fill still reads. Control: a prefetch of bytes the running fill holds
// reads nothing more.
func TestFillProbeOtherBytesOfRunningObjectStartAtOnce(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
	key := fs.pathToKey("object")
	g := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	defer release()
	ctx := context.Background()
	first := PrefetchRange{Offset: 0, Size: 1 << 10}
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: first.Size, Ranges: []PrefetchRange{first}}))
	require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 1 }, 5*time.Second, time.Millisecond)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: first.Size, Ranges: []PrefetchRange{first}}))
	require.Never(t, func() bool { cur, _, _ := g.counts(key); return cur > 1 }, 50*time.Millisecond, time.Millisecond,
		"bytes a running fill holds need nothing more")
	other := PrefetchRange{Offset: 100 << 10, Size: 1 << 10}
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: other.Size, Ranges: []PrefetchRange{other}}))
	require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 2 }, 2*time.Second, time.Millisecond,
		"other bytes of the object get a fill of their own at once")
	release()
	probeWaitIdle(t, fs)
	_, _, total := g.counts(key)
	require.Equal(t, 2, total)
}

// Two prefetches of the same bytes make one waiting fill with the sooner due time, the
// larger share and both scans: it stays live while one scan does, and is read once.
func TestFillProbeMergeOfSameBytes(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
	releaseBusy := probeBusy(t, fs)
	key := fs.pathToKey("object")
	g := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return false })
	fs.server.raw = g
	ctx := context.Background()
	r := PrefetchRange{Offset: 40 << 10, Size: 1 << 10}
	s1, s2 := &PrefetchScan{}, &PrefetchScan{}
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size, Due: 30, Reads: 1, Blocks: 10,
		Ranges: []PrefetchRange{r}, Scan: s1}))
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size, Due: 10, Reads: 5, Blocks: 10,
		Ranges: []PrefetchRange{r}, Scan: s2}))
	fills := probeWaiting(fs, "object")
	require.Len(t, fills, 1)
	require.Equal(t, int64(10), fills[0].due)
	require.Equal(t, 5, fills[0].share)
	require.ElementsMatch(t, []*PrefetchScan{s1, s2}, fills[0].scans)
	s1.End()
	releaseBusy()
	probeWaitIdle(t, fs)
	_, _, total := g.counts(key)
	require.Equal(t, 1, total, "the merged fill lives on its other scan and is read once")
}

// When the waiting background fills hold their most bytes, a new fill drops those whose
// scans ended first, then the last in order when it starts before it; one that starts
// after every fill, or is larger than the bound, is dropped and drops nothing.
func TestFillProbeByteBoundDropOrder(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
	probeBusy(t, fs)
	limit := probeQueueLimit(fs)
	q := limit / 4
	ctx := context.Background()
	add := func(name string, due int64, scan *PrefetchScan) {
		require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: name, Size: q, Due: due,
			Ranges: []PrefetchRange{{Offset: 0, Size: q}}, Scan: scan}))
	}
	var scans []*PrefetchScan
	scan := func() *PrefetchScan {
		s := &PrefetchScan{}
		scans = append(scans, s)
		return s
	}
	defer func() {
		for _, s := range scans {
			s.End()
		}
	}()
	ended := scan()
	for _, f := range []struct {
		name string
		due  int64
		scan *PrefetchScan
	}{{"a", 10, scan()}, {"ended", 5, ended}, {"c", 20, scan()}, {"d", 25, scan()}} {
		add(f.name, f.due, f.scan)
	}
	ended.End()
	queued := func() []string {
		var out []string
		for _, name := range []string{"a", "ended", "c", "d", "e", "f", "g", "h"} {
			if probeQueued(fs, name) {
				out = append(out, name)
			}
		}
		return out
	}
	require.Equal(t, []string{"a", "ended", "c", "d"}, queued())
	add("e", 15, scan())
	require.Equal(t, []string{"a", "c", "d", "e"}, queued(), "the fill of an ended scan goes first")
	add("f", 30, scan())
	require.Equal(t, []string{"a", "c", "d", "e"}, queued(), "a fill after every fill is dropped")
	add("g", 1, scan())
	require.Equal(t, []string{"a", "c", "e", "g"}, queued(), "the last in order is dropped for one before it")
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "h", Size: limit + 1, Due: 0,
		Ranges: []PrefetchRange{{Offset: 0, Size: limit + 1}}, Scan: scan()}))
	require.Equal(t, []string{"a", "c", "e", "g"}, queued(), "a fill larger than the bound drops nothing")
}

// The end of a scan drops its waiting fill no read joined, which never reads; a fill a
// read joined after its scan ended is read and serves the read.
func TestFillProbeScanEndDropsUnjoinedFill(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	releaseBusy := probeBusy(t, fs)
	ctx := context.Background()
	unjoined, joined := PrefetchRange{Offset: 0, Size: 1 << 10}, PrefetchRange{Offset: 100 << 10, Size: 1 << 10}
	scan := &PrefetchScan{}
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 2 << 10,
		Ranges: []PrefetchRange{unjoined, joined}, Scan: scan}))
	starts := &startRecorder{ObjectStorage: fs.server.raw}
	fs.server.raw = starts
	scan.End()
	got, err := probeRead(fs, joined)
	require.NoError(t, err)
	require.Equal(t, data[joined.Offset:joined.Offset+joined.Size], got)
	releaseBusy()
	probeWaitIdle(t, fs)
	require.Equal(t, []int64{joined.Offset}, starts.offsets(), "the fill no read joined is dropped")
}

// startRecorder records the offset each range read of an object starts at.
type startRecorder struct {
	ObjectStorage
	mu     sync.Mutex
	starts []int64
}

func (s *startRecorder) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	var off int64
	if min != nil {
		off = *min
	}
	s.mu.Lock()
	s.starts = append(s.starts, off)
	s.mu.Unlock()
	return s.ObjectStorage.Read(ctx, key, min, max)
}

func (s *startRecorder) offsets() []int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]int64(nil), s.starts...)
}

// The buffers of memory fills never exceed the memory budget: background fills take at
// most half of it, the others running file-backed at once, and a foreground read gets a
// buffer and returns while they hold theirs.
// A read that joins a running background fill moves its buffer to the foreground charge.
// Every range is read once.
func TestFillProbeFlightBytesWithinBudget(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	buf := flightBufSize(1 << 10)
	budget := 4 * buf
	fs.server.memoryLimit = budget
	key := fs.pathToKey("object")
	g := newBlackBoxStorage(fs.server.raw, func(_ string, off int64) bool { return off < 200<<10 })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	defer release()
	var over atomic.Int64
	stop := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		for {
			select {
			case <-stop:
				return
			default:
			}
			if bytes, _ := probeFlightBytes(fs); bytes > budget {
				over.Store(bytes)
			}
			time.Sleep(50 * time.Microsecond)
		}
	}()
	ranges := columnRanges(6, 20<<10, 1<<10)
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: 6 << 10,
		Ranges: ranges, Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == len(ranges) }, 5*time.Second, time.Millisecond,
		"every range runs at once")
	bytes, background := probeFlightBytes(fs)
	require.Equal(t, 2*buf, bytes)
	require.Equal(t, 2*buf, background)

	bystander := PrefetchRange{Offset: 250 << 10, Size: 1 << 10}
	got, err := probeRead(fs, bystander)
	require.NoError(t, err, "a foreground read gets a buffer while background fills hold theirs")
	require.Equal(t, data[bystander.Offset:bystander.Offset+bystander.Size], got)

	var inMemory PrefetchRange
	f := &fs.diskCache.prefetchObjects
	f.mu.Lock()
	for _, fl := range f.m[fs.diskCache.objectKey("object")] {
		if o := fl.joinableDownload(); o != nil && o.memory && o.class == backgroundRequest {
			inMemory = PrefetchRange{Offset: fl.start, Size: fl.end - fl.start}
		}
	}
	f.mu.Unlock()
	require.NotZero(t, inMemory.Size, "a background fill holds a buffer")
	joined := make(chan error, 1)
	go func() {
		_, err := probeRead(fs, inMemory)
		joined <- err
	}()
	require.Eventually(t, func() bool { _, background := probeFlightBytes(fs); return background == buf }, 2*time.Second,
		time.Millisecond, "the joined fill's buffer moves to the foreground charge")
	release()
	require.NoError(t, <-joined)
	probeWaitIdle(t, fs)
	close(stop)
	<-sampled
	require.Zero(t, over.Load(), "the flight bytes went over the budget")
	bytes, background = probeFlightBytes(fs)
	require.Zero(t, bytes)
	require.Zero(t, background)
	_, _, total := g.counts(key)
	require.Equal(t, len(ranges)+1, total, "each range and the bystander are read once")
}

// A read that joins a waiting fill of a whole object large enough to split starts its
// first part at the block the read needs.
func TestFillProbeSplitStartsAtJoinedReadBlock(t *testing.T) {
	fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
	probeBusy(t, fs)
	starts := &startRecorder{ObjectStorage: fs.server.raw}
	fs.server.raw = starts
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: int64(len(data))}))
	got, err := readFullObjectOf(fs, 100<<10, 10)
	require.NoError(t, err)
	require.Equal(t, data[100<<10:100<<10+10], got)
	require.Eventually(t, func() bool { return fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))) },
		10*time.Second, time.Millisecond)
	offsets := starts.offsets()
	require.Len(t, offsets, 2)
	require.Equal(t, int64(64<<10), offsets[0], "the first part starts at the read's block")
}

// The fills of one object into the index-file cache and the disk cache are apart: a read
// of the disk cache joins only the disk cache's fill, and each cache is filled once.
// Control: the index-file cache's fill of another object.
func TestFillProbeIndexAndDiskFillsIndependent(t *testing.T) {
	for _, indexPath := range []string{"object", "another-object"} {
		t.Run(indexPath, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			ctx := context.Background()
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "another-object", Policy: SkipAllCache,
				Entries: []IOEntry{{Size: int64(len(data)), Data: data}}}))
			releaseBusy := probeBusy(t, fs)
			require.NoError(t, fs.PrefetchObject(WithFileServicePolicy(ctx, IndexFileObject),
				ObjectPrefetch{FilePath: indexPath, Size: int64(len(data))}))
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: int64(len(data))}))
			vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache, Entries: []IOEntry{{Offset: 100, Size: 10}}}
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[100:110], vec.Entries[0].Data)
			vec.Release()
			require.True(t, probeQueued(fs, indexPath), "the index-file cache's fill still waits")
			require.False(t, fs.indexObjectHeld(indexPath))
			releaseBusy()
			probeWaitIdle(t, fs)
			require.True(t, fs.indexObjectHeld(indexPath))
			require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))))
			require.EqualValues(t, 2, storage.reads.Load(), "each cache is filled once")
		})
	}
}

// A delete of the object, or the closing of the file service, reaches its fills, memory or
// file backed ranges or a whole object, in each state: waiting, joined by a read and
// waiting for room (memory ranges), waiting for background room, and running beside a waiting
// sibling. Afterwards no
// fill, buffer or room is held, the joined read returns, no fill reads object storage
// later, and after a delete nothing of the object is cached and a read of the object
// written again gets its new bytes. A delete reaches a waiting fill through the path
// qualified by the service.
func TestFillProbeDeleteAndCloseReachEveryState(t *testing.T) {
	for _, event := range []string{"delete", "close"} {
		for _, kind := range []string{"range", "file range", "whole"} {
			for _, state := range []string{"waiting", "promoted", "room", "running"} {
				if kind == "whole" && state == "promoted" {
					continue
				}
				t.Run(event+"/"+kind+"/"+state, func(t *testing.T) {
					fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
					ctx := context.Background()
					r := PrefetchRange{Offset: 40 << 10, Size: 1 << 10}
					req := ObjectPrefetch{FilePath: "object", Size: int64(len(data)), Scan: &PrefetchScan{}}
					if kind != "whole" {
						req.Ranges = columnRanges(3, 40<<10, 1<<10)
						req.Size = 3 << 10
					}
					if kind == "file range" {
						fs.diskCache.memoryPrefetchMax = 1
					}
					key := fs.pathToKey("object")
					g := newBlackBoxStorage(fs.server.raw, func(k string, _ int64) bool { return state == "running" && k == key })
					fs.server.raw = g
					releaseGate := sync.OnceFunc(func() { close(g.gate) })
					defer releaseGate()
					joined := make(chan error, 1)
					freeRoom := func() {}
					switch state {
					case "waiting", "promoted":
						probeBusy(t, fs)
					case "running":
						fs.diskCache.setFillsMax(2)
					}
					baseRoom := heldRoom(fs.server)
					baseBytes, _ := probeFlightBytes(fs)
					if state == "room" {
						freeRoom = occupy(t, fs.server, backgroundRequest, fs.server.backgroundMax)
					}
					require.NoError(t, fs.PrefetchObject(ctx, req))
					opens := 0
					switch state {
					case "waiting":
						require.True(t, probeQueued(fs, "object"))
					case "promoted":
						freeRoom = fillRoom(t, fs.server)
						go func() {
							_, err := readColumnRange(withReadMark(ctx), fs, r)
							joined <- err
						}()
						require.Eventually(t, func() bool { return probeJoinedRunning(fs, "object", kind == "file range") },
							2*time.Second, time.Millisecond)
					case "room":
						require.Eventually(t, func() bool { _, waiting := fs.server.counts(); return waiting[backgroundRequest] > 0 },
							2*time.Second, time.Millisecond)
					case "running":
						opens = 1
						if kind != "whole" {
							opens = 2
						}
						require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == opens }, 5*time.Second, time.Millisecond)
					}
					if event == "delete" {
						path := "object"
						if state == "waiting" {
							path = JoinPath(fs.Name(), path)
						}
						// The delete drops the fills before its own request waits for room.
						deleted := make(chan error, 1)
						go func() { deleted <- fs.Delete(ctx, path) }()
						require.Eventually(t, func() bool { return !probeUnderWay(fs, "object") }, 2*time.Second, time.Millisecond)
						freeRoom()
						require.NoError(t, <-deleted)
					} else {
						fs.Close(ctx)
					}
					freeRoom()
					if state == "promoted" {
						select {
						case <-joined:
						case <-time.After(2 * time.Second):
							t.Fatal("the joined read did not return")
						}
					}
					require.Eventually(t, func() bool {
						return !probeQueued(fs, "object") && !probeUnderWay(fs, "object")
					}, 2*time.Second, time.Millisecond)
					require.Eventually(t, func() bool {
						bytes, _ := probeFlightBytes(fs)
						return heldRoom(fs.server) <= baseRoom && bytes <= baseBytes
					}, 2*time.Second, time.Millisecond, "the fill holds no room and no buffer")
					// The joined read reads the bytes of the stopped fill itself.
					require.Equal(t, opens, g.unmarked(key))
					releaseGate()
					require.Never(t, func() bool { return g.unmarked(key) > opens }, 50*time.Millisecond,
						time.Millisecond, "no fill reads object storage later")
					if event == "close" {
						return
					}
					require.Empty(t, fs.diskCache.objectFiles("object"))
					rewritten := slices.Clone(data)
					for i := range rewritten {
						rewritten[i] ^= 0xff
					}
					require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Policy: SkipAllCache,
						Entries: []IOEntry{{Size: int64(len(rewritten)), Data: rewritten}}}))
					got, err := readColumnRange(ctx, fs, r)
					require.NoError(t, err)
					require.Equal(t, rewritten[r.Offset:r.Offset+r.Size], got)
				})
			}
		}
	}
}

// holdForegroundBuffer takes n bytes of the memory of the server of fs, as a foreground
// flight does, until the returned function or the test's end.
func holdForegroundBuffer(t *testing.T, fs *S3FS, n int64) func() {
	require.True(t, fs.server.takeMemory(n, foregroundRequest))
	free := sync.OnceFunc(func() { probeUncharge(fs, n) })
	t.Cleanup(free)
	return free
}

// probeFill is what the probes observe of a waiting fill.
type probeFill struct {
	due   int64
	share int
	scans []*PrefetchScan
}

// probeBusy keeps the fills of fs prefetches make waiting until the returned function.
func probeBusy(t *testing.T, fs *S3FS) func() { return busyFills(t, fs) }

func probeQueued(fs *S3FS, filePath string) bool { return fs.diskCache.isQueued(filePath) }

func probeWaiting(fs *S3FS, filePath string) []probeFill {
	fills := fs.diskCache.waitingFillsOf(filePath)
	out := make([]probeFill, 0, len(fills))
	for _, q := range fills {
		out = append(out, probeFill{due: q.req.Due, share: q.req.share(), scans: q.scans})
	}
	return out
}

// probeJoinedRunning reports whether a read joined a started fill a prefetch made of the
// object at filePath: it holds its download, or, unless download is set, waits for it.
func probeJoinedRunning(fs *S3FS, filePath string, download bool) bool {
	f := &fs.diskCache.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, q := range f.m[fs.diskCache.objectKey(filePath)] {
		q.mu.Lock()
		holds := q.holds
		q.mu.Unlock()
		if q.queued && q.pos < 0 && (holds > 0 || !download && q.joiners > 0) {
			return true
		}
	}
	return false
}

func probeWaitIdle(t *testing.T, fs *S3FS) { waitFillsIdle(t, fs.diskCache) }

func probeQueueLimit(fs *S3FS) int64 { return fs.diskCache.fillQueueLimit() }

func probeFlightBytes(fs *S3FS) (bytes, background int64) {
	used := fs.server.memoryInUse()
	return used[foregroundRequest] + used[backgroundRequest], used[backgroundRequest]
}

func probeUncharge(fs *S3FS, n int64) {
	fs.server.freeMemory(n, foregroundRequest)
}

// probeUnderWay reports whether a fill a prefetch made of the object at filePath runs.
func probeUnderWay(fs *S3FS, filePath string) bool {
	f := &fs.diskCache.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.ContainsFunc(f.m[fs.diskCache.objectKey(filePath)], func(p *cacheFill) bool { return p.queued && p.pos < 0 })
}

// waitRoom waits for room of class c of g.
func waitRoom(g *objectStorageServer, ctx context.Context, c requestClass) error {
	_, err := g.wait(ctx, c)
	return err
}

// prefetchObject queues the fills of req, whatever is cached, and waits until they end.
func (s *S3FS) prefetchObject(_ context.Context, req ObjectPrefetch) {
	path, err := parseFilePathAtService(req.FilePath, s.name)
	if err != nil || req.Size <= 0 {
		return
	}
	req.FilePath = path.File
	s.diskCache.addFills(req, false, s.source(req.FilePath))
	waitFillsEnd(s.diskCache, req.FilePath)
}

// waitFillsEnd waits until no fill of the object at filePath waits or runs.
func waitFillsEnd(d *DiskCache, filePath string) { waitRegistered(d, filePath, true) }

// isQueued reports whether a fill of the object at filePath waits to start.
func (d *DiskCache) isQueued(filePath string) bool {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.ContainsFunc(f.m[d.objectKey(filePath)], func(q *cacheFill) bool { return q.pos >= 0 })
}

// fillState is what tests observe of a queued fill.
type fillState struct {
	req        ObjectPrefetch
	start, end int64
	whole      bool
	scans      []*PrefetchScan
	joiners    int
}

// waitingFillsOf returns the fills of the object at filePath into the disk cache waiting to
// start.
func (d *DiskCache) waitingFillsOf(filePath string) []fillState {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []fillState
	for _, q := range f.m[d.objectKey(filePath)] {
		if q.pos >= 0 && !q.index {
			out = append(out, fillState{req: q.req, start: q.start, end: q.end, whole: q.whole,
				scans: slices.Clone(q.scans), joiners: q.joiners})
		}
	}
	return out
}

// setFillsMax sets the most background fills of d run at once.
func (d *DiskCache) setFillsMax(n int) {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fillsMax = n
}

// busyStorage serves the object busy itself once gate is closed, and the other objects
// from its upstream.
type busyStorage struct {
	ObjectStorage
	key     string
	gate    chan struct{}
	reading chan struct{}
	once    sync.Once
}

func (b *busyStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	if key != b.key {
		return b.ObjectStorage.Read(ctx, key, min, max)
	}
	b.once.Do(func() { close(b.reading) })
	select {
	case <-b.gate:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return io.NopCloser(bytes.NewReader([]byte{1})), nil
}

// busyFills lets fs run one background fill at once, and runs one, of the object "busy",
// whose read of object storage waits until the returned function or the test's end.
func busyFills(t *testing.T, fs *S3FS) func() {
	b := &busyStorage{ObjectStorage: fs.server.raw, key: fs.pathToKey("busy"),
		gate: make(chan struct{}), reading: make(chan struct{})}
	fs.server.raw = b
	fs.diskCache.setFillsMax(1)
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "busy", Size: 1,
		Ranges: []PrefetchRange{{Offset: 0, Size: 1}}}))
	select {
	case <-b.reading:
	case <-time.After(5 * time.Second):
		t.Fatal("the busy fill did not start")
	}
	release := sync.OnceFunc(func() { close(b.gate) })
	t.Cleanup(release)
	return release
}

// fillsRunning returns the background fills g counts.
func fillsRunning(g *objectStorageServer) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.fills
}

// waitFillsIdle waits until d has no fill waiting or running and no prefetch object.
func waitFillsIdle(t *testing.T, d *DiskCache) {
	t.Helper()
	f := &d.prefetchObjects
	require.Eventually(t, func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return len(f.waitingFills) == 0 && fillsRunning(d.server) == 0 && len(f.m) == 0
	}, 10*time.Second, time.Millisecond)
}

// waitObjectsEnd waits until no fill of the object at filePath runs.
func waitObjectsEnd(d *DiskCache, filePath string) { waitRegistered(d, filePath, false) }

// waitRegistered waits until no fill of the object at filePath runs, nor waits when
// waiting is set.
func waitRegistered(d *DiskCache, filePath string, waiting bool) {
	f := &d.prefetchObjects
	diskPath := d.objectKey(filePath)
	for deadline := time.Now().Add(30 * time.Second); time.Now().Before(deadline); time.Sleep(time.Millisecond) {
		f.mu.Lock()
		n := 0
		for _, p := range f.m[diskPath] {
			if waiting || p.pos < 0 {
				n++
			}
		}
		f.mu.Unlock()
		if n == 0 {
			return
		}
	}
}

// keyRecorder records the key of each read of object storage.
type keyRecorder struct {
	ObjectStorage
	mu   sync.Mutex
	keys []string
}

func (k *keyRecorder) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	k.mu.Lock()
	k.keys = append(k.keys, key)
	k.mu.Unlock()
	return k.ObjectStorage.Read(ctx, key, min, max)
}

// The waiting fills start one at a time in order: a larger share first, an unknown share as
// the whole, then the sooner due time, then arrival.
func TestFillsStartInOrder(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 4<<10)
	ctx := context.Background()
	names := []string{"partial", "late", "unknown", "first", "second"}
	for _, name := range names {
		require.NoError(t, fs.Write(ctx, IOVector{FilePath: name, Entries: []IOEntry{{Size: 1 << 10, Data: data[:1<<10]}},
			Policy: SkipDiskCache | SkipMemoryCache}))
	}
	release := busyFills(t, fs)
	rec := &keyRecorder{ObjectStorage: fs.server.raw}
	fs.server.raw = rec
	add := func(name string, due int64, reads, blocks int) {
		require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: name, Size: 1 << 10, Due: due, Reads: reads, Blocks: blocks,
			Ranges: []PrefetchRange{{Offset: 0, Size: 1 << 10}}}))
	}
	add("partial", 1, 1, 10)
	add("late", 30, 10, 10)
	add("unknown", 20, 0, 10)
	add("first", 10, 10, 10)
	add("second", 10, 10, 10)
	release()
	for _, name := range names {
		waitFillsEnd(fs.diskCache, name)
	}
	want := []string{"first", "second", "unknown", "late", "partial"}
	for i := range want {
		want[i] = fs.pathToKey(want[i])
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	require.Equal(t, want, rec.keys)
}

// oldWholeFiles returns the files under dir named as whole objects were before.
func oldWholeFiles(t *testing.T, d *DiskCache) []string {
	var found []string
	require.NoError(t, filepath.WalkDir(d.path, func(path string, entry os.DirEntry, err error) error {
		require.NoError(t, err)
		if !entry.IsDir() && d.isOldWholeFile(path) {
			found = append(found, path)
		}
		return nil
	}))
	return found
}

// readObjectEntry reads size bytes at offset of the object at file from cache, and reports
// whether the cache served them and the bytes.
func readObjectEntry(t *testing.T, cache *DiskCache, file string, offset, size int64) (bool, []byte) {
	vec := &IOVector{FilePath: file, Entries: []IOEntry{{Offset: offset, Size: size}}}
	defer vec.Release()
	require.NoError(t, cache.Read(context.Background(), vec))
	return vec.Entries[0].done, bytes.Clone(vec.Entries[0].Data)
}

// A whole object is cached as the range [0, size), which serves reads of its sub-ranges.
func TestWholeObjectCachedAsRangeServesSubRange(t *testing.T) {
	ctx := context.Background()
	cache := newLifecycleTestDiskCache(t)
	data := rangeTestData(100)
	require.NoError(t, cache.SetFile(ctx, "obj", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(data)), nil
	}))

	whole := cache.wholePath("obj", int64(len(data)))
	require.Equal(t, []string{whole}, cache.objectFiles("obj"))
	require.Empty(t, oldWholeFiles(t, cache))
	require.Equal(t, []cachedRange{{diskPath: whole, offset: 0, size: int64(len(data))}}, cache.ranges.of(cache.objectKey("obj")))

	done, got := readObjectEntry(t, cache, "obj", 10, 20)
	require.True(t, done)
	require.Equal(t, data[10:30], got)
}

// A whole object of unknown size is published as the range of the size its fill finds,
// which is recorded.
func TestWholeObjectOfUnknownSizePublishedAtFoundSize(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
	d := fs.diskCache
	_, known := d.prefetched.size(d.objectKey("object"))
	require.False(t, known)

	require.NoError(t, fs.PrefetchFile(context.Background(), "object"))
	flushPublishers(fs)
	require.Equal(t, int64(1), storage.reads.Load())
	require.Equal(t, []string{d.wholePath("object", int64(len(data)))}, d.objectFiles("object"))
	size, known := d.prefetched.size(d.objectKey("object"))
	require.True(t, known)
	require.Equal(t, int64(len(data)), size)

	got, err := readFullObjectOf(fs, 1000, 10)
	require.NoError(t, err)
	require.Equal(t, data[1000:1010], got)
	require.Equal(t, int64(1), storage.reads.Load(), "served from the cached range")
}

// A prefetch of an object whose size no prefetch told learns it from object storage when
// a cached range starts at the object's start, and downloads only what is not cached; a
// failed stat leaves it to download the object as one of unknown size.
func TestPrefetchOfWarmWholeObjectLearnsItsSize(t *testing.T) {
	startCached := func(t *testing.T, fs *S3FS, data []byte) {
		require.NoError(t, fs.diskCache.Update(context.Background(), &IOVector{FilePath: "object",
			Entries: []IOEntry{{Offset: 0, Size: 100, Data: data[:100]}}}, false))
	}
	for _, tc := range []struct {
		name      string
		cached    func(*testing.T, *S3FS, []byte)
		stats     int64
		download  bool
		statFails bool
	}{
		{name: "whole object cached", stats: 1, cached: func(t *testing.T, fs *S3FS, _ []byte) {
			require.NoError(t, fs.PrefetchFile(context.Background(), "object"))
			flushPublishers(fs)
		}},
		{name: "start of the object cached", stats: 1, download: true, cached: startCached},
		{name: "stat fails", stats: 1, download: true, statFails: true, cached: startCached},
		{name: "nothing from the start cached", download: true, cached: func(t *testing.T, fs *S3FS, data []byte) {
			require.NoError(t, fs.diskCache.Update(context.Background(), &IOVector{FilePath: "object",
				Entries: []IOEntry{{Offset: 100, Size: 100, Data: data[100:200]}}}, false))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
			d := fs.diskCache
			tc.cached(t, fs, data)
			// The sizes known in the process are lost, as at a restart.
			d.prefetched = prefetchedObjects{}
			stats := &statCountingStorage{ObjectStorage: storage}
			if tc.statFails {
				stats.ObjectStorage = failingStatStorage{storage}
			}
			fs.server.raw = stats
			reads := storage.reads.Load()

			require.NoError(t, fs.PrefetchFile(context.Background(), "object"))
			flushPublishers(fs)
			require.Equal(t, tc.stats, stats.stats.Load())
			if tc.download {
				require.Equal(t, reads+1, storage.reads.Load())
			} else {
				require.Equal(t, reads, storage.reads.Load(), "the cached whole object is not downloaded again")
			}
			require.True(t, d.cache.Contains(d.wholePath("object", int64(len(data)))))
		})
	}
}

type failingStatStorage struct{ ObjectStorage }

func (failingStatStorage) Stat(context.Context, string) (int64, error) {
	return 0, os.ErrPermission
}

// A delete of an object removes its whole object as one of its ranges, and an eviction
// removes it as one range.
func TestWholeObjectDeleteAndEvictionAsOneRange(t *testing.T) {
	ctx := context.Background()
	data := rangeTestData(100)
	fill := func(t *testing.T, cache *DiskCache) string {
		require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "obj",
			Entries: []IOEntry{{Offset: 10, Size: 10, Data: data[10:20]}}}, false))
		require.NoError(t, cache.SetFile(ctx, "obj", func(context.Context) (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(data)), nil
		}))
		whole := cache.wholePath("obj", int64(len(data)))
		require.True(t, cache.cache.Contains(whole))
		require.Len(t, cache.ranges.of(cache.objectKey("obj")), 2)
		cache.preloads.mu.Lock()
		_, tracked := cache.preloads.served[whole]
		cache.preloads.mu.Unlock()
		require.True(t, tracked, "the whole object's file counts its reads")
		return whole
	}
	untracked := func(t *testing.T, cache *DiskCache, whole string) {
		cache.preloads.mu.Lock()
		defer cache.preloads.mu.Unlock()
		_, tracked := cache.preloads.served[whole]
		require.False(t, tracked)
	}

	t.Run("delete", func(t *testing.T) {
		cache := newLifecycleTestDiskCache(t)
		whole := fill(t, cache)
		require.NoError(t, cache.DeletePaths(ctx, []string{"obj"}))
		require.Empty(t, cache.objectFiles("obj"))
		require.Empty(t, cache.ranges.of(cache.objectKey("obj")))
		require.False(t, cache.cache.Contains(whole))
		untracked(t, cache, whole)
		_, known := cache.prefetched.size(cache.objectKey("obj"))
		require.False(t, known, "the delete forgets the size")
		done, _ := readObjectEntry(t, cache, "obj", 30, 10)
		require.False(t, done)
	})

	t.Run("eviction", func(t *testing.T) {
		cache := newLifecycleTestDiskCache(t)
		whole := fill(t, cache)
		cache.cache.Delete(ctx, whole)
		cache.Flush(ctx)
		require.NoFileExists(t, whole)
		require.Equal(t, []cachedRange{{diskPath: cache.pathForIOEntry("obj", IOEntry{Offset: 10, Size: 10}), offset: 10, size: 10}},
			cache.ranges.of(cache.objectKey("obj")), "the other range stays")
		untracked(t, cache, whole)
		done, _ := readObjectEntry(t, cache, "obj", 30, 10)
		require.False(t, done)
		done, got := readObjectEntry(t, cache, "obj", 12, 5)
		require.True(t, done)
		require.Equal(t, data[12:17], got)
	})
}

// Cache files named as whole objects by an earlier version are removed at startup, from
// the disk cache and the index-file cache; range files stay, also those of an object named
// "full", and no whole object is named as an old one.
func TestStartupRemovesOldWholeFiles(t *testing.T) {
	ctx := context.Background()
	for _, dirEntries := range []bool{false, true} {
		name := "disk cache"
		if dirEntries {
			name = "index-file cache"
		}
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			write := func(rel, content string) string {
				path := filepath.Join(dir, rel)
				require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
				require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
				return path
			}
			old := []string{
				write(filepath.Join(diskCacheShard("obj"), "fullobj"+cacheFileSuffix), "whole"),
				write(filepath.Join(diskCacheShard("dir/obj"), "fulldir", "obj"+cacheFileSuffix), "whole"),
				write("fullfoo"+cacheFileSuffix, "whole"),
			}
			kept := write(filepath.Join(diskCacheShard("obj"), "0-3obj"+cacheFileSuffix), "abc")
			keptFull := write(filepath.Join(diskCacheShard("fullobj"), "0-3fullobj"+cacheFileSuffix), "abc")

			cache, err := newDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "", "",
				diskCacheOptions{dirEntries: dirEntries})
			require.NoError(t, err)
			defer cache.Close(ctx)
			for _, path := range old {
				require.NoFileExists(t, path)
				require.False(t, cache.cache.Contains(path))
			}
			for _, path := range []string{kept, keptFull} {
				require.FileExists(t, path)
				require.True(t, cache.cache.Contains(path))
			}
			require.False(t, cache.isOldWholeFile(cache.wholePath("full", 3)))
			if !dirEntries {
				done, got := readObjectEntry(t, cache, "obj", 1, 2)
				require.True(t, done)
				require.Equal(t, []byte("bc"), got)
			}
		})
	}
}

// Fix-missing is accepted and ignored: no object is restored from the disk cache.
func TestFixMissingIsIgnored(t *testing.T) {
	ctx := context.Background()
	cacheDir := t.TempDir()
	old := filepath.Join(cacheDir, diskCacheShard("obj"), "fullobj"+cacheFileSuffix)
	require.NoError(t, os.MkdirAll(filepath.Dir(old), 0o755))
	require.NoError(t, os.WriteFile(old, []byte("whole"), 0o644))
	fs, err := NewFileService(ctx, Config{
		Name:       "s3",
		Backend:    s3FileServiceBackend,
		S3:         ObjectStorageArguments{Endpoint: "disk", Bucket: t.TempDir()},
		Cache:      CacheConfig{DiskPath: &cacheDir, DiskCapacity: ptrTo[toml.ByteSize](1 << 20)},
		FixMissing: true,
	}, nil)
	require.NoError(t, err)
	defer fs.Close(ctx)
	_, err = fs.StatFile(ctx, "obj")
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "%v", err)
	<-fs.(*S3FS).diskCache.load.done
	require.NoFileExists(t, old)
}

// ProbeStallStorage returns readers of whole reads that send their first n bytes, then wait
// For their context to end.
type probeStallStorage struct {
	ObjectStorage
	n     int64
	reads atomic.Int64
}

type probeStallReader struct {
	ctx  context.Context
	r    io.ReadCloser
	left int64
}

func (s *probeStallReader) Read(b []byte) (int, error) {
	if s.left <= 0 {
		<-s.ctx.Done()
		return 0, s.ctx.Err()
	}
	if int64(len(b)) > s.left {
		b = b[:s.left]
	}
	n, err := s.r.Read(b)
	s.left -= int64(n)
	return n, err
}

func (s *probeStallReader) Close() error { return s.r.Close() }

func (s *probeStallStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	s.reads.Add(1)
	r, err := s.ObjectStorage.Read(ctx, key, min, max)
	if err != nil || max != nil {
		return r, err
	}
	return &probeStallReader{ctx: ctx, r: r, left: s.n}, nil
}

// A disk-cache read under way when the cache closes keeps no file open past it.
func TestDiskCacheTransitionReadDuringCloseKeepsNoOpenFile(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 64<<10)
	ctx := context.Background()
	d := fs.diskCache
	require.NoError(t, d.Update(ctx, &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: 0, Size: 100, Data: data[:100]}}}, false))
	diskPath := d.pathForIOEntry("object", IOEntry{Offset: 0, Size: 100})
	require.True(t, d.cache.Contains(diskPath))
	// The read has taken the count of changes of the path when the cache closes.
	seq := d.files.seq(diskPath)
	d.Close(ctx)
	f, err := os.Open(diskPath)
	require.NoError(t, err)
	if kept := d.files.keep(diskPath, seq, f); kept != nil {
		d.files.close(openedDiskCacheFile{file: f, kept: kept})
	} else {
		require.NoError(t, f.Close())
	}
	d.files.mu.Lock()
	kept := len(d.files.m)
	d.files.mu.Unlock()
	require.Zero(t, kept, "a file kept open after Close is never closed")
}

// A range prefetch of at least the split size, memory- or file-backed, reads its range in
// one request and caches it as one range. Control: a whole object of the same size,
// prefetched first, splits into its parts.
// The whole-object download of a read or a prefetch of an object of at least the split
// size whose size is known reads its 64KiB-aligned parts, each its own request even with
// room for one, and caches the whole object, never asking object storage for its size; a
// smaller object or one of unknown size downloads in one open-ended request.
func TestRangeFillProbeOneRequestWholeObjectSplits(t *testing.T) {
	for _, tc := range []struct {
		name          string
		size          int
		min           int64
		unknown, room bool
		prefetch      bool
		starts        []int64
	}{
		{name: "split", size: 300<<10 + 17, min: 128 << 10, starts: []int64{0, 192 << 10}},
		{name: "room for one", size: 300 << 10, min: 128 << 10, room: true, starts: []int64{0, 192 << 10}},
		{name: "prefetch tells the size", size: 300 << 10, min: 128 << 10, unknown: true, prefetch: true, starts: []int64{0, 192 << 10}},
		{name: "small", size: 300 << 10, min: 1 << 20},
		{name: "unknown size", size: 300 << 10, min: 128 << 10, unknown: true},
	} {
		t.Run("whole/"+tc.name, func(t *testing.T) {
			fs, storage, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: tc.min}, tc.size)
			d := fs.diskCache
			if tc.unknown {
				d.prefetched = prefetchedObjects{}
			}
			if tc.room {
				limitRoom(t, fs.server, 1, 1)
			}
			stats := &statCountingStorage{ObjectStorage: fs.server.raw}
			fs.server.raw = stats
			if tc.prefetch {
				fs.prefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: int64(tc.size)})
			} else {
				got, err := readFullObjectOf(fs, 250<<10, 10)
				require.NoError(t, err)
				require.Equal(t, data[250<<10:250<<10+10], got)
			}
			flushPublishers(fs)
			reads := int64(max(1, len(tc.starts)))
			require.Equal(t, reads, storage.reads.Load())
			storage.mu.Lock()
			require.ElementsMatch(t, tc.starts, storage.starts)
			storage.mu.Unlock()
			require.Zero(t, stats.stats.Load())
			require.Zero(t, heldRoom(fs.server), "its requests free their room")
			require.True(t, d.cache.Contains(d.wholePath("object", int64(tc.size))))
			got, err := readFullObjectOf(fs, 0, int64(tc.size))
			require.NoError(t, err)
			require.Equal(t, data, got)
			require.Equal(t, reads, storage.reads.Load(), "the cached whole object serves later reads")
		})
	}
	const size = 1 << 20
	for _, backing := range []string{"memory", "file"} {
		t.Run(backing, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, 2<<20)
			d := fs.diskCache
			d.objectSplit = objectSplitConfig{parts: 2, min: 512 << 10}
			if backing == "file" {
				d.memoryPrefetchMax = 64 << 10
			}
			ctx := context.Background()
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "whole", Entries: []IOEntry{{Size: size, Data: data[:size]}},
				Policy: SkipDiskCache | SkipMemoryCache}))
			d.prefetched.store(d.objectKey("whole"), size)
			starts := &startRecorder{ObjectStorage: fs.server.raw}
			fs.server.raw = starts

			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "whole", Size: size, Scan: &PrefetchScan{}}))
			require.Eventually(t, func() bool { return d.cache.Contains(d.wholePath("whole", size)) },
				10*time.Second, time.Millisecond)
			probeWaitIdle(t, fs)
			require.ElementsMatch(t, []int64{0, 512 << 10}, starts.offsets(), "the whole object splits")

			r := PrefetchRange{Offset: 64 << 10, Size: size}
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size,
				Ranges: []PrefetchRange{r}, Scan: &PrefetchScan{}}))
			require.Eventually(t, func() bool {
				_, ok := d.ranges.covering(d.objectKey("object"), r.Offset, r.Size)
				return ok
			}, 10*time.Second, time.Millisecond)
			probeWaitIdle(t, fs)
			require.Equal(t, []int64{r.Offset}, starts.offsets()[2:], "the range is read in one request")
			cached, _ := d.ranges.covering(d.objectKey("object"), r.Offset, r.Size)
			require.Len(t, cached, 1, "the range is one cached range")
			got, err := readColumnRange(ctx, fs, PrefetchRange{Offset: r.Offset + 512<<10 - 100, Size: 200})
			require.NoError(t, err)
			require.Equal(t, data[r.Offset+512<<10-100:r.Offset+512<<10+100], got)
			require.Len(t, starts.offsets(), 3, "the read is served from the cache")
		})
	}
}

// The automatic prefetch unit is the bandwidth-delay product rounded up to a power of two
// within its bounds; a read of no bytes or no time leaves the cost model unchanged.
func TestAutoPrefetchUnit(t *testing.T) {
	require.Equal(t, int64(64<<10), autoPrefetchUnit(0))
	require.Equal(t, int64(64<<10), autoPrefetchUnit(1000))
	require.Equal(t, int64(4<<20), autoPrefetchUnit(25e-3*100*(1<<20)))
	require.Equal(t, int64(2<<20), autoPrefetchUnit(2<<20))
	require.Equal(t, int64(8<<20), autoPrefetchUnit(100e-3*2000*(1<<20)))
	require.Equal(t, int64(16<<20), prefetchRangeConfig{on: true, fixed: 16 << 20}.unit())
	require.Equal(t, int64(0), prefetchRangeConfig{fixed: 16 << 20}.unit())

	m := newRangeReadCostModel(0.02, 100<<20)
	m.observe(0, time.Second)
	m.observe(1<<20, 0)
	require.Equal(t, rangeReadCost{latency: 0.02, bytesPerS: 100 << 20}, m.get())
}

// A flight larger than the largest kept size gets a buffer of its own size, which is not
// kept; a buffer of a capacity not a kept size is not kept either.
func TestFlightBufBeyondKeptSizes(t *testing.T) {
	n := int64(1)<<flightBufMaxShift + 1
	require.Equal(t, n, flightBufSize(n))
	b := getFlightBuf(n)
	require.Equal(t, n, int64(len(b)))
	require.Equal(t, n, int64(cap(b)))
	putFlightBuf(b)
	putFlightBuf(make([]byte, 3))
	require.Equal(t, int64(1)<<flightBufMinShift, int64(cap(getFlightBuf(3))))
}

// A delete ends while fills of its object, a prefetch's and a read's, hold all the room in
// an object storage that does not answer: it stops them before its own request waits for
// room.
func TestDeleteEndsWhileFillsOfItsObjectHoldAllRoom(t *testing.T) {
	for _, kind := range []string{"memory", "file"} {
		t.Run(kind, func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
			if kind == "file" {
				fs.diskCache.memoryPrefetchMax = 1
			}
			ctx := context.Background()
			key := fs.pathToKey("object")
			g := newBlackBoxStorage(fs.server.raw, func(k string, _ int64) bool { return k == key })
			fs.server.raw = g
			release := sync.OnceFunc(func() { close(g.gate) })
			defer release()
			limitRoom(t, fs.server, 2, 1)
			ranges := columnRanges(2, 40<<10, 1<<10)
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 1 << 10, Ranges: ranges[:1], Scan: &PrefetchScan{}}))
			read := make(chan error, 1)
			go func() {
				_, err := readColumnRange(ctx, fs, ranges[1])
				read <- err
			}()
			require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 2 }, 5*time.Second, time.Millisecond,
				"the fills hold all the room")
			require.Equal(t, 2, heldRoom(fs.server))
			deadline, cancel := context.WithTimeout(ctx, 2*time.Second)
			defer cancel()
			require.NoError(t, fs.Delete(deadline, "object"))
			release()
			<-read
		})
	}
}
