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
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/defines"
	metric "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
	"github.com/matrixorigin/matrixone/pkg/util/toml"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

// probeStorage holds the first read from offset at until gate is closed, then answers it
// through answer, given the reader of object storage; it counts the other reads.
type probeStorage struct {
	ObjectStorage
	at     int64
	gate   chan struct{}
	answer func(io.ReadCloser) (io.ReadCloser, error)
	hit    atomic.Bool
	others atomic.Int64
}

func (s *probeStorage) Read(ctx context.Context, key string, from, to *int64) (io.ReadCloser, error) {
	if from != nil && *from == s.at && s.hit.CompareAndSwap(false, true) {
		select {
		case <-s.gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		r, err := s.ObjectStorage.Read(ctx, key, from, to)
		if err != nil {
			return nil, err
		}
		return s.answer(r)
	}
	s.others.Add(1)
	return s.ObjectStorage.Read(ctx, key, from, to)
}

type readCloser2 struct {
	io.Reader
	io.Closer
}

// failAfter returns a reader of n bytes of r that then fails with err.
func failAfter(n int64, err error) func(io.ReadCloser) (io.ReadCloser, error) {
	return func(r io.ReadCloser) (io.ReadCloser, error) {
		return readCloser2{Reader: io.MultiReader(io.LimitReader(r, n), failingReader{err}), Closer: r}, nil
	}
}

// shortBy returns a reader of r without its last n bytes.
func shortBy(r PrefetchRange, n int64) func(io.ReadCloser) (io.ReadCloser, error) {
	return func(rc io.ReadCloser) (io.ReadCloser, error) {
		return readCloser2{Reader: io.LimitReader(rc, r.Size-n), Closer: rc}, nil
	}
}

type failingReader struct{ err error }

func (r failingReader) Read([]byte) (int, error) { return 0, r.err }

// joinedColumnRead prefetches r of the object, held in s, and returns what a column read of
// r that joins the prefetch returns once s answers it.
func joinedColumnRead(t *testing.T, fileBacked bool, r PrefetchRange, answer func(io.ReadCloser) (io.ReadCloser, error)) ([]byte, []byte, error, *probeStorage) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
	<-fs.diskCache.load.done
	if fileBacked {
		fs.diskCache.memoryPrefetchMax = r.Size - 1
	}
	s := &probeStorage{ObjectStorage: storage, at: r.Offset, gate: make(chan struct{}), answer: answer}
	fs.server.raw = s
	release := sync.OnceFunc(func() { close(s.gate) })
	t.Cleanup(release)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}))
	require.Eventually(t, s.hit.Load, 5*time.Second, time.Millisecond)
	result := make(chan error, 1)
	var got []byte
	go func() {
		b, err := readColumnRange(ctx, fs, r)
		got = b
		result <- err
	}()
	diskPath := fs.diskCache.objectKey("object")
	require.Eventually(t, func() bool {
		if !fileBacked {
			return memoryWaiters(fs, diskPath) == 1
		}
		o, _ := fs.diskCache.fileDownload(diskPath, r.Offset, r.Offset+r.Size)
		if o == nil {
			return false
		}
		o.mu.Lock()
		defer o.mu.Unlock()
		return len(o.waiters) == 1
	}, 5*time.Second, time.Millisecond, "the read joins the prefetch")
	release()
	err := <-result
	waitPrefetcherIdle(t, fs)
	return got, data[r.Offset : r.Offset+r.Size], err, s
}

var joinedKinds = []struct {
	name       string
	fileBacked bool
}{{name: "memory"}, {name: "file", fileBacked: true}}

// BC2: a read whose joined prefetch object fails with an unexpected end of its source gets
// the error a read of its own reports.
func TestJoinedReadReportsUnexpectedEOFAsOwnRead(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	for _, k := range joinedKinds {
		t.Run(k.name, func(t *testing.T) {
			_, _, err, _ := joinedColumnRead(t, k.fileBacked, r, failAfter(100, io.ErrUnexpectedEOF))
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF), "got %v", err)
		})
	}
}

// BC1: a read whose joined prefetch object finds the object gone fails as its own read
// would, without reading object storage again.
func TestJoinedReadFailsOnMissingObject(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	for _, k := range joinedKinds {
		t.Run(k.name, func(t *testing.T) {
			_, _, err, s := joinedColumnRead(t, k.fileBacked, r, func(rc io.ReadCloser) (io.ReadCloser, error) {
				_ = rc.Close()
				return nil, moerr.NewFileNotFoundNoCtx("object")
			})
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "got %v", err)
			require.Zero(t, s.others.Load(), "the read does not read the object again")
		})
	}
}

// BC3: a read whose joined prefetch object's source ends short of its bytes reads them
// itself.
func TestJoinedReadOfShortSourceReadsItsBytes(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	for _, k := range joinedKinds {
		for _, short := range []int64{100, r.Size} {
			t.Run(k.name+"/"+map[bool]string{true: "empty", false: "short"}[short == r.Size], func(t *testing.T) {
				got, want, err, s := joinedColumnRead(t, k.fileBacked, r, shortBy(r, short))
				require.NoError(t, err)
				require.True(t, bytes.Equal(want, got))
				require.EqualValues(t, 1, s.others.Load(), "the read reads its bytes itself")
			})
		}
	}
}

// BC4: a read that started a whole-object download and cannot read the bytes the download
// wrote, from the download's file, reads them elsewhere.
func TestStreamedReadFallsBackOnUnreadableDownloadFile(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	d := fs.diskCache
	<-d.load.done
	storage := newBlackBoxStorage(fs.server.raw, func(_ string, offset int64) bool { return offset == 0 })
	fs.server.raw = storage
	release := sync.OnceFunc(func() { close(storage.gate) })
	t.Cleanup(release)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: 1000, Size: 4000}}}
	defer vec.Release()
	result := make(chan error, 1)
	go func() { result <- fs.Read(ctx, vec) }()
	diskPath := d.objectKey("object")
	var download *objectDownload
	require.Eventually(t, func() bool {
		if download = d.download(diskPath); download == nil {
			return false
		}
		download.mu.Lock()
		defer download.mu.Unlock()
		return len(download.waiters) == 1
	}, 5*time.Second, time.Millisecond, "the read waits for its download")
	download.mu.Lock()
	require.NoError(t, download.file.Close())
	download.mu.Unlock()
	release()
	require.NoError(t, <-result)
	require.True(t, bytes.Equal(data[1000:5000], vec.Entries[0].Data))
}

// stalledWholeDownload starts a whole-object download of the object whose source sends
// nothing until the returned function is called.
func stalledWholeDownload(t *testing.T, fs *S3FS) (*objectDownload, func()) {
	d := fs.diskCache
	<-d.load.done
	storage := newBlackBoxStorage(fs.server.raw, func(_ string, offset int64) bool { return offset == 0 })
	fs.server.raw = storage
	release := sync.OnceFunc(func() { close(storage.gate) })
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = fs.PrefetchFile(context.Background(), "object")
	}()
	t.Cleanup(func() {
		release()
		<-done
	})
	var download *objectDownload
	require.Eventually(t, func() bool {
		download = d.download(d.objectKey("object"))
		return download != nil
	}, 5*time.Second, time.Millisecond)
	return download, release
}

// BC12: a read of the disk cache whose context ends while it waits for a download returns
// its context's error.
func TestDownloadWaitEndsWithReadContext(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
	download, _ := stalledWholeDownload(t, fs)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Size: 4096}}}
	defer vec.Release()
	result := make(chan error, 1)
	go func() { result <- fs.diskCache.Read(ctx, vec) }()
	require.Eventually(t, func() bool {
		download.mu.Lock()
		defer download.mu.Unlock()
		return len(download.waiters) == 1
	}, 5*time.Second, time.Millisecond)
	cancel()
	require.ErrorIs(t, <-result, context.Canceled)
}

// BC19: a prefetch whose object is deleted while it downloads counts as ended, not failed.
func TestPrefetchOfDeletedObjectEnds(t *testing.T) {
	for _, whole := range []bool{false, true} {
		t.Run(map[bool]string{true: "whole", false: "ranges"}[whole], func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
			<-fs.diskCache.load.done
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
			fs.server.raw = storage
			t.Cleanup(func() { close(storage.gate) })
			failed := promtestutil.ToFloat64(metric.FSPrefetchFailedCounter)
			ended := promtestutil.ToFloat64(metric.FSPrefetchEndedCounter)
			req := ObjectPrefetch{FilePath: "object", Size: 300 << 10}
			if !whole {
				req.Ranges = []PrefetchRange{{Offset: 4096, Size: 4096}}
			}
			require.NoError(t, fs.PrefetchObject(context.Background(), req))
			key := fs.pathToKey("object")
			require.Eventually(t, func() bool {
				cur, _, _ := storage.counts(key)
				return cur == 1
			}, 5*time.Second, time.Millisecond)
			require.NoError(t, fs.Delete(context.Background(), "object"))
			waitPrefetcherIdle(t, fs)
			require.Equal(t, failed, promtestutil.ToFloat64(metric.FSPrefetchFailedCounter))
			require.Equal(t, ended+1, promtestutil.ToFloat64(metric.FSPrefetchEndedCounter))
		})
	}
}

// BC6: an entry of no bytes is an empty range for the disk cache too.
func TestDiskCacheReadOfEmptyEntryIsEmptyRange(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
	<-fs.diskCache.load.done
	vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: 10}}}
	defer vec.Release()
	err := fs.diskCache.Read(context.Background(), vec)
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrEmptyRange), "got %v", err)
}

// BC7: the index-file cache serves every entry it can, also those after one read to the
// end of the object.
func TestIndexObjectServesEntriesAfterReadToEnd(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	cacheIndexObject(t, fs, "object", data)
	vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
		Entries: []IOEntry{{Offset: 0, Size: -1}, {Offset: 100, Size: 10}}}
	defer vec.Release()
	require.NoError(t, fs.readIndexObject(context.Background(), vec))
	require.False(t, vec.Entries[0].done)
	require.True(t, vec.Entries[1].done)
	require.Equal(t, data[100:110], vec.Entries[1].Data)
}

// BC8: the bytes a read finds cached once a prefetch object it joined ended are read from
// cached ranges that hold them together, or from the cached whole object.
func TestCachedBytesAcrossRangesAndWholeObject(t *testing.T) {
	const part = 64 << 10
	ctx := context.Background()
	t.Run("ranges", func(t *testing.T) {
		fs, _, data := newPrefetchRangeFS(t, 0, 4*part)
		d := fs.diskCache
		<-d.load.done
		require.NoError(t, d.Update(ctx, &IOVector{FilePath: "object", Entries: []IOEntry{
			{Size: part, Data: data[:part]},
			{Offset: part, Size: part, Data: data[part : 2*part]},
		}}, false))
		dst := make([]byte, 32)
		var deadline time.Time
		require.True(t, d.readCachedBytes(ctx, "object", part-16, dst, &deadline))
		require.Equal(t, data[part-16:part+16], dst)
	})
	t.Run("whole", func(t *testing.T) {
		fs, _, data := newPrefetchRangeFS(t, 0, 4*part)
		d := fs.diskCache
		<-d.load.done
		require.NoError(t, d.setFileContent(ctx, "object", data, false))
		dst := make([]byte, 32)
		var deadline time.Time
		require.True(t, d.readCachedBytes(ctx, "object", part-16, dst, &deadline))
		require.Equal(t, data[part-16:part+16], dst)
		require.False(t, d.readCachedBytes(ctx, "object", int64(len(data))-16, dst, &deadline), "bytes past the object's end")
		require.True(t, d.cache.Contains(d.wholePath("object", int64(len(data)))), "a read past its end keeps the whole object")
	})
}

// BC17: a read whose whole-object fill was skipped, as for an object cached meanwhile,
// reads the disk cache again before object storage.
func TestSkippedWholeFillReadsCacheAgain(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 0, 64<<10)
	d := fs.diskCache
	<-d.load.done
	ctx := context.Background()
	require.NoError(t, d.setFileContent(ctx, "object", data, false))
	vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: 1000, Size: 100}}}
	defer vec.Release()
	require.NoError(t, fs.read(ctx, vec, false, false))
	require.Equal(t, data[1000:1100], vec.Entries[0].Data)
	require.Zero(t, storage.reads.Load(), "the cached object serves the read")
}

// BC16: an exact read joins a queued prefetch of ranges holding its bytes, as a column
// read does.
func TestExactReadJoinsQueuedRanges(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
	<-fs.diskCache.load.done
	busyFills(t, fs)
	storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
	fs.server.raw = storage
	t.Cleanup(func() { close(storage.gate) })
	r := PrefetchRange{Offset: 4096, Size: 4096}
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: 300 << 10, Ranges: []PrefetchRange{r}}))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead, Entries: []IOEntry{{Offset: r.Offset, Size: r.Size}}}
	require.True(t, fs.joinFills(ctx, vec, fs.sourcesFor(ctx, vec)))
}

// BC15: a read made under a download's context joins no prefetch object.
func TestReadOfDownloadJoinsNoPrefetchObject(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	for _, policy := range []Policy{ColumnPrefetch | SkipFullFilePreloads, SkipFullFilePreloads} {
		t.Run(map[bool]string{true: "column", false: "range"}[policy.Any(ColumnPrefetch)], func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
			<-fs.diskCache.load.done
			s := &probeStorage{ObjectStorage: storage, at: r.Offset, gate: make(chan struct{}),
				answer: func(rc io.ReadCloser) (io.ReadCloser, error) { return rc, nil }}
			fs.server.raw = s
			t.Cleanup(sync.OnceFunc(func() { close(s.gate) }))
			require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}))
			require.Eventually(t, s.hit.Load, 5*time.Second, time.Millisecond)
			ctx, cancel := context.WithTimeout(withDownloadStarted(context.Background(), func() {}), 5*time.Second)
			defer cancel()
			vec := &IOVector{FilePath: "object", Policy: policy | SkipMemoryCache, Entries: []IOEntry{{Offset: r.Offset, Size: r.Size}}}
			defer vec.Release()
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[r.Offset:r.Offset+r.Size], vec.Entries[0].Data)
			require.EqualValues(t, 1, s.others.Load(), "the read reads its bytes itself")
		})
	}
}

// BC13: a fill of an index object that skips every cache write fills neither the
// index-file cache nor the disk cache.
func TestIndexObjectFillSkippingCacheWrites(t *testing.T) {
	fs, _, _ := newIndexObjectFS(t, true)
	ctx := WithFileServicePolicy(context.Background(), IndexFileObject|SkipCacheWrites)
	require.NoError(t, fs.PrefetchFile(ctx, "object"))
	fs.diskCache.prefetchObjects.reads.Wait()
	_, cached := fs.indexObjectCached("object")
	require.False(t, cached)
	require.Zero(t, fs.diskCache.cache.Used())
	require.True(t, FullFilePreloadSkipped(ctx, fs, IndexFileObject|SkipCacheWrites, "object"))
}

// BC14: a read of an index object that skips disk-cache reads joins its download into the
// index-file cache.
func TestIndexObjectReadSkippingDiskReadsJoinsIndexDownload(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	d := fs.diskCache
	storage := newBlackBoxStorage(fs.server.raw, func(_ string, offset int64) bool { return offset == 0 })
	fs.server.raw = storage
	release := sync.OnceFunc(func() { close(storage.gate) })
	t.Cleanup(release)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	filled := make(chan error, 1)
	go func() { filled <- fs.PrefetchFile(WithFileServicePolicy(ctx, IndexFileObject), "object") }()
	var download *objectDownload
	require.Eventually(t, func() bool {
		download = d.download(d.objectKey("object"))
		return download != nil
	}, 5*time.Second, time.Millisecond)
	require.NotNil(t, d.wholeFill(d.objectKey("object"), true), "the download is of the index-file cache")
	vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipDiskCacheReads | SkipMemoryCache,
		Entries: []IOEntry{{Offset: 100, Size: 4096}}}
	defer vec.Release()
	result := make(chan error, 1)
	go func() { result <- fs.Read(ctx, vec) }()
	require.Eventually(t, func() bool {
		download.mu.Lock()
		defer download.mu.Unlock()
		return len(download.waiters) == 1
	}, 5*time.Second, time.Millisecond, "the read joins the download")
	release()
	require.NoError(t, <-result)
	require.NoError(t, <-filled)
	require.Equal(t, data[100:4196], vec.Entries[0].Data)
	_, _, total := storage.counts(fs.pathToKey("object"))
	require.Equal(t, 1, total, "the download alone reads the object")
}

// BC10: a read that caches nothing joins any prefetch object holding its bytes, also one
// that caches nothing.
func TestUncachingReadJoinsUncachingFlight(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
	<-fs.diskCache.load.done
	s := &probeStorage{ObjectStorage: storage, at: r.Offset, gate: make(chan struct{}),
		answer: func(rc io.ReadCloser) (io.ReadCloser, error) { return rc, nil }}
	fs.server.raw = s
	release := sync.OnceFunc(func() { close(s.gate) })
	t.Cleanup(release)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	first := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: []IOEntry{{Offset: r.Offset, Size: r.Size}}}
	defer first.Release()
	firstDone := make(chan error, 1)
	go func() { firstDone <- fs.Read(ctx, first) }()
	diskPath := fs.diskCache.objectKey("object")
	require.Eventually(t, s.hit.Load, 5*time.Second, time.Millisecond)
	var out bytes.Buffer
	second := &IOVector{FilePath: "object", Policy: ColumnPrefetch | SkipFullFilePreloads | SkipDiskCacheWrites | SkipMemoryCache,
		Entries: []IOEntry{{Offset: r.Offset, Size: r.Size, WriterForRead: &out}}}
	defer second.Release()
	secondDone := make(chan error, 1)
	go func() { secondDone <- fs.Read(ctx, second) }()
	require.Eventually(t, func() bool { return memoryWaiters(fs, diskPath) == 2 }, 5*time.Second, time.Millisecond,
		"the second read joins the first one's flight")
	release()
	require.NoError(t, <-firstDone)
	require.NoError(t, <-secondDone)
	require.Equal(t, data[r.Offset:r.Offset+r.Size], out.Bytes())
	require.Zero(t, s.others.Load())
}

// BC11: a read of the disk cache joins the first live FilePrefetchObject registered
// holding its bytes, never a stopped one.
func TestFileDownloadPicksLiveObject(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
	d := fs.diskCache
	<-d.load.done
	diskPath := d.objectKey("object")
	tempPath := t.TempDir() + "/download"
	require.NoError(t, os.WriteFile(tempPath, make([]byte, 64<<10), 0600))
	object := func(written int64, stopped bool) *filePrefetchObject {
		p := newFilePrefetchObject(context.Background(), 0, 64<<10)
		o := newObjectDownload(tempPath, nil)
		t.Cleanup(o.retire)
		o.begin(0)
		if written > 0 {
			o.advance(0, written)
		}
		p.download = o
		if stopped {
			p.cancelRead()
		}
		return p
	}
	slow, fast, stoppedFast := object(0, false), object(64<<10, false), object(64<<10, true)
	f := &d.prefetchObjects
	f.mu.Lock()
	f.registerLocked(diskPath, stoppedFast)
	f.registerLocked(diskPath, slow)
	f.registerLocked(diskPath, fast)
	f.mu.Unlock()
	got, _ := d.fileDownload(diskPath, 100, 200)
	require.Same(t, slow.download, got)
	f.mu.Lock()
	f.forgetLocked(diskPath, slow)
	f.mu.Unlock()
	got, _ = d.fileDownload(diskPath, 100, 200)
	require.Same(t, fast.download, got)
	f.mu.Lock()
	f.forgetLocked(diskPath, fast)
	f.mu.Unlock()
	got, _ = d.fileDownload(diskPath, 100, 200)
	require.Nil(t, got)
}

// tailHeldStorage returns the first 64 KiB of every read at once and holds the rest until
// gate is closed.
type tailHeldStorage struct {
	ObjectStorage
	gate  chan struct{}
	opens atomic.Int64
}

func (s *tailHeldStorage) Read(ctx context.Context, key string, from, to *int64) (io.ReadCloser, error) {
	s.opens.Add(1)
	r, err := s.ObjectStorage.Read(ctx, key, from, to)
	if err != nil {
		return nil, err
	}
	return readCloser2{Reader: io.MultiReader(io.LimitReader(r, 64<<10), &heldTail{ctx: ctx, gate: s.gate, r: r}), Closer: r}, nil
}

type heldTail struct {
	ctx  context.Context
	gate chan struct{}
	r    io.Reader
	open bool
}

func (h *heldTail) Read(b []byte) (int, error) {
	if !h.open {
		select {
		case <-h.gate:
		case <-h.ctx.Done():
			return 0, h.ctx.Err()
		}
		h.open = true
	}
	return h.r.Read(b)
}

// firstWriteSignal closes first on its first write. It has no ReadFrom, so a copy to it
// writes each chunk.
type firstWriteSignal struct {
	mu    sync.Mutex
	buf   []byte
	first chan struct{}
	once  sync.Once
}

func (w *firstWriteSignal) Write(b []byte) (int, error) {
	w.once.Do(func() { close(w.first) })
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, b...)
	return len(b), nil
}

func (w *firstWriteSignal) Bytes() []byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf
}

// A streamed entry larger than a prefetch object buffers opens its own reader: a reader
// gets its first bytes before a running range download that holds them sends its last, a
// writer gets its range once it is read; one of at most that many bytes joins the download.
func TestStreamedEntryPastPrefetchBufferReadsItself(t *testing.T) {
	const max = minDiskPrefetchMemoryMax
	for _, tc := range []struct {
		name   string
		size   int64
		reader bool
		policy Policy
	}{
		{name: "writer", size: max + 1<<20},
		{name: "writer_skip_disk_reads", size: max + 1<<20, policy: SkipDiskCacheReads},
		{name: "reader", size: max + 1<<20, reader: true},
		{name: "reader_skip_disk_reads", size: max + 1<<20, reader: true, policy: SkipDiskCacheReads},
		{name: "writer_joins_at_most_buffer", size: 128 << 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, max+2<<20)
			d := fs.diskCache
			d.memoryPrefetchMax = max
			d.objectSplit.parts = 1
			<-d.load.done
			storage := &tailHeldStorage{ObjectStorage: fs.server.raw, gate: make(chan struct{})}
			fs.server.raw = storage
			release := sync.OnceFunc(func() { close(storage.gate) })
			t.Cleanup(release)
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			r := PrefetchRange{Size: max + 1<<20}
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}))
			diskPath := d.objectKey("object")
			var download *objectDownload
			require.Eventually(t, func() bool {
				download, _ = d.fileDownload(diskPath, 0, r.Size)
				return download != nil
			}, 60*time.Second, time.Millisecond)

			entry := IOEntry{Size: tc.size}
			writer := &firstWriteSignal{first: make(chan struct{})}
			var stream io.ReadCloser
			if tc.reader {
				entry.ReadCloserForRead = &stream
			} else {
				entry.WriterForRead = writer
			}
			vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache | tc.policy, Entries: []IOEntry{entry}}
			defer vec.Release()
			result := make(chan error, 1)
			go func() { result <- fs.Read(ctx, vec) }()
			if tc.size <= max {
				require.Eventually(t, func() bool {
					download.mu.Lock()
					defer download.mu.Unlock()
					return len(download.waiters) == 1
				}, 60*time.Second, time.Millisecond, "the read joins the download")
				release()
				require.NoError(t, <-result)
				require.Equal(t, data[:tc.size], writer.Bytes())
				require.EqualValues(t, 1, storage.opens.Load())
				return
			}
			if tc.reader {
				require.NoError(t, <-result)
				var first [1]byte
				_, err := io.ReadFull(stream, first[:])
				require.NoError(t, err, "the reader returns bytes before the download's tail")
				require.Equal(t, data[0], first[0])
				release()
				rest, err := io.ReadAll(stream)
				require.NoError(t, err)
				require.Equal(t, data[1:tc.size], rest)
				require.NoError(t, stream.Close())
				return
			}
			require.Never(t, func() bool {
				select {
				case <-writer.first:
					return true
				default:
					return false
				}
			}, 100*time.Millisecond, time.Millisecond, "the writer runs once its range is read")
			release()
			require.NoError(t, <-result)
			require.Equal(t, data[:tc.size], writer.Bytes())
		})
	}
}

// A large exact read of bytes a stalled file prefetch holds joins it: it waits, opens no
// source of its own, and returns the bytes once the prefetch goes on.
func TestLargeReadJoinsStalledFilePrefetch(t *testing.T) {
	const size = 9 << 20
	for _, tc := range []struct {
		name   string
		policy Policy
	}{
		{name: "baseline_skip_cache_writes", policy: SkipDiskCacheWrites},
		{name: "cache_writes_enabled"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, 2*size)
			d := fs.diskCache
			d.memoryPrefetchMax = minDiskPrefetchMemoryMax
			d.objectSplit.parts = 1
			<-d.load.done

			var opens atomic.Int64
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool {
				return opens.Add(1) == 1
			})
			fs.server.raw = storage
			release := sync.OnceFunc(func() { close(storage.gate) })
			defer release()

			ctx := context.Background()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: size, Ranges: []PrefetchRange{{Size: size}},
			}))
			require.Eventually(t, func() bool { return opens.Load() == 1 }, 5*time.Second, time.Millisecond)
			download, _ := d.fileDownload(d.objectKey("object"), 0, size)
			require.NotNil(t, download)

			readCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()
			vec := &IOVector{
				FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache | tc.policy,
				Entries: []IOEntry{{Size: size}},
			}
			defer vec.Release()
			done := make(chan error, 1)
			go func() { done <- fs.Read(readCtx, vec) }()
			require.Never(t, func() bool { return len(done) > 0 }, 200*time.Millisecond, time.Millisecond,
				"the read waits for the prefetch it joined")
			require.EqualValues(t, 1, opens.Load(), "the read opens no source of its own")
			release()
			require.NoError(t, <-done)
			require.True(t, bytes.Equal(data[:size], vec.Entries[0].Data), "the exact range should be returned")
			require.EqualValues(t, 1, opens.Load())
		})
	}
}

func TestJoinedFileRangeWaitsForCaller(t *testing.T) {
	const (
		size = 192 << 10
		wait = 100 * time.Millisecond
	)
	for _, tc := range []struct {
		name    string
		ioBound time.Duration
	}{
		{name: "baseline_long_io_wait", ioBound: 5 * time.Second},
		{name: "short_io_wait", ioBound: wait},
	} {
		t.Run(tc.name, func(t *testing.T) {
			originalBound := maxIOWaitDuration
			maxIOWaitDuration = tc.ioBound
			defer func() { maxIOWaitDuration = originalBound }()
			fs, _, data := newPrefetchRangeFS(t, 0, 256<<10)
			d := fs.diskCache
			d.memoryPrefetchMax = 128 << 10
			d.objectSplit.parts = 1
			<-d.load.done
			storage := newBlackBoxStorage(fs.server.raw, func(_ string, start int64) bool { return start == 0 })
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.gate) })
			defer unblock()
			key := fs.pathToKey("object")
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: size, Ranges: []PrefetchRange{{Size: size}},
			}))
			require.Eventually(t, func() bool {
				cur, _, _ := storage.counts(key)
				return cur == 1
			}, time.Second, time.Millisecond)
			download, _ := d.fileDownload(d.objectKey("object"), 16, 80)
			require.NotNil(t, download)

			var dst bytes.Buffer
			vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache | SkipDiskCacheWrites,
				Entries: []IOEntry{{Offset: 16, Size: 64, WriterForRead: &dst}}}
			defer vec.Release()
			result := make(chan error, 1)
			readDone := make(chan struct{})
			go func() {
				defer close(readDone)
				result <- fs.Read(ctx, vec)
			}()
			defer func() {
				unblock()
				<-readDone
			}()
			require.Eventually(t, func() bool {
				download.mu.Lock()
				defer download.mu.Unlock()
				return len(download.waiters) == 1
			}, time.Second, time.Millisecond, "the read joins the file-backed range")
			select {
			case err := <-result:
				_, _, total := storage.counts(key)
				t.Fatalf("read returned before its joined range sent bytes: err=%v, source opens=%d", err, total)
			case <-time.After(3 * wait):
			}
			unblock()
			require.NoError(t, <-result)
			require.Equal(t, data[16:80], dst.Bytes())
			_, _, total := storage.counts(key)
			require.Equal(t, 1, total, "the joined range alone supplies the bytes")
		})
	}
}

func TestConcurrentSplitAdmissionAtLimitOne(t *testing.T) {
	const size = 300 << 10
	for _, tc := range []struct {
		name             string
		firstAlreadyOpen bool
	}{
		{name: "baseline_first_download_registered", firstAlreadyOpen: true},
		{name: "both_admitted_before_registration"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10, downloads: 1}, size)
			d := fs.diskCache
			<-d.load.done
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Policy: SkipAllCache,
				Entries: []IOEntry{{Size: size, Data: data}}}))
			d.prefetched.store(d.objectKey("other"), size)

			staging := make(chan chan struct{}, 2)
			cancelStaging := make(chan struct{})
			d.hooks.beforeCacheTempFileCreate = func() {
				gate := make(chan struct{})
				staging <- gate
				select {
				case <-gate:
				case <-cancelStaging:
				}
			}
			storage := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool {
				return key == fs.pathToKey("object")
			})
			fs.server.raw = storage
			allowSource := sync.OnceFunc(func() { close(storage.gate) })
			var readers sync.WaitGroup
			defer func() {
				close(cancelStaging)
				allowSource()
				readers.Wait()
			}()
			read := func(file string, result chan<- error) {
				readers.Go(func() { result <- fs.PrefetchFile(ctx, file) })
			}
			waitForStage := func() chan struct{} {
				t.Helper()
				select {
				case gate := <-staging:
					return gate
				case <-ctx.Done():
					t.Fatal("download did not reach staging", ctx.Err())
					return nil
				}
			}
			waitForFirstOpens := func() {
				t.Helper()
				require.Eventually(t, func() bool {
					cur, _, _ := storage.counts(fs.pathToKey("object"))
					return cur == 2
				}, time.Second, time.Millisecond, "the first fill's split parts should still be in flight")
			}

			first, second := make(chan error, 1), make(chan error, 1)
			read("object", first)
			firstStage := waitForStage()
			if tc.firstAlreadyOpen {
				close(firstStage)
				waitForFirstOpens()
			}
			read("other", second)
			secondStage := waitForStage()
			if !tc.firstAlreadyOpen {
				close(firstStage)
				waitForFirstOpens()
			}
			close(secondStage)
			require.NoError(t, <-second)
			allowSource()
			require.NoError(t, <-first)
			_, _, firstOpens := storage.counts(fs.pathToKey("object"))
			_, _, secondOpens := storage.counts(fs.pathToKey("other"))
			require.Equal(t, 3, firstOpens+secondOpens,
				"with a split-download limit of one, only one fill should split; source opens by fill: %d and %d", firstOpens, secondOpens)
		})
	}
}

func TestUnwritableIndexCacheKeepsTableCacheSeparate(t *testing.T) {
	const size = 9 << 20
	for _, tc := range []struct {
		name     string
		readOnly bool
	}{
		{name: "baseline_writable_index_cache"},
		{name: "unwritable_index_cache", readOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnly && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			config := indexCacheConfig(t, 64<<20, 64<<20)
			config.MemoryCapacity = ptrTo[toml.ByteSize](0)
			config.DiskPrefetchMemoryMax = ptrTo[toml.ByteSize](minDiskPrefetchMemoryMax)
			fs, err := NewS3FS(ctx, ObjectStorageArguments{
				Name: defines.SharedFileServiceName, Endpoint: "disk", Bucket: t.TempDir(),
			}, config, nil, false, false)
			require.NoError(t, err)
			defer fs.Close(ctx)
			fs.SetAsyncUpdate(false)
			<-fs.diskCache.load.done
			<-fs.indexFileCache.disk.load.done

			data := bytes.Repeat([]byte{0x59}, size)
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Policy: SkipAllCache,
				Entries: []IOEntry{{Size: size, Data: data}}}))
			if tc.readOnly {
				path := fs.indexFileCache.disk.path
				require.NoError(t, os.Chmod(path, 0o500))
				defer func() { require.NoError(t, os.Chmod(path, 0o755)) }()
			}
			vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
				Entries: []IOEntry{{Size: size}}}
			defer vec.Release()
			require.NoError(t, fs.Read(ctx, vec))
			require.True(t, bytes.Equal(data, vec.Entries[0].Data), "the index read should return the whole object")
			fs.diskCache.prefetchObjects.reads.Wait()
			require.Zero(t, fs.diskCache.cache.Used(), "an index-object read should take no space in the table-data cache")
		})
	}
}

type reviewFailOnceWriter struct {
	err   error
	calls int
}

func (w *reviewFailOnceWriter) Write(data []byte) (int, error) {
	w.calls++
	if w.calls == 1 {
		return 0, w.err
	}
	return len(data), nil
}

func TestSplitCacheReadReturnsWriterError(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
	}{
		{name: "baseline_source_read", policy: SkipDiskCache},
		{name: "read_across_two_cached_ranges"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const part = 64 << 10
			fs, storage, data := newPrefetchRangeFS(t, 0, 4*part)
			d := fs.diskCache
			<-d.load.done
			ctx := context.Background()
			require.NoError(t, d.Update(ctx, &IOVector{FilePath: "object", Entries: []IOEntry{
				{Size: part, Data: data[:part]},
				{Offset: part, Size: part, Data: data[part : 2*part]},
			}}, false))
			const offset, size = part - 16, 32
			rs, ok := d.ranges.covering(d.objectKey("object"), offset, size)
			require.True(t, ok)
			require.Len(t, rs, 2)
			wantErr := errors.New("review: caller's writer rejected output")
			writer := &reviewFailOnceWriter{err: wantErr}
			vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache | tc.policy,
				Entries: []IOEntry{{Offset: offset, Size: size, WriterForRead: writer}}}
			defer vec.Release()
			err := fs.Read(ctx, vec)
			require.ErrorIs(t, err, wantErr, "a caller writer failure should be returned; writer calls=%d, source opens=%d", writer.calls, storage.reads.Load())
			require.Equal(t, 1, writer.calls)
		})
	}
}

// wantSources is the table of the shared sources a read may take its bytes from, by its
// policy, whether it is a read of a download and whether it streams an entry larger than
// a prefetch object buffers.
func wantSources(s *S3FS, p Policy, download, streamsLarge bool) readSources {
	// BC15: the reads of a download join no prefetch object. A large streamed entry opens
	// its own reader.
	disk := s.diskCache != nil && !p.Any(SkipDiskCacheReads) && !download && !streamsLarge
	// BC13: index-file cache reads are gated by SkipCacheReads, its fills by
	// SkipCacheWrites. The bytes of an index object are cached in the index-file cache
	// only, never in the disk cache.
	index := p.Any(IndexFileObject) && s.cachesIndexObjects()
	tier := diskTier
	switch {
	case index && p&SkipCacheWrites != SkipCacheWrites:
		tier = indexTier
	case index, p.Any(SkipDiskCacheWrites):
		tier = noTier
	}
	return readSources{
		indexRead: index && p&SkipCacheReads != SkipCacheReads,
		tier:      tier,
		flights:   disk && p.Any(ColumnPrefetch, ExactRead),
		window:    disk && tier == diskTier,
		// BC10: a read that caches its bytes joins only objects that cache theirs.
		caches: p.CacheIOEntry() && tier == diskTier,
		fetch:  disk,
	}
}

func TestSourcesForTable(t *testing.T) {
	fs, _, _ := newIndexObjectFS(t, true)
	for p := Policy(0); p < IndexFileObject<<1; p++ {
		for _, download := range []bool{false, true} {
			for _, streams := range []int64{0, 1, fs.diskCache.memoryPrefetchMax + 1} {
				ctx := context.Background()
				if download {
					ctx = withDownloadStarted(ctx, func() {})
				}
				vec := &IOVector{FilePath: "object", Policy: p, Entries: []IOEntry{{Size: 1}}}
				if streams > 0 {
					vec.Entries[0].Size, vec.Entries[0].WriterForRead = streams, new(bytes.Buffer)
				}
				require.Equal(t, wantSources(fs, p, download, streams > fs.diskCache.memoryPrefetchMax), fs.sourcesFor(ctx, vec),
					"policy %b, download %v, streams %v", p, download, streams)
			}
		}
	}
}

func TestCloseCancelsUnclaimedPrefetchPublication(t *testing.T) {
	for _, tc := range []struct {
		name string
		size int64
	}{
		{name: "ordinary_async_entry", size: 64 << 10},
		{name: "file_backed_range", size: 144 << 10},
		{name: "memory_backed_range", size: 64 << 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newPrefetchRangeFS(t, 0, 256<<10)
			d := fs.diskCache
			d.memoryPrefetchMax = 128 << 10
			<-d.load.done

			atPublication := make(chan struct{})
			resume := make(chan struct{})
			release := sync.OnceFunc(func() { close(resume) })
			defer release()
			d.hooks.beforeFilePublication = func() {
				close(atPublication)
				<-resume
			}
			if tc.name == "ordinary_async_entry" {
				vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Size: tc.size, Data: data[:tc.size]}}}
				require.NoError(t, d.Update(ctx, vec, true))
			} else {
				require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: tc.size,
					Ranges: []PrefetchRange{{Size: tc.size}}}))
			}
			select {
			case <-atPublication:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			canceled := d.async.ctx.Done()
			if tc.name != "ordinary_async_entry" {
				objects := &d.prefetchObjects
				objects.mu.Lock()
				active := slices.Clone(objects.m[d.objectKey("object")])
				objects.mu.Unlock()
				require.Len(t, active, 1)
				switch flight := active[0].(type) {
				case *memoryPrefetchObject:
					canceled = flight.ctx.Done()
				case *filePrefetchObject:
					canceled = flight.ctx.Done()
				default:
					t.Fatalf("unexpected range owner %T", flight)
				}
			}

			closed := make(chan struct{})
			go func() {
				d.Close(ctx)
				close(closed)
			}()
			select {
			case <-canceled:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			release()
			select {
			case <-closed:
				require.NoError(t, ctx.Err())
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}

			path := d.pathForIOEntry("object", IOEntry{Size: tc.size})
			require.NoFileExists(t, path, "close canceled the flight before it could claim publication")
		})
	}
}

func TestSplitReadReservesFirstConnectionForNeededPart(t *testing.T) {
	for _, offset := range []int64{0, 128 << 10} {
		name := "needed_prefix"
		if offset > 0 {
			name = "needed_suffix"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
			d := fs.diskCache
			<-d.load.done
			limitRoom(t, fs.server, 1, 1)

			opened := make(chan int64, 2)
			storage := newBlackBoxStorage(fs.server.raw, func(_ string, start int64) bool {
				opened <- start
				return true
			})
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.gate) })
			defer unblock()

			result := make(chan error, 1)
			go func() {
				vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: offset, Size: 16}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				if err == nil && !bytes.Equal(data[offset:offset+16], vec.Entries[0].Data) {
					err = fmt.Errorf("incorrect bytes at offset %d", offset)
				}
				result <- err
			}()

			var first int64
			select {
			case first = <-opened:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			require.Equal(t, 1, heldRoom(fs.server))
			unblock()
			require.NoError(t, <-result)
			require.Zero(t, heldRoom(fs.server))
			require.Equal(t, offset, first, "with room for one the read's part must start first")
		})
	}
}

// An exact read of bytes a stalled file-backed fill holds joins it and waits for its
// bytes; one that skips disk-cache reads cannot read what the fill caches and reads its
// bytes itself.
func TestFileRangeReadJoinsStalledFileFill(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
		joins  bool
	}{
		{name: "bypass_disk_reads", policy: SkipDiskCacheReads},
		{name: "can_read_disk_cache", joins: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newPrefetchRangeFS(t, 0, 256<<10)
			d := fs.diskCache
			d.memoryPrefetchMax = 128 << 10
			<-d.load.done

			open := make(chan struct{})
			opened := sync.OnceFunc(func() { close(open) })
			storage := newBlackBoxStorage(fs.server.raw, func(_ string, offset int64) bool {
				if offset != 0 {
					return false
				}
				opened()
				return true
			})
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.gate) })
			defer unblock()
			const size = 192 << 10
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: size,
				Ranges: []PrefetchRange{{Size: size}}}))
			select {
			case <-open:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			download, _ := d.fileDownload(d.objectKey("object"), 16, 80)
			require.NotNil(t, download)

			vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipDiskCacheWrites | tc.policy,
				Entries: []IOEntry{{Offset: 16, Size: 64}}}
			defer vec.Release()
			readDone := make(chan error, 1)
			readFinished := make(chan struct{})
			go func() {
				defer close(readFinished)
				readDone <- fs.Read(ctx, vec)
			}()
			defer func() {
				unblock()
				<-readFinished
			}()
			var queuedWaiters int
			require.Eventually(t, func() bool {
				select {
				case <-readFinished:
					return true
				default:
				}
				download.mu.Lock()
				queuedWaiters = len(download.waiters)
				download.mu.Unlock()
				return queuedWaiters > 0
			}, 4*time.Second, time.Millisecond)

			unblock()
			require.NoError(t, <-readDone)
			require.Equal(t, data[16:80], vec.Entries[0].Data)
			require.Equal(t, tc.joins, queuedWaiters > 0, "the read waits on the file fill it joined")
		})
	}
}

func TestFlightBudgetDoesNotOverflow(t *testing.T) {
	for _, tc := range []struct {
		name  string
		limit toml.ByteSize
	}{
		{name: "default_sized_limit", limit: DefaultDiskPrefetchMemoryMax},
		{name: "large_limit", limit: 1 << 57},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prior := memoryAvailable
			memoryAvailable = func() (uint64, bool) { return 1 << 30, true }
			t.Cleanup(func() { memoryAvailable = prior })
			cfg := CacheConfig{DiskPrefetchMemoryMax: &tc.limit}
			fs, _, _ := newPrefetchRangeFS(t, 0, 1)
			fs.diskCache.memoryPrefetchMax = cfg.diskPrefetchMemoryMax()
			require.EqualValues(t, (uint64(1)<<30)/100*5, fs.server.memoryBudget(), "the measured ceiling should constrain an oversized configured ceiling")
		})
	}
}

// Overlapping exact reads larger than disk-prefetch-memory-max share one file-backed
// prefetch object (decision 2026-11-23).
func TestExactReadsShareLargeRange(t *testing.T) {
	for _, tc := range []struct {
		name string
		size int64
	}{
		{name: "memory_sized_range", size: 64 << 10},
		{name: "file_sized_range", size: 128<<10 + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newPrefetchRangeFS(t, 0, 256<<10)
			fs.diskCache.memoryPrefetchMax = 128 << 10
			<-fs.diskCache.load.done

			// Exact column reads skip the grid. Use distinct start offsets so the
			// identical-request I/O merger leaves deduplication to the range owner.
			started := make(chan struct{}, 2)
			gate := newBlackBoxStorage(fs.server.raw, func(string, int64) bool {
				select {
				case started <- struct{}{}:
				default:
				}
				return true
			})
			fs.server.raw = gate
			unblock := sync.OnceFunc(func() { close(gate.gate) })
			defer unblock()
			read := func(offset, size int64, result chan<- error) {
				vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead,
					Entries: []IOEntry{{Offset: offset, Size: size}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				if err == nil && !bytes.Equal(data[offset:offset+size], vec.Entries[0].Data) {
					err = fmt.Errorf("incorrect bytes at offset %d", offset)
				}
				result <- err
			}
			results := make(chan error, 2)
			go read(0, tc.size, results)
			select {
			case <-started:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			go read(1, tc.size-1, results)
			// If the second reader opens its own range we have the counterexample.
			// Otherwise, observe it waiting for bytes on the registered range.
			require.Eventually(t, func() bool {
				if _, _, reads := gate.counts("object"); reads == 2 {
					return true
				}
				objects := &fs.diskCache.prefetchObjects
				objects.mu.Lock()
				defer objects.mu.Unlock()
				for _, object := range objects.m[fs.diskCache.objectKey("object")] {
					var waiters int
					switch flight := object.(type) {
					case *memoryPrefetchObject:
						flight.mu.Lock()
						waiters = len(flight.waiters)
						flight.mu.Unlock()
					case *filePrefetchObject:
						if download := flight.joinableDownload(); download != nil {
							download.mu.Lock()
							waiters = len(download.waiters)
							download.mu.Unlock()
						}
					}
					if waiters == 2 {
						return true
					}
				}
				return false
			}, 4*time.Second, time.Millisecond)
			unblock()
			for range 2 {
				require.NoError(t, <-results)
			}
			_, _, reads := gate.counts("object")
			require.Equal(t, 1, reads, "overlapping exact reads should share a registered download")
		})
	}
}
