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
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// readUnit reads size bytes at offset of the object with prefetch ranges.
func readUnit(ctx context.Context, fs *S3FS, offset, size int64) ([]byte, error) {
	vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
		Entries: []IOEntry{{Offset: offset, Size: size}}}
	err := fs.Read(ctx, vec)
	data := slices.Clone(vec.Entries[0].Data)
	vec.Release()
	return data, err
}

func fileObjects(d *DiskCache) int {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, objects := range f.m {
		for _, o := range objects {
			if d := o.joinableDownload(); d == nil || !d.memory {
				n++
			}
		}
	}
	return n
}

// A FilePrefetchObject a read starts is stopped when the read ends before its download
// starts, and reads nothing.
func TestFilePrefetchObjectOfEndedReadStops(t *testing.T) {
	const unit = 64 << 10
	fs, storage, _ := newPrefetchRangeFS(t, unit, 4*unit)
	d := fs.diskCache
	d.memoryPrefetchMax = 16 << 10
	limitRoom(t, d.server, 1, 1)
	release := occupy(t, d.server, foregroundRequest, 1)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := readUnit(ctx, fs, 100, 10)
		done <- err
	}()
	require.Eventually(t, func() bool { _, waiting := d.server.counts(); return waiting[foregroundRequest] > 0 },
		10*time.Second, time.Millisecond, "the download waits for room")
	cancel()
	require.ErrorIs(t, <-done, context.Canceled)
	require.Eventually(t, func() bool { return fileObjects(d) == 0 }, 10*time.Second, time.Millisecond)
	d.prefetchObjects.reads.Wait()
	release()
	require.Zero(t, storage.reads.Load())
	require.Zero(t, heldRoom(d.server))
}

// A delete of an object stops the download a whole-object read started; the read then
// reads its range and returns its bytes.
func TestDeleteDuringWholeObjectReadReadsRange(t *testing.T) {
	fs, storage, data := newDownloadContractFS(t)
	done := storage.readAsync(fs, 6<<20, 17)
	require.Eventually(t, func() bool { return fs.diskCache.download(fs.diskCache.objectKey("object")) != nil },
		10*time.Second, time.Millisecond)
	require.NoError(t, fs.diskCache.DeletePaths(context.Background(), []string{"object"}))
	require.Equal(t, data[6<<20:6<<20+17], requireReturns(t, done))
	require.Positive(t, storage.ranges.Load(), "the read fetched its range")
}

// blackBoxStorage holds each range read until the channel gated returns for it is closed,
// none for nil, and counts the reads of each object under way and in all.
type blackBoxStorage struct {
	ObjectStorage
	gated func(key string, offset int64) chan struct{}
	gate  chan struct{}

	mu    sync.Mutex
	cur   map[string]int
	most  map[string]int
	total map[string]int
	// marked counts the reads under a context of withReadMark.
	marked map[string]int
	// ranges are the ranges read, the end -1 for none.
	ranges [][2]int64
}

type readMarkKey struct{}

// withReadMark returns ctx, whose reads of object storage blackBoxStorage counts as marked.
func withReadMark(ctx context.Context) context.Context {
	return context.WithValue(ctx, readMarkKey{}, true)
}

// newBlackBoxStorage returns a storage holding the reads gated accepts until its gate is
// closed.
func newBlackBoxStorage(upstream ObjectStorage, gated func(key string, offset int64) bool) *blackBoxStorage {
	g := &blackBoxStorage{ObjectStorage: upstream, gate: make(chan struct{}),
		cur: map[string]int{}, most: map[string]int{}, total: map[string]int{}, marked: map[string]int{}}
	g.gated = func(key string, offset int64) chan struct{} {
		if gated(key, offset) {
			return g.gate
		}
		return nil
	}
	return g
}

func (g *blackBoxStorage) Read(ctx context.Context, key string, from, to *int64) (io.ReadCloser, error) {
	var offset int64
	if from != nil {
		offset = *from
	}
	g.mu.Lock()
	g.cur[key]++
	g.total[key]++
	end := int64(-1)
	if to != nil {
		end = *to
	}
	g.ranges = append(g.ranges, [2]int64{offset, end})
	if ctx.Value(readMarkKey{}) != nil {
		g.marked[key]++
	}
	g.most[key] = max(g.most[key], g.cur[key])
	g.mu.Unlock()
	defer func() {
		g.mu.Lock()
		g.cur[key]--
		g.mu.Unlock()
	}()
	if gate := g.gated(key, offset); gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return g.ObjectStorage.Read(ctx, key, from, to)
}

func (g *blackBoxStorage) counts(key string) (cur, most, total int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.cur[key], g.most[key], g.total[key]
}

// unmarked returns the reads of key not under a context of withReadMark.
func (g *blackBoxStorage) unmarked(key string) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.total[key] - g.marked[key]
}

func columnRanges(n int, stride, size int64) []PrefetchRange {
	ranges := make([]PrefetchRange, n)
	for i := range ranges {
		ranges[i] = PrefetchRange{Offset: int64(i) * stride, Size: size}
	}
	return ranges
}

func readColumnRange(ctx context.Context, fs *S3FS, r PrefetchRange) ([]byte, error) {
	vec := &IOVector{FilePath: "object", Policy: ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache,
		Entries: []IOEntry{{Offset: r.Offset, Size: r.Size}}}
	if err := fs.Read(ctx, vec); err != nil {
		return nil, err
	}
	defer vec.Release()
	return append([]byte(nil), vec.Entries[0].Data...), nil
}

// A column read that joined a prefetch of the object returns once its range is read, while
// the prefetch's later ranges still wait.
func TestBlackBoxJoinedColumnReadReturnsWhileRangesWait(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	ranges := columnRanges(3, 40<<10, 1<<10)
	limitRoom(t, fs.server, 2, 1)
	first, rest := make(chan struct{}), make(chan struct{})
	g := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return false })
	g.gated = func(_ string, offset int64) chan struct{} {
		if offset == ranges[0].Offset {
			return first
		}
		return rest
	}
	fs.server.raw = g
	releaseFirst, releaseRest := sync.OnceFunc(func() { close(first) }), sync.OnceFunc(func() { close(rest) })
	defer releaseFirst()
	defer releaseRest()
	ctx := context.Background()
	key := fs.pathToKey("object")
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 3 << 10, Ranges: ranges, Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 1 }, 5*time.Second, time.Millisecond)
	readCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	type result struct {
		data []byte
		err  error
	}
	done := make(chan result, 1)
	go func() {
		got, err := readColumnRange(readCtx, fs, ranges[0])
		done <- result{got, err}
	}()
	require.Never(t, func() bool { return len(done) > 0 }, 100*time.Millisecond, time.Millisecond, "the read waits for its range")
	releaseFirst()
	r := <-done
	require.NoError(t, r.err, "the read returns while the later ranges wait")
	require.Equal(t, data[ranges[0].Offset:ranges[0].Offset+ranges[0].Size], r.data)
}

// A request for an object downloading does not take over what an ended scan deferred behind
// it, so another object's live request is still downloaded.
func TestBlackBoxEndedDeferredPrefetchNotInherited(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	ctx := context.Background()
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Entries: []IOEntry{{Size: int64(len(data)), Data: data}},
		Policy: SkipDiskCache | SkipMemoryCache}))
	objectKey, otherKey := fs.pathToKey("object"), fs.pathToKey("other")
	g := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return key == objectKey })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	defer release()
	limit := fs.diskCache.fillQueueLimit()
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 1 << 10,
		Ranges: []PrefetchRange{{Offset: 0, Size: 1 << 10}}, Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { cur, _, _ := g.counts(objectKey); return cur == 1 }, 5*time.Second, time.Millisecond)
	ended := &PrefetchScan{}
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: limit - (1 << 10),
		Ranges: []PrefetchRange{{Offset: 100 << 10, Size: limit - (1 << 10)}}, Scan: ended}))
	ended.End()
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 1 << 10,
		Ranges: []PrefetchRange{{Offset: 200 << 10, Size: 1 << 10}}, Scan: &PrefetchScan{}}))
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "other", Size: 1 << 10,
		Ranges: []PrefetchRange{{Offset: 0, Size: 1 << 10}}, Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { _, _, total := g.counts(otherKey); return total == 1 }, 5*time.Second, time.Millisecond,
		"the live request of another object is downloaded")
}

// A column prefetch whose range cannot get a download connection holds no buffer, so a
// column read of another range of its object, waiting for buffers, gets one and returns.
func TestBlackBoxForegroundReadNotHeldByBufferOfWaitingRange(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	ctx := context.Background()
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Entries: []IOEntry{{Size: int64(len(data)), Data: data}},
		Policy: SkipDiskCache | SkipMemoryCache}))
	ranges := columnRanges(2, 40<<10, 1<<10)
	limitRoom(t, fs.server, 2, 1)
	fs.server.memoryLimit = flightBufSize(ranges[0].Size)
	objectKey, otherKey := fs.pathToKey("object"), fs.pathToKey("other")
	g := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return key == otherKey })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	defer release()
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "other", Size: int64(len(data)), Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { cur, _, _ := g.counts(otherKey); return cur == 1 }, 5*time.Second, time.Millisecond,
		"the whole download of other holds the connection")
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: ranges[0].Size,
		Ranges: ranges[:1], Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { return fs.diskCache.fillsUnderWay("object") }, 5*time.Second, time.Millisecond)
	readCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	got, err := readColumnRange(readCtx, fs, ranges[1])
	require.NoError(t, err)
	require.Equal(t, data[ranges[1].Offset:ranges[1].Offset+ranges[1].Size], got)
	_, _, total := g.counts(objectKey)
	require.Equal(t, 1, total, "only the read reads the object")
}

// A column read of a range of a running prefetch whose range waits for a connection, held
// by another object's download or by a sibling range of the prefetch, reads its own bytes;
// once the connection is free, the prefetch caches its ranges. A column read of an object
// whose whole-object prefetch waits joins and promotes it, and the object is read once.
func TestBlackBoxColumnReadNotHeldByPrefetchWaitingForConnection(t *testing.T) {
	for _, tc := range []struct {
		name              string
		ranges            []PrefetchRange
		memoryPrefetchMax int64
		// sibling holds the connection with the first range the prefetch reads, else a
		// whole download of another object holds it.
		sibling bool
	}{
		{name: "memory", ranges: columnRanges(2, 40<<10, 1<<10)},
		{name: "file", ranges: columnRanges(2, 40<<10, 16<<10), memoryPrefetchMax: 16 << 10},
		{name: "sibling", ranges: columnRanges(2, 40<<10, 1<<10), sibling: true},
		{name: "whole object"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
			ctx := context.Background()
			limitRoom(t, fs.server, 2, 1)
			if tc.memoryPrefetchMax > 0 {
				fs.diskCache.memoryPrefetchMax = tc.memoryPrefetchMax
			}
			objectKey, otherKey := fs.pathToKey("object"), fs.pathToKey("other")
			g := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return key == otherKey })
			fs.server.raw = g
			release := sync.OnceFunc(func() { close(g.gate) })
			defer release()
			var heldOffset atomic.Int64
			heldOffset.Store(-1)
			if tc.sibling {
				fs.server.memoryLimit = 4 * flightBufSize(tc.ranges[0].Size)
				g.gated = func(key string, offset int64) chan struct{} {
					if key == objectKey && heldOffset.CompareAndSwap(-1, offset) {
						return g.gate
					}
					return nil
				}
			} else {
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Entries: []IOEntry{{Size: int64(len(data)), Data: data}},
					Policy: SkipDiskCache | SkipMemoryCache}))
				require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "other", Size: int64(len(data)), Scan: &PrefetchScan{}}))
				require.Eventually(t, func() bool { cur, _, _ := g.counts(otherKey); return cur == 1 }, 5*time.Second, time.Millisecond,
					"the whole download of other holds the connection")
			}
			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data)), Ranges: tc.ranges, Scan: &PrefetchScan{}}
			if tc.ranges != nil {
				req.Size = 0
				for _, r := range tc.ranges {
					req.Size += r.Size
				}
			}
			require.NoError(t, fs.PrefetchObject(ctx, req))
			r := PrefetchRange{Offset: 40 << 10, Size: 1 << 10}
			if tc.sibling {
				require.Eventually(t, func() bool { cur, _, _ := g.counts(objectKey); return cur == 1 }, 5*time.Second, time.Millisecond)
				r = tc.ranges[0]
				if heldOffset.Load() == r.Offset {
					r = tc.ranges[1]
				}
			} else {
				require.Eventually(t, func() bool { return fs.diskCache.fillsUnderWay("object") }, 5*time.Second, time.Millisecond)
				if tc.ranges != nil {
					r = tc.ranges[0]
				}
			}
			readCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
			defer cancel()
			got, err := readColumnRange(readCtx, fs, r)
			require.NoError(t, err)
			require.Equal(t, data[r.Offset:r.Offset+r.Size], got)
			if tc.ranges == nil {
				require.Eventually(t, func() bool { return fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))) },
					5*time.Second, time.Millisecond)
				_, _, total := g.counts(objectKey)
				require.Equal(t, 1, total, "the read is served by the whole-object download it joined")
				return
			}
			require.True(t, fs.diskCache.fillsUnderWay("object"), "the prefetch still waits")
			release()
			require.Eventually(t, func() bool { return !fs.diskCache.fillsUnderWay("object") }, 5*time.Second, time.Millisecond)
			key := fs.diskCache.objectKey("object")
			for _, r := range tc.ranges {
				_, cached := fs.diskCache.ranges.covering(key, r.Offset, r.Size)
				require.True(t, cached)
			}
			// The prefetch's range the read read was registered before the read cached it.
			_, _, total := g.counts(objectKey)
			require.LessOrEqual(t, total, 3)
		})
	}
}

// Reads at offsets of the same run, past the bytes of a MemoryPrefetchObject or within
// them, that wait for a connection, held by another object's download, download the run
// once and cache it.
func TestBlackBoxConcurrentReadsOfFileRunDownloadOnce(t *testing.T) {
	const unit = 64 << 10
	for _, size := range []int64{4 * unit, unit} {
		fs, _, data := newPrefetchRangeFS(t, unit, 1<<20)
		ctx := context.Background()
		require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Entries: []IOEntry{{Size: int64(len(data)), Data: data}},
			Policy: SkipDiskCache | SkipMemoryCache}))
		limitRoom(t, fs.server, 2, 1)
		fs.diskCache.memoryPrefetchMax = 2 * unit
		objectKey, otherKey := fs.pathToKey("object"), fs.pathToKey("other")
		g := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return key == otherKey })
		fs.server.raw = g
		release := sync.OnceFunc(func() { close(g.gate) })
		defer release()
		require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "other", Size: int64(len(data)), Scan: &PrefetchScan{}}))
		require.Eventually(t, func() bool { cur, _, _ := g.counts(otherKey); return cur == 1 }, 5*time.Second, time.Millisecond,
			"the whole download of other holds the connection")
		freeForeground := occupy(t, fs.server, foregroundRequest, 1)
		const readers = 3
		errs := make(chan error, readers)
		for i := range int64(readers) {
			go func() {
				off := i / 2 * 100
				vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
					Entries: []IOEntry{{Offset: off, Size: size - off}}}
				err := fs.Read(ctx, vec)
				if err == nil && !bytes.Equal(data[off:size], vec.Entries[0].Data) {
					err = errors.New("wrong bytes")
				}
				vec.Release()
				errs <- err
			}()
		}
		require.Never(t, func() bool { return len(errs) > 0 }, 100*time.Millisecond, time.Millisecond, "the reads wait for room")
		_, _, total := g.counts(objectKey)
		require.Zero(t, total)
		release()
		freeForeground()
		for range readers {
			require.NoError(t, <-errs)
		}
		fs.diskCache.prefetchObjects.reads.Wait()
		_, _, total = g.counts(objectKey)
		require.Equal(t, 1, total, "the run downloads once")
		_, cached := fs.diskCache.ranges.covering(fs.diskCache.objectKey("object"), 0, size)
		require.True(t, cached, "size %d", size)
		f := &fs.diskCache.prefetchObjects
		f.mu.Lock()
		objects := len(f.m[fs.diskCache.objectKey("object")])
		f.mu.Unlock()
		require.Zero(t, objects, "no prefetch object stays")
	}
}

// A range of a prefetch that ended frees its buffer while an earlier range of the prefetch
// still reads, so a read of another range takes the buffer and returns.
func TestBlackBoxEndedPrefetchRangeFreesItsBuffer(t *testing.T) {
	const unit = 64 << 10
	fs, _, data := newPrefetchRangeFS(t, unit, 4*unit)
	ctx := context.Background()
	d := fs.diskCache
	d.server.memoryLimit = 2 * flightBufSize(unit)
	key := fs.pathToKey("object")
	g := newBlackBoxStorage(fs.server.raw, func(k string, offset int64) bool { return k == key && offset == 0 })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	t.Cleanup(release)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 2 * unit,
		Ranges: []PrefetchRange{{Offset: 0, Size: unit}, {Offset: 2 * unit, Size: unit}}, Scan: &PrefetchScan{}}))
	f := &d.prefetchObjects
	diskPath := d.objectKey("object")
	require.Eventually(t, func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return len(f.m[diskPath]) == 1 && heldMemory(d) <= flightBufSize(unit)
	}, 5*time.Second, time.Millisecond, "the second range ended and freed its buffer while the first reads")

	read := make(chan []byte, 1)
	go func() {
		vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
			Entries: []IOEntry{{Offset: 3*unit + 100, Size: 10}}}
		if err := fs.Read(ctx, vec); err != nil {
			read <- nil
			return
		}
		read <- bytes.Clone(vec.Entries[0].Data)
		vec.Release()
	}()
	var got []byte
	require.Eventually(t, func() bool {
		select {
		case got = <-read:
			return true
		default:
			return false
		}
	}, 5*time.Second, time.Millisecond, "a read of another range takes the freed buffer")
	require.Equal(t, data[3*unit+100:3*unit+110], got)
	cur, _, _ := g.counts(key)
	require.Equal(t, 1, cur, "the first range still reads")
	release()
	f.reads.Wait()
}

// The ranges of a prefetch waiting for the only connection take no goroutine each, and a
// delete of the object ends them all.
func TestBlackBoxWideRangePrefetchWaitingTakesNoGoroutinePerRange(t *testing.T) {
	const n = 2048
	fs, _, _ := newPrefetchRangeFS(t, 0, n*64)
	ctx := context.Background()
	d := fs.diskCache
	limitRoom(t, d.server, 1, 0)
	d.setFillsMax(8)
	f := &d.prefetchObjects
	waiting := func() int {
		_, waiting := d.server.counts()
		return waiting[backgroundRequest]
	}
	before := runtime.NumGoroutine()
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: n * 16,
		Ranges: columnRanges(n, 64, 16), Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { return waiting() == 8 }, 5*time.Second, time.Millisecond,
		"the fills started wait for room")
	require.Less(t, runtime.NumGoroutine()-before, 32, "the fills waiting to start take no goroutine")

	require.NoError(t, fs.Delete(ctx, "object"))
	require.Eventually(t, func() bool {
		return waiting() == 0 && !fs.diskCache.fillsUnderWay("object") && runtime.NumGoroutine()-before < 32
	}, 5*time.Second, time.Millisecond, "the delete ends every range")
	f.mu.Lock()
	require.Zero(t, heldMemory(d))
	f.mu.Unlock()
	require.Zero(t, heldRoom(d.server), "no range took room")
}

// A prefetch whose bytes a range download a read started holds keeps it downloading once
// the read leaves, and the range is cached.
func TestBlackBoxPrefetchKeepsJoinedReadDownload(t *testing.T) {
	const unit = 64 << 10
	fs, storage, data := newPrefetchRangeFS(t, unit, 4*unit)
	d := fs.diskCache
	d.memoryPrefetchMax = unit / 2
	entered, gate := make(chan struct{}), make(chan struct{})
	var once sync.Once
	d.hooks.beforeCacheTempFileCreate = func() {
		once.Do(func() {
			close(entered)
			<-gate
		})
	}
	release := sync.OnceFunc(func() { close(gate) })
	t.Cleanup(release)

	readCtx, cancelRead := context.WithCancel(context.Background())
	readDone := make(chan error, 1)
	go func() {
		vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
			Entries: []IOEntry{{Offset: 100, Size: 10}}}
		readDone <- fs.Read(readCtx, vec)
		vec.Release()
	}()
	<-entered
	f := &d.prefetchObjects
	diskPath := d.objectKey("object")
	var started *cacheFill
	f.mu.Lock()
	for _, p := range f.m[diskPath] {
		if p.readStarted {
			started = p
		}
	}
	f.mu.Unlock()
	require.NotNil(t, started, "the read started a range download")

	ctx := context.Background()
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: unit,
		Ranges: []PrefetchRange{{Offset: 0, Size: unit}}, Scan: &PrefetchScan{}}))
	kept := func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return started.kept
	}
	require.Eventually(t, kept, 5*time.Second, time.Millisecond, "the prefetch's fill relies on it")
	cancelRead()
	require.ErrorIs(t, <-readDone, context.Canceled)
	require.NoError(t, started.ctx.Err(), "the download the prefetch holds goes on")

	release()
	require.Eventually(t, func() bool { return !fs.diskCache.fillsUnderWay("object") }, 5*time.Second, time.Millisecond)
	f.reads.Wait()
	reads := storage.reads.Load()
	require.Equal(t, data[100:110], readRangeOf(t, fs, 0, 100, 10))
	require.Equal(t, reads, storage.reads.Load(), "the range was cached")
}

// A prefetch of many disjoint ranges leaves fills and room to the prefetch of another
// object, which reads while the wide prefetch waits.
func TestBlackBoxWideRangePrefetchLeavesConnectionsToOthers(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	ctx := context.Background()
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Entries: []IOEntry{{Size: 64 << 10, Data: data[:64<<10]}}, Policy: SkipDiskCache | SkipMemoryCache}))
	share := 14
	require.Less(t, share, fs.server.backgroundMax)
	key := fs.pathToKey("object")
	g := newBlackBoxStorage(fs.server.raw, func(k string, _ int64) bool { return k == key })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	t.Cleanup(release)
	ranges := columnRanges(14, 20<<10, 1<<10)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 300 << 10, Ranges: ranges, Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == share }, 5*time.Second, time.Millisecond)
	require.Never(t, func() bool { cur, _, _ := g.counts(key); return cur > share }, 100*time.Millisecond, time.Millisecond)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "other", Size: 64 << 10, Ranges: []PrefetchRange{{Offset: 0, Size: 1 << 10}}, Scan: &PrefetchScan{}}))
	otherKey := fs.diskCache.objectKey("other")
	require.Eventually(t, func() bool { _, ok := fs.diskCache.ranges.covering(otherKey, 0, 1<<10); return ok },
		5*time.Second, time.Millisecond, "the other object's prefetch read while the wide prefetch waited")
	release()
	require.Eventually(t, func() bool { _, _, total := g.counts(key); return total == len(ranges) }, 5*time.Second, time.Millisecond)
}

// A read of an object whose download waits, while the background fills and all background
// room are held, joins that download and promotes it: it is read once, in two parts from
// the read's block, serves the read and is cached whole.
func TestS3FSReadOfQueuedObjectJoinsItsDownload(t *testing.T) {
	fs, storage, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
	busyFills(t, fs)
	defer occupy(t, fs.server, backgroundRequest, fs.server.backgroundMax-1)()
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: int64(len(data)), Due: 1 << 62}))
	require.True(t, fs.diskCache.isQueued("object"))
	got, err := readFullObjectOf(fs, 100, 10)
	require.NoError(t, err)
	require.Equal(t, data[100:110], got)
	require.False(t, fs.diskCache.isQueued("object"))
	require.Eventually(t, func() bool { return fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))) },
		10*time.Second, time.Millisecond)
	require.Equal(t, int64(2), storage.reads.Load(), "the whole object in two parts, read once")
	require.ElementsMatch(t, []int64{0, 192 << 10}, storage.starts)
}

// A prefetch of a cached object marks it used: the disk cache evicts first the objects no
// prefetch asked for.
func TestPrefetchOfCachedObjectMarksItUsed(t *testing.T) {
	fs, _, _ := newObjectSplitFS(t, objectSplitConfig{parts: 1}, 1)
	d := fs.diskCache
	ctx := context.Background()
	<-d.load.done
	a, b := d.wholePath("a", 1), d.wholePath("b", 1)
	for _, p := range []string{a, b} {
		require.NoError(t, os.MkdirAll(filepath.Dir(p), 0o755))
		require.NoError(t, os.WriteFile(p, []byte{1}, 0o644))
		d.cache.Set(ctx, p, struct{}{}, 1)
	}
	busyFills(t, fs)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "a", Size: 1}))
	require.False(t, fs.diskCache.isQueued("a"), "a cached object is not queued")
	d.cache.ForceEvict(ctx, 1)
	require.True(t, d.cache.Contains(a))
	require.False(t, d.cache.Contains(b))
}

// A prefetch of ranges, or a read of entries, downloads each range on its own, adjacent
// ones together, never the bytes between them, and caches no whole object; a later read
// within them is served from the cache.
func TestS3FSPrefetchRanges(t *testing.T) {
	type then func(t *testing.T, fs *S3FS, storage *countingStorage, req ObjectPrefetch)
	for _, tc := range []struct {
		name     string
		unit     int64
		prefetch bool
		// policy reads the entries, unless prefetch is set, and the cached range.
		policy       Policy
		entries      []IOEntry
		reads, bytes int64
		cached       IOEntry
		then         then
	}{
		{name: "prefetch apart", unit: 256 << 10, prefetch: true, policy: ColumnPrefetch,
			entries: []IOEntry{{Offset: 100, Size: 10}, {Offset: 200 << 10, Size: 20}}, reads: 2, bytes: 30,
			cached: IOEntry{Offset: 200<<10 + 5, Size: 10},
			// Once the ranges are cached, a prefetch of them is not queued.
			then: func(t *testing.T, fs *S3FS, _ *countingStorage, req ObjectPrefetch) {
				ctx := context.Background()
				require.True(t, fs.haveObject(req))
				busyFills(t, fs)
				require.NoError(t, fs.PrefetchObject(ctx, req))
				require.False(t, fs.diskCache.isQueued("object"), "its ranges are cached")
				require.False(t, fs.haveObject(ObjectPrefetch{FilePath: "object", Size: 300 << 10}),
					"a whole request wants the object whatever ranges are cached")
				req.Ranges = []PrefetchRange{{Offset: 1000, Size: 10}}
				require.NoError(t, fs.PrefetchObject(ctx, req))
				require.True(t, fs.diskCache.isQueued("object"))
			}},
		{name: "prefetch adjacent", unit: 256 << 10, prefetch: true, policy: ColumnPrefetch,
			entries: []IOEntry{{Offset: 4000, Size: 10}, {Offset: 1100, Size: 50}, {Offset: 1000, Size: 100}}, reads: 2, bytes: 160,
			cached: IOEntry{Offset: 1050, Size: 70},
			// A prefetch of cached and uncached ranges reads only the uncached ones.
			then: func(t *testing.T, fs *S3FS, storage *countingStorage, req ObjectPrefetch) {
				req.Ranges = append(req.Ranges[1:], PrefetchRange{Offset: 1150, Size: 20})
				req.Size = 180
				fs.prefetchObject(context.Background(), req)
				require.Equal(t, int64(3), storage.reads.Load(), "only the uncached range is read")
				require.Equal(t, int64(180), storage.bytes.Load())
			}},
		{name: "column read apart", unit: 256 << 10, policy: ColumnPrefetch,
			entries: []IOEntry{{Offset: 100, Size: 10}, {Offset: 200 << 10, Size: 10}}, reads: 2, bytes: 20,
			cached: IOEntry{Offset: 102, Size: 5}},
		{name: "column read touching", unit: 256 << 10, policy: ColumnPrefetch,
			entries: []IOEntry{{Offset: 1000, Size: 10}, {Offset: 1010, Size: 20}}, reads: 1, bytes: 30,
			cached: IOEntry{Offset: 1012, Size: 10}},
		{name: "read apart", unit: 4096,
			entries: []IOEntry{{Offset: 100, Size: 10}, {Offset: 40000, Size: 5000}}, reads: 2, bytes: 4096 + 8192,
			cached: IOEntry{Offset: 41000, Size: 100}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, tc.unit, 300<<10)
			ctx := context.Background()
			req := ObjectPrefetch{FilePath: "object"}
			for _, e := range tc.entries {
				req.Size += e.Size
				req.Ranges = append(req.Ranges, PrefetchRange{Offset: e.Offset, Size: e.Size})
			}
			if tc.prefetch {
				fs.prefetchObject(ctx, req)
			} else {
				vec := &IOVector{FilePath: "object", Policy: tc.policy | SkipFullFilePreloads | SkipMemoryCache,
					Entries: slices.Clone(tc.entries)}
				require.NoError(t, fs.Read(ctx, vec))
				for i, e := range tc.entries {
					require.Equal(t, data[e.Offset:e.Offset+e.Size], vec.Entries[i].Data)
				}
				vec.Release()
				waitObjectsEnd(fs.diskCache, "object")
				flushPublishers(fs)
			}
			require.Equal(t, tc.reads, storage.reads.Load())
			require.Equal(t, tc.bytes, storage.bytes.Load(), "not the bytes between the ranges")
			require.False(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))))
			c := tc.cached
			require.Equal(t, data[c.Offset:c.Offset+c.Size], readRangeOf(t, fs, tc.policy, c.Offset, c.Size))
			require.Equal(t, tc.reads, storage.reads.Load(), "a read within a range is served from the cache")
			if tc.then != nil {
				tc.then(t, fs, storage, req)
			}
		})
	}
}

// A download at a size other than the object's caches nothing.
func TestS3FSPrefetchObjectOfWrongSizeCachesNothing(t *testing.T) {
	for _, delta := range []int{-1000, 1000} {
		fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
		fs.diskCache.prefetched = prefetchedObjects{}
		fs.prefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: int64(len(data) + delta)})
		require.Empty(t, fs.diskCache.objectFiles("object"), "size off by %d", delta)
	}
}

func TestPrefetchedObjectsBound(t *testing.T) {
	var o prefetchedObjects
	_, ok := o.size("a")
	require.False(t, ok)
	o.store("a", 10)
	o.store("a", 10)
	size, ok := o.size("a")
	require.True(t, ok)
	require.Equal(t, int64(10), size)
	require.Len(t, o.m, 1)
	for i := len(o.m); i < prefetchedObjectsMax; i++ {
		o.store(fmt.Sprint(i), 1)
	}
	o.store("b", 20)
	require.Len(t, o.m, 1, "past the bound the sizes are dropped")
	_, ok = o.size("a")
	require.False(t, ok)
}

// A column read of an object whose column ranges are queued joins the fill holding its
// bytes, which starts as foreground, and is served from it; the fill of another range
// stays queued, while a read of a column no fill holds reads it, until a read of it joins
// it. A column read joins a queued fill of the whole object.
func TestS3FSColumnReadJoinsQueuedColumns(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 256<<10, 300<<10)
	ctx := context.Background()
	busyFills(t, fs)
	req := ObjectPrefetch{
		FilePath: "object",
		Size:     30,
		Due:      1 << 62,
		Ranges:   []PrefetchRange{{Offset: 100, Size: 10}, {Offset: 200 << 10, Size: 20}},
	}
	require.NoError(t, fs.PrefetchObject(ctx, req))
	require.True(t, fs.diskCache.isQueued("object"))

	cache := &publicationVectorCache{}
	vec := &IOVector{FilePath: "object", Policy: ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache,
		Caches: []IOVectorCache{cache}, Entries: []IOEntry{{Offset: 200<<10 + 5, Size: 10}}}
	require.NoError(t, fs.Read(ctx, vec))
	waitObjectsEnd(fs.diskCache, "object")
	require.Equal(t, data[200<<10+5:200<<10+15], vec.Entries[0].Data)
	vec.Release()
	require.Equal(t, int64(1), cache.updates.Load(), "the read updates the caches it missed")
	require.True(t, fs.diskCache.isQueued("object"), "the fill of the other range stays queued")
	require.Equal(t, int64(1), storage.reads.Load())
	require.Equal(t, int64(20), storage.bytes.Load(), "the column range, not the bytes around it")
	require.Equal(t, data[50<<10:50<<10+5], readRangeOf(t, fs, ColumnPrefetch, 50<<10, 5), "a column no fill holds")
	require.True(t, fs.diskCache.isQueued("object"), "the fill of the other range stays queued")
	require.Equal(t, int64(2), storage.reads.Load())
	require.Equal(t, data[102:107], readRangeOf(t, fs, ColumnPrefetch, 102, 5))
	require.False(t, fs.diskCache.isQueued("object"))
	require.Equal(t, int64(3), storage.reads.Load())
	require.Equal(t, int64(35), storage.bytes.Load(), "the column ranges, not the bytes around them")
	require.Equal(t, data[103:106], readRangeOf(t, fs, ColumnPrefetch, 103, 3))
	require.Equal(t, int64(3), storage.reads.Load(), "served from the column ranges")

	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: int64(len(data)), Due: 1 << 62}))
	require.Equal(t, data[150<<10:150<<10+10], readRangeOf(t, fs, ColumnPrefetch, 150<<10, 10))
	require.False(t, fs.diskCache.isQueued("object"), "a column read joins a whole-object download")
	require.Eventually(t, func() bool { return fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))) },
		10*time.Second, time.Millisecond)
}

// A prefetch of ranges of more bytes than a MemoryPrefetchObject downloads them to their
// cache files as FilePrefetchObjects, not the object.
func TestS3FSPrefetchPastMemoryMaxDownloadsFiles(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 256<<10, 300<<10)
	fs.diskCache.memoryPrefetchMax = 20
	ranges := []PrefetchRange{{Offset: 100, Size: 10}, {Offset: 200 << 10, Size: 20}}
	fs.prefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: 30, Ranges: ranges})
	require.Equal(t, int64(30), storage.bytes.Load(), "only the ranges")
	key := fs.diskCache.objectKey("object")
	for _, r := range ranges {
		_, cached := fs.diskCache.ranges.covering(key, r.Offset, r.Size)
		require.True(t, cached)
	}
	require.False(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))))
	require.Empty(t, fs.diskCache.prefetchObjects.m, "no MemoryPrefetchObject")
	require.Equal(t, data[200<<10+5:200<<10+15], readRangeOf(t, fs, ColumnPrefetch, 200<<10+5, 10))
	require.Equal(t, int64(30), storage.bytes.Load(), "served from the cached range")
}

// An exact read reads only its entry, and leaves a queued download of the object's column
// ranges to their fill.
func TestS3FSExactReadLeavesQueuedColumns(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 64<<10, 300<<10)
	ctx := context.Background()
	busyFills(t, fs)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
		FilePath: "object", Size: 10, Due: 1 << 62, Ranges: []PrefetchRange{{Offset: 0, Size: 10}},
	}))
	require.Equal(t, data[200<<10:200<<10+10], readRangeOf(t, fs, ExactRead, 200<<10, 10))
	require.Equal(t, int64(10), storage.bytes.Load())
	require.True(t, fs.diskCache.isQueued("object"), "the column ranges stay queued")
}

// A column prefetch past the bytes of the flights under way reads its ranges one by one,
// and those reads do not wait for the prefetch itself.
func TestS3FSPrefetchPastFlightBytesFinishes(t *testing.T) {
	fs, storage, _ := newPrefetchRangeFS(t, 256<<10, 300<<10)
	fs.server.memoryLimit = 1
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{
		FilePath: "object", Size: 30, Due: 1 << 62,
		Ranges: []PrefetchRange{{Offset: 100, Size: 10}, {Offset: 200 << 10, Size: 20}},
	}))
	require.Eventually(t, func() bool { return !fs.diskCache.fillsUnderWay("object") && !fs.diskCache.isQueued("object") },
		10*time.Second, time.Millisecond)
	require.Equal(t, int64(2), storage.reads.Load())
}

// A column prefetch past the background share of the memory budget runs file-backed at
// once, never waiting for memory, and caches its range.
func TestS3FSColumnPrefetchPastMemoryRunsFileBacked(t *testing.T) {
	fs, storage, _ := newPrefetchRangeFS(t, 256<<10, 300<<10)
	fs.server.memoryLimit = 2 * flightBufSize(10)
	held := &heldStorage{ObjectStorage: fs.server.raw, read: make(chan struct{}), released: make(chan struct{})}
	fs.server.raw = held
	ctx := context.Background()
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 10, Ranges: []PrefetchRange{{Offset: 100, Size: 10}}}))
	<-held.read
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 20, Ranges: []PrefetchRange{{Offset: 200 << 10, Size: 20}}}))
	require.Eventually(t, func() bool {
		_, cached := fs.diskCache.ranges.covering(fs.diskCache.objectKey("object"), 200<<10, 20)
		return cached
	}, 10*time.Second, time.Millisecond, "the second prefetch runs while the first holds the background memory")
	require.Equal(t, int64(2), storage.reads.Load())
	require.Equal(t, flightBufSize(10), heldMemory(fs.diskCache), "the second prefetch took no memory")
	close(held.released)
	waitFillsEnd(fs.diskCache, "object")
}

// A column read the disk cache holds returns from it while a download of other columns of
// its object stays queued, without waiting to join that download.
func TestS3FSCachedColumnReadDoesNotWaitForQueuedColumns(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 256<<10, 300<<10)
	ctx := context.Background()
	fs.prefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 10, Ranges: []PrefetchRange{{Offset: 100, Size: 10}}})
	reads := storage.reads.Load()
	busyFills(t, fs)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
		FilePath: "object", Size: 20, Due: 1 << 62, Ranges: []PrefetchRange{{Offset: 200 << 10, Size: 20}},
	}))
	require.True(t, fs.diskCache.isQueued("object"))

	got := make(chan []byte, 1)
	go func() { got <- readRangeOf(t, fs, ColumnPrefetch, 102, 5) }()
	select {
	case b := <-got:
		require.Equal(t, data[102:107], b)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the cached read waited for the queued download")
	}
	require.Equal(t, reads, storage.reads.Load(), "served from the disk cache")
	require.True(t, fs.diskCache.isQueued("object"), "the other columns stay queued")
}

// A read that skips the disk cache reads its bytes from object storage without waiting
// for a queued download of its object, which stays queued, or for a prefetch range of it
// under way, which then returns its own bytes.
func TestS3FSCacheBypassReadLeavesQueuedObject(t *testing.T) {
	for _, active := range []bool{false, true} {
		fs, storage, data := newPrefetchRangeFS(t, 256<<10, 300<<10)
		ctx, cancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
		defer cancel()
		first := make(chan []byte, 1)
		release := func() {}
		if active {
			blocking := &blockingStorage{ObjectStorage: fs.server.raw, started: make(chan struct{}), release: make(chan struct{})}
			fs.server.raw = blocking
			release = sync.OnceFunc(func() { close(blocking.release) })
			t.Cleanup(release)
			go func() { first <- readRangeOf(t, fs, 0, 5000, 2) }()
			<-blocking.started
		} else {
			busyFills(t, fs)
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: int64(len(data)), Due: 1 << 62}))
			require.True(t, fs.diskCache.isQueued("object"))
		}
		before := storage.reads.Load()
		vec := &IOVector{FilePath: "object", Policy: SkipAllCache, Entries: []IOEntry{{Offset: 6000, Size: 100}}}
		require.NoError(t, fs.Read(ctx, vec), "active %v", active)
		require.Equal(t, data[6000:6100], vec.Entries[0].Data)
		vec.Release()
		require.Equal(t, before+1, storage.reads.Load(), "the read reads object storage itself")
		if active {
			release()
			require.Equal(t, data[5000:5002], <-first)
		} else {
			require.True(t, fs.diskCache.isQueued("object"), "the download stays queued")
		}
	}
}

// errStressInjected is the error of a read of object storage the stress storage fails.
var errStressInjected = errors.New("stress: injected storage failure")

// stressStorage delays reads of object storage by up to a millisecond, and fails a share
// of them, at open or part way through the body, while faults is set.
type stressStorage struct {
	ObjectStorage
	faults atomic.Bool
	seed   atomic.Uint64
}

func (s *stressStorage) rand() *rand.Rand {
	return rand.New(rand.NewPCG(s.seed.Add(1), 7))
}

func (s *stressStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	r := s.rand()
	select {
	case <-time.After(time.Duration(r.IntN(1000)) * time.Microsecond):
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	faults := s.faults.Load()
	if faults && r.IntN(20) == 0 {
		return nil, errStressInjected
	}
	rc, err := s.ObjectStorage.Read(ctx, key, min, max)
	if err != nil || !faults || r.IntN(20) != 0 {
		return rc, err
	}
	return &failingBody{ReadCloser: rc, left: r.IntN(64 << 10)}, nil
}

// failingBody fails after left bytes.
type failingBody struct {
	io.ReadCloser
	left int
}

func (b *failingBody) Read(p []byte) (int, error) {
	if b.left <= 0 {
		return 0, errStressInjected
	}
	if len(p) > b.left {
		p = p[:b.left]
	}
	n, err := b.ReadCloser.Read(p)
	b.left -= n
	return n, err
}

// stressContent is version v of object i, of size bytes.
func stressContent(i, v, size int) []byte {
	b := make([]byte, size)
	for j := range b {
		b[j] = byte(j*7 + j/509 + v*31 + i*13)
	}
	return b
}

// Workers read, prefetch, cancel, delete and rewrite objects at once through the public
// file-service calls, against few download connections, a tight buffer budget, small
// MemoryPrefetchObjects, split runs, and an object storage that delays and fails reads.
// Every read returns, with bytes of one version of its object or an error the workload
// explains; once the workers stop, no prefetch object, buffer, connection or prefetch
// stays, and every object reads its last version.
func TestPrefetchObjectsUnderStress(t *testing.T) {
	const (
		// A unit of four blocks makes runs of blocks a read takes no bytes of.
		unit    = 256 << 10
		objects = 6
		workers = 16
	)
	duration := 3 * time.Second
	if testing.Short() {
		duration = time.Second
	}
	sizes := []int{100<<10 + 3, 300 << 10, 700<<10 + 11, 1 << 20, 1500 << 10, 2<<20 + 5}
	fs, _, _ := newPrefetchRangeFS(t, unit, 1)
	d := fs.diskCache
	d.server.memoryLimit = 4 * prefetchBlockMax
	limitRoom(t, d.server, 6, 3)
	d.memoryPrefetchMax = 256 << 10
	d.objectSplit = objectSplitConfig{parts: 2, min: 256 << 10}
	storage := &stressStorage{ObjectStorage: fs.server.raw}
	fs.server.raw = storage
	ctx := context.Background()

	// versions[i] is the newest version of object i written; a read may see any version
	// from the one current when it began on.
	var versions [objects]atomic.Int64
	var rewrite [objects]sync.Mutex
	name := func(i int) string { return fmt.Sprintf("object-%d", i) }
	write := func(i, v int) {
		require.NoError(t, fs.Write(ctx, IOVector{FilePath: name(i), Entries: []IOEntry{{Size: int64(sizes[i]), Data: stressContent(i, v, sizes[i])}},
			Policy: SkipDiskCache | SkipMemoryCache}))
	}
	for i := range objects {
		write(i, 0)
	}
	// matches reports whether data is bytes [off, off+len(data)) of a version of object i
	// from first on.
	matches := func(i, first int, off int64, data []byte) bool {
		for v := first; v <= int(versions[i].Load()); v++ {
			want := stressContent(i, v, sizes[i])[off : off+int64(len(data))]
			if string(want) == string(data) {
				return true
			}
		}
		return false
	}
	explained := func(err error) bool {
		return errors.Is(err, errStressInjected) || errors.Is(err, context.Canceled) ||
			errors.Is(err, context.DeadlineExceeded) || moerr.IsMoErrCode(err, moerr.ErrFileNotFound)
	}

	storage.faults.Store(true)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	var failures atomic.Value
	fail := func(format string, args ...any) { failures.CompareAndSwap(nil, fmt.Sprintf(format, args...)) }
	var scans sync.Map
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := rand.New(rand.NewPCG(uint64(w), 11))
			for {
				select {
				case <-stop:
					return
				default:
				}
				// Half of the work is on one object, so reads and prefetches overlap.
				i := r.IntN(objects)
				if r.IntN(2) == 0 {
					i = 2
				}
				size := int64(sizes[i])
				switch op := r.IntN(100); {
				case op < 45:
					// A column read, or a read that may download the whole object, of
					// entries anywhere in the object, some under a short deadline.
					var policy Policy = SkipMemoryCache
					if op < 30 {
						policy |= ColumnPrefetch | SkipFullFilePreloads
					}
					entries := make([]IOEntry, 1+r.IntN(4))
					for e := range entries {
						n := 1 + r.Int64N(min(size, 200<<10))
						if r.IntN(2) == 0 {
							// A short entry across a block boundary.
							n = 1 + r.Int64N(256)
						}
						entries[e] = IOEntry{Offset: r.Int64N(size - n + 1), Size: n}
					}
					readCtx, cancel := context.WithCancel(ctx)
					if r.IntN(5) == 0 {
						readCtx, cancel = context.WithTimeout(ctx, time.Duration(r.IntN(3000))*time.Microsecond)
					}
					first := int(versions[i].Load())
					vec := &IOVector{FilePath: name(i), Policy: policy, Entries: entries}
					done := make(chan error, 1)
					go func() { done <- fs.Read(readCtx, vec) }()
					var err error
					select {
					case err = <-done:
					case <-time.After(diskCacheLifecycleTestTimeout):
						fail("a read of %s did not return", name(i))
						cancel()
						return
					}
					cancel()
					if err != nil {
						if !explained(err) {
							fail("read of %s: unexplained error %v", name(i), err)
						}
					} else {
						for e := range vec.Entries {
							if !matches(i, first, vec.Entries[e].Offset, vec.Entries[e].Data) {
								fail("read of %s at %d: bytes of no version", name(i), vec.Entries[e].Offset)
							}
						}
					}
					vec.Release()
				case op < 75:
					// A prefetch of ranges, or of the whole object, whose scan some
					// workers end soon after.
					scan := &PrefetchScan{}
					req := ObjectPrefetch{FilePath: name(i), Size: size, Scan: scan}
					if op < 65 {
						for range 1 + r.IntN(4) {
							n := 1 + r.Int64N(min(size, 600<<10))
							req.Ranges = append(req.Ranges, PrefetchRange{Offset: r.Int64N(size - n + 1), Size: n})
						}
					}
					if err := fs.PrefetchObject(ctx, req); err != nil {
						fail("prefetch of %s: %v", name(i), err)
					}
					if r.IntN(2) == 0 {
						scan.End()
					} else {
						scans.Store(scan, struct{}{})
					}
				case op < 80:
					// A delete of the object, then its next version.
					rewrite[i].Lock()
					if err := fs.Delete(ctx, name(i)); err != nil {
						fail("delete of %s: %v", name(i), err)
					}
					v := int(versions[i].Add(1))
					write(i, v)
					rewrite[i].Unlock()
				default:
					time.Sleep(time.Duration(r.IntN(500)) * time.Microsecond)
				}
			}
		}()
	}
	time.Sleep(duration)
	close(stop)
	wg.Wait()
	if msg := failures.Load(); msg != nil {
		t.Fatal(msg)
	}

	// Quiesce: end every scan, then nothing stays held.
	storage.faults.Store(false)
	scans.Range(func(k, _ any) bool {
		k.(*PrefetchScan).End()
		return true
	})
	waitFillsIdle(t, d)
	f := &d.prefetchObjects
	f.reads.Wait()
	require.Eventually(t, func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return len(f.m) == 0 && heldMemory(d) == 0
	}, 10*time.Second, time.Millisecond, "no prefetch object or buffer stays")
	require.Eventually(t, func() bool { return heldRoom(d.server) == 0 }, 10*time.Second, time.Millisecond,
		"no connection stays held")

	// Every object reads its last version, whole and by columns.
	for i := range objects {
		want := stressContent(i, int(versions[i].Load()), sizes[i])
		for _, policy := range []Policy{SkipMemoryCache, ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache} {
			vec := &IOVector{FilePath: name(i), Policy: policy, Entries: []IOEntry{{Offset: 0, Size: int64(sizes[i])}}}
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, want, vec.Entries[0].Data, "%s reads its last version", name(i))
			vec.Release()
		}
	}
}

func TestFilePrefetchRetriesKnownObjectEnd(t *testing.T) {
	for _, tc := range []struct {
		name      string
		overshoot int64
	}{
		{name: "request_ends_at_eof"},
		{name: "request_ends_past_eof", overshoot: 1024},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newPrefetchRangeFS(t, 0, 256<<10)
			fs.diskCache.memoryPrefetchMax = 128 << 10
			<-fs.diskCache.load.done
			r := PrefetchRange{Size: int64(len(data)) + tc.overshoot}
			req := ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}

			submitAndWait := func() {
				t.Helper()
				require.NoError(t, fs.PrefetchObject(ctx, req))
				waitFillsIdle(t, fs.diskCache)
			}
			submitAndWait()
			require.EqualValues(t, 1, storage.reads.Load())
			if tc.overshoot > 0 {
				size, ok := fs.diskCache.prefetched.size(fs.diskCache.objectKey(req.FilePath))
				require.True(t, ok)
				require.EqualValues(t, len(data), size, "the first download should learn the real size")
			}
			submitAndWait()

			_, cached := fs.diskCache.ranges.covering(fs.diskCache.objectKey(req.FilePath), 0, int64(len(data)))
			require.True(t, cached, "the retry should cache the range through the known EOF")
		})
	}
}

// slowCancelReader stalls until its context ends, signals it, and returns only once
// proceed is closed, as an HTTP body does while its connection is torn down.
// slowCancelReader returns from Read only once its context is canceled and proceed is
// closed.
type slowCancelReader struct {
	ctx      context.Context
	canceled chan struct{}
	proceed  chan struct{}
}

func (r *slowCancelReader) Read(p []byte) (int, error) {
	<-r.ctx.Done()
	close(r.canceled)
	<-r.proceed
	return 0, r.ctx.Err()
}

func waitPrefetcherIdle(t *testing.T, fs *S3FS) {
	t.Helper()
	waitFillsIdle(t, fs.diskCache)
	fs.diskCache.prefetchObjects.reads.Wait()
}

// A prefetch of an object or a column range whose cached file is gone downloads it
// again, whatever the order of a cold range prefetched with it, and later reads are
// served from the disk cache.
func TestPrefetchObjectRefillsMissingCachedFile(t *testing.T) {
	for _, whole := range []bool{true, false} {
		for _, removed := range []bool{true, false} {
			for _, coldFirst := range []bool{false, true} {
				if whole && coldFirst {
					continue
				}
				name := map[bool]string{true: "whole", false: "range"}[whole] + "/" + map[bool]string{true: "removed", false: "control"}[removed] +
					map[bool]string{true: "/cold first", false: ""}[coldFirst]
				t.Run(name, func(t *testing.T) {
					fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
					ctx := context.Background()
					r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
					cold := PrefetchRange{Offset: 80 << 10, Size: 4 << 10}
					req := func(again bool) ObjectPrefetch {
						if whole {
							return ObjectPrefetch{FilePath: "object", Size: int64(len(data)), Scan: &PrefetchScan{}}
						}
						ranges := []PrefetchRange{r}
						if again && coldFirst {
							ranges = []PrefetchRange{cold, r}
						} else if again {
							ranges = []PrefetchRange{r, cold}
						}
						return ObjectPrefetch{FilePath: "object", Size: int64(len(ranges)) * r.Size, Ranges: ranges, Scan: &PrefetchScan{}}
					}
					read := func(i int) {
						b := PrefetchRange{Offset: r.Offset + int64(i)<<10, Size: 1 << 10}
						var got []byte
						if whole {
							vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
								Entries: []IOEntry{{Offset: b.Offset, Size: b.Size}}}
							require.NoError(t, fs.Read(ctx, vec))
							got = append(got, vec.Entries[0].Data...)
							vec.Release()
						} else {
							var err error
							got, err = readColumnRange(ctx, fs, b)
							require.NoError(t, err)
						}
						require.Equal(t, data[b.Offset:b.Offset+b.Size], got)
					}

					require.NoError(t, fs.PrefetchObject(ctx, req(false)))
					waitPrefetcherIdle(t, fs)
					diskPath := fs.diskCache.wholePath("object", int64(len(data)))
					if !whole {
						rs := fs.diskCache.ranges.of(fs.diskCache.objectKey("object"))
						require.Len(t, rs, 1)
						diskPath = rs[0].diskPath
					}
					require.Eventually(t, func() bool { return fs.diskCache.cache.Contains(diskPath) }, 5*time.Second, time.Millisecond)
					if removed {
						require.NoError(t, os.Remove(diskPath))
					}
					fs.diskCache.files.closeAll()

					before := storage.reads.Load()
					require.NoError(t, fs.PrefetchObject(ctx, req(true)))
					waitPrefetcherIdle(t, fs)
					_, err := os.Stat(diskPath)
					require.NoError(t, err, "the prefetch downloads the file again")
					if !whole {
						require.FileExists(t, fs.diskCache.pathForIOEntry("object", cold.entry()))
						require.Equal(t, map[bool]int64{true: 2, false: 1}[removed], storage.reads.Load()-before)
					}
					base := storage.reads.Load()
					for i := 1; i <= 3; i++ {
						read(i)
					}
					require.Equal(t, base, storage.reads.Load(), "the reads are served from the disk cache")
				})
			}
		}
	}
}

func prefetchBytes(fs *S3FS) int64 {
	return heldMemory(fs.diskCache)
}

// A read waiting on two flights holds only the flight it still waits for: once either
// flight ends, the read holds no budget for it, a read of another range is not held by the
// read, and the range of a flight that ended before the read reached it is read from the
// disk cache, not object storage; once all reads return, no buffer stays counted. The
// flights are started by a column prefetch, or by the read and another read of prefetch
// ranges.
func TestFlightTransitionColumnReadReleasesCopiedFlight(t *testing.T) {
	const unit = 64 << 10
	columns := columnRanges(3, 40<<10, 1<<10)
	for _, tc := range []struct {
		name string
		// unit is the prefetch unit; with none, a column prefetch starts the flights.
		unit  int64
		first int
	}{
		{"the first range ends first", 0, 0},
		{"the second range ends first", 0, 1},
		{"a read joins the flight another read started", unit, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, tc.unit, 300<<10)
			starts := []int64{0, columns[1].Offset}
			other := columns[2].entry()
			buf := flightBufSize(columns[0].Size)
			fs.server.memoryLimit = 4 * buf
			if tc.unit > 0 {
				starts = []int64{0, unit}
				other = IOEntry{Offset: 3*unit + 5, Size: 5}
				buf = flightBufSize(unit)
				fs.server.memoryLimit = 2 * buf
			}
			gates := []chan struct{}{make(chan struct{}), make(chan struct{})}
			release := []func(){sync.OnceFunc(func() { close(gates[0]) }), sync.OnceFunc(func() { close(gates[1]) })}
			defer release[0]()
			defer release[1]()
			g := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return false })
			g.gated = func(_ string, offset int64) chan struct{} {
				for i := range gates {
					if offset == starts[i] {
						return gates[i]
					}
				}
				return nil
			}
			fs.server.raw = g
			ctx := context.Background()
			key := fs.pathToKey("object")
			read := func(ctx context.Context, policy Policy, caches []IOVectorCache, entries ...IOEntry) ([][]byte, error) {
				vec := &IOVector{FilePath: "object", Policy: policy | SkipFullFilePreloads | SkipMemoryCache,
					Caches: caches, Entries: entries}
				if err := fs.Read(ctx, vec); err != nil {
					return nil, err
				}
				defer vec.Release()
				var got [][]byte
				for _, e := range vec.Entries {
					got = append(got, bytes.Clone(e.Data))
				}
				return got, nil
			}
			readCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()
			type result struct {
				data [][]byte
				err  error
			}
			readAsync := func(policy Policy, caches []IOVectorCache, entries ...IOEntry) <-chan result {
				done := make(chan result, 1)
				go func() {
					got, err := read(readCtx, policy, caches, entries...)
					done <- result{got, err}
				}()
				return done
			}
			cache := &publicationVectorCache{}
			var entries []IOEntry
			var done, started <-chan result
			if tc.unit == 0 {
				require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 2 << 10, Ranges: columns[:2], Scan: &PrefetchScan{}}))
				require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 2 }, 5*time.Second, time.Millisecond)
				entries = []IOEntry{columns[0].entry(), columns[1].entry()}
				done = readAsync(ColumnPrefetch, []IOVectorCache{cache}, entries...)
				// The read waits for the first range.
				require.Eventually(t, func() bool { return flightRefs(fs, columns[0].Offset) >= 2 }, 5*time.Second, time.Millisecond)
			} else {
				// Another read starts the flight of unit 1.
				started = readAsync(0, nil, IOEntry{Offset: unit + 5, Size: 5})
				require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 1 }, 5*time.Second, time.Millisecond)
				// The read starts the flight of unit 0 and joins that of unit 1.
				entries = []IOEntry{{Offset: 10, Size: unit}}
				done = readAsync(0, nil, entries...)
				require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 2 }, 5*time.Second, time.Millisecond)
			}
			release[tc.first]()
			// Only the flight still reading holds budget.
			assertEventuallyBytes(t, fs, buf)

			otherCtx, cancelOther := context.WithTimeout(ctx, 2*time.Second)
			defer cancelOther()
			if got, err := read(otherCtx, 0, nil, other); assert.NoError(t, err, "a read of another range is not held by the read") {
				assert.Equal(t, data[other.Offset:other.Offset+other.Size], got[0])
			}
			release[1-tc.first]()
			r := <-done
			require.NoError(t, r.err)
			for i, e := range entries {
				require.Equal(t, data[e.Offset:e.Offset+e.Size], r.data[i])
			}
			if started != nil {
				require.NoError(t, (<-started).err)
				g.mu.Lock()
				ranges := slices.Clone(g.ranges)
				g.mu.Unlock()
				require.ElementsMatch(t, [][2]int64{{unit, 2 * unit}, {0, unit}, {3 * unit, 4 * unit}}, ranges,
					"the read joins the flight of unit 1")
			} else {
				_, _, total := g.counts(key)
				require.Equal(t, 3, total, "each range is read from object storage once")
				require.Equal(t, int64(1), cache.updates.Load(), "the read updates the caches it missed")
			}
			fs.diskCache.prefetchObjects.reads.Wait()
			require.Zero(t, prefetchBytes(fs), "no buffer stays counted")
		})
	}
}

// flightRefs returns the holders of the flight of the object that starts at offset.
func flightRefs(fs *S3FS, offset int64) int {
	f := &fs.diskCache.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, fl := range f.m[fs.diskCache.objectKey("object")] {
		if fl.start == offset {
			fl.mu.Lock()
			defer fl.mu.Unlock()
			return fl.holds + 1
		}
	}
	return 0
}

// A range read whose runs end out of order holds no budget for a run that ended and
// cached its bytes while it waits for an earlier one.
func TestFlightTransitionRangeReadReleasesEndedRun(t *testing.T) {
	const unit = 64 << 10
	fs, _, data := newPrefetchRangeFS(t, unit, 5*unit)
	fs.server.memoryLimit = 2 * unit
	// Unit 1 is cached, so the read runs units 0 and 2 apart.
	readRangeOf(t, fs, 0, unit+5, 5)
	gate := make(chan struct{})
	release := sync.OnceFunc(func() { close(gate) })
	defer release()
	g := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return false })
	g.gated = func(_ string, offset int64) chan struct{} {
		if offset == 0 {
			return gate
		}
		return nil
	}
	fs.server.raw = g
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	key := fs.pathToKey("object")
	read := func(ctx context.Context, offset, size int64) ([]byte, error) {
		vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
			Entries: []IOEntry{{Offset: offset, Size: size}}}
		if err := fs.Read(ctx, vec); err != nil {
			return nil, err
		}
		defer vec.Release()
		return append([]byte(nil), vec.Entries[0].Data...), nil
	}
	done := make(chan error, 1)
	go func() {
		got, err := read(ctx, 10, 2*unit)
		if err == nil && string(got) != string(data[10:10+2*unit]) {
			err = context.Canceled
		}
		done <- err
	}()
	require.Eventually(t, func() bool { _, _, total := g.counts(key); return total == 2 }, 5*time.Second, time.Millisecond)
	// Unit 2 ended; only unit 0, which reads, holds budget.
	assertEventuallyBytes(t, fs, unit)
	otherCtx, cancelOther := context.WithTimeout(ctx, 2*time.Second)
	defer cancelOther()
	got, err := read(otherCtx, 3*unit+5, 5)
	if assert.NoError(t, err, "a read of another unit is not held by the ended run") {
		assert.Equal(t, data[3*unit+5:3*unit+10], got)
	}
	release()
	require.NoError(t, <-done)
	_, _, total := g.counts(key)
	require.Equal(t, 3, total, "each unit is read from object storage once")
}

func (r PrefetchRange) entry() IOEntry { return IOEntry{Offset: r.Offset, Size: r.Size} }

// failOnceStorage fails the first read at offset fail once gate is closed.
type failOnceStorage struct {
	ObjectStorage
	fail   int64
	gate   chan struct{}
	failed atomic.Bool
}

func (s *failOnceStorage) Read(ctx context.Context, key string, from, to *int64) (io.ReadCloser, error) {
	if from != nil && *from == s.fail && s.failed.CompareAndSwap(false, true) {
		select {
		case <-s.gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return nil, errors.New("probe: storage failed")
	}
	return s.ObjectStorage.Read(ctx, key, from, to)
}

func memoryWaiters(fs *S3FS, diskPath string) int {
	f := &fs.diskCache.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, fl := range f.m[diskPath] {
		if o := fl.joinableDownload(); o != nil && o.memory {
			o.mu.Lock()
			n += len(o.waiters)
			o.mu.Unlock()
		}
	}
	return n
}

// A range read joining a read-started flight whose storage read fails fails, as does the
// read that started it, with or without prefetch ranges.
func TestFlightTransitionRangeReadJoinedFailedFlightFails(t *testing.T) {
	for _, tc := range []struct {
		name       string
		unit, size int64
		fail, at   int64
	}{
		{name: "range", size: 300 << 10, fail: 1000, at: 1000},
		{name: "prefetch range", unit: 64 << 10, size: 320 << 10, fail: 64 << 10, at: 64<<10 + 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, tc.unit, int(tc.size))
			s := &failOnceStorage{ObjectStorage: fs.server.raw, fail: tc.fail, gate: make(chan struct{})}
			fs.server.raw = s
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			read := func(offset, size int64) error {
				vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: []IOEntry{{Offset: offset, Size: size}}}
				err := fs.Read(ctx, vec)
				if err == nil {
					vec.Release()
				}
				return err
			}
			diskPath := fs.diskCache.objectKey("object")
			a, b := make(chan error, 1), make(chan error, 1)
			go func() { a <- read(tc.at, 100) }()
			require.Eventually(t, func() bool { return memoryWaiters(fs, diskPath) == 1 }, 5*time.Second, time.Millisecond)
			go func() { b <- read(tc.at+10, 40) }()
			require.Eventually(t, func() bool { return memoryWaiters(fs, diskPath) == 2 }, 5*time.Second, time.Millisecond)
			close(s.gate)
			require.Error(t, <-a, "the read that started the flight")
			require.Error(t, <-b, "the read that joined the flight")
			fs.diskCache.prefetchObjects.reads.Wait()
			require.Zero(t, prefetchBytes(fs))
		})
	}
}

// Budget and room return to what they were after a background column prefetch, whose
// requests never pass their room, and after one of ranges downloading to files, which
// runs at most fillsMax fills at once; every range is cached.
func TestFlightTransitionColumnPrefetchReturnsBudgetAndConns(t *testing.T) {
	for _, file := range []bool{false, true} {
		fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
		n, bound := 8, 2
		ranges := columnRanges(n, 20<<10, 1<<10)
		if file {
			n, bound = 14, 3
			ranges = columnRanges(n, 20<<10, 1<<10)
			fs.diskCache.memoryPrefetchMax = 512
			fs.diskCache.setFillsMax(bound)
		} else {
			limitRoom(t, fs.server, 18, bound)
			fs.server.memoryLimit = 4 * flightBufSize(ranges[0].Size)
		}
		key := fs.pathToKey("object")
		g := newBlackBoxStorage(fs.server.raw, func(k string, _ int64) bool { return k == key })
		fs.server.raw = g
		release := sync.OnceFunc(func() { close(g.gate) })
		defer release()
		require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: int64(n) << 10, Ranges: ranges, Scan: &PrefetchScan{}}))
		inFlight := func() int { cur, _, _ := g.counts(key); return cur }
		require.Eventually(t, func() bool { return inFlight() == bound }, 5*time.Second, time.Millisecond)
		require.Never(t, func() bool { return inFlight() > bound }, 100*time.Millisecond, time.Millisecond)
		require.LessOrEqual(t, heldRoom(fs.server), bound)
		release()
		require.Eventually(t, func() bool { return !fs.diskCache.fillsUnderWay("object") }, 5*time.Second, time.Millisecond)
		fs.diskCache.prefetchObjects.reads.Wait()
		require.Zero(t, prefetchBytes(fs))
		require.Zero(t, heldRoom(fs.server))
		_, most, total := g.counts(key)
		require.Equal(t, n, total)
		require.Equal(t, bound, most)
		objectKey := fs.diskCache.objectKey("object")
		for _, r := range ranges {
			_, ok := fs.diskCache.ranges.covering(objectKey, r.Offset, r.Size)
			require.True(t, ok, "file %v: range %d cached", file, r.Offset)
		}
	}
}

// A background flight no read waits for, or a whole-object download its read no longer
// waits for, whose source stalls, stops after the idle timeout and frees its buffer, its
// connection and its publication, caching nothing.
func TestFlightTransitionIdleFlightFreesBudgetAndConn(t *testing.T) {
	shortIdle(t, 50*time.Millisecond)
	for _, whole := range []bool{false, true} {
		fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
		d := fs.diskCache
		r := PrefetchRange{Offset: 0, Size: 1 << 10}
		if whole {
			fs.server.raw = &probeStallStorage{ObjectStorage: fs.server.raw, n: 64 << 10}
			got, err := readFullObjectOf(fs, 10, 10)
			require.NoError(t, err)
			require.Equal(t, data[10:20], got, "the read returns bytes the download wrote before it stalled")
		} else {
			g := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
			fs.server.raw = g
			defer close(g.gate)
			require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}, Scan: &PrefetchScan{}}))
			key := fs.pathToKey("object")
			require.Eventually(t, func() bool { cur, _, _ := g.counts(key); return cur == 1 }, 5*time.Second, time.Millisecond)
		}
		require.Eventually(t, func() bool {
			d.pub.mu.Lock()
			pending := d.pub.mu.pending
			d.pub.mu.Unlock()
			return !d.fillsUnderWay("object") && prefetchBytes(fs) == 0 && heldRoom(fs.server) == 0 &&
				d.download(d.objectKey("object")) == nil && pending == 0 && fillsRunning(fs.server) == 0
		}, 5*time.Second, time.Millisecond, "whole %v", whole)
		_, ok := d.ranges.covering(d.objectKey("object"), r.Offset, r.Size)
		require.False(t, ok)
		require.Empty(t, d.objectFiles("object"))
	}
}

func assertEventuallyBytes(t *testing.T, fs *S3FS, want int64) bool {
	return assert.Eventually(t, func() bool { return prefetchBytes(fs) == want }, 2*time.Second, time.Millisecond,
		"the buffer of the copied, ended flight is still charged: %d", prefetchBytes(fs))
}

// offsetGateStorage blocks the first read at offset 0 until open0 is closed, and every
// read at offset gated until openGated is closed, or their contexts end.
type offsetGateStorage struct {
	ObjectStorage
	gated     int64
	first0    atomic.Bool
	open0     chan struct{}
	openGated chan struct{}
	started   chan int64
}

func (s *offsetGateStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	var gate chan struct{}
	switch {
	case min != nil && *min == 0 && s.first0.CompareAndSwap(false, true):
		gate = s.open0
	case min != nil && *min == s.gated:
		gate = s.openGated
	}
	if gate != nil {
		s.started <- *min
		select {
		case <-gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return s.ObjectStorage.Read(ctx, key, min, max)
}

// A read joining two prefetch objects, one of which a delete and rewrite of the object
// stops, returns the bytes of one version of the object, not the first object's bytes
// of the old version followed by the new version read in place of the stopped one.
func TestReadJoiningPrefetchObjectsAcrossDeleteReturnsOneVersion(t *testing.T) {
	const half = 16 << 10
	ctx := context.Background()
	fs, _, v1 := newPrefetchRangeFS(t, 0, 2*half)
	v2 := slices.Clone(v1)
	for i := range v2 {
		v2[i]++
	}
	storage := &offsetGateStorage{ObjectStorage: fs.server.raw, gated: half,
		open0: make(chan struct{}), openGated: make(chan struct{}), started: make(chan int64, 16)}
	fs.server.raw = storage
	d := fs.diskCache
	read := func(offset, size int64) ([]byte, error) {
		vec := &IOVector{FilePath: "object", Policy: ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache,
			Entries: []IOEntry{{Offset: offset, Size: size}}}
		defer vec.Release()
		if err := fs.Read(ctx, vec); err != nil {
			return nil, err
		}
		return slices.Clone(vec.Entries[0].Data), nil
	}
	type result struct {
		data []byte
		err  error
	}
	start := func(offset, size int64) chan result {
		ch := make(chan result, 1)
		go func() {
			data, err := read(offset, size)
			ch <- result{data, err}
		}()
		return ch
	}
	// Each half is read by a prefetch object of its own, held in object storage.
	first := start(0, half)
	<-storage.started
	second := start(half, half)
	<-storage.started
	key := d.objectKey("object")
	refs := func() []int {
		f := &d.prefetchObjects
		f.mu.Lock()
		defer f.mu.Unlock()
		out := make([]int, 0, len(f.m[key]))
		for _, m := range f.m[key] {
			m.mu.Lock()
			out = append(out, m.holds)
			m.mu.Unlock()
		}
		return out
	}
	before := refs()
	require.Len(t, before, 2)
	// The read of both halves joins both.
	both := start(0, 2*half)
	require.Eventually(t, func() bool {
		now := refs()
		return len(now) == 2 && now[0] == before[0]+1 && now[1] == before[1]+1
	}, diskCacheLifecycleTestTimeout, time.Millisecond)
	// The first half arrives; then the object is deleted and rewritten before the second
	// half is read again.
	close(storage.open0)
	r := <-first
	require.NoError(t, r.err)
	require.Equal(t, v1[:half], r.data)
	require.NoError(t, fs.Delete(ctx, "object"))
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: int64(len(v2)), Data: v2}},
		Policy: SkipDiskCache | SkipMemoryCache}))
	close(storage.openGated)
	<-second
	r = <-both
	require.NoError(t, r.err)
	require.True(t, bytes.Equal(r.data, v1) || bytes.Equal(r.data, v2),
		"the bytes of one version: first half v1 %v, second half v2 %v",
		bytes.Equal(r.data[:half], v1[:half]), bytes.Equal(r.data[half:], v2[half:]))
}

// A read whose entry is held together by cached ranges, one opened for an earlier entry
// of the same read and the other published after a delete and rewrite of the object
// meanwhile, returns the bytes of one version of the object. The control, with no delete,
// reads the entry from the cached ranges.
func TestReadCoveringRangesAcrossDeleteReturnsOneVersion(t *testing.T) {
	for _, rewrite := range []bool{true, false} {
		t.Run(map[bool]string{true: "rewrite", false: "control"}[rewrite], func(t *testing.T) {
			const half = 4 << 10
			ctx := context.Background()
			fs, storage, v1 := newPrefetchRangeFS(t, 0, 2*half)
			v2 := slices.Clone(v1)
			for i := range v2 {
				v2[i]++
			}
			// The halves are cached as ranges of their own.
			require.Equal(t, v1[:half], readRangeOf(t, fs, 0, 0, half))
			require.Equal(t, v1[half:], readRangeOf(t, fs, 0, half, half))
			reading, resume := make(chan struct{}), make(chan struct{})
			vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: []IOEntry{
				{Offset: 0, Size: half, WriterForRead: readShareWriterFunc(func(b []byte) (int, error) {
					close(reading)
					<-resume
					return len(b), nil
				})},
				{Offset: half / 2, Size: half},
			}}
			defer vec.Release()
			done := make(chan error, 1)
			go func() { done <- fs.Read(ctx, vec) }()
			// The first entry is read from the first half; then the object is deleted,
			// rewritten and its halves cached again before the second entry is read.
			<-reading
			if rewrite {
				require.NoError(t, fs.Delete(ctx, "object"))
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: int64(len(v2)), Data: v2}},
					Policy: SkipDiskCache | SkipMemoryCache}))
				require.Equal(t, v2[:half], readRangeOf(t, fs, 0, 0, half))
				require.Equal(t, v2[half:], readRangeOf(t, fs, 0, half, half))
			}
			reads := storage.reads.Load()
			close(resume)
			require.NoError(t, <-done)
			got := vec.Entries[1].Data
			if !rewrite {
				require.Equal(t, v1[half/2:half/2+half], got)
				require.Equal(t, reads, storage.reads.Load(), "the entry is read from the cached ranges")
				return
			}
			require.True(t, bytes.Equal(got, v1[half/2:half/2+half]) || bytes.Equal(got, v2[half/2:half/2+half]),
				"the bytes of one version: first part v1 %v, second part v2 %v",
				bytes.Equal(got[:half/2], v1[half/2:half]), bytes.Equal(got[half/2:], v2[half:half/2+half]))
		})
	}
}

// /debug/prefetch_objects lists the prefetch objects, the download holding background room
// and the server's requests: a range waiting for room is registered while its request
// waits.
func TestDebugPrefetchObjectsListsWaitingObjects(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	ctx := context.Background()
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Entries: []IOEntry{{Size: int64(len(data)), Data: data}},
		Policy: SkipDiskCache | SkipMemoryCache}))
	limitRoom(t, fs.server, 2, 1)
	otherKey := fs.pathToKey("other")
	g := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return key == otherKey })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	defer release()
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "other", Size: int64(len(data)), Scan: &PrefetchScan{}}))
	require.Eventually(t, func() bool { cur, _, _ := g.counts(otherKey); return cur == 1 }, 5*time.Second, time.Millisecond)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 2 << 10, Ranges: columnRanges(1, 40<<10, 1<<10), Scan: &PrefetchScan{}}))
	page := func() string {
		w := httptest.NewRecorder()
		http.DefaultServeMux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/debug/prefetch_objects", nil))
		return w.Body.String()
	}
	objectPath := fs.diskCache.objectKey("object")
	require.Eventually(t, func() bool {
		return strings.Contains(page(), "fill "+objectPath+" [0, 1024) waiting for room")
	}, 5*time.Second, time.Millisecond, "the waiting object is listed")
	require.Eventually(t, func() bool {
		return strings.Contains(page(), fmt.Sprintf("background running %d of %d waiting 1", fs.server.backgroundMax, fs.server.backgroundMax))
	}, 5*time.Second, time.Millisecond, "its request waits")
	require.Contains(t, page(), "file "+fs.diskCache.objectKey("other")+" [0, ")
	release()
	require.Eventually(t, func() bool { return !strings.Contains(page(), objectPath) }, 5*time.Second, time.Millisecond,
		"an object that ended is not listed")
}
