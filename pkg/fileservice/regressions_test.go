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
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/util/toml"
	"github.com/stretchr/testify/require"
)

func TestExactReadSkipsGapBeforeTail(t *testing.T) {
	for _, tc := range []struct {
		name     string
		tailSize int64
	}{
		{name: "known_size", tailSize: 16},
		{name: "read_to_end", tailSize: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 0, 1024)
			vector := &IOVector{
				FilePath: "object",
				Policy:   ExactColumnsRead | SkipAllCache,
				Entries:  []IOEntry{{Offset: 0, Size: 16}, {Offset: 1008, Size: tc.tailSize}},
			}
			defer vector.Release()
			require.NoError(t, fs.Read(context.Background(), vector))
			require.Equal(t, data[:16], vector.Entries[0].Data)
			require.Equal(t, data[1008:], vector.Entries[1].Data)
			storage.mu.Lock()
			defer storage.mu.Unlock()
			require.Equal(t, []int64{0, 1008}, storage.starts, "exact reads should open each entry without reading the intervening bytes")
		})
	}
}

type idlePartStorage struct {
	ObjectStorage
	opened   atomic.Bool
	canceled chan struct{}
	proceed  chan struct{}
}

func (s *idlePartStorage) Read(ctx context.Context, path string, min, max *int64) (io.ReadCloser, error) {
	if min != nil && *min == 0 && s.opened.CompareAndSwap(false, true) {
		return io.NopCloser(&slowCancelReader{ctx: ctx, canceled: s.canceled, proceed: s.proceed}), nil
	}
	return s.ObjectStorage.Read(ctx, path, min, max)
}

func TestJoinedWholeReadFallsBackAfterPartIdleTimeout(t *testing.T) {
	shortIdle(t, 50*time.Millisecond)
	for _, tc := range []struct {
		name  string
		parts int
	}{
		{name: "single_stream", parts: 1},
		{name: "split_download", parts: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: tc.parts, min: 1}, 256<<10)
			<-fs.diskCache.load.done
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			storage := &idlePartStorage{
				ObjectStorage: fs.server.raw,
				canceled:      make(chan struct{}),
				proceed:       make(chan struct{}),
			}
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.proceed) })
			prefetchDone := make(chan struct{})
			go func() {
				defer close(prefetchDone)
				_ = fs.PrefetchFile(ctx, "object")
			}()
			defer func() { cancel(); unblock(); <-prefetchDone }()
			select {
			case <-storage.canceled:
			case <-ctx.Done():
				t.Fatal("prefetch part did not hit its idle timeout")
			}
			download := fs.diskCache.download(fs.diskCache.objectKey("object"))
			require.NotNil(t, download)
			vector := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache, Entries: []IOEntry{{Offset: 0, Size: 16}}}
			readDone := make(chan struct{})
			var readErr error
			go func() {
				defer close(readDone)
				readErr = fs.Read(ctx, vector)
			}()
			defer func() { cancel(); unblock(); <-readDone; vector.Release() }()
			require.Eventually(t, func() bool {
				download.idle.mu.Lock()
				defer download.idle.mu.Unlock()
				return download.idle.waiting > 0
			}, time.Second, time.Millisecond, "foreground read should join before the stalled part returns")
			unblock()
			select {
			case <-readDone:
				require.NoError(t, readErr, "a late joiner should fall back to its exact bytes after a prefetch idle timeout")
				require.Equal(t, data[:16], vector.Entries[0].Data)
			case <-ctx.Done():
				t.Fatal("joined read failed to return")
			}
		})
	}
}

func TestJoinedColumnReadReleasesBufferOnDecoderPanic(t *testing.T) {
	for _, tc := range []struct {
		name   string
		panics bool
	}{
		{name: "decoder_returns_error"},
		{name: "decoder_panics", panics: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 4096)
			<-fs.diskCache.load.done
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			publishing := make(chan struct{})
			resume := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(resume) })
			defer unblock()
			fs.diskCache.hooks.beforeFilePublication = func() {
				close(publishing)
				<-resume
			}
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: 128, Ranges: []PrefetchRange{{Size: 128}},
			}))
			select {
			case <-publishing:
			case <-ctx.Done():
				t.Fatal("prefetch did not reach publication")
			}
			decoderError := errors.New("decoder rejected input")
			vector := &IOVector{
				FilePath: "object",
				Policy:   ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache,
				Entries: []IOEntry{{Offset: 8, Size: 8, ToCacheData: func(context.Context, io.Reader, []byte, CacheDataAllocator) (fscache.Data, error) {
					if tc.panics {
						panic(decoderError)
					}
					return nil, decoderError
				}}},
			}
			defer vector.Release()
			if tc.panics {
				require.PanicsWithValue(t, decoderError, func() { _ = fs.Read(ctx, vector) })
			} else {
				require.ErrorIs(t, fs.Read(ctx, vector), decoderError)
			}
			unblock()
			waitFillsIdle(t, fs.diskCache)
			fs.diskCache.prefetchObjects.reads.Wait()
			flights := &fs.diskCache.prefetchObjects
			flights.mu.Lock()
			charged := heldMemory(fs.diskCache)
			flights.mu.Unlock()
			require.Zero(t, charged, "a finished prefetch must return its buffer charge after the reading decoder exits")
		})
	}
}

func TestReadStartedSplitHonorsConfiguredPartCount(t *testing.T) {
	for _, tc := range []struct {
		name   string
		offset int64
	}{
		{name: "read_starts_at_object_start"},
		{name: "read_starts_at_interior_block", offset: 64 << 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 0, 256<<10)
			<-fs.diskCache.load.done
			fs.diskCache.objectSplit = (&CacheConfig{
				DiskPrefetchObjectSplit:    ptrTo(4),
				DiskPrefetchObjectSplitMin: ptrTo[toml.ByteSize](0),
			}).diskObjectSplit()
			fs.diskCache.storePrefetchedSize(fs.diskCache.objectKey("object"), int64(len(data)), fs.diskCache.deletes.now())
			vector := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: tc.offset, Size: 16}}}
			defer vector.Release()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.Read(ctx, vector))
			require.Equal(t, data[tc.offset:tc.offset+16], vector.Entries[0].Data)
			require.EqualValues(t, 4, storage.reads.Load(), "the known-size object fits four aligned parts, including a boundary at the read's first needed block")
		})
	}
}

func TestFileRangeReadUsesCachePublishedWhileAdmitted(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "read_after_publication"},
		{name: "read_admitted_before_publication", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const size = minDiskPrefetchMemoryMax + 64
			fs, storage, data := newPrefetchRangeFS(t, 0, size)
			fs.diskCache.memoryPrefetchMax = minDiskPrefetchMemoryMax
			<-fs.diskCache.load.done
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			creating := make(chan struct{})
			proceed := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(proceed) })
			defer unblock()
			fs.diskCache.hooks.beforeCacheTempFileCreate = sync.OnceFunc(func() {
				close(creating)
				<-proceed
			})
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: size, Ranges: []PrefetchRange{{Size: size}},
			}))
			select {
			case <-creating:
			case <-ctx.Done():
				t.Fatal("background range did not acquire the cache path")
			}
			if !tc.overlap {
				unblock()
				waitFillsIdle(t, fs.diskCache)
			}
			vector := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache, Entries: []IOEntry{{Size: size}}}
			readDone := make(chan struct{})
			var readErr error
			go func() {
				defer close(readDone)
				readErr = fs.Read(ctx, vector)
			}()
			defer func() { cancel(); unblock(); <-readDone; vector.Release() }()
			if tc.overlap {
				path := fs.diskCache.objectKey("object")
				objects := &fs.diskCache.prefetchObjects
				require.Eventually(t, func() bool {
					objects.mu.Lock()
					defer objects.mu.Unlock()
					for _, object := range objects.m[path] {
						file, ok := object.(*filePrefetchObject)
						if !ok {
							continue
						}
						file.mu.Lock()
						foregroundAttached := file.readStarted || file.holds > 1
						file.mu.Unlock()
						if foregroundAttached {
							return true
						}
					}
					return false
				}, 3*time.Second, time.Millisecond, "foreground read should reach range admission while the background range owns the path")
				unblock()
			}
			select {
			case <-readDone:
				require.NoError(t, readErr)
			case <-ctx.Done():
				t.Fatal("foreground read did not finish")
			}
			require.True(t, bytes.Equal(data, vector.Entries[0].Data))
			require.EqualValues(t, 1, storage.reads.Load(), "publication of the same range should avoid a second storage read")
		})
	}
}

func TestIndexPrefetchPreservesConcurrentDiskCacheFill(t *testing.T) {
	for _, tc := range []struct {
		name           string
		prefetchPolicy Policy
	}{
		{name: "prefetch_into_disk_cache"},
		{name: "prefetch_into_index_cache", prefetchPolicy: IndexFileObject},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			gated := newBlackBoxStorage(storage, func(_ string, offset int64) bool { return offset == 0 })
			fs.server.raw = gated
			unblock := sync.OnceFunc(func() { close(gated.gate) })
			prefetchDone := make(chan struct{})
			var prefetchErr error
			go func() {
				defer close(prefetchDone)
				prefetchErr = fs.PrefetchFile(WithFileServicePolicy(ctx, tc.prefetchPolicy), "object")
			}()
			defer func() { unblock(); <-prefetchDone }()
			require.Eventually(t, func() bool {
				active, _, _ := gated.counts("object")
				return active == 1
			}, time.Second, time.Millisecond, "whole prefetch should have opened storage")
			// The first exact read bypasses disk-cache reads but allows its writes.
			require.Equal(t, data[128:144], readIndexObjectRange(t, fs, ExactColumnsRead|SkipDiskCacheReads, 128, 16))
			unblock()
			select {
			case <-prefetchDone:
				require.NoError(t, prefetchErr)
			case <-ctx.Done():
				t.Fatal("whole prefetch did not finish")
			}
			_, indexed := fs.indexObjectCached("object")
			require.Equal(t, tc.prefetchPolicy == IndexFileObject, indexed)
			before := storage.reads.Load()
			require.Equal(t, data[128:144], readIndexObjectRange(t, fs, ExactColumnsRead, 128, 16))
			require.Equal(t, before, storage.reads.Load(), "the normal read should find the whole object or its exact range in the disk cache")
		})
	}
}

func TestDeleteInvalidatesPrefetchedRangeForNestedPath(t *testing.T) {
	for _, tc := range []struct {
		name string
		path string
	}{
		{name: "baseline_flat_object_path", path: "flat-object"},
		{name: "nested_object_path", path: "prefix/nested-object"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const size = 4 << 10
			fs, storage, _ := newPrefetchRangeFS(t, 0, size)
			<-fs.diskCache.load.done
			ctx := context.Background()
			write := func(value byte) {
				t.Helper()
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: tc.path, Policy: SkipAllCache,
					Entries: []IOEntry{{Size: size, Data: bytes.Repeat([]byte{value}, size)}}}))
			}
			read := func() byte {
				t.Helper()
				vec := &IOVector{FilePath: tc.path, Policy: ExactColumnsRead | SkipMemoryCache | SkipDiskCacheWrites,
					Entries: []IOEntry{{Size: size}}}
				defer vec.Release()
				require.NoError(t, fs.Read(ctx, vec))
				require.Len(t, vec.Entries[0].Data, size)
				return vec.Entries[0].Data[0]
			}

			write(1)
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: tc.path, Size: size, Ranges: []PrefetchRange{{Size: size}},
			}))
			require.Eventually(t, func() bool { return !fs.diskCache.fillsUnderWay(tc.path) }, time.Second, time.Millisecond)
			require.EqualValues(t, 1, storage.reads.Load(), "the prefetch should read the former object once")
			require.Equal(t, byte(1), read())
			require.EqualValues(t, 1, storage.reads.Load(), "the former data should be read from the prefetched cache range")

			require.NoError(t, fs.Delete(ctx, tc.path))
			write(2)
			require.Equal(t, byte(2), read(), "a read after delete and recreate should return the new object")
		})
	}
}

func TestAsyncWholeFillDoesNotPublishAfterDeleteMark(t *testing.T) {
	for _, tc := range []struct {
		name   string
		delete bool
	}{
		{name: "baseline_no_delete"},
		{name: "delete_marks_object_before_publication", delete: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
			d := fs.diskCache
			<-d.load.done
			fs.SetAsyncUpdate(true)
			reached, release := make(chan struct{}), make(chan struct{})
			allowPublish := sync.OnceFunc(func() { close(release) })
			defer allowPublish()
			d.hooks.beforeFilePublication = func() {
				close(reached)
				<-release
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 64 << 10}))
			select {
			case <-reached:
			case <-ctx.Done():
				t.Fatal("whole-object fill did not reach the publication gate", ctx.Err())
			}

			diskPath := d.objectKey("object")
			if tc.delete {
				mark := d.deletes.now()
				deleteCtx, cancelDelete := context.WithCancel(ctx)
				defer cancelDelete()
				deleted := make(chan error, 1)
				go func() { deleted <- fs.Delete(deleteCtx, "object") }()
				require.Eventually(t, func() bool {
					return d.deletes.since(diskPath, mark)
				}, time.Second, time.Millisecond, "the source delete completed and the cache delete was counted")
				_, err := fs.StatFile(ctx, "object")
				require.Error(t, err, "object storage no longer holds the object")
				require.NoError(t, <-deleted, "the delete waits for no fill whose publication it prevents")
			}
			allowPublish()
			d.Flush(ctx)
			require.NoError(t, ctx.Err())
			if !tc.delete {
				require.FileExists(t, d.wholePath("object", 64<<10), "the staged whole object normally publishes")
				return
			}
			require.NoFileExists(t, d.wholePath("object", 64<<10), "the old whole object must not appear once its delete has been counted")
		})
	}
}

func TestSyncWholeFillDoesNotPublishAfterDeleteMark(t *testing.T) {
	for _, tc := range []struct {
		name   string
		delete bool
	}{
		{name: "baseline_no_delete"},
		{name: "delete_during_publication", delete: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
			d := fs.diskCache
			<-d.load.done
			reached, release := make(chan struct{}), make(chan struct{})
			allowPublish := sync.OnceFunc(func() { close(release) })
			defer allowPublish()
			d.hooks.afterPublicationClaim = func() {
				close(reached)
				<-release
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 64 << 10}))
			select {
			case <-reached:
			case <-ctx.Done():
				t.Fatal("whole-object fill did not reach the publication claim", ctx.Err())
			}

			diskPath := d.objectKey("object")
			deleted := make(chan error, 1)
			if tc.delete {
				mark := d.deletes.now()
				deleteCtx, cancelDelete := context.WithCancel(ctx)
				defer cancelDelete()
				go func() { deleted <- fs.Delete(deleteCtx, "object") }()
				counted := false
				for i := 0; i < 100 && !counted; i++ {
					time.Sleep(time.Millisecond)
					counted = d.deletes.since(diskPath, mark)
				}
				if counted {
					cancelDelete()
					require.ErrorIs(t, <-deleted, context.Canceled)
					deleted <- nil
				}
			}
			allowPublish()
			if tc.delete {
				require.NoError(t, <-deleted)
			}
			require.Eventually(t, func() bool { return !fs.diskCache.fillsUnderWay("object") }, time.Second, time.Millisecond)
			if !tc.delete {
				require.FileExists(t, d.wholePath("object", 64<<10))
				return
			}
			require.NoFileExists(t, d.wholePath("object", 64<<10), "a delete counted during the publication of the whole object removes it")
		})
	}
}

func TestDeleteInvalidatesCachedEntryForNestedPath(t *testing.T) {
	for _, path := range []string{"flat-object", "prefix/nested-object"} {
		t.Run(path, func(t *testing.T) {
			const size = 4 << 10
			fs, storage, _ := newPrefetchRangeFS(t, 0, size)
			<-fs.diskCache.load.done
			ctx := context.Background()
			write := func(value byte) {
				t.Helper()
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: path, Policy: SkipAllCache,
					Entries: []IOEntry{{Size: size, Data: bytes.Repeat([]byte{value}, size)}}}))
			}
			read := func() byte {
				t.Helper()
				vec := &IOVector{FilePath: path, Policy: SkipMemoryCache | SkipFullFilePreloads,
					Entries: []IOEntry{{Offset: 16, Size: 64}}}
				defer vec.Release()
				require.NoError(t, fs.Read(ctx, vec))
				return vec.Entries[0].Data[0]
			}

			write(1)
			require.Equal(t, byte(1), read())
			reads := storage.reads.Load()
			require.Equal(t, byte(1), read())
			require.Equal(t, reads, storage.reads.Load(), "the entry is read from the disk cache")

			require.NoError(t, fs.Delete(ctx, path))
			write(2)
			require.Equal(t, byte(2), read())
		})
	}
}

func TestPrefetchBufferReleasedBeforeDecode(t *testing.T) {
	for _, blockDecode := range []bool{false, true} {
		name := "completed_decode"
		if blockDecode {
			name = "blocked_decode_after_copy"
		}
		t.Run(name, func(t *testing.T) {
			const budget = 64 << 10
			const size = 4 << 10
			fs, _, data := newPrefetchRangeFS(t, 0, 64<<10)
			d := fs.diskCache
			<-d.load.done
			d.server.memoryLimit = 2 * budget
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "bystander", Policy: SkipAllCache,
				Entries: []IOEntry{{Size: size, Data: data[:size]}}}))
			storage := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return key == fs.pathToKey("object") })
			fs.server.raw = storage
			resumeStorage := sync.OnceFunc(func() { close(storage.gate) })
			defer resumeStorage()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: size,
				Ranges: []PrefetchRange{{Size: size}}}))
			require.Eventually(t, func() bool {
				cur, _, _ := storage.counts(fs.pathToKey("object"))
				return cur == 1
			}, time.Second, time.Millisecond)

			decoding := make(chan struct{})
			resumeDecode := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(resumeDecode) })
			if !blockDecode {
				unblock()
			}
			vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache,
				Entries: []IOEntry{{Size: size, ToCacheData: func(ctx context.Context, r io.Reader, data []byte, alloc CacheDataAllocator) (fscache.Data, error) {
					close(decoding)
					<-resumeDecode
					return CacheOriginalData(ctx, r, data, alloc)
				}}}}
			defer vec.Release()
			done := make(chan struct{})
			var readErr error
			go func() {
				defer close(done)
				readErr = fs.Read(ctx, vec)
			}()
			defer func() {
				unblock()
				resumeStorage()
				<-done
				require.NoError(t, readErr)
			}()
			require.Eventually(t, func() bool {
				return memoryWaiters(fs, d.objectKey("object")) == 1
			}, time.Second, time.Millisecond, "the read should be joined to the prefetch")
			resumeStorage()
			select {
			case <-decoding:
			case <-ctx.Done():
				t.Fatal("the reader did not reach the decoder")
			}
			waitPrefetcherIdle(t, fs)
			d.prefetchObjects.reads.Wait()
			if !blockDecode {
				<-done
			}

			// The rest of the budget is held, so the bystander gets a buffer only once the
			// prefetch released its own.
			holdForegroundBuffer(t, fs, budget)
			other := &IOVector{FilePath: "bystander", Policy: ExactColumnsRead | SkipMemoryCache | SkipDiskCacheWrites,
				Entries: []IOEntry{{Size: size}}}
			defer other.Release()
			otherDone := make(chan struct{})
			var otherErr error
			go func() {
				defer close(otherDone)
				otherErr = fs.Read(ctx, other)
			}()
			defer func() {
				unblock()
				<-otherDone
				require.NoError(t, otherErr)
				require.Equal(t, data[:size], other.Entries[0].Data)
			}()
			var charged int64
			require.Eventually(t, func() bool {
				select {
				case <-otherDone:
					return true
				default:
				}
				f := &d.prefetchObjects
				f.mu.Lock()
				defer f.mu.Unlock()
				for o := range f.waiting {
					if target := o.target(); target.filePath == "bystander" && target.state == objectWaitingRoom {
						charged = heldMemory(d)
						return true
					}
				}
				return false
			}, time.Second, time.Millisecond)
			require.LessOrEqual(t, charged, int64(budget), "the bystander should finish while the decoder runs on its own copy")
		})
	}
}

func TestIndexPrefetchWithDirectDiskDownload(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "after_direct_disk_download"},
		{name: "during_direct_disk_download", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				fs, storage, data := newIndexObjectFS(t, true)
				publishing := make(chan struct{})
				publish := make(chan struct{})
				release := sync.OnceFunc(func() { close(publish) })
				defer release()
				fs.diskCache.hooks.beforeFilePublication = sync.OnceFunc(func() {
					close(publishing)
					<-publish
				})

				diskPrefetch := make(chan error, 1)
				go func() { diskPrefetch <- fs.PrefetchFile(ctx, "object") }()
				<-publishing
				if !tc.overlap {
					release()
					require.NoError(t, <-diskPrefetch)
				}

				require.NoError(t, fs.PrefetchObject(WithFileServicePolicy(ctx, IndexFileObject), ObjectPrefetch{
					FilePath: "object", Size: int64(len(data)),
				}))
				if tc.overlap {
					// Let the index prefetch reach its wait or complete before publishing
					// the other tier. A worker that defers the fill can then proceed.
					synctest.Wait()
					release()
					require.NoError(t, <-diskPrefetch)
				}
				waitPrefetcherIdle(t, fs)

				require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))))
				require.True(t, fs.indexObjectHeld("object"),
					"index prefetch must warm its own tier after the direct disk fill (backend reads: %d)", storage.reads.Load())
				require.EqualValues(t, 2, storage.reads.Load())
			})
		})
	}
}

func TestIndexRangePrefetchHonorsCacheWritePolicy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		ranged bool
		policy Policy
	}{
		{name: "whole_prefetch_skips_all_cache_writes", policy: SkipCacheWrites},
		{name: "whole_prefetch_with_default_writes"},
		{name: "range_prefetch_skips_all_cache_writes", ranged: true, policy: SkipCacheWrites},
		{name: "range_prefetch_with_default_writes", ranged: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			ctx := WithFileServicePolicy(context.Background(), IndexFileObject|tc.policy)
			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
			if tc.ranged {
				req.Size = 4096
				req.Ranges = []PrefetchRange{{Offset: 64 << 10, Size: req.Size}}
			}
			require.NoError(t, fs.PrefetchObject(ctx, req))
			waitPrefetcherIdle(t, fs)
			require.Zero(t, fs.diskCache.cache.Used(),
				"index-object bytes must stay out of the table-data tier (backend reads: %d)", storage.reads.Load())
			if tc.policy.Any(SkipCacheWrites) {
				require.False(t, fs.indexObjectHeld("object"), "skipping all cache writes must leave both tiers empty")
			} else if !tc.ranged {
				require.True(t, fs.indexObjectHeld("object"), "a whole prefetch with writes enabled warms the index tier")
			}
		})
	}
}

func TestIndexObjectIdleTTLStartsAfterReadReleasesHold(t *testing.T) {
	for _, tc := range []struct {
		name     string
		readTime time.Duration
	}{
		{name: "read_finishes_before_idle_ttl"},
		{name: "read_outlasts_idle_ttl", readTime: 3 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				fs, storage, data := newIndexObjectFS(t, true)
				cacheIndexObject(t, fs, "object", data)
				cache := fs.indexFileCache
				cache.idleTTL = 2 * time.Second
				ctx := context.Background()
				var received bytes.Buffer
				vec := &IOVector{
					FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
					Entries: []IOEntry{{
						Offset: 100, Size: 64, Data: make([]byte, 64),
						WriterForRead: readShareWriterFunc(func(data []byte) (int, error) {
							// Advance virtual time while the cached entry is held by the read.
							time.Sleep(tc.readTime)
							return received.Write(data)
						}),
					}},
				}
				defer vec.Release()
				require.NoError(t, fs.Read(ctx, vec))
				require.Equal(t, data[100:164], received.Bytes())
				require.Zero(t, storage.reads.Load())
				removed, err := cache.collectIdle(ctx, time.Now())
				require.NoError(t, err)
				require.Zero(t, removed, "an index object just released by a successful read is not idle")
			})
		})
	}
}

func TestExactReadDoesNotWidenRange(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
	}{
		{name: "exact_columns_read", policy: ExactColumnsRead},
		{name: "exact_read_on_its_own", policy: ExactRead},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, false)
			var offset atomic.Int64
			tracked := newBlackBoxStorage(storage, func(_ string, from int64) bool {
				offset.Store(from)
				return false
			})
			fs.server.raw = tracked
			require.Equal(t, data[100:164], readIndexObjectRange(t, fs, tc.policy, 100, 64))
			require.EqualValues(t, 1, storage.reads.Load())
			require.EqualValues(t, 100, offset.Load(), "an exact read opens storage at the requested offset")
		})
	}
}

func TestIndexReadKeepsDownloadTierAcrossEntries(t *testing.T) {
	for _, tc := range []struct {
		name              string
		startDiskPrefetch bool
	}{
		{name: "index_publication_only"},
		{name: "table_download_replaces_index_download", startDiskPrefetch: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				fs, _, data := newIndexObjectFS(t, true)
				indexCtx := WithFileServicePolicy(ctx, IndexFileObject)

				indexPublishing, allowIndexPublish := make(chan struct{}), make(chan struct{})
				releaseIndex := sync.OnceFunc(func() { close(allowIndexPublish) })
				defer releaseIndex()
				fs.diskCache.hooks.beforeFilePublication = sync.OnceFunc(func() {
					close(indexPublishing)
					<-allowIndexPublish
				})
				diskFailure := errors.New("table download source failed")
				probe := &probeStorage{
					ObjectStorage: fs.server.raw, at: 0, gate: make(chan struct{}),
					answer: func(r io.ReadCloser) (io.ReadCloser, error) {
						_ = r.Close()
						return nil, diskFailure
					},
				}
				probe.hit.Store(true) // Leave the initial index download unchanged.
				fs.server.raw = probe
				releaseDisk := sync.OnceFunc(func() { close(probe.gate) })
				defer releaseDisk()

				indexPrefetch := make(chan error, 1)
				go func() { indexPrefetch <- fs.PrefetchFile(indexCtx, "object") }()
				<-indexPublishing

				firstEntry, resumeRead := make(chan struct{}), make(chan struct{})
				releaseRead := sync.OnceFunc(func() { close(resumeRead) })
				defer releaseRead()
				var received bytes.Buffer
				vec := &IOVector{
					FilePath: "object", Policy: IndexFileObject | SkipDiskCacheReads | SkipMemoryCache,
					Entries: []IOEntry{
						{Offset: 0, Size: 64, Data: make([]byte, 64), WriterForRead: readShareWriterFunc(func(b []byte) (int, error) {
							close(firstEntry)
							<-resumeRead
							return received.Write(b)
						})},
						{Offset: 64, Size: 64, Data: make([]byte, 64)},
					},
				}
				result := make(chan error, 1)
				go func() { result <- fs.Read(ctx, vec) }()
				<-firstEntry
				releaseIndex()
				require.NoError(t, <-indexPrefetch)
				require.True(t, fs.indexObjectHeld("object"))

				diskPrefetch := make(chan error, 1)
				if tc.startDiskPrefetch {
					probe.hit.Store(false)
					go func() { diskPrefetch <- fs.PrefetchFile(ctx, "object") }()
					synctest.Wait()
					require.True(t, probe.hit.Load(), "the replacement table-data download reached storage")
				}
				releaseRead()
				synctest.Wait()
				if tc.startDiskPrefetch {
					releaseDisk()
					require.ErrorIs(t, <-diskPrefetch, diskFailure)
				}
				readErr := <-result
				defer vec.Release()
				require.NoError(t, readErr, "a read skipping the table-data tier should use the published index entry")
				require.Equal(t, data[:64], received.Bytes())
				require.Equal(t, data[64:128], vec.Entries[1].Data)
			})
		})
	}
}

func TestNegativeConcurrencyIsRejected(t *testing.T) {
	noProbe := func(context.Context, string) (string, error) { return "", nil }
	for _, tc := range []struct {
		concurrency int64
		ok          bool
	}{{0, true}, {1, true}, {minRequestRoom - 1, true}, {minRequestRoom, true}, {1024, true}, {-1, false}} {
		args := ObjectStorageArguments{Endpoint: "disk", Bucket: t.TempDir(), Concurrency: tc.concurrency}
		err := args.validateWithRegionProbe(noProbe)
		if tc.ok {
			require.NoError(t, err, "concurrency=%d", tc.concurrency)
		} else {
			require.ErrorContains(t, err, fmt.Sprintf("concurrency must not be negative, got %d", tc.concurrency))
		}
		fs, err := NewS3FS(context.Background(), ObjectStorageArguments{Name: "s3", Endpoint: "disk", Bucket: t.TempDir(), Concurrency: tc.concurrency},
			CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](0)}, nil, false, true)
		if tc.ok {
			require.NoError(t, err, "concurrency=%d", tc.concurrency)
			fs.Close(context.Background())
		} else {
			require.ErrorContains(t, err, "concurrency must not be negative")
		}
	}
}

func TestDownloadConnsAreHalfTheStorageConcurrency(t *testing.T) {
	for _, tc := range []struct {
		conns       int
		concurrency int64
		want        int
	}{
		{0, 0, maxConnsPerHost / 2},
		{1, 0, 32},
		{0, 1, 32},
		{2, 3, 32},
		{64, 0, 32},
		{256, 0, 128},
		{0, 64, 32},
		{64, 64, 32},
		{256, 100, 50},
		{64, 1 << 20, 32},
		{4096, 0, 512},
	} {
		ctx := context.Background()
		fs, err := NewS3FS(ctx, ObjectStorageArguments{
			Name: "s3", Endpoint: "disk", Bucket: t.TempDir(),
			Concurrency: tc.concurrency, MaxConnsPerHost: tc.conns,
		}, CacheConfig{DiskPath: ptrTo(t.TempDir()), DiskCapacity: ptrTo[toml.ByteSize](64 << 20), MemoryCapacity: ptrTo[toml.ByteSize](0)}, nil, false, false)
		require.NoError(t, err)
		require.Equal(t, tc.want, fs.server.backgroundMax, "conns %d concurrency %d", tc.conns, tc.concurrency)
		require.Equal(t, 2*tc.want, fs.server.capacity, "conns %d concurrency %d", tc.conns, tc.concurrency)
		fs.Close(ctx)
	}
}

func newStorageSlotFS(t *testing.T, concurrency int64, conns int, paths []string, data []byte, stalled func(string) bool) (*S3FS, *blackBoxStorage) {
	ctx := context.Background()
	config := indexCacheConfig(t, 64<<20, 64<<20)
	config.IndexFileCacheEnabled = false
	config.MemoryCapacity = DisabledCacheConfig.MemoryCapacity
	fs, err := NewS3FS(ctx, ObjectStorageArguments{
		Name: "s3", Endpoint: "disk", Bucket: t.TempDir(),
		Concurrency: concurrency, MaxConnsPerHost: conns,
	}, config, nil, false, false)
	require.NoError(t, err)
	<-fs.diskCache.load.done
	for _, path := range paths {
		require.NoError(t, fs.Write(ctx, IOVector{
			FilePath: path, Policy: SkipAllCache,
			Entries: []IOEntry{{Size: int64(len(data)), Data: data}},
		}))
	}
	// Stall prefetch opens after the server admits them.
	gated := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return stalled(key) })
	fs.server.raw = gated
	return fs, gated
}

func requireForegroundReads(t *testing.T, fs *S3FS, gated *blackBoxStorage, data []byte, policies ...Policy) {
	for _, policy := range policies {
		readCtx, cancel := context.WithTimeout(context.Background(), time.Second)
		vec := &IOVector{FilePath: "foreground", Policy: policy, Entries: []IOEntry{{Offset: 64, Size: 16}}}
		readErr := fs.Read(readCtx, vec)
		_, _, opens := gated.counts("foreground")
		require.NoError(t, readErr, "policy %v, foreground opens: %d", policy, opens)
		require.Equal(t, data[64:80], vec.Entries[0].Data)
		vec.Release()
		cancel()
	}
}

// Two stalled whole-object prefetches leave storage slots and download connections for
// foreground reads, cache-filling ones included.
func TestPrefetchLeavesStorageSlotForForegroundRead(t *testing.T) {
	for _, tc := range []struct {
		name        string
		concurrency int64
	}{
		{name: "default_storage_concurrency"},
		{name: "storage_concurrency_64", concurrency: minRequestRoom},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				data := bytes.Repeat([]byte("data"), 64<<10)
				fs, gated := newStorageSlotFS(t, tc.concurrency, 64, []string{"prefetch-a", "prefetch-b", "foreground"}, data,
					func(key string) bool { return key == "prefetch-a" || key == "prefetch-b" })
				defer fs.Close(ctx)
				defer close(gated.gate)
				for _, path := range []string{"prefetch-a", "prefetch-b"} {
					require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: path, Size: int64(len(data))}))
				}
				synctest.Wait()
				first, _, _ := gated.counts("prefetch-a")
				second, _, _ := gated.counts("prefetch-b")
				require.Positive(t, first+second, "at least one prefetch must still be in flight")
				requireForegroundReads(t, fs, gated, data, SkipAllCache|ExactColumnsRead, SkipMemoryCache)
			})
		})
	}
}

// Stalled column prefetches holding every connection downloads may hold leave storage
// slots for a foreground read when the storage concurrency is below the connections per host.
func TestStalledDownloadsLeaveStorageConcurrencyForForegroundRead(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx := context.Background()
		data := bytes.Repeat([]byte("data"), 64<<10)
		paths := make([]string, 0, 33)
		paths = append(paths, "foreground")
		for i := range 32 {
			paths = append(paths, fmt.Sprintf("prefetch-%d", i))
		}
		fs, gated := newStorageSlotFS(t, minRequestRoom, 0, paths, data,
			func(key string) bool { return strings.HasPrefix(key, "prefetch-") })
		defer fs.Close(ctx)
		defer close(gated.gate)
		for _, path := range paths[1:] {
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: path, Size: int64(len(data)), Ranges: columnRanges(8, 20<<10, 1<<10), Scan: &PrefetchScan{},
			}))
		}
		synctest.Wait()
		inFlight := 0
		for _, path := range paths[1:] {
			cur, _, _ := gated.counts(path)
			inFlight += cur
		}
		require.EqualValues(t, fs.server.backgroundMax, inFlight, "stalled downloads hold every background room")
		requireForegroundReads(t, fs, gated, data, SkipAllCache|ExactColumnsRead)
	})
}

func waitReviewPrefetchIdle(t *testing.T, fs *S3FS) {
	t.Helper()
	waitFillsIdle(t, fs.diskCache)
}

func TestPrefetchDiskTierAfterIndex(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "after_index_download"},
		{name: "during_index_download", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			fs, _, data := newIndexObjectFS(t, true)
			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
			fs.server.raw = storage
			resume := sync.OnceFunc(func() { close(storage.gate) })
			defer resume()

			require.NoError(t, fs.PrefetchObject(WithFileServicePolicy(ctx, IndexFileObject), req))
			require.Eventually(t, func() bool {
				current, _, _ := storage.counts(fs.pathToKey(req.FilePath))
				return current == 1
			}, 5*time.Second, time.Millisecond)
			if !tc.overlap {
				resume()
				waitReviewPrefetchIdle(t, fs)
			}
			require.NoError(t, fs.PrefetchObject(ctx, req))
			resume()
			waitReviewPrefetchIdle(t, fs)

			require.True(t, fs.indexObjectHeld(req.FilePath), "the index prefetch should fill the index-file cache")
			_, _, reads := storage.counts(fs.pathToKey(req.FilePath))
			require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath(req.FilePath, int64(len(data)))),
				"the disk prefetch should fill the disk cache (storage opens: %d)", reads)
			require.Equal(t, 2, reads)
		})
	}
}

func TestPrefetchBothQueuedTiers(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "separate_queue_entries"},
		{name: "same_object_queued_in_both_tiers", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newIndexObjectFS(t, true)
			resume := busyFills(t, fs)

			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
			indexReq := req
			if !tc.overlap {
				indexReq.FilePath = "another-object"
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: indexReq.FilePath, Policy: SkipAllCache,
					Entries: []IOEntry{{Size: req.Size, Data: data}}}))
			}
			require.NoError(t, fs.PrefetchObject(ctx, req))
			require.True(t, fs.diskCache.isQueued(req.FilePath))
			require.NoError(t, fs.PrefetchObject(WithFileServicePolicy(ctx, IndexFileObject), indexReq))
			require.True(t, fs.diskCache.isQueued(indexReq.FilePath))
			resume()
			waitReviewPrefetchIdle(t, fs)

			require.True(t, fs.indexObjectHeld(indexReq.FilePath), "the index prefetch should fill the index-file cache")
			require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath(req.FilePath, int64(len(data)))),
				"the disk prefetch should fill the disk cache (storage opens: %d)", storage.reads.Load())
			require.EqualValues(t, 2, storage.reads.Load())
		})
	}
}

func TestIndexPrefetchAfterAsyncDiskDownload(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "after_disk_publication"},
		{name: "while_disk_publication_is_pending", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				fs, _, data := newIndexObjectFS(t, true)
				fs.SetAsyncUpdate(true)
				storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
				fs.server.raw = storage
				resumeStorage := sync.OnceFunc(func() { close(storage.gate) })
				defer resumeStorage()
				publicationEntered, publish := make(chan struct{}), make(chan struct{})
				resumePublication := sync.OnceFunc(func() { close(publish) })
				defer resumePublication()
				fs.diskCache.hooks.beforeFilePublication = sync.OnceFunc(func() {
					close(publicationEntered)
					<-publish
				})

				req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
				indexCtx := WithFileServicePolicy(ctx, IndexFileObject)
				require.NoError(t, fs.PrefetchObject(ctx, req))
				require.Eventually(t, func() bool {
					current, _, _ := storage.counts(fs.pathToKey(req.FilePath))
					return current == 1
				}, 5*time.Second, time.Millisecond)
				if tc.overlap {
					require.NoError(t, fs.PrefetchObject(indexCtx, req))
				}
				resumeStorage()
				select {
				case <-publicationEntered:
				case <-time.After(5 * time.Second):
					t.Fatal("the disk cache did not reach publication")
				}
				// Let the deferred prefetch reach its wait or complete before resuming publication.
				synctest.Wait()
				resumePublication()
				waitReviewPrefetchIdle(t, fs)
				fs.diskCache.Flush(ctx)
				require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath(req.FilePath, int64(len(data)))))
				if !tc.overlap {
					require.NoError(t, fs.PrefetchObject(indexCtx, req))
					waitReviewPrefetchIdle(t, fs)
					fs.diskCache.Flush(ctx)
				}

				_, _, reads := storage.counts(fs.pathToKey(req.FilePath))
				require.True(t, fs.indexObjectHeld(req.FilePath),
					"the index prefetch should survive the disk cache's asynchronous publication (storage opens: %d)", reads)
				require.Equal(t, 2, reads)
			})
		})
	}
}

func TestQueuedRangeMergeChargesOnlyUniqueBytes(t *testing.T) {
	for _, tc := range []struct {
		name         string
		secondOffset int64
	}{
		{name: "adjacent_requests", secondOffset: 128},
		{name: "overlapping_requests", secondOffset: 64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const total = 192
			ctx := context.Background()
			fs, storage, _ := newIndexObjectFS(t, false)
			resume := busyFills(t, fs)

			for _, r := range []PrefetchRange{{Size: 128}, {Offset: tc.secondOffset, Size: total - tc.secondOffset}} {
				require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}))
			}
			resume()
			waitReviewPrefetchIdle(t, fs)

			d := fs.diskCache
			key := d.objectKey("object")
			_, prefixCached := d.ranges.covering(key, 0, 128)
			require.True(t, prefixCached)
			_, unionCached := d.ranges.covering(key, 0, total)
			require.True(t, unionCached, "all %d unique bytes are cached; cached ranges: %+v", total, d.ranges.of(key))
			require.EqualValues(t, 2, storage.reads.Load(), "the bytes the first fill does not hold get a fill of their own")
		})
	}
}

func TestBatchDeleteInvalidatesOtherIndexObjects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		readOnly bool
	}{
		{name: "both_index_entries_writable"},
		{name: "first_index_shard_read_only", readOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnly && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			fs, _, data := newIndexObjectFS(t, true)
			const first, second = "first-object", "object"
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: first, Policy: SkipAllCache,
				Entries: []IOEntry{{Size: int64(len(data)), Data: data}}}))
			cacheIndexObject(t, fs, first, data)
			cacheIndexObject(t, fs, second, data)
			firstDir, firstCached := fs.indexObjectCached(first)
			secondDir, secondCached := fs.indexObjectCached(second)
			require.True(t, firstCached && secondCached)
			firstShard := filepath.Dir(firstDir)
			require.NotEqual(t, firstShard, filepath.Dir(secondDir))
			if tc.readOnly {
				require.NoError(t, os.Chmod(firstShard, 0o500))
				defer func() { require.NoError(t, os.Chmod(firstShard, 0o755)) }()
			}

			err := fs.Delete(ctx, first, second)
			if tc.readOnly {
				require.ErrorIs(t, err, os.ErrPermission)
			} else {
				require.NoError(t, err)
			}
			_, err = fs.StatFile(ctx, second)
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "the second object is deleted from storage: %v", err)
			vec := &IOVector{FilePath: second, Policy: IndexFileObject | ExactColumnsRead,
				Entries: []IOEntry{{Size: 16}}}
			defer vec.Release()
			err = fs.Read(ctx, vec)
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound),
				"the second object's index entry should be invalidated independently: read returned %d bytes, error %v", len(vec.Entries[0].Data), err)
		})
	}
}

func TestBatchDeleteRemovesOtherDiskCacheObjects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		readOnly bool
	}{
		{name: "both_shards_writable"},
		{name: "first_shard_read_only", readOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnly && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			fs, _, data := newPrefetchRangeFS(t, 4096, 300<<10)
			const first, second = "first-object", "object"
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: first, Policy: SkipDiskCache | SkipMemoryCache,
				Entries: []IOEntry{{Size: int64(len(data)), Data: data}}}))
			read := func(file string, policy Policy, offset int64) error {
				vec := &IOVector{FilePath: file, Policy: policy | SkipMemoryCache,
					Entries: []IOEntry{{Offset: offset, Size: 10}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				fs.diskCache.prefetchObjects.reads.Wait()
				return err
			}
			for _, file := range []string{first, second} {
				require.NoError(t, read(file, SkipFullFilePreloads, 100))
				require.NoError(t, read(file, 0, 200<<10))
			}
			firstPath, secondPath := fs.diskCache.wholePath(first, int64(len(data))), fs.diskCache.wholePath(second, int64(len(data)))
			require.True(t, fs.diskCache.cache.Contains(firstPath) && fs.diskCache.cache.Contains(secondPath))
			require.NotEmpty(t, fs.diskCache.ranges.of(fs.diskCache.objectKey(second)))
			firstShard := filepath.Dir(firstPath)
			require.NotEqual(t, firstShard, filepath.Dir(secondPath))
			if tc.readOnly {
				require.NoError(t, os.Chmod(firstShard, 0o500))
				defer func() { require.NoError(t, os.Chmod(firstShard, 0o755)) }()
			}

			err := fs.Delete(ctx, first, second)
			if tc.readOnly {
				require.ErrorIs(t, err, os.ErrPermission)
			} else {
				require.NoError(t, err)
			}
			for _, policy := range []Policy{0, SkipFullFilePreloads} {
				err = read(second, policy, 100)
				require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "policy %v: %v", policy, err)
			}
			require.False(t, fs.diskCache.cache.Contains(secondPath), "the second object's whole-object entry is removed")
			require.Empty(t, fs.diskCache.ranges.of(fs.diskCache.objectKey(second)), "the second object's range files are removed")
		})
	}
}

func TestLocalFSBatchDeleteInvalidatesDeletedFiles(t *testing.T) {
	for _, tc := range []struct {
		name     string
		readOnly bool
	}{
		{name: "both_directories_writable"},
		{name: "second_directory_read_only", readOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnly && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			root := t.TempDir()
			fs, err := NewLocalFS(ctx, "test", root, CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](1 << 20)}, nil)
			require.NoError(t, err)
			t.Cleanup(func() { fs.Close(ctx) })
			const first, second = "a/first", "b/second"
			read := func(file string) ([]byte, error) {
				vec := &IOVector{FilePath: file, Entries: []IOEntry{{Size: 3, ToCacheData: CacheOriginalData}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				return append([]byte(nil), vec.Entries[0].Data...), err
			}
			for _, file := range []string{first, second} {
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: file, Entries: []IOEntry{{Size: 3, Data: []byte("abc")}}}))
				data, err := read(file)
				require.NoError(t, err)
				require.Equal(t, "abc", string(data))
			}
			if tc.readOnly {
				dir := filepath.Join(root, "b")
				require.NoError(t, os.Chmod(dir, 0o500))
				defer func() { require.NoError(t, os.Chmod(dir, 0o755)) }()
			}

			err = fs.Delete(ctx, first, second)
			if tc.readOnly {
				require.ErrorIs(t, err, os.ErrPermission)
			} else {
				require.NoError(t, err)
			}
			data, err := read(first)
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "the deleted first file is not read from the cache: read %q, error %v", data, err)
		})
	}
}

func TestCachingReadDoesNotJoinUncachedFlight(t *testing.T) {
	for _, tc := range []struct {
		name       string
		skipLookup bool
	}{
		{name: "baseline_direct_caching_read", skipLookup: true},
		{name: "caching_read_with_shared_lookup"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const offset, size = 100, 8
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, storage, data := newPrefetchRangeFS(t, 0, 1024)
			<-fs.diskCache.load.done
			failedOpen := &failOnceStorage{ObjectStorage: storage, fail: offset, gate: make(chan struct{})}
			fs.server.raw = failedOpen
			release := sync.OnceFunc(func() { close(failedOpen.gate) })
			defer release()
			type result struct {
				data string
				err  error
			}
			start := func(policy Policy) <-chan result {
				done := make(chan result, 1)
				go func() {
					vector := &IOVector{FilePath: "object", Policy: policy | ExactColumnsRead | SkipMemoryCache,
						Entries: []IOEntry{{Offset: offset, Size: size}}}
					defer vector.Release()
					err := fs.Read(ctx, vector)
					done <- result{data: string(vector.Entries[0].Data), err: err}
				}()
				return done
			}

			path := fs.diskCache.objectKey("object")
			first := start(SkipDiskCacheWrites)
			require.Eventually(t, func() bool { return failedOpen.failed.Load() && memoryWaiters(fs, path) == 1 },
				time.Second, time.Millisecond, "the uncached flight must be waiting for its first storage open")

			var policy Policy
			if tc.skipLookup {
				policy = SkipDiskCacheReads
			}
			second := start(policy)
			require.Eventually(t, func() bool { return len(second) == 1 || memoryWaiters(fs, path) == 2 },
				time.Second, time.Millisecond, "the second read must either complete or join the stalled flight")
			release()
			require.ErrorContains(t, (<-first).err, "probe: storage failed")
			got := <-second
			require.NoError(t, got.err, "the caching read should open storage independently of the non-caching flight")
			require.Equal(t, string(data[offset:offset+size]), got.data)
			require.EqualValues(t, 1, storage.reads.Load(), "the independent storage open succeeds")
		})
	}
}

func TestAsyncDiskPrefetchSurvivesIndexDownload(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "baseline_index_published_first"},
		{name: "disk_prefetch_during_index_download", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				fs, _, data := newIndexObjectFS(t, true)
				fs.SetAsyncUpdate(true)
				storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
				fs.server.raw = storage
				resume := sync.OnceFunc(func() { close(storage.gate) })
				defer resume()

				req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
				indexCtx := WithFileServicePolicy(ctx, IndexFileObject)
				require.NoError(t, fs.PrefetchObject(indexCtx, req))
				synctest.Wait()
				current, _, attempts := storage.counts("object")
				require.Equal(t, 1, current, "the index download holds the object path")
				require.Equal(t, 1, attempts)
				if !tc.overlap {
					resume()
					waitReviewPrefetchIdle(t, fs)
					fs.diskCache.Flush(ctx)
				}

				require.NoError(t, fs.PrefetchObject(ctx, req))
				synctest.Wait()
				resume()
				waitReviewPrefetchIdle(t, fs)
				fs.diskCache.Flush(ctx)

				require.True(t, fs.indexObjectHeld("object"))
				_, _, attempts = storage.counts("object")
				require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))),
					"the disk cache stayed empty after %d storage open(s), despite a prefetch for each cache", attempts)
				require.Equal(t, 2, attempts)
			})
		})
	}
}

func TestStartupLoadPreservesRefilledFile(t *testing.T) {
	for _, tc := range []struct {
		name       string
		deletePath string
	}{
		{name: "baseline_unrelated_delete", deletePath: "unrelated"},
		{name: "delete_and_refill_same_object", deletePath: "object"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			d := newLifecycleTestDiskCache(t)
			path := d.wholePath("object", 3)
			require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
			require.NoError(t, os.WriteFile(path, []byte("old"), 0o644))

			found, resume, loaded := make(chan struct{}), make(chan struct{}), make(chan struct{})
			unblock := sync.OnceFunc(func() { close(resume) })
			d.hooks.beforeLoadFile = func(file string) {
				if file == path {
					close(found)
					<-resume
				}
			}
			go func() {
				defer close(loaded)
				d.loadCache(ctx)
			}()
			defer func() { unblock(); <-loaded }()
			select {
			case <-found:
			case <-ctx.Done():
				t.Fatal("startup load did not find the old file")
			}

			require.NoError(t, d.DeletePaths(ctx, []string{tc.deletePath}))
			want := "old"
			if tc.deletePath == "object" {
				want = "new"
				require.NoError(t, d.SetFile(ctx, "object", func(context.Context) (io.ReadCloser, error) {
					return io.NopCloser(strings.NewReader(want)), nil
				}))
				got, err := os.ReadFile(path)
				require.NoError(t, err)
				require.Equal(t, want, string(got), "refill published before the loader resumes")
			}

			unblock()
			<-loaded
			got, err := os.ReadFile(path)
			require.NoError(t, err, "startup load removed the current cache file")
			require.Equal(t, want, string(got))
			require.True(t, d.cache.Contains(path))
		})
	}
}

func TestDeleteRetryInvalidatesRange(t *testing.T) {
	for _, tc := range []struct {
		name          string
		readOnlyShard bool
	}{
		{name: "baseline_writable_shard"},
		{name: "first_delete_cannot_unlink_range", readOnlyShard: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnlyShard && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			fs, _, data := newPrefetchRangeFS(t, 0, 1024)
			<-fs.diskCache.load.done
			const offset, size = 100, 4
			rangeEntry := IOEntry{Offset: offset, Size: size}
			require.Equal(t, data[offset:offset+size], readRangeOf(t, fs, ExactRead, offset, size))
			path := fs.diskCache.pathForIOEntry("object", rangeEntry)
			require.FileExists(t, path)
			if tc.readOnlyShard {
				parent := filepath.Dir(path)
				require.NoError(t, os.Chmod(parent, 0o555))
				t.Cleanup(func() { require.NoError(t, os.Chmod(parent, 0o755)) })
				require.ErrorIs(t, fs.Delete(ctx, "object"), os.ErrPermission)
				require.FileExists(t, path)
				require.NoError(t, os.Chmod(parent, 0o755))
			} else {
				require.NoError(t, fs.Delete(ctx, "object"))
			}
			require.NoError(t, fs.Delete(ctx, "object"), "the delete retry succeeds")
			replacement := make([]byte, len(data))
			copy(replacement[offset:], "new!")
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Policy: SkipAllCache,
				Entries: []IOEntry{{Size: int64(len(replacement)), Data: replacement}}}))
			got := readRangeOf(t, fs, ExactRead, offset, size)
			require.Equal(t, "new!", string(got), "a successful delete retry must invalidate the old cached range")
		})
	}
}

func TestIndexPrefetchRepairsNonFileData(t *testing.T) {
	for _, tc := range []struct {
		name        string
		isDirectory bool
	}{
		{name: "baseline_missing_data"},
		{name: "data_replaced_by_directory", isDirectory: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			path, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			name := filepath.Join(path, IndexObjectFile)
			require.NoError(t, os.Remove(name))
			if tc.isDirectory {
				require.NoError(t, os.Mkdir(name, 0o755))
			}
			ctx := WithFileServicePolicy(context.Background(), IndexFileObject)
			require.NoError(t, fs.PrefetchFile(ctx, "object"))
			require.EqualValues(t, 1, storage.reads.Load(), "prefetch must replace the invalid index-cache entry from storage")
			require.Equal(t, data[100:104], readIndexObjectRange(t, fs, IndexFileObject, 100, 4))
			require.EqualValues(t, 1, storage.reads.Load(), "the repaired entry serves the next read")
		})
	}
}

func TestDropCachedRangeKeepsRangeUntilUnlinked(t *testing.T) {
	for _, tc := range []struct {
		name          string
		readOnlyShard bool
	}{
		{name: "baseline_writable_shard"},
		{name: "drop_cannot_unlink_range", readOnlyShard: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnlyShard && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			fs, _, data := newPrefetchRangeFS(t, 0, 1024)
			<-fs.diskCache.load.done
			const offset, size = 100, 4
			require.Equal(t, data[offset:offset+size], readRangeOf(t, fs, ExactRead, offset, size))
			path := fs.diskCache.pathForIOEntry("object", IOEntry{Offset: offset, Size: size})
			require.NoError(t, os.Truncate(path, 1))
			if tc.readOnlyShard {
				parent := filepath.Dir(path)
				require.NoError(t, os.Chmod(parent, 0o555))
				t.Cleanup(func() { require.NoError(t, os.Chmod(parent, 0o755)) })
				fs.diskCache.dropCachedRange(ctx, path)
				require.FileExists(t, path)
				require.NoError(t, os.Chmod(parent, 0o755))
			} else {
				fs.diskCache.dropCachedRange(ctx, path)
			}
			require.NoError(t, fs.Delete(ctx, "object"))
			require.NoFileExists(t, path, "a delete after a failed drop must remove the range file")
			require.False(t, fs.diskCache.cache.Contains(path))
		})
	}
}

func TestIndexReadRepairsNonFileData(t *testing.T) {
	for _, tc := range []struct {
		name        string
		isDirectory bool
	}{
		{name: "baseline_missing_data"},
		{name: "data_replaced_by_directory", isDirectory: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			path, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			name := filepath.Join(path, IndexObjectFile)
			require.NoError(t, os.Remove(name))
			if tc.isDirectory {
				require.NoError(t, os.Mkdir(name, 0o755))
			}
			require.Equal(t, data[100:104], readIndexObjectRange(t, fs, IndexFileObject, 100, 4))
			require.EqualValues(t, 1, storage.reads.Load())
			require.True(t, fs.indexObjectHeld("object"), "the read must replace the invalid index-cache entry")
			require.Equal(t, data[100:104], readIndexObjectRange(t, fs, IndexFileObject, 100, 4))
			require.EqualValues(t, 1, storage.reads.Load(), "the repaired entry serves the next read")
		})
	}
}
