// Copyright 2026 Matrix Origin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fileservice

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zaptest/observer"

	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/logutil"
	metric "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
	"github.com/stretchr/testify/require"
)

// Lifecycle tests use explicit phase barriers. This timeout is only a hang
// guard for a broken transition, so leave enough room for a loaded race/CI
// worker to schedule the goroutine that owns the release edge.
const diskCacheLifecycleTestTimeout = 5 * time.Second

func newLifecycleTestDiskCache(t *testing.T) *DiskCache {
	t.Helper()
	cache, err := NewDiskCache(
		context.Background(),
		t.TempDir(),
		fscache.ConstCapacity(1<<20),
		nil,
		false,
		nil,
		"",
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		cache.Close(context.Background())
	})
	return cache
}

type blockingDiskCacheWriter struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

// installBlockedDiskCacheFileSync isolates lifecycle scheduling from real
// filesystem sync latency. The injected call still marks the exact finalizer
// phase and blocks until released; normal cache writes retain the production
// file.Sync implementation.
func installBlockedDiskCacheFileSync(cache *DiskCache) (<-chan struct{}, func()) {
	started := make(chan struct{})
	markStarted := sync.OnceFunc(func() { close(started) })
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	cache.fileSync = func(*os.File) error {
		markStarted()
		<-release
		return nil
	}
	return started, unblock
}

func (w *blockingDiskCacheWriter) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.started) })
	<-w.release
	return len(p), nil
}

func TestDiskCacheIOEntryWaitIsBoundedDuringAsyncFinalize(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	syncStarted, unblock := installBlockedDiskCacheFileSync(cache)
	t.Cleanup(unblock)
	require.NoError(t, cache.Update(context.Background(), &IOVector{
		FilePath: "foo",
		Entries:  []IOEntry{{Offset: 0, Size: 1, Data: []byte("x")}},
	}, true))
	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async IOEntry finalizer did not start")
	}

	vector := &IOVector{FilePath: "foo", Entries: []IOEntry{{Offset: 0, Size: 1}}}
	readDone := make(chan struct{})
	var readErr error
	t.Cleanup(func() {
		unblock()
		<-readDone
		vector.Release()
	})
	go func() {
		defer close(readDone)
		readErr = cache.Read(context.Background(), vector)
	}()
	select {
	case <-readDone:
		require.NoError(t, readErr)
	case <-time.After(diskCacheLifecycleTestTimeout):
		unblock()
		<-readDone
		t.Fatal("IOEntry cache read waited for async disk finalization")
	}
	require.False(t, vector.Entries[0].done)

	unblock()
	flushCtx, cancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer cancel()
	cache.Flush(flushCtx)
	require.NoError(t, flushCtx.Err())
	require.NoError(t, cache.Read(context.Background(), vector))
	require.True(t, vector.Entries[0].done)
	require.Equal(t, []byte("x"), vector.Entries[0].Data)
}

func TestDiskCacheOpenReaderCannotReindexEvictedPath(t *testing.T) {
	tests := []struct {
		name  string
		evict func(context.Context, *DiskCache, string)
	}{
		{
			name: "Delete",
			evict: func(ctx context.Context, cache *DiskCache, diskPath string) {
				cache.cache.Delete(ctx, diskPath)
			},
		},
		{
			name: "ForceEvict",
			evict: func(ctx context.Context, cache *DiskCache, _ string) {
				cache.cache.ForceEvictWithWait(ctx, cache.cache.Used())
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cache := newLifecycleTestDiskCache(t)
			ctx := context.Background()
			require.NoError(t, cache.SetFile(ctx, "foo", func(context.Context) (io.ReadCloser, error) {
				return io.NopCloser(strings.NewReader("ab")), nil
			}))

			diskPath := cache.wholePath("foo", 2)
			writer := &blockingDiskCacheWriter{
				started: make(chan struct{}),
				release: make(chan struct{}),
			}
			releaseWriter := sync.OnceFunc(func() { close(writer.release) })
			vector := &IOVector{
				FilePath: "foo",
				Entries: []IOEntry{
					{Offset: 0, Size: 1, WriterForRead: writer},
					{Offset: 1, Size: 1},
				},
			}
			readDone := make(chan struct{})
			var readErr error
			t.Cleanup(func() {
				releaseWriter()
				<-readDone
				vector.Release()
			})
			go func() {
				defer close(readDone)
				readErr = cache.Read(ctx, vector)
			}()

			select {
			case <-writer.started:
			case <-time.After(diskCacheLifecycleTestTimeout):
				t.Fatal("disk-cache reader did not open the full-file cache entry")
			}
			test.evict(ctx, cache, diskPath)
			require.False(t, cache.cache.Contains(diskPath))
			cache.Flush(ctx)
			_, err := os.Stat(diskPath)
			require.ErrorIs(t, err, os.ErrNotExist)

			releaseWriter()
			select {
			case <-readDone:
				require.NoError(t, readErr)
			case <-time.After(diskCacheLifecycleTestTimeout):
				t.Fatal("disk-cache reader did not finish after eviction")
			}
			require.True(t, vector.Entries[0].done)
			require.False(t, vector.Entries[1].done, "the range left the index")
			cache.Flush(ctx)
			require.False(t, cache.cache.Contains(diskPath))
			_, err = os.Stat(diskPath)
			require.ErrorIs(t, err, os.ErrNotExist)
		})
	}
}

func TestDiskCacheCurrentReaderReindexesUntrackedPath(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	ctx := context.Background()
	diskPath := cache.pathForIOEntry("foo", IOEntry{Offset: 1, Size: 1})
	require.NoError(t, os.MkdirAll(filepath.Dir(diskPath), 0o755))
	require.NoError(t, os.WriteFile(diskPath, []byte("b"), 0o644))
	require.False(t, cache.cache.Contains(diskPath))

	vector := &IOVector{
		FilePath: "foo",
		Entries:  []IOEntry{{Offset: 1, Size: 1}},
	}
	defer vector.Release()
	require.NoError(t, cache.Read(ctx, vector))
	require.True(t, vector.Entries[0].done)
	require.Equal(t, []byte("b"), vector.Entries[0].Data)
	cache.Flush(ctx)
	require.True(t, cache.cache.Contains(diskPath))
	_, err := os.Stat(diskPath)
	require.NoError(t, err)
}

// The asynchronous updates of different paths are written by several workers at once.
func TestDiskCacheAsyncUpdatesWriteConcurrently(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	var inSync atomic.Int32
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	t.Cleanup(unblock)
	cache.fileSync = func(*os.File) error {
		inSync.Add(1)
		<-release
		return nil
	}
	ctx := context.Background()
	for i := 0; i < publisherWorkers; i++ {
		require.NoError(t, cache.Update(ctx, &IOVector{
			FilePath: fmt.Sprintf("file-%02d", i),
			Entries:  []IOEntry{{Offset: 0, Size: 1, Data: []byte("x")}},
		}, true))
	}
	require.Eventually(t, func() bool { return inSync.Load() == publisherWorkers },
		diskCacheLifecycleTestTimeout, time.Millisecond)
	unblock()
	cache.Flush(ctx)
	for i := 0; i < publisherWorkers; i++ {
		vec := &IOVector{FilePath: fmt.Sprintf("file-%02d", i), Entries: []IOEntry{{Offset: 0, Size: 1}}}
		require.NoError(t, cache.Read(ctx, vec))
		require.True(t, vec.Entries[0].done)
		vec.Release()
	}
}

func TestDiskCacheAsyncSetFileReturnsBeforeFinalize(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	syncStarted, unblock := installBlockedDiskCacheFileSync(cache)
	t.Cleanup(unblock)

	readerOpened := false
	require.NoError(t, cache.setFile(
		context.Background(),
		"foo",
		func(context.Context) (io.ReadCloser, error) {
			readerOpened = true
			return io.NopCloser(strings.NewReader("payload")), nil
		},
		true,
	))
	require.True(t, readerOpened)
	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async full-file finalizer did not start")
	}

	diskPath := cache.wholePath("foo", int64(len("payload")))
	_, err := os.Stat(diskPath)
	require.ErrorIs(t, err, os.ErrNotExist)

	duplicateReaderOpened := false
	require.NoError(t, cache.setFile(
		context.Background(),
		"foo",
		func(context.Context) (io.ReadCloser, error) {
			duplicateReaderOpened = true
			return io.NopCloser(strings.NewReader("duplicate")), nil
		},
		true,
	))
	require.False(t, duplicateReaderOpened)

	unblock()
	flushCtx, cancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer cancel()
	cache.Flush(flushCtx)
	require.NoError(t, flushCtx.Err())
	data, err := os.ReadFile(diskPath)
	require.NoError(t, err)
	require.Equal(t, []byte("payload"), data)
	require.False(t, cache.writeFailures.failed.Load())
}

func TestDiskCacheAsyncSetFileSourceErrorReleasesOwnership(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	expectedErr := io.ErrUnexpectedEOF
	err := cache.setFile(
		context.Background(),
		"foo",
		func(context.Context) (io.ReadCloser, error) {
			return nil, expectedErr
		},
		true,
	)
	require.ErrorIs(t, err, expectedErr)
	requireNoPublications(t, cache)
	require.False(t, cache.writeFailures.failed.Load())
}

func TestDiskCacheAsyncSetFileFinalizeErrorIsFailOpenAndObservable(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	core, logs := observer.New(zap.DebugLevel)
	cache.writeFailures.logger = logutil.NewRateLimitedLoggerWithConfig(
		zap.New(core),
		logutil.RateLimitedLoggerConfig{MaxKeys: 1},
	)
	cache.fileSync = func(*os.File) error { return os.ErrPermission }

	require.NoError(t, cache.setFile(
		context.Background(),
		"foo",
		func(context.Context) (io.ReadCloser, error) {
			return io.NopCloser(strings.NewReader("payload")), nil
		},
		true,
	))
	flushCtx, cancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer cancel()
	cache.Flush(flushCtx)
	require.NoError(t, flushCtx.Err())
	require.Equal(t, 1, logs.FilterMessage("write disk cache error").Len())
	require.True(t, cache.writeFailures.failed.Load())
	requireNoPublications(t, cache)
	require.Empty(t, cache.objectFiles("foo"))
}

func TestDiskCacheCloseDrainsQueuedFileFinalize(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	var inSync atomic.Int32
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	t.Cleanup(unblock)
	cache.fileSync = func(*os.File) error {
		inSync.Add(1)
		<-release
		return nil
	}

	open := func(value string) func(context.Context) (io.ReadCloser, error) {
		return func(context.Context) (io.ReadCloser, error) {
			return io.NopCloser(strings.NewReader(value)), nil
		}
	}
	// Every worker is blocked finalizing a file, so the next finalizer stays queued.
	var firstPaths []string
	for i := 0; i < publisherWorkers; i++ {
		name := fmt.Sprintf("first-%d", i)
		require.NoError(t, cache.setFile(context.Background(), name, open(name), true))
		firstPaths = append(firstPaths, cache.wholePath(name, int64(len(name))))
	}
	require.Eventually(t, func() bool { return inSync.Load() == publisherWorkers },
		diskCacheLifecycleTestTimeout, time.Millisecond, "the first full-file finalizers did not start")
	require.NoError(t, cache.setFile(context.Background(), "queued", open("queued"), true))
	queuedPath := cache.wholePath("queued", int64(len("queued")))
	// Canceling a queued finalizer only cleans its temporary artifact; it must
	// not claim that a previous disk-write failure recovered.
	cache.writeFailures.failed.Store(true)

	closeCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	cache.Close(closeCtx)
	require.ErrorIs(t, closeCtx.Err(), context.DeadlineExceeded)
	// A timed-out Close starts the background drain but does not join it. The first
	// finalizers are still blocked in file.Sync, so only the queued finalizer is released.
	require.Eventually(t, func() bool {
		cache.pub.mu.Lock()
		defer cache.pub.mu.Unlock()
		return cache.pub.mu.pending == publisherWorkers
	}, diskCacheLifecycleTestTimeout, time.Millisecond, "the queued finalizer was not released")
	require.True(t, cache.writeFailures.failed.Load())
	_, err := os.Stat(queuedPath)
	require.ErrorIs(t, err, os.ErrNotExist)

	unblock()
	finishCtx, finishCancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer finishCancel()
	cache.Close(finishCtx)
	require.NoError(t, finishCtx.Err())
	requireNoPublications(t, cache)
	select {
	case <-cache.pub.done:
	default:
		t.Fatal("successful Close did not join the async finalizer generation")
	}
	for _, diskPath := range append(firstPaths, queuedPath) {
		_, err := os.Stat(diskPath)
		require.ErrorIs(t, err, os.ErrNotExist)
	}
	tempFiles, err := filepath.Glob(filepath.Join(cache.path, "*"+cacheFileTempSuffix))
	require.NoError(t, err)
	require.Empty(t, tempFiles)
	require.True(t, cache.writeFailures.failed.Load())
}

func TestDiskCacheCloseOwnsAsyncLoaderGeneration(t *testing.T) {
	dir := t.TempDir()
	diskPath := filepath.Join(dir, diskCacheShard("foo"), "0-3foo"+cacheFileSuffix)
	require.NoError(t, os.MkdirAll(filepath.Dir(diskPath), 0o755))
	require.NoError(t, os.WriteFile(diskPath, []byte("old"), 0o644))

	loaderReachedEviction := make(chan struct{})
	releaseLoader := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(releaseLoader) })
	t.Cleanup(unblock)
	var capacityCalls atomic.Int32
	var signalLoader sync.Once
	capacity := func() int64 {
		if capacityCalls.Add(1) > 1 {
			signalLoader.Do(func() { close(loaderReachedEviction) })
			<-releaseLoader
			return 0
		}
		return 1 << 20
	}

	oldCache, err := NewDiskCache(
		context.Background(),
		dir,
		capacity,
		nil,
		true,
		nil,
		"",
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		unblock()
		cleanupCtx, cleanupCancel := context.WithTimeout(
			context.Background(),
			diskCacheLifecycleTestTimeout,
		)
		defer cleanupCancel()
		oldCache.Close(cleanupCtx)
	})
	select {
	case <-loaderReachedEviction:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async loader did not reach eviction")
	}

	expiredCtx, expiredCancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer expiredCancel()
	oldCache.Close(expiredCtx)
	require.ErrorIs(t, expiredCtx.Err(), context.DeadlineExceeded)
	select {
	case <-oldCache.load.done:
		t.Fatal("Close returned before the old loader generation stopped")
	default:
	}

	unblock()
	finishCtx, finishCancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer finishCancel()
	oldCache.Close(finishCtx)
	require.NoError(t, finishCtx.Err())
	select {
	case <-oldCache.load.done:
	default:
		t.Fatal("successful Close did not join the old loader generation")
	}

	require.NoError(t, os.WriteFile(diskPath, []byte("new"), 0o644))
	newCache, err := NewDiskCache(
		context.Background(),
		dir,
		fscache.ConstCapacity(1<<20),
		nil,
		false,
		nil,
		"",
	)
	require.NoError(t, err)
	t.Cleanup(func() { newCache.Close(context.Background()) })
	require.True(t, newCache.cache.Contains(diskPath))
	data, err := os.ReadFile(diskPath)
	require.NoError(t, err)
	require.Equal(t, []byte("new"), data)
}

func TestDiskCacheLoadCacheKeepsForegroundDirectory(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	foregroundDir := filepath.Dir(cache.objectKey("foo"))

	writerReachedTempFileCreate := make(chan struct{})
	releaseWriter := make(chan struct{})
	unblockWriter := sync.OnceFunc(func() { close(releaseWriter) })
	cache.hooks.beforeCacheTempFileCreate = func() {
		close(writerReachedTempFileCreate)
		<-releaseWriter
	}
	writerDone := make(chan struct{})
	writerResult := make(chan struct {
		file *os.File
		err  error
	}, 1)
	go func() {
		defer close(writerDone)
		file, err := cache.createCacheTempFile(foregroundDir)
		writerResult <- struct {
			file *os.File
			err  error
		}{file: file, err: err}
	}()
	t.Cleanup(func() {
		unblockWriter()
		<-writerDone
	})
	select {
	case <-writerReachedTempFileCreate:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("foreground cache write did not reach temporary file creation")
	}

	loaderReachedDirectoryCleanup := make(chan struct{})
	cache.hooks.beforeLoadDirectoryCleanup = func(path string) {
		if path == foregroundDir {
			close(loaderReachedDirectoryCleanup)
		}
	}
	loadDone := make(chan struct{})
	go func() {
		cache.loadCache(context.Background())
		close(loadDone)
	}()
	t.Cleanup(func() {
		unblockWriter()
		<-loadDone
	})
	select {
	case <-loaderReachedDirectoryCleanup:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("disk-cache loader did not process the foreground directory")
	}

	unblockWriter()
	result := <-writerResult
	if result.file != nil {
		t.Cleanup(func() { _ = result.file.Close() })
	}
	require.NoError(t, result.err)
	require.NoError(t, result.file.Close())
	<-loadDone
	require.DirExists(t, foregroundDir)

	cache.hooks.beforeLoadDirectoryCleanup = nil
	staleDir := filepath.Join(cache.path, "stale")
	require.NoError(t, os.MkdirAll(staleDir, 0o755))
	cache.loadCache(context.Background())
	_, err := os.Stat(staleDir)
	require.ErrorIs(t, err, os.ErrNotExist)
}

func TestDiskCacheCloseCancelsActiveFileFinalizeBeforePublish(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	syncStarted, unblock := installBlockedDiskCacheFileSync(cache)
	t.Cleanup(unblock)

	require.NoError(t, cache.setFile(
		context.Background(),
		"active",
		func(context.Context) (io.ReadCloser, error) {
			return io.NopCloser(strings.NewReader("payload")), nil
		},
		true,
	))
	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async full-file finalizer did not start")
	}

	closeCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	cache.Close(closeCtx)
	require.ErrorIs(t, closeCtx.Err(), context.DeadlineExceeded)
	unblock()

	finishCtx, finishCancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer finishCancel()
	cache.Close(finishCtx)
	require.NoError(t, finishCtx.Err())
	require.Empty(t, cache.objectFiles("active"))
}

func TestDiskCacheCloseWinsAtFilePublicationBoundary(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	// Exercise publication/Close ordering without waiting for real fsync.
	// Persistence errors and Close during an active sync have separate tests;
	// this test must reach the publication gate before initiating Close.
	cache.fileSync = func(*os.File) error { return nil }
	publicationReached := make(chan struct{})
	releasePublication := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(releasePublication) })
	t.Cleanup(unblock)
	cache.hooks.beforeFilePublication = func() {
		close(publicationReached)
		<-releasePublication
	}

	require.NoError(t, cache.setFile(
		context.Background(),
		"boundary",
		func(context.Context) (io.ReadCloser, error) {
			return io.NopCloser(strings.NewReader("payload")), nil
		},
		true,
	))
	select {
	case <-publicationReached:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async finalizer did not reach the publication boundary")
	}

	closeCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	cache.Close(closeCtx)
	require.ErrorIs(t, closeCtx.Err(), context.DeadlineExceeded)
	unblock()

	finishCtx, finishCancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer finishCancel()
	cache.Close(finishCtx)
	require.NoError(t, finishCtx.Err())
	require.Empty(t, cache.objectFiles("boundary"))
}

func TestDiskCachePublicationClaimWinsBeforeClose(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	// This test owns the publication/Close linearization boundary. Keep the
	// unrelated persistence syscall out of that ownership contract; the real
	// file.Sync path is covered by the dedicated error and finalize tests.
	cache.fileSync = func(*os.File) error { return nil }
	diskPath := cache.objectKey("claimed")
	require.True(t, cache.pub.begin())
	var releaseReservationOnce sync.Once
	releaseReservation := func() {
		releaseReservationOnce.Do(func() {
			cache.pub.end()
		})
	}
	// Own the reservation immediately. The worker below takes this ownership
	// over only after all setup succeeds, and the once keeps failure cleanup
	// from racing with worker cleanup.
	t.Cleanup(releaseReservation)
	require.NoError(t, os.MkdirAll(filepath.Dir(diskPath), 0o755))
	tempFile, err := os.CreateTemp(filepath.Dir(diskPath), "claimed-*"+cacheFileTempSuffix)
	require.NoError(t, err)
	var cleanupTempOnce sync.Once
	var cleanupTempErr error
	cleanupTemp := func() error {
		cleanupTempOnce.Do(func() {
			cleanupTempErr = cleanupDiskCacheTempFile(tempFile)
		})
		return cleanupTempErr
	}
	t.Cleanup(func() { _ = cleanupTemp() })
	_, err = tempFile.WriteString("payload")
	require.NoError(t, err)

	claimReached := make(chan struct{})
	releaseClaim := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(releaseClaim) })
	finalizeDone := make(chan error, 1)
	workerDone := make(chan struct{})
	go func() {
		defer close(workerDone)
		finalizeErr := cache.finalizeFileTo(
			cache.pub.ctx,
			diskPath,
			tempFile,
			func() bool {
				claimed := cache.pub.claim()
				close(claimReached)
				<-releaseClaim
				return claimed
			},
			diskFillTarget{d: cache, diskPath: diskPath},
		)
		cleanupErr := cleanupTemp()
		releaseReservation()
		finalizeDone <- errors.Join(finalizeErr, cleanupErr)
	}()
	// Always release the barrier before waiting for the worker. If the test
	// fails before the normal release, this cleanup still lets the worker finish
	// before the fallback temp/update/reservation cleanups run.
	t.Cleanup(func() {
		unblock()
		select {
		case <-workerDone:
		case <-time.After(diskCacheLifecycleTestTimeout):
			t.Fatal("publication finalizer did not finish during test cleanup")
		}
	})
	claimObserved := false
	claimWaitTimedOut := false
	select {
	case <-claimReached:
		claimObserved = true
	case err := <-finalizeDone:
		require.NoError(t, err, "publication finalizer exited before the claim barrier")
		select {
		case <-claimReached:
			claimObserved = true
		default:
		}
	case <-time.After(diskCacheLifecycleTestTimeout):
		claimWaitTimedOut = true
		unblock()
		select {
		case err := <-finalizeDone:
			require.NoError(t, err)
			select {
			case <-claimReached:
				claimObserved = true
			default:
			}
		case <-time.After(diskCacheLifecycleTestTimeout):
			t.Fatal("async finalizer did not claim the publication generation")
		}
	}
	require.True(t, claimObserved, "async finalizer did not claim the publication generation")
	if claimWaitTimedOut {
		t.Fatal("async finalizer did not claim the publication generation within the guard")
	}

	closeCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	cache.Close(closeCtx)
	require.ErrorIs(t, closeCtx.Err(), context.DeadlineExceeded)
	unblock()
	select {
	case err := <-finalizeDone:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("claimed publication did not complete")
	}

	finishCtx, finishCancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer finishCancel()
	cache.Close(finishCtx)
	require.NoError(t, finishCtx.Err())
	data, err := os.ReadFile(diskPath)
	require.NoError(t, err)
	require.Equal(t, []byte("payload"), data)
}

func TestDiskCacheCloseTracksReservedFileFinalize(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	reader, writer := io.Pipe()
	readerOpened := make(chan struct{})
	closeWriter := sync.OnceFunc(func() { _ = writer.Close() })
	t.Cleanup(closeWriter)
	setFileDone := make(chan error, 1)
	go func() {
		setFileDone <- cache.setFile(
			context.Background(),
			"reserved",
			func(context.Context) (io.ReadCloser, error) {
				close(readerOpened)
				return reader, nil
			},
			true,
		)
	}()
	select {
	case <-readerOpened:
	case <-time.After(diskCacheLifecycleTestTimeout):
		closeWriter()
		<-setFileDone
		t.Fatal("async full-file source reader did not open")
	}

	const numClosers = 4
	closeResults := make(chan error, numClosers)
	for i := 0; i < numClosers; i++ {
		go func() {
			closeCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			cache.Close(closeCtx)
			closeResults <- closeCtx.Err()
		}()
	}
	for i := 0; i < numClosers; i++ {
		select {
		case err := <-closeResults:
			require.ErrorIs(t, err, context.DeadlineExceeded)
		case <-time.After(diskCacheLifecycleTestTimeout):
			t.Fatal("concurrent Close did not honor its deadline")
		}
	}

	_, err := writer.Write([]byte("payload"))
	require.NoError(t, err)
	closeWriter()
	select {
	case err := <-setFileDone:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async SetFile did not transfer canceled finalizer cleanup")
	}
	finishCtx, finishCancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer finishCancel()
	cache.Close(finishCtx)
	require.NoError(t, finishCtx.Err())
	require.Empty(t, cache.objectFiles("reserved"))
}

func TestDiskCacheCloseDeadlineDoesNotWaitForQueuedCleanup(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	diskPath := cache.objectKey("queued")
	cleanupStarted := make(chan struct{})
	releaseCleanup := make(chan struct{})
	unblockCleanup := sync.OnceFunc(func() { close(releaseCleanup) })
	t.Cleanup(unblockCleanup)
	tempFile, err := os.CreateTemp(cache.path, "close-deadline-*")
	require.NoError(t, err)
	p := &cache.pub
	require.True(t, p.begin())
	p.jobs <- &publication{
		diskPath: diskPath,
		publish:  func(context.Context) (bool, error) { return false, nil },
		release: func() error {
			close(cleanupStarted)
			<-releaseCleanup
			return cleanupDiskCacheTempFile(tempFile)
		},
	}
	metric.FSDiskCachePublisherQueued.Inc()
	p.submitters.Done()

	closeCtx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	closeReturned := make(chan struct{})
	go func() {
		cache.Close(closeCtx)
		close(closeReturned)
	}()

	select {
	case <-cleanupStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		unblockCleanup()
		<-closeReturned
		t.Fatal("queued cleanup did not start during close")
	}
	select {
	case <-closeReturned:
		require.ErrorIs(t, closeCtx.Err(), context.DeadlineExceeded)
	case <-time.After(diskCacheLifecycleTestTimeout):
		unblockCleanup()
		<-closeReturned
		t.Fatal("Close exceeded its context deadline while queued cleanup was blocked")
	}

	unblockCleanup()
	finishCtx, finishCancel := context.WithTimeout(context.Background(), diskCacheLifecycleTestTimeout)
	defer finishCancel()
	cache.Close(finishCtx)
	require.NoError(t, finishCtx.Err())
	requireNoPublications(t, cache)
	_, err = os.Stat(tempFile.Name())
	require.ErrorIs(t, err, os.ErrNotExist)
}

// requireNoPublications requires that cache has no publication begun.
func requireNoPublications(t *testing.T, cache *DiskCache) {
	t.Helper()
	cache.pub.mu.Lock()
	defer cache.pub.mu.Unlock()
	require.Zero(t, cache.pub.mu.pending)
}

// shrinkPublisher makes the publisher of cache publish one publication at a time from a
// queue of size, before its first submission.
func shrinkPublisher(cache *DiskCache, size int) {
	cache.pub.workers = 1
	cache.pub.jobs = make(chan *publication, size)
}

// requireSubmitterWaits requires that the submission done reports has not returned while
// the publisher queue of cache is full.
func requireSubmitterWaits(t *testing.T, cache *DiskCache, done <-chan error) {
	t.Helper()
	require.Equal(t, cap(cache.pub.jobs), len(cache.pub.jobs), "the queue is full")
	select {
	case err := <-done:
		t.Fatalf("a submission returned while the publisher queue is full: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
}

// requireNoTempFiles requires that no temporary file is left in the directory of cache.
func requireNoTempFiles(t *testing.T, cache *DiskCache) {
	t.Helper()
	var temps []string
	require.NoError(t, filepath.WalkDir(cache.path, func(path string, _ os.DirEntry, err error) error {
		if strings.HasSuffix(path, cacheFileTempSuffix) || strings.HasSuffix(path, cacheDirTempSuffix) {
			temps = append(temps, path)
		}
		return err
	}))
	require.Empty(t, temps)
}

// A submission to a full publisher queue waits for room; one whose context ends returns,
// caching nothing and leaving nothing behind, and one that waits on completes once room
// frees. The control is a submission with room, which returns at once.
func TestDiskCacheUpdateWaitsForPublisherRoom(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	shrinkPublisher(cache, 1)
	syncStarted, unblock := installBlockedDiskCacheFileSync(cache)
	t.Cleanup(unblock)
	update := func(ctx context.Context, name string) error {
		return cache.Update(ctx, &IOVector{
			FilePath: name,
			Entries:  []IOEntry{{Offset: 0, Size: 1, Data: []byte("x")}},
		}, true)
	}
	ctx := context.Background()
	require.NoError(t, update(ctx, "published"))
	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the publisher did not take the first publication")
	}
	// The control: with room in the queue, a submission returns at once.
	require.NoError(t, update(ctx, "queued"))

	waitCtx, cancel := context.WithCancel(ctx)
	canceled := make(chan error, 1)
	go func() { canceled <- update(waitCtx, "canceled") }()
	requireSubmitterWaits(t, cache, canceled)
	cancel()
	select {
	case err := <-canceled:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the waiting submission did not end with its context")
	}

	waiting := make(chan error, 1)
	go func() { waiting <- update(ctx, "waiting") }()
	requireSubmitterWaits(t, cache, waiting)
	unblock()
	select {
	case err := <-waiting:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the waiting submission did not complete once room freed")
	}
	flushCtx, cancelFlush := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
	defer cancelFlush()
	cache.Flush(flushCtx)
	require.NoError(t, flushCtx.Err())
	for name, cached := range map[string]bool{"published": true, "queued": true, "waiting": true, "canceled": false} {
		vec := &IOVector{FilePath: name, Entries: []IOEntry{{Offset: 0, Size: 1}}}
		require.NoError(t, cache.Read(ctx, vec))
		require.Equal(t, cached, vec.Entries[0].done, name)
		vec.Release()
	}
	requireNoPublications(t, cache)
	requireNoTempFiles(t, cache)
}

func TestDiskCacheAsyncUpdateRejectsCanceledCaller(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := cache.Update(ctx, &IOVector{
		FilePath: "foo",
		Entries:  []IOEntry{{Offset: 0, Size: 1, Data: []byte("x")}},
	}, true)
	require.ErrorIs(t, err, context.Canceled)
	requireNoPublications(t, cache)
}

func TestDiskCacheWriteErrorsAndAlternatingRecoveryAreRateLimited(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	core, logs := observer.New(zap.DebugLevel)
	cache.writeFailures.logger = logutil.NewRateLimitedLoggerWithConfig(
		zap.New(core),
		logutil.RateLimitedLoggerConfig{MaxKeys: 1},
	)

	require.NoError(t, os.RemoveAll(cache.path))
	require.NoError(t, os.WriteFile(cache.path, []byte("not-a-directory"), 0o644))
	for i := 0; i < 10; i++ {
		require.NoError(t, cache.Update(context.Background(), &IOVector{
			FilePath: "broken",
			Entries:  []IOEntry{{Offset: int64(i), Size: 1, Data: []byte("x")}},
		}, false))
	}
	require.Equal(t, 1, logs.FilterMessage("write disk cache error").Len())
	require.True(t, cache.writeFailures.failed.Load())

	require.NoError(t, os.Remove(cache.path))
	require.NoError(t, os.MkdirAll(cache.path, 0o755))
	require.NoError(t, cache.Update(context.Background(), &IOVector{
		FilePath: "recovered",
		Entries:  []IOEntry{{Offset: 0, Size: 1, Data: []byte("x")}},
	}, false))
	require.Equal(t, 1, logs.FilterMessage("disk cache write recovered").Len())
	require.False(t, cache.writeFailures.failed.Load())

	for i := 0; i < 10; i++ {
		require.NoError(t, os.RemoveAll(cache.path))
		require.NoError(t, os.WriteFile(cache.path, []byte("not-a-directory"), 0o644))
		require.NoError(t, cache.Update(context.Background(), &IOVector{
			FilePath: "broken-again",
			Entries:  []IOEntry{{Offset: int64(i), Size: 1, Data: []byte("x")}},
		}, false))

		require.NoError(t, os.Remove(cache.path))
		require.NoError(t, os.MkdirAll(cache.path, 0o755))
		require.NoError(t, cache.Update(context.Background(), &IOVector{
			FilePath: "recovered-again",
			Entries:  []IOEntry{{Offset: int64(i), Size: 1, Data: []byte("x")}},
		}, false))
	}
	require.Equal(t, 1, logs.FilterMessage("write disk cache error").Len())
	require.Equal(t, 1, logs.FilterMessage("disk cache write recovered").Len())
	require.False(t, cache.writeFailures.failed.Load())
}

func TestDiskCacheCacheHitDoesNotClaimWriteRecovery(t *testing.T) {
	cache := newLifecycleTestDiskCache(t)
	cache.writeFailures.logger = logutil.NewRateLimitedLoggerWithConfig(
		zap.NewNop(),
		logutil.RateLimitedLoggerConfig{MaxKeys: 1},
	)
	ctx := context.Background()
	require.NoError(t, cache.SetFile(ctx, "cached", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader("cached")), nil
	}))

	cache.fileSync = func(*os.File) error { return os.ErrPermission }
	require.NoError(t, cache.SetFile(ctx, "broken", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader("broken")), nil
	}))
	require.True(t, cache.writeFailures.failed.Load())

	readerOpened := false
	require.NoError(t, cache.SetFile(ctx, "cached", func(context.Context) (io.ReadCloser, error) {
		readerOpened = true
		return io.NopCloser(strings.NewReader("duplicate")), nil
	}))
	require.False(t, readerOpened)
	require.True(t, cache.writeFailures.failed.Load())

	cache.fileSync = func(file *os.File) error { return file.Sync() }
	require.NoError(t, cache.SetFile(ctx, "recovered", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader("recovered")), nil
	}))
	require.False(t, cache.writeFailures.failed.Load())
}

// A delete of an object while the load has found a range file of it, but not yet indexed
// it, removes the file: reads after the delete never get its bytes.
func TestDiskCacheLoadDropsRangeFileOfObjectDeletedMeanwhile(t *testing.T) {
	ctx := context.Background()
	cache := newLifecycleTestDiskCache(t)
	entry := IOEntry{Offset: 100, Size: 4}
	rangePath := cache.pathForIOEntry("obj", entry)
	require.NoError(t, os.MkdirAll(filepath.Dir(rangePath), 0o755))
	require.NoError(t, os.WriteFile(rangePath, []byte{0x11, 0x11, 0x11, 0x11}, 0o644))

	reached, release := make(chan struct{}), make(chan struct{})
	cache.hooks.beforeLoadFile = func(diskPath string) {
		if diskPath == rangePath {
			close(reached)
			<-release
		}
	}
	loadDone := make(chan struct{})
	go func() {
		cache.loadCache(ctx)
		close(loadDone)
	}()
	select {
	case <-reached:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the load did not reach the range file")
	}
	require.NoError(t, cache.DeletePaths(ctx, []string{"obj"}))
	close(release)
	<-loadDone
	cache.Flush(ctx)

	require.False(t, cache.cache.Contains(rangePath))
	require.Empty(t, cache.ranges.of(cache.objectKey("obj")))
	_, err := os.Stat(rangePath)
	require.ErrorIs(t, err, os.ErrNotExist)
}

// A read of a range file the load has found but not yet indexed, after a delete of its
// object, neither gets the file's bytes nor indexes the file.
func TestDiskCacheReadSkipsUnloadedRangeFileOfDeletedObject(t *testing.T) {
	ctx := context.Background()
	cache := newLifecycleTestDiskCache(t)
	entry := IOEntry{Offset: 100, Size: 4}
	rangePath := cache.pathForIOEntry("obj", entry)
	require.NoError(t, os.MkdirAll(filepath.Dir(rangePath), 0o755))
	require.NoError(t, os.WriteFile(rangePath, []byte{0x11, 0x11, 0x11, 0x11}, 0o644))

	reached, release := make(chan struct{}), make(chan struct{})
	cache.hooks.beforeLoadFile = func(diskPath string) {
		if diskPath == rangePath {
			close(reached)
			<-release
		}
	}
	loadDone := make(chan struct{})
	go func() {
		cache.loadCache(ctx)
		close(loadDone)
	}()
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
		<-loadDone
	}()
	select {
	case <-reached:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the load did not reach the range file")
	}
	require.NoError(t, cache.DeletePaths(ctx, []string{"obj"}))

	vector := &IOVector{FilePath: "obj", Entries: []IOEntry{entry}}
	require.NoError(t, cache.Read(ctx, vector))
	require.False(t, vector.Entries[0].done)
	require.False(t, cache.cache.Contains(rangePath))
	require.Empty(t, cache.ranges.of(cache.objectKey("obj")))

	close(release)
	<-loadDone
	cache.Flush(ctx)
	_, err := os.Stat(rangePath)
	require.ErrorIs(t, err, os.ErrNotExist)
}

// holdPublisher gives the publisher of d one worker, which stops at the first file or
// entry directory it publishes until release is called; held is closed once it stops.
func holdPublisher(t *testing.T, d *DiskCache) (held <-chan struct{}, release func()) {
	t.Helper()
	d.pub.workers = 1
	h, r := make(chan struct{}), make(chan struct{})
	var once sync.Once
	d.hooks.beforeFilePublication = func() {
		once.Do(func() {
			close(h)
			<-r
		})
	}
	release = sync.OnceFunc(func() { close(r) })
	t.Cleanup(release)
	return h, release
}

// holdObject holds a publication of "object" by the disk cache of fs, that of a read of
// size bytes at offset, and returns its release, which waits for the read.
func holdObject(t *testing.T, fs *S3FS, offset, size int64) func() {
	t.Helper()
	held, release := holdPublisher(t, fs.diskCache)
	read := make(chan error, 1)
	go func() {
		vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache | SkipFullFilePreloads,
			Entries: []IOEntry{{Offset: offset, Size: size}}}
		defer vec.Release()
		read <- fs.Read(context.Background(), vec)
	}()
	<-held
	return func() {
		release()
		require.NoError(t, <-read)
	}
}

// cacheRange reads the range [offset, offset+size) of "object" into the disk cache of fs
// and returns its cache file.
func cacheRange(t *testing.T, fs *S3FS, offset, size int64) string {
	t.Helper()
	readRangeOf(t, fs, 0, offset, size)
	path := fs.diskCache.pathForIOEntry("object", IOEntry{Offset: offset, Size: size})
	require.True(t, fs.diskCache.cache.Contains(path))
	return path
}

// A read waits for no publication: one of the bytes whose publication is held reads them
// from object storage, and the control, once they are published, from the cache.
func TestReadDoesNotWaitForPublication(t *testing.T) {
	ctx := context.Background()
	fs, storage, data := newPrefetchRangeFS(t, 0, 64<<10)
	read := func() []byte {
		vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache, Entries: []IOEntry{{Offset: 100, Size: 10}}}
		defer vec.Release()
		require.NoError(t, fs.Read(ctx, vec))
		return slices.Clone(vec.Entries[0].Data)
	}
	release := holdObject(t, fs, 100, 10)
	done := make(chan []byte, 1)
	go func() { done <- read() }()
	select {
	case got := <-done:
		require.Equal(t, data[100:110], got)
	case <-time.After(5 * time.Second):
		release()
		t.Fatal("a read waited for a publication")
	}
	release()
	fs.diskCache.Flush(ctx)
	reads := storage.reads.Load()
	require.Equal(t, data[100:110], read())
	require.Equal(t, reads, storage.reads.Load(), "the control reads the published range")
}

// An evicted range file is removed by the publisher, after the publications of its object
// taken before; once the publisher is idle no file of the path is left unindexed. The
// control holds no publication.
func TestEvictionRemovesFileOnPublisher(t *testing.T) {
	for _, held := range []bool{true, false} {
		t.Run(map[bool]string{true: "held", false: "control"}[held], func(t *testing.T) {
			ctx := context.Background()
			fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
			d := fs.diskCache
			path := cacheRange(t, fs, 100, 10)
			release := func() {}
			if held {
				release = holdObject(t, fs, 20000, 10)
			}
			d.cache.Delete(ctx, path)
			require.False(t, d.cache.Contains(path))
			if held {
				require.FileExists(t, path, "the removal waits for the publisher")
			}
			release()
			d.Flush(ctx)
			// A file left is one published again since, and indexed.
			if _, err := os.Stat(path); err == nil {
				require.True(t, d.cache.Contains(path), "a file left on disk is indexed")
			}
		})
	}
}

// Reads of a file the cache does not hold, while the publisher is busy, queue one job to
// index it, however many read it; once the job runs the file is indexed. The control reads
// it once.
func TestReadsOfUnindexedFileQueueOneJob(t *testing.T) {
	for _, reads := range []int{100, 1} {
		t.Run(map[int]string{100: "many", 1: "control"}[reads], func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newPrefetchRangeFS(t, 0, 64<<10)
			d := fs.diskCache
			<-d.load.done
			path := d.pathForIOEntry("object", IOEntry{Offset: 200, Size: 10})
			require.NoError(t, os.MkdirAll(filepath.Dir(path), 0755))
			require.NoError(t, os.WriteFile(path, data[200:210], 0644))
			release := holdObject(t, fs, 20000, 10)
			before := storage.reads.Load()
			for range reads {
				require.Equal(t, data[200:210], readRangeOfNoWait(t, fs, 200, 10))
			}
			require.Equal(t, before, storage.reads.Load(), "the reads read the file")
			d.pub.mu.Lock()
			queued := len(d.pub.mu.queued)
			d.pub.mu.Unlock()
			require.Equal(t, 1, queued)
			release()
			d.Flush(ctx)
			require.True(t, d.cache.Contains(path))
		})
	}
}

// readRangeOfNoWait reads size bytes at offset of "object" through fs.
func readRangeOfNoWait(t *testing.T, fs *S3FS, offset, size int64) []byte {
	t.Helper()
	vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
		Entries: []IOEntry{{Offset: offset, Size: size}}}
	defer vec.Release()
	require.NoError(t, fs.Read(context.Background(), vec))
	return slices.Clone(vec.Entries[0].Data)
}

// A cached range found short by a read is removed by the publisher, after the publications
// of its object taken before, and the range is cached again; the control removes it once
// the publisher is idle.
func TestBrokenRangeIsRemovedOnPublisher(t *testing.T) {
	for _, held := range []bool{true, false} {
		t.Run(map[bool]string{true: "held", false: "control"}[held], func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newPrefetchRangeFS(t, 4096, 64<<10)
			d := fs.diskCache
			require.Equal(t, data[5000:5002], readRangeOf(t, fs, 0, 5000, 2))
			rs := d.ranges.of(d.objectKey("object"))
			require.Len(t, rs, 1)
			path := rs[0].diskPath
			require.NoError(t, os.Truncate(path, 100))
			d.files.closeAll()
			release := func() {}
			if held {
				release = holdObject(t, fs, 40000, 10)
			}
			vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache | SkipFullFilePreloads,
				Entries: []IOEntry{{Offset: 6000, Size: 10}}}
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[6000:6010], vec.Entries[0].Data)
			vec.Release()
			if held {
				info, err := os.Stat(path)
				require.NoError(t, err)
				require.Equal(t, int64(100), info.Size(), "the removal waits for the publisher")
			}
			release()
			waitObjectsEnd(d, "object")
			d.Flush(ctx)
			require.Equal(t, data[6100:6110], readRangeOf(t, fs, 0, 6100, 10))
			reads := storage.reads.Load()
			require.Equal(t, data[6200:6210], readRangeOf(t, fs, 0, 6200, 10))
			require.Equal(t, reads, storage.reads.Load(), "the range is cached again")
		})
	}
}

// waitQueued waits until n publications wait in the queue of the publisher of d.
func waitQueued(t *testing.T, d *DiskCache, n int) {
	t.Helper()
	require.Eventually(t, func() bool { return len(d.pub.jobs) == n }, diskCacheLifecycleTestTimeout, time.Millisecond)
}

// deleteReleasing runs del while a publisher is held, releases the publisher once del
// returns or a moment passed, and returns the error of del.
func deleteReleasing(release func(), del func() error) error {
	deleted := make(chan error, 1)
	go func() { deleted <- del() }()
	select {
	case err := <-deleted:
		release()
		return err
	case <-time.After(50 * time.Millisecond):
	}
	release()
	return <-deleted
}

// A publication of an object queued before a delete of the object is never published
// after the delete returns: the delete removes it once it is published, or it is not.
func TestPublicationQueuedBeforeDeleteIsNotPublishedAfterIt(t *testing.T) {
	ctx := context.Background()
	cachingWrite := func(fs *S3FS, name string) {
		require.NoError(t, fs.Write(ctx, IOVector{FilePath: name, Policy: SkipMemoryCache,
			Entries: []IOEntry{{Size: 4096, Data: make([]byte, 4096)}}}))
	}
	t.Run("whole object", func(t *testing.T) {
		fs, _, _ := newIndexObjectFS(t, true)
		d := fs.diskCache
		held, release := holdPublisher(t, d)
		cachingWrite(fs, "other")
		<-held
		cachingWrite(fs, "target")
		waitQueued(t, d, 1)
		require.NoError(t, deleteReleasing(release, func() error { return fs.Delete(ctx, "target") }))
		d.Flush(ctx)
		require.Empty(t, d.ranges.of(d.objectKey("target")))
	})
	t.Run("range", func(t *testing.T) {
		fs, _, _ := newIndexObjectFS(t, true)
		d := fs.diskCache
		held, release := holdPublisher(t, d)
		cachingWrite(fs, "other")
		<-held
		vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache, Entries: []IOEntry{{Offset: 100, Size: 10}}}
		require.NoError(t, fs.Read(ctx, vec))
		vec.Release()
		waitQueued(t, d, 1)
		require.NoError(t, deleteReleasing(release, func() error { return fs.Delete(ctx, "object") }))
		d.Flush(ctx)
		require.Empty(t, d.ranges.of(d.objectKey("object")))
	})
	putOther := func(c *IndexFileCache) chan error {
		put := make(chan error, 1)
		go func() {
			_, err := c.Put(ctx, "other", []IndexCacheFile{{Name: "f", Open: func(context.Context) (io.ReadCloser, error) {
				return io.NopCloser(strings.NewReader("other")), nil
			}}})
			put <- err
		}()
		return put
	}
	t.Run("index object", func(t *testing.T) {
		fs, _, _ := newIndexObjectFS(t, true)
		c := fs.indexFileCache
		held, release := holdPublisher(t, c.disk)
		other := putOther(c)
		<-held
		require.NoError(t, fs.PrefetchFile(WithFileServicePolicy(ctx, IndexFileObject), "object"))
		waitQueued(t, c.disk, 1)
		require.NoError(t, deleteReleasing(release, func() error { return fs.Delete(ctx, "object") }))
		require.NoError(t, <-other)
		flushPublishers(fs)
		_, cached := fs.indexObjectCached("object")
		require.False(t, cached)
	})
	t.Run("index entry", func(t *testing.T) {
		fs, _, _ := newIndexObjectFS(t, true)
		c := fs.indexFileCache
		held, release := holdPublisher(t, c.disk)
		other := putOther(c)
		<-held
		put := make(chan error, 1)
		go func() {
			_, err := c.Put(ctx, "entry", []IndexCacheFile{{Name: "f", Open: func(context.Context) (io.ReadCloser, error) {
				return io.NopCloser(strings.NewReader("entry")), nil
			}}})
			put <- err
		}()
		waitQueued(t, c.disk, 1)
		require.NoError(t, deleteReleasing(release, func() error { return c.Delete(ctx, "entry") }))
		require.NoError(t, <-other)
		require.NoError(t, <-put)
		c.disk.Flush(ctx)
		dir, err := c.disk.pathForDir("entry")
		require.NoError(t, err)
		require.False(t, c.disk.cache.Contains(dir))
		_, err = os.Stat(dir)
		require.True(t, os.IsNotExist(err), "%v", err)
	})
}

// firstReadGateStorage blocks the first read until open is closed, or its context ends;
// started is closed once it blocks.
type firstReadGateStorage struct {
	ObjectStorage
	once    sync.Once
	started chan struct{}
	open    chan struct{}
}

func (s *firstReadGateStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	first := false
	s.once.Do(func() { first = true })
	if first {
		close(s.started)
		select {
		case <-s.open:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return s.ObjectStorage.Read(ctx, key, min, max)
}

// A delete of an object while a fill of it runs, waits for its publication or publishes
// leaves no range of the old version: the next reads after the object is rewritten get
// the new version.
func TestDeleteDuringFillReadsNewVersion(t *testing.T) {
	const size = 64 << 10
	ctx := context.Background()
	for _, state := range []string{"running", "queued", "publishing"} {
		t.Run(state, func(t *testing.T) {
			fs, _, v1 := newPrefetchRangeFS(t, 0, size)
			d := fs.diskCache
			d.memoryPrefetchMax = 1 << 10
			read := func(offset, n int64) ([]byte, error) {
				vec := &IOVector{FilePath: "object", Policy: ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache,
					Entries: []IOEntry{{Offset: offset, Size: n}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				return append([]byte(nil), vec.Entries[0].Data...), err
			}
			release := func() {}
			switch state {
			case "running":
				gate := &firstReadGateStorage{ObjectStorage: fs.server.raw, started: make(chan struct{}), open: make(chan struct{})}
				fs.server.raw = gate
				release = sync.OnceFunc(func() { close(gate.open) })
				t.Cleanup(release)
				go func() { _, _ = read(0, size/4) }()
				<-gate.started
			case "queued":
				var held <-chan struct{}
				held, release = holdPublisher(t, d)
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Policy: SkipMemoryCache,
					Entries: []IOEntry{{Size: 4096, Data: make([]byte, 4096)}}}))
				<-held
				go func() { _, _ = read(0, size/4) }()
				waitQueued(t, d, 1)
			case "publishing":
				var held <-chan struct{}
				held, release = holdPublisher(t, d)
				go func() { _, _ = read(0, size/4) }()
				<-held
			}
			require.NoError(t, deleteReleasing(release, func() error { return fs.Delete(ctx, "object") }))
			v2 := make([]byte, size)
			for i := range v2 {
				v2[i] = v1[i] + 1
			}
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: size, Data: v2}},
				Policy: SkipDiskCache | SkipMemoryCache}))
			waitObjectsEnd(d, "object")
			d.Flush(ctx)
			for _, r := range [][2]int64{{0, size / 4}, {0, size}, {size / 8, size / 4}} {
				data, err := read(r[0], r[1])
				require.NoError(t, err)
				require.Equal(t, v2[r[0]:r[0]+r[1]], data, "%s: [%d, +%d)", state, r[0], r[1])
			}
		})
	}
}

// A range whose file a delete cannot remove stays indexed, and a later delete removes it.
func TestDeleteKeepsUnremovableRangeIndexed(t *testing.T) {
	ctx := context.Background()
	fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
	d := fs.diskCache
	vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache, Entries: []IOEntry{{Offset: 100, Size: 10}}}
	require.NoError(t, fs.Read(ctx, vec))
	vec.Release()
	d.Flush(ctx)
	key := d.objectKey("object")
	ranges := d.ranges.of(key)
	require.Len(t, ranges, 1)
	path := ranges[0].diskPath
	// A non-empty directory in place of the file cannot be removed as a file.
	require.NoError(t, os.Remove(path))
	require.NoError(t, os.Mkdir(path, 0755))
	child := filepath.Join(path, "child")
	require.NoError(t, os.WriteFile(child, []byte("x"), 0644))
	require.Error(t, fs.Delete(ctx, "object"))
	require.Equal(t, ranges, d.ranges.of(key))
	require.True(t, d.cache.Contains(path))
	require.NoError(t, os.Remove(child))
	require.NoError(t, fs.Delete(ctx, "object"))
	require.Empty(t, d.ranges.of(key))
	require.False(t, d.cache.Contains(path))
	_, err := os.Stat(path)
	require.True(t, os.IsNotExist(err), "%v", err)
}

// A delete waits for room in a full publisher queue, and ends once the publisher drains
// it, as do the submitters waiting for room.
func TestDeleteWithFullPublisherQueue(t *testing.T) {
	ctx := context.Background()
	fs, _, _ := newIndexObjectFS(t, true)
	d := fs.diskCache
	d.pub.jobs = make(chan *publication, 1)
	held, release := holdPublisher(t, d)
	write := func(name string) error {
		return fs.Write(ctx, IOVector{FilePath: name, Policy: SkipMemoryCache,
			Entries: []IOEntry{{Size: 4096, Data: make([]byte, 4096)}}})
	}
	require.NoError(t, write("a"))
	<-held
	require.NoError(t, write("b"))
	waitQueued(t, d, 1)
	written := make(chan error, 1)
	go func() { written <- write("c") }()
	deleted := make(chan error, 1)
	go func() { deleted <- fs.Delete(ctx, "b") }()
	require.Never(t, func() bool { return len(deleted) > 0 }, 50*time.Millisecond, time.Millisecond,
		"the delete waits for the publication of b queued before it")
	release()
	select {
	case err := <-deleted:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the delete did not return")
	}
	require.NoError(t, <-written)
	d.Flush(ctx)
	require.Empty(t, d.ranges.of(d.objectKey("b")))
}
