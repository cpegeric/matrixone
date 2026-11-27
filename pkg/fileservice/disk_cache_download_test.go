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
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	btoml "github.com/BurntSushi/toml"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/logutil"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/buffer"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

// Settings the redesigned cache does not use are accepted from a config and ignored, with
// one log each when a file service starts, and the file service caches as without them.
// The control sets none and logs none.
func TestIgnoredCacheSettingsAreLogged(t *testing.T) {
	for _, set := range []bool{true, false} {
		t.Run(map[bool]string{true: "set", false: "control"}[set], func(t *testing.T) {
			ctx := context.Background()
			text := "disk-capacity = \"64MiB\"\nmemory-capacity = \"1B\"\n"
			if set {
				text += "disk-download-max-extra-wait = \"200ms\"\ndisk-prefetch-object-split-downloads = 3\n"
			}
			var config CacheConfig
			_, err := btoml.Decode(text, &config)
			require.NoError(t, err)
			config.DiskPath = ptrTo(t.TempDir())
			const event = "fileservice: setting ignored"
			core, logs := observer.New(zap.DebugLevel)
			previous := logutil.GetReportZapFunc()
			logutil.SetLogReporter(&logutil.TraceReporter{
				ReportZap: func(enc zapcore.Encoder, entry zapcore.Entry, fields []zapcore.Field) (*buffer.Buffer, error) {
					if entry.Message == event {
						_ = core.Write(entry, fields)
					}
					return previous(enc, entry, fields)
				},
			})
			defer logutil.SetLogReporter(&logutil.TraceReporter{ReportZap: previous})
			fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "ignored", Endpoint: "disk", Bucket: t.TempDir()}, config, nil, false, false)
			require.NoError(t, err)
			defer fs.Close(ctx)
			entries := logs.FilterMessage(event).All()
			settings := make([]string, 0, len(entries))
			for _, e := range entries {
				settings = append(settings, e.ContextMap()["setting"].(string))
			}
			if set {
				require.ElementsMatch(t, []string{"disk-download-max-extra-wait", "disk-prefetch-object-split-downloads"}, settings)
			} else {
				require.Empty(t, settings)
			}
			data := bytes.Repeat([]byte("x"), 4096)
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: 4096, Data: data}}}))
			vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache, Entries: []IOEntry{{Offset: 10, Size: 10}}}
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[10:20], vec.Entries[0].Data)
			vec.Release()
			fs.diskCache.Flush(ctx)
			require.NotEmpty(t, fs.diskCache.ranges.of(fs.diskCache.objectKey("object")), "the disk cache caches")
		})
	}
}

// downloading starts a whole-object fill of "obj" whose source is a pipe the test writes.
func downloading(t *testing.T) (*DiskCache, *io.PipeWriter, chan error) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	t.Cleanup(func() { cache.Close(ctx) })
	pr, pw := io.Pipe()
	filled := make(chan error, 1)
	done := make(chan struct{})
	t.Cleanup(func() {
		_ = pw.CloseWithError(io.ErrClosedPipe)
		<-done
	})
	go func() {
		defer close(done)
		filled <- cache.SetFile(ctx, "obj", func(context.Context) (io.ReadCloser, error) { return pr, nil })
	}()
	require.Eventually(t, func() bool { return cache.download(cache.objectKey("obj")) != nil },
		diskCacheLifecycleTestTimeout, time.Millisecond)
	return cache, pw, filled
}

func readRange(t *testing.T, ctx context.Context, cache *DiskCache, offset, size int64) chan *IOVector {
	t.Helper()
	ctx, cancel := context.WithCancel(ctx)
	result := make(chan *IOVector, 1)
	done := make(chan struct{})
	vec := &IOVector{FilePath: "obj", Entries: []IOEntry{{Offset: offset, Size: size}}}
	t.Cleanup(func() {
		cancel()
		<-done
		vec.Release()
	})
	go func() {
		defer close(done)
		_ = cache.Read(ctx, vec)
		result <- vec
	}()
	return result
}

func waitVector(t *testing.T, done chan *IOVector) *IOVector {
	select {
	case vec := <-done:
		return vec
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("read did not return")
		return nil
	}
}

// A read whose bytes the download never writes is not served from it.
func TestDiskCacheReadFallsBackWhenDownloadFails(t *testing.T) {
	cache, pw, filled := downloading(t)
	_, err := pw.Write([]byte("hello"))
	require.NoError(t, err)

	pending := readRange(t, context.Background(), cache, 6, 5)
	require.NoError(t, pw.CloseWithError(errors.New("source failed")))
	vec := waitVector(t, pending)
	require.False(t, vec.Entries[0].done)
	require.Error(t, <-filled)
}

func TestDiskCacheDownloadCleanupJoinsAbandonedFill(t *testing.T) {
	var filled chan error
	// Establish TempDir's cleanup before registering the witness. downloading
	// creates another directory under the same test-owned parent.
	t.TempDir()
	// The witness runs after downloading's cleanups and before TempDir removal.
	t.Cleanup(func() {
		select {
		case err := <-filled:
			require.ErrorIs(t, err, io.ErrClosedPipe)
		default:
			t.Error("abandoned cache fill outlived its cleanup")
		}
	})
	_, _, filled = downloading(t)
}

// A download that ends with the object shorter than a read leaves the read unserved.
func TestDiskCacheReadBeyondDownloadedObject(t *testing.T) {
	cache, pw, filled := downloading(t)
	pending := readRange(t, context.Background(), cache, 2, 10)
	_, err := pw.Write([]byte("hello"))
	require.NoError(t, err)
	require.NoError(t, pw.Close())
	require.False(t, waitVector(t, pending).Entries[0].done)
	require.NoError(t, <-filled)
}

// A reader holding the download's file reads the written bytes after the file is
// published, and after a failed download's file is removed.
func TestObjectDownloadReadableAfterPublishOrRemoval(t *testing.T) {
	for _, fail := range []bool{false, true} {
		cache, pw, filled := downloading(t)
		download := cache.download(cache.objectKey("obj"))
		_, err := pw.Write([]byte("hello"))
		require.NoError(t, err)
		require.Eventually(t, func() bool { return download.writtenTo(0, 5) },
			diskCacheLifecycleTestTimeout, time.Millisecond)
		f, ok := download.acquire()
		require.True(t, ok)
		if fail {
			require.NoError(t, pw.CloseWithError(errors.New("source failed")))
			require.Error(t, <-filled)
		} else {
			require.NoError(t, pw.Close())
			require.NoError(t, <-filled)
		}
		b := make([]byte, 5)
		_, err = f.ReadAt(b, 0)
		require.NoError(t, err, "failed %v", fail)
		require.Equal(t, []byte("hello"), b)
		download.release()
		_, ok = download.acquire()
		require.False(t, ok, "the file is closed once the last reader is done")
	}
}

// A read waits for a download whose source stops sending until its context ends; the
// download then fails after the idle timeout.
func TestDiskCacheReadLeavesStalledDownload(t *testing.T) {
	idle0 := sourceIdleTimeout
	sourceIdleTimeout = 50 * time.Millisecond
	t.Cleanup(func() { sourceIdleTimeout = idle0 })
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	t.Cleanup(func() { cache.Close(ctx) })
	filled := make(chan error, 1)
	go func() {
		filled <- cache.SetFile(ctx, "obj", func(ctx context.Context) (io.ReadCloser, error) {
			return io.NopCloser(&stalledReader{ctx: ctx, data: []byte("0")}), nil
		})
	}()
	require.Eventually(t, func() bool { return cache.download(cache.objectKey("obj")) != nil },
		diskCacheLifecycleTestTimeout, time.Millisecond)
	readCtx, cancel := context.WithTimeout(ctx, 10*sourceIdleTimeout)
	defer cancel()
	vec := waitVector(t, readRange(t, readCtx, cache, 1, 3))
	require.False(t, vec.Entries[0].done)
	select {
	case err := <-filled:
		require.ErrorIs(t, err, errSourceIdle)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the stalled download did not fail")
	}
}

// stalledReader returns data, then blocks until ctx is done.
type stalledReader struct {
	ctx  context.Context
	data []byte
}

func (r *stalledReader) Read(b []byte) (int, error) {
	if len(r.data) > 0 {
		n := copy(b, r.data)
		r.data = r.data[n:]
		return n, nil
	}
	<-r.ctx.Done()
	return 0, r.ctx.Err()
}

// lateReader returns data once release is closed, then io.EOF.
type lateReader struct {
	ctx     context.Context
	release chan struct{}
	data    []byte
}

func (r *lateReader) Read(b []byte) (int, error) {
	select {
	case <-r.release:
	case <-r.ctx.Done():
		return 0, r.ctx.Err()
	}
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	n := copy(b, r.data)
	r.data = r.data[n:]
	return n, nil
}

// shortIdle sets the source idle timeout for the test.
func shortIdle(t *testing.T, d time.Duration) {
	idle0 := sourceIdleTimeout
	sourceIdleTimeout = d
	t.Cleanup(func() { sourceIdleTimeout = idle0 })
}

// A whole-object download that a read waits for is not stopped by the idle timeout.
func TestDiskCacheWaitedDownloadIsNotTimed(t *testing.T) {
	shortIdle(t, 200*time.Millisecond)
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	t.Cleanup(func() { cache.Close(ctx) })
	release := make(chan struct{})
	filled := make(chan error, 1)
	go func() {
		filled <- cache.SetFile(ctx, "obj", func(ctx context.Context) (io.ReadCloser, error) {
			return io.NopCloser(&lateReader{ctx: ctx, release: release, data: []byte("abcdef")}), nil
		})
	}()
	var o *objectDownload
	require.Eventually(t, func() bool { o = cache.download(cache.objectKey("obj")); return o != nil },
		diskCacheLifecycleTestTimeout, time.Millisecond)
	waited := make(chan bool, 1)
	go func() {
		ctx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
		defer cancel()
		_, ok := o.waitFor(ctx, 0, 3)
		waited <- ok
	}()
	require.Eventually(t, func() bool {
		o.idle.mu.Lock()
		defer o.idle.mu.Unlock()
		return o.idle.waiting == 1
	}, diskCacheLifecycleTestTimeout, time.Millisecond)
	time.AfterFunc(3*sourceIdleTimeout, func() { close(release) })
	require.True(t, <-waited)
	select {
	case err := <-filled:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the download did not end")
	}
	require.True(t, cache.cache.Contains(cache.wholePath("obj", 6)))
}

// Bytes read from a download count toward its file's read share also when the file is
// evicted as soon as it is published: those of a read, of a read still running when the
// download publishes the file, and those the read that started the download captures. A
// read that fails across publication counts none.
func TestDiskCacheDownloadReadsCountWhenEvictedOnPublish(t *testing.T) {
	type readFunc func(t *testing.T, ctx context.Context, cache *DiskCache, pw *io.PipeWriter, filled <-chan error)
	acrossPublication := func(werr error, read func(ctx context.Context, cache *DiskCache, w blockingWriter) error) readFunc {
		return func(t *testing.T, ctx context.Context, cache *DiskCache, pw *io.PipeWriter, filled <-chan error) {
			w := blockingWriter{entered: make(chan struct{}), release: make(chan struct{}), err: werr}
			done, release := startTestBlockedRead(t, pw, w, func() error { return read(ctx, cache, w) })
			select {
			case <-w.entered:
			case <-time.After(diskCacheLifecycleTestTimeout):
				t.Fatal("read did not reach its writer")
			}
			require.NoError(t, pw.Close())
			require.NoError(t, <-filled)
			require.False(t, cache.cache.Contains(cache.wholePath("obj", 4096)), "a file larger than the cache is evicted")
			release()
			if werr == nil {
				require.NoError(t, <-done)
			} else {
				require.Error(t, <-done)
			}
		}
	}
	for _, c := range []struct {
		name string
		// read reads the 4096 bytes written to pw and ends the fill.
		read readFunc
		// want is the group's average share, or -1 for any positive share.
		want float64
	}{
		{"read", func(t *testing.T, ctx context.Context, cache *DiskCache, pw *io.PipeWriter, filled <-chan error) {
			vec := waitVector(t, readRange(t, ctx, cache, 0, 4096))
			require.True(t, vec.Entries[0].done)
			require.NoError(t, pw.Close())
			require.NoError(t, <-filled)
		}, -1},
		{"read across publication", acrossPublication(nil, func(ctx context.Context, cache *DiskCache, w blockingWriter) error {
			vec := &IOVector{FilePath: "obj", Entries: []IOEntry{{Offset: 0, Size: 4096, WriterForRead: w}}}
			defer vec.Release()
			return cache.Read(ctx, vec)
		}), -1},
		{"failed read across publication", acrossPublication(errors.New("injected writer error"), func(ctx context.Context, cache *DiskCache, w blockingWriter) error {
			entry := IOEntry{Offset: 0, Size: 4096, WriterForRead: w}
			_, _, err := cache.readFromDownload(ctx, cache.objectKey("obj"), false, &entry, &IOVector{})
			return err
		}), 0},
		{"capture", func(t *testing.T, ctx context.Context, cache *DiskCache, pw *io.PipeWriter, filled <-chan error) {
			read := newFullObjectDiskCacheReader(&IOVector{FilePath: "obj", Entries: []IOEntry{{Offset: 0, Size: 4096}}})
			download := cache.download(cache.objectKey("obj"))
			file, finish := acquireCapture(t, read, download)
			defer finish(false)
			captured, err := read.capture(ctx, download, file, "obj")
			require.NoError(t, err)
			require.True(t, captured)
			require.NoError(t, pw.Close())
			require.NoError(t, <-filled)
			finish(true)
		}, 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			ctx := context.Background()
			cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1024), nil, false, nil, "")
			require.NoError(t, err)
			t.Cleanup(func() { cache.Close(ctx) })
			const group = 7
			diskPath := cache.objectKey("obj")
			cache.preloads.narrowRead(diskPath, group, func() bool { return true })
			pw, filled := startTestCacheFill(t, ctx, cache)
			require.Eventually(t, func() bool { return cache.download(diskPath) != nil },
				diskCacheLifecycleTestTimeout, time.Millisecond)
			_, err = pw.Write(make([]byte, 4096))
			require.NoError(t, err)
			c.read(t, ctx, cache, pw, filled)

			require.False(t, cache.cache.Contains(cache.wholePath("obj", 4096)), "a file larger than the cache is evicted")
			cache.preloads.mu.Lock()
			defer cache.preloads.mu.Unlock()
			require.NotNil(t, cache.preloads.groups[group])
			switch c.want {
			case -1:
				require.Greater(t, cache.preloads.groups[group].average, 0.0)
			case 0:
				require.Zero(t, cache.preloads.groups[group].average)
			default:
				require.Equal(t, c.want, cache.preloads.groups[group].average)
				require.Equal(t, 1, cache.preloads.groups[group].samples)
			}
		})
	}
}

func startTestCacheFill(t *testing.T, ctx context.Context, cache *DiskCache) (*io.PipeWriter, <-chan error) {
	t.Helper()
	pr, pw := io.Pipe()
	filled := make(chan error, 1)
	done := make(chan struct{})
	t.Cleanup(func() {
		_ = pw.Close()
		select {
		case <-done:
		case <-time.After(diskCacheLifecycleTestTimeout):
			t.Error("cache fill did not stop during cleanup")
		}
	})
	go func() {
		defer close(done)
		filled <- cache.SetFile(ctx, "obj", func(context.Context) (io.ReadCloser, error) { return pr, nil })
	}()
	return pw, filled
}

type blockingWriter struct {
	entered, release chan struct{}
	err              error
}

func (w blockingWriter) Write(p []byte) (int, error) {
	close(w.entered)
	<-w.release
	if w.err != nil {
		return 0, w.err
	}
	return len(p), nil
}

func startTestBlockedRead(t *testing.T, pw *io.PipeWriter, w blockingWriter, fn func() error) (<-chan error, func()) {
	t.Helper()
	read := make(chan error, 1)
	done := make(chan struct{})
	release := sync.OnceFunc(func() { close(w.release) })
	t.Cleanup(func() {
		_ = pw.Close()
		release()
		select {
		case <-done:
		case <-time.After(diskCacheLifecycleTestTimeout):
			t.Error("blocked read did not stop during cleanup")
		}
	})
	go func() {
		defer close(done)
		read <- fn()
	}()
	return read, release
}

func acquireCapture(t *testing.T, read *fullObjectDiskCacheReader, download *objectDownload) (io.ReaderAt, func(bool)) {
	t.Helper()
	file, ok := download.acquire()
	require.True(t, ok)
	released := false
	return file, func(success bool) {
		if released {
			return
		}
		released = true
		defer download.release()
		read.finish(download, success)
	}
}

// download returns the download of the registered fill of the whole object at diskPath,
// of either cache.
func (d *DiskCache) download(diskPath string) *objectDownload {
	p := d.wholeFill(diskPath, false)
	if p == nil {
		p = d.wholeFill(diskPath, true)
	}
	if p == nil {
		return nil
	}
	return p.joinableDownload()
}

// registerWholeFill registers o as the download of a fill of the whole object at diskPath.
func registerWholeFill(d *DiskCache, diskPath string, o *objectDownload) *cacheFill {
	p := d.newContentFill(diskPath)
	d.startDownload(p, o)
	p.started(o)
	return p
}

func newPreloadGeneration(t *testing.T, p *preloadUsage, diskPath string, data []byte) *objectDownload {
	t.Helper()
	tempPath := filepath.Join(t.TempDir(), "download")
	require.NoError(t, os.WriteFile(tempPath, data, 0600))
	o := newObjectDownload(tempPath, nil)
	t.Cleanup(o.retire)
	o.credit = func(n int64) { p.readDownload(diskPath, o, n) }
	p.cachedDownload(diskPath, diskPath, o)
	return o
}

type readShareWriterFunc func([]byte) (int, error)

func (f readShareWriterFunc) Write(p []byte) (int, error) { return f(p) }

func newReadShareDownload(t *testing.T, data []byte, advertised int64) (*DiskCache, *objectDownload, string) {
	t.Helper()
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	t.Cleanup(func() { cache.Close(ctx) })

	diskPath := cache.objectKey("obj")
	o := newPreloadGeneration(t, cache.preloads, diskPath, data)
	registerWholeFill(cache, diskPath, o)
	o.advance(0, advertised)
	o.end(true)
	return cache, o, diskPath
}

// Read share counts only reads that complete successfully, both before and after
// the download's bytes are transferred to the published file.
func TestDiskCacheDownloadReadShareOnlyCountsSuccessfulReads(t *testing.T) {
	for _, stage := range []string{"before_publish", "after_publish"} {
		for _, scenario := range []string{"success", "short_read", "writer_error", "cancel", "conversion_error"} {
			t.Run(stage+"/"+scenario, func(t *testing.T) {
				data := []byte("01234567")
				if scenario == "short_read" {
					data = []byte("012")
				}
				cache, o, diskPath := newReadShareDownload(t, data, 8)
				ctx := context.Background()
				entry := IOEntry{Size: 8}
				switch scenario {
				case "writer_error":
					entry.WriterForRead = readShareWriterFunc(func([]byte) (int, error) {
						return 0, errors.New("injected writer error")
					})
				case "cancel":
					var cancel context.CancelFunc
					ctx, cancel = context.WithCancel(ctx)
					defer cancel()
					entry.WriterForRead = readShareWriterFunc(func([]byte) (int, error) {
						cancel()
						return 0, ctx.Err()
					})
				case "conversion_error":
					entry.ToCacheData = func(context.Context, io.Reader, []byte, CacheDataAllocator) (fscache.Data, error) {
						return nil, errors.New("injected conversion error")
					}
				}

				if stage == "after_publish" {
					cache.preloads.readDownload(diskPath, o, o.takeServed())
				}
				downloading, filled, err := cache.readFromDownload(ctx, diskPath, false, &entry, &IOVector{})
				vector := IOVector{Entries: []IOEntry{entry}}
				defer vector.Release()
				require.True(t, downloading)
				wantShare := int64(0)
				if scenario == "success" {
					require.NoError(t, err)
					require.True(t, filled)
					require.True(t, entry.done)
					wantShare = 8
				} else {
					require.Error(t, err)
					require.False(t, filled)
				}
				if stage == "before_publish" {
					cache.preloads.readDownload(diskPath, o, o.takeServed())
				}
				cache.preloads.mu.Lock()
				share, ok := cache.preloads.served[diskPath]
				cache.preloads.mu.Unlock()
				require.True(t, ok)
				require.Equal(t, wantShare, share)
			})
		}
	}
}

// Eviction after finishDownload still waits for an acquired reader. The final
// sample includes that reader only when its read succeeds.
func TestDiskCacheReadShareRemovalAfterDownloadFinishWaitsForReader(t *testing.T) {
	cache, o, diskPath := newReadShareDownload(t, []byte("01234567"), 8)
	p := cache.preloads
	p.narrowRead(diskPath, 7, func() bool { return true })
	p.readDownload(diskPath, o, o.takeServed())
	_, ok := o.acquire()
	require.True(t, ok)
	release := sync.OnceFunc(o.release)
	t.Cleanup(release)
	o.retire() // finishDownload removes the public map entry while the reader stays alive.

	p.removed(diskPath, 8)
	p.mu.Lock()
	pending, group := len(p.pending), p.groups[7]
	p.mu.Unlock()
	require.Equal(t, 1, pending)
	require.NotNil(t, group)
	require.Zero(t, group.samples, "the pending read has not produced a sample")

	o.serve(8)
	release()
	p.mu.Lock()
	defer p.mu.Unlock()
	require.Empty(t, p.pending)
	require.NotNil(t, p.groups[7])
	require.Equal(t, 1.0, p.groups[7].average)
	require.Equal(t, 1, p.groups[7].samples)
}

// A delayed old-generation eviction cannot delete or charge the new generation
// that reused the same cache path.
func TestDiskCacheReadShareRemovalDoesNotTouchReusedPath(t *testing.T) {
	cache, old, diskPath := newReadShareDownload(t, []byte("01234567"), 8)
	p := cache.preloads
	p.narrowRead(diskPath, 7, func() bool { return true })
	p.readDownload(diskPath, old, old.takeServed())
	_, ok := old.acquire()
	require.True(t, ok)
	release := sync.OnceFunc(old.release)
	t.Cleanup(release)
	old.retire()
	p.removed(diskPath, 8)

	current := newPreloadGeneration(t, p, diskPath, []byte("01234567"))
	p.narrowRead(diskPath, 7, func() bool { return true })
	p.readDownload(diskPath, current, current.takeServed())
	current.serve(4)
	old.serve(8)
	release()

	p.mu.Lock()
	served, currentDownload := p.served[diskPath], p.downloads[diskPath]
	average, samples := p.groups[7].average, p.groups[7].samples
	p.mu.Unlock()
	require.Equal(t, int64(4), served)
	require.Same(t, current, currentDownload)
	require.Equal(t, 1.0, average)
	require.Equal(t, 1, samples)

	// A late credit from the old object cannot mutate the current generation.
	p.readDownload(diskPath, old, 100)
	p.mu.Lock()
	served = p.served[diskPath]
	p.mu.Unlock()
	require.Equal(t, int64(4), served)

	p.removed(diskPath, 8)
	p.mu.Lock()
	defer p.mu.Unlock()
	require.Empty(t, p.pending)
	require.NotContains(t, p.served, diskPath)
	require.Equal(t, 2, p.groups[7].samples)
	require.Greater(t, p.groups[7].average, 0.0)
	require.Less(t, p.groups[7].average, 1.0)
}

func TestDiskCacheReadShareRemovalSurvivesLaterDelete(t *testing.T) {
	cache, old, diskPath := newReadShareDownload(t, []byte("01234567"), 8)
	p := cache.preloads
	p.narrowRead(diskPath, 7, func() bool { return true })
	p.readDownload(diskPath, old, old.takeServed())
	_, ok := old.acquire()
	require.True(t, ok)
	release := sync.OnceFunc(old.release)
	t.Cleanup(release)
	p.removed(diskPath, 8)

	newPreloadGeneration(t, p, diskPath, []byte("abcdefgh"))
	p.narrowRead(diskPath, 8, func() bool { return true })
	p.deleted(diskPath)
	p.removed(diskPath, 8)
	old.serve(8)
	release()

	p.mu.Lock()
	oldGroup, newGroup, pending := p.groups[7], p.groups[8], len(p.pending)
	p.mu.Unlock()
	require.Zero(t, pending)
	require.NotNil(t, oldGroup)
	require.Equal(t, 1, oldGroup.samples)
	require.Equal(t, 1.0, oldGroup.average)
	require.Nil(t, newGroup, "explicit deletion is not an eviction sample")
}

// Full-object captures charge only after fillVector succeeds.
func TestFullObjectDiskCacheReaderReadShareWaitsForRequestOutcome(t *testing.T) {
	for _, scenario := range []string{"success", "conversion_error"} {
		t.Run(scenario, func(t *testing.T) {
			cache, o, diskPath := newReadShareDownload(t, []byte("01234567"), 8)
			cache.preloads.readDownload(diskPath, o, o.takeServed())
			entry := IOEntry{Size: 8}
			if scenario == "conversion_error" {
				entry.ToCacheData = func(context.Context, io.Reader, []byte, CacheDataAllocator) (fscache.Data, error) {
					return nil, errors.New("injected conversion error")
				}
			}
			read := newFullObjectDiskCacheReader(&IOVector{FilePath: "obj", Entries: []IOEntry{entry}})
			file, finish := acquireCapture(t, read, o)
			defer finish(false)
			ctx := context.Background()
			captured, err := read.capture(ctx, o, file, "obj")
			require.NoError(t, err)
			require.True(t, captured)

			success := false
			switch scenario {
			case "success":
				require.NoError(t, read.fillVector(ctx, cache.cacheDataAllocator, "obj"))
				success = true
			case "conversion_error":
				require.Error(t, read.fillVector(ctx, cache.cacheDataAllocator, "obj"))
			}
			finish(success)

			cache.preloads.mu.Lock()
			share := cache.preloads.served[diskPath]
			cache.preloads.mu.Unlock()
			if success {
				require.Equal(t, int64(8), share)
			} else {
				require.Zero(t, share)
			}
		})
	}
}

// A request admitted before retirement keeps its file and credits all of its
// ranges before releasing the last reader.
func TestFullObjectDiskCacheReaderCapturesAfterRetirement(t *testing.T) {
	cache, o, diskPath := newReadShareDownload(t, []byte("01234567"), 8)
	cache.preloads.readDownload(diskPath, o, o.takeServed())
	vector := &IOVector{FilePath: "obj", Entries: []IOEntry{
		{Offset: 1, Size: 3},
		{Offset: 5, Size: -1},
		{Offset: 0, Size: 2},
	}}
	read := newFullObjectDiskCacheReader(vector)
	file, finish := acquireCapture(t, read, o)
	defer finish(false)
	o.retire()
	_, ok := o.acquire()
	require.False(t, ok, "retirement must reject a new reader")

	ctx := context.Background()
	for i := 0; i < 2; i++ {
		captured, err := read.capture(ctx, o, file, "obj")
		require.NoError(t, err)
		require.True(t, captured)
	}
	require.NoError(t, read.fillVector(ctx, cache.cacheDataAllocator, "obj"))
	require.Equal(t, []byte("123"), vector.Entries[0].Data)
	require.Equal(t, []byte("567"), vector.Entries[1].Data)
	require.Equal(t, []byte("01"), vector.Entries[2].Data)
	read.finish(o, true)
	read.finish(o, true)
	finish(true)

	cache.preloads.mu.Lock()
	served := cache.preloads.served[diskPath]
	cache.preloads.mu.Unlock()
	require.Equal(t, int64(8), served)
	o.mu.Lock()
	closed := o.file == nil
	o.mu.Unlock()
	require.True(t, closed, "the last release must close the retired file")
}

type failingCaptureSource struct {
	err      error
	onFailed func()
}

func (s failingCaptureSource) ReadAt(p []byte, offset int64) (int, error) {
	if offset == 0 {
		return copy(p, "ab"), nil
	}
	if s.onFailed != nil {
		s.onFailed()
	}
	return copy(p, "cde"), s.err
}

// A capture that reads fewer bytes than its entries need, or whose file read fails,
// captures nothing and is not charged; cancellation takes precedence over the read error.
func TestDiskCacheDownloadCaptureReportsUnreadableFile(t *testing.T) {
	for _, tc := range []struct {
		name       string
		data       string
		advertised int64
		entries    []IOEntry
		failing    bool
		cancel     bool
	}{
		{name: "short fixed range", advertised: 1, entries: []IOEntry{{Size: 1}}},
		{name: "short read to end", data: "abc", advertised: 4, entries: []IOEntry{{Size: -1}}},
		{name: "read error", data: "abcdef", advertised: 6, entries: []IOEntry{{Size: 2}, {Offset: 2, Size: -1}}, failing: true},
		{name: "cancellation takes precedence", data: "abcdef", advertised: 6, entries: []IOEntry{{Size: 2}, {Offset: 2, Size: -1}}, failing: true, cancel: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, o, _ := newReadShareDownload(t, []byte(tc.data), tc.advertised)
			vector := &IOVector{FilePath: "obj", Entries: tc.entries}
			read := newFullObjectDiskCacheReader(vector)
			file, finish := acquireCapture(t, read, o)
			defer finish(false)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var want error = errCaptureUnreadable
			if tc.failing {
				source := failingCaptureSource{err: errors.New("injected file read error")}
				if tc.cancel {
					source.onFailed = cancel
					want = context.Canceled
				}
				file = source
			}
			captured, err := read.capture(ctx, o, file, "obj")
			require.False(t, captured)
			require.ErrorIs(t, err, want)
			for _, e := range vector.Entries {
				require.Empty(t, e.Data)
			}
			finish(false)
			require.Zero(t, o.takeServed())
		})
	}
}

// A download the disk cache keeps for its cached file holds no reader of its source once
// the fill ends, so the bytes it was filled from are not kept.
func TestDiskCacheDownloadKeepsNoSourceReader(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	diskPath := cache.wholePath("obj", 4096)
	data := make([]byte, 4096)
	require.NoError(t, cache.fillFile(ctx, "obj", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(data)), nil
	}, false, contentFill, nil))
	require.True(t, cache.cache.Contains(diskPath))
	cache.preloads.mu.Lock()
	o := cache.preloads.downloads[diskPath]
	cache.preloads.mu.Unlock()
	require.NotNil(t, o, "the cached file keeps its download")
	o.idle.mu.Lock()
	defer o.idle.mu.Unlock()
	require.Empty(t, o.idle.readers)
}

// An asynchronous update writes its entry's file before it returns, so the caller's bytes
// are not kept for its publication; an entry read before its object was deleted is not
// published.
func TestDiskCacheAsyncUpdateWritesBeforePublication(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	data := []byte("0123456789")
	require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "obj", Entries: []IOEntry{{Offset: 0, Size: 10, Data: data}}}, true))
	clear(data)
	cache.Flush(ctx)
	vec := &IOVector{FilePath: "obj", Entries: []IOEntry{{Offset: 0, Size: 10}}}
	require.NoError(t, cache.Read(ctx, vec))
	require.True(t, vec.Entries[0].done)
	require.Equal(t, []byte("0123456789"), vec.Entries[0].Data)
	vec.Release()

	mark := cache.readDeletes()
	require.NoError(t, cache.DeletePaths(ctx, []string{"gone"}))
	require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "gone", Entries: []IOEntry{{Offset: 0, Size: 10, Data: []byte("0123456789"), readDeletes: mark}}}, true))
	cache.Flush(ctx)
	vec = &IOVector{FilePath: "gone", Entries: []IOEntry{{Offset: 0, Size: 10}}}
	require.NoError(t, cache.Read(ctx, vec))
	require.False(t, vec.Entries[0].done, "an entry read before the delete is not cached")
}
