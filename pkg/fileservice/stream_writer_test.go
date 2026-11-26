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
	"io"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// heldTailStorage returns the first head bytes of a read at once and the rest once
// proceed is closed.
type heldTailStorage struct {
	ObjectStorage
	head    int
	proceed chan struct{}
}

func (s *heldTailStorage) Read(ctx context.Context, path string, min, max *int64) (io.ReadCloser, error) {
	r, err := s.ObjectStorage.Read(ctx, path, min, max)
	if err != nil {
		return nil, err
	}
	return &heldTailReader{ReadCloser: r, left: s.head, proceed: s.proceed, ctx: ctx}, nil
}

type heldTailReader struct {
	io.ReadCloser
	left    int
	proceed chan struct{}
	ctx     context.Context
}

func (r *heldTailReader) Read(p []byte) (int, error) {
	if r.left <= 0 {
		select {
		case <-r.proceed:
		case <-r.ctx.Done():
			return 0, r.ctx.Err()
		}
		return r.ReadCloser.Read(p)
	}
	n, err := r.ReadCloser.Read(p[:min(len(p), r.left)])
	r.left -= n
	return n, err
}

// firstWriteWriter signals first once it is written to.
type firstWriteWriter struct {
	mu    sync.Mutex
	buf   bytes.Buffer
	once  sync.Once
	first chan struct{}
}

func (w *firstWriteWriter) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.first) })
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(p)
}

func (w *firstWriteWriter) bytes() []byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return bytes.Clone(w.buf.Bytes())
}

func TestStreamedRangeReadWritesBeforeRangeArrives(t *testing.T) {
	const size = minDiskPrefetchMemoryMax + 1<<20
	for _, tc := range []struct {
		name   string
		writer bool
	}{
		{name: "read_closer"},
		{name: "writer_for_read", writer: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, size)
			fs.diskCache.memoryPrefetchMax = minDiskPrefetchMemoryMax
			<-fs.diskCache.load.done
			storage := &heldTailStorage{ObjectStorage: fs.server.raw, head: 64 << 10, proceed: make(chan struct{})}
			fs.server.raw = storage
			release := sync.OnceFunc(func() { close(storage.proceed) })
			defer release()
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()

			w := &firstWriteWriter{first: make(chan struct{})}
			var rc io.ReadCloser
			entry := IOEntry{Offset: 0, Size: size}
			if tc.writer {
				entry.WriterForRead = w
			} else {
				entry.ReadCloserForRead = &rc
			}
			vector := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache, Entries: []IOEntry{entry}}
			done := make(chan error, 1)
			copied := make(chan error, 1)
			go func() {
				err := fs.Read(ctx, vector)
				if err == nil && rc != nil {
					go func() {
						defer rc.Close()
						_, err := io.Copy(w, rc)
						copied <- err
					}()
				} else {
					copied <- nil
				}
				done <- err
			}()
			if tc.writer {
				// The range is read before the writer runs.
				require.Never(t, func() bool {
					select {
					case <-w.first:
						return true
					default:
						return false
					}
				}, 100*time.Millisecond, time.Millisecond)
			} else {
				select {
				case <-w.first:
				case <-time.After(10 * time.Second):
					t.Fatal("the streamed entry got no bytes before the rest of its range arrived")
				}
			}
			release()
			require.NoError(t, <-done)
			require.NoError(t, <-copied)
			require.Equal(t, data, w.bytes())
		})
	}
}

// openProbeStreams opens n streams of size bytes each of the object, from offset 0 on, and
// returns them open.
func openProbeStreams(t *testing.T, fs *S3FS, n int, size int64) []io.ReadCloser {
	streams := make([]io.ReadCloser, n)
	for i := range streams {
		var rc io.ReadCloser
		vec := &IOVector{FilePath: "object", Policy: SkipAllCache,
			Entries: []IOEntry{{Offset: int64(i) * size, Size: size, ReadCloserForRead: &rc}}}
		require.NoError(t, fs.Read(context.Background(), vec))
		require.NotNil(t, rc)
		streams[i] = rc
	}
	return streams
}

// A caller that keeps open streams whose buffers fill the memory budget and then reads a
// column range whose flight needs a buffer gets its bytes promptly. Control: the same read
// with no stream open returns, and the streams read their bytes.
func TestStreamProbeOpenStreamsDoNotBlockFlights(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	const size = 1 << 10
	fs.server.memoryLimit = 2 * flightBufSize(size)

	control := PrefetchRange{Offset: 200 << 10, Size: size}
	got, err := probeRead(fs, control)
	require.NoError(t, err)
	require.Equal(t, data[control.Offset:control.Offset+control.Size], got)

	streams := openProbeStreams(t, fs, 2, size)
	defer func() {
		for _, s := range streams {
			_ = s.Close()
		}
	}()
	r := PrefetchRange{Offset: 250 << 10, Size: size}
	start := time.Now()
	got, err = probeRead(fs, r)
	require.NoError(t, err, "a read waited for the buffers of streams its caller holds open")
	require.Less(t, time.Since(start), time.Second)
	require.Equal(t, data[r.Offset:r.Offset+r.Size], got)

	for i, s := range streams {
		b, err := io.ReadAll(s)
		require.NoError(t, err)
		require.Equal(t, data[i*size:(i+1)*size], b)
	}
}

func probeStreamBytes(fs *S3FS) int64 { return heldMemory(fs.diskCache) }

// Streams take buffers only while their bytes fit the memory budget when they open; the
// others read into their temporary files, and every stream reads its bytes. Control: once
// they are closed, a stream takes a buffer again.
func TestStreamProbeStreamBytesWithinBound(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	const size = 1 << 10
	buf := flightBufSize(size)
	bound := 2 * buf
	fs.server.memoryLimit = bound

	streams := openProbeStreams(t, fs, 6, size)
	require.Equal(t, bound, probeStreamBytes(fs), "the streams that fit take buffers")
	memory := 0
	for _, s := range streams {
		if s.(*streamReader).f.buf != nil {
			memory++
		}
	}
	require.Equal(t, 2, memory)
	for i, s := range streams {
		b, err := io.ReadAll(s)
		require.NoError(t, err)
		require.Equal(t, data[i*size:(i+1)*size], b)
		require.LessOrEqual(t, probeStreamBytes(fs), bound)
		require.NoError(t, s.Close())
	}
	require.Zero(t, probeStreamBytes(fs))

	again := openProbeStreams(t, fs, 1, size)
	require.Equal(t, buf, probeStreamBytes(fs))
	require.NoError(t, again[0].Close())
	require.Zero(t, probeStreamBytes(fs))
}
