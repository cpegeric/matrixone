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
	"iter"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aliyun/aliyun-oss-go-sdk/oss"
	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/util/toml"
	"github.com/stretchr/testify/require"
)

// occupied is the room of each server that occupy holds.
var occupied sync.Map

func occupiedRoom(g *objectStorageServer) *atomic.Int64 {
	n, _ := occupied.LoadOrStore(g, new(atomic.Int64))
	return n.(*atomic.Int64)
}

// occupy holds n room of class c of g until the returned function frees it.
func occupy(t *testing.T, g *objectStorageServer, c requestClass, n int) func() {
	t.Helper()
	for range n {
		require.NoError(t, waitRoom(g, context.Background(), c))
		occupiedRoom(g).Add(1)
	}
	return sync.OnceFunc(func() {
		for range n {
			g.done(c)
			occupiedRoom(g).Add(-1)
		}
	})
}

// limitRoom holds room of g until the test ends or the returned function frees it, so
// that n of it is free, at most background of it for background requests.
func limitRoom(t *testing.T, g *objectStorageServer, n, background int) func() {
	t.Helper()
	freeBackground := occupy(t, g, backgroundRequest, g.backgroundMax-background)
	freeForeground := occupy(t, g, foregroundRequest, g.capacity-g.backgroundMax-(n-background))
	free := sync.OnceFunc(func() {
		freeForeground()
		freeBackground()
	})
	t.Cleanup(free)
	return free
}

// waitingFor returns whether n requests of class c wait for room of g.
func waitingFor(g *objectStorageServer, c requestClass, n int) func() bool {
	return func() bool {
		_, waiting := g.counts()
		return waiting[c] == n
	}
}

// heldRoom returns the room of g requests hold, beyond the room occupy holds.
func heldRoom(g *objectStorageServer) int {
	running, _ := g.counts()
	return running[foregroundRequest] + running[backgroundRequest] - int(occupiedRoom(g).Load())
}

// openerStorage is object storage whose reads open their range through its function.
type openerStorage struct {
	ObjectStorage
	open func(ctx context.Context, start, end int64) (io.ReadCloser, error)
}

func (o openerStorage) Read(ctx context.Context, _ string, min, max *int64) (io.ReadCloser, error) {
	end := int64(-1)
	if max != nil {
		end = *max
	}
	return o.open(ctx, *min, end)
}

// openerSource returns the source of an object whose ranges open through open.
func openerSource(open func(ctx context.Context, start, end int64) (io.ReadCloser, error)) objectSource {
	return objectSource{g: newObjectStorageServer(openerStorage{open: open}, 1024), key: "object", end: -1}
}

// admitted reports whether a wait of class c on g is admitted within a short time.
func admitted(g *objectStorageServer, c requestClass) (func(), bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if waitRoom(g, ctx, c) != nil {
		return nil, false
	}
	return func() { g.done(c) }, true
}

// Waiting requests are admitted foreground first, then background, each in arrival order.
func TestServerOrder(t *testing.T) {
	g := newObjectStorageServer(nil, minRequestRoom)
	defer occupy(t, g, foregroundRequest, minRequestRoom-1)()
	free := occupy(t, g, foregroundRequest, 1)
	var mu sync.Mutex
	var order []string
	var wg sync.WaitGroup
	start := func(name string, c requestClass, waiting int) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			require.NoError(t, waitRoom(g, context.Background(), c))
			mu.Lock()
			order = append(order, name)
			mu.Unlock()
			g.done(c)
		}()
		require.Eventually(t, waitingFor(g, c, waiting), 5*time.Second, time.Millisecond)
	}
	start("b1", backgroundRequest, 1)
	start("f1", foregroundRequest, 1)
	start("b2", backgroundRequest, 2)
	start("f2", foregroundRequest, 2)
	free()
	wg.Wait()
	require.Equal(t, []string{"f1", "f2", "b1", "b2"}, order)
	require.Zero(t, heldRoom(g))
}

// A background request arriving while foreground waits is not admitted before it.
func TestServerBackgroundWaitsBehindForeground(t *testing.T) {
	g := newObjectStorageServer(nil, minRequestRoom)
	free := occupy(t, g, foregroundRequest, minRequestRoom)
	done := make(chan struct{})
	go func() {
		defer close(done)
		require.NoError(t, waitRoom(g, context.Background(), foregroundRequest))
		g.done(foregroundRequest)
	}()
	require.Eventually(t, waitingFor(g, foregroundRequest, 1), 5*time.Second, time.Millisecond)
	_, ok := admitted(g, backgroundRequest)
	require.False(t, ok)
	free()
	<-done
	require.Zero(t, heldRoom(g))
}

// Background requests run at most half of the room; foreground may use the rest.
func TestServerBackgroundCap(t *testing.T) {
	g := newObjectStorageServer(nil, minRequestRoom)
	free := occupy(t, g, backgroundRequest, minRequestRoom/2)
	_, ok := admitted(g, backgroundRequest)
	require.False(t, ok, "background holds its half")
	freeForeground := occupy(t, g, foregroundRequest, minRequestRoom/2)
	_, ok = admitted(g, foregroundRequest)
	require.False(t, ok, "the room is full")
	freeForeground()
	free()
	require.Zero(t, heldRoom(g))
}

// A room below minRequestRoom is raised to it, half of it for background requests.
func TestServerRoomAtLeastMinimum(t *testing.T) {
	for _, room := range []int{-1, 0, 1, 2, minRequestRoom - 1, minRequestRoom} {
		g := newObjectStorageServer(nil, room)
		require.Equal(t, minRequestRoom, g.capacity, "room %d", room)
		require.Equal(t, minRequestRoom/2, g.backgroundMax, "room %d", room)
	}
	g := newObjectStorageServer(nil, 3*minRequestRoom+1)
	require.Equal(t, 3*minRequestRoom+1, g.capacity)
	require.Equal(t, (3*minRequestRoom+1)/2, g.backgroundMax)
}

// A running request is never stopped: a foreground request waits for a background one to
// end.
func TestServerNoPreemption(t *testing.T) {
	g := newObjectStorageServer(nil, minRequestRoom)
	free := occupy(t, g, backgroundRequest, minRequestRoom/2)
	freeForeground := occupy(t, g, foregroundRequest, minRequestRoom/2)
	defer freeForeground()
	_, ok := admitted(g, foregroundRequest)
	require.False(t, ok)
	running, _ := g.counts()
	require.Equal(t, minRequestRoom/2, running[backgroundRequest])
	free()
	release, ok := admitted(g, foregroundRequest)
	require.True(t, ok)
	release()
}

// A wait canceled while waiting, or racing its admission, leaves no room held.
func TestServerCancel(t *testing.T) {
	g := newObjectStorageServer(nil, minRequestRoom)
	free := occupy(t, g, foregroundRequest, minRequestRoom)
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- waitRoom(g, ctx, foregroundRequest) }()
	require.Eventually(t, waitingFor(g, foregroundRequest, 1), 5*time.Second, time.Millisecond)
	cancel()
	require.ErrorIs(t, <-errCh, context.Canceled)
	free()
	require.Zero(t, heldRoom(g))
	_, waiting := g.counts()
	require.Zero(t, waiting[foregroundRequest])

	for range 200 {
		free := occupy(t, g, foregroundRequest, minRequestRoom)
		ctx, cancel := context.WithCancel(context.Background())
		go func() { errCh <- waitRoom(g, ctx, foregroundRequest) }()
		require.Eventually(t, waitingFor(g, foregroundRequest, 1), 5*time.Second, time.Microsecond)
		go free()
		cancel()
		if err := <-errCh; err == nil {
			g.done(foregroundRequest)
		}
		require.Eventually(t, func() bool { return heldRoom(g) == 0 }, 5*time.Second, time.Microsecond)
	}
}

type panicStorage struct{ ObjectStorage }

func (panicStorage) Stat(context.Context, string) (int64, error) { panic("stat") }

// A request whose call panics frees its room.
func TestServerPanicFreesRoom(t *testing.T) {
	g := newObjectStorageServer(panicStorage{}, minRequestRoom)
	defer occupy(t, g, foregroundRequest, minRequestRoom-1)()
	require.Panics(t, func() { _, _ = g.Do(context.Background(), statRequest{key: "key"}) })
	require.Zero(t, heldRoom(g))
	release, ok := admitted(g, foregroundRequest)
	require.True(t, ok)
	release()
}

// Close ends the waits for room; running requests keep theirs until they end.
func TestServerCloseWithWaiters(t *testing.T) {
	g := newObjectStorageServer(nil, minRequestRoom)
	free := occupy(t, g, foregroundRequest, minRequestRoom)
	errCh := make(chan error, 2)
	for _, c := range []requestClass{foregroundRequest, backgroundRequest} {
		go func() { errCh <- waitRoom(g, context.Background(), c) }()
		require.Eventually(t, waitingFor(g, c, 1), 5*time.Second, time.Millisecond)
	}
	g.close()
	require.ErrorIs(t, <-errCh, errServerClosed)
	require.ErrorIs(t, <-errCh, errServerClosed)
	running, _ := g.counts()
	require.Equal(t, minRequestRoom, running[foregroundRequest])
	free()
	require.Zero(t, heldRoom(g))
	require.ErrorIs(t, waitRoom(g, context.Background(), foregroundRequest), errServerClosed)
}

type retryStorage struct {
	ObjectStorage
	g     *objectStorageServer
	held  []int
	fails int
}

func (r *retryStorage) Stat(ctx context.Context, key string) (int64, error) {
	return DoWithRetryContext(ctx, "stat", func() (int64, error) {
		r.held = append(r.held, heldRoom(r.g))
		if len(r.held) <= r.fails {
			return 0, io.ErrUnexpectedEOF
		}
		return 1, nil
	}, maxRetryAttemps, IsRetryableError)
}

// A retry waits with no room held, and its next attempt holds room again: a request
// waiting behind it runs during the wait.
func TestServerRetryWaitsWithoutRoom(t *testing.T) {
	s := &retryStorage{fails: 2}
	g := newObjectStorageServer(s, minRequestRoom)
	s.g = g
	ctx := context.Background()
	freeOne := occupy(t, g, foregroundRequest, 1)
	defer occupy(t, g, foregroundRequest, minRequestRoom-1)()
	type statResult struct {
		size int64
		err  error
	}
	statDone := make(chan statResult, 1)
	go func() {
		res, err := g.Do(ctx, statRequest{key: "key"})
		statDone <- statResult{res.n, err}
	}()
	require.Eventually(t, waitingFor(g, foregroundRequest, 1), 5*time.Second, time.Millisecond)
	attemptsBefore := make(chan int, 1)
	go func() {
		if err := waitRoom(g, ctx, foregroundRequest); err != nil {
			attemptsBefore <- -1
			return
		}
		attemptsBefore <- len(s.held)
		g.done(foregroundRequest)
	}()
	require.Eventually(t, waitingFor(g, foregroundRequest, 2), 5*time.Second, time.Millisecond)
	freeOne()
	r := <-statDone
	require.NoError(t, r.err)
	require.Equal(t, int64(1), r.size)
	require.Equal(t, 1, <-attemptsBefore, "the request behind runs during the first retry's wait")
	require.Equal(t, []int{1, 1, 1}, s.held, "each attempt holds room")
	require.Zero(t, heldRoom(g))
}

// readFlight reads the range of fl through open.
func readFlight(fl *memoryPrefetchObject, open func(ctx context.Context, start, end int64) (io.ReadCloser, error)) error {
	fl.src = openerSource(open)
	return fl.read(context.Background())
}

// getterStorage is object storage whose reads open their range through its function.
type getterStorage struct {
	ObjectStorage
	get func(ctx context.Context, min, max *int64) (io.ReadCloser, error)
}

func (g getterStorage) Read(ctx context.Context, _ string, min, max *int64) (io.ReadCloser, error) {
	return g.get(ctx, min, max)
}

// getterSource returns the source of a whole object whose ranges open through get.
func getterSource(get func(ctx context.Context, min, max *int64) (io.ReadCloser, error)) objectSource {
	return objectSource{g: newObjectStorageServer(getterStorage{get: get}, 1024), key: "object", end: -1}
}

// No request type has a function-typed field.
func TestServerRequestsHoldNoFunctions(t *testing.T) {
	for _, req := range requestTypes {
		typ := reflect.TypeOf(req)
		for i := range typ.NumField() {
			require.NotEqual(t, reflect.Func, typ.Field(i).Type.Kind(), "%s.%s", typ.Name(), typ.Field(i).Name)
		}
	}
}

// finishes runs f, failing the test when it does not return within 20s.
func finishes(t *testing.T, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		f()
	}()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("hold and wait: the call did not return")
	}
}

// openCountingStorage counts the requests open on its upstream at once: a read's from its
// call until its body ends or closes, any other request during its call. Each request
// takes delay first, and a read of a key gated accepts waits until gate is closed.
type openCountingStorage struct {
	ObjectStorage
	delay time.Duration
	gate  chan struct{}
	gated func(key string) bool

	cur, most, total atomic.Int64
	stats            atomic.Int64
}

func (o *openCountingStorage) open() func() {
	n := o.cur.Add(1)
	o.total.Add(1)
	for most := o.most.Load(); n > most && !o.most.CompareAndSwap(most, n); most = o.most.Load() {
	}
	time.Sleep(o.delay)
	return sync.OnceFunc(func() { o.cur.Add(-1) })
}

func (o *openCountingStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	closed := o.open()
	if o.gated != nil && o.gated(key) {
		select {
		case <-o.gate:
		case <-ctx.Done():
			closed()
			return nil, ctx.Err()
		}
	}
	r, err := o.ObjectStorage.Read(ctx, key, min, max)
	if err != nil {
		closed()
		return nil, err
	}
	return &readCloser{
		r: readerFunc(func(b []byte) (int, error) {
			n, err := r.Read(b)
			if err != nil {
				closed()
			}
			return n, err
		}),
		closeFunc: func() error {
			defer closed()
			return r.Close()
		},
	}, nil
}

func (o *openCountingStorage) Write(ctx context.Context, key string, r io.Reader, size *int64, expire *time.Time) error {
	defer o.open()()
	return o.ObjectStorage.Write(ctx, key, r, size, expire)
}

func (o *openCountingStorage) Stat(ctx context.Context, key string) (int64, error) {
	o.stats.Add(1)
	defer o.open()()
	return o.ObjectStorage.Stat(ctx, key)
}

func (o *openCountingStorage) Exists(ctx context.Context, key string) (bool, error) {
	defer o.open()()
	return o.ObjectStorage.Exists(ctx, key)
}

func (o *openCountingStorage) Delete(ctx context.Context, keys ...string) error {
	defer o.open()()
	return o.ObjectStorage.Delete(ctx, keys...)
}

func (o *openCountingStorage) List(ctx context.Context, prefix string) iter.Seq2[*DirEntry, error] {
	return func(yield func(*DirEntry, error) bool) {
		closed := o.open()
		defer func() { closed() }()
		for e, err := range o.ObjectStorage.List(ctx, prefix) {
			closed()
			if !yield(e, err) {
				return
			}
			closed = o.open()
		}
	}
}

// newRoomFS returns an S3FS on a disk object storage whose server has room free for
// room requests, at most half of it, or one, for background requests, the rest held until
// the test ends, and objects of size bytes written.
func newRoomFS(t *testing.T, room int, objects []string, size int) (*S3FS, []byte) {
	ctx := context.Background()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "s3", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{
		DiskPath: ptrTo(t.TempDir()), DiskCapacity: ptrTo[toml.ByteSize](256 << 20),
		MemoryCapacity: ptrTo[toml.ByteSize](0),
	}, nil, false, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(ctx) })
	<-fs.diskCache.load.done
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i*7 + i/1021)
	}
	for _, name := range objects {
		require.NoError(t, fs.Write(ctx, IOVector{FilePath: name, Entries: []IOEntry{{Size: int64(len(data)), Data: data}}, Policy: SkipAllCache}))
	}
	limitRoom(t, fs.server, room, max(room/2, 1))
	return fs, data
}

// newGovernedFS is newRoomFS whose requests are counted.
func newGovernedFS(t *testing.T, room int, objects []string, size int) (*S3FS, *openCountingStorage, []byte) {
	fs, data := newRoomFS(t, room, objects, size)
	counting := &openCountingStorage{ObjectStorage: fs.server.raw, gate: make(chan struct{})}
	fs.server.raw = counting
	return fs, counting, data
}

func names(prefix string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("%s-%d", prefix, i)
	}
	return out
}

// openStream returns the stream of the bytes of filePath from offset, size bytes or to the
// end when size is -1.
func openStream(t *testing.T, ctx context.Context, fs FileService, filePath string, offset, size int64) io.ReadCloser {
	var r io.ReadCloser
	vec := &IOVector{FilePath: filePath, Policy: SkipAllCache, Entries: []IOEntry{{Offset: offset, Size: size, ReadCloserForRead: &r}}}
	require.NoError(t, fs.Read(ctx, vec))
	return r
}

// otherRequests makes a read, a write, a listing and a stat on fs.
func otherRequests(t *testing.T, ctx context.Context, fs FileService, data []byte, name string) {
	vec := &IOVector{FilePath: "b", Policy: SkipAllCache, Entries: []IOEntry{{Offset: 3, Size: 10}}}
	require.NoError(t, fs.Read(ctx, vec))
	require.Equal(t, data[3:13], vec.Entries[0].Data)
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: name, Entries: []IOEntry{{Size: 4, Data: data[:4]}}}))
	for _, err := range fs.List(ctx, "") {
		require.NoError(t, err)
	}
	_, err := fs.StatFile(ctx, "b")
	require.NoError(t, err)
}

// Probe 1: a caller holding a stream unread makes other requests, then reads the stream.
func TestServerRuleStreamHeldWhileOtherRequests(t *testing.T) {
	for _, c := range []struct {
		name         string
		offset, size int64
	}{{"unknown size", 0, -1}, {"known size", 100, 200 << 10}, {"small", 7, 100}} {
		t.Run(c.name, func(t *testing.T) {
			fs, data := newRoomFS(t, 1, []string{"a", "b"}, 300<<10)
			fs.diskCache.memoryPrefetchMax = 64 << 10
			ctx := context.Background()
			finishes(t, func() {
				r := openStream(t, ctx, fs, "a", c.offset, c.size)
				otherRequests(t, ctx, fs, data, "new-"+strings.ReplaceAll(c.name, " ", "-"))
				got, err := io.ReadAll(r)
				require.NoError(t, err)
				want := data[c.offset:]
				if c.size > 0 {
					want = data[c.offset : c.offset+c.size]
				}
				require.Equal(t, want, got)
				require.NoError(t, r.Close())
			})
			require.Zero(t, heldRoom(fs.server))
		})
	}
}

// streamCopiedIntoWrite copies a stream of a into a write of the object copy on fs, and
// reads copy back.
func streamCopiedIntoWrite(t *testing.T, fs FileService, data []byte) {
	ctx := context.Background()
	finishes(t, func() {
		r := openStream(t, ctx, fs, "a", 0, -1)
		require.NoError(t, fs.Write(ctx, IOVector{FilePath: "copy", Entries: []IOEntry{{Size: -1, ReaderForWrite: r}}}))
		require.NoError(t, r.Close())
		vec := &IOVector{FilePath: "copy", Policy: SkipAllCache, Entries: []IOEntry{{Size: -1}}}
		require.NoError(t, fs.Read(ctx, vec))
		require.Equal(t, data, vec.Entries[0].Data)
	})
}

// Probe 2: a stream of an object is the body of a write of another on the same file
// service, on the disk object storage, on object storage written through a temporary file,
// and on object storage written in parts.
func TestServerRuleStreamCopiedIntoWrite(t *testing.T) {
	t.Run("disk", func(t *testing.T) {
		fs, data := newRoomFS(t, 1, []string{"a"}, 300<<10)
		streamCopiedIntoWrite(t, fs, data)
	})
	t.Run("temporary file", func(t *testing.T) {
		fs, _, data := newGovernedFS(t, 1, []string{"a"}, 300<<10)
		streamCopiedIntoWrite(t, fs, data)
	})
	t.Run("parts", func(t *testing.T) {
		fs, data := newRoomFS(t, 1, []string{"a"}, int(minMultipartPartSize)*2+17)
		prev := writePartSize
		writePartSize = minMultipartPartSize
		t.Cleanup(func() { writePartSize = prev })
		storage := newPartStorage()
		parts := &partsOnDisk{ObjectStorage: fs.server.raw, partStorage: storage}
		fs.server.raw = parts
		streamCopiedIntoWrite(t, fs, data)
		require.Len(t, storage.parts, 3)
	})
}

// partsOnDisk is object storage that reads from its ObjectStorage and writes objects in
// parts into it.
type partsOnDisk struct {
	ObjectStorage
	*partStorage
}

func (p *partsOnDisk) Exists(ctx context.Context, key string) (bool, error) {
	return p.ObjectStorage.Exists(ctx, key)
}

func (p *partsOnDisk) Write(ctx context.Context, key string, r io.Reader, size *int64, expire *time.Time) error {
	return p.ObjectStorage.Write(ctx, key, r, size, expire)
}

func (p *partsOnDisk) completeMultipart(ctx context.Context, key, uploadID string, parts []multipartPart) error {
	if err := p.partStorage.completeMultipart(ctx, key, uploadID, parts); err != nil {
		return err
	}
	size := int64(len(p.done))
	return p.ObjectStorage.Write(ctx, key, bytes.NewReader(p.done), &size, nil)
}

type writerFunc func([]byte) (int, error)

func (f writerFunc) Write(b []byte) (int, error) { return f(b) }

// Probe 3: the writer of a read makes another read.
func TestServerRuleWriterForReadReads(t *testing.T) {
	fs, data := newRoomFS(t, 1, []string{"a", "b"}, 300<<10)
	ctx := context.Background()
	for _, size := range []int64{200 << 10, -1} {
		finishes(t, func() {
			var got bytes.Buffer
			w := writerFunc(func(b []byte) (int, error) {
				vec := &IOVector{FilePath: "b", Policy: SkipAllCache, Entries: []IOEntry{{Offset: 1, Size: 5}}}
				if err := fs.Read(ctx, vec); err != nil {
					return 0, err
				}
				return got.Write(b)
			})
			vec := &IOVector{FilePath: "a", Policy: SkipAllCache, Entries: []IOEntry{{Offset: 10, Size: size, WriterForRead: w}}}
			require.NoError(t, fs.Read(ctx, vec))
			want := data[10:]
			if size > 0 {
				want = data[10 : 10+size]
			}
			require.Equal(t, want, got.Bytes())
		})
	}
	require.Zero(t, heldRoom(fs.server))
}

// Probe 4: the decoder of a read makes another read.
func TestServerRuleToCacheDataReads(t *testing.T) {
	fs, data := newRoomFS(t, 1, []string{"a", "b"}, 300<<10)
	ctx := context.Background()
	for _, policy := range []Policy{SkipAllCache, SkipMemoryCache, 0} {
		finishes(t, func() {
			vec := &IOVector{FilePath: "a", Policy: policy, Entries: []IOEntry{{Offset: 10, Size: 100,
				ToCacheData: func(ctx context.Context, r io.Reader, b []byte, allocator CacheDataAllocator) (fscache.Data, error) {
					other := &IOVector{FilePath: "b", Policy: SkipAllCache, Entries: []IOEntry{{Offset: 2, Size: 3}}}
					if err := fs.Read(ctx, other); err != nil {
						return nil, err
					}
					return allocator.CopyToCacheData(ctx, b), nil
				}}}}
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[10:110], vec.Entries[0].CachedData.Bytes())
			vec.Release()
		})
	}
}

// Probe 5: the body of a listing's loop reads and stats each listed file.
func TestServerRuleListBodyReads(t *testing.T) {
	fs, data := newRoomFS(t, 1, names("dir/object", 3), 4<<10)
	ctx := context.Background()
	finishes(t, func() {
		n := 0
		for entry, err := range fs.List(ctx, "dir") {
			require.NoError(t, err)
			vec := &IOVector{FilePath: "dir/" + entry.Name, Policy: SkipAllCache, Entries: []IOEntry{{Offset: 1, Size: 8}}}
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[1:9], vec.Entries[0].Data)
			_, err = fs.StatFile(ctx, "dir/"+entry.Name)
			require.NoError(t, err)
			n++
		}
		require.Equal(t, 3, n)
	})
	require.Zero(t, heldRoom(fs.server))
}

// Probe 6: a whole-object fill of unknown size learns its size from object storage in a
// request of its own before it downloads, with room for one request.
func TestServerRuleUnknownSizeFillStats(t *testing.T) {
	fs, counting, data := newGovernedFS(t, 1, []string{"object"}, 300<<10)
	fs.SetAsyncUpdate(false)
	ctx := context.Background()
	finishes(t, func() {
		vec := &IOVector{FilePath: "object", Policy: SkipMemoryCache | SkipFullFilePreloads, Entries: []IOEntry{{Offset: 0, Size: 100}}}
		require.NoError(t, fs.Read(ctx, vec))
		vec.Release()
		fs.diskCache.prefetchObjects.reads.Wait()
		require.True(t, fs.diskCache.ranges.startsAt0(fs.diskCache.objectKey("object")))
		_, sized := fs.diskCache.prefetched.size(fs.diskCache.objectKey("object"))
		require.False(t, sized)
		stats := counting.stats.Load()
		require.NoError(t, fs.PrefetchFile(ctx, "object"))
		require.Equal(t, stats+1, counting.stats.Load(), "it stats")
	})
	_, cached := fs.diskCache.ranges.covering(fs.diskCache.objectKey("object"), 0, int64(len(data)))
	require.True(t, cached)
	require.Equal(t, int64(1), counting.most.Load())
	require.Zero(t, heldRoom(fs.server))
}

// Probe 6: split whole-object reads of many objects at once with room for one request
// cache them all: each part is its own request.
func TestServerRuleSplitFills(t *testing.T) {
	objects := names("object", 6)
	fs, counting, data := newGovernedFS(t, 1, objects, 300<<10+17)
	fs.diskCache.objectSplit = objectSplitConfig{parts: 3, min: 128 << 10}
	ctx := context.Background()
	for _, name := range objects {
		fs.diskCache.prefetched.store(fs.diskCache.objectKey(name), int64(len(data)))
	}
	finishes(t, func() {
		var wg sync.WaitGroup
		for _, name := range objects {
			wg.Add(1)
			go func() {
				defer wg.Done()
				vec := &IOVector{FilePath: name, Policy: SkipMemoryCache, Entries: []IOEntry{{Offset: 250 << 10, Size: 10}}}
				defer vec.Release()
				require.NoError(t, fs.Read(ctx, vec))
				require.Equal(t, data[250<<10:250<<10+10], vec.Entries[0].Data)
			}()
		}
		wg.Wait()
	})
	require.Equal(t, int64(1), counting.most.Load())
	for _, name := range objects {
		require.Eventually(t, func() bool {
			return fs.diskCache.cache.Contains(fs.diskCache.wholePath(name, int64(len(data))))
		}, 5*time.Second, time.Millisecond, "%s is cached", name)
	}
	require.Eventually(t, func() bool { return heldRoom(fs.server) == 0 }, 5*time.Second, time.Millisecond)
}

// Reads with every policy, whole-object and range prefetches, writes, listings, stats and
// deletes made at once through the public entry points never have more requests open than
// the server's room, and free all of it.
func TestServerProbePeakOpensWithinRoom(t *testing.T) {
	const room = 4
	objects := names("object", 12)
	fs, counting, data := newGovernedFS(t, room, objects, 300<<10)
	counting.delay = time.Millisecond
	fs.diskCache.objectSplit = objectSplitConfig{parts: 2, min: 128 << 10}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	policies := []Policy{0, SkipMemoryCache, SkipAllCache, SkipMemoryCache | SkipFullFilePreloads, ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache}
	var wg sync.WaitGroup
	errs := make(chan error, 256)
	run := func(f func() error) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := f(); err != nil {
				errs <- err
			}
		}()
	}
	for i, name := range objects {
		if i%3 == 0 {
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: name, Size: int64(len(data)), Scan: &PrefetchScan{}}))
		} else if i%3 == 1 {
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: name, Size: int64(len(data)),
				Ranges: columnRanges(6, 40<<10, 2<<10), Scan: &PrefetchScan{}}))
		}
		for j, policy := range policies {
			offset := int64((i*37 + j*53) % (len(data) - 64))
			run(func() error {
				vec := &IOVector{FilePath: name, Policy: policy, Entries: []IOEntry{{Offset: offset, Size: 64}}}
				defer vec.Release()
				if err := fs.Read(ctx, vec); err != nil {
					return err
				}
				if !bytes.Equal(data[offset:offset+64], vec.Entries[0].Data) {
					return moerr.NewInternalErrorNoCtxf("wrong bytes of %s at %d", name, offset)
				}
				return nil
			})
		}
		run(func() error {
			var r io.ReadCloser
			if err := fs.Read(ctx, &IOVector{FilePath: name, Policy: SkipAllCache, Entries: []IOEntry{{Size: -1, ReadCloserForRead: &r}}}); err != nil {
				return err
			}
			defer r.Close()
			got, err := io.ReadAll(r)
			if err == nil && !bytes.Equal(data, got) {
				err = moerr.NewInternalErrorNoCtxf("wrong stream of %s", name)
			}
			return err
		})
		run(func() error {
			_ = fs.PrefetchFile(ctx, name)
			return nil
		})
		run(func() error {
			_, err := fs.StatFile(ctx, name)
			return err
		})
	}
	for i := range 6 {
		run(func() error {
			return fs.Write(ctx, IOVector{FilePath: fmt.Sprintf("new-%d", i), Entries: []IOEntry{{Size: 1 << 10, Data: data[:1<<10]}}})
		})
		run(func() error {
			for _, err := range fs.List(ctx, "") {
				if err != nil {
					return err
				}
			}
			return nil
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
	for _, name := range objects[:4] {
		require.NoError(t, fs.Delete(ctx, name))
	}
	fs.diskCache.prefetchObjects.reads.Wait()
	require.Eventually(t, func() bool {
		return !slices.ContainsFunc(objects, fs.diskCache.fillsUnderWay) && heldRoom(fs.server) == 0
	}, 10*time.Second, time.Millisecond)
	require.Positive(t, counting.total.Load())
	require.LessOrEqual(t, counting.most.Load(), int64(room), "peak requests open")
	require.Zero(t, counting.cur.Load())
	_, waiting := fs.server.counts()
	require.Zero(t, waiting[foregroundRequest]+waiting[backgroundRequest])
}

// With background downloads stalled on all the room background may hold and more waiting,
// a foreground read of every kind with a deadline of 2s returns its bytes.
func TestServerProbeBystanderForegroundRead(t *testing.T) {
	const room = 8
	objects := append(names("prefetch", 12), "foreground")
	fs, counting, data := newGovernedFS(t, room, objects, 300<<10)
	foregroundKey := fs.pathToKey("foreground")
	counting.gated = func(key string) bool { return key != foregroundKey }
	defer close(counting.gate)
	ctx := context.Background()
	for i, name := range objects[:12] {
		req := ObjectPrefetch{FilePath: name, Size: int64(len(data)), Scan: &PrefetchScan{}}
		if i%2 == 1 {
			req.Ranges = columnRanges(6, 40<<10, 2<<10)
		}
		require.NoError(t, fs.PrefetchObject(ctx, req))
		go func() { _ = fs.PrefetchFile(ctx, name) }()
	}
	require.Eventually(t, func() bool {
		running, waiting := fs.server.counts()
		return running[backgroundRequest] == fs.server.backgroundMax && waiting[backgroundRequest] > 0
	}, 5*time.Second, time.Millisecond, "background holds its half and more waits")
	for _, policy := range []Policy{0, SkipMemoryCache, SkipAllCache, SkipMemoryCache | SkipFullFilePreloads,
		ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache, SkipAllCache | ExactColumnsRead} {
		readCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		vec := &IOVector{FilePath: "foreground", Policy: policy, Entries: []IOEntry{{Offset: 100 << 10, Size: 64}}}
		require.NoError(t, fs.Read(readCtx, vec), "policy %v", policy)
		require.Equal(t, data[100<<10:100<<10+64], vec.Entries[0].Data)
		vec.Release()
		cancel()
	}
	running, _ := fs.server.counts()
	require.LessOrEqual(t, running[backgroundRequest], fs.server.backgroundMax)
}

// Two file services that leave max-conns-per-host unset have pools of their own: with
// every connection of one held, a request of the other gets a connection at once.
// Control: a request of the saturated one waits.
func TestServerProbeOwnConnectionPools(t *testing.T) {
	release := make(chan struct{})
	var held atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/block" {
			held.Add(1)
			<-release
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	defer close(release)
	first := newHTTPClient(ObjectStorageArguments{})
	second := newHTTPClient(ObjectStorageArguments{})
	get := func(c *http.Client, path string, timeout time.Duration) error {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.URL+path, nil)
		if err != nil {
			return err
		}
		resp, err := c.Do(req)
		if err != nil {
			return err
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		return resp.Body.Close()
	}
	for range maxConnsPerHost {
		go func() { _ = get(first, "/block", time.Minute) }()
	}
	require.Eventually(t, func() bool { return held.Load() == int64(maxConnsPerHost) }, 10*time.Second, time.Millisecond)

	require.Error(t, get(first, "/ok", 200*time.Millisecond), "control: the saturated pool has no connection")
	start := time.Now()
	require.NoError(t, get(second, "/ok", 2*time.Second), "the other file service shares no connection")
	require.Less(t, time.Since(start), time.Second)
}

// orderStorage records the order in which reads of readKey and stats of statKey reach it.
type orderStorage struct {
	ObjectStorage
	readKey, statKey string
	mu               sync.Mutex
	order            []string
}

func (o *orderStorage) record(op string) {
	o.mu.Lock()
	o.order = append(o.order, op)
	o.mu.Unlock()
}

func (o *orderStorage) first() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.order) == 0 {
		return ""
	}
	return o.order[0]
}

func (o *orderStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	if key == o.readKey {
		o.record("read")
	}
	return o.ObjectStorage.Read(ctx, key, min, max)
}

func (o *orderStorage) Stat(ctx context.Context, key string) (int64, error) {
	if key == o.statKey {
		o.record("stat")
		return 0, nil
	}
	return o.ObjectStorage.Stat(ctx, key)
}

// With the room full and the memory full, a foreground read of a column range a fill
// held in memory would read, and a background request that needs only room, both wait;
// when one room frees, the foreground read is admitted first, and, its buffer not
// fitting, reads through a file fill. Control: with room free, both run at once.
func TestServerProbeOneOrderForRoom(t *testing.T) {
	const size = 16 << 10
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	probeSetMemory(fs, flightBufSize(size))
	order := &orderStorage{ObjectStorage: probeRaw(fs), readKey: fs.pathToKey("object"), statKey: fs.pathToKey("bg")}
	probeSetRaw(fs, order)
	ctx := context.Background()

	r := PrefetchRange{Offset: 200 << 10, Size: size}
	got, err := probeRead(fs, r)
	require.NoError(t, err, "control")
	require.Equal(t, data[r.Offset:r.Offset+r.Size], got)
	require.NoError(t, probeStat(withBackgroundRequests(ctx), fs, "bg"), "control")
	order.mu.Lock()
	order.order = nil
	order.mu.Unlock()

	freeMemory := holdForegroundBuffer(t, fs, flightBufSize(size))
	defer freeMemory()
	frees := probeFillRoom(t, fs)
	defer func() {
		for _, free := range frees {
			free()
		}
	}()
	statDone := make(chan error, 1)
	go func() { statDone <- probeStat(withBackgroundRequests(ctx), fs, "bg") }()
	require.Eventually(t, func() bool { _, waiting := probeCounts(fs); return waiting[backgroundRequest] == 1 },
		5*time.Second, time.Millisecond)
	r = PrefetchRange{Offset: 100 << 10, Size: size}
	readDone := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		b, err := readColumnRange(ctx, fs, r)
		if err == nil && !bytes.Equal(b, data[r.Offset:r.Offset+r.Size]) {
			err = fmt.Errorf("wrong bytes")
		}
		readDone <- err
	}()
	waitedInOrder := false
	for deadline := time.Now().Add(time.Second); time.Now().Before(deadline) && !waitedInOrder; time.Sleep(time.Millisecond) {
		_, waiting := probeCounts(fs)
		waitedInOrder = waiting[foregroundRequest] == 1
	}
	frees[0]()
	require.Eventually(t, func() bool { return order.first() != "" }, 5*time.Second, time.Millisecond)
	require.Equal(t, "read", order.first(), "the foreground read is admitted before the background request")
	require.True(t, waitedInOrder, "the foreground read waits for room among the requests")
	for _, free := range frees {
		free()
	}
	require.NoError(t, <-readDone)
	require.NoError(t, <-statDone)
}

// notePeak raises peak to n when n is larger.
func notePeak(peak *atomic.Int64, n int64) {
	for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
	}
}

// The memory a file service holds never passes its budget while reads, streams and
// background prefetches run at once; with the whole budget held, a read whose range would
// be held in memory reads it through a file fill at once, never waiting.
func TestServerProbeMemoryWithinBudget(t *testing.T) {
	const unit = 64 << 10
	fs, _, data := newPrefetchRangeFS(t, unit, 4<<20)
	budget := 4 * flightBufSize(unit)
	probeSetMemory(fs, budget)
	var peak atomic.Int64
	stop := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		for {
			notePeak(&peak, probeMemoryHeld(fs))
			select {
			case <-stop:
				return
			default:
				runtime.Gosched()
			}
		}
	}()
	ctx := context.Background()
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for i := range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			off := int64(i) * 200 << 10
			vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
				Entries: []IOEntry{{Offset: off + 100, Size: 10}, {Offset: off + 2*unit + 7, Size: 20}}}
			cctx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
			err := fs.Read(cctx, vec)
			if err == nil && (!bytes.Equal(vec.Entries[0].Data, data[off+100:off+110]) || !bytes.Equal(vec.Entries[1].Data, data[off+2*unit+7:off+2*unit+27])) {
				err = fmt.Errorf("wrong bytes at %d", off)
			}
			vec.Release()
			errs <- err
		}()
	}
	for i := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var rc io.ReadCloser
			off := int64(i) * 300 << 10
			vec := &IOVector{FilePath: "object", Policy: SkipAllCache,
				Entries: []IOEntry{{Offset: off, Size: unit, ReadCloserForRead: &rc}}}
			err := fs.Read(ctx, vec)
			if err == nil {
				notePeak(&peak, probeMemoryHeld(fs))
				var b []byte
				b, err = io.ReadAll(rc)
				_ = rc.Close()
				if err == nil && !bytes.Equal(b, data[off:off+unit]) {
					err = fmt.Errorf("wrong stream bytes at %d", off)
				}
			}
			errs <- err
		}()
	}
	ranges := make([]PrefetchRange, 8)
	for i := range ranges {
		ranges[i] = PrefetchRange{Offset: int64(i)*400<<10 + 50<<10, Size: 32 << 10}
	}
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 4 << 20, Ranges: ranges, Scan: &PrefetchScan{}}))
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
	waitFillsIdle(t, fs.diskCache)
	fs.diskCache.prefetchObjects.reads.Wait()
	close(stop)
	<-sampled
	require.LessOrEqual(t, peak.Load(), budget, "the memory held never passes the budget")
	require.Positive(t, peak.Load(), "some memory was held")

	free := holdForegroundBuffer(t, fs, budget-probeMemoryHeld(fs))
	defer free()
	r := PrefetchRange{Offset: 3<<20 + 11, Size: 8 << 10}
	start := time.Now()
	got, err := probeRead(fs, r)
	require.NoError(t, err, "a read whose buffer does not fit waits for no memory")
	require.Less(t, time.Since(start), time.Second)
	require.Equal(t, data[r.Offset:r.Offset+r.Size], got)
}

// Open streams that hold the whole memory budget and writes whose caller readers block,
// pipes whose producers wait on a read of the same file service, block neither a read nor
// another write from a caller's reader: each completes at once, while the streams keep
// their memory. Control: the same read and write with no stream open and no write
// blocked.
func TestServerProbeCallerMemoryBlocksNothing(t *testing.T) {
	const size = 1 << 10
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	probeSetRaw(fs, newProbeParts(probeRaw(fs)))
	prevPart := writePartSize
	writePartSize = defaultParallelMultipartPartSize
	t.Cleanup(func() { writePartSize = prevPart })
	probeSetMemory(fs, 2*flightBufSize(size))
	ctx := context.Background()
	var seq atomic.Int64
	probeOps := func() error {
		r := PrefetchRange{Offset: 200<<10 + seq.Load(), Size: 8 << 10}
		got, err := probeRead(fs, r)
		if err != nil {
			return fmt.Errorf("read: %w", err)
		}
		if !bytes.Equal(got, data[r.Offset:r.Offset+r.Size]) {
			return fmt.Errorf("read: wrong bytes")
		}
		wctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		name := fmt.Sprintf("written-%d", seq.Add(1))
		if err := fs.Write(wctx, IOVector{FilePath: name, Policy: SkipAllCache,
			Entries: []IOEntry{{Size: -1, ReaderForWrite: bytes.NewReader(data)}}}); err != nil {
			return fmt.Errorf("write: %w", err)
		}
		return nil
	}
	require.NoError(t, probeOps(), "control")

	streams := openProbeStreams(t, fs, 2, size)
	defer func() {
		for _, s := range streams {
			_ = s.Close()
		}
	}()
	held := probeMemoryHeld(fs)
	gate := make(chan struct{})
	openGate := sync.OnceFunc(func() { close(gate) })
	defer openGate()
	blocked := runtime.NumCPU() + 1
	written := make(chan error, blocked)
	for i := range blocked {
		pr, pw := io.Pipe()
		go func() {
			<-gate
			r := PrefetchRange{Offset: int64(i) * 10 << 10, Size: 4 << 10}
			b, err := probeRead(fs, r)
			if err != nil {
				_ = pw.CloseWithError(err)
				return
			}
			_, err = pw.Write(b)
			_ = pw.CloseWithError(err)
		}()
		go func() {
			written <- fs.Write(ctx, IOVector{FilePath: fmt.Sprintf("blocked-%d", i), Policy: SkipAllCache,
				Entries: []IOEntry{{Size: -1, ReaderForWrite: pr}}})
		}()
	}
	time.Sleep(100 * time.Millisecond)
	start := time.Now()
	require.NoError(t, probeOps(), "a read and a write run while caller readers block")
	require.Less(t, time.Since(start), 2*time.Second)
	require.Equal(t, held, probeMemoryHeld(fs), "the streams keep their memory")
	openGate()
	for range blocked {
		require.NoError(t, <-written)
	}
}

// zeroReader reads zeros.
type zeroReader struct{}

func (zeroReader) Read(p []byte) (int, error) {
	clear(p)
	return len(p), nil
}

// allocated returns the bytes f allocates.
func allocated(f func()) uint64 {
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// A write of bytes the caller holds in memory sends them as they are: with one put, and as
// a multipart upload, it allocates far less than their size.
func TestServerProbeInMemoryWriteCopiesNothing(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 1)
	probeSetRaw(fs, newProbeParts(probeRaw(fs)))
	prevPart := writePartSize
	writePartSize = 8 << 20
	t.Cleanup(func() { writePartSize = prevPart })
	ctx := context.Background()
	for _, size := range []int{4 << 20, 32 << 20} {
		data := bytes.Repeat([]byte{7}, size)
		n := allocated(func() {
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: fmt.Sprintf("in-memory-%d", size), Policy: SkipAllCache,
				Entries: []IOEntry{{Size: int64(size), Data: data}}}))
		})
		require.Less(t, n, uint64(size/4), "a write of %d bytes allocated %d", size, n)
	}
}

// A write from a caller's reader of many parts reuses part buffers: it allocates far less
// than a part for each part, less than one part for each part with the race detector,
// which drops some of the buffers put back.
func TestServerProbePartBuffersReused(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 1)
	probeSetRaw(fs, newProbeParts(probeRaw(fs)))
	probeSetMemory(fs, 1<<30)
	prevPart := writePartSize
	writePartSize = minMultipartPartSize
	t.Cleanup(func() { writePartSize = prevPart })
	const parts = 24
	ctx := context.Background()
	write := func(name string) {
		require.NoError(t, fs.Write(ctx, IOVector{FilePath: name, Policy: SkipAllCache,
			Entries: []IOEntry{{Size: -1, ReaderForWrite: io.LimitReader(zeroReader{}, parts*minMultipartPartSize)}}}))
	}
	write("warm")
	n := allocated(func() { write("parts") })
	most := uint64(minMultipartPartSize / 2)
	if raceEnabled {
		most = minMultipartPartSize
	}
	require.Less(t, n/parts, most, "allocated %d for %d parts", n, parts)
}

// object-storage-memory sets the memory budget of the server; unset, the budget is 5% of
// the memory available. A budget that shrinks below the memory in use takes nothing back,
// and new memory is taken only once it fits again.
func TestServerProbeMemoryBudgetConfig(t *testing.T) {
	var args ObjectStorageArguments
	require.NoError(t, args.SetFromString([]string{"object-storage-memory=3MiB"}))
	require.Equal(t, toml.ByteSize(3<<20), probeArgsMemory(args))
	newFS := func(args ObjectStorageArguments) *S3FS {
		args.Name, args.Endpoint, args.Bucket = "s3", "disk", t.TempDir()
		fs, err := NewS3FS(context.Background(), args, CacheConfig{
			DiskPath: ptrTo(t.TempDir()), DiskCapacity: ptrTo[toml.ByteSize](64 << 20), MemoryCapacity: ptrTo[toml.ByteSize](0),
		}, nil, false, false)
		require.NoError(t, err)
		t.Cleanup(func() { fs.Close(context.Background()) })
		return fs
	}
	require.Equal(t, int64(3<<20), probeMemoryBudget(newFS(args)))

	prev := memoryAvailable
	t.Cleanup(func() { memoryAvailable = prev })
	memoryAvailable = func() (uint64, bool) { return 1 << 30, true }
	fs := newFS(ObjectStorageArguments{})
	require.Equal(t, int64((1<<30)/100*5), probeMemoryBudget(fs), "the default is 5% of the memory available")

	buf := flightBufSize(64 << 10)
	probeSetMemory(fs, 4*buf)
	free := holdForegroundBuffer(t, fs, 4*buf)
	probeSetMemory(fs, buf)
	require.Equal(t, 4*buf, probeMemoryHeld(fs), "a shrunk budget takes nothing back")
	require.False(t, probeMemoryFits(fs, buf), "new memory waits for none and is refused")
	free()
	require.True(t, probeMemoryFits(fs, buf), "memory is taken once it fits")
}

func probeSetMemory(fs *S3FS, n int64) { fs.server.memoryLimit = n }

func probeMemoryBudget(fs *S3FS) int64 { return fs.server.memoryBudget() }

func probeMemoryHeld(fs *S3FS) int64 { return heldMemory(fs.diskCache) }

func probeMemoryFits(fs *S3FS, n int64) bool { return fs.server.memoryFits(n, foregroundRequest) }

func probeArgsMemory(args ObjectStorageArguments) toml.ByteSize { return args.ObjectStorageMemory }

func probeRaw(fs *S3FS) ObjectStorage { return fs.server.raw }

func probeSetRaw(fs *S3FS, raw ObjectStorage) { fs.server.raw = raw }

func probeCounts(fs *S3FS) (running, waiting [2]int) { return fs.server.counts() }

func probeStat(ctx context.Context, fs *S3FS, name string) error {
	_, err := fs.server.Do(ctx, statRequest{key: fs.pathToKey(name)})
	return err
}

// probeFillRoom holds all the free room of fs, background first, and returns the function
// freeing each.
func probeFillRoom(t *testing.T, fs *S3FS) []func() {
	var frees []func()
	for _, c := range []requestClass{backgroundRequest, foregroundRequest} {
		for {
			free, ok := admitted(fs.server, c)
			if !ok {
				break
			}
			frees = append(frees, sync.OnceFunc(free))
		}
	}
	return frees
}

// probeParts is object storage that writes in parts, reading and dropping what it is
// sent; other calls go to the object storage it wraps.
type probeParts struct {
	ObjectStorage
}

func newProbeParts(raw ObjectStorage) ObjectStorage { return probeParts{raw} }

func (probeParts) Write(_ context.Context, _ string, r io.Reader, _ *int64, _ *time.Time) error {
	_, err := io.Copy(io.Discard, r)
	return err
}

func (probeParts) createMultipart(context.Context, string, *time.Time) (string, error) {
	return "upload", nil
}

func (probeParts) uploadPart(_ context.Context, _, _ string, _ int32, body io.ReaderAt, size int64) (string, error) {
	_, err := io.Copy(io.Discard, io.NewSectionReader(body, 0, size))
	return "etag", err
}

func (probeParts) completeMultipart(context.Context, string, string, []multipartPart) error {
	return nil
}

func (probeParts) abortMultipart(context.Context, string, string) error { return nil }

// A max-conns-per-host or a concurrency below minRequestRoom is raised to it: the
// server has room for minRequestRoom requests, half of them background; a negative one
// is rejected.
func TestRoomBelowMinimumIsRaised(t *testing.T) {
	noProbe := func(context.Context, string) (string, error) { return "", nil }
	for _, conns := range []int{-1, 0, 1, 2, minRequestRoom - 1, minRequestRoom, 256} {
		args := ObjectStorageArguments{Endpoint: "disk", Bucket: t.TempDir(), MaxConnsPerHost: conns}
		err := args.validateWithRegionProbe(noProbe)
		if conns < 0 {
			require.ErrorContains(t, err, "max-conns-per-host", "max-conns-per-host=%d", conns)
		} else {
			require.NoError(t, err, "max-conns-per-host=%d", conns)
		}
	}
	for _, tc := range []struct {
		conns       int
		concurrency int64
		room        int
	}{
		{1, 0, minRequestRoom},
		{2, 0, minRequestRoom},
		{minRequestRoom - 1, 0, minRequestRoom},
		{0, 1, minRequestRoom},
		{0, minRequestRoom - 1, minRequestRoom},
		{3, 5, minRequestRoom},
		{256, 7, minRequestRoom},
		{minRequestRoom, 0, minRequestRoom},
		{0, 0, maxConnsPerHost},
	} {
		args := ObjectStorageArguments{Name: "s3", Endpoint: "disk", Bucket: t.TempDir(), MaxConnsPerHost: tc.conns, Concurrency: tc.concurrency}
		raised := args
		raised.raiseRequestRoom()
		if tc.conns > 0 {
			require.Equal(t, max(tc.conns, minRequestRoom), raised.MaxConnsPerHost, "conns %d", tc.conns)
		}
		if tc.concurrency > 0 {
			require.Equal(t, max(tc.concurrency, minRequestRoom), raised.Concurrency, "concurrency %d", tc.concurrency)
		}
		fs, err := NewS3FS(context.Background(), args, CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](0)}, nil, false, true)
		require.NoError(t, err, "conns %d concurrency %d", tc.conns, tc.concurrency)
		require.Equal(t, tc.room, fs.server.capacity, "conns %d concurrency %d", tc.conns, tc.concurrency)
		require.Equal(t, tc.room/2, fs.server.backgroundMax, "conns %d concurrency %d", tc.conns, tc.concurrency)
		fs.Close(context.Background())
	}
	_, err := NewS3FS(context.Background(), ObjectStorageArguments{Name: "s3", Endpoint: "disk", Bucket: t.TempDir(), MaxConnsPerHost: -1},
		CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](0)}, nil, false, true)
	require.ErrorContains(t, err, "max-conns-per-host")
}

func TestDownloadConnsAreHalfTheConfiguredLimit(t *testing.T) {
	ctx := context.Background()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "s3", Endpoint: "disk", Bucket: t.TempDir(), MaxConnsPerHost: 100},
		CacheConfig{DiskPath: ptrTo(t.TempDir()), DiskCapacity: ptrTo[toml.ByteSize](64 << 20), MemoryCapacity: ptrTo[toml.ByteSize](0)}, nil, false, false)
	require.NoError(t, err)
	defer fs.Close(ctx)
	require.Equal(t, 100, fs.server.capacity)
	require.Equal(t, 50, fs.server.backgroundMax)
}

// failingHeads is an object storage server whose HEAD requests fail with the status and
// body of fail for the first fails of them, and succeed after. Before the first fails,
// it waits until a foreground request waits for room of g, and that request, once it
// runs, sends the HEADs made so far to ranWith.
type failingHeads struct {
	t        *testing.T
	g        *objectStorageServer
	fails    int32
	status   int
	body     string
	attempts atomic.Int32
	ranWith  chan int32
}

func newFailingHeads(t *testing.T, fails int32, status int, body string) (*failingHeads, *httptest.Server) {
	f := &failingHeads{t: t, fails: fails, status: status, body: body, ranWith: make(chan int32, 1)}
	server := httptest.NewServer(f)
	t.Cleanup(server.Close)
	return f, server
}

func (f *failingHeads) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && r.URL.Query().Has("location") {
		w.Header().Set("Content-Type", "application/xml")
		_, _ = io.WriteString(w, `<LocationConstraint>us-east-1</LocationConstraint>`)
		return
	}
	if r.Method != http.MethodHead {
		w.WriteHeader(http.StatusNotImplemented)
		return
	}
	n := f.attempts.Add(1)
	if n == 1 {
		go func() {
			if err := waitRoom(f.g, context.Background(), foregroundRequest); err != nil {
				f.ranWith <- -1
				return
			}
			attempts := f.attempts.Load()
			f.g.done(foregroundRequest)
			f.ranWith <- attempts
		}()
		deadline := time.Now().Add(5 * time.Second)
		for _, waiting := f.g.counts(); waiting[foregroundRequest] == 0 && time.Now().Before(deadline); _, waiting = f.g.counts() {
			time.Sleep(time.Millisecond)
		}
	}
	if n <= f.fails {
		w.Header().Set("Content-Type", "application/xml")
		w.WriteHeader(f.status)
		_, _ = io.WriteString(w, f.body)
		return
	}
	w.Header().Set("Content-Length", "6")
	w.Header().Set("ETag", `"etag"`)
	w.Header().Set("Last-Modified", time.Unix(1, 0).UTC().Format(http.TimeFormat))
	w.WriteHeader(http.StatusOK)
}

// statHoldingAllOtherRoom stats object on raw through a server whose room, but for
// one, is held, and returns its result and the HEADs made when the request that waited
// for room behind it ran.
func (f *failingHeads) statHoldingAllOtherRoom(raw ObjectStorage) (int64, int32, error) {
	f.g = newObjectStorageServer(raw, minRequestRoom)
	defer occupy(f.t, f.g, foregroundRequest, minRequestRoom-1)()
	res, err := f.g.Do(context.Background(), statRequest{key: "object"})
	ranWith := <-f.ranWith
	require.Zero(f.t, heldRoom(f.g))
	return res.n, ranWith, err
}

// requireRetriedWithoutRoom requires that the stat retried the failed HEAD once, and that
// the request waiting for the room the stat held ran before the retry.
func (f *failingHeads) requireRetriedWithoutRoom(size int64, ranWith int32, err error) {
	require.NoError(f.t, err)
	require.Equal(f.t, int64(6), size)
	require.Equal(f.t, int32(2), f.attempts.Load(), "the failed HEAD is made again once")
	require.Equal(f.t, int32(1), ranWith, "the request waiting for room runs during the retry's wait")
}

const serviceUnavailableXML = `<Error><Code>SlowDown</Code><Message>slow down</Message></Error>`

// A HEAD that fails once with a status the AWS SDK retries is made again by the file
// service's retry, which waits with no room held.
func TestAwsSDKv2RetriesWithoutRoom(t *testing.T) {
	f, server := newFailingHeads(t, 1, http.StatusServiceUnavailable, serviceUnavailableXML)
	sdk, err := NewAwsSDKv2(context.Background(), ObjectStorageArguments{
		Name: "aws", Bucket: "bucket", Endpoint: server.URL, Region: "us-east-1",
		KeyID: "id", KeySecret: "secret", NoBucketValidation: true,
	}, nil)
	require.NoError(t, err)
	f.requireRetriedWithoutRoom(f.statHoldingAllOtherRoom(sdk))
}

// A HEAD that fails once with a status the MinIO SDK retries is made again by the file
// service's retry, which waits with no room held.
func TestMinioSDKRetriesWithoutRoom(t *testing.T) {
	f, server := newFailingHeads(t, 1, http.StatusServiceUnavailable, serviceUnavailableXML)
	sdk, err := NewMinioSDK(context.Background(), ObjectStorageArguments{
		Endpoint: server.URL, Region: "us-east-1", Bucket: "bucket",
		KeyID: "id", KeySecret: "secret", NoBucketValidation: true,
	}, nil)
	require.NoError(t, err)
	f.requireRetriedWithoutRoom(f.statHoldingAllOtherRoom(sdk))
}

// A HEAD that fails with a status the file service does not retry is made once by the
// MinIO SDK.
func TestMinioSDKMakesOneAttempt(t *testing.T) {
	f, server := newFailingHeads(t, 1, http.StatusNotImplemented, `<Error><Code>NotImplemented</Code></Error>`)
	sdk, err := NewMinioSDK(context.Background(), ObjectStorageArguments{
		Endpoint: server.URL, Region: "us-east-1", Bucket: "bucket",
		KeyID: "id", KeySecret: "secret", NoBucketValidation: true,
	}, nil)
	require.NoError(t, err)
	_, ranWith, err := f.statHoldingAllOtherRoom(sdk)
	require.Error(t, err)
	require.Equal(t, int32(1), f.attempts.Load())
	require.Equal(t, int32(1), ranWith)
}

// A HEAD that fails once with a status the file service retries for COS is made again by
// the file service's retry, which waits with no room held.
func TestQCloudSDKRetriesWithoutRoom(t *testing.T) {
	f, server := newFailingHeads(t, 1, http.StatusServiceUnavailable, serviceUnavailableXML)
	sdk, err := NewQCloudSDK(context.Background(), ObjectStorageArguments{
		Name: "cos", Bucket: "bucket", Region: "ap-guangzhou",
		KeyID: "id", KeySecret: "secret", NoBucketValidation: true,
	}, nil)
	require.NoError(t, err)
	u, err := url.Parse(server.URL)
	require.NoError(t, err)
	sdk.client.BaseURL.BucketURL = u
	f.requireRetriedWithoutRoom(f.statHoldingAllOtherRoom(sdk))
}

// The Aliyun SDK makes one attempt per call: a HEAD that fails with a status the file
// service does not retry is made once.
func TestAliyunSDKMakesOneAttempt(t *testing.T) {
	f, server := newFailingHeads(t, 1, http.StatusServiceUnavailable, serviceUnavailableXML)
	client, err := oss.New(server.URL, "id", "secret", oss.ForcePathStyle(true), oss.HTTPClient(server.Client()))
	require.NoError(t, err)
	bucket, err := client.Bucket("bucket")
	require.NoError(t, err)
	_, ranWith, err := f.statHoldingAllOtherRoom(&AliyunSDK{bucket: bucket})
	require.Error(t, err)
	require.Equal(t, int32(1), f.attempts.Load())
	require.Equal(t, int32(1), ranWith)
}

// Long waits for a connection are counted until a warning reports and resets them.
func TestConnWaitReporterCountsLongWaits(t *testing.T) {
	var r connWaitReporter
	r.last = time.Now()
	r.record("host", connWaitWarn/2, false)
	require.Zero(t, r.waits, "a short wait is not counted")
	r.record("host-a", connWaitWarn, true)
	r.record("host-b", 3*connWaitWarn, false)
	require.Equal(t, int64(2), r.waits)
	require.Equal(t, int64(1), r.reused)
	require.Equal(t, 3*connWaitWarn, r.max)
	require.Equal(t, "host-b", r.host)

	r.last = time.Now().Add(-connWaitWarnInterval)
	r.record("host-c", connWaitWarn, false)
	require.Zero(t, r.waits, "the warning resets the counts")
	require.Zero(t, r.max)
}

// gateStorage holds every read until release is closed or the read's context ends.
type gateStorage struct {
	ObjectStorage
	opens   atomic.Int64
	release chan struct{}
}

func (h *gateStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	h.opens.Add(1)
	select {
	case <-h.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return h.ObjectStorage.Read(ctx, key, min, max)
}

// heldMemory returns the memory the server of d holds.
func heldMemory(d *DiskCache) int64 {
	used := d.server.memoryInUse()
	return used[foregroundRequest] + used[backgroundRequest]
}

// fillRoom holds all the free room of g until the returned function frees it.
func fillRoom(t *testing.T, g *objectStorageServer) func() {
	t.Helper()
	var frees []func()
	for _, c := range []requestClass{backgroundRequest, foregroundRequest} {
		for {
			free, ok := admitted(g, c)
			if !ok {
				break
			}
			occupiedRoom(g).Add(1)
			frees = append(frees, func() {
				free()
				occupiedRoom(g).Add(-1)
			})
		}
	}
	return sync.OnceFunc(func() {
		for _, free := range frees {
			free()
		}
	})
}

// Concurrent reads of different prefetch ranges run MemoryPrefetchObjects only up to the
// bytes the memory budget holds; the others read their ranges through file fills at once,
// never waiting for memory. A read whose context ends returns, and once storage answers
// every read gets its bytes and no buffer stays counted.
func TestPrefetchMemoryBoundUnderConcurrentReads(t *testing.T) {
	const (
		unit    = 64 << 10
		units   = 16
		budget  = 4 * unit
		waiting = 4
	)
	fs, _, data := newPrefetchRangeFS(t, unit, (units+waiting)*unit)
	fs.server.memoryLimit = budget
	storage := &gateStorage{ObjectStorage: fs.server.raw, release: make(chan struct{})}
	fs.server.raw = storage
	f := &fs.diskCache.prefetchObjects

	var peak atomic.Int64
	stopSampling := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		for {
			if n := heldMemory(fs.diskCache); n > peak.Load() {
				peak.Store(n)
			}
			select {
			case <-stopSampling:
				return
			default:
				runtime.Gosched()
			}
		}
	}()

	type result struct {
		i    int
		data []byte
		err  error
	}
	read := func(ctx context.Context, i int, out chan<- result) {
		vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
			Entries: []IOEntry{{Offset: int64(i)*unit + 100, Size: 10}}}
		err := fs.Read(ctx, vec)
		r := result{i: i, err: err}
		if err == nil {
			r.data = slices.Clone(vec.Entries[0].Data)
		}
		vec.Release()
		out <- r
	}

	done := make(chan result, units)
	for i := range units {
		go read(context.Background(), i, done)
	}
	require.Eventually(t, func() bool { return storage.opens.Load() == units }, 10*time.Second, time.Millisecond,
		"the reads past the bound read at once")

	ctx, cancel := context.WithCancel(context.Background())
	canceled := make(chan result, waiting)
	var started sync.WaitGroup
	for i := range waiting {
		started.Add(1)
		go func() {
			started.Done()
			read(ctx, units+i, canceled)
		}()
	}
	started.Wait()
	require.Eventually(t, func() bool { return storage.opens.Load() == units+waiting }, 10*time.Second, time.Millisecond)
	cancel()
	for range waiting {
		r := <-canceled
		require.ErrorIs(t, r.err, context.Canceled)
	}

	close(storage.release)
	for range units {
		select {
		case r := <-done:
			require.NoError(t, r.err)
			off := r.i*unit + 100
			require.Equal(t, data[off:off+10], r.data)
		case <-time.After(diskCacheLifecycleTestTimeout):
			t.Fatal("a read did not return")
		}
	}
	f.reads.Wait()
	close(stopSampling)
	<-sampled

	require.LessOrEqual(t, peak.Load(), int64(budget))
	require.Zero(t, heldMemory(fs.diskCache))
	require.Empty(t, f.m)
}

// suffixGateStorage holds the reads from offset from until release is closed.
type suffixGateStorage struct {
	ObjectStorage
	from    int64
	held    atomic.Int64
	release chan struct{}
}

func (h *suffixGateStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	if min != nil && *min >= h.from {
		h.held.Add(1)
		select {
		case <-h.release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return h.ObjectStorage.Read(ctx, key, min, max)
}

// A MemoryPrefetchObject whose read ended stops being counted once the read copying from
// it copied its bytes, while that read waits for another of its ranges.
func TestPrefetchBufferCountedUntilLastRelease(t *testing.T) {
	const unit = 64 << 10
	fs, _, data := newPrefetchRangeFS(t, unit, 4*unit)
	storage := &suffixGateStorage{ObjectStorage: fs.server.raw, from: 2 * unit, release: make(chan struct{})}
	fs.server.raw = storage
	f := &fs.diskCache.prefetchObjects
	bytes := func() int64 { return heldMemory(fs.diskCache) }
	memoryObjects := func() int {
		f.mu.Lock()
		defer f.mu.Unlock()
		n := 0
		for _, objects := range f.m {
			for _, o := range objects {
				if _, ok := o.(*memoryPrefetchObject); ok {
					n++
				}
			}
		}
		return n
	}

	vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
		Entries: []IOEntry{{Offset: 100, Size: 10}, {Offset: 2*unit + 100, Size: 10}}}
	done := make(chan error, 1)
	go func() { done <- fs.Read(context.Background(), vec) }()
	require.Eventually(t, func() bool { return storage.held.Load() == 1 && memoryObjects() == 1 },
		10*time.Second, time.Millisecond, "the first range's object ended; the second waits")
	// The read releases the ended object's buffer once it has copied from it, while its
	// second range still waits.
	require.Eventually(t, func() bool { return bytes() == flightBufSize(unit) }, 10*time.Second, time.Millisecond,
		"the read copied from the ended object and released its buffer")

	close(storage.release)
	require.NoError(t, <-done)
	require.Equal(t, data[100:110], vec.Entries[0].Data)
	require.Equal(t, data[2*unit+100:2*unit+110], vec.Entries[1].Data)
	vec.Release()
	f.reads.Wait()
	require.Zero(t, bytes())
}

// A cold read whose entries make more runs than the memory budget holds reads every run at
// once, those past the budget through file fills; the buffers never exceed the bound, each
// run is read once, and the read gets its bytes.
func TestPrefetchMemoryBoundForColdMultiRunRead(t *testing.T) {
	const (
		unit   = 64 << 10
		runs   = 8
		budget = 2 * unit
	)
	fs, _, data := newPrefetchRangeFS(t, unit, 2*runs*unit)
	fs.server.memoryLimit = budget
	storage := &gateStorage{ObjectStorage: fs.server.raw, release: make(chan struct{})}
	fs.server.raw = storage
	f := &fs.diskCache.prefetchObjects

	entries := make([]IOEntry, runs)
	for i := range entries {
		entries[i] = IOEntry{Offset: int64(2*i)*unit + 100, Size: 10}
	}
	vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: entries}
	done := make(chan error, 1)
	go func() { done <- fs.Read(context.Background(), vec) }()
	require.Eventually(t, func() bool { return storage.opens.Load() == runs }, 10*time.Second, time.Millisecond,
		"every run is read at once")
	require.LessOrEqual(t, heldMemory(fs.diskCache), int64(budget))
	_ = f

	close(storage.release)
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the read did not return")
	}
	for i := range vec.Entries {
		off := 2*i*unit + 100
		require.Equal(t, data[off:off+10], vec.Entries[i].Data)
	}
	vec.Release()
	require.Equal(t, int64(runs), storage.opens.Load(), "no run is read twice")
}

// A cold column prefetch of more ranges than the memory budget holds reads every range at
// once, background buffers within half the budget and the others through file fills, and
// once storage answers every range is read once.
func TestColumnPrefetchMemoryBoundForColdRanges(t *testing.T) {
	const (
		unit   = 64 << 10
		ranges = 8
		size   = 16 << 10
	)
	fs, _, data := newPrefetchRangeFS(t, unit, 2*ranges*unit)
	budget := 4 * flightBufSize(size)
	fs.server.memoryLimit = budget
	f := &fs.diskCache.prefetchObjects
	var opens atomic.Int64
	release := make(chan struct{})
	open := func(ctx context.Context, start, end int64) (io.ReadCloser, error) {
		opens.Add(1)
		select {
		case <-release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return io.NopCloser(bytes.NewReader(data[start:end])), nil
	}
	rs := make([]PrefetchRange, ranges)
	for i := range rs {
		rs[i] = PrefetchRange{Offset: int64(2*i) * unit, Size: size}
	}
	done := make(chan error, 1)
	go func() {
		prefetchRanges(fs, rs, open)
		done <- nil
	}()
	require.Eventually(t, func() bool { return opens.Load() == ranges }, 10*time.Second, time.Millisecond)
	require.LessOrEqual(t, heldMemory(fs.diskCache), budget/2, "background buffers take at most half the budget")
	_ = f

	close(release)
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the prefetch did not return")
	}
	require.Equal(t, int64(ranges), opens.Load(), "no range is read twice")
}

// Concurrent cold reads, each of more runs than the memory budget holds, all return with
// their bytes: no run waits for memory. The buffers never exceed the bound, and none stays
// counted.
func TestPrefetchConcurrentMultiRunReadsUnderTightBudget(t *testing.T) {
	const (
		unit    = 64 << 10
		runs    = 6
		readers = 16
		budget  = 2 * unit
	)
	fs, _, data := newPrefetchRangeFS(t, unit, 2*runs*unit*2)
	fs.server.memoryLimit = budget
	f := &fs.diskCache.prefetchObjects
	var peak atomic.Int64
	stop := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		for {
			if n := heldMemory(fs.diskCache); n > peak.Load() {
				peak.Store(n)
			}
			select {
			case <-stop:
				return
			default:
				runtime.Gosched()
			}
		}
	}()
	errs := make(chan error, readers)
	for r := range readers {
		go func() {
			entries := make([]IOEntry, runs)
			for i := range entries {
				// Readers overlap: each reads every other unit, shifted by r%2.
				entries[i] = IOEntry{Offset: int64(2*i+r%2)*unit + int64(r), Size: 10}
			}
			vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: entries}
			err := fs.Read(context.Background(), vec)
			for i := range vec.Entries {
				off := vec.Entries[i].Offset
				if err == nil && !bytes.Equal(data[off:off+10], vec.Entries[i].Data) {
					err = errors.New("wrong bytes")
				}
			}
			vec.Release()
			errs <- err
		}()
	}
	for range readers {
		select {
		case err := <-errs:
			require.NoError(t, err)
		case <-time.After(diskCacheLifecycleTestTimeout):
			t.Fatal("a read did not return")
		}
	}
	f.reads.Wait()
	close(stop)
	<-sampled
	require.LessOrEqual(t, peak.Load(), int64(budget))
	require.Zero(t, heldMemory(fs.diskCache))
}

// A read whose entry spans a block another read's run is reading and a block no run reads
// takes the first block from that run and reads the other; once both reads return, no
// buffer stays counted.
func TestPrefetchReadOfJoinedBlockReleasesIt(t *testing.T) {
	const unit = 64 << 10
	fs, _, data := newPrefetchRangeFS(t, unit, 4*unit)
	storage := &suffixGateStorage{ObjectStorage: fs.server.raw, from: unit, release: make(chan struct{})}
	fs.server.raw = storage
	f := &fs.diskCache.prefetchObjects
	read := func(off, size int64, done chan<- error) {
		vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
			Entries: []IOEntry{{Offset: off, Size: size}}}
		err := fs.Read(context.Background(), vec)
		if err == nil && !bytes.Equal(data[off:off+size], vec.Entries[0].Data) {
			err = errors.New("wrong bytes")
		}
		vec.Release()
		done <- err
	}
	first := make(chan error, 1)
	go read(unit+100, 10, first)
	require.Eventually(t, func() bool { return storage.held.Load() == 1 }, 10*time.Second, time.Millisecond,
		"the first read's run waits for storage")
	second := make(chan error, 1)
	go read(2*unit-5, 10, second)
	require.Eventually(t, func() bool { return storage.held.Load() == 2 }, 10*time.Second, time.Millisecond,
		"the second read reads the block no run reads")
	close(storage.release)
	require.NoError(t, <-first)
	require.NoError(t, <-second)
	f.reads.Wait()
	require.Zero(t, heldMemory(fs.diskCache), "no buffer stays counted")
}

// A read of units of which it takes some blocks, with memory for one run, returns.
func TestPrefetchReadHoldsNoUnneededRun(t *testing.T) {
	const (
		block = 64 << 10
		unit  = 4 * block
	)
	fs, _, data := newPrefetchRangeFS(t, unit, 4*unit)
	fs.server.memoryLimit = flightBufSize(2 * block)
	f := &fs.diskCache.prefetchObjects
	// Each entry is in the third block of a unit, the units apart: the read takes the
	// third and fourth blocks of each as one run, and reads the first two as another.
	entries := []IOEntry{{Offset: 2*block + 100, Size: 10}, {Offset: 2*unit + 2*block + 100, Size: 10}}
	vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: entries}
	done := make(chan error, 1)
	go func() { done <- fs.Read(context.Background(), vec) }()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the read did not return")
	}
	for _, e := range vec.Entries {
		require.Equal(t, data[e.Offset:e.Offset+e.Size], e.Data)
	}
	vec.Release()
	f.reads.Wait()
	require.Zero(t, heldMemory(fs.diskCache))
}

// keepOnDeleteStorage serves an object after it is deleted, as object storage may for a
// while.
type keepOnDeleteStorage struct{ ObjectStorage }

func (keepOnDeleteStorage) Delete(context.Context, ...string) error { return nil }

// A read whose run waits for room while its object is deleted leaves no range of the
// deleted object cached, even while storage still serves it.
func TestPrefetchReadRunWaitingAcrossDeleteCachesNothing(t *testing.T) {
	const unit = 64 << 10
	fs, _, data := newPrefetchRangeFS(t, unit, 4*unit)
	ctx := context.Background()
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "other", Entries: []IOEntry{{Size: int64(len(data)), Data: data}},
		Policy: SkipDiskCache | SkipMemoryCache}))
	fs.server.memoryLimit = flightBufSize(unit)
	otherKey := fs.pathToKey("other")
	g := newBlackBoxStorage(keepOnDeleteStorage{fs.server.raw}, func(key string, _ int64) bool { return key == otherKey })
	fs.server.raw = g
	release := sync.OnceFunc(func() { close(g.gate) })
	defer release()
	read := func(name string, done chan<- error) {
		vec := &IOVector{FilePath: name, Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: []IOEntry{{Offset: 100, Size: 10}}}
		done <- fs.Read(ctx, vec)
		vec.Release()
	}
	other := make(chan error, 1)
	go read("other", other)
	require.Eventually(t, func() bool { cur, _, _ := g.counts(otherKey); return cur == 1 }, 5*time.Second, time.Millisecond,
		"the read of other holds its room")
	freeRoom := fillRoom(t, fs.server)
	object := make(chan error, 1)
	go read("object", object)
	f := &fs.diskCache.prefetchObjects
	require.Eventually(t, waitingFor(fs.server, foregroundRequest, 1), 5*time.Second, time.Millisecond,
		"the read of object waits for room")
	// The delete's own request waits for room behind the run.
	deleted := make(chan error, 1)
	go func() { deleted <- fs.Delete(ctx, "object") }()
	require.Eventually(t, waitingFor(fs.server, foregroundRequest, 2), 5*time.Second, time.Millisecond)
	freeRoom()
	require.NoError(t, <-deleted)
	release()
	require.NoError(t, <-other)
	<-object
	f.reads.Wait()
	_, cached := fs.diskCache.ranges.covering(fs.diskCache.objectKey("object"), 0, unit)
	require.False(t, cached, "no range of the deleted object is cached")
}

// A read of a run past the bytes of a MemoryPrefetchObject whose part, cut at a stale
// object size, holds none of its blocks returns.
func TestPrefetchReadOfRunCutAtStaleSizeReturns(t *testing.T) {
	const unit = 64 << 10
	fs, _, data := newPrefetchRangeFS(t, unit, 4*unit)
	fs.diskCache.memoryPrefetchMax = unit
	fs.diskCache.prefetched.store(fs.diskCache.objectKey("object"), 30<<10)
	vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: []IOEntry{{Offset: 60 << 10, Size: 10 << 10}}}
	done := make(chan error, 1)
	go func() { done <- fs.Read(context.Background(), vec) }()
	select {
	case err := <-done:
		require.NoError(t, err)
		require.Equal(t, data[60<<10:70<<10], vec.Entries[0].Data)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the read did not return")
	}
	vec.Release()
}

// A read of an object deleted while it reads returns none of the bytes it assembled: they
// may be of two objects at the path, so the caller reads them again.
func TestPrefetchReadAcrossDeleteReturnsNoBytes(t *testing.T) {
	const unit = 64 << 10
	fs, _, data := newPrefetchRangeFS(t, unit, 4*unit)
	d := fs.diskCache
	diskPath := d.objectKey("object")
	var opens atomic.Int64
	open := func(ctx context.Context, start, end int64) (io.ReadCloser, error) {
		// The second run opens once the object is deleted.
		if opens.Add(1) == 2 {
			d.deletes.deleted(diskPath)
		}
		return io.NopCloser(bytes.NewReader(data[start:end])), nil
	}
	entries := []IOEntry{{Offset: 100, Size: 10}, {Offset: 2*unit + 100, Size: 10}}
	_, _, ok, err := d.prefetchRead(context.Background(), "object", SkipFullFilePreloads|SkipMemoryCache, entries, 100, 2*unit+110, unit, openerSource(open))
	require.NoError(t, err)
	require.Equal(t, int64(2), opens.Load())
	require.False(t, ok, "no bytes read across a delete are returned")
}

// With no memory available, concurrent reads hold no buffer: each reads its range through a
// file fill, and all of them return their bytes.
func TestPrefetchMemoryExhaustedHoldsNoBuffer(t *testing.T) {
	const (
		unit  = 64 << 10
		units = 8
	)
	prev := memoryAvailable
	t.Cleanup(func() { memoryAvailable = prev })
	memoryAvailable = func() (uint64, bool) { return 0, true }
	fs, _, data := newPrefetchRangeFS(t, unit, units*unit)
	fs.server.memoryLimit = 0
	fs.server.measuredAt.Store(0)
	storage := &gateStorage{ObjectStorage: fs.server.raw, release: make(chan struct{})}
	fs.server.raw = storage
	release := sync.OnceFunc(func() { close(storage.release) })
	t.Cleanup(release)
	f := &fs.diskCache.prefetchObjects

	type result struct {
		i    int
		data []byte
		err  error
	}
	done := make(chan result, units)
	for i := range units {
		go func() {
			vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache,
				Entries: []IOEntry{{Offset: int64(i)*unit + 100, Size: 10}}}
			err := fs.Read(context.Background(), vec)
			r := result{i: i, err: err}
			if err == nil {
				r.data = slices.Clone(vec.Entries[0].Data)
			}
			vec.Release()
			done <- r
		}()
	}
	require.Eventually(t, func() bool { return storage.opens.Load() == units }, 10*time.Second, time.Millisecond,
		"every read reads at once")
	require.Zero(t, heldMemory(fs.diskCache), "no read holds a buffer")

	release()
	for range units {
		r := <-done
		require.NoError(t, r.err)
		off := r.i*unit + 100
		require.Equal(t, data[off:off+10], r.data)
	}
	f.reads.Wait()
	require.Zero(t, heldMemory(fs.diskCache))
}
