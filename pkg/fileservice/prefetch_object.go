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
	"context"
	"fmt"
	"io"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
)

// prefetchObject is a download of the bytes [start, end) of an object, which reads of
// them join while it runs: a memoryPrefetchObject, into a buffer, or a
// filePrefetchObject, into its cache file.
type prefetchObject interface {
	// span returns the bytes of the object it downloads.
	span() (start, end int64)
	// target returns the object it downloads and the cache it is of.
	target() *prefetchTarget
	// live reports whether it was not stopped.
	live() bool
	// joinable reports whether reads can join it yet.
	joinable() bool
	// hasBytes reports whether it read its bytes [lo, hi), or may still.
	hasBytes(lo, hi int64) bool
	// caches reports whether it caches its bytes.
	caches() bool
	// endsCached reports whether its bytes end cached, by it or by the read that started
	// it: a read that caches its bytes joins only one whose bytes end cached.
	endsCached() bool
	// hold keeps its bytes readable for a read until release. Holds are taken while it
	// is registered.
	hold()
	release()
	// use calls fn with its bytes [off, off+n) of the object once they are written. It
	// returns fn's error, the error it failed with, ctx's, or errPrefetchMissed when it
	// ended without them.
	use(ctx context.Context, off, n int64, fn func([]byte) error) error
	// readAt reads its bytes [off, off+len(dst)) of the object into dst, as use does.
	readAt(ctx context.Context, off int64, dst []byte) error
	// stop cancels it. ended is closed once it ended, when err is why it cached nothing.
	stop()
	ended() <-chan struct{}
	err() error
}

// errPrefetchMissed is returned by use of a prefetch object that ended without the bytes.
var errPrefetchMissed = moerr.NewInternalErrorNoCtx("fileservice: prefetch object ended without the bytes")

var (
	_ prefetchObject = (*memoryPrefetchObject)(nil)
	_ prefetchObject = (*filePrefetchObject)(nil)
)

func (f *memoryPrefetchObject) span() (int64, int64) { return f.start, f.end }

// joinable reports true: reads join f once it is registered.
func (f *memoryPrefetchObject) joinable() bool { return true }

func (f *memoryPrefetchObject) live() bool { return f.ctx == nil || f.ctx.Err() == nil }

func (f *memoryPrefetchObject) hasBytes(lo, hi int64) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return !f.finished || f.start+f.n >= hi
}

func (f *memoryPrefetchObject) caches() bool { return !f.skipCache }

func (f *memoryPrefetchObject) endsCached() bool { return !f.skipCache || f.needs.joinsCaching }

func (f *memoryPrefetchObject) use(ctx context.Context, off, n int64, fn func([]byte) error) error {
	content, ok := f.waitFor(ctx, off+n)
	if !ok {
		if err := ctx.Err(); err != nil {
			return err
		}
		// A stopped object misses the bytes; the error of a failed read is the read's.
		if err := f.failure(); err != nil {
			return err
		}
		return errPrefetchMissed
	}
	return fn(content[off-f.start : off-f.start+n])
}

func (f *memoryPrefetchObject) readAt(ctx context.Context, off int64, dst []byte) error {
	return f.use(ctx, off, int64(len(dst)), func(bs []byte) error {
		copy(dst, bs)
		return nil
	})
}

func (f *memoryPrefetchObject) stop() { f.cancelRead() }

func (f *memoryPrefetchObject) ended() <-chan struct{} { return f.endedCh }

func (f *memoryPrefetchObject) err() error { return f.cacheErr }

func (p *filePrefetchObject) span() (int64, int64) { return p.start, p.end }

func (p *filePrefetchObject) joinable() bool { return p.joinableDownload() != nil }

// live reports whether p was not stopped, or its download wrote every byte and no delete
// of its object stopped it.
func (p *filePrefetchObject) live() bool {
	if p.dropped.Load() {
		return false
	}
	if p.ctx.Err() == nil {
		return true
	}
	o := p.joinableDownload()
	return o != nil && o.isComplete()
}

// hasBytes reports whether its download wrote the bytes [lo, hi), or may still.
func (p *filePrefetchObject) hasBytes(lo, hi int64) bool {
	o := p.joinableDownload()
	return o == nil || o.hasBytes(lo-p.start, hi-p.start)
}

func (p *filePrefetchObject) caches() bool { return true }

func (p *filePrefetchObject) endsCached() bool { return true }

func (p *filePrefetchObject) hold() {
	p.mu.Lock()
	p.holds++
	p.mu.Unlock()
}

// release stops one a read started when no read uses it, no queued fill keeps it and no
// part of its download began. It stops it under the lock joins take, so no join takes one that stops.
func (p *filePrefetchObject) release() {
	if p.readStarted {
		f := &p.d.prefetchObjects
		f.mu.Lock()
		defer f.mu.Unlock()
	}
	p.mu.Lock()
	p.holds--
	stop := p.readStarted && !p.kept && p.holds == 0 && (p.download == nil || !p.download.hasBegun())
	p.mu.Unlock()
	if stop {
		p.cancelRead()
	}
}

func (p *filePrefetchObject) use(ctx context.Context, off, n int64, fn func([]byte) error) error {
	buf := make([]byte, n)
	if err := p.readAt(ctx, off, buf); err != nil {
		return err
	}
	return fn(buf)
}

// readAt waits until reads can join its download, then reads from its file as the bytes
// arrive.
func (p *filePrefetchObject) readAt(ctx context.Context, off int64, dst []byte) error {
	select {
	case <-p.ready:
	case <-ctx.Done():
		return ctx.Err()
	}
	o := p.joinableDownload()
	if o == nil {
		return errPrefetchMissed
	}
	if read, written := readDownload(ctx, o, off-p.start, dst); !read {
		if err := ctx.Err(); err != nil {
			return err
		}
		// A failure that ended the download before it wrote the bytes fails the read.
		if err := o.failure(); err != nil && !written {
			return err
		}
		return errPrefetchMissed
	}
	return nil
}

// readOwnAt is readAt for the read that started p: it reads through the reader the
// download holds for it, so a download that ends first is still read, then releases it.
func (p *filePrefetchObject) readOwnAt(ctx context.Context, off int64, dst []byte) error {
	defer p.releaseStarter()
	select {
	case <-p.ready:
	case <-ctx.Done():
		return ctx.Err()
	}
	p.mu.Lock()
	o, held := p.download, p.starterHeld
	p.mu.Unlock()
	if o == nil || !held {
		return p.readAt(ctx, off, dst)
	}
	if _, ok := o.waitFor(ctx, off-p.start, off-p.start+int64(len(dst))); !ok {
		if err := ctx.Err(); err != nil {
			return err
		}
		// The download ended before it wrote the bytes.
		if err := o.failure(); err != nil {
			return err
		}
		return errPrefetchMissed
	}
	o.mu.Lock()
	f := o.file
	o.mu.Unlock()
	if _, err := f.ReadAt(dst, off-p.start); err != nil {
		return errPrefetchMissed
	}
	return nil
}

// releaseStarter releases the reader the download holds for the read that started p.
func (p *filePrefetchObject) releaseStarter() {
	p.mu.Lock()
	o, held := p.download, p.starterHeld
	p.starterHeld, p.starterDone = false, true
	p.mu.Unlock()
	if held {
		o.release()
	}
}

func (p *filePrefetchObject) stop() {
	p.dropped.Store(true)
	p.cancelRead()
}

func (p *filePrefetchObject) ended() <-chan struct{} { return p.endedCh }

func (p *filePrefetchObject) err() error { return p.endErr }

// registerLocked registers o as a prefetch object of the object at diskPath.
func (f *prefetchObjects) registerLocked(diskPath string, o prefetchObject) {
	if f.m == nil {
		f.m = make(map[string][]prefetchObject)
	}
	f.m[diskPath] = append(f.m[diskPath], o)
}

// forgetLocked forgets o as a prefetch object of the object at diskPath.
func (f *prefetchObjects) forgetLocked(diskPath string, o prefetchObject) {
	objects := f.m[diskPath]
	for i := range objects {
		if objects[i] == o {
			objects = append(objects[:i], objects[i+1:]...)
			break
		}
	}
	if len(objects) == 0 {
		delete(f.m, diskPath)
	} else {
		f.m[diskPath] = objects
	}
}

var errObjectSizeUnknown = moerr.NewInternalErrorNoCtx("object size unknown")

// prefetchedObjectsMax is the most object sizes kept; past it they are dropped.
const prefetchedObjectsMax = 1 << 16

// prefetchedObjects holds, by disk path, the size of each object a prefetch told of.
type prefetchedObjects struct {
	mu sync.Mutex
	m  map[string]int64
}

// forget drops the size of the object at diskPath, which is deleted.
func (o *prefetchedObjects) forget(diskPath string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	delete(o.m, diskPath)
}

func (o *prefetchedObjects) store(diskPath string, size int64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.storeLocked(diskPath, size)
}

func (o *prefetchedObjects) storeLocked(diskPath string, size int64) {
	if o.m == nil || len(o.m) >= prefetchedObjectsMax {
		o.m = make(map[string]int64)
	}
	o.m[diskPath] = size
}

func (o *prefetchedObjects) size(diskPath string) (int64, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	size, ok := o.m[diskPath]
	return size, ok
}

func init() {
	http.HandleFunc("/debug/prefetch_objects", func(w http.ResponseWriter, _ *http.Request) {
		allDiskCaches.Range(func(k, v any) bool {
			fmt.Fprintf(w, "disk cache %v\n", v)
			k.(*DiskCache).writePrefetchObjects(w)
			return true
		})
	})
}

// writePrefetchObjects writes the prefetch objects of d, registered or waiting for their
// resources, and its queued fills, one a line: its kind, object, bytes, where it is in its life and for how long,
// and the bytes and connections the prefetch objects hold. It copies them under the lock
// and writes them after.
func (d *DiskCache) writePrefetchObjects(w io.Writer) {
	type row struct {
		file, fill     bool
		diskPath       string
		start, end     int64
		state          string
		since          time.Time
		joinable, kept bool
	}
	f := &d.prefetchObjects
	f.mu.Lock()
	var rows []row
	add := func(diskPath string, o prefetchObject) {
		r := row{diskPath: diskPath, state: "registered", joinable: o.joinable()}
		_, r.file = o.(*filePrefetchObject)
		r.start, r.end = o.span()
		if t, ok := o.(waitingObject); ok && t.target().d != nil {
			r.state, r.since, r.kept = t.target().state.String(), t.target().since, true
		}
		rows = append(rows, r)
	}
	for diskPath, objects := range f.m {
		for _, o := range objects {
			add(diskPath, o)
		}
	}
	for o := range f.waiting {
		add(o.target().diskPath, o)
	}
	for key, fills := range f.fills {
		for _, q := range fills {
			r := row{fill: true, diskPath: key.diskPath, start: q.start, end: q.end, state: "waiting", joinable: true}
			if q.running {
				r.state = "running"
			}
			if q.isPromoted() {
				r.state += ", foreground"
			}
			rows = append(rows, r)
		}
	}
	queued := f.queuedBytes
	f.mu.Unlock()

	now := time.Now()
	lines := make([]string, 0, len(rows))
	for _, r := range rows {
		kind, state, age := "memory", r.state, ""
		if r.file {
			kind = "file"
		}
		if r.fill {
			kind = "queued fill"
		}
		if r.kept {
			age = now.Sub(r.since).Round(time.Millisecond).String()
		}
		if state == "registered" && !r.joinable {
			state = "registered, not joinable yet"
		}
		lines = append(lines, fmt.Sprintf("  %s %s [%d, %d) %s %s", kind, r.diskPath, r.start, r.end, state, age))
	}
	sort.Strings(lines)
	for _, l := range lines {
		fmt.Fprintln(w, l)
	}
	fmt.Fprintf(w, "  queued background fill bytes %d of %d\n", queued, d.fillQueueLimit())
	if g := d.server; g != nil {
		used := g.memoryInUse()
		fmt.Fprintf(w, "  memory foreground %d background %d of %d\n", used[foregroundRequest], used[backgroundRequest], g.memoryBudget())
		running, waiting := g.counts()
		fmt.Fprintf(w, "  requests of %d: foreground running %d waiting %d, background running %d of %d waiting %d\n",
			g.capacity, running[foregroundRequest], waiting[foregroundRequest],
			running[backgroundRequest], g.backgroundMax, waiting[backgroundRequest])
	}
}
