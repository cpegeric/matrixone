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
	"cmp"
	"container/heap"
	"context"
	"slices"
	"sync/atomic"

	"github.com/matrixorigin/matrixone/pkg/logutil"
	metric "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
	"go.uber.org/zap"
)

// ObjectPrefetch asks to download, ahead of the read, the object at FilePath, of Size
// bytes, which the read reaches at Due, in Unix nanoseconds, and of whose Blocks blocks
// it reads Reads, 0 when unknown. The download is dropped if Scan ends before it starts.
// With Ranges, only those ranges of the object, of Size bytes in all, are downloaded.
type ObjectPrefetch struct {
	FilePath string
	Size     int64
	Due      int64
	Reads    int
	Blocks   int
	Ranges   []PrefetchRange
	Scan     *PrefetchScan
	// deletes is the disk cache's count of deleted objects when the request was made.
	deletes uint64
	// indexObject publishes the download of the whole object into the index-file cache.
	indexObject bool
}

// PrefetchRange is a range of an object.
type PrefetchRange struct {
	Offset int64
	Size   int64
}

// PrefetchScan is the read prefetches are for. A nil PrefetchScan never ends.
type PrefetchScan struct {
	ended atomic.Bool
}

// End ends the read: its downloads not started are dropped.
func (s *PrefetchScan) End() {
	if s != nil {
		s.ended.Store(true)
	}
}

func (s *PrefetchScan) live() bool {
	return s == nil || !s.ended.Load()
}

// share returns, in tenths, the share of the object's blocks the read reads.
func (r ObjectPrefetch) share() int {
	if r.Blocks <= 0 || r.Reads <= 0 || r.Reads >= r.Blocks {
		return 10
	}
	return r.Reads * 10 / r.Blocks
}

// unionRanges returns the ranges holding the bytes of a, in order, those that overlap or
// touch made one.
func unionRanges(a []PrefetchRange) []PrefetchRange {
	all := slices.Clone(a)
	slices.SortFunc(all, func(x, y PrefetchRange) int { return cmp.Compare(x.Offset, y.Offset) })
	out := all[:0]
	for _, r := range all {
		if r.Size <= 0 {
			continue
		}
		if n := len(out); n > 0 && r.Offset <= out[n-1].Offset+out[n-1].Size {
			last := &out[n-1]
			last.Size = max(last.Offset+last.Size, r.Offset+r.Size) - last.Offset
			continue
		}
		out = append(out, r)
	}
	return out
}

// subtractRange returns the parts of ranges outside [start, end).
func subtractRange(ranges []PrefetchRange, start, end int64) []PrefetchRange {
	var out []PrefetchRange
	for _, r := range ranges {
		lo, hi := r.Offset, r.Offset+r.Size
		if hi <= start || lo >= end {
			out = append(out, r)
			continue
		}
		if lo < start {
			out = append(out, PrefetchRange{Offset: lo, Size: start - lo})
		}
		if hi > end {
			out = append(out, PrefetchRange{Offset: end, Size: hi - end})
		}
	}
	return out
}

// ObjectPrefetcher is a file service that downloads objects ahead of the reads that need
// them.
type ObjectPrefetcher interface {
	// PrefetchObject queues req and returns without waiting for it.
	PrefetchObject(ctx context.Context, req ObjectPrefetch) error
}

var _ ObjectPrefetcher = new(S3FS)

// fillKey identifies the fills of an object into one cache: the disk cache, or the
// index-file cache when index is set.
type fillKey struct {
	diskPath string
	index    bool
}

// liveScan reports whether a read joined p, waiting, or a scan p is for has not ended.
func (p *cacheFill) liveScan() bool {
	return p.joiners > 0 || slices.ContainsFunc(p.scans, (*PrefetchScan).live)
}

// before reports whether p starts before o: it is foreground and o is not, the read reads
// a larger share of it, a read reaches it sooner, or as soon and it was asked for first.
func (p *cacheFill) before(o *cacheFill) bool {
	if a, b := p.isPromoted(), o.isPromoted(); a != b {
		return a
	}
	if a, b := p.req.share(), o.req.share(); a != b {
		return a > b
	}
	if p.req.Due != o.req.Due {
		return p.req.Due < o.req.Due
	}
	return p.seq < o.seq
}

// merge merges into p, waiting, the request req for bytes it holds: the sooner due time,
// the larger share and its scan.
func (p *cacheFill) merge(req ObjectPrefetch) {
	p.req.Due = min(p.req.Due, req.Due)
	p.req.deletes = min(p.req.deletes, req.deletes)
	if req.share() > p.req.share() {
		p.req.Reads, p.req.Blocks = req.Reads, req.Blocks
	}
	p.scans = slices.DeleteFunc(p.scans, func(s *PrefetchScan) bool { return !s.live() })
	if !slices.Contains(p.scans, req.Scan) {
		p.scans = append(p.scans, req.Scan)
	}
}

// holdsNeeded reports whether p holds bytes of an entry of vector not done.
func (p *cacheFill) holdsNeeded(vector *IOVector) bool {
	for i := range vector.Entries {
		e := &vector.Entries[i]
		if !e.done && (p.whole || (e.Size < 0 || p.start < e.Offset+e.Size) && e.Offset < p.end) {
			return true
		}
	}
	return false
}

type fillHeap []*cacheFill

func (h fillHeap) Len() int           { return len(h) }
func (h fillHeap) Less(i, j int) bool { return h[i].before(h[j]) }
func (h fillHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].pos, h[j].pos = i, j
}
func (h *fillHeap) Push(x any) {
	p := x.(*cacheFill)
	p.pos = len(*h)
	*h = append(*h, p)
}
func (h *fillHeap) Pop() any {
	old := *h
	p := old[len(old)-1]
	old[len(old)-1] = nil
	*h = old[:len(old)-1]
	p.pos = -1
	return p
}

// rangeFillPolicy is the policy a fill of ranges caches them under.
const rangeFillPolicy = SkipMemoryCacheWrites | SkipFullFilePreloads | ColumnPrefetch

// addFills makes waiting background fills of the bytes req asks for, have reporting
// whether they are cached. Bytes a waiting fill of the same cache holds merge into it,
// bytes a running fill holds need nothing more, and the other bytes get fills of their
// own; a fill of the whole object takes over the waiting fills of ranges of it no read
// joined.
func (d *DiskCache) addFills(req ObjectPrefetch, have bool, src objectSource) {
	if !req.Scan.live() {
		return
	}
	key := fillKey{diskPath: d.objectKey(req.FilePath), index: req.indexObject}
	var pieces []PrefetchRange
	if len(req.Ranges) > 0 {
		size, sized := d.prefetched.size(key.diskPath)
		for _, r := range req.Ranges {
			if sized && r.Offset < size && size < r.Offset+r.Size {
				// A range ends at the object's known end, not past it.
				r.Size = size - r.Offset
			}
			if _, cached := d.ranges.covering(key.diskPath, r.Offset, r.Size); !cached {
				pieces = append(pieces, r)
			}
		}
		if pieces = unionRanges(pieces); len(pieces) == 0 {
			return
		}
	}
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed {
		return
	}
	defer d.startFillsLocked()
	if have {
		metric.FSPrefetchCachedCounter.Inc()
		return
	}
	if pieces == nil {
		d.addWholeLocked(req, key, src)
		return
	}
	for _, o := range f.m[key.diskPath] {
		if len(pieces) == 0 {
			break
		}
		if o.index != key.index || !o.live() || !o.endsCached() {
			continue
		}
		held := subtractRange(pieces, o.start, o.end)
		if slices.Equal(held, pieces) {
			continue
		}
		if o.queued {
			metric.FSPrefetchMergedCounter.Inc()
		}
		if o.pos >= 0 {
			o.merge(req)
			heap.Fix(&f.waitingFills, o.pos)
		} else {
			o.kept = true
		}
		pieces = held
	}
	for _, r := range pieces {
		p := d.newFill(req.FilePath, r.Offset, r.Offset+r.Size, src)
		p.req, p.index = req, key.index
		d.pushFillLocked(p)
	}
}

// addWholeLocked is addFills of req, a request of the whole object of key.
func (d *DiskCache) addWholeLocked(req ObjectPrefetch, key fillKey, src objectSource) {
	f := &d.prefetchObjects
	for _, o := range f.m[key.diskPath] {
		if o.whole && o.index == key.index && o.live() {
			metric.FSPrefetchMergedCounter.Inc()
			if o.pos >= 0 {
				o.merge(req)
				heap.Fix(&f.waitingFills, o.pos)
			}
			return
		}
	}
	whole := d.newFill(req.FilePath, 0, req.Size, src)
	whole.req, whole.index, whole.whole = req, key.index, true
	var taken []*cacheFill
	for _, o := range f.m[key.diskPath] {
		if o.index == key.index && o.pos >= 0 && o.joiners == 0 && !o.isPromoted() {
			taken = append(taken, o)
		}
	}
	for _, o := range taken {
		whole.req.Due = min(whole.req.Due, o.req.Due)
		whole.scans = append(whole.scans, o.scans...)
		heap.Remove(&f.waitingFills, o.pos)
		f.queuedBytes -= o.size()
		f.forgetLocked(o)
	}
	if d.pushFillLocked(whole) {
		for _, o := range taken {
			o.done = true
			close(o.endedCh)
			o.markReady()
		}
		return
	}
	// The fills of ranges stay queued when the whole object does not fit.
	for _, o := range taken {
		heap.Push(&f.waitingFills, o)
		f.queuedBytes += o.size()
		f.registerLocked(o)
	}
}

// pushFillLocked registers p, a new waiting background fill, unless it does not fit the
// bytes of the waiting background fills, and reports whether it did.
func (d *DiskCache) pushFillLocked(p *cacheFill) bool {
	f := &d.prefetchObjects
	if !d.roomLocked(p) {
		metric.FSPrefetchDroppedCounter.Inc()
		return false
	}
	f.fillSeq++
	p.seq, p.queued = f.fillSeq, true
	if !slices.Contains(p.scans, p.req.Scan) {
		p.scans = append(p.scans, p.req.Scan)
	}
	heap.Push(&f.waitingFills, p)
	f.queuedBytes += p.size()
	f.registerLocked(p)
	metric.FSPrefetchQueuedCounter.Inc()
	return true
}

// fillQueueLimit is the most bytes of waiting background fills: half the disk cache keeps.
func (d *DiskCache) fillQueueLimit() int64 { return d.capacityFunc() / 2 }

// roomLocked reports whether p fits the bytes of the waiting background fills, after
// dropping those whose scans all ended and no read joined, then the last in order while p
// starts before it.
func (d *DiskCache) roomLocked(p *cacheFill) bool {
	f := &d.prefetchObjects
	limit := d.fillQueueLimit()
	if p.size() > limit {
		return false
	}
	for f.queuedBytes+p.size() > limit {
		var last *cacheFill
		for _, o := range f.waitingFills {
			if o.isPromoted() {
				continue
			}
			if !o.liveScan() {
				last = o
				break
			}
			if last == nil || last.before(o) {
				last = o
			}
		}
		if last == nil || last.liveScan() && !p.before(last) {
			return false
		}
		d.dropLocked(last)
	}
	return true
}

// dropLocked drops p, waiting.
func (d *DiskCache) dropLocked(p *cacheFill) {
	if p.liveScan() {
		metric.FSPrefetchDroppedCounter.Inc()
	} else {
		metric.FSPrefetchEndedCounter.Inc()
	}
	d.endLocked(p, nil)
}

// memoryBacked reports whether a fill of n bytes is made to be held in memory: it is at
// most memoryPrefetchMax bytes and its buffer fits the memory of the server for its class
// now. Its buffer is taken only once its request is admitted, if it still fits.
func (d *DiskCache) memoryBacked(n int64, background bool) bool {
	c := foregroundRequest
	if background {
		c = backgroundRequest
	}
	return n <= d.memoryPrefetchMax && d.server.memoryFits(flightBufSize(n), c)
}

// startFillsLocked starts the waiting fills in order while the first can start: it is
// foreground, or the server counts it as a background fill. A fill whose scans all ended
// and no read joined is dropped instead.
func (d *DiskCache) startFillsLocked() {
	f := &d.prefetchObjects
	for len(f.waitingFills) > 0 && !f.closed {
		p := f.waitingFills[0]
		if !p.liveScan() {
			d.dropLocked(p)
			continue
		}
		background := !p.isPromoted()
		if background && !d.server.takeFill(f.fillsMax) {
			return
		}
		heap.Pop(&f.waitingFills)
		if background {
			f.queuedBytes -= p.size()
		}
		p.background = background
		d.startFillLocked(p)
	}
}

// startFillLocked starts p, taken from the waiting fills, on its own goroutine. Its
// requests are background until it is promoted and wait in its order: its share, then its
// due time. A fill of a range that finds its bytes cached, held by another fill or its
// object deleted ends instead.
func (d *DiskCache) startFillLocked(p *cacheFill) {
	f := &d.prefetchObjects
	switch {
	case !p.whole && d.deletes.since(p.diskPath, p.req.deletes):
		metric.FSPrefetchEndedCounter.Inc()
		d.endLocked(p, nil)
		return
	case !p.whole && d.rangesCover(p):
		metric.FSPrefetchCachedCounter.Inc()
		d.endLocked(p, nil)
		return
	}
	if o := f.holdingLocked(p); o != nil && !p.whole {
		o.kept = true
		metric.FSPrefetchMergedCounter.Inc()
		d.endLocked(p, nil)
		return
	}
	p.closing, _ = f.openLocked()
	p.running, p.counted = true, true
	p.deletes, p.policy = &p.req.deletes, rangeFillPolicy
	f.reads.Add(1)
	p.setContext(withRequestOrder(withPromotion(context.Background(), p.promoted), requestOrder{share: p.req.share(), due: p.req.Due}))
	if !p.whole {
		go p.run(nil)
		return
	}
	run := f.runWhole
	go func() {
		defer context.AfterFunc(p.closing, p.cancel)()
		run(p.ctx, p)
		p.finish(nil)
	}()
}

// rangesCover reports whether cached ranges hold the bytes of p.
func (d *DiskCache) rangesCover(p *cacheFill) bool {
	_, cached := d.ranges.covering(p.diskPath, p.start, p.size())
	return cached
}

// endFill ends a background fill the server counted, and starts the waiting fills.
func (d *DiskCache) endFill() {
	d.server.endFill()
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	d.startFillsLocked()
}

// promoteObjectLocked makes p, which a read joins, foreground: it starts before every
// background fill and leaves the bytes of the waiting background fills, its requests are
// foreground, and its buffer moves to the foreground memory.
func (d *DiskCache) promoteObjectLocked(p *cacheFill) {
	if o := p.joinableDownload(); o != nil {
		o.promote()
	}
	if p.isPromoted() {
		return
	}
	close(p.promoted)
	if p.pos >= 0 {
		f := &d.prefetchObjects
		metric.FSPrefetchMissedCounter.Inc()
		f.queuedBytes -= p.size()
		heap.Fix(&f.waitingFills, p.pos)
		d.startFillsLocked()
	}
}

// promoteObject promotes p, which a read joins.
func (d *DiskCache) promoteObject(p *cacheFill) {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	d.promoteObjectLocked(p)
}

// joinFills joins the fills of the cache of vector waiting, or running before reads can
// join their downloads, that hold bytes it needs: each is promoted, and the read waits
// until reads can join what it downloads, it ends, or ctx ends. A waiting fill of the
// whole object starts at the first block the read needs. It reports whether it joined
// one.
func (s *S3FS) joinFills(ctx context.Context, vector *IOVector, src readSources) bool {
	if !src.fetch {
		return false
	}
	path, err := parseFilePathAtService(vector.FilePath, s.name)
	if err != nil {
		return false
	}
	d := s.diskCache
	index := s.indexObject(vector.Policy)
	f := &d.prefetchObjects
	f.mu.Lock()
	var joined []*cacheFill
	for _, p := range f.m[d.objectKey(path.File)] {
		if !p.queued || p.index != index || p.joinableDownload() != nil || !p.holdsNeeded(vector) {
			continue
		}
		p.joiners++
		if p.whole && p.pos >= 0 && !p.isPromoted() {
			p.splitAt = firstNeeded(vector)
		}
		joined = append(joined, p)
	}
	for _, p := range joined {
		d.promoteObjectLocked(p)
	}
	f.mu.Unlock()
	if len(joined) == 0 {
		return false
	}
	for _, p := range joined {
		select {
		case <-p.ready:
		case <-ctx.Done():
		}
	}
	f.mu.Lock()
	for _, p := range joined {
		if p.joiners--; p.pos >= 0 && !p.liveScan() {
			d.dropLocked(p)
		}
	}
	f.mu.Unlock()
	return true
}

// dropFills ends the waiting fills of the object at filePath into both caches and stops
// the running ones a prefetch made.
func (d *DiskCache) dropFills(filePath string) {
	key := d.objectKey(filePath)
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, p := range slices.Clone(f.m[key]) {
		if p.pos >= 0 {
			d.endLocked(p, nil)
		} else if p.queued {
			p.stop()
		}
	}
	d.startFillsLocked()
}

// fillsUnderWay reports whether a fill of the object at filePath runs.
func (d *DiskCache) fillsUnderWay(filePath string) bool {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.ContainsFunc(f.m[d.objectKey(filePath)], func(p *cacheFill) bool { return p.pos < 0 })
}

func (s *S3FS) PrefetchObject(ctx context.Context, req ObjectPrefetch) error {
	// A fill keys an object by its path at the service, whatever its spelling.
	path, err := parseFilePathAtService(req.FilePath, s.name)
	if err != nil {
		return nil
	}
	req.FilePath = path.File
	// An object the index-file cache holds is not downloaded into the disk cache, nor one
	// that cannot fill it.
	if policy := GetFileServicePolicy(ctx); s.indexObject(policy) {
		if s.indexObjectReadable(policy) && s.indexObjectHeld(req.FilePath) {
			return nil
		}
		// No cache holds the ranges of an index object.
		if len(req.Ranges) > 0 || !s.indexObjectFillable(policy) {
			return nil
		}
		req.indexObject = true
	}
	if s.diskCache != nil && req.Size > 0 && req.Scan.live() {
		req.deletes = s.diskCache.deletes.now()
		s.diskCache.addFills(req, s.haveObject(req), s.source(req.FilePath))
	}
	return nil
}

// haveObject reports whether the whole object of req is cached or being downloaded, or
// whether cached ranges cover the ranges req needs. A request of the whole object marks a
// cached one used.
func (s *S3FS) haveObject(req ObjectPrefetch) bool {
	path, err := parseFilePathAtService(req.FilePath, s.name)
	if err != nil {
		return true
	}
	// A download into the index-file cache is had only once that cache holds the object.
	if req.indexObject && s.cachesIndexObjects() {
		return s.indexObjectHeld(req.FilePath)
	}
	// A fill of the index-file cache does not fill the disk cache.
	key := s.diskCache.objectKey(path.File)
	if s.diskCache.wholeFill(key, false) != nil {
		return true
	}
	ctx := context.Background()
	// The whole object is the range [0, size).
	ranges, whole := req.Ranges, len(req.Ranges) == 0
	if whole {
		ranges = []PrefetchRange{{Offset: 0, Size: req.Size}}
	}
	// Every range is checked, so the record of a missing file is dropped before the
	// download decides which ranges to fetch.
	size, sized := s.diskCache.prefetched.size(key)
	checked := make(map[string]struct{})
	have := true
	for _, r := range ranges {
		if sized && r.Offset < size && size < r.Offset+r.Size {
			// A range ends at the object's known end, not past it.
			r.Size = size - r.Offset
		}
		rs, ok := s.diskCache.ranges.covering(key, r.Offset, r.Size)
		if !ok {
			have = false
			continue
		}
		for _, c := range rs {
			if _, ok := checked[c.diskPath]; ok {
				continue
			}
			checked[c.diskPath] = struct{}{}
			if !s.diskCache.cachedFileExists(ctx, c.diskPath) {
				have = false
			} else if whole {
				s.diskCache.cache.Get(ctx, c.diskPath)
			}
		}
	}
	return have
}

// runWholeFill downloads the whole object of q at its known size, from the block of its
// split.
func (s *S3FS) runWholeFill(ctx context.Context, q *cacheFill) {
	req := q.req
	ctx = withBackgroundRequests(ctx)
	// A fill started after its object was deleted downloads nothing.
	objectPath := s.diskCache.objectKey(req.FilePath)
	if s.diskCache.deletes.since(objectPath, req.deletes) {
		metric.FSPrefetchEndedCounter.Inc()
		return
	}
	// A size told before an object was deleted may be another object's at that path.
	s.diskCache.storePrefetchedSize(objectPath, req.Size, req.deletes)
	ctx = withDeletesMark(ctx, req.deletes)
	if req.indexObject {
		ctx = WithFileServicePolicy(ctx, GetFileServicePolicy(ctx)|IndexFileObject)
	}
	s.diskCache.countPrefetch(req, s.prefetchFile(ctx, req.FilePath, q.splitAt))
}

// countPrefetch counts the outcome of the prefetch of req that ended with err: a prefetch
// of an object deleted since it was asked for ended.
func (d *DiskCache) countPrefetch(req ObjectPrefetch, err error) {
	switch {
	case err == nil:
		metric.FSPrefetchDownloadedCounter.Inc()
	case d.deletes.since(d.objectKey(req.FilePath), req.deletes):
		metric.FSPrefetchEndedCounter.Inc()
	default:
		metric.FSPrefetchFailedCounter.Inc()
		logutil.Debug("fileservice: prefetch object", zap.String("path", req.FilePath), zap.Error(err))
	}
}
