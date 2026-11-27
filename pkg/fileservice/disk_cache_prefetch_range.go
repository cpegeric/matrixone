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
	"container/heap"
	"context"
	"fmt"
	"io"
	"math"
	"math/bits"
	"net/http"
	"os"
	pathpkg "path"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
)

const (
	// The automatic prefetch unit is the bandwidth-delay product of range reads,
	// rounded up to a power of two between prefetchRangeAutoMin and prefetchRangeAutoMax.
	prefetchRangeAutoMin = 64 << 10
	prefetchRangeAutoMax = 8 << 20
)

// prefetchRangeConfig sets the prefetch ranges of range reads.
type prefetchRangeConfig struct {
	// on enables prefetch ranges; only tests turn them off.
	on bool
	// fixed, when positive, is the unit instead of the bandwidth-delay product of range
	// reads.
	fixed int64
}

// unit returns the size of prefetch ranges, 0 when they are off.
func (c prefetchRangeConfig) unit() int64 {
	if !c.on {
		return 0
	}
	if c.fixed > 0 {
		return c.fixed
	}
	return autoPrefetchUnit(rangeReadCosts.bandwidthDelay())
}

func autoPrefetchUnit(bdp int64) int64 {
	if bdp <= prefetchRangeAutoMin {
		return prefetchRangeAutoMin
	}
	if bdp >= prefetchRangeAutoMax {
		return prefetchRangeAutoMax
	}
	return 1 << bits.Len64(uint64(bdp-1))
}

// prefetchUnit returns the prefetch unit for a read of the vector, or false to read
// its ranges alone. Objects with a separator in their path read their ranges alone.
func (d *DiskCache) prefetchUnit(vector *IOVector) (int64, bool) {
	unit := d.prefetchRange.unit()
	if unit <= 0 || vector.Policy.Any(SkipDiskCacheReads, SkipDiskCacheWrites, ColumnPrefetch, ExactRead) || !vector.Policy.CacheIOEntry() {
		return 0, false
	}
	if path, err := ParsePath(vector.FilePath); err != nil || strings.Contains(pathpkg.Clean(path.File), "/") {
		return 0, false
	}
	for i := range vector.Entries {
		e := &vector.Entries[i]
		if !e.done && (e.Size <= 0 || !e.materializes(d.memoryPrefetchMax)) {
			return 0, false
		}
	}
	return unit, true
}

const (
	// prefetchBlockMax is the largest block of the grid prefetch ranges are made of.
	prefetchBlockMax = 64 << 10
	// prefetchWindowBlocks is the most blocks a read through prefetch ranges spans.
	prefetchWindowBlocks = 4096
	// prefetchChunk is the most bytes of a prefetch range read at a time.
	prefetchChunk = 64 << 10
)

type blockState int

const (
	blockMissing blockState = iota
	blockCached
	blockInFlight
	// blockJoined is a block another fill was reading when the read began.
	blockJoined
)

// prefetchBlock is a block of a prefetch window, or a piece of one, and where its bytes
// come from.
type prefetchBlock struct {
	lo, hi     int64 // the block's bytes
	start, end int64 // the block's bytes the read takes, or all of them
	needed     bool  // the read takes bytes of the block
	wanted     bool  // the block is in a unit the read takes bytes of
	state      blockState
	cached     cachedRange
	// run is the run of the read reading the block's bytes.
	run *readRun
}

// readRun is a run of blocks a read reads by a fill of its own, or by the fill holding
// their bytes it joined; got is held until the last needed block of the run is copied,
// or, when copyHolds is set, only while a block is copied from it.
type readRun struct {
	first, last int
	got         *cacheFill
	err         error
	release     func()
	copyHolds   bool
}

// prefetchRead returns a buffer of the bytes [lo, hi) of filePath that holds the bytes
// of the entries not done, and the number of fills it started. It splits the unit-aligned
// window around them into blocks, a block fills hold only some bytes of into pieces at
// their bounds, and takes each block an entry needs from a cached range, from a fill
// holding it, or from a fill of its own: each run of blocks in units the entries take
// bytes of that are neither cached nor being read and holds a needed block is one fill,
// served to other reads as its bytes arrive, and cached. It returns false when the bytes
// are not all read, or the object was deleted since it began, and the error of a failed
// read of object storage it waited on.
func (d *DiskCache) prefetchRead(
	ctx context.Context,
	filePath string,
	policy Policy,
	entries []IOEntry,
	lo, hi, unit int64,
	src objectSource,
) ([]byte, int, bool, error) {
	if lo < 0 || hi <= lo || ctx.Err() != nil {
		return nil, 0, false, nil
	}
	block := int64(prefetchBlockMax)
	if unit < block || unit%block != 0 {
		block = unit
	}
	windowStart := lo / unit * unit
	windowEnd := (hi + unit - 1) / unit * unit
	if (windowEnd-windowStart)/block > prefetchWindowBlocks {
		return nil, 0, false, nil
	}
	// No fill of the read starts after a delete of the object counted since it began, and
	// no bytes it read across one are returned.
	deletes := d.deletes.now()
	diskPath := d.objectKey(filePath)
	ranges := d.ranges.of(diskPath)

	n := int((windowEnd - windowStart) / block)
	needed := make([]bool, n)
	for i := range entries {
		e := &entries[i]
		if !e.ranged() {
			continue
		}
		for j := (e.Offset - windowStart) / block; j <= (e.Offset+e.Size-1-windowStart)/block; j++ {
			needed[j] = true
		}
	}
	perUnit := int(unit / block)
	wanted := make([]bool, (n+perUnit-1)/perUnit)
	for j := range needed {
		wanted[j/perUnit] = wanted[j/perUnit] || needed[j]
	}

	// newBlock returns the block of the bytes [start, end), of a grid block needed or
	// wanted as given, and whether a cached range holds it.
	newBlock := func(start, end int64, isNeeded, isWanted bool) prefetchBlock {
		b := prefetchBlock{lo: start, hi: end, start: start, end: end, wanted: isWanted}
		if isNeeded && start < hi && lo < end {
			b.needed = true
			b.start, b.end = max(start, lo), min(end, hi)
		}
		if r, ok := smallestContaining(ranges, b.start, b.end-b.start, nil); ok && d.cache.Contains(r.diskPath) {
			b.state, b.cached = blockCached, r
		}
		return b
	}
	grid := make([]prefetchBlock, 0, n)
	for j, start := 0, windowStart; start < windowEnd; j, start = j+1, start+block {
		grid = append(grid, newBlock(start, start+block, needed[j], wanted[j/perUnit]))
	}

	// Find the bytes being read under one lock: a block a fill holds is joined, and a
	// block fills hold only some bytes of is split into pieces, those they hold joined.
	f := &d.prefetchObjects
	f.mu.Lock()
	if _, ok := f.openLocked(); !ok {
		f.mu.Unlock()
		return nil, 0, false, nil
	}
	blocks := make([]prefetchBlock, 0, len(grid))
	for _, b := range grid {
		if b.state != blockMissing {
			blocks = append(blocks, b)
			continue
		}
		pieces := f.coverLocked(diskPath, false, b.lo, b.hi, true)
		if len(pieces) == 1 {
			if pieces[0].o != nil {
				b.state = blockJoined
			}
			blocks = append(blocks, b)
			continue
		}
		for _, p := range pieces {
			piece := newBlock(p.start, p.end, b.needed, b.wanted)
			if p.o != nil {
				piece.state = blockJoined
			}
			blocks = append(blocks, piece)
		}
	}
	f.mu.Unlock()
	var pending [][2]int
	for i := 0; i < len(blocks); {
		if blocks[i].state != blockMissing || !blocks[i].wanted {
			i++
			continue
		}
		j, first := i, -1
		for j < len(blocks) && blocks[j].state == blockMissing && blocks[j].wanted {
			if first < 0 && blocks[j].needed {
				first = j
			}
			j++
		}
		if first >= 0 {
			pending = append(pending, [2]int{first, j - 1})
			if first > i {
				pending = append(pending, [2]int{i, first - 1})
			}
		}
		i = j
	}

	// Each run is a fill of its own, registered at once, or the fill holding its bytes it
	// joins. Its own fills start in order; each serves its bytes as they arrive and stays
	// registered until it is cached, outliving the read.
	var runs []*readRun
	var own []*cacheFill
	for _, pr := range pending {
		start, end := blocks[pr[0]].lo, blocks[pr[1]].hi
		if size, ok := d.prefetched.size(diskPath); ok && start < size && size < end {
			// A run reads to the object's end, not past it.
			end = size
		}
		p := d.newFill(filePath, start, end, src)
		p.setContext(context.WithoutCancel(ctx))
		// A buffered run outlives the read; one in a file is stopped once no read uses it
		// before it began.
		p.readStarted = !d.memoryBacked(end-start, requestClassOf(ctx) == backgroundRequest)
		p.policy, p.deletes = policy, &deletes
		r := &readRun{first: pr[0], last: pr[1]}
		r.got, r.err = d.registerFill(ctx, p, true)
		if r.got == p {
			own = append(own, p)
		}
		r.release = func() {}
		if r.got != nil {
			r.release = sync.OnceFunc(r.got.release)
		}
		if r.got == p && !p.readStarted {
			// A buffered run holds its buffer only while the read copies from it.
			r.copyHolds = true
			r.release()
		}
		for r.last > r.first && blocks[r.last].lo >= end {
			r.last--
		}
		runs = append(runs, r)
	}
	defer func() {
		for _, r := range runs {
			r.release()
		}
	}()
	startInOrder(own)
	remaining := make(map[*readRun]int)
	for _, r := range runs {
		for j := r.first; j <= r.last; j++ {
			if blocks[j].needed {
				blocks[j].state, blocks[j].run = blockInFlight, r
				remaining[r]++
			}
		}
		if remaining[r] == 0 {
			r.release()
		}
	}

	// Assemble the bytes. A fill is released once the last needed block it holds is
	// copied.
	out := make([]byte, hi-lo)
	// copyBlock copies the bytes of b from the fill reading them, holding it while it
	// copies. It reports false when they are not read, with the error failing the read.
	copyBlock := func(b *prefetchBlock, dst []byte) (bool, error) {
		var o *cacheFill
		var release func()
		if r := b.run; r != nil {
			if r.err != nil {
				return false, ctx.Err()
			}
			o = r.got
			release = func() {
				if remaining[r]--; remaining[r] == 0 {
					r.release()
				}
			}
			if r.copyHolds {
				f.mu.Lock()
				if o.done {
					o = nil
				} else {
					o.hold()
					release = o.release
				}
				f.mu.Unlock()
			}
		} else {
			// The read joins the block as it chose to when it planned it.
			f.mu.Lock()
			o, release = heldLocked(f.joinableLocked(diskPath, false, b.start, b.end, true))
			f.mu.Unlock()
		}
		var err error = errPrefetchMissed
		if o != nil {
			err = o.readAt(ctx, b.start, dst)
		}
		release()
		if err == nil {
			return true, nil
		}
		// A failed read of object storage fails the read. A fill that caches its bytes and
		// ended before the read reached the block left them in a cached range.
		if err := readFailure(ctx, filePath, fromSource, err); err != nil {
			return false, err
		}
		if o != nil && !o.caches() {
			return false, nil
		}
		return d.readCachedBytes(ctx, filePath, b.start, dst), nil
	}
	for i := 0; i < len(blocks); i++ {
		b := &blocks[i]
		if !b.needed {
			continue
		}
		dst := out[b.start-lo : b.end-lo]
		switch b.state {
		case blockInFlight, blockJoined:
			if ok, err := copyBlock(b, dst); !ok {
				return nil, len(own), false, err
			}
		case blockCached:
			j := i
			for j+1 < len(blocks) && blocks[j+1].needed && blocks[j+1].state == blockCached &&
				blocks[j+1].cached.diskPath == b.cached.diskPath {
				j++
			}
			dst = out[b.start-lo : blocks[j].end-lo]
			if !d.readCachedRange(ctx, diskPath, b.cached, b.start, dst) {
				return nil, len(own), false, nil
			}
			i = j
		default:
			return nil, len(own), false, nil
		}
	}
	// Bytes copied across a delete of the object may be of two objects at the path.
	if d.deletes.since(diskPath, deletes) {
		return nil, len(own), false, nil
	}
	return out, len(own), true, nil
}

// startInOrder runs the fills, registered, each on its own goroutine, starting each once
// the one before holds room, is file-backed or ended.
func startInOrder(fills []*cacheFill) {
	if len(fills) == 0 {
		return
	}
	go func() {
		for _, p := range fills {
			admitted := make(chan struct{})
			go p.run(func() { close(admitted) })
			<-admitted
		}
	}()
}

// readDownload reads the bytes of download o from off into dst, once they are written. It
// reports whether it read them, and whether the download wrote them.
func readDownload(ctx context.Context, o *objectDownload, off int64, dst []byte) (read, written bool) {
	if _, ok := o.waitFor(ctx, off, off+int64(len(dst))); !ok {
		return false, false
	}
	held, err := o.with(func(f io.ReaderAt) error {
		_, err := f.ReadAt(dst, off)
		return err
	})
	return held && err == nil, true
}

// fetchRange returns the bytes [start, end) of the object at filePath from the fills
// holding them: it joins each fill holding some of them, and takes the bytes none holds,
// and those of a joined fill that ended without them, through fetchPiece. It reports
// whether it returned them, and whether the fills cache them; it returns false when a
// delete of the object was counted since it began. A read whose cache is the index-file
// cache, index, joins and starts fills of that cache.
func (d *DiskCache) fetchRange(
	ctx context.Context,
	filePath string,
	policy Policy,
	index bool,
	start, end int64,
	src objectSource,
) (content []byte, ok, cached bool, err error) {
	if start < 0 || end <= start {
		return nil, false, false, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, false, false, err
	}
	// Bytes copied across a delete of the object may be of two objects at the path.
	key, deletes := d.objectKey(filePath), d.deletes.now()
	defer func() {
		if ok && d.deletes.since(key, deletes) {
			content, ok, cached = nil, false, false
		}
	}()
	f := &d.prefetchObjects
	f.mu.Lock()
	pieces := f.coverLocked(key, index, start, end, policy.CacheIOEntry() && !policy.Any(SkipDiskCacheWrites))
	if len(pieces) == 1 {
		f.mu.Unlock()
		return d.fetchPiece(ctx, filePath, policy, index, start, end, src)
	}
	releases := make([]func(), len(pieces))
	for i := range pieces {
		pieces[i].o, releases[i] = heldLocked(pieces[i].o)
	}
	f.mu.Unlock()

	out := make([]byte, end-start)
	type pieceResult struct {
		ok, cached bool
		err        error
	}
	results := make([]pieceResult, len(pieces))
	var wg sync.WaitGroup
	for i, p := range pieces {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dst := out[p.start-start : p.end-start]
			if o := p.o; o != nil {
				err := o.readAt(ctx, p.start, dst)
				releases[i]()
				if err == nil {
					results[i] = pieceResult{ok: true, cached: o.caches()}
					return
				}
				if err := readFailure(ctx, filePath, fromSource, err); err != nil {
					results[i] = pieceResult{err: err}
					return
				}
				// A fill that caches its bytes and ended without them may have left them
				// cached in a range.
				if o.caches() && d.readCachedBytes(ctx, filePath, p.start, dst) {
					results[i] = pieceResult{ok: true, cached: true}
					return
				}
			}
			content, ok, cached, err := d.fetchPiece(ctx, filePath, policy, index, p.start, p.end, src)
			if ok {
				copy(dst, content)
			}
			results[i] = pieceResult{ok: ok, cached: cached, err: err}
		}()
	}
	wg.Wait()
	cached = true
	for _, r := range results {
		if r.err != nil {
			return nil, false, false, r.err
		}
		if !r.ok {
			return nil, false, false, nil
		}
		cached = cached && r.cached
	}
	return out, true, cached, nil
}

// fetchPiece returns the bytes [start, end) of the object at filePath from the fill
// holding them, or from a fill it starts: buffered, it caches nothing, as the read caches
// its entries; file-backed, it caches them. A read that skips disk-cache writes, of bytes
// not buffered, only joins a fill. It reports whether it returned them, and whether the
// fill caches them.
func (d *DiskCache) fetchPiece(
	ctx context.Context,
	filePath string,
	policy Policy,
	index bool,
	start, end int64,
	src objectSource,
) (content []byte, ok, cached bool, err error) {
	caches := policy.CacheIOEntry() && !policy.Any(SkipDiskCacheWrites)
	f := &d.prefetchObjects
	var o *cacheFill
	memory := d.memoryBacked(end-start, requestClassOf(ctx) == backgroundRequest)
	if !memory && policy.Any(SkipDiskCacheWrites) {
		f.mu.Lock()
		o, _ = heldLocked(f.joinableLocked(d.objectKey(filePath), index, start, end, caches))
		f.mu.Unlock()
	} else {
		deletes := d.deletes.now()
		p := d.newFill(filePath, start, end, src)
		p.setContext(context.WithoutCancel(ctx))
		p.index, p.policy, p.deletes = index, policy, &deletes
		p.readStarted = true
		p.skipCache, p.joinsCaching, p.noFile = memory, caches, policy.Any(SkipDiskCacheWrites)
		if o, err = d.registerFill(ctx, p, false); o == p {
			go p.run(nil)
		}
	}
	if err != nil || o == nil {
		return nil, false, false, readFailure(ctx, filePath, fromSource, err)
	}
	defer o.release()
	// A fill that failed to read the bytes fails the read; one that ended without them
	// leaves the read to the caller.
	out := make([]byte, end-start)
	if err := o.readAt(ctx, start, out); err != nil {
		if err := readFailure(ctx, filePath, fromSource, err); err != nil {
			return nil, false, false, err
		}
		// A fill that caches its bytes and ended without them may have found them cached
		// in a range.
		if o.caches() && d.readCachedBytes(ctx, filePath, start, out) {
			return out, true, true, nil
		}
		return nil, false, false, nil
	}
	return out, true, o.caches(), nil
}

// errPrefetchMissed is returned by a read of a fill that ended without the bytes.
var errPrefetchMissed = moerr.NewInternalErrorNoCtx("fileservice: prefetch object ended without the bytes")

// errPrefetchDeleted is the error of a fill of an object deleted before it started.
var errPrefetchDeleted = moerr.NewInternalErrorNoCtx("fileservice: object deleted during a prefetch")

// errPrefetchClosed is the error of a fill the disk cache closed on before it started.
var errPrefetchClosed = moerr.NewInternalErrorNoCtx("fileservice: disk cache closed during a prefetch")

// acquireUntil waits for room of g for a request under ctx until ctx or closing ends.
func acquireUntil(ctx context.Context, g *objectStorageServer, closing context.Context) (*requestRoom, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	defer context.AfterFunc(closing, cancel)()
	return g.acquire(ctx)
}

// readFromFlights fills the entries of vector not done, of the object at filePath, from the
// fills of the read's cache holding their bytes, waiting for the bytes to arrive. Each
// entry holds its fill only while it waits for and copies its bytes. One no fill holds, or
// whose fill ends without its bytes, is left to the caller. A read that caches its bytes
// joins only fills that cache theirs.
func (d *DiskCache) readFromFlights(ctx context.Context, filePath string, vector *IOVector, caches bool) error {
	diskPath := d.objectKey(filePath)
	f := &d.prefetchObjects
	for i := range vector.Entries {
		e := &vector.Entries[i]
		if !e.ranged() || !e.materializes(d.memoryPrefetchMax) {
			continue
		}
		f.mu.Lock()
		fl, release := heldLocked(f.joinableLocked(diskPath, vector.indexTier, e.Offset, e.Offset+e.Size, caches))
		f.mu.Unlock()
		if fl == nil {
			continue
		}
		buf := make([]byte, e.Size)
		err := fl.readAt(ctx, e.Offset, buf)
		release()
		if err != nil {
			// A fill that ended without the bytes leaves them to the caller; a failed read
			// of object storage fails the read.
			if err := readFailure(ctx, filePath, fromSource, err); err != nil {
				return err
			}
			continue
		}
		if err := e.readFrom(ctx, bytes.NewReader(buf), d.cacheDataAllocator); err != nil {
			return err
		}
		e.done = true
		e.fromCache = d
	}
	return nil
}

// readCachedBytes reads len(dst) bytes at offset of the object at filePath from what the
// disk cache holds of them: the smallest cached range holding them, or the cached ranges
// holding them together while no delete of the object is counted meanwhile. It reports
// whether it read them.
func (d *DiskCache) readCachedBytes(ctx context.Context, filePath string, offset int64, dst []byte) bool {
	key, end := d.objectKey(filePath), offset+int64(len(dst))
	if r, ok := smallestContaining(d.ranges.of(key), offset, int64(len(dst)), nil); ok && d.cache.Contains(r.diskPath) {
		return d.readCachedRange(ctx, key, r, offset, dst)
	}
	deletes := d.deletes.now()
	if rs, ok := d.ranges.covering(key, offset, int64(len(dst))); ok {
		for _, r := range rs {
			lo, hi := max(offset, r.offset), min(end, r.offset+r.size)
			if !d.cache.Contains(r.diskPath) || !d.readCachedRange(ctx, key, r, lo, dst[lo-offset:hi-offset]) {
				return false
			}
		}
		return !d.deletes.since(key, deletes)
	}
	return false
}

// readCachedRange reads len(dst) bytes at offset of the object with key from the cached
// range r. A range it cannot read is dropped.
func (d *DiskCache) readCachedRange(ctx context.Context, key string, r cachedRange, offset int64, dst []byte) bool {
	opened, ok := d.files.open(ctx, r.diskPath)
	if !ok {
		d.cachedFileExists(ctx, r.diskPath)
		return false
	}
	defer d.files.close(opened)
	if _, err := opened.file.ReadAt(dst, offset-r.offset); err != nil {
		d.dropCachedRange(key, r.diskPath, opened.file)
		return false
	}
	d.cache.Get(ctx, r.diskPath)
	return true
}

// dropCachedRange has the publisher remove the cached range at diskPath of the object with
// key, whose file f is short or unreadable, while f is still its file. A range whose file
// is not removed stays in the cache, stale.
func (d *DiskCache) dropCachedRange(key, diskPath string, f *os.File) {
	info, err := f.Stat()
	if err != nil {
		return
	}
	d.pub.queue(key, queuedJob{"repair", diskPath}, func() error {
		if !sameFileAt(info, diskPath) {
			return nil
		}
		err := os.Remove(diskPath)
		d.files.changed(diskPath)
		if err != nil && !os.IsNotExist(err) {
			d.ranges.retire(diskPath, key)
			return err
		}
		d.cache.Delete(context.Background(), diskPath)
		return nil
	})
}

// cachedFileExists reports whether the cache file at diskPath is on disk; when it is
// missing, its index entry is dropped.
func (d *DiskCache) cachedFileExists(ctx context.Context, diskPath string) bool {
	if _, err := os.Stat(diskPath); !os.IsNotExist(err) {
		return true
	}
	d.cache.Delete(ctx, diskPath)
	return false
}

const (
	// flightBufMinShift and flightBufMaxShift bound the power-of-two sizes of the flight
	// buffers kept for reuse; a larger flight allocates its own.
	flightBufMinShift = 16
	flightBufMaxShift = 25
)

// flightBufs keeps released flight buffers for reuse, by power-of-two size.
var flightBufs [flightBufMaxShift - flightBufMinShift + 1]sync.Pool

// flightBufShift returns the power of two of the buffer of a flight of n bytes.
func flightBufShift(n int64) int {
	return max(bits.Len64(uint64(max(n, 1)-1)), flightBufMinShift)
}

// flightBufSize returns the bytes of the buffer of a flight of n bytes.
func flightBufSize(n int64) int64 {
	if shift := flightBufShift(n); shift <= flightBufMaxShift {
		return 1 << shift
	}
	return n
}

// getFlightBuf returns a buffer of n bytes, of capacity flightBufSize(n).
func getFlightBuf(n int64) []byte {
	shift := flightBufShift(n)
	if shift > flightBufMaxShift {
		return make([]byte, n)
	}
	if b, ok := flightBufs[shift-flightBufMinShift].Get().(*[]byte); ok {
		return (*b)[:n]
	}
	return make([]byte, n, 1<<shift)
}

// putFlightBuf keeps b, from getFlightBuf, for reuse.
func putFlightBuf(b []byte) {
	c := int64(cap(b))
	if c == 0 || flightBufSize(c) != c || c > 1<<flightBufMaxShift {
		return
	}
	b = b[:c]
	flightBufs[flightBufShift(c)-flightBufMinShift].Put(&b)
}

// cacheFill is a fill: a download of the bytes [start, end) of an object into one cache,
// the disk cache or, when index is set, the index-file cache. A whole object is the range
// [0, size). From when it is registered until it ends it is in the fills of the disk
// cache: waiting, in the order of the waiting fills, holding nothing, or running. Reads
// join it for the bytes it holds, written into its buffer or its cache file.
type cacheFill struct {
	d                  *DiskCache
	filePath, diskPath string
	index              bool
	src                objectSource
	start, end         int64
	whole              bool
	// content is set on a fill of the whole object from bytes the write path holds, which
	// is registered with its download and ends once its file is published.
	content bool
	// policy is the policy a buffered range is cached under. skipCache leaves a buffered
	// range out of the cache; joinsCaching is then set when the read that starts it caches
	// its entries, so reads that cache join it. noFile is set when it is buffered or reads
	// nothing.
	policy                          Policy
	skipCache, joinsCaching, noFile bool
	// deletes, when set, is the count of deletes after which it does not start.
	deletes *uint64
	// readStarted is set on a fill a read started, which is stopped once no read uses it,
	// no prefetch keeps it and no part of its download began.
	readStarted bool

	// Guarded by the lock of the fills.
	req   ObjectPrefetch
	scans []*PrefetchScan
	seq   uint64
	pos   int // in the waiting fills; -1 when not waiting
	// joiners counts the reads waiting for it to start; splitAt is where the first read
	// that joined it, waiting and of the whole object, needs its bytes from.
	joiners int
	splitAt int64
	// queued is set on a fill a prefetch made; running once it started; background while
	// the server counts it among its background fills; counted while it is counted among
	// the runs the closing of the cache waits for.
	queued, running, background, counted bool
	// kept is set on one a read started that a prefetch relies on. handedOff is set once
	// its file is handed to its publisher: it stays registered until the file is
	// published. done is set once it ended.
	kept, handedOff, done bool
	closing               context.Context
	promoted              chan struct{}

	ctx       context.Context
	cancel    context.CancelFunc
	ready     chan struct{} // closed once reads can join its download, or it ended
	readyOnce sync.Once
	endedCh   chan struct{}
	dropped   atomic.Bool

	mu       sync.Mutex
	download *objectDownload
	holds    int
	// retireDue is set once its download is done: it is retired once no hold is left.
	retireDue   bool
	stopOnClose func() bool
}

type fillCtxKey struct{}

// newFill returns an unregistered fill of the bytes [start, end) of the object at
// filePath, read from src.
func (d *DiskCache) newFill(filePath string, start, end int64, src objectSource) *cacheFill {
	return &cacheFill{d: d, filePath: filePath, diskPath: d.objectKey(filePath), src: src, start: start, end: end, pos: -1,
		ready: make(chan struct{}), endedCh: make(chan struct{}), promoted: make(chan struct{})}
}

func (p *cacheFill) setContext(ctx context.Context) {
	ctx, p.cancel = context.WithCancel(ctx)
	p.ctx = context.WithValue(ctx, fillCtxKey{}, p)
}

// fillOf returns the fill whose download ctx is of, or nil.
func fillOf(ctx context.Context) *cacheFill {
	p, _ := ctx.Value(fillCtxKey{}).(*cacheFill)
	return p
}

// isDownload reports whether ctx is that of a fill's download.
func isDownload(ctx context.Context) bool { return fillOf(ctx) != nil }

func (p *cacheFill) size() int64 { return p.end - p.start }

func (p *cacheFill) isPromoted() bool {
	select {
	case <-p.promoted:
		return true
	default:
		return false
	}
}

func (p *cacheFill) markReady() { p.readyOnce.Do(func() { close(p.ready) }) }

func (p *cacheFill) stopContext() {
	if p.cancel != nil {
		p.cancel()
	}
}

// joinable reports whether reads can join p: a range once registered, a whole object once
// its download began.
func (p *cacheFill) joinable() bool { return !p.whole || p.joinableDownload() != nil }

// live reports whether p was not stopped, or its download wrote every byte and no delete
// of its object stopped it.
func (p *cacheFill) live() bool {
	if p.dropped.Load() {
		return false
	}
	if p.ctx == nil || p.ctx.Err() == nil {
		return true
	}
	o := p.joinableDownload()
	return o != nil && o.isComplete()
}

// hasBytes reports whether its download wrote the bytes [lo, hi), or may still.
func (p *cacheFill) hasBytes(lo, hi int64) bool {
	o := p.joinableDownload()
	return o == nil || o.hasBytes(lo-p.start, hi-p.start)
}

// fileBacked reports whether p writes its bytes into its cache file.
func (p *cacheFill) fileBacked() bool {
	o := p.joinableDownload()
	return o != nil && !o.memory
}

// caches reports whether p caches its bytes: all but a buffered one that skips the cache.
func (p *cacheFill) caches() bool {
	if !p.skipCache {
		return true
	}
	o := p.joinableDownload()
	return !p.noFile && (o == nil || !o.memory)
}

// endsCached reports whether its bytes end cached, by it or by the read that started it:
// a read that caches its bytes joins only one whose bytes end cached.
func (p *cacheFill) endsCached() bool { return !p.skipCache || p.joinsCaching || p.fileBacked() }

func (p *cacheFill) hold() {
	p.mu.Lock()
	p.holds++
	p.mu.Unlock()
}

// release drops a hold. One a read started is stopped when no read uses it, no prefetch
// keeps it and no part of its download began, under the lock joins take, so no join takes
// one that stops. Its download is retired once it is done and no hold is left.
func (p *cacheFill) release() {
	if p.readStarted {
		f := &p.d.prefetchObjects
		f.mu.Lock()
		defer f.mu.Unlock()
	}
	p.mu.Lock()
	p.holds--
	o := p.download
	stop := p.readStarted && !p.kept && p.holds == 0 && (o == nil || !o.hasBegun())
	retire := p.holds == 0 && p.retireDue && o != nil
	p.mu.Unlock()
	if stop {
		p.stopContext()
	}
	if retire {
		o.retire()
	}
}

// readAt waits until p began its download, then reads its bytes [off, off+len(dst)) of
// the object into dst as they arrive. It returns the error p failed with, ctx's, or
// errPrefetchMissed when it ended without them.
func (p *cacheFill) readAt(ctx context.Context, off int64, dst []byte) error {
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

// retireDownload retires the download of p, done, once no hold is left.
func (p *cacheFill) retireDownload() {
	p.mu.Lock()
	p.retireDue = true
	o := p.download
	retire := p.holds == 0 && o != nil
	p.mu.Unlock()
	if retire {
		o.retire()
	}
}

// stop cancels p.
func (p *cacheFill) stop() {
	p.dropped.Store(true)
	p.stopContext()
}

func (p *cacheFill) joinableDownload() *objectDownload {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.download
}

// started gives p its download o, which reads can join from then.
func (p *cacheFill) started(o *objectDownload) {
	p.mu.Lock()
	p.download = o
	p.mu.Unlock()
	p.markReady()
}

// endLocked ends p with err, why it cached nothing: p leaves the waiting fills and the
// fills, unless its file is handed to its publisher, which forgets it once published, and
// frees what it was counted for. It reports false when p had ended.
func (d *DiskCache) endLocked(p *cacheFill, err error) bool {
	if p.done {
		return false
	}
	p.done = true
	f := &d.prefetchObjects
	if p.pos >= 0 {
		heap.Remove(&f.waitingFills, p.pos)
		if !p.isPromoted() {
			f.queuedBytes -= p.size()
		}
	}
	if !p.handedOff {
		f.forgetLocked(p)
		p.stopContext()
	}
	if p.background {
		d.server.endFill()
		p.background = false
	}
	if p.counted {
		p.counted = false
		f.reads.Done()
	}
	if p.queued && p.running && !p.whole {
		d.countPrefetch(p.req, err)
	}
	p.mu.Lock()
	buffered := p.download != nil && p.download.memory
	p.mu.Unlock()
	if buffered {
		p.retireDownload()
	}
	close(p.endedCh)
	p.markReady()
	return true
}

// finish ends p with err, why it cached nothing, and starts the waiting fills.
func (p *cacheFill) finish(err error) {
	f := &p.d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	p.d.endLocked(p, err)
	p.d.startFillsLocked()
}

// registerFill registers p, made by a read under ctx, running, and returns it held, for
// the caller to run. It returns instead, held, the fill holding the bytes of p that reads
// can join, promoting it for a foreground read; neither once they are cached, when
// caching is set; errPrefetchClosed once the cache is closed, and errPrefetchDeleted after
// a delete of the object counted since its deletes.
func (d *DiskCache) registerFill(ctx context.Context, p *cacheFill, caching bool) (*cacheFill, error) {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	closing, ok := f.openLocked()
	if !ok {
		return nil, errPrefetchClosed
	}
	if p.deletes != nil && d.deletes.since(p.diskPath, *p.deletes) {
		return nil, errPrefetchDeleted
	}
	if caching {
		if _, cached := d.ranges.covering(p.diskPath, p.start, p.size()); cached {
			return nil, nil
		}
	}
	if j := f.holdingLocked(p); j != nil {
		if requestClassOf(ctx) != backgroundRequest {
			d.promoteObjectLocked(j)
		}
		j.hold()
		return j, nil
	}
	p.closing, p.running, p.counted = closing, true, true
	f.registerLocked(p)
	f.reads.Add(1)
	p.hold()
	return p, nil
}

// run downloads the bytes of p, registered and running, and ends it: into a buffer when
// they are at most disk-prefetch-memory-max bytes and the buffer fits the memory of the
// server once its request holds room, then caches them unless skipCache is set; otherwise,
// unless noFile is set, into its cache file. admitted, when set, is called once its
// request holds room, or it is file-backed or ended. The closing of the cache or a delete
// of the object cancels it.
func (p *cacheFill) run(admitted func()) {
	d := p.d
	once := sync.OnceFunc(func() {
		if admitted != nil {
			admitted()
		}
	})
	defer once()
	defer context.AfterFunc(p.closing, p.cancel)()
	if d.memoryBacked(p.size(), requestClassOf(p.ctx) == backgroundRequest) {
		buffered, err := p.runBuffered(once)
		if buffered || err != nil || p.noFile {
			p.finish(err)
			return
		}
	} else if p.noFile {
		p.finish(nil)
		return
	}
	once()
	_, _ = d.downloadFile(p.ctx, p.filePath, p.start, p.end, p.src, p.finish)
}

// runBuffered reads the bytes of p into a buffer in one request, once its request holds
// room and the buffer fits the memory of the server, and caches them unless skipCache is
// set. It reports false, with no error, when the buffer does not fit.
func (p *cacheFill) runBuffered(admitted func()) (bool, error) {
	d := p.d
	room, err := acquireUntil(p.ctx, p.src.g, p.closing)
	if err != nil {
		return false, err
	}
	defer room.end()
	n, charge := p.size(), flightBufSize(p.size())
	if !d.server.takeMemory(charge, room.class) {
		return false, nil
	}
	o := newBufferDownload(getFlightBuf(n), d.server, charge, room.class)
	readDeletes := d.readDeletes()
	p.started(o)
	admitted()
	ctx, idle := newIdleReader(p.ctx)
	defer idle.Close()
	defer o.idle.add(idle)()
	_, err = p.src.g.doHeld(ctx, room, rangeRead{key: p.src.key, start: p.start, end: p.end, buf: o.buf, progress: fillProgress{download: o}, idle: idle})
	written := o.writtenFrom(0)
	failure := err
	if err != nil && (sourceStopped(err) || p.ctx.Err() != nil) {
		failure = nil
	}
	o.endWith(err == nil && written == n, failure)
	if err == nil && written == 0 {
		err = errSourceShort
	}
	if err == nil && !p.skipCache {
		err = d.Update(p.ctx, &IOVector{
			FilePath: p.filePath,
			Policy:   p.policy,
			Entries:  []IOEntry{{Offset: p.start, Size: written, Data: o.buf[:written], readDeletes: readDeletes}},
		}, false)
	}
	return true, err
}

// prefetchObjects is the fills of the disk cache, in every state, by the disk path of
// their object.
type prefetchObjects struct {
	mu     sync.Mutex
	m      map[string][]*cacheFill
	closed bool
	// waitingFills are the fills waiting to start, in order; queuedBytes are the bytes of
	// the waiting background fills.
	waitingFills fillHeap
	queuedBytes  int64
	fillSeq      uint64
	// fillsMax is the most background fills of the cache run at once, 0 for the server's
	// bound.
	fillsMax int
	// runWhole runs a fill of the whole object.
	runWhole func(context.Context, *cacheFill)
	reads    sync.WaitGroup // running fills
	// closing is canceled when the cache closes; it cancels the running fills.
	closing     context.Context
	cancelReads context.CancelFunc
}

func (f *prefetchObjects) registerLocked(p *cacheFill) {
	if f.m == nil {
		f.m = make(map[string][]*cacheFill)
	}
	f.m[p.diskPath] = append(f.m[p.diskPath], p)
}

func (f *prefetchObjects) forgetLocked(p *cacheFill) {
	fills := slices.DeleteFunc(f.m[p.diskPath], func(o *cacheFill) bool { return o == p })
	if len(fills) == 0 {
		delete(f.m, p.diskPath)
	} else {
		f.m[p.diskPath] = fills
	}
}

// openLocked returns the context the closing of the cache cancels, or false once the cache
// is closed.
func (f *prefetchObjects) openLocked() (context.Context, bool) {
	if f.closed {
		return nil, false
	}
	if f.closing == nil {
		f.closing, f.cancelReads = context.WithCancel(context.Background())
	}
	return f.closing, true
}

// heldLocked holds o, when set, for a read that joins it, promoting it, and returns it with
// the function releasing it once.
func heldLocked(o *cacheFill) (*cacheFill, func()) {
	if o == nil {
		return nil, func() {}
	}
	o.d.promoteObjectLocked(o)
	o.hold()
	return o, sync.OnceFunc(o.release)
}

// dropPrefetches ends the waiting fills of the object at diskPath, stops the others, and
// waits for them and their cache writes to end, or ctx.
func (d *DiskCache) dropPrefetches(ctx context.Context, diskPath string) error {
	f := &d.prefetchObjects
	f.mu.Lock()
	var running []*cacheFill
	for _, p := range slices.Clone(f.m[diskPath]) {
		if p.pos >= 0 {
			d.endLocked(p, nil)
		} else {
			running = append(running, p)
		}
	}
	d.startFillsLocked()
	f.mu.Unlock()
	for _, p := range running {
		p.stop()
	}
	for _, p := range running {
		select {
		case <-p.endedCh:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// closePrefetches ends the waiting fills, stops the others at once and waits for them, or
// ctx.
func (d *DiskCache) closePrefetches(ctx context.Context) {
	f := &d.prefetchObjects
	f.mu.Lock()
	f.closed = true
	for len(f.waitingFills) > 0 {
		d.endLocked(f.waitingFills[0], nil)
	}
	for _, fills := range f.m {
		for _, p := range fills {
			p.stop()
		}
	}
	cancelReads := f.cancelReads
	f.mu.Unlock()
	if cancelReads != nil {
		cancelReads()
	}
	done := make(chan struct{})
	go func() {
		f.reads.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
	}
}

// joinsLocked reports whether a read of the bytes [lo, hi) of the cache index names may
// join o: o is a live fill of that cache holding them that reads can join and, when cached
// is set, whose bytes end cached.
func joinsLocked(o *cacheFill, index bool, lo, hi int64, cached bool) bool {
	return o.start <= lo && hi <= o.end && o.index == index && o.live() && o.joinable() && o.hasBytes(lo, hi) &&
		(!cached || o.endsCached())
}

// joinableLocked returns the first fill of diskPath, in the order they registered, that
// joinsLocked allows for a read of [lo, hi).
func (f *prefetchObjects) joinableLocked(diskPath string, index bool, lo, hi int64, cached bool) *cacheFill {
	for _, o := range f.m[diskPath] {
		if joinsLocked(o, index, lo, hi, cached) {
			return o
		}
	}
	return nil
}

// coverPiece is a piece of a read's bytes, [start, end), and the fill holding them that
// the read joins, nil when none does.
type coverPiece struct {
	start, end int64
	o          *cacheFill
}

// coverLocked splits [lo, hi) into pieces in order: each run of bytes a fill of diskPath
// that joinsLocked allows holds, taking at each byte the one that holds the most bytes from
// it, and each run of bytes none holds.
func (f *prefetchObjects) coverLocked(diskPath string, index bool, lo, hi int64, cached bool) []coverPiece {
	var held []*cacheFill
	for _, o := range f.m[diskPath] {
		if o.start < hi && lo < o.end && joinsLocked(o, index, max(o.start, lo), min(o.end, hi), cached) {
			held = append(held, o)
		}
	}
	var pieces []coverPiece
	for pos := lo; pos < hi; {
		var best *cacheFill
		next := hi
		for _, o := range held {
			if o.start <= pos && pos < o.end {
				if best == nil || o.end > best.end {
					best = o
				}
			} else if o.start > pos {
				next = min(next, o.start)
			}
		}
		if best != nil {
			next = min(best.end, hi)
		}
		pieces = append(pieces, coverPiece{start: pos, end: next, o: best})
		pos = next
	}
	return pieces
}

// holdingLocked returns the fill other than p, of its object and cache, holding the bytes
// of p that p joins instead of downloading them: for a whole object, a live fill of the
// whole object; for a range, one reads can join that caches them, or a live fill of a
// range or of a whole object a read started.
func (f *prefetchObjects) holdingLocked(p *cacheFill) *cacheFill {
	for _, o := range f.m[p.diskPath] {
		if o == p || o.index != p.index || !o.live() {
			continue
		}
		if p.whole {
			if o.whole {
				return o
			}
			continue
		}
		if joinsLocked(o, p.index, p.start, p.end, !p.skipCache || p.joinsCaching) ||
			(o.readStarted || !o.whole) && o.start <= p.start && p.end <= o.end && !o.skipCache {
			return o
		}
	}
	return nil
}

type deletesMarkKey struct{}

// withDeletesMark returns ctx under which a whole-object download is not registered after
// a delete of its object counted since mark.
func withDeletesMark(ctx context.Context, mark uint64) context.Context {
	return context.WithValue(ctx, deletesMarkKey{}, mark)
}

// newWholeObject returns the fill of the whole object at diskPath, into the index-file
// cache when index is set: the fill of ctx when it is that fill, otherwise a new one,
// unregistered, downloaded under ctx and not registered after a delete of the object since
// the mark of ctx, or since now.
func (d *DiskCache) newWholeObject(ctx context.Context, diskPath string, index bool) *cacheFill {
	if p := fillOf(ctx); p != nil && p.whole && p.diskPath == diskPath && p.index == index && p.running {
		return p
	}
	deletes, ok := ctx.Value(deletesMarkKey{}).(uint64)
	if !ok {
		deletes = d.deletes.now()
	}
	p := &cacheFill{d: d, diskPath: diskPath, index: index, end: math.MaxInt64, whole: true, pos: -1, deletes: &deletes,
		ready: make(chan struct{}), endedCh: make(chan struct{}), promoted: make(chan struct{})}
	p.setContext(ctx)
	return p
}

// newContentFill returns the unregistered fill of the whole object at diskPath from the
// bytes the write path holds, which registers with its download.
func (d *DiskCache) newContentFill(diskPath string) *cacheFill {
	p := &cacheFill{d: d, diskPath: diskPath, end: math.MaxInt64, whole: true, content: true, pos: -1,
		ready: make(chan struct{}), endedCh: make(chan struct{}), promoted: make(chan struct{})}
	p.setContext(context.Background())
	return p
}

// splitParts sets the end of p, the fill of a whole object, at the object's size layout
// gives, and returns the bounds of the parts it reads at once and the part a read needs
// first, or nil to read it as one range.
func (p *cacheFill) splitParts(layout objectLayout) (bounds []int64, lead int) {
	if layout == nil {
		return nil, 0
	}
	size, splitAt, err := layout(p.ctx)
	if err != nil {
		return nil, 0
	}
	p.endAt(size)
	b := p.d.objectSplit.bounds(size, splitAt)
	return b, leadPart(b, splitAt)
}

// wholeFill returns the registered fill of the whole object at diskPath whose download
// started, of the index-file cache when index is set, or nil.
func (d *DiskCache) wholeFill(diskPath string, index bool) *cacheFill {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.wholeFillLocked(diskPath, index)
}

func (f *prefetchObjects) wholeFillLocked(diskPath string, index bool) *cacheFill {
	for _, p := range f.m[diskPath] {
		if p.whole && p.index == index && p.joinableDownload() != nil {
			return p
		}
	}
	return nil
}

// endAt sets the end of p, the fill of a whole object, at the object's size.
func (p *cacheFill) endAt(size int64) {
	if size <= 0 {
		return
	}
	f := &p.d.prefetchObjects
	f.mu.Lock()
	p.end = size
	f.mu.Unlock()
}

// admitWhole registers p, a whole-object download, unless a prefetch registered it. It
// reports false when the download is skipped, as for a whole object another fill of its
// cache downloads or publishes, or with why it stopped; a delete of the object or the
// closing of the cache stops it as a cancel does.
func (p *cacheFill) admitWhole() (bool, error) {
	d := p.d
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := p.ctx.Err(); err != nil {
		return false, err
	}
	closing, ok := f.openLocked()
	if !ok || d.deletes.since(p.diskPath, *p.deletes) {
		return false, context.Canceled
	}
	if j := f.holdingLocked(p); j != nil {
		if requestClassOf(p.ctx) != backgroundRequest {
			d.promoteObjectLocked(j)
		}
		return false, nil
	}
	if !p.running {
		p.closing, p.running, p.counted = closing, true, true
		f.registerLocked(p)
		f.reads.Add(1)
	}
	p.stopOnClose = context.AfterFunc(p.closing, p.cancel)
	return true, nil
}

// endWhole ends p, a whole-object download, with why it cached nothing.
func (p *cacheFill) endWhole(err error) {
	if p.stopOnClose == nil {
		if !p.running {
			p.cancel()
		}
		return
	}
	p.stopOnClose()
	p.finish(err)
}

// downloadFile downloads under ctx, the context of a fill, its bytes [start, end) of the
// object at filePath from src, waits for the publication of their file, and ends it
// through done.
func (d *DiskCache) downloadFile(ctx context.Context, filePath string, start, end int64, src objectSource, done func(error)) (bool, error) {
	var written bool
	var err error
	defer func() { done(err) }()
	readDeletes := d.readDeletes()
	diskPath := d.pathForIOEntry(filePath, IOEntry{Offset: start, Size: end - start})
	// objectEnd is where a source ended before its range did: the object's size.
	var objectEnd atomic.Int64
	src.start, src.end, src.objectEnd = start, end, &objectEnd
	openReader := func(context.Context) (io.ReadCloser, error) { return &src, nil }
	layout := func(context.Context) (int64, int64, error) { return end - start, 0, nil }
	mark := readMark{object: d.objectKey(filePath), deletes: readDeletes - 1, set: true}
	written, err = d.writeFileTo(ctx, diskPath, openReader, false, rangeFill, layout, diskFillTarget{d: d, diskPath: diskPath}, mark)
	if err != nil {
		if size := objectEnd.Load(); size > start {
			// A later run of the object ends at its size.
			d.storePrefetchedSize(d.objectKey(filePath), size, readDeletes-1)
		}
		err = d.observeWriteResult(diskPath, err)
		return false, err
	}
	return written, nil
}

// fileFill returns the live file-backed fill of the object at objectPath, of the
// index-file cache when index is set, holding the bytes [lo, hi), and its download.
func (d *DiskCache) fileFill(objectPath string, index bool, lo, hi int64) (*cacheFill, *objectDownload) {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, p := range f.m[objectPath] {
		if o := p.joinableDownload(); o != nil && !o.memory && joinsLocked(p, index, lo, hi, false) {
			return p, o
		}
	}
	return nil, nil
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

// writePrefetchObjects writes the fills of d, one a line: its backing, object, bytes and
// state, then the bytes of the waiting background fills and the room and memory of the
// server. It copies them under the lock and writes them after.
func (d *DiskCache) writePrefetchObjects(w io.Writer) {
	f := &d.prefetchObjects
	f.mu.Lock()
	var lines []string
	for diskPath, fills := range f.m {
		for _, p := range fills {
			kind, state := "fill", "waiting for room"
			if o := p.joinableDownload(); o != nil {
				kind, state = map[bool]string{true: "memory", false: "file"}[o.memory], "running"
			}
			if p.pos >= 0 {
				state = "waiting"
			}
			if p.isPromoted() {
				state += ", foreground"
			}
			lines = append(lines, fmt.Sprintf("  %s %s [%d, %d) %s", kind, diskPath, p.start, p.end, state))
		}
	}
	queued := f.queuedBytes
	f.mu.Unlock()
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
