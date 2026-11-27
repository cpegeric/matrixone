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
	"hash/maphash"
	"os"
	pathpkg "path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// cachedRange is a range of an object cached in a file of its own.
type cachedRange struct {
	diskPath string
	offset   int64
	size     int64
}

// cachedRanges indexes the cached range files by object, so a read can find a cached
// range that contains its own. Reads take no lock: each object's ranges are a snapshot
// that add and remove replace.
type cachedRanges struct {
	mu       sync.Mutex        // held by add and remove
	byObject sync.Map          // object key -> *rangeSnapshot
	objectOf map[string]string // diskPath -> object key, under mu
}

// rangeSnapshot is the cached ranges of an object, sorted by offset. reach[i] is the index
// of the range among the first i+1 that reaches furthest.
type rangeSnapshot struct {
	ranges []cachedRange
	reach  []int
}

// with returns a snapshot of the ranges of s and r: r is inserted where it sorts, and the
// reach is recomputed from there.
func (s *rangeSnapshot) with(r cachedRange) *rangeSnapshot {
	var old []cachedRange
	var oldReach []int
	if s != nil {
		old, oldReach = s.ranges, s.reach
	}
	at, _ := slices.BinarySearchFunc(old, r.offset, func(x cachedRange, t int64) int { return cmp.Compare(x.offset, t) })
	ranges := make([]cachedRange, len(old)+1)
	copy(ranges, old[:at])
	ranges[at] = r
	copy(ranges[at+1:], old[at:])
	reach := make([]int, len(ranges))
	copy(reach, oldReach[:at])
	fillReach(ranges, reach, at)
	return &rangeSnapshot{ranges: ranges, reach: reach}
}

// without returns a snapshot of the ranges of s but the one at index at, the reach
// recomputed from there.
func (s *rangeSnapshot) without(at int) *rangeSnapshot {
	ranges := slices.Delete(slices.Clone(s.ranges), at, at+1)
	reach := make([]int, len(ranges))
	copy(reach, s.reach[:at])
	fillReach(ranges, reach, at)
	return &rangeSnapshot{ranges: ranges, reach: reach}
}

// fillReach sets reach[i], for i from from on, to the index of the range among the first
// i+1 that reaches furthest.
func fillReach(ranges []cachedRange, reach []int, from int) {
	for i := from; i < len(ranges); i++ {
		reach[i] = i
		if i > 0 && ranges[reach[i-1]].offset+ranges[reach[i-1]].size >= ranges[i].offset+ranges[i].size {
			reach[i] = reach[i-1]
		}
	}
}

func (c *cachedRanges) snapshot(key string) *rangeSnapshot {
	if v, ok := c.byObject.Load(key); ok {
		return v.(*rangeSnapshot)
	}
	return nil
}

// startingBy returns the number of ranges that start at or before pos.
func (s *rangeSnapshot) startingBy(pos int64) int {
	n, _ := slices.BinarySearchFunc(s.ranges, pos+1, func(r cachedRange, t int64) int { return cmp.Compare(r.offset, t) })
	return n
}

// objectKey returns the key of the object at filePath: the disk path of its range files
// without the offset and size. No file is published under it.
func (d *DiskCache) objectKey(filePath string) string {
	filePath = pathpkg.Clean(filePath)
	return filepath.Join(d.path, diskCacheShard(filePath), toOSPath(filePath)+cacheFileSuffix)
}

// parseRangeFile returns the object key, offset and size of the range file at diskPath
// under the cache directory root. The size digits run into the object path, so the size
// is the prefix of them that equals the file's length.
func parseRangeFile(root, diskPath string, length int64) (key string, offset, size int64, ok bool) {
	rel, err := filepath.Rel(root, diskPath)
	if err != nil {
		return "", 0, 0, false
	}
	shard, name, cut := strings.Cut(rel, string(filepath.Separator))
	if !cut || !isDiskCacheShard(shard) || !strings.HasSuffix(name, cacheFileSuffix) {
		return "", 0, 0, false
	}
	dash := strings.IndexByte(name, '-')
	if dash <= 0 {
		return "", 0, 0, false
	}
	offset, err = strconv.ParseInt(name[:dash], 10, 64)
	if err != nil {
		return "", 0, 0, false
	}
	digits := name[dash+1:]
	for i := 1; i <= len(digits) && digits[i-1] >= '0' && digits[i-1] <= '9'; i++ {
		size, err = strconv.ParseInt(digits[:i], 10, 64)
		if err != nil || size > length {
			break
		}
		object := strings.TrimSuffix(digits[i:], cacheFileSuffix)
		if size == length && object != "" && diskCacheShard(fromOSPath(object)) == shard {
			return filepath.Join(root, shard, digits[i:]), offset, size, true
		}
	}
	return "", 0, 0, false
}

// add indexes the range file at diskPath under root, if it is one.
func (c *cachedRanges) add(root, diskPath string) {
	if !strings.HasSuffix(diskPath, cacheFileSuffix) {
		return
	}
	if rel, err := filepath.Rel(root, diskPath); err != nil {
		return
	} else if _, name, _ := strings.Cut(rel, string(filepath.Separator)); name == "" || name[0] < '0' || name[0] > '9' {
		return
	}
	info, err := os.Stat(diskPath)
	if err != nil {
		return
	}
	key, offset, size, ok := parseRangeFile(root, diskPath, info.Size())
	if !ok {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.objectOf[diskPath]; ok {
		return
	}
	if c.objectOf == nil {
		c.objectOf = make(map[string]string)
	}
	c.objectOf[diskPath] = key
	c.byObject.Store(key, c.snapshot(key).with(cachedRange{diskPath: diskPath, offset: offset, size: size}))
}

// remove forgets the range file at diskPath.
func (c *cachedRanges) remove(diskPath string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	key, ok := c.objectOf[diskPath]
	if !ok {
		return
	}
	delete(c.objectOf, diskPath)
	old := c.snapshot(key)
	if old == nil {
		return
	}
	at := slices.IndexFunc(old.ranges, func(r cachedRange) bool { return r.diskPath == diskPath })
	if at < 0 {
		return
	}
	if len(old.ranges) == 1 {
		c.byObject.Delete(key)
		return
	}
	c.byObject.Store(key, old.without(at))
}

// covering returns cached ranges of the object with key that together hold size bytes
// at offset, in order, each starting where the one before it ends or earlier, or false
// if the cached ranges leave a gap. Each range is the one that reaches furthest of those
// holding the byte where it starts.
func (c *cachedRanges) covering(key string, offset, size int64) ([]cachedRange, bool) {
	s := c.snapshot(key)
	if s == nil {
		return nil, false
	}
	var out []cachedRange
	end := offset + size
	for pos := offset; pos < end; {
		n := s.startingBy(pos)
		if n == 0 {
			return nil, false
		}
		best := s.ranges[s.reach[n-1]]
		if best.offset+best.size <= pos {
			return nil, false
		}
		out = append(out, best)
		pos = best.offset + best.size
	}
	return out, true
}

// containing returns the smallest cached range of the object with key that contains
// size bytes at offset, passing over the range files in tried.
func (c *cachedRanges) containing(key string, offset, size int64, tried []string) (cachedRange, bool) {
	s := c.snapshot(key)
	if s == nil {
		return cachedRange{}, false
	}
	return smallestContaining(s.ranges[:s.startingBy(offset)], offset, size, tried)
}

// startsAt0 reports whether a cached range of the object with key starts at offset 0.
func (c *cachedRanges) startsAt0(key string) bool {
	s := c.snapshot(key)
	return s != nil && len(s.ranges) > 0 && s.ranges[0].offset == 0
}

// of returns a copy of the cached ranges of the object with key.
func (c *cachedRanges) of(key string) []cachedRange {
	if s := c.snapshot(key); s != nil {
		return slices.Clone(s.ranges)
	}
	return nil
}

// smallestContaining returns the smallest of ranges that contains size bytes at offset,
// passing over the range files in tried.
func smallestContaining(ranges []cachedRange, offset, size int64, tried []string) (cachedRange, bool) {
	var found cachedRange
	ok := false
	for _, r := range ranges {
		if r.offset <= offset && offset+size <= r.offset+r.size && (!ok || r.size < found.size) &&
			!slices.Contains(tried, r.diskPath) {
			found, ok = r, true
		}
	}
	return found, ok
}

// objectDeletesMax is the most deleted objects remembered; past it they are forgotten.
const objectDeletesMax = 1 << 16

// objectDeletesStripes is the number of locks the publications and deletes of objects
// share by their disk paths.
const objectDeletesStripes = 64

// objectDeletes counts the deletes of objects, so an entry read before its object was
// deleted is not cached after the delete.
type objectDeletes struct {
	count atomic.Uint64
	mu    sync.Mutex
	// last holds, by object disk path, the count after the object's last delete.
	last map[string]uint64
	// floor is the count when last was cleared: earlier deletes are forgotten.
	floor uint64
	// loading keeps every delete while the cache loads its files: a file found by the load
	// is indexed only when its object was not deleted since the cache was created.
	loading bool
	// stripes are held shared by the publications of entries and exclusively by the
	// counting of deletes, by object disk path.
	stripes [objectDeletesStripes]sync.RWMutex
}

var objectDeletesSeed = maphash.MakeSeed()

func (o *objectDeletes) stripe(diskPath string) *sync.RWMutex {
	return &o.stripes[maphash.String(objectDeletesSeed, diskPath)%objectDeletesStripes]
}

func (o *objectDeletes) now() uint64 {
	return o.count.Load()
}

// deleted counts a delete of the object at diskPath, after the publications of its
// entries under way end.
func (o *objectDeletes) deleted(diskPath string) {
	stripe := o.stripe(diskPath)
	stripe.Lock()
	defer stripe.Unlock()
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.last) >= objectDeletesMax && !o.loading {
		clear(o.last)
		o.floor = o.count.Load()
	}
	if o.last == nil {
		o.last = make(map[string]uint64)
	}
	o.last[diskPath] = o.count.Add(1)
}

// setLoading sets whether the cache loads its files.
func (o *objectDeletes) setLoading(loading bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.loading = loading
}

// since reports whether the object at diskPath may have been deleted after the count
// was start.
func (o *objectDeletes) since(diskPath string, start uint64) bool {
	if o.count.Load() == start {
		return false
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if start < o.floor {
		return true
	}
	return o.last[diskPath] > start
}

// storePrefetchedSize stores the size of the object at objectPath, told by a prefetch made
// when the delete count was mark, unless the object was deleted since. The delete of the
// object forgets a size stored before it.
func (d *DiskCache) storePrefetchedSize(objectPath string, size int64, mark uint64) {
	stripe := d.deletes.stripe(objectPath)
	stripe.RLock()
	defer stripe.RUnlock()
	if !d.deletes.since(objectPath, mark) {
		d.prefetched.store(objectPath, size)
	}
}

// readDeletes returns the mark of an entry read now: 1 plus the delete count.
func (d *DiskCache) readDeletes() uint64 {
	return d.deletes.now() + 1
}

// publishUnlessDeleted runs write, which caches entry once the claim it is given holds,
// with a claim that also fails when the object at filePath was deleted after the entry
// was read. A delete is counted after the publications claimed end, so its removal of the
// object's cached files finds them.
func (d *DiskCache) publishUnlessDeleted(
	filePath string,
	entry IOEntry,
	claim func() bool,
	write func(claim func() bool) (bool, error),
) (bool, error) {
	if entry.readDeletes == 0 {
		return write(claim)
	}
	path, err := ParsePath(filePath)
	if err != nil {
		return write(claim)
	}
	return d.publishUnlessDeletedSince(d.objectKey(path.File), entry.readDeletes-1, claim, write)
}

// publishUnlessDeletedSince runs write with a claim that also fails when the object at
// objectPath was deleted after the delete count was mark; it reports false without the
// error of write when it was.
func (d *DiskCache) publishUnlessDeletedSince(
	objectPath string,
	mark uint64,
	claim func() bool,
	write func(claim func() bool) (bool, error),
) (bool, error) {
	stripe := d.deletes.stripe(objectPath)
	locked, deleted := false, false
	written, err := write(func() bool {
		if claim != nil && !claim() {
			return false
		}
		stripe.RLock()
		locked = true
		deleted = d.deletes.since(objectPath, mark)
		return !deleted
	})
	if locked {
		stripe.RUnlock()
	}
	if deleted {
		return false, nil
	}
	return written, err
}

// diskCacheMissingPathsMax is the most paths noted missing; past it they are forgotten.
const diskCacheMissingPathsMax = 1 << 16

// openCacheFile is a cache file kept open for the reads of its path.
type openCacheFile struct {
	file *os.File
	// refs counts the reads holding the file.
	refs int
	// dropped marks a file whose path changed; it is closed once no read holds it.
	dropped bool
	// used is when a read last took the file, in takes.
	used uint64
}

// openFilePool keeps, by disk path, the cache files reads opened, so the reads after
// them do not open them again. A change of a path drops its file. It is guarded by the
// lock of the disk cache's pathOwners.
type openFilePool struct {
	m map[string]*openCacheFile
	// missing are the paths a read found no file at, until they change.
	missing map[string]struct{}
	takes   uint64
	// limit, when set, is the most files kept instead of DefaultDiskOpenFilesMax; less than
	// 0 keeps none and notes no path missing.
	limit int
}

// takeLocked returns the file kept open for path, held for a read, or nil.
func (o *openFilePool) takeLocked(path string) *openCacheFile {
	f := o.m[path]
	if f == nil {
		return nil
	}
	f.refs++
	o.takes++
	f.used = o.takes
	return f
}

// keepLocked keeps file open for path, held for the read that opened it. With no file to
// keep, or the most files kept and all held, it keeps nothing and returns nil.
func (o *openFilePool) keepLocked(path string, file *os.File) (kept *openCacheFile, closeFile *os.File) {
	if o.limit < 0 || o.m[path] != nil {
		return nil, nil
	}
	if len(o.m) >= cmp.Or(o.limit, DefaultDiskOpenFilesMax) {
		var lru string
		var oldest *openCacheFile
		for p, f := range o.m {
			if f.refs == 0 && (oldest == nil || f.used < oldest.used) {
				lru, oldest = p, f
			}
		}
		if oldest == nil {
			return nil, nil
		}
		delete(o.m, lru)
		closeFile = oldest.file
	}
	if o.m == nil {
		o.m = make(map[string]*openCacheFile)
	}
	o.takes++
	kept = &openCacheFile{file: file, refs: 1, used: o.takes}
	o.m[path] = kept
	return kept, closeFile
}

// releaseLocked ends a read's hold of f, and returns the file to close when f was
// dropped and no read holds it any more.
func (o *openFilePool) releaseLocked(f *openCacheFile) *os.File {
	f.refs--
	if f.dropped && f.refs == 0 {
		return f.file
	}
	return nil
}

// missingLocked reports whether path was noted missing.
func (o *openFilePool) missingLocked(path string) bool {
	_, ok := o.missing[path]
	return ok
}

// noteMissingLocked notes that path has no file.
func (o *openFilePool) noteMissingLocked(path string) {
	if o.limit < 0 {
		return
	}
	if o.missing == nil || len(o.missing) >= diskCacheMissingPathsMax {
		o.missing = make(map[string]struct{})
	}
	o.missing[path] = struct{}{}
}

// dropLocked forgets the file of path, whose path changes, and returns it to close when
// no read holds it.
func (o *openFilePool) dropLocked(path string) *os.File {
	delete(o.missing, path)
	f := o.m[path]
	if f == nil {
		return nil
	}
	delete(o.m, path)
	f.dropped = true
	if f.refs == 0 {
		return f.file
	}
	return nil
}

// dropAllLocked forgets every file, and returns those no read holds to close. No file is
// kept after it.
func (o *openFilePool) dropAllLocked() (files []*os.File) {
	o.missing = nil
	o.limit = -1
	for p := range o.m {
		if f := o.dropLocked(p); f != nil {
			files = append(files, f)
		}
	}
	return files
}
