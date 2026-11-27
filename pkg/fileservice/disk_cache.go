// Copyright 2022 Matrix Origin
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
	"cmp"
	"context"
	"errors"
	"fmt"
	"hash/maphash"
	"io"
	"io/fs"
	"math"
	"os"
	pathpkg "path"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/fileservice/lrucache"
	"github.com/matrixorigin/matrixone/pkg/logutil"
	"github.com/matrixorigin/matrixone/pkg/perfcounter"
	metric "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
	"go.uber.org/zap"
)

type DiskCache struct {
	path               string
	cacheDataAllocator CacheDataAllocator
	perfCounterSets    []*perfcounter.CounterSet

	files openFilePool

	cache        *lrucache.Cache[string, struct{}]
	capacityFunc fscache.CapacityFunc
	preloads     *preloadUsage
	ranges       cachedRanges
	dirEntries   bool
	// held counts the holds on entry directories.
	held dirHolds
	// trash holds the entry directories whose removal failed.
	trash       cacheTrash
	directoryMu sync.RWMutex
	// diskFullEviction is held by the single eviction that responds to
	// disk-full write failures.
	diskFullEviction atomic.Bool
	prefetchRange    prefetchRangeConfig
	objectSplit      objectSplitConfig
	// memoryPrefetchMax is the most bytes of a MemoryPrefetchObject.
	memoryPrefetchMax int64
	prefetched        prefetchedObjects
	// server makes the requests of fills; nil when the cache has no object storage.
	server          *objectStorageServer
	prefetchObjects prefetchObjects
	// wholeFills counts whole-object fills in progress; wholeFileMax is the size of
	// the largest whole-object file written.
	wholeFills   atomic.Int64
	wholeFileMax atomic.Int64
	// wholePublications holds, by range file, the whole-object fills publishing it.
	wholePublications sync.Map
	// statFilesystem overrides statDiskCacheFilesystem in tests.
	statFilesystem func(path string) (diskCacheFilesystemStat, bool)

	// pub publishes the files and entry directories written into the cache.
	pub           publisher
	writeFailures diskCacheWriteFailureState
	fileSync      func(*os.File) error
	load          struct {
		cancel context.CancelFunc
		done   chan struct{}
	}
	hooks diskCacheHooks
	// deletes counts the deletes of objects; see objectDeletes.
	deletes objectDeletes
}

// diskCacheHooks are test-only phase barriers; production instances leave them nil.
type diskCacheHooks struct {
	beforeLoadDirectoryCleanup func(path string)
	beforeCacheTempFileCreate  func()
	// beforeIndexLoadedDirs runs before the entry directories found by the load are
	// indexed, beforeLoadFile before a file found by the load is, and
	// beforeRemoveUnlistedDir before an entry directory the load cannot list is removed.
	beforeIndexLoadedDirs   func()
	beforeLoadFile          func(diskPath string)
	beforeRemoveUnlistedDir func(path string)
	// beforeFilePublication, afterPublicationClaim and afterFilePublication run before a
	// file's publication claim, between it and its caching, and after its caching.
	beforeFilePublication func()
	afterPublicationClaim func()
	afterFilePublication  func()
	// beforeCacheDirFileOpen runs between the files of one entry open.
	beforeCacheDirFileOpen func()
}

func runHook(hook func()) {
	if hook != nil {
		hook()
	}
}

func runPathHook(hook func(path string), path string) {
	if hook != nil {
		hook(path)
	}
}

func NewDiskCache(
	ctx context.Context,
	path string,
	capacity fscache.CapacityFunc,
	perfCounterSets []*perfcounter.CounterSet,
	asyncLoad bool,
	cacheDataAllocator CacheDataAllocator,
	name string,
) (ret *DiskCache, err error) {
	return newDiskCacheWithMetricScope(ctx, path, capacity, perfCounterSets, asyncLoad, cacheDataAllocator, name, "")
}

func newDiskCacheWithMetricScope(
	ctx context.Context,
	path string,
	capacity fscache.CapacityFunc,
	perfCounterSets []*perfcounter.CounterSet,
	asyncLoad bool,
	cacheDataAllocator CacheDataAllocator,
	name string,
	metricScope string,
) (ret *DiskCache, err error) {
	return newDiskCache(ctx, path, capacity, perfCounterSets, asyncLoad, cacheDataAllocator, name, metricScope,
		diskCacheOptions{})
}

type diskCacheOptions struct {
	// dirEntries makes directories named with cacheDirSuffix cache entries
	// (IndexFileCache), never evicted while held, and whose
	// capacity does not follow GlobalDiskCacheSizeHint.
	dirEntries bool
	// prefetchRange sets the prefetch ranges of range reads.
	prefetchRange prefetchRangeConfig
	// objectSplit sets the parts a whole-object download reads at once.
	objectSplit objectSplitConfig
	// memoryPrefetchMax is the most bytes of a MemoryPrefetchObject; 0 is
	// DefaultDiskPrefetchMemoryMax.
	memoryPrefetchMax int64
	// openFilesMax is the most cache files kept open for later reads; 0 is
	// DefaultDiskOpenFilesMax, and less keeps none.
	openFilesMax int
}

func newDiskCache(
	ctx context.Context,
	path string,
	capacity fscache.CapacityFunc,
	perfCounterSets []*perfcounter.CounterSet,
	asyncLoad bool,
	cacheDataAllocator CacheDataAllocator,
	name string,
	metricScope string,
	opts diskCacheOptions,
) (ret *DiskCache, err error) {

	err = os.MkdirAll(path, 0755)
	if err != nil {
		return nil, err
	}

	if cacheDataAllocator == nil {
		cacheDataAllocator = DefaultCacheDataAllocator()
	}

	inuseBytes, capacityBytes := metric.GetFsCacheBytesGaugeWithScope(metricScope, name, "disk")
	capacityBytes.Set(float64(capacity()))

	capacityFunc := func() int64 {
		// read from global size hint
		if n := GlobalDiskCacheSizeHint.Load(); n > 0 && !opts.dirEntries {
			return n
		}
		// fallback
		c := max(capacity()-ret.trash.bytes.Load(), 0)
		if l := ret.trash.limit.Load(); l >= 0 {
			c = min(c, l)
		}
		return c
	}

	ret = &DiskCache{
		path:               path,
		cacheDataAllocator: cacheDataAllocator,
		perfCounterSets:    perfCounterSets,
		fileSync:           func(file *os.File) error { return file.Sync() },

		capacityFunc:      capacityFunc,
		dirEntries:        opts.dirEntries,
		prefetchRange:     opts.prefetchRange,
		objectSplit:       opts.objectSplit,
		memoryPrefetchMax: cmp.Or(opts.memoryPrefetchMax, DefaultDiskPrefetchMemoryMax),
		preloads:          newPreloadUsage(),
	}
	ret.trash.limit.Store(-1)

	postSet := func(_ context.Context, path string, _ struct{}, size int64, _ uint64) {
		// The file of a whole object is the one its fill publishes.
		if v, ok := ret.wholePublications.Load(path); ok {
			pub := v.(wholePublication)
			o := pub.download
			ret.preloads.cachedDownload(path, pub.key, o)
			if n := o.creditTo(func(n int64) { ret.preloads.readDownload(path, o, n) }); n > 0 {
				ret.preloads.readDownload(path, o, n)
			}
		}
		ret.ranges.add(ret.path, path)
		inuseBytes.Add(float64(size))
		capacityBytes.Set(float64(capacityFunc()))
	}
	prepareEvict := func(path string, _ struct{}, size int64, _ uint64) func() {
		object := ret.ranges.objectOfFile(path)
		return func() {
			ret.pub.queue(object, queuedJob{"evict", path}, func() error {
				err := ret.removeEvictedFile(path, size)
				if err != nil {
					logutil.Error("delete disk cache file",
						zap.Any("error", err),
					)
				}
				return err
			})
		}
	}
	postEvict := func(_ context.Context, path string, _ struct{}, size int64, _ uint64) {
		ret.preloads.removed(path, size)
		ret.ranges.remove(path)
		// A read may have cached the file again since it was evicted.
		if ret.cache.Contains(path) {
			ret.ranges.add(ret.path, path)
		}
		inuseBytes.Add(float64(-size))
		capacityBytes.Set(float64(capacityFunc()))
	}
	ret.cache = lrucache.NewWithPrepareEvict(capacityFunc, postSet, nil, prepareEvict, postEvict)
	if opts.dirEntries {
		ret.cache.SetEvictable(func(path string) bool { return !ret.dirHeld(path) })
	}
	ret.files.limit = opts.openFilesMax
	ret.pub.init(ret)
	ret.initWriteFailureState()
	loadCtx, loadCancel := context.WithCancel(ctx)
	ret.load.cancel = loadCancel
	ret.load.done = make(chan struct{})
	ret.deletes.setLoading(true)
	runLoad := func() {
		defer func() {
			ret.deletes.setLoading(false)
			loadCancel()
			close(ret.load.done)
		}()
		ret.loadCache(loadCtx)
	}

	if asyncLoad {
		go runLoad()
	} else {
		runLoad()
	}

	if name != "" {
		allDiskCaches.Store(ret, name)
	}

	return ret, nil
}

// loadFile indexes the cache file at diskPath found at startup, unless its object was
// deleted since the cache was created: the delete may have missed a file not yet
// indexed, so the file is removed instead (see removeLoadedFile). As for the publication
// of an entry, a delete counted later finds the file indexed.
func (d *DiskCache) loadFile(ctx context.Context, diskPath string, info fs.FileInfo) {
	objectPath := diskPath
	if key, _, _, ok := parseRangeFile(d.path, diskPath, info.Size()); ok {
		objectPath = key
	}
	runPathHook(d.hooks.beforeLoadFile, diskPath)
	if !d.indexLoadedFile(ctx, objectPath, diskPath, info) {
		d.removeLoadedFile(objectPath, diskPath, info)
	}
}

// indexLoadedFile indexes the file at diskPath found at startup and reports true, unless
// the object at objectPath was deleted since the cache was created.
func (d *DiskCache) indexLoadedFile(ctx context.Context, objectPath, diskPath string, info fs.FileInfo) bool {
	stripe := d.deletes.stripe(objectPath)
	stripe.RLock()
	defer stripe.RUnlock()
	if d.deletes.since(objectPath, 0) {
		return false
	}
	d.cache.SetOldest(ctx, diskPath, struct{}{}, int64(fileSize(info)))
	return true
}

// removeLoadedFile has the publisher remove the file at diskPath of the object at
// objectPath, found at startup, while it is not indexed and is still the file info
// describes.
func (d *DiskCache) removeLoadedFile(objectPath, diskPath string, info fs.FileInfo) {
	d.pub.queue(objectPath, queuedJob{"unload", diskPath}, func() error {
		if !d.cache.Contains(diskPath) && sameFileAt(info, diskPath) {
			_ = os.Remove(diskPath)
			d.files.changed(diskPath)
		}
		return nil
	})
}

// sameFileAt reports whether info describes the file at path.
func sameFileAt(info fs.FileInfo, path string) bool {
	cur, err := os.Lstat(path)
	return err == nil && os.SameFile(info, cur)
}

// reindexRead reports whether the cache file at diskPath that a read opened and the cache
// does not hold may be read, and has the publisher index it while it is still that file.
// As in loadFile, a file of an object deleted since the cache was created is neither read
// nor indexed: a delete during the load may have missed it.
func (d *DiskCache) reindexRead(ctx context.Context, diskPath, objectPath string, file *os.File) (bool, error) {
	if d.deletes.since(objectPath, 0) {
		return false, nil
	}
	LogEvent(ctx, str_disk_cache_file_stat_begin)
	stat, err := file.Stat()
	LogEvent(ctx, str_disk_cache_file_stat_end)
	if err != nil {
		return false, err
	}
	d.pub.queue(objectPath, queuedJob{"reindex", diskPath}, func() error {
		if !d.cache.Contains(diskPath) && !d.deletes.since(objectPath, 0) && sameFileAt(stat, diskPath) {
			d.cache.Set(context.Background(), diskPath, struct{}{}, fileSize(stat))
		}
		return nil
	})
	return true, nil
}

func (d *DiskCache) loadCache(ctx context.Context) {
	t0 := time.Now()
	d.startLoading()

	type Info struct {
		Path  string
		Entry os.DirEntry
		Dir   bool
	}
	works := make(chan Info)

	var (
		dirsMu sync.Mutex
		dirs   []loadedDir
	)

	numWorkers := runtime.NumCPU()
	wg := new(sync.WaitGroup)
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case work, ok := <-works:
					if !ok {
						return
					}
					if work.Dir {
						size, err := cacheDirSize(work.Path)
						if os.IsPermission(err) {
							d.removeUnlistedDir(ctx, work.Path)
							continue
						}
						if err != nil {
							continue // ignore
						}
						info, err := work.Entry.Info()
						if err != nil {
							continue // ignore
						}
						d.loadingDir(work.Path, size)
						dirsMu.Lock()
						dirs = append(dirs, loadedDir{path: work.Path, size: size, lastUse: info.ModTime()})
						dirsMu.Unlock()
						continue
					}
					info, err := work.Entry.Info()
					if err != nil {
						continue // ignore
					}
					d.loadFile(ctx, work.Path, info)
				}
			}
		}()
	}

	var numFiles, numCacheFiles, numTempFiles, numDeleted int

	_ = filepath.WalkDir(d.path, func(path string, entry os.DirEntry, err error) error {
		if ctx.Err() != nil {
			return fs.SkipAll
		}
		numFiles++
		if err != nil {
			return nil //ignore
		}

		if entry.IsDir() {
			if path == d.path {
				return nil
			}
			switch name := entry.Name(); {
			case !d.dirEntries:
			case strings.HasSuffix(name, cacheDirTrashSuffix):
				_ = d.removeTrash(path, trashSize(path))
				numDeleted++
				return fs.SkipDir
			case strings.HasSuffix(name, cacheDirTempSuffix):
				numTempFiles++
				info, err := entry.Info()
				if err == nil && time.Since(info.ModTime()) > time.Hour*8 {
					_ = os.RemoveAll(path)
					numDeleted++
				}
				return fs.SkipDir
			case strings.HasSuffix(name, cacheDirSuffix):
				numCacheFiles++
				select {
				case works <- Info{Path: path, Entry: entry, Dir: true}:
				case <-ctx.Done():
					return fs.SkipAll
				}
				return fs.SkipDir
			}
			d.removeEmptyDirectory(path)
			return nil

		} else {
			// plain files
			if d.isOldWholeFile(path) {
				_ = os.Remove(path)
				numDeleted++
				return nil
			}
			if !strings.HasSuffix(entry.Name(), cacheFileSuffix) {
				// not cache file
				if strings.HasSuffix(entry.Name(), cacheFileTempSuffix) {
					numTempFiles++
					// temp file
					info, err := entry.Info()
					if err == nil && time.Since(info.ModTime()) > time.Hour*8 {
						// old temp file
						_ = os.Remove(path)
						numDeleted++
					}
				} else {
					// unknown file
					_ = os.Remove(path)
					numDeleted++
				}
				return nil
			}
		}

		numCacheFiles++
		select {
		case works <- Info{
			Path: path, Entry: entry,
		}:
		case <-ctx.Done():
			return fs.SkipAll
		}

		return nil
	})

	close(works)
	wg.Wait()
	if len(dirs) > 0 {
		d.indexLoadedDirs(ctx, dirs)
	}
	d.loadingDir("", 0)

	logutil.Info("disk cache info loaded",
		zap.Any("all files", numFiles),
		zap.Any("cache files", numCacheFiles),
		zap.Any("temp files", numTempFiles),
		zap.Any("deleted files", numDeleted),
		zap.Any("time", time.Since(t0)),
	)
	if ctx.Err() != nil {
		return
	}

	done := make(chan int64, 1)
	d.cache.Evict(ctx, done, 0)
	target := <-done
	logutil.Info("disk cache evict done",
		zap.Any("target", target),
	)

}

// isOldWholeFile reports whether the file at diskPath is a whole-object cache file of an
// earlier version: "full" and the object path, in a shard directory or the cache's.
func (d *DiskCache) isOldWholeFile(diskPath string) bool {
	rel, err := filepath.Rel(d.path, diskPath)
	if err != nil {
		return false
	}
	if shard, rest, ok := strings.Cut(rel, string(filepath.Separator)); ok && isDiskCacheShard(shard) {
		rel = rest
	}
	return strings.HasPrefix(rel, "full") && strings.HasSuffix(rel, cacheFileSuffix)
}

var _ IOVectorCache = new(DiskCache)

func (d *DiskCache) Read(
	ctx context.Context,
	vector *IOVector,
) (
	err error,
) {

	if vector.Policy.Any(SkipDiskCacheReads) {
		return nil
	}

	var numHit, numRead, numOpenIOEntry, numError int64
	defer func() {
		LogEvent(ctx, str_update_metrics_begin)

		metric.FSReadHitDiskCounter.Add(float64(numHit))
		metric.FSReadReadDiskCounter.Add(float64(numRead))
		if numError > 0 {
			metric.FSDiskCacheErrorCounter.Add(float64(numError))
		}
		perfcounter.Update(ctx, func(c *perfcounter.CounterSet) {
			c.FileService.Cache.Read.Add(numRead)
			c.FileService.Cache.Hit.Add(numHit)
			c.FileService.Cache.Disk.Read.Add(numRead)
			c.FileService.Cache.Disk.Hit.Add(numHit)
		}, d.perfCounterSets...)

		LogEvent(ctx, str_update_metrics_end)
	}()

	path, err := ParsePath(vector.FilePath)
	if err != nil {
		return err
	}

	// The files opened are of one version of the object while no delete of it is counted
	// since deletes.
	deletes := d.deletes.now()
	openedFiles := make(map[string]openedDiskCacheFile)
	defer func() {
		LogEvent(ctx, str_close_disk_files_begin)
		for _, opened := range openedFiles {
			d.files.close(opened)
		}
		LogEvent(ctx, str_close_disk_files_end)
	}()

	var key string
	objectKey := func() string {
		if key == "" {
			key = d.objectKey(path.File)
		}
		return key
	}
	openPath := func(diskPath string) (opened openedDiskCacheFile, newlyOpened bool, ok bool) {
		if opened, ok = openedFiles[diskPath]; ok {
			return opened, false, true
		}
		if opened, ok = d.files.open(ctx, diskPath); !ok {
			if d.cache.Contains(diskPath) {
				d.cachedFileExists(ctx, diskPath)
			}
			return openedDiskCacheFile{}, false, false
		}
		openedFiles[diskPath] = opened
		return opened, true, true
	}
	closePath := func(diskPath string) {
		if opened, ok := openedFiles[diskPath]; ok {
			d.files.close(opened)
			delete(openedFiles, diskPath)
		}
	}

	fillEntry := func(entry *IOEntry) error {
		LogEvent(ctx, str_disk_cache_fill_entry_begin)
		defer LogEvent(ctx, str_disk_cache_fill_entry_end)

		if entry.done {
			return nil
		}
		if entry.Size == 0 {
			return &failingReadError{err: moerr.NewEmptyRangeNoCtx(path.File)}
		}
		if entry.Size < 0 {
			// ignore size unknown entry
			return nil
		}

		numRead++

		var opened openedDiskCacheFile
		var openedOK bool

		// entry file
		diskPath := d.pathForIOEntry(path.File, *entry)
		if f, newlyOpened, ok := openPath(diskPath); ok {
			if newlyOpened {
				numOpenIOEntry++
			}
			opened = f
			openedOK = true
		}

		var downloading bool
		// The reads of a download join no other, nor does a large streamed entry.
		if !openedOK && !isDownload(ctx) && entry.materializes(d.memoryPrefetchMax) {
			var filled bool
			downloading, filled, err = d.readFromDownload(ctx, objectKey(), vector.indexTier, entry, vector)
			if err != nil {
				return err
			}
			if filled {
				entry.done = true
				entry.fromCache = d
				numHit++
				return nil
			}
		}

		// try a cached range that contains the entry; rangeOffset is where the entry
		// starts in it
		var rangeOffset int64
		fromRange := false
		for tried := []string(nil); !openedOK; {
			r, ok := d.ranges.containing(objectKey(), entry.Offset, entry.Size, tried)
			if !ok {
				break
			}
			tried = append(tried, r.diskPath)
			if !d.cache.Contains(r.diskPath) {
				d.ranges.remove(r.diskPath)
				continue
			}
			f, newlyOpened, ok := openPath(r.diskPath)
			if !ok {
				continue
			}
			if newlyOpened {
				numOpenIOEntry++
			}
			diskPath, opened, openedOK, fromRange = r.diskPath, f, true, true
			rangeOffset = entry.Offset - r.offset
		}

		// try cached ranges that hold the entry together
		if !openedOK {
			if rs, ok := d.ranges.covering(objectKey(), entry.Offset, entry.Size); ok && len(rs) > 1 {
				readers := make([]io.Reader, 0, len(rs))
				pos, end := entry.Offset, entry.Offset+entry.Size
				for _, r := range rs {
					if !d.cache.Contains(r.diskPath) {
						d.ranges.remove(r.diskPath)
						readers = nil
						break
					}
					f, newlyOpened, ok := openPath(r.diskPath)
					if !ok {
						readers = nil
						break
					}
					if newlyOpened {
						numOpenIOEntry++
					}
					next := min(r.offset+r.size, end)
					readers = append(readers, io.NewSectionReader(f.file, pos-r.offset, next-pos))
					pos = next
				}
				if readers != nil && d.deletes.since(objectKey(), deletes) {
					readers = nil
				}
				if readers != nil {
					if local, err := d.readEntryFrom(ctx, entry, io.MultiReader(readers...)); err != nil {
						if local != nil {
							for _, r := range rs {
								d.dropCachedRange(objectKey(), r.diskPath, openedFiles[r.diskPath].file)
								closePath(r.diskPath)
							}
						}
						return err
					}
					for _, r := range rs {
						d.cache.Get(ctx, r.diskPath)
					}
					entry.done = true
					entry.fromCache = d
					numHit++
					return nil
				}
			}
		}

		if !openedOK && downloading {
			// The fill the read joined stopped without these bytes.
			vector.fillStopped = true
			return nil
		}

		if !openedOK {
			// no file available
			return nil
		}
		file := opened.file

		LogEvent(ctx, str_disk_cache_update_states_begin)
		if _, ok := d.cache.Get(ctx, diskPath); !ok {
			readable, err := d.reindexRead(ctx, diskPath, objectKey(), file)
			if err != nil {
				return err
			}
			if !readable {
				return nil
			}
		}
		LogEvent(ctx, str_disk_cache_update_states_end)

		readOffset, readSize := rangeOffset, entry.Size
		// Only a file from the object's start can be its whole object.
		if readOffset == entry.Offset {
			d.preloads.read(diskPath, entry.Size)
		}
		// Reads of a file kept open share it, so each reads at its own offset.
		if local, err := d.readEntryFrom(ctx, entry, io.NewSectionReader(file, readOffset, readSize)); err != nil {
			if local != nil && fromRange {
				d.dropCachedRange(objectKey(), diskPath, file)
				closePath(diskPath)
			}
			return err
		}
		fadviseDontNeed(file, readOffset, readSize)

		entry.done = true
		entry.fromCache = d
		numHit++

		return nil
	}

	for i := range vector.Entries {
		if err := fillEntry(&vector.Entries[i]); err != nil {
			// A failed read of object storage by a download the read joined, and an error
			// of the entry, fail the read; an error of the cache itself is ignored.
			var failing *failingReadError
			if errors.As(err, &failing) {
				return failing.err
			}
			numError++
			logutil.Warn(
				"read disk cache error",
				zap.Any("error", err),
				zap.Any("path", vector.FilePath),
				zap.Any("entry", vector.Entries[i]),
			)
			// A process allocation failure is not a cache miss; falling back
			// can immediately fail another allocation in the storage path.
			if errors.Is(err, syscall.ENOMEM) {
				return err
			}
		}
	}

	return nil
}

// Update has the publisher publish the entries of vector the cache does not hold; unless
// async is set, it waits for their publications.
func (d *DiskCache) Update(
	ctx context.Context,
	vector *IOVector,
	async bool,
) (
	err error,
) {
	if err := ctx.Err(); err != nil {
		return err
	}

	if vector.Policy.Any(SkipDiskCacheWrites) {
		return nil
	}

	path, err := ParsePath(vector.FilePath)
	if err != nil {
		return err
	}
	if d.wholeFill(d.objectKey(path.File), false) != nil {
		// The fill of the whole object caches it.
		return nil
	}

	for _, entry := range vector.Entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(entry.Data) == 0 {
			// no data
			continue
		}
		if entry.Size < 0 {
			// ignore size unknown entry
			continue
		}
		if entry.fromCache == d || entry.inPrefetchRange {
			// no need to update
			continue
		}

		diskPath := d.pathForIOEntry(path.File, entry)
		if err := d.writeEntry(ctx, d.objectKey(path.File), diskPath, entry, async); err != nil && !async {
			return err
		}
	}

	return nil
}

// writeEntry writes entry, of the object with key, to a file and has the publisher publish
// it, unless the object was deleted after the entry was read. Unless async is set, it
// waits for the publication.
func (d *DiskCache) writeEntry(ctx context.Context, key, diskPath string, entry IOEntry, async bool) error {
	mark := readMark{object: key, deletes: entry.readDeletes - 1, set: entry.readDeletes > 0}
	_, err := d.writeFileTo(ctx, diskPath, func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(entry.Data)), nil
	}, async, notWholeFill, nil, diskFillTarget{d: d, diskPath: diskPath}, mark)
	if err != nil {
		return d.observeWriteResult(diskPath, err)
	}
	return nil
}

func (d *DiskCache) removeEmptyDirectory(path string) {
	runPathHook(d.hooks.beforeLoadDirectoryCleanup, path)
	d.directoryMu.Lock()
	defer d.directoryMu.Unlock()
	// os.Remove will not delete a non-empty directory.
	_ = os.Remove(path)
}

type diskCacheFilesystemStat struct {
	totalBytes     uint64
	availableBytes uint64
	totalInodes    uint64
	freeInodes     uint64
}

// evictForDiskFull frees cache space after a cache write failed because the
// disk is full.
func (d *DiskCache) evictForDiskFull(ctx context.Context, err error) {
	var sourceErr *diskCacheSourceError
	if errors.As(err, &sourceErr) {
		// Reading the source failed; the cache filesystem is not involved.
		return
	}
	kind := classifyDiskFull(err)
	if kind == notDiskFull {
		return
	}
	n := d.capacityFunc() / 10
	if n <= 0 {
		return
	}
	// Concurrent failures observe the same shortage. Each additional eviction
	// would cut another n bytes before the first one has freed anything, so
	// only one runs at a time, and the shortage is checked inside it so that a
	// failure observed before a completed eviction does not evict again.
	if !d.diskFullEviction.CompareAndSwap(false, true) {
		return
	}
	defer d.diskFullEviction.Store(false)
	if kind == diskNoSpace && !d.filesystemExhausted(n) {
		// ENOSPC while blocks and inodes are available comes from a structural
		// limit such as a full directory index. Evicting cannot fix that and
		// would only discard cached data on every failed write.
		return
	}
	d.cache.ForceEvictWithWait(ctx, n)
}

func (d *DiskCache) statFS() (diskCacheFilesystemStat, bool) {
	if d.statFilesystem != nil {
		return d.statFilesystem(d.path)
	}
	return statDiskCacheFilesystem(d.path)
}

func (d *DiskCache) filesystemExhausted(wantBytes int64) bool {
	stat, ok := d.statFS()
	if !ok {
		// Unknown: keep evicting, as before ENOSPC was distinguished.
		return true
	}
	if stat.availableBytes < uint64(wantBytes) {
		return true
	}
	// Filesystems with dynamic inode allocation report zero total inodes.
	// Cross-multiply instead of dividing totalInodes by 100: on a filesystem
	// with under 100 inodes total, that division truncates to zero and
	// freeInodes < 0 can never be true, so zero free inodes would never be
	// recognized as exhaustion.
	return stat.totalInodes > 0 && stat.freeInodes*100 < stat.totalInodes
}

func (d *DiskCache) createCacheTempFile(dir string) (*os.File, error) {
	d.directoryMu.RLock()
	defer d.directoryMu.RUnlock()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	runHook(d.hooks.beforeCacheTempFileCreate)
	return os.CreateTemp(dir, "*"+cacheFileTempSuffix)
}

// readMark is the object of a file and, when set, the delete count its bytes were read
// at: the file is not published once the object was deleted since.
type readMark struct {
	object  string
	deletes uint64
	set     bool
}

// writeFileTo writes the file of the object at diskPath and has the publisher of target
// publish it, unless mark is set and its object was deleted since. Unless async is set,
// it waits for the publication; written reports that the file was published, or, under
// async, queued.
func (d *DiskCache) writeFileTo(
	ctx context.Context,
	diskPath string,
	openReader func(context.Context) (io.ReadCloser, error),
	async bool,
	fill wholeFill,
	objectSize objectLayout,
	target fillTarget,
	mark readMark,
) (written bool, err error) {
	callerCtx := ctx
	defer func() {
		target.cache().evictForDiskFull(callerCtx, err)
	}()
	// A whole-object download is a FilePrefetchObject, not registered after a delete of
	// the object since the mark of ctx.
	var whole *filePrefetchObject
	if fill == storageFill {
		_, index := target.(indexFillTarget)
		whole = d.newWholeObject(ctx, diskPath, index)
		defer func() { whole.endWhole(err) }()
	}
	if target.held(ctx) {
		return false, nil
	}
	pub := &target.cache().pub
	if !pub.begin() {
		return false, nil
	}
	w := fileWrite{pub: pub}
	defer func() { d.endFileWrite(diskPath, &w, &err) }()
	// Each part of a whole-object download is its own request.
	var bounds []int64
	lead, taken := 0, false
	if fill == storageFill {
		var ok bool
		if ok, err = whole.admitWhole(); !ok {
			return false, err
		}
		ctx = whole.ctx
		bounds, lead, taken = whole.splitParts(objectSize)
		if t, ok := target.(*wholeFillTarget); ok && whole.end != math.MaxInt64 {
			t.sizeAt(whole.end)
		}
		if taken {
			defer func() {
				if taken {
					d.prefetchObjects.downloads.n.Add(-1)
				}
			}()
		}
	}

	p, _ := ctx.Value(fileDownloadKey{}).(*filePrefetchObject)
	if err := ctx.Err(); err != nil {
		return false, err
	}
	// The size of a whole object is known from here.
	if target.held(ctx) {
		return false, nil
	}

	// write data
	f, err := target.createTemp()
	if err != nil {
		return false, err
	}
	w.file = f
	var download *objectDownload
	var dst io.Writer = f
	if fill != notWholeFill {
		// The fill of ctx, or one of the bytes the write path holds.
		own := p
		if own == nil {
			own = d.newContentFill(diskPath)
		}
		download = newObjectDownload(f.Name(), bounds)
		if t, ok := target.(*wholeFillTarget); ok {
			t.download = download
		}
		d.startDownload(own, download, taken)
		taken = false
		w.finish = func() { d.finishDownload(download, own) }
		own.started(download)
		downloadStarted(ctx)
		dst = downloadProgressWriter{w: f, download: download}
	}

	n, err := d.copyFrom(ctx, openReader, f, dst, download, bounds, lead)
	if err != nil {
		return false, err
	}

	if t, ok := target.(*wholeFillTarget); ok {
		if !t.place(ctx, n) {
			return false, nil
		}
	}
	if download != nil && fill != rangeFill {
		target.noteWritten(n)
	}
	if whole != nil {
		mark = readMark{object: diskPath, deletes: *whole.needs.deletes, set: true}
	} else if t, ok := target.(*wholeFillTarget); ok {
		mark = readMark{object: t.key, deletes: t.mark, set: true}
	}
	object := mark.object
	if t, ok := target.(indexFillTarget); ok {
		object = t.dir
	}

	finish := w.finish
	job := &publication{
		diskPath: diskPath,
		object:   object,
		publish: func(ctx context.Context) (bool, error) {
			return d.publishFile(ctx, pub, diskPath, f, target, mark)
		},
		release: func() error {
			err := cleanupDiskCacheTempFile(f)
			if finish != nil {
				finish()
			}
			return err
		},
	}
	if !async {
		job.done = make(chan publishResult, 1)
	}
	if !pub.submit(ctx, job) {
		return false, ctx.Err()
	}
	w = fileWrite{}
	if whole != nil {
		d.prefetchObjects.mu.Lock()
		whole.handedOff = true
		d.prefetchObjects.mu.Unlock()
	}
	if async {
		return true, nil
	}
	return pub.wait(ctx, job)
}

// copyFrom copies the bytes openReader opens into f, through dst, noting the progress of
// download when set.
func (d *DiskCache) copyFrom(
	ctx context.Context,
	openReader func(context.Context) (io.ReadCloser, error),
	f *os.File,
	dst io.Writer,
	download *objectDownload,
	bounds []int64,
	lead int,
) (n int64, err error) {
	readCtx, idle := newIdleReader(ctx)
	defer idle.Close()
	if download != nil {
		defer download.idle.add(idle)()
	}
	from, err := idle.open(func() (io.ReadCloser, error) { return openReader(readCtx) })
	if err != nil {
		err = &diskCacheSourceError{err: err}
		if download != nil {
			download.endWith(false, sourceFailure(readCtx, err))
		}
		return 0, err
	}
	defer from.Close()
	source := &diskCacheSourceReader{Reader: idle}

	if src, ok := from.(*objectSource); ok && download != nil {
		n, err = copyParts(readCtx, f, download, src, bounds, lead)
	} else {
		if download != nil {
			download.begin(0)
		}
		var buf []byte
		put := ioBufferPool.Get(&buf)
		defer put.Put()
		n, err = io.CopyBuffer(dst, source, buf)
		if err != nil && source.err != nil {
			err = &diskCacheSourceError{err: source.err}
		}
	}
	if download != nil {
		download.endWith(err == nil, sourceFailure(readCtx, err))
	}
	return n, err
}

// publishFile publishes f, the file of diskPath into target, on the publisher pub, unless
// mark is set and its object was deleted since. A delete of the object queued later
// removes it.
func (d *DiskCache) publishFile(ctx context.Context, pub *publisher, diskPath string, f *os.File, target fillTarget, mark readMark) (bool, error) {
	if mark.set && d.deletes.since(mark.object, mark.deletes) {
		return false, nil
	}
	err := d.finalizeFileTo(ctx, diskPath, f, pub.claim, target)
	return err == nil, err
}

// sourceFailure returns err when it is a failed read of object storage by a fill whose
// context readCtx did not end: it fails the reads waiting for the fill with the error of
// object storage itself, as a read of their own would. A fill stopped (idle, delete,
// close, its query) leaves them to read their own bytes.
func sourceFailure(readCtx context.Context, err error) error {
	var source *diskCacheSourceError
	if err == nil || readCtx.Err() != nil || !errors.As(err, &source) || sourceStopped(err) {
		return nil
	}
	return source.err
}

// sourceStopped reports whether err ends a source without failing the reads of its
// bytes: it was canceled, sent no bytes within the idle timeout, or ended short of its
// range, so the reads read their own bytes.
func sourceStopped(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, errSourceIdle) || errors.Is(err, errSourceShort)
}

// fileWrite is what a write of a cache file holds until it hands the file to its
// publisher.
type fileWrite struct {
	pub  *publisher // the publisher the write began a publication of
	file *os.File
	// finish ends the download of the file, once the file is published or dropped.
	finish func()
}

// endFileWrite releases what w holds, joining the errors into err.
func (d *DiskCache) endFileWrite(diskPath string, w *fileWrite, err *error) {
	if w.file != nil {
		d.mergeWriteCleanupError(diskPath, err, cleanupDiskCacheTempFile(w.file))
	}
	if w.finish != nil {
		w.finish()
	}
	if w.pub != nil {
		w.pub.end()
	}
}

// finalizeFileTo makes f durable, makes room for it and publishes it into target once
// claimPublication holds.
func (d *DiskCache) finalizeFileTo(
	ctx context.Context,
	diskPath string,
	f *os.File,
	claimPublication func() bool,
	target fillTarget,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := d.fileSync(f); err != nil {
		return err
	}
	// Sync is an uninterruptible syscall on the supported platforms. Recheck
	// cancellation after it returns so Close cannot publish a cache file after the cache
	// has entered its terminal generation.
	if err := ctx.Err(); err != nil {
		return err
	}
	fadviseDontNeed(f, 0, 0)

	stat, err := f.Stat()
	if err != nil {
		return err
	}
	size := fileSize(stat)
	tempPath := f.Name()

	if err := f.Close(); err != nil {
		return err
	}
	target.reserve(ctx, size)
	runHook(d.hooks.beforeFilePublication)
	if !claimPublication() {
		return context.Canceled
	}
	runHook(d.hooks.afterPublicationClaim)
	if err := target.publish(ctx, tempPath, size); err != nil {
		return err
	}
	runHook(d.hooks.afterFilePublication)

	return nil
}

// Flush waits until every publication begun is published or dropped, or ctx ends.
func (d *DiskCache) Flush(ctx context.Context) {
	d.pub.flush(ctx)
}

const (
	cacheFileSuffix     = ".mofscache"
	cacheFileTempSuffix = cacheFileSuffix + ".tmp"
)

// Cache files are spread over diskCacheShardCount sub-directories so that no
// single directory accumulates every cached object: filesystems cap the
// entries of one directory independently of free space (ext4 without
// large_dir fails with ENOSPC once a directory's htree index is full).
//
// The shard must be stable across processes and versions, otherwise a restart
// or upgrade would no longer find existing cache files. Do not change the hash
// or the shard count without handling the previous layout.
const (
	diskCacheShardHexDigits = 3
	diskCacheShardCount     = 1 << (4 * diskCacheShardHexDigits)
)

func diskCacheShard(path string) string {
	// FNV-1a, 32 bit.
	h := uint32(2166136261)
	for i := 0; i < len(path); i++ {
		h ^= uint32(path[i])
		h *= 16777619
	}
	h %= diskCacheShardCount
	const digits = "0123456789abcdef"
	var shard [diskCacheShardHexDigits]byte
	for i := len(shard) - 1; i >= 0; i-- {
		shard[i] = digits[h&0xf]
		h >>= 4
	}
	return string(shard[:])
}

func isDiskCacheShard(name string) bool {
	if len(name) != diskCacheShardHexDigits {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func (d *DiskCache) pathForIOEntry(path string, entry IOEntry) string {
	if entry.Size < 0 {
		panic("should not cache size -1 entry")
	}
	// Different spellings of the same object path (for example "foo/./bar"
	// and "foo/bar") must land on the same disk path, or a delete issued
	// with one spelling leaves a stale entry reachable under another.
	// filepath.Join below already cleans the "N-N..." component,
	// so the shard has to be derived from the same cleaned form or the two
	// spellings would be sharded apart even though the filename collides.
	path = pathpkg.Clean(path)
	return filepath.Join(
		d.path,
		diskCacheShard(path),
		fmt.Sprintf("%d-%d%s%s", entry.Offset, entry.Size, toOSPath(path), cacheFileSuffix),
	)
}

// removeEvictedFile removes, on the publisher, the file of size bytes at path, evicted,
// unless path was indexed again.
func (d *DiskCache) removeEvictedFile(path string, size int64) error {
	if d.cache.Contains(path) {
		return nil
	}
	if d.dirEntries && strings.HasSuffix(path, cacheDirSuffix) {
		return d.removeEvictedDir(path, size)
	}
	err := os.Remove(path)
	d.files.changed(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	metric.FSDiskCacheEvictCounter.Add(1)
	return nil
}

var _ FileCache = new(DiskCache)

func (d *DiskCache) SetFile(
	ctx context.Context,
	filePath string,
	openReader func(context.Context) (io.ReadCloser, error),
) error {
	return d.setFile(ctx, filePath, openReader, false)
}

func (d *DiskCache) setFile(
	ctx context.Context,
	filePath string,
	openReader func(context.Context) (io.ReadCloser, error),
	async bool,
) error {
	return d.fillFile(ctx, filePath, openReader, async, storageFill, nil)
}

// setFileContent caches content already in memory as the whole object of filePath.
func (d *DiskCache) setFileContent(
	ctx context.Context,
	filePath string,
	content []byte,
	async bool,
) error {
	return d.fillFile(ctx, filePath, func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(content)), nil
	}, async, contentFill, nil)
}

// wholeFill is how a write fills a disk-cache file.
type wholeFill int

const (
	// notWholeFill writes a range or an index file.
	notWholeFill wholeFill = iota
	// contentFill writes a whole object from bytes in memory.
	contentFill
	// storageFill downloads a whole object from object storage.
	storageFill
	// rangeFill downloads a range of an object from object storage.
	rangeFill
)

// fillFile caches the whole object of filePath read through openReader; objectSize, when
// set, returns the object's size before a download starts, so a large object downloads
// as parts.
func (d *DiskCache) fillFile(
	ctx context.Context,
	filePath string,
	openReader func(context.Context) (io.ReadCloser, error),
	async bool,
	fill wholeFill,
	objectSize objectLayout,
) error {
	return d.fillFileTo(ctx, filePath, openReader, async, fill, objectSize, nil)
}

// fillFileTo fills the file of the object at filePath and has it published into target,
// the disk cache when nil; unless async is set, it waits for the publication.
func (d *DiskCache) fillFileTo(
	ctx context.Context,
	filePath string,
	openReader func(context.Context) (io.ReadCloser, error),
	async bool,
	fill wholeFill,
	objectSize objectLayout,
	target fillTarget,
) error {
	path, err := ParsePath(filePath)
	if err != nil {
		return err
	}
	diskPath := d.objectKey(path.File)
	// Only a fill published into the disk cache takes its room.
	if target == nil {
		target = d.newWholeFillTarget(ctx, path.File)
		d.wholeFills.Add(1)
		defer d.wholeFills.Add(-1)
	}
	// The publisher observes the result of a publication.
	if _, err = d.writeFileTo(ctx, diskPath, openReader, async, fill, objectSize, target, readMark{object: diskPath}); err != nil {
		return d.observeWriteResult(diskPath, err)
	}
	return nil
}

// DeletePaths deletes the entries of every path of paths, whatever the others return,
// and returns their errors joined.
func (d *DiskCache) DeletePaths(
	ctx context.Context,
	paths []string,
) error {
	canonical, err := canonicalFilePaths(paths)
	if err != nil {
		return err
	}
	errs := make([]error, 0, len(canonical))
	for _, path := range canonical {
		errs = append(errs, d.deletePath(ctx, path))
	}
	return errors.Join(errs...)
}

func (d *DiskCache) deletePath(ctx context.Context, path string) error {
	key := d.objectKey(path)
	// Counted first: an entry read before it and cached after it is retracted. Counted
	// again once done: bytes read while it runs are of no version after it.
	d.deletes.deleted(key)
	defer d.deletes.deleted(key)
	// No prefetch object of the object caches its bytes after the delete; one under
	// way is canceled rather than waited for.
	if err := d.dropPrefetches(ctx, key); err != nil {
		return err
	}
	d.prefetched.forget(key)
	return d.pub.remove(ctx, key, func() error { return d.removeRanges(key) })
}

// removeRanges removes the cached ranges of the object with key, each one whatever the
// others return, and returns their errors joined. A range whose file is not removed stays
// indexed.
func (d *DiskCache) removeRanges(key string) error {
	ctx := context.Background()
	var errs []error
	for _, r := range d.ranges.of(key) {
		err := os.Remove(r.diskPath)
		d.files.changed(r.diskPath)
		if err != nil && !os.IsNotExist(err) {
			errs = append(errs, err)
			continue
		}
		if d.cache.Contains(r.diskPath) {
			d.preloads.deleted(r.diskPath)
		}
		d.ranges.remove(r.diskPath)
		d.cache.Delete(ctx, r.diskPath)
	}
	return errors.Join(errs...)
}

func (d *DiskCache) Evict(ctx context.Context, done chan int64) {
	d.cache.Evict(ctx, done, 0)
}

func fileSize(info fs.FileInfo) int64 {
	if sys, ok := info.Sys().(*syscall.Stat_t); ok {
		return int64(sys.Blocks) * 512 // it's always 512, not sys.Blksize
	}
	return info.Size()
}

func (d *DiskCache) Close(ctx context.Context) {
	// A canceled ctx means terminal draining is incomplete. Callers must not
	// reuse the cache directory for a new DiskCache generation until a later
	// Close returns with a live context.
	d.load.cancel()
	d.pub.seal()
	d.closePrefetches(ctx)
	d.pub.close(ctx)
	d.files.closeAll()
	select {
	case <-d.load.done:
	case <-ctx.Done():
	}
	allDiskCaches.Delete(d)
}

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

// objectOfFile returns the object key of the range file at diskPath, or diskPath when it
// is not an indexed range file.
func (c *cachedRanges) objectOfFile(diskPath string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if key, ok := c.objectOf[diskPath]; ok {
		return key
	}
	return diskPath
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

// objectDeletesStripes is the number of locks the deletes of objects share by their disk
// paths.
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
	// stripes are held shared by the stores of prefetched sizes and the indexing of files
	// found on disk, and exclusively by the counting of deletes, by object disk path.
	stripes [objectDeletesStripes]sync.RWMutex
}

var objectDeletesSeed = maphash.MakeSeed()

func (o *objectDeletes) stripe(diskPath string) *sync.RWMutex {
	return &o.stripes[maphash.String(objectDeletesSeed, diskPath)%objectDeletesStripes]
}

func (o *objectDeletes) now() uint64 {
	return o.count.Load()
}

// deleted counts a delete of the object at diskPath.
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

// openFileStripes is the number of counts the changes of paths share by path.
const openFileStripes = 64

// openFilePool keeps, by disk path, the cache files reads opened, so the reads after
// them do not open them again. A change of a path, made by the publisher, drops its file.
type openFilePool struct {
	mu sync.Mutex
	m  map[string]*openCacheFile
	// missing are the paths a read found no file at, until they change.
	missing map[string]struct{}
	takes   uint64
	// limit, when set, is the most files kept instead of DefaultDiskOpenFilesMax; less than
	// 0 keeps none and notes no path missing.
	limit int
	// changes counts the changes of the paths of each stripe: a file opened before a change
	// of its stripe is not kept, nor is its path noted missing.
	changes [openFileStripes]uint64
}

type openedDiskCacheFile struct {
	file *os.File
	// kept is the file kept open for later reads, when the read holds one.
	kept *openCacheFile
}

func (o *openFilePool) stripe(path string) *uint64 {
	return &o.changes[maphash.String(objectDeletesSeed, path)%openFileStripes]
}

// seq returns the count of changes of the stripe of path.
func (o *openFilePool) seq(path string) uint64 {
	o.mu.Lock()
	defer o.mu.Unlock()
	return *o.stripe(path)
}

// take returns the file kept open for path, held for a read, or nil.
func (o *openFilePool) take(path string) *openCacheFile {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.takeLocked(path)
}

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

// keep keeps file open for path, held for the read that opened it, unless the stripe of
// path changed since its count was seq. With the most files kept and all held, it keeps
// nothing. It returns the kept file, or nil.
func (o *openFilePool) keep(path string, seq uint64, file *os.File) *openCacheFile {
	var closeFile *os.File
	defer func() {
		if closeFile != nil {
			_ = closeFile.Close()
		}
	}()
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.limit < 0 || o.m[path] != nil || *o.stripe(path) != seq {
		return nil
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
			return nil
		}
		delete(o.m, lru)
		closeFile = oldest.file
	}
	if o.m == nil {
		o.m = make(map[string]*openCacheFile)
	}
	o.takes++
	kept := &openCacheFile{file: file, refs: 1, used: o.takes}
	o.m[path] = kept
	return kept
}

// release ends a read's hold of f, and closes it when it was dropped and no read holds it.
func (o *openFilePool) release(f *openCacheFile) {
	o.mu.Lock()
	f.refs--
	closeIt := f.dropped && f.refs == 0
	o.mu.Unlock()
	if closeIt {
		_ = f.file.Close()
	}
}

// drop forgets kept, the file kept open for path a read took, which path no longer
// holds, and ends the read's hold of it.
func (o *openFilePool) drop(path string, kept *openCacheFile) {
	o.mu.Lock()
	if o.m[path] == kept {
		o.dropLocked(path)
	}
	o.mu.Unlock()
	o.release(kept)
}

// changed counts a change of path, made by the publisher once the file at path is
// renamed in or removed, and drops its file.
func (o *openFilePool) changed(path string) {
	o.mu.Lock()
	*o.stripe(path)++
	f := o.dropLocked(path)
	o.mu.Unlock()
	if f != nil {
		_ = f.Close()
	}
}

// dropLocked forgets the file of path and returns it to close when no read holds it.
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

// open opens the cache file at path for a read: the file kept open for path, or a new
// one, kept open for later reads. A path noted missing is not opened again until it
// changes.
func (o *openFilePool) open(ctx context.Context, path string) (openedDiskCacheFile, bool) {
	o.mu.Lock()
	kept := o.takeLocked(path)
	_, missing := o.missing[path]
	seq := *o.stripe(path)
	o.mu.Unlock()
	if kept != nil {
		return openedDiskCacheFile{file: kept.file, kept: kept}, true
	}
	if missing {
		return openedDiskCacheFile{}, false
	}
	LogEvent(ctx, str_disk_cache_file_open_begin)
	file, err := os.Open(path)
	LogEvent(ctx, str_disk_cache_file_open_end)
	if err != nil {
		if os.IsNotExist(err) {
			o.mu.Lock()
			if o.limit >= 0 && *o.stripe(path) == seq {
				if o.missing == nil || len(o.missing) >= diskCacheMissingPathsMax {
					o.missing = make(map[string]struct{})
				}
				o.missing[path] = struct{}{}
			}
			o.mu.Unlock()
		}
		return openedDiskCacheFile{}, false
	}
	return openedDiskCacheFile{file: file, kept: o.keep(path, seq, file)}, true
}

// close ends the read of opened.
func (o *openFilePool) close(opened openedDiskCacheFile) {
	if opened.kept == nil {
		_ = opened.file.Close()
		return
	}
	o.release(opened.kept)
}

// closeAll closes the files kept open; no file is kept after it.
func (o *openFilePool) closeAll() {
	o.mu.Lock()
	o.missing = nil
	o.limit = -1
	var files []*os.File
	for p := range o.m {
		if f := o.dropLocked(p); f != nil {
			files = append(files, f)
		}
	}
	o.mu.Unlock()
	for _, f := range files {
		_ = f.Close()
	}
}
