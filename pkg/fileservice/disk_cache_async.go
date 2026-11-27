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
	"io"
	"os"
	pathpkg "path"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/matrixorigin/matrixone/pkg/logutil"
	metric "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
)

const (
	// publisherQueueSize is the most publications queued at once.
	publisherQueueSize = 512
	// publisherWorkers is the number of publications a publisher publishes at once.
	publisherWorkers = 4
	// publisherLongWait is the wait for room in the queue that is logged.
	publisherLongWait            = time.Second
	publisherLongWaitLogKey      = "disk-cache-publisher-wait"
	diskCacheWriteErrorLogKey    = "disk-cache-write-error"
	diskCacheWriteRecoveryLogKey = "disk-cache-write-recovery"
	diskCacheWriteErrorLogPeriod = time.Minute
)

// publication is a file or entry directory written under a temporary name, published by
// the publisher of its cache, or, when removal is set, the removal of the cached files of
// a deleted object.
type publication struct {
	diskPath string
	// object orders it: a removal runs after every publication of its object taken
	// before it, and a publication after every removal of its object taken before it.
	object  string
	removal bool
	// job names a removal queued without waiting.
	job queuedJob
	// publish makes it durable, renames it into place and indexes it, and reports whether
	// it did.
	publish func(ctx context.Context) (bool, error)
	// release frees what it holds once it is published or dropped.
	release func() error
	// done, when set, receives the result once it is released.
	done chan publishResult

	attempted bool
	published bool
	err       error
	// releasing is set once its release starts, which may end the context of its submitter.
	releasing   atomic.Bool
	releaseOnce sync.Once
}

// publishResult is what a publication did: whether it was published, and its error as the
// cache observed it.
type publishResult struct {
	published bool
	err       error
}

// publisher publishes the publications of one cache, from a bounded queue: a submission
// waits for room until its context ends.
type publisher struct {
	d      *DiskCache
	ctx    context.Context
	cancel context.CancelFunc
	jobs   chan *publication
	done   chan struct{}
	// workers is the number of publications published at once.
	workers int
	logger  *logutil.RateLimitedLogger

	startOnce  sync.Once
	closeOnce  sync.Once
	submitters sync.WaitGroup
	// taking, holding one token, orders the taking of publications from jobs and queued
	// with their entry in taken.
	taking chan struct{}
	// wake tells a worker waiting for jobs that one is queued.
	wake chan struct{}

	mu struct {
		sync.Mutex
		closed bool
		// pending counts the publications begun and not yet released.
		pending int
		idle    chan struct{}
		// taken holds, by object, the publications taken from jobs and not yet done, in
		// the order they were taken; finished is closed and replaced when one is done.
		taken    map[string][]*publication
		finished chan struct{}
		// queued holds the removals queued without waiting, taken before jobs; waiting
		// holds them by job, one of each.
		queued  []*publication
		waiting map[queuedJob]*publication
	}
}

// queuedJob names a removal queued without waiting: its kind ("evict", "reindex",
// "repair" or "unload") and the path it changes.
type queuedJob struct{ kind, path string }

type diskCacheWriteFailureState struct {
	failed          atomic.Bool
	logger          *logutil.RateLimitedLogger
	recoveryLimiter *logutil.EventRateLimiter
}

type diskCacheSourceError struct {
	err error
}

func (e *diskCacheSourceError) Error() string { return e.err.Error() }
func (e *diskCacheSourceError) Unwrap() error { return e.err }

type diskCacheSourceReader struct {
	io.Reader
	err error
}

func (r *diskCacheSourceReader) Read(buf []byte) (int, error) {
	n, err := r.Reader.Read(buf)
	if err != nil && !errors.Is(err, io.EOF) {
		r.err = err
	}
	return n, err
}

func (p *publisher) init(d *DiskCache) {
	p.d = d
	p.ctx, p.cancel = context.WithCancel(context.Background())
	p.jobs = make(chan *publication, publisherQueueSize)
	p.done = make(chan struct{})
	p.workers = publisherWorkers
	p.logger = logutil.NewRateLimitedLoggerWithConfig(
		logutil.GetGlobalLogger(),
		logutil.RateLimitedLoggerConfig{MaxKeys: 1},
	)
	p.mu.idle = make(chan struct{})
	close(p.mu.idle)
	p.mu.taken = make(map[string][]*publication)
	p.mu.waiting = make(map[queuedJob]*publication)
	p.mu.finished = make(chan struct{})
	p.taking = make(chan struct{}, 1)
	p.taking <- struct{}{}
	p.wake = make(chan struct{}, 1)
}

func (d *DiskCache) initWriteFailureState() {
	d.writeFailures.logger = logutil.NewRateLimitedLoggerWithConfig(
		logutil.GetGlobalLogger(),
		logutil.RateLimitedLoggerConfig{MaxKeys: 1},
	)
	d.writeFailures.recoveryLimiter = logutil.NewEventRateLimiter(
		logutil.RateLimitedLoggerConfig{MaxKeys: 1},
	)
}

func (d *DiskCache) observeWriteResult(diskPath string, err error) error {
	if err == nil {
		if d.writeFailures.failed.CompareAndSwap(true, false) {
			decision, ok := d.writeFailures.recoveryLimiter.Allow(
				diskCacheWriteRecoveryLogKey,
				logutil.RateLimitConfig{
					Interval:   diskCacheWriteErrorLogPeriod,
					BurstCount: 1,
				},
			)
			if ok {
				fields := logutil.EventFieldsWithDecision([]zap.Field{
					zap.String("path", diskPath),
					zap.String("cache-path", d.path),
				}, decision)
				d.writeFailures.logger.Logger().Info(
					"disk cache write recovered",
					fields...,
				)
			}
		}
		return nil
	}
	var sourceErr *diskCacheSourceError
	if errors.As(err, &sourceErr) {
		return sourceErr.err
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	metric.FSDiskCacheErrorCounter.Inc()
	d.writeFailures.failed.Store(true)
	d.writeFailures.logger.WarnWithConfig(
		diskCacheWriteErrorLogKey,
		"write disk cache error",
		logutil.RateLimitConfig{
			Interval:   diskCacheWriteErrorLogPeriod,
			BurstCount: 1,
		},
		zap.Error(err),
		zap.String("path", diskPath),
		zap.String("cache-path", d.path),
	)
	// Disk cache is an optional acceleration layer. Filesystem failures are
	// observable above but remain fail-open for the storage read/write result.
	return nil
}

func (d *DiskCache) mergeWriteCleanupError(diskPath string, primary *error, cleanupErr error) {
	if cleanupErr == nil {
		return
	}
	var sourceErr *diskCacheSourceError
	if *primary != nil && (errors.As(*primary, &sourceErr) ||
		errors.Is(*primary, context.Canceled) ||
		errors.Is(*primary, context.DeadlineExceeded)) {
		// Preserve the source/control-path contract while still surfacing an
		// independent local cleanup failure.
		_ = d.observeWriteResult(diskPath, cleanupErr)
		return
	}
	*primary = errors.Join(*primary, cleanupErr)
}

func cleanupDiskCacheTempFile(file *os.File) error {
	closeErr := file.Close()
	if errors.Is(closeErr, os.ErrClosed) {
		closeErr = nil
	}
	removeErr := os.Remove(file.Name())
	if os.IsNotExist(removeErr) {
		removeErr = nil
	}
	return errors.Join(closeErr, removeErr)
}

// begin counts a publication to come, which ends with its release or with end. It
// reports false when the publisher is closed.
func (p *publisher) begin() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.mu.closed {
		return false
	}
	if p.mu.pending == 0 {
		p.mu.idle = make(chan struct{})
	}
	p.mu.pending++
	// Close seals admission under the same mutex before waiting for submitters, so it
	// cannot miss a publication begun that has not reached the queue yet.
	p.submitters.Add(1)
	return true
}

// end ends a publication begun that was not submitted.
func (p *publisher) end() {
	p.settle()
	p.submitters.Done()
}

// settle ends the count of a publication begun.
func (p *publisher) settle() {
	p.mu.Lock()
	p.mu.pending--
	if p.mu.pending == 0 {
		close(p.mu.idle)
	}
	p.mu.Unlock()
}

// submit queues pub, begun, waiting for room until ctx ends or the publisher closes. It
// reports false when pub is not queued; its caller then frees what pub holds and calls
// end. A submitter holds nothing the publisher waits for.
func (p *publisher) submit(ctx context.Context, pub *publication) bool {
	p.startOnce.Do(p.start)
	select {
	case p.jobs <- pub:
		metric.FSDiskCachePublisherQueued.Inc()
		metric.FSDiskCachePublisherWaitDuration.Observe(0)
		p.submitters.Done()
		return true
	default:
	}
	start := time.Now()
	metric.FSDiskCachePublisherWaiting.Inc()
	long := time.NewTimer(publisherLongWait)
	defer long.Stop()
	queued, waiting := false, true
	for waiting {
		select {
		case p.jobs <- pub:
			queued, waiting = true, false
			metric.FSDiskCachePublisherQueued.Inc()
		case <-ctx.Done():
			waiting = false
		case <-p.ctx.Done():
			waiting = false
		case <-long.C:
			p.logger.WarnWithConfig(
				publisherLongWaitLogKey,
				"disk cache publisher queue full",
				logutil.RateLimitConfig{Interval: diskCacheWriteErrorLogPeriod, BurstCount: 1},
				zap.Duration("wait", time.Since(start)),
				zap.Int("queued", len(p.jobs)),
				zap.String("path", pub.diskPath),
				zap.String("cache-path", p.d.path),
			)
			long.Reset(publisherLongWait)
		}
	}
	metric.FSDiskCachePublisherWaiting.Dec()
	metric.FSDiskCachePublisherWaitDuration.Observe(time.Since(start).Seconds())
	if queued {
		p.submitters.Done()
	}
	return queued
}

// wait waits for the result of pub, queued with done set, until ctx ends before its
// release starts.
func (p *publisher) wait(ctx context.Context, pub *publication) (bool, error) {
	select {
	case r := <-pub.done:
		return r.published, r.err
	case <-ctx.Done():
		if !pub.releasing.Load() {
			return false, ctx.Err()
		}
		r := <-pub.done
		return r.published, r.err
	}
}

// claim reports whether a publication may still be published: the publisher is not
// closed. Its mutex orders publications and Close.
func (p *publisher) claim() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return !p.mu.closed
}

// start starts the workers; done is closed once all of them exit.
func (p *publisher) start() {
	var workers sync.WaitGroup
	for range p.workers {
		workers.Add(1)
		go func() {
			defer workers.Done()
			p.run()
		}()
	}
	go func() {
		workers.Wait()
		close(p.done)
	}()
}

func (p *publisher) run() {
	for {
		select {
		case <-p.ctx.Done():
			return
		case <-p.taking:
		}
		pub := p.takeQueued()
		if pub == nil {
			select {
			case <-p.ctx.Done():
				p.taking <- struct{}{}
				return
			case <-p.wake:
				p.taking <- struct{}{}
				continue
			case pub = <-p.jobs:
			}
			metric.FSDiskCachePublisherQueued.Dec()
			p.take(pub)
		}
		p.taking <- struct{}{}
		p.runTaken(pub)
	}
}

// take enters pub, just taken from jobs, in the order of its object.
func (p *publisher) take(pub *publication) {
	p.mu.Lock()
	p.mu.taken[pub.object] = append(p.mu.taken[pub.object], pub)
	p.mu.Unlock()
}

// takeQueued takes the first removal queued, entered in the order of its object, or
// returns nil.
func (p *publisher) takeQueued() *publication {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.mu.queued) == 0 {
		return nil
	}
	pub := p.mu.queued[0]
	p.mu.queued = p.mu.queued[1:]
	delete(p.mu.waiting, pub.job)
	p.mu.taken[pub.object] = append(p.mu.taken[pub.object], pub)
	return pub
}

// queue has a worker run remove as job, a removal of object, without waiting for it. A
// job queued and not yet taken is replaced by the later one in its place. Nothing is
// queued once the publisher is closed.
func (p *publisher) queue(object string, job queuedJob, remove func() error) {
	publish := func(context.Context) (bool, error) { return true, remove() }
	p.mu.Lock()
	if p.mu.closed {
		p.mu.Unlock()
		return
	}
	if pub, ok := p.mu.waiting[job]; ok {
		pub.object, pub.publish = object, publish
		p.mu.Unlock()
		return
	}
	if p.mu.pending == 0 {
		p.mu.idle = make(chan struct{})
	}
	p.mu.pending++
	pub := &publication{
		diskPath: object,
		object:   object,
		removal:  true,
		job:      job,
		publish:  publish,
		release:  func() error { return nil },
	}
	p.mu.queued = append(p.mu.queued, pub)
	p.mu.waiting[job] = pub
	p.mu.Unlock()
	p.startOnce.Do(p.start)
	select {
	case p.wake <- struct{}{}:
	default:
	}
}

// runTaken waits for the turn of pub, taken, and publishes it, or runs it when it is a
// removal; once the publisher stops, a publication is released unpublished.
func (p *publisher) runTaken(pub *publication) {
	for {
		p.mu.Lock()
		turn, finished := p.turnLocked(pub), p.mu.finished
		p.mu.Unlock()
		if turn {
			break
		}
		var stopped <-chan struct{}
		if !pub.removal {
			stopped = p.ctx.Done()
		}
		select {
		case <-finished:
			continue
		case <-stopped:
		}
		break
	}
	if pub.removal || p.ctx.Err() == nil {
		pub.attempted = true
		pub.published, pub.err = pub.publish(p.ctx)
		p.d.evictForDiskFull(p.ctx, pub.err)
	}
	p.mu.Lock()
	taken := p.mu.taken[pub.object]
	if i := slices.Index(taken, pub); i >= 0 {
		taken = slices.Delete(taken, i, i+1)
	}
	if len(taken) == 0 {
		delete(p.mu.taken, pub.object)
	} else {
		p.mu.taken[pub.object] = taken
	}
	close(p.mu.finished)
	p.mu.finished = make(chan struct{})
	p.mu.Unlock()
	p.release(pub)
}

// turnLocked reports whether pub, taken, may run: a removal once every publication of
// its object taken before it is done, any other once every removal taken before it is.
func (p *publisher) turnLocked(pub *publication) bool {
	for _, q := range p.mu.taken[pub.object] {
		if q == pub {
			return true
		}
		if pub.removal || q.removal {
			return false
		}
	}
	return true
}

// remove has the publisher run remove, which removes the cached files of the object
// deleted, after every publication of the object queued before it, and returns its error.
// Once the publisher is closed, it runs remove after the workers exit.
func (p *publisher) remove(ctx context.Context, object string, remove func() error) error {
	if p.begin() {
		job := &publication{
			diskPath: object,
			object:   object,
			removal:  true,
			publish:  func(context.Context) (bool, error) { return true, remove() },
			release:  func() error { return nil },
			done:     make(chan publishResult, 1),
		}
		if p.submit(ctx, job) {
			_, err := p.wait(ctx, job)
			return err
		}
		p.end()
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	select {
	case <-p.done:
	case <-ctx.Done():
		return ctx.Err()
	}
	return remove()
}

// release frees what pub holds and reports its result.
func (p *publisher) release(pub *publication) {
	pub.releaseOnce.Do(func() {
		pub.releasing.Store(true)
		cleanupErr := pub.release()
		var resultErr error
		if pub.removal {
			resultErr = pub.err
		} else if pub.attempted {
			resultErr = pub.err
			p.d.mergeWriteCleanupError(pub.diskPath, &resultErr, cleanupErr)
			resultErr = p.d.observeWriteResult(pub.diskPath, resultErr)
		} else if cleanupErr != nil {
			// One dropped before it was published has not shown a recovery: only a
			// failed cleanup is observed.
			_ = p.d.observeWriteResult(pub.diskPath, cleanupErr)
		}
		if pub.done != nil {
			pub.done <- publishResult{published: pub.attempted && pub.published && resultErr == nil, err: resultErr}
		}
		p.settle()
	})
}

// flush waits until every publication begun is released, or ctx ends.
func (p *publisher) flush(ctx context.Context) {
	p.mu.Lock()
	idle := p.mu.idle
	p.mu.Unlock()
	select {
	case <-idle:
	case <-ctx.Done():
	}
}

// seal stops the publisher: nothing is published from then, the publications queued are
// dropped, the removals queued run, and submissions end.
func (p *publisher) seal() {
	p.closeOnce.Do(func() {
		p.mu.Lock()
		p.mu.closed = true
		p.mu.Unlock()
		p.cancel()

		// Waiting for submitters and dropping queued publications runs file cleanup,
		// kept off the Close caller so its context bounds Close.
		go func() {
			p.submitters.Wait()
			for {
				<-p.taking
				if pub := p.takeQueued(); pub != nil {
					p.taking <- struct{}{}
					p.runTaken(pub)
					continue
				}
				select {
				case pub := <-p.jobs:
					metric.FSDiskCachePublisherQueued.Dec()
					if !pub.removal {
						p.taking <- struct{}{}
						p.release(pub)
						continue
					}
					p.take(pub)
					p.taking <- struct{}{}
					p.runTaken(pub)
				default:
					p.taking <- struct{}{}
					p.startOnce.Do(func() {
						close(p.done)
					})
					return
				}
			}
		}()
	})
}

// close seals the publisher and waits until its publications are released and its
// workers exit, or ctx ends.
func (p *publisher) close(ctx context.Context) {
	p.seal()
	p.flush(ctx)
	select {
	case <-p.done:
	case <-ctx.Done():
	}
}

// fillTarget is the cache a fill of the disk cache publishes the file it writes into: the
// disk cache itself, or the index-file cache for the objects it keeps. The download, the
// file written and the reads of it while it is written are the disk cache's either way.
type fillTarget interface {
	// held reports whether the target already holds the file.
	held(ctx context.Context) bool
	// reserve makes room for a file of size bytes about to be written.
	reserve(ctx context.Context, size int64)
	// noteWritten tells the target the size of a whole object written.
	noteWritten(size int64)
	// createTemp creates the file the fill writes, on the file system publish renames it
	// within.
	createTemp() (*os.File, error)
	// publish makes the file written at tempPath, of size bytes, the target's file.
	publish(ctx context.Context, tempPath string, size int64) error
	// cache returns the cache whose files the target holds, which frees room when a write
	// of its files finds the disk full.
	cache() *DiskCache
}

func (t diskFillTarget) cache() *DiskCache { return t.d }

// diskFillTarget publishes into the disk cache, at diskPath.
type diskFillTarget struct {
	d        *DiskCache
	diskPath string
}

// held reports whether the file at diskPath is indexed and on disk. Otherwise the file
// written is renamed over what is there, and replaces its index entry.
func (t diskFillTarget) held(ctx context.Context) bool {
	if _, ok := t.d.cache.Get(ctx, t.diskPath); ok {
		if _, err := os.Stat(t.diskPath); err == nil {
			return true
		}
	}
	return false
}

func (t diskFillTarget) reserve(ctx context.Context, size int64) {
	t.d.cache.Evict(ctx, nil, size)
}

func (t diskFillTarget) noteWritten(size int64) {
	t.d.noteWholeFile(size)
}

func (t diskFillTarget) createTemp() (*os.File, error) {
	return t.d.createCacheTempFile(filepath.Dir(t.diskPath))
}

func (t diskFillTarget) publish(ctx context.Context, tempPath string, size int64) error {
	err := os.Rename(tempPath, t.diskPath)
	t.d.files.changed(t.diskPath)
	if err != nil {
		return err
	}
	t.index(ctx, size)
	return nil
}

// index indexes the file published at diskPath, of size bytes.
func (t diskFillTarget) index(ctx context.Context, size int64) {
	logutil.Debug("disk cache file written",
		zap.Any("path", t.diskPath),
	)
	if !t.d.cache.Replace(ctx, t.diskPath, struct{}{}, size) {
		t.d.cache.Set(ctx, t.diskPath, struct{}{}, size)
	}
}

// wholeFillTarget publishes a whole object into the disk cache as the range [0, size) of
// filePath, at the size its fill finds.
type wholeFillTarget struct {
	d        *DiskCache
	filePath string
	key      string
	// mark is the delete count the size of the object is stored at.
	mark uint64
	// size is the object's size, once known; path is the range file of [0, size), once
	// the fill holds it.
	size     int64
	sized    bool
	path     string
	download *objectDownload
}

// wholePublication is a whole-object fill publishing its range file.
type wholePublication struct {
	key      string
	download *objectDownload
}

func (d *DiskCache) newWholeFillTarget(ctx context.Context, filePath string) *wholeFillTarget {
	mark, ok := ctx.Value(deletesMarkKey{}).(uint64)
	if !ok {
		mark = d.deletes.now()
	}
	t := &wholeFillTarget{d: d, filePath: filePath, key: d.objectKey(filePath), mark: mark}
	t.size, t.sized = d.prefetched.size(t.key)
	return t
}

// sizeAt sets the object's size.
func (t *wholeFillTarget) sizeAt(size int64) {
	t.size, t.sized = size, true
}

func (t *wholeFillTarget) cached(ctx context.Context) bool {
	if !t.sized {
		return false
	}
	path := t.d.pathForIOEntry(t.filePath, IOEntry{Size: t.size})
	if !(diskFillTarget{d: t.d, diskPath: path}).held(ctx) {
		return false
	}
	t.d.storePrefetchedSize(t.key, t.size, t.mark)
	return true
}

func (t *wholeFillTarget) held(ctx context.Context) bool { return t.cached(ctx) }

func (t *wholeFillTarget) reserve(ctx context.Context, size int64) {
	t.d.cache.Evict(ctx, nil, size)
}

func (t *wholeFillTarget) noteWritten(size int64) {
	t.d.noteWholeFile(size)
}

func (t *wholeFillTarget) createTemp() (*os.File, error) {
	return t.d.createCacheTempFile(filepath.Join(t.d.path, diskCacheShard(pathpkg.Clean(t.filePath))))
}

func (t *wholeFillTarget) cache() *DiskCache { return t.d }

// place makes the file of the fill the range [0, n). It reports false when the range file
// is cached.
func (t *wholeFillTarget) place(ctx context.Context, n int64) bool {
	t.sizeAt(n)
	t.path = t.d.pathForIOEntry(t.filePath, IOEntry{Size: n})
	t.d.storePrefetchedSize(t.key, n, t.mark)
	return !(diskFillTarget{d: t.d, diskPath: t.path}).held(ctx)
}

func (t *wholeFillTarget) publish(ctx context.Context, tempPath string, size int64) error {
	if t.download != nil {
		t.d.wholePublications.Store(t.path, wholePublication{key: t.key, download: t.download})
		defer t.d.wholePublications.Delete(t.path)
	}
	err := t.rename(tempPath)
	t.d.files.changed(t.path)
	if err != nil {
		return err
	}
	diskFillTarget{d: t.d, diskPath: t.path}.index(ctx, size)
	return nil
}

// rename moves the file at tempPath to the range file, in its directory, which stays
// while the file is moved into it.
func (t *wholeFillTarget) rename(tempPath string) error {
	t.d.directoryMu.RLock()
	defer t.d.directoryMu.RUnlock()
	if err := os.MkdirAll(filepath.Dir(t.path), 0755); err != nil {
		return err
	}
	return os.Rename(tempPath, t.path)
}
