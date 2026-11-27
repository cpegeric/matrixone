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
	"io/fs"
	iofs "io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/defines"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/util/toml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestIndexFileCache(t *testing.T, dir string, capacity int64) *IndexFileCache {
	t.Helper()
	c, err := NewIndexFileCache(context.Background(), dir, fscache.ConstCapacity(capacity), "", IndexFileRebuildPolicyDefault)
	require.NoError(t, err)
	<-c.disk.load.done
	t.Cleanup(func() { c.Close(context.Background()) })
	return c
}

func indexCacheBytes(name string, content []byte) IndexCacheFile {
	return IndexCacheFile{
		Name: name,
		Open: func(context.Context) (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(content)), nil
		},
	}
}

func indexCacheEntry(data []byte) []IndexCacheFile {
	return []IndexCacheFile{
		indexCacheBytes("data", data),
		indexCacheBytes("manifest", []byte(fmt.Sprintf("len=%d", len(data)))),
	}
}

func readAllFile(t *testing.T, f *os.File) []byte {
	t.Helper()
	_, err := f.Seek(0, io.SeekStart)
	require.NoError(t, err)
	b, err := io.ReadAll(f)
	require.NoError(t, err)
	return b
}

func closeFiles(files []*os.File) {
	for _, f := range files {
		_ = f.Close()
	}
}

func requireIndexCacheMiss(t *testing.T, c *IndexFileCache, name string) {
	t.Helper()
	files, err := c.Open(context.Background(), name, "data", "manifest")
	require.Nil(t, files)
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "want miss, got %v", err)
}

// listCacheDirs returns entry, temp and trash directories under root.
func listCacheDirs(t *testing.T, root string) (entries, temps, trash []string) {
	t.Helper()
	require.NoError(t, filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return err
		}
		switch name := d.Name(); {
		case strings.HasSuffix(name, cacheDirTrashSuffix):
			trash = append(trash, path)
			return fs.SkipDir
		case strings.HasSuffix(name, cacheDirTempSuffix):
			temps = append(temps, path)
			return fs.SkipDir
		case strings.HasSuffix(name, cacheDirSuffix):
			entries = append(entries, path)
			return fs.SkipDir
		}
		return nil
	}))
	return
}

func evictIndexCache(c *IndexFileCache) {
	done := make(chan int64, 1)
	c.disk.cache.Evict(context.Background(), done, 0)
	<-done
	c.disk.Flush(context.Background())
}

func TestIndexFileCachePutOpen(t *testing.T) {
	ctx := context.Background()
	c := newTestIndexFileCache(t, t.TempDir(), 1<<30)
	data := bytes.Repeat([]byte("x"), 10000)

	written, err := c.Put(ctx, "1/abcd/0/sum", indexCacheEntry(data))
	require.NoError(t, err)
	require.True(t, written)

	files, err := c.Open(ctx, "1/abcd/0/sum", "data", "manifest")
	require.NoError(t, err)
	defer closeFiles(files)
	require.Equal(t, data, readAllFile(t, files[0]))
	require.Equal(t, []byte("len=10000"), readAllFile(t, files[1]))

	written, err = c.Put(ctx, "1/abcd/0/sum", indexCacheEntry([]byte("other")))
	require.NoError(t, err)
	require.False(t, written, "an existing entry is never rewritten")

	requireIndexCacheMiss(t, c, "1/abcd/0/other")
	_, err = c.Open(ctx, "1/abcd/0/sum", "data", "absent")
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound))
}

func TestIndexFileCacheRejectsInvalidNames(t *testing.T) {
	ctx := context.Background()
	c := newTestIndexFileCache(t, t.TempDir(), 1<<30)
	for _, name := range []string{"", ".", "..", "../x", "/abs", "a/b" + cacheDirSuffix + "/c",
		"a" + cacheDirTempSuffix, "a" + cacheDirTrashSuffix, "a" + cacheFileSuffix} {
		_, err := c.Put(ctx, name, indexCacheEntry([]byte("x")))
		require.Error(t, err, name)
		_, err = c.Open(ctx, name, "data")
		require.Error(t, err, name)
		require.Error(t, c.Delete(ctx, name), name)
	}
	for _, files := range [][]IndexCacheFile{
		nil,
		{indexCacheBytes("", nil)},
		{indexCacheBytes("a/b", nil)},
		{indexCacheBytes("..", nil)},
		{indexCacheBytes("data", nil), indexCacheBytes("data", nil)},
	} {
		_, err := c.Put(ctx, "ok", files)
		require.Error(t, err)
	}
	_, err := c.Open(ctx, "ok", "../data")
	require.Error(t, err)
}

func TestIndexFileCacheSourceErrorLeavesNothing(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c := newTestIndexFileCache(t, dir, 1<<30)
	boom := errors.New("boom")

	for _, files := range [][]IndexCacheFile{
		{indexCacheBytes("data", []byte("x")), {Name: "manifest", Open: func(context.Context) (io.ReadCloser, error) {
			return nil, boom
		}}},
		{{Name: "data", Open: func(context.Context) (io.ReadCloser, error) {
			return io.NopCloser(io.MultiReader(bytes.NewReader([]byte("part")), indexCacheErrReader{boom})), nil
		}}},
	} {
		written, err := c.Put(ctx, "e", files)
		require.ErrorIs(t, err, boom)
		require.False(t, written)
		requireIndexCacheMiss(t, c, "e")
	}
	entries, temps, trash := listCacheDirs(t, dir)
	require.Empty(t, entries)
	require.Empty(t, temps)
	require.Empty(t, trash)
}

type indexCacheErrReader struct{ err error }

func (r indexCacheErrReader) Read([]byte) (int, error) { return 0, r.err }

// A Put whose publication waits for room in the full queue of the publisher returns once
// its context ends, publishing nothing and leaving nothing behind; the Put that found room
// is published.
func TestIndexFileCacheCanceledPutLeavesNothing(t *testing.T) {
	dir := t.TempDir()
	c := newTestIndexFileCache(t, dir, 1<<30)
	shrinkPublisher(c.disk, 1)
	syncStarted, unblock := installBlockedDiskCacheFileSync(c.disk)
	t.Cleanup(unblock)
	put := func(ctx context.Context, name string) chan error {
		done := make(chan error, 1)
		go func() {
			_, err := c.Put(ctx, name, indexCacheEntry([]byte("x")))
			done <- err
		}()
		return done
	}
	first := put(context.Background(), "first")
	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the publisher did not take the first entry")
	}
	queued := put(context.Background(), "queued")
	require.Eventually(t, func() bool { return len(c.disk.pub.jobs) == cap(c.disk.pub.jobs) },
		diskCacheLifecycleTestTimeout, time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())
	canceled := put(ctx, "e")
	requireSubmitterWaits(t, c.disk, canceled)
	cancel()
	select {
	case err := <-canceled:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the waiting Put did not end with its context")
	}
	unblock()
	require.NoError(t, <-first)
	require.NoError(t, <-queued)
	requireIndexCacheMiss(t, c, "e")
	files, err := c.Open(context.Background(), "queued", "data")
	require.NoError(t, err)
	require.NoError(t, files[0].Close())
	entries, temps, _ := listCacheDirs(t, dir)
	require.Len(t, entries, 2)
	require.Empty(t, temps)
}

func TestIndexFileCacheEntryInvisibleBeforePublication(t *testing.T) {
	ctx := context.Background()
	c := newTestIndexFileCache(t, t.TempDir(), 1<<30)
	diskPath, err := c.disk.pathForDir("e")
	require.NoError(t, err)
	var sawMiss bool
	c.disk.hooks.beforeFilePublication = func() {
		_, statErr := os.Stat(diskPath)
		sawMiss = os.IsNotExist(statErr)
	}
	written, err := c.Put(ctx, "e", indexCacheEntry([]byte("x")))
	require.NoError(t, err)
	require.True(t, written)
	require.True(t, sawMiss, "the entry must not exist under its name before publication")
}

func TestIndexFileCacheEvictedWhileOpenStaysReadable(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	data := bytes.Repeat([]byte("a"), 64<<10)
	c := newTestIndexFileCache(t, dir, 200<<10)

	_, err := c.Put(ctx, "first", indexCacheEntry(data))
	require.NoError(t, err)
	files, err := c.Open(ctx, "first", "data")
	require.NoError(t, err)
	defer closeFiles(files)

	for i := 0; i < 8; i++ {
		_, err := c.Put(ctx, fmt.Sprintf("other-%d", i), indexCacheEntry(data))
		require.NoError(t, err)
	}
	evictIndexCache(c)

	requireIndexCacheMiss(t, c, "first")
	require.Equal(t, data, readAllFile(t, files[0]), "an evicted entry stays readable through an open handle")

	entries, _, trash := listCacheDirs(t, dir)
	require.Empty(t, trash)
	var total int64
	for _, e := range entries {
		require.True(t, c.disk.cache.Contains(e), e)
		size, err := cacheDirSize(e)
		require.NoError(t, err)
		total += size
	}
	require.LessOrEqual(t, total, int64(200<<10))
}

func TestIndexFileCacheDelete(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c := newTestIndexFileCache(t, dir, 1<<30)
	data := []byte("payload")
	_, err := c.Put(ctx, "e", indexCacheEntry(data))
	require.NoError(t, err)
	files, err := c.Open(ctx, "e", "data")
	require.NoError(t, err)
	defer closeFiles(files)

	require.NoError(t, c.Delete(ctx, "e"))
	requireIndexCacheMiss(t, c, "e")
	require.Equal(t, data, readAllFile(t, files[0]))
	require.NoError(t, c.Delete(ctx, "e"), "deleting a missing entry is not an error")

	written, err := c.Put(ctx, "e", indexCacheEntry([]byte("new")))
	require.NoError(t, err)
	require.True(t, written)
	files2, err := c.Open(ctx, "e", "data")
	require.NoError(t, err)
	defer closeFiles(files2)
	require.Equal(t, []byte("new"), readAllFile(t, files2[0]))

	entries, temps, trash := listCacheDirs(t, dir)
	require.Len(t, entries, 1)
	require.Empty(t, temps)
	require.Empty(t, trash)
}

func requireEntryDir(t *testing.T, c *IndexFileCache, name string, exists bool) {
	t.Helper()
	c.disk.Flush(context.Background())
	path, err := c.disk.pathForDir(name)
	require.NoError(t, err)
	_, err = os.Stat(path)
	if exists {
		require.NoError(t, err, "entry %s is on disk", name)
	} else {
		require.True(t, os.IsNotExist(err), "entry %s is removed, got %v", name, err)
	}
}

// Delete removes a held entry at once, and eviction passes over a held entry.
func TestIndexFileCacheHoldDefersRemoval(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	data := bytes.Repeat([]byte("h"), 64<<10)
	c := newTestIndexFileCache(t, dir, 200<<10)

	_, err := c.Put(ctx, "deleted", indexCacheEntry(data))
	require.NoError(t, err)
	release1, err := c.Hold("deleted")
	require.NoError(t, err)
	release2, err := c.Hold("deleted")
	require.NoError(t, err)
	files, err := c.Open(ctx, "deleted", "data")
	require.NoError(t, err)
	require.NoError(t, c.Delete(ctx, "deleted"))
	requireEntryDir(t, c, "deleted", false)
	requireIndexCacheMiss(t, c, "deleted")
	require.Equal(t, data, readAllFile(t, files[0]), "a file open before Delete stays readable")
	closeFiles(files)
	replacement := bytes.Repeat([]byte("r"), 64<<10)
	written, err := c.Put(ctx, "deleted", indexCacheEntry(replacement))
	require.NoError(t, err)
	require.True(t, written, "a held entry deleted as invalid is replaced")
	release1.Release()
	release1.Release()
	release2.Release()
	files, err = c.Open(ctx, "deleted", "data")
	require.NoError(t, err)
	require.Equal(t, replacement, readAllFile(t, files[0]), "releasing the old holds keeps the replacement")
	closeFiles(files)

	_, err = c.Put(ctx, "held", indexCacheEntry(data))
	require.NoError(t, err)
	release, err := c.Hold("held")
	require.NoError(t, err)
	for i := 0; i < 8; i++ {
		_, err := c.Put(ctx, fmt.Sprintf("other-%d", i), indexCacheEntry(data))
		require.NoError(t, err)
	}
	evictIndexCache(c)
	heldPath, err := c.disk.pathForDir("held")
	require.NoError(t, err)
	requireEntryDir(t, c, "held", true)
	require.True(t, c.disk.cache.Contains(heldPath), "a held entry is not evicted")
	release.Release()
	requireEntryDir(t, c, "held", true)
	require.True(t, c.disk.cache.Contains(heldPath), "a released entry stays")

	_, temps, trash := listCacheDirs(t, dir)
	require.Empty(t, temps)
	require.Empty(t, trash)
}

// A hold taken after eviction chose the entry, but before it is removed, keeps the entry
// on disk and in the cache's accounting.
func TestIndexFileCacheHoldWinsOverChosenEviction(t *testing.T) {
	ctx := context.Background()
	data := bytes.Repeat([]byte("w"), 64<<10)
	entrySize := measuredEntrySize(t, data)
	dir := t.TempDir()
	c := newTestIndexFileCache(t, dir, 2*entrySize+10)
	for _, name := range []string{"a", "b"} {
		written, err := c.Put(ctx, name, indexCacheEntry(data))
		require.NoError(t, err)
		require.True(t, written, name)
	}
	aPath, err := c.disk.pathForDir("a")
	require.NoError(t, err)

	lru := c.disk.cache
	var release *IndexFileHold
	lru.SetEvictable(func(path string) bool {
		if c.disk.dirHeld(path) {
			return false
		}
		if path == aPath && release == nil {
			release, err = c.Hold("a")
			require.NoError(t, err)
		}
		return true
	})
	written, err := c.Put(ctx, "c", indexCacheEntry(data))
	require.NoError(t, err)
	require.True(t, written)
	require.NotNil(t, release, "eviction chose a")

	requireEntryDir(t, c, "a", true)
	require.True(t, c.disk.cache.Contains(aPath), "a held entry stays in the cache")
	require.GreaterOrEqual(t, c.disk.cache.Used(), 2*entrySize, "a held entry stays counted")
	files, err := c.Open(ctx, "a", "data")
	require.NoError(t, err)
	require.Equal(t, data, readAllFile(t, files[0]))
	closeFiles(files)

	lru.SetEvictable(func(path string) bool { return !c.disk.dirHeld(path) })
	release.Release()
	for _, name := range []string{"d", "e", "f"} {
		_, err := c.Put(ctx, name, indexCacheEntry(data))
		require.NoError(t, err)
	}
	requireEntryDir(t, c, "a", false)
	require.False(t, c.disk.cache.Contains(aPath), "a released entry is evicted again")
	require.LessOrEqual(t, c.disk.cache.Used(), 2*entrySize+10)
	_, temps, trash := listCacheDirs(t, dir)
	require.Empty(t, temps)
	require.Empty(t, trash)
}

// Entries are evicted least recently used first: a new entry is never the victim, an
// opened or released entry is used again, and a held one is passed over.
func TestIndexFileCacheEvictsLeastRecentlyUsed(t *testing.T) {
	ctx := context.Background()
	data := bytes.Repeat([]byte("p"), 64<<10)
	entrySize := measuredEntrySize(t, data)
	c := newTestIndexFileCache(t, t.TempDir(), 3*entrySize+10)
	put := func(name string) {
		t.Helper()
		written, err := c.Put(ctx, name, indexCacheEntry(data))
		require.NoError(t, err)
		require.True(t, written, name)
	}
	cached := func(names ...string) {
		t.Helper()
		for _, n := range []string{"a", "b", "c", "d", "e", "f"} {
			path, err := c.disk.pathForDir(n)
			require.NoError(t, err)
			require.Equal(t, slices.Contains(names, n), c.disk.cache.Contains(path), n)
			requireEntryDir(t, c, n, slices.Contains(names, n))
		}
	}

	put("a")
	put("b")
	put("c")
	files, err := c.Open(ctx, "a", "data")
	require.NoError(t, err)
	closeFiles(files)
	put("d")
	cached("a", "c", "d")

	release, err := c.Hold("c")
	require.NoError(t, err)
	put("e")
	cached("c", "d", "e")

	release.Release()
	put("f")
	cached("c", "e", "f")
	require.LessOrEqual(t, c.disk.cache.Used(), 3*entrySize+10)

	_, err = c.Hold("../e")
	require.Error(t, err)
}

// Releasing a hold counts as a use: the entry becomes the most recently used and its last use
// moves to now. Releasing the hold of a superseded entry leaves both where they were.
func TestIndexFileCacheReleaseSuperseded(t *testing.T) {
	ctx := context.Background()
	data := bytes.Repeat([]byte("p"), 64<<10)
	entrySize := measuredEntrySize(t, data)
	for _, superseded := range []bool{false, true} {
		c := newTestIndexFileCache(t, t.TempDir(), 3*entrySize+10)
		put := func(name string) {
			t.Helper()
			written, err := c.Put(ctx, name, indexCacheEntry(data))
			require.NoError(t, err)
			require.True(t, written, name)
		}
		put("a")
		hold, err := c.Hold("a")
		require.NoError(t, err)
		files, err := c.Open(ctx, "a", "data")
		require.NoError(t, err)
		closeFiles(files)
		put("b")
		put("c")
		if superseded {
			hold.ReleaseSuperseded()
		} else {
			hold.Release()
		}
		hold.Release()
		put("d")
		requireEntryDir(t, c, "a", !superseded)
		requireEntryDir(t, c, "b", superseded)
	}

	c, err := newIndexFileCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<30), "", "", IndexFileRebuildPolicyDefault, time.Hour, 0)
	require.NoError(t, err)
	<-c.disk.load.done
	t.Cleanup(func() { c.Close(context.Background()) })
	for _, n := range []string{"released", "superseded"} {
		_, err := c.Put(ctx, n, indexCacheEntry([]byte(n)))
		require.NoError(t, err)
		ageEntry(t, c, n, 2*time.Hour)
	}
	used, err := c.Hold("released")
	require.NoError(t, err)
	used.Release()
	replaced, err := c.Hold("superseded")
	require.NoError(t, err)
	replaced.ReleaseSuperseded()
	n, err := c.collectIdle(ctx, time.Now())
	require.NoError(t, err)
	require.Equal(t, 1, n)
	requireEntryDir(t, c, "superseded", false)
	requireEntryDir(t, c, "released", true)

	var none *IndexFileHold
	none.Release()
	none.ReleaseSuperseded()
}

// After a restart the cache evicts the entries reloaded from disk least recently used
// first, by their last use before the restart.
func TestIndexFileCacheEvictionOrderSurvivesRestart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	data := bytes.Repeat([]byte("p"), 64<<10)
	capacity := 3*measuredEntrySize(t, data) + 10
	c, err := NewIndexFileCache(ctx, dir, fscache.ConstCapacity(capacity), "", IndexFileRebuildPolicyDefault)
	require.NoError(t, err)
	<-c.disk.load.done
	for i, n := range []string{"new", "old", "mid"} {
		_, err := c.Put(ctx, n, indexCacheEntry(data))
		require.NoError(t, err)
		ageEntry(t, c, n, []time.Duration{time.Minute, 3 * time.Hour, time.Hour}[i])
	}
	c.Close(ctx)

	c = newTestIndexFileCache(t, dir, capacity)
	_, err = c.Put(ctx, "next", indexCacheEntry(data))
	require.NoError(t, err)
	requireEntryDir(t, c, "old", false)
	for _, n := range []string{"new", "mid", "next"} {
		requireEntryDir(t, c, n, true)
	}
}

// Entries found when the cache loads rank behind every entry published while it loads,
// an entry removed meanwhile is not counted, and an entry published again becomes the
// most recently used.
func TestIndexFileCacheLoadRanksBehindUse(t *testing.T) {
	ctx := context.Background()
	data := bytes.Repeat([]byte("p"), 64<<10)
	size := measuredEntrySize(t, data)
	put := func(c *IndexFileCache, name string) {
		t.Helper()
		written, err := c.Put(ctx, name, indexCacheEntry(data))
		require.NoError(t, err)
		require.True(t, written, name)
	}

	// entries left on disk by an earlier process
	earlier := newTestIndexFileCache(t, t.TempDir(), 1<<30)
	for i, n := range []string{"old1", "old2", "gone"} {
		put(earlier, n)
		ageEntry(t, earlier, n, time.Duration(3-i)*time.Hour)
	}
	c := newTestIndexFileCache(t, t.TempDir(), 3*size+10)
	for _, n := range []string{"old1", "old2", "gone"} {
		from, err := earlier.disk.pathForDir(n)
		require.NoError(t, err)
		to, err := c.disk.pathForDir(n)
		require.NoError(t, err)
		require.NoError(t, os.MkdirAll(filepath.Dir(to), 0755))
		require.NoError(t, os.Rename(from, to))
	}
	gonePath, err := c.disk.pathForDir("gone")
	require.NoError(t, err)
	c.disk.hooks.beforeIndexLoadedDirs = func() {
		put(c, "fresh")
		require.NoError(t, os.RemoveAll(gonePath))
	}
	c.disk.loadCache(ctx)
	c.disk.hooks.beforeIndexLoadedDirs = nil
	require.False(t, c.disk.cache.Contains(gonePath), "an entry removed while the cache loads is not counted")
	require.Equal(t, 3*size, c.disk.cache.Used())

	put(c, "next")
	requireEntryDir(t, c, "old1", false)
	for _, n := range []string{"fresh", "old2", "next"} {
		requireEntryDir(t, c, n, true)
	}

	old2, err := c.disk.pathForDir("old2")
	require.NoError(t, err)
	require.NoError(t, os.RemoveAll(old2))
	put(c, "old2")
	put(c, "last")
	requireEntryDir(t, c, "fresh", false)
	for _, n := range []string{"old2", "next", "last"} {
		requireEntryDir(t, c, n, true)
	}
}

// measuredEntrySize is the accounted size of an entry of data, in disk blocks.
func measuredEntrySize(t *testing.T, data []byte) int64 {
	t.Helper()
	c := newTestIndexFileCache(t, t.TempDir(), 1<<30)
	_, err := c.Put(context.Background(), "e", indexCacheEntry(data))
	require.NoError(t, err)
	return c.disk.cache.Used()
}

func ageEntry(t *testing.T, c *IndexFileCache, name string, age time.Duration) {
	t.Helper()
	path, err := c.disk.pathForDir(name)
	require.NoError(t, err)
	at := time.Now().Add(-age)
	require.NoError(t, os.Chtimes(path, at, at))
}

// The GC deletes an entry with no holds last used longer than the idle TTL ago; a held
// entry, a recently used one and one whose last hold was just released stay.
// Below the low-water mark of free space, entries with no holds are removed, least recently
// used first, until enough is freed to reach the target, whatever the idle TTL.
func TestIndexFileCacheRelievesDiskPressure(t *testing.T) {
	ctx := context.Background()
	c, err := newIndexFileCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<30), "", "", IndexFileRebuildPolicyDefault, 0, 0)
	require.NoError(t, err)
	<-c.disk.load.done
	t.Cleanup(func() { c.Close(context.Background()) })
	names := []string{"e0", "e1", "e2", "e3", "e4"}
	for i, n := range names {
		_, err := c.Put(ctx, n, indexCacheEntry(bytes.Repeat([]byte{'x'}, 400)))
		require.NoError(t, err)
		ageEntry(t, c, n, time.Duration(len(names)-i)*time.Minute)
	}
	release, err := c.Hold("e0")
	require.NoError(t, err)
	defer release.Release()

	var stat diskCacheFilesystemStat
	ok := true
	c.disk.statFilesystem = func(string) (diskCacheFilesystemStat, bool) { return stat, ok }
	present := func() (out []string) {
		for _, n := range names {
			path, err := c.disk.pathForDir(n)
			require.NoError(t, err)
			if _, err := os.Stat(path); err == nil {
				out = append(out, n)
			}
		}
		return out
	}

	stat = diskCacheFilesystemStat{totalBytes: 10000, availableBytes: 1000}
	short, err := c.relieveDiskPressure(ctx)
	require.NoError(t, err)
	require.False(t, short)
	require.Equal(t, names, present(), "at the low-water mark nothing is removed")

	ok = false
	stat.availableBytes = 0
	short, err = c.relieveDiskPressure(ctx)
	require.NoError(t, err)
	require.False(t, short)
	require.Equal(t, names, present(), "an unknown filesystem is left alone")

	// 1000 bytes short of the 15% target: three 407-byte entries
	ok = true
	stat.availableBytes = 500
	short, err = c.relieveDiskPressure(ctx)
	require.NoError(t, err)
	require.False(t, short)
	require.Equal(t, []string{"e0", "e4"}, present(), "the oldest unheld entries go first; a held one stays")
	requireIndexCacheMiss(t, c, "e1")

	short, err = c.relieveDiskPressure(ctx)
	require.NoError(t, err)
	require.True(t, short, "only a held entry is left")
	require.Equal(t, []string{"e0"}, present())

	// a last use recorded ahead of the clock, as after the clock is set back
	_, err = c.Put(ctx, "future", indexCacheEntry(bytes.Repeat([]byte{'x'}, 400)))
	require.NoError(t, err)
	ageEntry(t, c, "future", -time.Hour)
	_, err = c.relieveDiskPressure(ctx)
	require.NoError(t, err)
	requireEntryDir(t, c, "future", false)
}

func TestIndexFileCacheCollectsIdleEntries(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(1<<30), "", "", IndexFileRebuildPolicyDefault, time.Hour, 0)
	require.NoError(t, err)
	<-c.disk.load.done
	t.Cleanup(func() { c.Close(context.Background()) })
	for _, n := range []string{"idle", "young", "held", "released"} {
		_, err := c.Put(ctx, n, indexCacheEntry([]byte(n)))
		require.NoError(t, err)
	}
	ageEntry(t, c, "idle", 2*time.Hour)
	ageEntry(t, c, "held", 2*time.Hour)
	ageEntry(t, c, "released", 2*time.Hour)
	releaseHeld, err := c.Hold("held")
	require.NoError(t, err)
	releaseUsed, err := c.Hold("released")
	require.NoError(t, err)
	releaseUsed.Release()

	n, err := c.collectIdle(ctx, time.Now())
	require.NoError(t, err)
	require.Equal(t, 1, n)
	requireEntryDir(t, c, "idle", false)
	idlePath, err := c.disk.pathForDir("idle")
	require.NoError(t, err)
	require.False(t, c.disk.cache.Contains(idlePath))
	requireIndexCacheMiss(t, c, "idle")
	for _, name := range []string{"young", "held", "released"} {
		requireEntryDir(t, c, name, true)
	}

	n, err = c.collectIdle(ctx, time.Now().Add(2*time.Hour))
	require.NoError(t, err)
	require.Equal(t, 2, n, "a held entry is never collected")
	requireEntryDir(t, c, "held", true)
	releaseHeld.Release()
	n, err = c.collectIdle(ctx, time.Now().Add(2*time.Hour))
	require.NoError(t, err)
	require.Equal(t, 1, n)

	entries, temps, trash := listCacheDirs(t, dir)
	require.Empty(t, entries)
	require.Empty(t, temps)
	require.Empty(t, trash)
}

// The GC runs on its own timer until Close, also with an idle TTL of 0, when it only
// watches free space.
func TestIndexFileCacheGCLoop(t *testing.T) {
	ctx := context.Background()
	c, err := newIndexFileCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<30), "", "", IndexFileRebuildPolicyDefault, time.Second, 0)
	require.NoError(t, err)
	<-c.disk.load.done
	_, err = c.Put(ctx, "e", indexCacheEntry([]byte("e")))
	require.NoError(t, err)
	ageEntry(t, c, "e", time.Hour)
	require.Eventually(t, func() bool {
		path, _ := c.disk.pathForDir("e")
		_, err := os.Stat(path)
		return os.IsNotExist(err)
	}, 10*time.Second, 50*time.Millisecond)
	c.Close(ctx)

	noTTL := newTestIndexFileCache(t, t.TempDir(), 1<<30)
	require.NotNil(t, noTTL.stopGC, "without an idle TTL the loop still watches free space")
}

func TestIndexFileIdleTTLConfig(t *testing.T) {
	var c CacheConfig
	ttl, err := c.indexFileIdleTTL()
	require.NoError(t, err)
	require.Equal(t, DefaultIndexFileIdleTTL, ttl)
	c.IndexFileIdleTTL = &toml.Duration{Duration: 0}
	ttl, err = c.indexFileIdleTTL()
	require.NoError(t, err)
	require.Zero(t, ttl)
	c.IndexFileIdleTTL = &toml.Duration{Duration: -time.Second}
	_, err = c.indexFileIdleTTL()
	require.Error(t, err)

	config := indexCacheConfig(t, 100<<20, 300<<20)
	config.IndexFileIdleTTL = &toml.Duration{Duration: -time.Second}
	_, err = newFileServiceCaches(context.Background(), config, nil, "test", true)
	require.Error(t, err)
	config.IndexFileIdleTTL = &toml.Duration{Duration: time.Minute}
	caches := newTestFileServiceCaches(t, config)
	require.Equal(t, time.Minute, caches.index.idleTTL)
	require.NotNil(t, caches.index.stopGC)
}

func TestIndexFileCacheReusedAfterRestart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c, err := NewIndexFileCache(ctx, dir, fscache.ConstCapacity(1<<30), "", IndexFileRebuildPolicyDefault)
	require.NoError(t, err)
	<-c.disk.load.done
	data := []byte("persisted")
	_, err = c.Put(ctx, "1/ab/0/sum", indexCacheEntry(data))
	require.NoError(t, err)
	diskPath, err := c.disk.pathForDir("1/ab/0/sum")
	require.NoError(t, err)
	c.Close(ctx)

	parent := filepath.Dir(diskPath)
	oldTemp := filepath.Join(parent, "old"+cacheDirTempSuffix)
	newTemp := filepath.Join(parent, "new"+cacheDirTempSuffix)
	leftTrash := filepath.Join(parent, "x"+cacheDirTrashSuffix)
	for _, d := range []string{oldTemp, newTemp, leftTrash} {
		require.NoError(t, os.MkdirAll(d, 0755))
		require.NoError(t, os.WriteFile(filepath.Join(d, "data"), []byte("junk"), 0644))
	}
	old := time.Now().Add(-9 * time.Hour)
	require.NoError(t, os.Chtimes(oldTemp, old, old))

	c2 := newTestIndexFileCache(t, dir, 1<<30)
	require.True(t, c2.disk.cache.Contains(diskPath), "restart load must index entry directories")
	files, err := c2.Open(ctx, "1/ab/0/sum", "data", "manifest")
	require.NoError(t, err)
	defer closeFiles(files)
	require.Equal(t, data, readAllFile(t, files[0]))

	_, err = os.Stat(oldTemp)
	require.True(t, os.IsNotExist(err), "old temp directory is removed")
	_, err = os.Stat(newTemp)
	require.NoError(t, err, "recent temp directory may belong to a writer and is kept")
	_, err = os.Stat(leftTrash)
	require.True(t, os.IsNotExist(err), "trash is removed")
}

func TestIndexFileCacheTwoInstancesShareDirectory(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c1 := newTestIndexFileCache(t, dir, 1<<30)
	c2 := newTestIndexFileCache(t, dir, 1<<30)
	data := []byte("shared")

	_, err := c1.Put(ctx, "e", indexCacheEntry(data))
	require.NoError(t, err)
	files, err := c2.Open(ctx, "e", "data")
	require.NoError(t, err)
	require.Equal(t, data, readAllFile(t, files[0]))
	closeFiles(files)
	c2.disk.Flush(ctx)
	diskPath, err := c2.disk.pathForDir("e")
	require.NoError(t, err)
	require.True(t, c2.disk.cache.Contains(diskPath), "a hit indexes the entry in the reading instance")

	written, err := c2.Put(ctx, "e", indexCacheEntry([]byte("other")))
	require.NoError(t, err)
	require.False(t, written)

	require.NoError(t, c1.Delete(ctx, "e"))
	requireIndexCacheMiss(t, c2, "e")

	var wg sync.WaitGroup
	results := make([]bool, 2)
	for i, c := range []*IndexFileCache{c1, c2} {
		wg.Add(1)
		go func(i int, c *IndexFileCache) {
			defer wg.Done()
			written, err := c.Put(ctx, "race", indexCacheEntry(data))
			assert.NoError(t, err)
			results[i] = written
		}(i, c)
	}
	wg.Wait()
	require.NotEqual(t, results[0], results[1], "exactly one instance publishes the entry")
	for _, c := range []*IndexFileCache{c1, c2} {
		files, err := c.Open(ctx, "race", "data", "manifest")
		require.NoError(t, err)
		require.Equal(t, data, readAllFile(t, files[0]))
		closeFiles(files)
	}
	_, temps, trash := listCacheDirs(t, dir)
	require.Empty(t, temps)
	require.Empty(t, trash)
}

// CNs of a process configured with the same index-file cache directory share one cache,
// created with the settings of the first and closed with the last; CNs with different
// directories have their own.
func TestIndexFileCacheSharedPerDirectory(t *testing.T) {
	ctx := context.Background()
	config := indexCacheConfig(t, 100<<20, 300<<20)
	c1 := newTestFileServiceCaches(t, config)
	same := indexCacheConfig(t, 100<<20, 50<<20)
	path := c1.index.disk.path + string(filepath.Separator)
	same.IndexFileDiskPath = &path
	c2 := newTestFileServiceCaches(t, same)
	require.Same(t, c1.index, c2.index)
	require.Equal(t, int64(300<<20), c2.index.disk.capacityFunc(), "the first CN's settings are kept")

	other := newTestFileServiceCaches(t, indexCacheConfig(t, 100<<20, 50<<20))
	require.NotSame(t, c1.index, other.index)
	require.Equal(t, int64(50<<20), other.index.disk.capacityFunc())

	shared := c1.index
	_, err := shared.Put(ctx, "e", indexCacheEntry([]byte("e")))
	require.NoError(t, err)
	c1.close(ctx)
	files, err := shared.Open(ctx, "e", "data")
	require.NoError(t, err, "the cache stays open while a CN uses it")
	closeFiles(files)
	c2.close(ctx)
	c3 := newTestFileServiceCaches(t, config)
	require.NotSame(t, shared, c3.index, "the last CN to stop closed the cache")
	files, err = c3.index.Open(ctx, "e", "data")
	require.NoError(t, err, "the entry is reused after the cache is reopened")
	closeFiles(files)
}

func TestIndexFileCacheConcurrentOperations(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c := newTestIndexFileCache(t, dir, 96<<10)
	names := []string{"a", "b", "c", "d", "e", "f"}
	payload := func(name string) []byte { return bytes.Repeat([]byte(name), 16<<10) }

	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			r := rand.New(rand.NewSource(seed))
			for i := 0; i < 200; i++ {
				name := names[r.Intn(len(names))]
				switch r.Intn(6) {
				case 0, 1:
					_, err := c.Put(ctx, name, indexCacheEntry(payload(name)))
					assert.NoError(t, err)
				case 2:
					release, err := c.Hold(name)
					assert.NoError(t, err)
					files, err := c.Open(ctx, name, "data", "manifest")
					if err != nil {
						release.Release()
						assert.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), err)
						continue
					}
					data, derr := io.ReadAll(io.NewSectionReader(files[0], 0, 1<<30))
					manifest, merr := io.ReadAll(io.NewSectionReader(files[1], 0, 1<<30))
					assert.NoError(t, derr)
					assert.NoError(t, merr)
					assert.Equal(t, payload(name), data)
					assert.Equal(t, []byte(fmt.Sprintf("len=%d", len(payload(name)))), manifest)
					closeFiles(files)
					release.Release()
				case 3:
					assert.NoError(t, c.Delete(ctx, name))
				case 4:
					release, err := c.Hold(name)
					assert.NoError(t, err)
					assert.NoError(t, c.Delete(ctx, name))
					release.Release()
				case 5:
					_, err := c.collectIdle(ctx, time.Now().Add(time.Hour))
					assert.NoError(t, err)
				}
			}
		}(int64(w))
	}
	wg.Wait()
	evictIndexCache(c)

	entries, temps, trash := listCacheDirs(t, dir)
	require.Empty(t, temps)
	require.Empty(t, trash)
	for _, e := range entries {
		require.True(t, c.disk.cache.Contains(e), "entry on disk but not in the cache: %s", e)
	}
	for _, name := range names {
		diskPath, err := c.disk.pathForDir(name)
		require.NoError(t, err)
		if c.disk.cache.Contains(diskPath) {
			_, err := os.Stat(diskPath)
			require.NoError(t, err, "entry in the cache but not on disk: %s", diskPath)
		}
	}
}

// An entry replaced while it is being opened yields its own files or a miss, never a mix.
func TestIndexFileCacheOpenReturnsOneGeneration(t *testing.T) {
	ctx := context.Background()
	c := newTestIndexFileCache(t, t.TempDir(), 1<<30)
	oldData := []byte("old")
	_, err := c.Put(ctx, "e", indexCacheEntry(oldData))
	require.NoError(t, err)

	replaced := false
	c.disk.hooks.beforeCacheDirFileOpen = func() {
		if replaced {
			return
		}
		replaced = true
		require.NoError(t, c.Delete(ctx, "e"))
		written, err := c.Put(ctx, "e", indexCacheEntry([]byte("newer content")))
		require.NoError(t, err)
		require.True(t, written)
	}
	files, err := c.Open(ctx, "e", "data", "manifest")
	require.True(t, replaced)
	if err != nil {
		require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), err)
		return
	}
	defer closeFiles(files)
	require.Equal(t, oldData, readAllFile(t, files[0]))
	require.Equal(t, []byte("len=3"), readAllFile(t, files[1]), "manifest must come from the same entry as data")
}

func TestDiskCacheIgnoresDirectoryEntriesWithoutIndexCache(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	entry := filepath.Join(dir, "abc", "x"+cacheDirSuffix)
	require.NoError(t, os.MkdirAll(entry, 0755))
	require.NoError(t, os.WriteFile(filepath.Join(entry, "data"), []byte("x"), 0644))

	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	require.False(t, cache.cache.Contains(entry), "a table-data cache never indexes a directory")
	_, err = os.Stat(filepath.Join(entry, "data"))
	require.True(t, os.IsNotExist(err), "a table-data cache removes unknown files, as before")
}

func indexCacheConfig(t *testing.T, capacity, indexCapacity int64) CacheConfig {
	t.Helper()
	path := filepath.Join(t.TempDir(), "disk-cache")
	c, ic := toml.ByteSize(capacity), toml.ByteSize(indexCapacity)
	return CacheConfig{
		DiskPath:              &path,
		DiskCapacity:          &c,
		IndexFileDiskCapacity: &ic,
		IndexFileCacheEnabled: true,
	}
}

func newTestFileServiceCaches(t *testing.T, config CacheConfig) *fileServiceCaches {
	t.Helper()
	caches, err := newFileServiceCaches(context.Background(), config, nil, "test", true)
	require.NoError(t, err)
	t.Cleanup(func() { caches.close(context.Background()) })
	return &caches
}

func TestIndexFileCacheHasItsOwnCapacity(t *testing.T) {
	config := indexCacheConfig(t, 100<<20, 300<<20)
	caches := newTestFileServiceCaches(t, config)
	require.NotNil(t, caches.disk)
	require.NotNil(t, caches.index)
	require.Equal(t, int64(100<<20), caches.disk.capacityFunc(), "the table-data cache keeps disk-capacity")
	require.Equal(t, int64(300<<20), caches.index.disk.capacityFunc())
	require.Equal(t, *config.DiskPath+"-index-file", caches.index.disk.path)
	require.True(t, caches.index.disk.dirEntries)
	require.False(t, caches.disk.dirEntries)
	require.Equal(t, IndexFileRebuildPolicyDefault, caches.index.RebuildPolicy())
	require.False(t, caches.index.CheckOnStartup())
	caches.close(context.Background())

	path := filepath.Join(t.TempDir(), "index")
	config = indexCacheConfig(t, 100<<20, 50<<20)
	config.IndexFileDiskPath = &path
	config.IndexFileRebuildCache = IndexFileRebuildCacheNone
	config.IndexFileCheckOnStartup = true
	caches = newTestFileServiceCaches(t, config)
	require.Equal(t, int64(50<<20), caches.index.disk.capacityFunc())
	require.Equal(t, path, caches.index.disk.path)
	require.Equal(t, Policy(SkipAllCache|SkipFullFilePreloads), caches.index.RebuildPolicy())
	require.True(t, caches.index.CheckOnStartup())
}

func TestIndexFileCacheCapacityDefaultAndDisable(t *testing.T) {
	config := indexCacheConfig(t, 100<<20, 0)
	config.IndexFileDiskCapacity = nil
	caches := newTestFileServiceCaches(t, config)
	require.NotNil(t, caches.index)
	require.Equal(t, int64(DefaultIndexFileDiskCapacity), caches.index.disk.capacityFunc())
	require.Equal(t, int64(100<<20), caches.disk.capacityFunc())
	caches.close(context.Background())

	for name, mutate := range map[string]func(*CacheConfig){
		"not enabled for this service": func(c *CacheConfig) { c.IndexFileCacheEnabled = false },
		"capacity 0":                   func(c *CacheConfig) { *c.IndexFileDiskCapacity = 0 },
		"capacity disabled":            func(c *CacheConfig) { *c.IndexFileDiskCapacity = DisableCacheCapacity },
	} {
		config := indexCacheConfig(t, 100<<20, 300<<20)
		mutate(&config)
		caches := newTestFileServiceCaches(t, config)
		require.Nil(t, caches.index, name)
		require.Equal(t, int64(100<<20), caches.disk.capacityFunc(), name)
	}

	// a disabled table-data cache leaves the index-file cache on
	config = indexCacheConfig(t, DisableCacheCapacity, 300<<20)
	caches = newTestFileServiceCaches(t, config)
	require.Nil(t, caches.disk)
	require.NotNil(t, caches.index)
	require.Equal(t, int64(300<<20), caches.index.disk.capacityFunc())
}

func TestIndexFileCacheWithoutTableDiskCache(t *testing.T) {
	config := indexCacheConfig(t, 100<<20, 300<<20)
	caches, err := newFileServiceCaches(context.Background(), config, nil, "test", false)
	require.NoError(t, err)
	defer caches.close(context.Background())
	require.Nil(t, caches.disk)
	require.NotNil(t, caches.index)
	require.Equal(t, int64(300<<20), caches.index.disk.capacityFunc())
}

func TestIndexFileCacheRejectsInvalidConfig(t *testing.T) {
	config := indexCacheConfig(t, 100<<20, 300<<20)
	config.IndexFileRebuildCache = "sometimes"
	_, err := newFileServiceCaches(context.Background(), config, nil, "test", true)
	require.Error(t, err)

	for s, want := range map[string]Policy{
		"":                             0,
		IndexFileRebuildCacheReadWrite: 0,
		IndexFileRebuildCacheRead:      SkipMemoryCacheWrites | SkipDiskCacheWrites | SkipFullFilePreloads,
		IndexFileRebuildCacheNone:      SkipAllCache | SkipFullFilePreloads,
	} {
		got, err := ParseIndexFileRebuildCache(s)
		require.NoError(t, err, s)
		require.Equal(t, want, got, s)
	}
}

func TestIndexFileCacheIgnoresSizeHint(t *testing.T) {
	old := GlobalDiskCacheSizeHint.Load()
	t.Cleanup(func() { GlobalDiskCacheSizeHint.Store(old) })

	caches := newTestFileServiceCaches(t, indexCacheConfig(t, 100<<20, 300<<20))
	GlobalDiskCacheSizeHint.Store(1000 << 20)
	require.Equal(t, int64(1000<<20), caches.disk.capacityFunc(), "the table-data cache takes the whole hint")
	require.Equal(t, int64(300<<20), caches.index.disk.capacityFunc())
}

func TestGetIndexFileCacheFromFileServices(t *testing.T) {
	ctx := context.Background()
	config := indexCacheConfig(t, 100<<20, 300<<20)
	shared, err := NewLocalFS(ctx, defines.SharedFileServiceName, t.TempDir(), config, nil)
	require.NoError(t, err)
	local, err := NewLocalFS(ctx, defines.LocalFileServiceName, t.TempDir(), CacheConfig{}, nil)
	require.NoError(t, err)
	fs, err := NewFileServices(defines.SharedFileServiceName, shared, local)
	require.NoError(t, err)
	defer fs.Close(ctx)

	c := GetIndexFileCache(fs)
	require.NotNil(t, c)
	require.Same(t, shared.IndexFileCache(), c)
	require.Nil(t, local.IndexFileCache())
	require.Nil(t, GetIndexFileCache(nil))
	require.Nil(t, GetIndexFileCache(local))

	<-c.disk.load.done
	written, err := c.Put(ctx, "e", indexCacheEntry([]byte("x")))
	require.NoError(t, err)
	require.True(t, written)
}

func TestIndexFileCachePutMovesTempFile(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c := newTestIndexFileCache(t, dir, 1<<30)

	f, remove, err := c.CreateTempFile()
	require.NoError(t, err)
	defer remove()
	require.True(t, strings.HasPrefix(f.Name(), dir))
	data := bytes.Repeat([]byte("m"), 70000)
	_, err = f.Write(data)
	require.NoError(t, err)
	require.NoError(t, f.Close())

	written, err := c.Put(ctx, "moved", []IndexCacheFile{
		{Name: "data", Path: f.Name()},
		indexCacheBytes("manifest", []byte("m")),
	})
	require.NoError(t, err)
	require.True(t, written)
	_, err = os.Stat(f.Name())
	require.True(t, os.IsNotExist(err), "the temp file is moved, not copied")
	remove()
	_, temps, _ := listCacheDirs(t, dir)
	require.Empty(t, temps)

	files, err := c.Open(ctx, "moved", "data")
	require.NoError(t, err)
	defer closeFiles(files)
	require.Equal(t, data, readAllFile(t, files[0]))

	_, err = c.Put(ctx, "missing", []IndexCacheFile{{Name: "data", Path: filepath.Join(dir, "absent")}})
	require.Error(t, err, "a missing source file is the caller's error")
	_, err = c.Put(ctx, "both", []IndexCacheFile{{Name: "data", Path: "x", Open: indexCacheBytes("d", nil).Open}})
	require.Error(t, err)
}

func TestIndexFileCachePathMustNotOverlapDiskCache(t *testing.T) {
	config := indexCacheConfig(t, 100<<20, 300<<20)
	trailing := *config.DiskPath + string(filepath.Separator)
	config.DiskPath = &trailing
	caches := newTestFileServiceCaches(t, config)
	require.Equal(t, filepath.Clean(trailing)+"-index-file", caches.index.disk.path,
		"a trailing slash on disk-path does not put the index cache inside it")

	for name, indexPath := range map[string]func(disk string) string{
		"same":   func(disk string) string { return disk },
		"inside": func(disk string) string { return filepath.Join(disk, "index") },
		"parent": func(disk string) string { return filepath.Dir(disk) },
	} {
		config := indexCacheConfig(t, 100<<20, 300<<20)
		p := indexPath(*config.DiskPath)
		config.IndexFileDiskPath = &p
		_, err := newFileServiceCaches(context.Background(), config, nil, "test", true)
		require.ErrorContains(t, err, "overlaps", name)
	}
}

func TestIndexFileCachePutRepairsIncompleteEntry(t *testing.T) {
	ctx := context.Background()
	c := newTestIndexFileCache(t, t.TempDir(), 1<<30)
	diskPath, err := c.disk.pathForDir("e")
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(diskPath, 0755))
	require.NoError(t, os.WriteFile(filepath.Join(diskPath, "data"), []byte("stale"), 0644))
	requireIndexCacheMiss(t, c, "e")

	written, err := c.Put(ctx, "e", indexCacheEntry([]byte("fresh")))
	require.NoError(t, err)
	require.True(t, written, "an entry missing files is replaced")
	files, err := c.Open(ctx, "e", "data", "manifest")
	require.NoError(t, err)
	defer closeFiles(files)
	require.Equal(t, []byte("fresh"), readAllFile(t, files[0]))
}

func TestIndexFileCacheRemovalLeavesNoDirectories(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	var capacity atomic.Int64
	capacity.Store(1 << 30)
	c, err := NewIndexFileCache(ctx, root, func() int64 { return capacity.Load() }, "", IndexFileRebuildPolicyDefault)
	require.NoError(t, err)
	<-c.disk.load.done
	t.Cleanup(func() { c.Close(context.Background()) })
	names := []string{"7/6964/1/40-1/c1", "7/6964/1/41-1/c2", "7/6964/1/42-1/c3", "7/6964/1/43-1/c4"}
	for _, name := range names {
		written, err := c.Put(ctx, name, indexCacheEntry([]byte("x")))
		require.NoError(t, err)
		require.True(t, written)
	}
	entries, _, _ := listCacheDirs(t, root)
	require.Len(t, entries, len(names))

	capacity.Store(0)
	evictIndexCache(c)
	for _, name := range names {
		requireIndexCacheMiss(t, c, name)
	}
	capacity.Store(1 << 30)
	_, err = c.Put(ctx, names[0], indexCacheEntry([]byte("y")))
	require.NoError(t, err)
	require.NoError(t, c.Delete(ctx, names[0]))

	// only the root and its shard directories remain
	require.NoError(t, filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		require.Equal(t, root, filepath.Dir(path), "left behind: %s", path)
		require.True(t, d.IsDir(), "left behind: %s", path)
		return nil
	}))

	long := strings.Repeat("a/", 200) + "z"
	_, err = c.Put(ctx, long, indexCacheEntry([]byte("z")))
	require.NoError(t, err)
	files, err := c.Open(ctx, long, "data")
	require.NoError(t, err)
	require.Equal(t, []byte("z"), readAllFile(t, files[0]))
	closeFiles(files)
}

type readCountingStorage struct {
	ObjectStorage
	reads atomic.Int64
}

func (c *readCountingStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	c.reads.Add(1)
	return c.ObjectStorage.Read(ctx, key, min, max)
}

func newIndexObjectFS(t *testing.T, indexCache bool) (*S3FS, *readCountingStorage, []byte) {
	ctx := context.Background()
	config := indexCacheConfig(t, 64<<20, 64<<20)
	config.MemoryCapacity = ptrTo[toml.ByteSize](0)
	config.IndexFileCacheEnabled = indexCache
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: defines.SharedFileServiceName, Endpoint: "disk", Bucket: t.TempDir()}, config, nil, false, false)
	require.NoError(t, err)
	<-fs.diskCache.load.done
	if fs.indexFileCache != nil {
		<-fs.indexFileCache.disk.load.done
	}
	// The disk cache caches exactly the entries read, so a read it serves is told apart.
	fs.diskCache.prefetchRange = prefetchRangeConfig{}
	t.Cleanup(func() { fs.Close(ctx) })
	data := make([]byte, 300<<10)
	for i := range data {
		data[i] = byte(i*7 + i/1021)
	}
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: int64(len(data)), Data: data}}, Policy: SkipAllCache}))
	storage := &readCountingStorage{ObjectStorage: fs.server.raw}
	fs.server.raw = storage
	return fs, storage, data
}

// cacheIndexObject puts data as the entry of the object at file.
func cacheIndexObject(t *testing.T, fs *S3FS, file string, data []byte) {
	_, err := fs.indexFileCache.Put(context.Background(), IndexObjectEntry(file), []IndexCacheFile{{
		Name: IndexObjectFile,
		Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil },
	}})
	require.NoError(t, err)
}

func readIndexObjectRange(t *testing.T, fs *S3FS, policy Policy, offset, size int64) []byte {
	t.Helper()
	vec := &IOVector{FilePath: "object", Policy: policy | SkipMemoryCache, Entries: []IOEntry{{Offset: offset, Size: size}}}
	require.NoError(t, fs.Read(context.Background(), vec))
	defer vec.Release()
	flushPublishers(fs)
	return slices.Clone(vec.Entries[0].Data)
}

// IndexFileObject reads of an object the index-file cache holds are served from there.
func TestIndexObjectServesReads(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, true)
	cacheIndexObject(t, fs, "object", data)
	require.Equal(t, data[1000:1010], readIndexObjectRange(t, fs, IndexFileObject, 1000, 10))
	require.Equal(t, data[200<<10:200<<10+100], readIndexObjectRange(t, fs, IndexFileObject|SkipFullFilePreloads, 200<<10, 100))
	require.Equal(t, data[2000:2010], readIndexObjectRange(t, fs, IndexFileObject|SkipDiskCacheReads, 2000, 10), "a skip of disk-cache reads does not apply to the index-file cache")
	require.Zero(t, storage.reads.Load())
	require.Equal(t, data[3000:3010], readIndexObjectRange(t, fs, IndexFileObject|SkipAllCache, 3000, 10))
	require.NotZero(t, storage.reads.Load(), "a read that skips every cache skips the index-file cache")
	storage.reads.Store(0)
	require.Empty(t, fs.diskCache.objectFiles("object"))

	readIndexObjectRange(t, fs, 0, 250<<10, 10)
	require.NotZero(t, storage.reads.Load(), "a read without IndexFileObject does not use the entry")
}

// A decoded IndexFileObject read served from the index-file cache with no memory cache
// gets its cache data from the default allocator.
func TestIndexObjectDecodedReadWithoutMemoryCache(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, true)
	require.Nil(t, fs.memCache)
	cacheIndexObject(t, fs, "object", data)
	vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
		Entries: []IOEntry{{Offset: 1000, Size: 10, ToCacheData: CacheOriginalData}}}
	require.NoError(t, fs.Read(context.Background(), vec))
	defer vec.Release()
	require.NotNil(t, vec.Entries[0].CachedData)
	require.Equal(t, data[1000:1010], vec.Entries[0].CachedData.Bytes())
	require.Zero(t, storage.reads.Load())
}

// Without an index-file cache, IndexFileObject reads take the normal path.
func TestIndexObjectsWithoutIndexFileCache(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, false)
	require.Nil(t, fs.indexFileCache)
	require.Equal(t, data[1000:1010], readIndexObjectRange(t, fs, IndexFileObject, 1000, 10))
	require.NotZero(t, storage.reads.Load())
	cache, _ := IndexObjectCache(fs)
	require.Nil(t, cache)
}

// A read of an object the cache does not hold takes the normal path.
func TestIndexObjectMissReadsStorage(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, true)
	require.Equal(t, data[1000:1010], readIndexObjectRange(t, fs, IndexFileObject, 1000, 10))
	require.NotZero(t, storage.reads.Load())
	cache, shared := IndexObjectCache(fs)
	require.Same(t, fs.indexFileCache, cache)
	require.Same(t, FileService(fs), shared)
}

// The prefetches of an object the index-file cache holds, through a queued fill or the
// IO pipeline, are not downloaded into the disk cache; those of one it does not hold are.
func TestPrefetchOfCachedIndexObjectIsSkipped(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	ctx := WithFileServicePolicy(context.Background(), IndexFileObject)
	busyFills(t, fs)
	require.False(t, FullFilePreloadSkipped(ctx, fs, IndexFileObject, "object"))
	require.False(t, FullFilePreloadSkipped(ctx, fs, IndexFileObject|SkipFullFilePreloads|NarrowRead, "object"),
		"a filtered search still preloads an object the index-file cache keeps")
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: int64(len(data))}))
	require.True(t, fs.diskCache.isQueued("object"))
	fs.diskCache.dropFills("object")

	cacheIndexObject(t, fs, "object", data)
	require.True(t, FullFilePreloadSkipped(ctx, fs, IndexFileObject, "object"))
	require.False(t, FullFilePreloadSkipped(ctx, fs, 0, "object"))
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: int64(len(data))}))
	require.False(t, fs.diskCache.isQueued("object"))
}

// Deleting an object deletes its entry; after it is written again, reads return the new
// bytes.
func TestDeleteRemovesIndexObject(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	ctx := context.Background()
	cacheIndexObject(t, fs, "object", data)
	require.NoError(t, fs.Delete(ctx, "object"))
	_, cached := fs.indexObjectCached("object")
	require.False(t, cached)
	recreated := slices.Clone(data)
	for i := range recreated {
		recreated[i] ^= 0xff
	}
	require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Entries: []IOEntry{{Size: int64(len(recreated)), Data: recreated}}, Policy: SkipAllCache}))
	require.Equal(t, recreated[2000:2010], readIndexObjectRange(t, fs, IndexFileObject, 2000, 10))
}

// A read of an entry marks it used, so idle collection keeps it; one not read is
// collected.
func TestReadKeepsIndexObject(t *testing.T) {
	for _, read := range []bool{false, true} {
		fs, _, data := newIndexObjectFS(t, true)
		cacheIndexObject(t, fs, "object", data)
		diskPath, cached := fs.indexObjectCached("object")
		require.True(t, cached)
		old := time.Now().Add(-2 * time.Hour)
		require.NoError(t, os.Chtimes(diskPath, old, old))
		if read {
			require.Equal(t, data[5000:5010], readIndexObjectRange(t, fs, IndexFileObject, 5000, 10))
		}
		c := fs.indexFileCache
		c.idleTTL = time.Hour
		_, err := c.collectIdle(context.Background(), time.Now())
		require.NoError(t, err)
		_, cached = fs.indexObjectCached("object")
		require.Equal(t, read, cached, "read %v", read)
	}
}

// A whole-object fill of an IndexFileObject read publishes the object into the index-file
// cache, not the disk cache; later reads are served from there without storage reads.
func TestIndexObjectReadFillsIndexFileCache(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, true)
	require.NoError(t, fs.PrefetchFile(WithFileServicePolicy(context.Background(), IndexFileObject), "object"))
	flushPublishers(fs)
	diskPath, cached := fs.indexObjectCached("object")
	require.True(t, cached)
	got, err := os.ReadFile(filepath.Join(diskPath, IndexObjectFile))
	require.NoError(t, err)
	require.Equal(t, data, got)
	require.Empty(t, fs.diskCache.objectFiles("object"), "not in the disk cache")

	reads := storage.reads.Load()
	require.Equal(t, data[7000:7010], readIndexObjectRange(t, fs, IndexFileObject, 7000, 10))
	require.Equal(t, reads, storage.reads.Load())
}

// A fill without IndexFileObject publishes into the disk cache as before.
func TestNonIndexObjectFillUsesDiskCache(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	require.NoError(t, fs.PrefetchFile(context.Background(), "object"))
	flushPublishers(fs)
	_, cached := fs.indexObjectCached("object")
	require.False(t, cached)
	require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))))
}

// A queued whole-object prefetch of an IndexFileObject request publishes into the
// index-file cache, though its fill runs without the request's context.
func TestIndexObjectPrefetchFillsIndexFileCache(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	busyFills(t, fs)
	ctx := WithFileServicePolicy(context.Background(), IndexFileObject)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: int64(len(data))}))
	f := &fs.diskCache.prefetchObjects
	f.mu.Lock()
	req := f.fills[fillKey{diskPath: fs.diskCache.objectKey("object"), index: true}][0].req
	f.mu.Unlock()
	require.True(t, req.indexObject)
	fs.diskCache.dropFills("object")
	fs.diskCache.setFillsMax(0)

	require.False(t, fs.haveObject(req))
	fs.prefetchObject(context.Background(), req)
	_, cached := fs.indexObjectCached("object")
	require.True(t, cached)
	require.Empty(t, fs.diskCache.objectFiles("object"))
	require.True(t, fs.haveObject(req))
}

// The fill of an object deleted after its download started publishes nothing, and its
// file is removed.
func TestIndexObjectFillOfDeletedObjectIsNotPublished(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	target := fs.fillTarget(IndexFileObject, "object").(indexFillTarget)
	require.NoError(t, fs.Delete(context.Background(), "object"))
	tempPath := t.TempDir() + "/object.tmp"
	require.NoError(t, os.WriteFile(tempPath, data, 0o644))
	require.NoError(t, target.publish(context.Background(), tempPath, int64(len(data))))
	_, cached := fs.indexObjectCached("object")
	require.False(t, cached)
	_, err := os.Stat(tempPath)
	require.NoError(t, err, "the file stays its fill's to remove")
}

// A miss read under IndexFileObject that reads the object whole streams it into the
// index-file cache and returns its bytes.
func TestIndexObjectMissReadStreamsIntoIndexFileCache(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, true)
	require.Equal(t, data[1000:1010], readIndexObjectRange(t, fs, IndexFileObject, 1000, 10))
	_, cached := fs.indexObjectCached("object")
	require.True(t, cached)
	require.Empty(t, fs.diskCache.objectFiles("object"))
	reads := storage.reads.Load()
	require.Equal(t, data[9000:9010], readIndexObjectRange(t, fs, IndexFileObject, 9000, 10))
	require.Equal(t, reads, storage.reads.Load())
}

// Publishing copies the file when it cannot be moved into the index-file cache, and the
// file written is removed either way.
func TestIndexObjectPublishRemovesWrittenFile(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	target := fs.fillTarget(IndexFileObject, "object").(indexFillTarget)
	tempPath := filepath.Join(t.TempDir(), "object.tmp")
	require.NoError(t, os.WriteFile(tempPath, data, 0o644))
	require.NoError(t, target.publish(context.Background(), tempPath, int64(len(data))))
	diskPath, cached := fs.indexObjectCached("object")
	require.True(t, cached)
	got, err := os.ReadFile(filepath.Join(diskPath, IndexObjectFile))
	require.NoError(t, err)
	require.Equal(t, data, got)
	_, err = os.Stat(tempPath)
	require.True(t, os.IsNotExist(err))
}

// An asynchronous finalize of a fill publishes into the index-file cache once the disk
// cache's updates are flushed.
func TestIndexObjectAsyncFillPublishes(t *testing.T) {
	fs, _, _ := newIndexObjectFS(t, true)
	require.NoError(t, fs.PrefetchFile(WithFileServicePolicy(context.Background(), IndexFileObject), "object"))
	fs.FlushCache(context.Background())
	_, cached := fs.indexObjectCached("object")
	require.True(t, cached)
	require.Empty(t, fs.diskCache.objectFiles("object"))
}

// An index-object fill writes its file in the index-file cache's directory, so publishing
// it never copies it across file systems, and a fill leaves no temp file in either cache.
func TestIndexObjectFillWritesInIndexFileCache(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	f, err := fs.fillTarget(IndexFileObject, "object").createTemp()
	require.NoError(t, err)
	require.Equal(t, fs.indexFileCache.Path(), filepath.Dir(f.Name()))
	require.NoError(t, cleanupDiskCacheTempFile(f))

	var published []string
	objectPath := fs.diskCache.objectKey("object")
	fs.diskCache.hooks.beforeFilePublication = func() {
		if o := fs.diskCache.download(objectPath); o != nil {
			o.mu.Lock()
			if o.file != nil {
				published = append(published, o.file.Name())
			}
			o.mu.Unlock()
		}
	}
	require.NoError(t, fs.PrefetchFile(WithFileServicePolicy(context.Background(), IndexFileObject), "object"))
	flushPublishers(fs)
	require.NotEmpty(t, published)
	for _, p := range published {
		require.Equal(t, fs.indexFileCache.Path(), filepath.Dir(p))
	}
	diskPath, cached := fs.indexObjectCached("object")
	require.True(t, cached)
	got, err := os.ReadFile(filepath.Join(diskPath, IndexObjectFile))
	require.NoError(t, err)
	require.Equal(t, data, got)
	for _, dir := range []string{fs.indexFileCache.Path(), fs.diskCache.path} {
		require.NoError(t, filepath.WalkDir(dir, func(path string, d iofs.DirEntry, err error) error {
			require.NoError(t, err)
			require.False(t, strings.HasSuffix(path, cacheFileTempSuffix), "temp file left: %s", path)
			return nil
		}))
	}
}

// A read of an index object keeps the entry's file open for the reads after it; a delete
// of the object closes it.
func TestIndexObjectReadKeepsFileOpen(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	d := fs.indexFileCache.disk
	<-d.load.done
	cacheIndexObject(t, fs, "object", data)
	diskPath, err := d.pathForDir(IndexObjectEntry("object"))
	require.NoError(t, err)
	kept := func() *openCacheFile {
		d.files.mu.Lock()
		defer d.files.mu.Unlock()
		return d.files.m[diskPath]
	}

	require.Equal(t, data[1000:1010], readIndexObjectRange(t, fs, IndexFileObject, 1000, 10))
	first := kept()
	require.NotNil(t, first)
	require.Zero(t, first.refs)
	require.Equal(t, data[5000:5010], readIndexObjectRange(t, fs, IndexFileObject, 5000, 10))
	require.Same(t, first, kept(), "the next read reads the kept file")

	require.NoError(t, fs.Delete(context.Background(), "object"))
	require.Nil(t, kept())
	_, err = first.file.Stat()
	require.ErrorIs(t, err, os.ErrClosed)
}

// An IndexFileObject read of an object the index-file cache holds coordinates its shared
// decode as a read of an object the disk cache holds does; a read without the policy, or
// of an object not held, does not.
func TestIndexObjectSharedDecodeFillEligibility(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	vec := func(policy Policy) *IOVector {
		return &IOVector{FilePath: "object", Policy: policy, Entries: []IOEntry{{Offset: 0, Size: 10}}}
	}
	require.False(t, fs.sharedDecodeIndexFillAvailable(vec(IndexFileObject)))
	cacheIndexObject(t, fs, "object", data)
	require.True(t, fs.sharedDecodeIndexFillAvailable(vec(IndexFileObject)))
	require.False(t, fs.sharedDecodeIndexFillAvailable(vec(0)))
}

// HoldCached holds a cached entry until the hold is released, and holds nothing for an
// entry the cache does not hold.
func TestIndexFileCacheHoldCached(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	d := fs.indexFileCache.disk
	name := IndexObjectEntry("object")
	diskPath, err := d.pathForDir(name)
	require.NoError(t, err)
	holds := func() int {
		d.held.mu.Lock()
		defer d.held.mu.Unlock()
		return d.held.m[diskPath]
	}
	require.Nil(t, fs.indexFileCache.HoldCached(name))
	require.Zero(t, holds())

	cacheIndexObject(t, fs, "object", data)
	hold := fs.indexFileCache.HoldCached(name)
	require.NotNil(t, hold)
	require.Equal(t, 1, holds())
	hold.Release()
	require.Zero(t, holds())
}

// A read of an entry whose cached file ends before the read's range drops the entry and
// reads storage, which fills it again; an error of the read's decoding is returned.
func TestIndexObjectShortCacheFileReadsStorage(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, true)
	cacheIndexObject(t, fs, "object", data[:1000])
	require.Equal(t, data[2000:2010], readIndexObjectRange(t, fs, IndexFileObject, 2000, 10))
	require.Equal(t, int64(1), storage.reads.Load())
	require.Equal(t, data[len(data)-10:], readIndexObjectRange(t, fs, IndexFileObject, int64(len(data)-10), 10))
	require.Equal(t, int64(1), storage.reads.Load(), "the short entry is replaced by the object the read filled")

	cacheIndexObject(t, fs, "object", data)
	storage.reads.Store(0)
	decodeErr := errors.New("decode")
	vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache, Entries: []IOEntry{{Offset: 1000, Size: 10,
		ToCacheData: func(context.Context, io.Reader, []byte, CacheDataAllocator) (fscache.Data, error) {
			return nil, decodeErr
		}}}}
	require.ErrorIs(t, fs.Read(context.Background(), vec), decodeErr)
	vec.Release()
	require.Zero(t, storage.reads.Load())
	_, cached := fs.indexObjectCached("object")
	require.True(t, cached, "a decoding error keeps the entry")
}

// A reader that found the entry short drops only the file it read: a healthy entry that
// another read filled meanwhile is kept.
func TestIndexObjectShortReaderKeepsRefill(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, true)
	cacheIndexObject(t, fs, "object", data[:1000])
	started := make(chan struct{})
	resume := make(chan struct{})
	readErr := make(chan error, 1)
	go func() {
		vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache, Entries: []IOEntry{
			{Offset: 100, Size: 10, ToCacheData: func(ctx context.Context, r io.Reader, b []byte, a CacheDataAllocator) (fscache.Data, error) {
				close(started)
				<-resume
				return CacheOriginalData(ctx, r, b, a)
			}},
			{Offset: 2000, Size: 10},
		}}
		readErr <- fs.Read(context.Background(), vec)
		vec.Release()
	}()
	<-started
	require.Equal(t, data[2000:2010], readIndexObjectRange(t, fs, IndexFileObject, 2000, 10))
	var healthy os.FileInfo
	require.Eventually(t, func() bool {
		diskPath, ok := fs.indexObjectCached("object")
		if !ok {
			return false
		}
		fi, err := os.Stat(filepath.Join(diskPath, IndexObjectFile))
		healthy = fi
		return err == nil && fi.Size() == int64(len(data))
	}, 5*time.Second, time.Millisecond)
	close(resume)
	require.NoError(t, <-readErr)
	diskPath, ok := fs.indexObjectCached("object")
	require.True(t, ok)
	fi, err := os.Stat(filepath.Join(diskPath, IndexObjectFile))
	require.NoError(t, err)
	require.True(t, os.SameFile(healthy, fi), "the healthy entry is kept")
	storage.reads.Store(0)
	require.Equal(t, data[len(data)-10:], readIndexObjectRange(t, fs, IndexFileObject, int64(len(data)-10), 10))
	require.Zero(t, storage.reads.Load())
}

// An open failing for lack of resources keeps the entry; other failures drop it.
func TestIndexObjectOpenResourcesExhausted(t *testing.T) {
	for _, err := range []error{syscall.EMFILE, syscall.ENFILE, syscall.ENOMEM, syscall.EAGAIN,
		&os.PathError{Op: "open", Path: "x", Err: syscall.EMFILE}} {
		require.True(t, openResourcesExhausted(err), err)
	}
	for _, err := range []error{syscall.ENOTDIR, syscall.ELOOP, syscall.EACCES, syscall.EIO,
		&os.PathError{Op: "open", Path: "x", Err: syscall.ENOTDIR}} {
		require.False(t, openResourcesExhausted(err), err)
	}
}

// A deleted entry whose files cannot be removed is gone, its files count against the
// capacity, listed or not, and the collector removes them once it can.
func TestIndexFileCacheTrashRemovedOnceItCanBe(t *testing.T) {
	for _, mode := range []os.FileMode{0o500, 0o000, 0o755} {
		fault := mode != 0o755
		t.Run(mode.String(), func(t *testing.T) {
			fs, _, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			diskPath, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			d := fs.indexFileCache.disk
			capacity := d.capacityFunc()
			trashDirs := func() []string {
				matches, err := filepath.Glob(diskPath + ".*" + cacheDirTrashSuffix)
				require.NoError(t, err)
				return matches
			}
			if fault {
				require.NoError(t, os.Chmod(diskPath, mode))
				t.Cleanup(func() {
					for _, m := range trashDirs() {
						_ = os.Chmod(m, 0o755)
					}
				})
				if f, err := os.Create(filepath.Join(diskPath, "probe")); err == nil {
					_ = f.Close()
					t.Skip("permissions are not enforced")
				}
			}
			require.NoError(t, fs.indexFileCache.Delete(context.Background(), IndexObjectEntry("object")))
			_, err := os.Stat(diskPath)
			require.True(t, os.IsNotExist(err))
			_, ok = fs.indexObjectCached("object")
			require.False(t, ok)
			if !fault {
				require.Empty(t, trashDirs())
				require.Equal(t, capacity, d.capacityFunc())
				return
			}
			trash := trashDirs()
			require.Len(t, trash, 1)
			require.NoError(t, os.Chmod(trash[0], 0o755))
			size, err := cacheDirSize(trash[0])
			require.NoError(t, err)
			require.Equal(t, int64(len(data)), size, "the files are kept")
			require.NoError(t, os.Chmod(trash[0], mode))
			require.Equal(t, int64(len(data)), d.trash.bytes.Load())
			require.Equal(t, capacity-int64(len(data)), d.capacityFunc())
			d.retryTrash()
			require.Len(t, trashDirs(), 1, "the fault holds")
			require.Equal(t, int64(len(data)), d.trash.bytes.Load())
			require.NoError(t, os.Chmod(trash[0], 0o755))
			d.retryTrash()
			require.Empty(t, trashDirs())
			require.Zero(t, d.trash.bytes.Load())
			require.Equal(t, capacity, d.capacityFunc())
		})
	}
}

// A read of a short entry repairs only the file it read: an entry another cache instance
// on the directory published meanwhile is kept.
func TestIndexObjectStaleShortRepairKeepsOtherInstanceEntry(t *testing.T) {
	for _, replaced := range []bool{true, false} {
		t.Run(map[bool]string{true: "replaced", false: "control"}[replaced], func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newIndexObjectFS(t, true)
			name := IndexObjectEntry("object")
			cacheIndexObject(t, fs, "object", data[:1000])
			diskPath, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			other, err := newIndexFileCache(ctx, fs.indexFileCache.disk.path, fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			t.Cleanup(func() { other.Close(ctx) })
			require.Equal(t, data[10:20], readIndexObjectRange(t, fs, IndexFileObject, 10, 10))
			require.Zero(t, storage.reads.Load())
			var replacement os.FileInfo
			if replaced {
				require.NoError(t, other.Delete(ctx, name))
				written, err := other.Put(ctx, name, []IndexCacheFile{{Name: IndexObjectFile,
					Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }}})
				require.NoError(t, err)
				require.True(t, written)
				replacement, err = os.Stat(filepath.Join(diskPath, IndexObjectFile))
				require.NoError(t, err)
			}
			require.Equal(t, data[2000:2010], readIndexObjectRange(t, fs, IndexFileObject, 2000, 10))
			info, err := os.Stat(filepath.Join(diskPath, IndexObjectFile))
			require.NoError(t, err)
			require.Equal(t, int64(len(data)), info.Size())
			if replaced {
				require.True(t, os.SameFile(replacement, info), "the other instance's entry was dropped")
			}
		})
	}
}

// Trash a removal failed to remove before a restart counts against the capacity of the
// cache loaded after it, and the collector removes it once it can.
func TestIndexFileCacheTrashChargedAfterRestart(t *testing.T) {
	for _, mode := range []os.FileMode{0o500, 0o000, 0o755} {
		fault := mode != 0o755
		t.Run(mode.String(), func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			data := bytes.Repeat([]byte{7}, 300<<10)
			name := "entry"
			c1, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			_, err = c1.Put(ctx, name, []IndexCacheFile{{Name: "data",
				Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }}})
			require.NoError(t, err)
			diskPath, err := c1.disk.pathForDir(name)
			require.NoError(t, err)
			trashDirs := func() []string {
				m, err := filepath.Glob(diskPath + ".*" + cacheDirTrashSuffix)
				require.NoError(t, err)
				return m
			}
			t.Cleanup(func() {
				for _, m := range trashDirs() {
					_ = os.Chmod(m, 0o755)
				}
			})
			if fault {
				require.NoError(t, os.Chmod(diskPath, mode))
				if f, err := os.Create(filepath.Join(diskPath, "probe")); err == nil {
					_ = f.Close()
					t.Skip("permissions are not enforced")
				}
			}
			require.NoError(t, c1.Delete(ctx, name))
			c1.Close(ctx)
			c2, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			defer c2.Close(ctx)
			<-c2.disk.load.done
			if !fault {
				require.Empty(t, trashDirs())
				require.Zero(t, c2.disk.trash.bytes.Load())
				return
			}
			require.Len(t, trashDirs(), 1)
			require.Equal(t, int64(len(data)), c2.disk.trash.bytes.Load())
			require.NoError(t, os.Chmod(trashDirs()[0], 0o755))
			c2.disk.retryTrash()
			require.Empty(t, trashDirs())
			require.Zero(t, c2.disk.trash.bytes.Load())
		})
	}
}

// Trash of bytes a removal cannot measure counts against the capacity after a restart:
// with the bytes the load measured when the entry is removed while the cache loads, or,
// when its bytes are unknown, by holding the cache at what it used, until it is removed.
func TestIndexFileCacheTrashOfUnknownSize(t *testing.T) {
	for _, by := range []string{"deleted while loading", "old trash", "unlistable entry", "control"} {
		t.Run(by, func(t *testing.T) {
			if by != "control" && !permissionsEnforced(t) {
				t.Skip("permissions are not enforced")
			}
			ctx := context.Background()
			dir := t.TempDir()
			data := bytes.Repeat([]byte{7}, 300<<10)
			capacity := int64(64 << 20)
			put := func(c *IndexFileCache, name string) {
				t.Helper()
				written, err := c.Put(ctx, name, []IndexCacheFile{{Name: "data",
					Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }}})
				require.NoError(t, err)
				require.True(t, written)
			}
			restart := func() *IndexFileCache {
				c, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(capacity), "", "", 0, 0, 0)
				require.NoError(t, err)
				<-c.disk.load.done
				return c
			}
			earlier, err := newIndexFileCache(ctx, t.TempDir(), fscache.ConstCapacity(capacity), "", "", 0, 0, 0)
			require.NoError(t, err)
			<-earlier.disk.load.done
			c2 := restart()
			var diskPath string
			var size int64
			for _, name := range []string{"kept", "entry"} {
				put(earlier, name)
				from, err := earlier.disk.pathForDir(name)
				require.NoError(t, err)
				if name == "entry" {
					size, _ = earlier.disk.cache.Size(from)
					require.Positive(t, size)
				}
				to, err := c2.disk.pathForDir(name)
				require.NoError(t, err)
				require.NoError(t, os.MkdirAll(filepath.Dir(to), 0o755))
				require.NoError(t, os.Rename(from, to))
				diskPath = to
			}
			earlier.Close(ctx)
			trashDirs := func() []string {
				m, err := filepath.Glob(diskPath + ".*" + cacheDirTrashSuffix)
				require.NoError(t, err)
				return m
			}
			t.Cleanup(func() {
				for _, m := range append(trashDirs(), diskPath) {
					_ = os.Chmod(m, 0o755)
				}
			})
			switch by {
			case "deleted while loading":
				c2.disk.hooks.beforeIndexLoadedDirs = func() {
					require.NoError(t, os.Chmod(diskPath, 0))
					require.NoError(t, c2.Delete(ctx, "entry"))
				}
				c2.disk.loadCache(ctx)
				c2.disk.hooks.beforeIndexLoadedDirs = nil
			case "old trash":
				require.NoError(t, os.Rename(diskPath, fmt.Sprintf("%s.%d-%d%s", diskPath, 1, 1, cacheDirTrashSuffix)))
				require.NoError(t, os.Chmod(trashDirs()[0], 0))
			case "unlistable entry":
				require.NoError(t, os.Chmod(diskPath, 0))
			}
			c2.Close(ctx)
			c := restart()
			defer c.Close(ctx)
			used := c.disk.cache.Used()
			if by == "control" {
				require.Equal(t, capacity, c.disk.capacityFunc())
				return
			}
			require.Len(t, trashDirs(), 1)
			_, err = os.Stat(diskPath)
			require.True(t, os.IsNotExist(err))
			if by == "deleted while loading" {
				require.Equal(t, size, c.disk.trash.bytes.Load())
				require.Equal(t, capacity-size, c.disk.capacityFunc())
			} else {
				require.Zero(t, c.disk.trash.bytes.Load())
				require.Equal(t, used, c.disk.capacityFunc(), "the cache is held at what it used")
			}
			require.NoError(t, os.Chmod(trashDirs()[0], 0o755))
			c.disk.retryTrash()
			require.Empty(t, trashDirs())
			require.Zero(t, c.disk.trash.bytes.Load())
			require.Equal(t, capacity, c.disk.capacityFunc())
		})
	}
}

// A read that finds its entry missing a file repairs it although the load indexed the
// entry meanwhile: indexing changes no file of it.
func TestIndexObjectRepairOutlivesLoadIndexing(t *testing.T) {
	for _, indexed := range []bool{true, false} {
		t.Run(map[bool]string{true: "indexed", false: "control"}[indexed], func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newIndexObjectFS(t, true)
			<-fs.indexFileCache.disk.load.done
			cacheIndexObject(t, fs, "object", data)
			diskPath, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			require.NoError(t, os.Remove(filepath.Join(diskPath, IndexObjectFile)))
			name := IndexObjectEntry("object")
			_, read, err := fs.indexFileCache.openEntryFile(ctx, name, IndexObjectFile)
			require.Error(t, err)
			if indexed {
				fs.indexFileCache.disk.indexLoadedDirs(ctx, []loadedDir{{path: diskPath}})
			}
			fs.indexFileCache.dropUnopened(ctx, name, IndexObjectFile, read)
			read.done()
			fs.indexFileCache.disk.Flush(ctx)
			_, cached := fs.indexObjectCached("object")
			require.False(t, cached, "the broken entry is dropped")
			storage.reads.Store(0)
			require.Equal(t, data[2000:2010], readIndexObjectRange(t, fs, IndexFileObject, 2000, 10))
			require.Equal(t, data[len(data)-10:], readIndexObjectRange(t, fs, IndexFileObject, int64(len(data)-10), 10))
			require.Equal(t, int64(1), storage.reads.Load(), "the read filled the entry again")
		})
	}
}

// permissionsEnforced reports whether a directory without permissions cannot be listed.
func permissionsEnforced(t *testing.T) bool {
	dir := filepath.Join(t.TempDir(), "probe")
	require.NoError(t, os.Mkdir(dir, 0))
	defer func() { _ = os.Chmod(dir, 0o755) }()
	_, err := os.ReadDir(dir)
	return err != nil
}

// The load holds the cache to what it used once it finds trash of unknown bytes: a
// publication while it loads is not admitted against that space, and the entries it
// found are indexed. Its removal of an entry it cannot list keeps a replacement published
// since.
func TestIndexFileCacheLoadWithUnknownTrash(t *testing.T) {
	for _, by := range []string{"publication", "publication control", "replacement", "replacement control"} {
		t.Run(by, func(t *testing.T) {
			if !permissionsEnforced(t) {
				t.Skip("permissions are not enforced")
			}
			ctx := context.Background()
			data := bytes.Repeat([]byte{7}, 300<<10)
			files := func() []IndexCacheFile {
				return []IndexCacheFile{{Name: "data",
					Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }}}
			}
			earlier, err := newIndexFileCache(ctx, t.TempDir(), fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			<-earlier.disk.load.done
			c, err := newIndexFileCache(ctx, t.TempDir(), fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			defer c.Close(ctx)
			<-c.disk.load.done
			move := func(name string) string {
				written, err := earlier.Put(ctx, name, files())
				require.NoError(t, err)
				require.True(t, written)
				from, err := earlier.disk.pathForDir(name)
				require.NoError(t, err)
				to, err := c.disk.pathForDir(name)
				require.NoError(t, err)
				require.NoError(t, os.MkdirAll(filepath.Dir(to), 0o755))
				require.NoError(t, os.Rename(from, to))
				return to
			}
			kept := move("kept")
			var trash string
			t.Cleanup(func() {
				m, _ := filepath.Glob(filepath.Join(c.disk.path, "*", "*"))
				for _, p := range m {
					_ = os.Chmod(p, 0o755)
				}
			})
			switch by {
			case "publication", "publication control":
				entry := move("entry")
				trash = fmt.Sprintf("%s.%d-%d%s", entry, 1, 1, cacheDirTrashSuffix)
				require.NoError(t, os.Rename(entry, trash))
				if by == "publication" {
					require.NoError(t, os.Chmod(trash, 0))
				}
				fresh, err := c.disk.pathForDir("fresh")
				require.NoError(t, err)
				c.disk.hooks.beforeIndexLoadedDirs = func() {
					_, err := c.Put(ctx, "fresh", files())
					require.NoError(t, err)
				}
				c.disk.loadCache(ctx)
				c.disk.hooks.beforeIndexLoadedDirs = nil
				require.True(t, c.disk.cache.Contains(kept), "the entry the load found is indexed")
				require.Equal(t, by == "publication control", c.disk.cache.Contains(fresh))
				require.LessOrEqual(t, c.disk.cache.Used(), c.disk.capacityFunc())
				if by == "publication" {
					keptSize, _ := c.disk.cache.Size(kept)
					require.Equal(t, keptSize, c.disk.capacityFunc(), "the cache is held at what it used")
				}
			case "replacement", "replacement control":
				entry := move("entry")
				require.NoError(t, os.Chmod(entry, 0))
				if by == "replacement" {
					c.disk.hooks.beforeRemoveUnlistedDir = func(path string) {
						require.Equal(t, entry, path)
						require.NoError(t, os.Chmod(entry, 0o755))
						require.NoError(t, c.Delete(ctx, "entry"))
						written, err := c.Put(ctx, "entry", files())
						require.NoError(t, err)
						require.True(t, written)
					}
				}
				c.disk.loadCache(ctx)
				c.disk.hooks.beforeRemoveUnlistedDir = nil
				size, err := cacheDirSize(entry)
				if by == "replacement" {
					require.NoError(t, err, "the replacement is kept")
					require.Positive(t, size)
					require.True(t, c.disk.cache.Contains(entry))
				} else {
					require.True(t, os.IsNotExist(err), "the unlistable entry is removed")
					require.False(t, c.disk.cache.Contains(entry))
				}
				require.True(t, c.disk.cache.Contains(kept))
			}
		})
	}
}

// A read whose open of a broken entry fails repairs only the entry it read: a
// replacement published before its repair, by a read or by another cache instance on the
// directory, is kept, also when the read could not open the entry.
func TestIndexObjectStaleOpenRepairKeepsReplacement(t *testing.T) {
	for _, by := range []string{"read", "other instance", "unopenable", "control"} {
		replaced := by != "control"
		t.Run(by, func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			diskPath, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			if by == "unopenable" {
				require.NoError(t, os.Chmod(diskPath, 0))
				t.Cleanup(func() {
					m, _ := filepath.Glob(diskPath + "*")
					for _, p := range m {
						_ = os.Chmod(p, 0o755)
					}
				})
				if f, err := os.Open(diskPath); err == nil {
					_ = f.Close()
					t.Skip("permissions are not enforced")
				}
			} else {
				file := filepath.Join(diskPath, IndexObjectFile)
				require.NoError(t, os.Remove(file))
				require.NoError(t, os.Symlink(IndexObjectFile, file))
			}
			name := IndexObjectEntry("object")
			_, read, err := fs.indexFileCache.openEntryFile(context.Background(), name, IndexObjectFile)
			require.Error(t, err)
			require.NotNil(t, read)
			if by == "unopenable" {
				require.NoError(t, os.Chmod(diskPath, 0o755))
			}
			switch by {
			case "read":
				storage.reads.Store(0)
				require.Equal(t, data[2000:2010], readIndexObjectRange(t, fs, IndexFileObject, 2000, 10))
				require.Equal(t, int64(1), storage.reads.Load(), "the read filled the entry again")
			case "other instance", "unopenable":
				other, err := newIndexFileCache(ctx, fs.indexFileCache.disk.path, fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
				require.NoError(t, err)
				t.Cleanup(func() { other.Close(ctx) })
				require.NoError(t, other.Delete(ctx, name))
				written, err := other.Put(ctx, name, []IndexCacheFile{{Name: IndexObjectFile,
					Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }}})
				require.NoError(t, err)
				require.True(t, written)
			}
			fs.indexFileCache.dropUnopened(ctx, name, "", read)
			read.done()
			fs.indexFileCache.disk.Flush(ctx)
			_, cached := fs.indexObjectCached("object")
			require.Equal(t, replaced, cached)
			storage.reads.Store(0)
			require.Equal(t, data[len(data)-10:], readIndexObjectRange(t, fs, IndexFileObject, int64(len(data)-10), 10))
			if replaced {
				require.Zero(t, storage.reads.Load(), "the replacement was dropped")
			}
		})
	}
}

// The repair of a missing entry file is no delete of the object: it waits for no
// publication holding the entry's delete stripe, and keeps the delete mark.
func TestIndexObjectMissingFileRepairWaitsForNoPublication(t *testing.T) {
	fs, storage, data := newIndexObjectFS(t, true)
	cacheIndexObject(t, fs, "object", data)
	diskPath, ok := fs.indexObjectCached("object")
	require.True(t, ok)
	require.NoError(t, os.Remove(filepath.Join(diskPath, IndexObjectFile)))
	stripe := fs.indexFileCache.disk.deletes.stripe(diskPath)
	stripe.RLock()
	unlocked := false
	unlock := func() {
		if !unlocked {
			unlocked = true
			stripe.RUnlock()
		}
	}
	defer unlock()
	mark := fs.indexFileCache.DeleteMark()
	storage.reads.Store(0)
	done := make(chan []byte, 1)
	go func() {
		// The read fills no cache, so the dropped entry stays dropped.
		done <- readIndexObjectRange(t, fs, IndexFileObject|SkipCacheWrites, 2000, 10)
	}()
	select {
	case got := <-done:
		require.Equal(t, data[2000:2010], got)
	case <-time.After(10 * time.Second):
		unlock()
		t.Fatal("the repair waited for the publication")
	}
	fs.indexFileCache.disk.Flush(context.Background())
	_, cached := fs.indexObjectCached("object")
	require.False(t, cached, "the entry is dropped")
	require.Equal(t, mark, fs.indexFileCache.DeleteMark())
}

// An entry whose file cannot be opened, or is missing, is dropped and read from storage,
// and the read fills it again.
func TestIndexObjectUnopenableEntryReadsStorage(t *testing.T) {
	for _, what := range []string{"file", "dir", "missing"} {
		t.Run(what, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			diskPath, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			file := filepath.Join(diskPath, IndexObjectFile)
			switch what {
			case "file":
				require.NoError(t, os.Remove(file))
				require.NoError(t, os.Symlink(IndexObjectFile, file))
			case "dir":
				require.NoError(t, os.RemoveAll(diskPath))
				require.NoError(t, os.MkdirAll(filepath.Dir(diskPath), 0o755))
				require.NoError(t, os.WriteFile(diskPath, nil, 0o644))
			case "missing":
				require.NoError(t, os.Remove(file))
			}
			require.Equal(t, data[3000:3010], readIndexObjectRange(t, fs, IndexFileObject|SkipAllCache, 3000, 10))
			storage.reads.Store(0)
			require.Equal(t, data[2000:2010], readIndexObjectRange(t, fs, IndexFileObject, 2000, 10))
			require.Equal(t, int64(1), storage.reads.Load())
			require.Equal(t, data[len(data)-10:], readIndexObjectRange(t, fs, IndexFileObject, int64(len(data)-10), 10))
			require.Equal(t, int64(1), storage.reads.Load(), "the read filled the entry again")
		})
	}
}

func TestIndexObjectFillWithDiskPolicy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
	}{
		{name: "default_policy"},
		{name: "skip_disk_writes", policy: SkipDiskCacheWrites},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newIndexObjectFS(t, true)
			// Unlike SkipCacheWrites, skipping the table-data tier still permits an index fill.
			vec := &IOVector{FilePath: "object", Policy: IndexFileObject | tc.policy,
				Entries: []IOEntry{{Offset: 100, Size: 64}}}
			defer vec.Release()
			require.NoError(t, fs.Read(context.Background(), vec))
			flushPublishers(fs)
			require.Equal(t, data[100:164], vec.Entries[0].Data)
			_, cached := fs.indexObjectCached("object")
			require.True(t, cached, "the read should cache the whole object in the index-file tier")
			require.Zero(t, fs.diskCache.cache.Used(), "the table-data tier should stay empty")
		})
	}
}

func TestIndexObjectRangeDoesNotFillDiskCache(t *testing.T) {
	for _, tc := range []struct {
		name          string
		cacheWritable bool
	}{
		{name: "skip_disk_writes"},
		{name: "narrow_index_object_read", cacheWritable: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newIndexObjectFS(t, true)
			policy := Policy(IndexFileObject | ExactColumnsRead)
			if !tc.cacheWritable {
				policy |= SkipDiskCacheWrites
			}
			require.Equal(t, data[100:164], readIndexObjectRange(t, fs, policy, 100, 64))
			d := fs.diskCache
			rangePath := d.pathForIOEntry("object", IOEntry{Offset: 100, Size: 64})
			require.False(t, d.cache.Contains(rangePath), "a narrow index-object read should not cache a range in the table-data tier")
		})
	}
}

func TestIndexPrefetchRefillsMissingDataFile(t *testing.T) {
	for _, missing := range []bool{false, true} {
		name := "existing_file"
		if missing {
			name = "missing_file"
		}
		t.Run(name, func(t *testing.T) {
			fs, _, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			entryDir, cached := fs.indexObjectCached("object")
			require.True(t, cached)
			file := filepath.Join(entryDir, IndexObjectFile)
			if missing {
				require.NoError(t, os.Remove(file))
			}
			ctx := WithFileServicePolicy(context.Background(), IndexFileObject)
			require.NoError(t, fs.PrefetchFile(ctx, "object"))
			flushPublishers(fs)
			require.FileExists(t, file, "prefetch should leave a readable data file in the index-file cache")
		})
	}
}

func TestIndexObjectActiveReadSurvivesEviction(t *testing.T) {
	for _, explicitHold := range []bool{true, false} {
		name := "explicit_hold"
		if !explicitHold {
			name = "read_hold"
		}
		t.Run(name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			cache := fs.indexFileCache
			if explicitHold {
				hold, err := cache.Hold(IndexObjectEntry("object"))
				require.NoError(t, err)
				defer hold.Release()
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			entered := make(chan struct{})
			resume := make(chan struct{})
			vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
				Entries: []IOEntry{{Offset: 100, Size: 64, ToCacheData: func(ctx context.Context, r io.Reader, data []byte, alloc CacheDataAllocator) (fscache.Data, error) {
					close(entered)
					<-resume
					return CacheOriginalData(ctx, r, data, alloc)
				}}}}
			defer vec.Release()
			done := make(chan struct{})
			var readErr error
			go func() {
				defer close(done)
				readErr = fs.Read(ctx, vec)
			}()
			defer func() {
				close(resume)
				<-done
				require.NoError(t, readErr)
				require.Zero(t, storage.reads.Load(), "the reader should be served by the index file")
			}()
			select {
			case <-entered:
			case <-ctx.Done():
				t.Fatal("the read never entered the decoder")
			}
			used := cache.disk.cache.Used()
			require.Positive(t, used)
			remaining := cache.disk.cache.ForceEvictWithWait(ctx, used)
			require.Equal(t, used, remaining, "eviction should skip an index object with an active read")
		})
	}
}

func TestMixedIndexObjectReadDoesNotFillDiskCache(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "buffered_entries"
		if stream {
			name = "buffered_entry_and_writer"
		}
		t.Run(name, func(t *testing.T) {
			fs, _, data := newIndexObjectFS(t, true)
			vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
				Entries: []IOEntry{{Offset: 100, Size: 64}, {Offset: 164, Size: 64}}}
			defer vec.Release()
			var out bytes.Buffer
			if stream {
				vec.Entries[1].WriterForRead = &out
			}
			require.NoError(t, fs.Read(context.Background(), vec))
			require.Equal(t, data[100:164], vec.Entries[0].Data)
			if stream {
				require.Equal(t, data[164:228], out.Bytes())
			} else {
				require.Equal(t, data[164:228], vec.Entries[1].Data)
			}
			require.Zero(t, fs.diskCache.cache.Used(), "an index-object read must not fill the table-data cache")
		})
	}
}

func TestIndexPrefetchWhileDiskPrefetchRunning(t *testing.T) {
	for _, concurrent := range []bool{false, true} {
		name := "after_disk_fill"
		if concurrent {
			name = "during_disk_fill"
		}
		t.Run(name, func(t *testing.T) {
			fs, _, data := newIndexObjectFS(t, true)
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
			fs.server.raw = storage
			resume := sync.OnceFunc(func() { close(storage.gate) })
			defer resume()
			ctx := context.Background()
			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
			require.NoError(t, fs.PrefetchObject(ctx, req))
			require.Eventually(t, func() bool {
				cur, _, _ := storage.counts(fs.pathToKey(req.FilePath))
				return cur == 1
			}, 5*time.Second, time.Millisecond, "the table-data download should be under way")
			if !concurrent {
				resume()
				waitPrefetcherIdle(t, fs)
			}
			indexCtx := WithFileServicePolicy(ctx, IndexFileObject)
			require.NoError(t, fs.PrefetchObject(indexCtx, req))
			resume()
			waitPrefetcherIdle(t, fs)
			require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath(req.FilePath, int64(len(data)))))
			_, cached := fs.indexObjectCached(req.FilePath)
			require.True(t, cached, "the index-object prefetch should fill its own cache tier")
		})
	}
}

func TestIndexObjectExternalRemovalIsCacheMiss(t *testing.T) {
	for _, pooled := range []bool{false, true} {
		name := "no_retained_file"
		if pooled {
			name = "default_open_file_pool"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newIndexObjectFS(t, true)
			cache := fs.indexFileCache
			if !pooled {
				cache.disk.files.mu.Lock()
				cache.disk.files.limit = -1
				cache.disk.files.mu.Unlock()
			}
			cacheIndexObject(t, fs, "object", data)
			entryDir, cached := fs.indexObjectCached("object")
			require.True(t, cached)
			// The second cache has its own publisher, as in a different process.
			other, err := NewIndexFileCache(ctx, cache.Path(), fscache.ConstCapacity(64<<20), "other", 0)
			require.NoError(t, err)
			defer other.Close(ctx)
			<-other.disk.load.done
			const policy = IndexFileObject | ExactColumnsRead | SkipCacheWrites
			require.Equal(t, data[100:164], readIndexObjectRange(t, fs, policy, 100, 64))
			require.Zero(t, storage.reads.Load(), "the initial read should come from the cached file")

			require.NoError(t, other.Delete(ctx, IndexObjectEntry("object")))
			require.NoDirExists(t, entryDir)
			require.Equal(t, data[100:164], readIndexObjectRange(t, fs, policy, 100, 64))
			require.EqualValues(t, 1, storage.reads.Load(), "a file removed by another cache instance should be a cache miss")
		})
	}
}

func TestIndexQuotaErrorDoesNotEvictTableCache(t *testing.T) {
	for _, quotaExceeded := range []bool{false, true} {
		name := "successful_index_fill"
		if quotaExceeded {
			name = "index_fill_exceeds_quota"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			fs, _, data := newIndexObjectFS(t, true)
			d := fs.diskCache
			require.NoError(t, d.setFileContent(ctx, "bystander", data, false))
			warmFile := d.wholePath("bystander", int64(len(data)))
			require.FileExists(t, warmFile)
			if quotaExceeded {
				syncFile := d.fileSync
				indexDir := fs.indexFileCache.Path()
				d.fileSync = func(f *os.File) error {
					if filepath.Dir(f.Name()) == indexDir {
						return syscall.EDQUOT
					}
					return syncFile(f)
				}
			}
			indexCtx := WithFileServicePolicy(ctx, IndexFileObject)
			require.NoError(t, fs.PrefetchFile(indexCtx, "object"), "an index-cache write failure is best effort")
			flushPublishers(fs)
			_, cached := fs.indexObjectCached("object")
			require.Equal(t, !quotaExceeded, cached)
			require.FileExists(t, warmFile, "an index-only disk quota failure must not evict table-data files")
		})
	}
}

type reviewIndexJoinContext struct {
	context.Context
	beforeJoin func()
}

func (c reviewIndexJoinContext) Value(key any) any {
	if _, ok := key.(downloadStartedKey); ok {
		c.beforeJoin()
	}
	return c.Context.Value(key)
}

func TestIndexReadSeesConcurrentPublication(t *testing.T) {
	for _, overlap := range []bool{false, true} {
		name := "published_before_read"
		if overlap {
			name = "published_after_cache_lookup"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newIndexObjectFS(t, true)
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
			fs.server.raw = storage
			resumeStorage := sync.OnceFunc(func() { close(storage.gate) })
			fillDone := make(chan struct{})
			var fillErr error
			go func() {
				defer close(fillDone)
				fillErr = fs.PrefetchFile(WithFileServicePolicy(ctx, IndexFileObject), "object")
			}()
			defer func() {
				resumeStorage()
				<-fillDone
			}()
			objectKey := fs.pathToKey("object")
			require.Eventually(t, func() bool {
				cur, _, _ := storage.counts(objectKey)
				return cur == 1
			}, time.Second, time.Millisecond)

			atJoin := make(chan struct{})
			resumeJoin := make(chan struct{})
			unblockJoin := sync.OnceFunc(func() { close(resumeJoin) })
			var readCtx context.Context = ctx
			if overlap {
				// With disk reads skipped, the first lookup of downloadStartedKey is
				// inside readIndexDownload, after the index-file lookup missed.
				readCtx = reviewIndexJoinContext{Context: ctx, beforeJoin: sync.OnceFunc(func() {
					close(atJoin)
					<-resumeJoin
				})}
			} else {
				resumeStorage()
				<-fillDone
				require.NoError(t, fillErr)
			}
			vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipDiskCacheReads | SkipMemoryCache | ExactColumnsRead,
				Entries: []IOEntry{{Offset: 100, Size: 64}}}
			defer vec.Release()
			readDone := make(chan struct{})
			var readErr error
			go func() {
				defer close(readDone)
				readErr = fs.Read(readCtx, vec)
			}()
			defer func() {
				unblockJoin()
				resumeStorage()
				<-readDone
			}()
			if overlap {
				select {
				case <-atJoin:
				case <-ctx.Done():
					t.Fatal("the read never tried to join the index download")
				}
				resumeStorage()
				<-fillDone
				require.NoError(t, fillErr)
				flushPublishers(fs)
				_, cached := fs.indexObjectCached("object")
				require.True(t, cached)
				require.Nil(t, fs.diskCache.download(fs.diskCache.objectKey("object")))
			}
			unblockJoin()
			<-readDone
			require.NoError(t, readErr)
			require.Equal(t, data[100:164], vec.Entries[0].Data)
			_, _, total := storage.counts(objectKey)
			require.Equal(t, 1, total, "the read should use the index file published before it checked the download registry")
		})
	}
}

func recoveryFiles(data []byte) []IndexCacheFile {
	return []IndexCacheFile{{Name: "data",
		Open: func(context.Context) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }}}
}

// recoveryMove publishes name in a scratch cache and moves it into c's directory, where c
// has not indexed it (an entry of another instance on the directory).
func recoveryMove(t *testing.T, c *IndexFileCache, name string, data []byte) string {
	t.Helper()
	ctx := context.Background()
	earlier, err := newIndexFileCache(ctx, t.TempDir(), fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
	require.NoError(t, err)
	<-earlier.disk.load.done
	defer earlier.Close(ctx)
	written, err := earlier.Put(ctx, name, recoveryFiles(data))
	require.NoError(t, err)
	require.True(t, written)
	from, err := earlier.disk.pathForDir(name)
	require.NoError(t, err)
	to, err := c.disk.pathForDir(name)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(filepath.Dir(to), 0o755))
	require.NoError(t, os.Rename(from, to))
	return to
}

func recoveryChmodAll(dir string) {
	_ = filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		_ = os.Chmod(p, 0o755)
		return nil
	})
	m, _ := filepath.Glob(filepath.Join(dir, "*", "*"))
	for _, p := range m {
		_ = os.Chmod(p, 0o755)
	}
}

// Idle collection of an entry the cache has not indexed and cannot list keeps it as trash
// of unknown bytes, which holds the cache, also after a restart, as a Delete does.
func TestIndexFileCacheIdleRemovalChargesUnlistableEntry(t *testing.T) {
	for _, by := range []string{"idle", "delete control"} {
		t.Run(by, func(t *testing.T) {
			if !permissionsEnforced(t) {
				t.Skip("permissions are not enforced")
			}
			ctx := context.Background()
			dir := t.TempDir()
			capacity := int64(64 << 20)
			data := bytes.Repeat([]byte{7}, 300<<10)
			c, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(capacity), "", "", 0, time.Hour, 0)
			require.NoError(t, err)
			<-c.disk.load.done
			t.Cleanup(func() { recoveryChmodAll(dir) })
			written, err := c.Put(ctx, "kept", recoveryFiles(data))
			require.NoError(t, err)
			require.True(t, written)
			hold, err := c.Hold("kept")
			require.NoError(t, err)
			entry := recoveryMove(t, c, "entry", data)
			require.NoError(t, os.Chmod(entry, 0))
			used := c.disk.cache.Used()
			require.Positive(t, used)
			if by == "idle" {
				n, err := c.collectIdle(ctx, time.Now().Add(2*time.Hour))
				t.Logf("collectIdle: n %d err %v", n, err)
			} else {
				require.NoError(t, c.Delete(ctx, "entry"))
			}
			trash, _ := filepath.Glob(entry + ".*" + cacheDirTrashSuffix)
			require.Len(t, trash, 1)
			t.Logf("trash name %s, trash.bytes %d, unknown %d, capacity %d, used %d",
				filepath.Base(trash[0]), c.disk.trash.bytes.Load(), c.disk.trash.unknown, c.disk.capacityFunc(), used)
			if !(c.disk.trash.bytes.Load() > 0 || c.disk.capacityFunc() <= used) {
				t.Errorf("the unremoved trash is charged nothing: capacity %d with %d used", c.disk.capacityFunc(), used)
			}
			hold.Release()
			c.Close(ctx)
			c2, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(capacity), "", "", 0, 0, 0)
			require.NoError(t, err)
			defer c2.Close(ctx)
			<-c2.disk.load.done
			used2 := c2.disk.cache.Used()
			t.Logf("after restart: trash.bytes %d, capacity %d, used %d", c2.disk.trash.bytes.Load(), c2.disk.capacityFunc(), used2)
			require.True(t, c2.disk.trash.bytes.Load() > 0 || c2.disk.capacityFunc() <= used2,
				"after a restart the unremoved trash is charged nothing")
		})
	}
}

// A Put that started before the load found trash of unknown bytes publishes nothing while
// the load holds publications.
func TestIndexFileCacheHoldStopsPublicationInFlight(t *testing.T) {
	for _, by := range []string{"in flight", "control"} {
		t.Run(by, func(t *testing.T) {
			if !permissionsEnforced(t) {
				t.Skip("permissions are not enforced")
			}
			ctx := context.Background()
			dir := t.TempDir()
			data := bytes.Repeat([]byte{7}, 300<<10)
			c, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			defer c.Close(ctx)
			<-c.disk.load.done
			t.Cleanup(func() { recoveryChmodAll(dir) })
			kept := recoveryMove(t, c, "kept", data)
			entry := recoveryMove(t, c, "entry", data)
			require.NoError(t, os.Chmod(entry, 0))
			fresh, err := c.disk.pathForDir("fresh")
			require.NoError(t, err)
			reached := make(chan struct{})
			release := make(chan struct{})
			putDone := make(chan bool, 1)
			startPut := func() {
				go func() {
					written, err := c.Put(ctx, "fresh", recoveryFiles(data))
					require.NoError(t, err)
					putDone <- written
				}()
			}
			c.disk.hooks.beforeFilePublication = func() {
				close(reached)
				<-release
			}
			c.disk.hooks.beforeRemoveUnlistedDir = func(path string) {
				if by == "in flight" {
					startPut()
					<-reached
				}
			}
			c.disk.hooks.beforeIndexLoadedDirs = func() {
				if by == "control" {
					startPut()
				}
				require.True(t, c.disk.trashHoldsPublications(), "the load holds publications")
				close(release)
				<-putDone
			}
			c.disk.loadCache(ctx)
			c.disk.hooks = diskCacheHooks{}
			require.True(t, c.disk.cache.Contains(kept))
			require.False(t, c.disk.cache.Contains(fresh), "published while the load holds publications")
		})
	}
}

// An eviction whose rename fails keeps the entry indexed, as a failed Delete does.
func TestIndexFileCacheEvictionRenameFailureKeepsEntry(t *testing.T) {
	for _, by := range []string{"eviction", "delete control"} {
		t.Run(by, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			data := bytes.Repeat([]byte{7}, 300<<10)
			c := newTestIndexFileCache(t, dir, 400<<10)
			t.Cleanup(func() { recoveryChmodAll(dir) })
			written, err := c.Put(ctx, "a", recoveryFiles(data))
			require.NoError(t, err)
			require.True(t, written)
			a, err := c.disk.pathForDir("a")
			require.NoError(t, err)
			size, _ := c.disk.cache.Size(a)
			shard := filepath.Dir(a)
			var other string
			for i := 0; ; i++ {
				other = "b" + string(rune('a'+i%26)) + string(rune('a'+i/26))
				p, err := c.disk.pathForDir(other)
				require.NoError(t, err)
				if filepath.Dir(p) != shard {
					break
				}
			}
			require.NoError(t, os.Chmod(shard, 0o500))
			if f, err := os.Create(filepath.Join(shard, "probe")); err == nil {
				_ = f.Close()
				t.Skip("permissions are not enforced")
			}
			if by == "eviction" {
				_, err = c.Put(ctx, other, recoveryFiles(data))
				require.NoError(t, err)
			} else {
				require.Error(t, c.Delete(ctx, "a"))
			}
			c.disk.Flush(ctx)
			_, statErr := os.Stat(a)
			onDisk := statErr == nil
			t.Logf("a on disk %v, indexed %v, trash.bytes %d, used %d, capacity %d",
				onDisk, c.disk.cache.Contains(a), c.disk.trash.bytes.Load(), c.disk.cache.Used(), c.disk.capacityFunc())
			require.True(t, onDisk)
			require.True(t, c.disk.cache.Contains(a), "the entry left on disk is indexed")
			got, _ := c.disk.cache.Size(a)
			require.Equal(t, size, got)
			require.LessOrEqual(t, c.disk.cache.Used(), c.disk.capacityFunc())
		})
	}
}

// The load indexes an entry with the bytes it holds when indexed, also when another
// instance replaced it after it was measured.
func TestIndexFileCacheLoadIndexesCurrentSize(t *testing.T) {
	for _, by := range []string{"replaced", "control"} {
		t.Run(by, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			c, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			defer c.Close(ctx)
			<-c.disk.load.done
			entry := recoveryMove(t, c, "entry", bytes.Repeat([]byte{7}, 100<<10))
			other, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			<-other.disk.load.done
			defer other.Close(ctx)
			c.disk.hooks.beforeIndexLoadedDirs = func() {
				if by == "replaced" {
					require.NoError(t, other.Delete(ctx, "entry"))
					written, err := other.Put(ctx, "entry", recoveryFiles(bytes.Repeat([]byte{8}, 2<<20)))
					require.NoError(t, err)
					require.True(t, written)
				}
			}
			c.disk.loadCache(ctx)
			c.disk.hooks = diskCacheHooks{}
			indexed, ok := c.disk.cache.Size(entry)
			require.True(t, ok)
			actual, err := cacheDirSize(entry)
			require.NoError(t, err)
			t.Logf("indexed %d, on disk %d", indexed, actual)
			require.Equal(t, actual, indexed, "indexed with the bytes of the entry it replaced")
		})
	}
}

// An entry whose trash cannot be removed does not stop the idle collection of the entries
// after it in the same pass.
func TestIndexFileCacheIdleCollectionPassesUnremovableTrash(t *testing.T) {
	for _, by := range []string{"fault", "control"} {
		t.Run(by, func(t *testing.T) {
			if by == "fault" && !permissionsEnforced(t) {
				t.Skip("permissions are not enforced")
			}
			ctx := context.Background()
			dir := t.TempDir()
			data := bytes.Repeat([]byte{7}, 100<<10)
			c, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(64<<20), "", "", 0, time.Hour, 0)
			require.NoError(t, err)
			defer c.Close(ctx)
			<-c.disk.load.done
			t.Cleanup(func() { recoveryChmodAll(dir) })
			var paths []string
			for i := 0; i < 8; i++ {
				name := "e" + string(rune('a'+i))
				written, err := c.Put(ctx, name, recoveryFiles(data))
				require.NoError(t, err)
				require.True(t, written)
				p, _ := c.disk.pathForDir(name)
				paths = append(paths, p)
			}
			dirs, err := c.entryDirs()
			require.NoError(t, err)
			if by == "fault" {
				require.NoError(t, os.Chmod(dirs[0], 0o500))
			}
			n, err := c.collectIdle(ctx, time.Now().Add(2*time.Hour))
			left, _ := c.entryDirs()
			t.Logf("collected %d, err %v, entries left %d, indexed used %d", n, err, len(left), c.disk.cache.Used())
			require.Empty(t, left, "idle entries are left after the pass")
		})
	}
}

// Concurrent puts, opens, holds, deletes, evictions, idle collection and trash
// retries finish and leave the directory and the index in agreement.
func TestIndexFileCacheConcurrentLifecycleAgreement(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	c, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(1<<20), "", "", 0, time.Hour, 0)
	require.NoError(t, err)
	defer c.Close(ctx)
	<-c.disk.load.done
	names := []string{"n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"}
	stop := time.Now().Add(3 * time.Second)
	done := make(chan struct{})
	var workers int
	run := func(f func(i int)) {
		workers++
		go func() {
			defer func() { done <- struct{}{} }()
			for i := 0; time.Now().Before(stop); i++ {
				f(i)
			}
		}()
	}
	for w := 0; w < 3; w++ {
		run(func(i int) {
			n := names[(i*7+w)%len(names)]
			_, _ = c.Put(ctx, n, recoveryFiles(bytes.Repeat([]byte{byte(i)}, (100+i%5*50)<<10)))
		})
		run(func(i int) {
			n := names[(i*5+w)%len(names)]
			if fs, err := c.Open(ctx, n, "data"); err == nil {
				for _, f := range fs {
					_ = f.Close()
				}
			}
		})
	}
	run(func(i int) { _ = c.Delete(ctx, names[i%len(names)]) })
	run(func(i int) {
		h, err := c.Hold(names[(i*3)%len(names)])
		if err == nil {
			time.Sleep(time.Millisecond)
			h.Release()
		}
	})
	run(func(i int) {
		if h := c.HoldCached(names[(i*11)%len(names)]); h != nil {
			h.ReleaseSuperseded()
		}
	})
	run(func(i int) { _, _ = c.collectIdle(ctx, time.Now().Add(2*time.Hour)); time.Sleep(time.Millisecond) })
	run(func(i int) { c.disk.retryTrash(); c.disk.cache.ForceEvict(ctx, 200<<10); time.Sleep(time.Millisecond) })
	timeout := time.After(60 * time.Second)
	for i := 0; i < workers; i++ {
		select {
		case <-done:
		case <-timeout:
			t.Fatal("deadlock: workers did not finish")
		}
	}
	dirs, err := c.entryDirs()
	require.NoError(t, err)
	var sum int64
	for _, p := range dirs {
		size, ok := c.disk.cache.Size(p)
		require.True(t, ok, "entry on disk not indexed: %s", p)
		actual, err := cacheDirSize(p)
		require.NoError(t, err)
		require.Equal(t, actual, size, "indexed size of %s", p)
		sum += size
	}
	require.Equal(t, sum, c.disk.cache.Used(), "used bytes match the entries on disk")
	trash, _ := filepath.Glob(filepath.Join(dir, "*", "*"+cacheDirTrashSuffix))
	require.Empty(t, trash)
	require.Zero(t, c.disk.trash.bytes.Load())
}

// The load finishes when it indexes an entry larger than the capacity that it cannot
// rename away: the eviction that fails is not rescheduled.
func TestIndexFileCacheLoadFinishesWhenEvictionCannotRemove(t *testing.T) {
	for _, by := range []string{"rename denied", "control"} {
		t.Run(by, func(t *testing.T) {
			if by != "control" && !permissionsEnforced(t) {
				t.Skip("permissions are not enforced")
			}
			ctx := context.Background()
			dir := t.TempDir()
			c1, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(64<<20), "", "", 0, 0, 0)
			require.NoError(t, err)
			<-c1.disk.load.done
			written, err := c1.Put(ctx, "entry", recoveryFiles(bytes.Repeat([]byte{7}, 300<<10)))
			require.NoError(t, err)
			require.True(t, written)
			entry, err := c1.disk.pathForDir("entry")
			require.NoError(t, err)
			c1.Close(ctx)
			shard := filepath.Dir(entry)
			if by != "control" {
				require.NoError(t, os.Chmod(shard, 0o500))
				t.Cleanup(func() { _ = os.Chmod(shard, 0o755) })
			}
			c2, err := newIndexFileCache(ctx, dir, fscache.ConstCapacity(100<<10), "", "", 0, 0, 0)
			require.NoError(t, err)
			defer c2.Close(ctx)
			select {
			case <-c2.disk.load.done:
			case <-time.After(10 * time.Second):
				t.Fatal("the load did not finish")
			}
			c2.disk.Flush(ctx)
			_, err = os.Stat(entry)
			if by == "control" {
				require.True(t, os.IsNotExist(err), "the oversized entry is evicted")
			} else {
				require.NoError(t, err)
				require.True(t, c2.disk.cache.Contains(entry), "the entry it cannot remove stays charged")
			}
		})
	}
}
