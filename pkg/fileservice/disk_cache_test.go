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
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"io/fs"
	mrand "math/rand"
	"os"
	pathpkg "path"
	"path/filepath"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/malloc"
	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/perfcounter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func countLoggedEvents(ctx context.Context, ev stringRef) int {
	logger, _ := ctx.Value(EventLoggerKey).(*eventLogger)
	if logger == nil {
		return 0
	}
	logger.mu.Lock()
	defer logger.mu.Unlock()

	var count int
	for _, event := range *logger.events {
		if event.ev == ev {
			count++
		}
	}
	return count
}

func TestDiskCache(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	// new
	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	// update
	testUpdate := func(cache *DiskCache) {
		vec := &IOVector{
			FilePath: "foo",
			Entries: []IOEntry{
				{
					Offset: 0,
					Size:   1,
					Data:   []byte("a"),
				},
				// no data
				{
					Offset: 98,
					Size:   0,
				},
				// size unknown
				{
					Offset: 9999,
					Size:   -1,
					Data:   []byte("abc"),
				},
			},
		}
		err = cache.Update(ctx, vec, false)
		assert.Nil(t, err)
	}
	testUpdate(cache)

	// written is the cache file of the entry written.
	stat := func() os.FileInfo {
		info, err := os.Stat(cache.pathForIOEntry("foo", IOEntry{Offset: 0, Size: 1}))
		require.NoError(t, err)
		return info
	}
	written := stat()

	// update again
	testUpdate(cache)

	assert.True(t, os.SameFile(written, stat()))

	// read
	testRead := func(cache *DiskCache) {
		buf := new(bytes.Buffer)
		var r io.ReadCloser
		vec := &IOVector{
			FilePath: "foo",
			Entries: []IOEntry{
				// written data
				{
					Offset:            0,
					Size:              1,
					WriterForRead:     buf,
					ReadCloserForRead: &r,
				},
				// not exists
				{
					Offset: 1,
					Size:   1,
				},
				// bad offset
				{
					Offset: 999,
					Size:   1,
				},
			},
		}
		err = cache.Read(ctx, vec)
		assert.Nil(t, err)
		assert.NotNil(t, r)
		defer r.Close()
		assert.True(t, vec.Entries[0].done)
		assert.Equal(t, []byte("a"), vec.Entries[0].Data)
		assert.Equal(t, []byte("a"), buf.Bytes())
		bs, err := io.ReadAll(r)
		assert.Nil(t, err)
		assert.Equal(t, []byte("a"), bs)
		assert.False(t, vec.Entries[1].done)
		assert.False(t, vec.Entries[2].done)
		vec.Release()
	}
	testRead(cache)

	// read again
	testRead(cache)

	// new cache instance and read
	cache, err = NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	testRead(cache)

	assert.True(t, os.SameFile(written, stat()))

	// new cache instance and update
	cache, err = NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	testUpdate(cache)

	assert.True(t, os.SameFile(written, stat()))

	// delete file
	err = cache.DeletePaths(ctx, []string{"foo"})
	assert.Nil(t, err)
}

func TestDiskCacheWriteAgain(t *testing.T) {

	dir := t.TempDir()
	ctx := context.Background()
	var counterSet perfcounter.CounterSet
	ctx = perfcounter.WithCounterSet(ctx, &counterSet)

	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(4096), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	// update
	err = cache.Update(ctx, &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{
				Size: 3,
				Data: []byte("foo"),
			},
		},
	}, false)
	assert.Nil(t, err)

	// update another entry
	err = cache.Update(ctx, &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{
				Size:   3,
				Data:   []byte("foo"),
				Offset: 99,
			},
		},
	}, false)
	assert.Nil(t, err)

	// update again, should write cache file again
	err = cache.Update(ctx, &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{
				Size: 3,
				Data: []byte("foo"),
			},
		},
	}, false)
	assert.Nil(t, err)

	vec := &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{
				Size: 3,
			},
		},
	}
	err = cache.Read(ctx, vec)
	assert.Nil(t, err)
	assert.Equal(t, int64(1), counterSet.FileService.Cache.Disk.Hit.Load())
	vec.Release()

}

func TestIOEntryReadErrorClearsAllocatedDataOwner(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "truncated")
	require.NoError(t, os.WriteFile(path, []byte("ab"), 0o644))

	file, err := os.Open(path)
	require.NoError(t, err)
	defer file.Close()

	entry := IOEntry{Size: 3}
	err = entry.ReadFromOSFile(ctx, file, DefaultCacheDataAllocator())
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)
	require.Nil(t, entry.Data)
	require.Nil(t, entry.releaseData)
}

func TestIOEntryAllocatedDataErrorCleanupIsIdempotent(t *testing.T) {
	entry := IOEntry{Size: 3}
	finally, err := entry.prepareData(context.Background())
	require.NoError(t, err)
	err = io.ErrUnexpectedEOF

	finally(&err)
	finally(&err)

	require.Nil(t, entry.Data)
	require.Nil(t, entry.releaseData)
}

func TestIOEntryCacheConversionErrorClearsAllocatedReader(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "data")
	require.NoError(t, os.WriteFile(path, []byte("foo"), 0o644))

	file, err := os.Open(path)
	require.NoError(t, err)
	defer file.Close()

	var reader io.ReadCloser
	entry := IOEntry{
		Size:              3,
		ReadCloserForRead: &reader,
		ToCacheData: func(context.Context, io.Reader, []byte, CacheDataAllocator) (fscache.Data, error) {
			return nil, fmt.Errorf("cache conversion failed")
		},
	}
	err = entry.ReadFromOSFile(ctx, file, DefaultCacheDataAllocator())
	require.EqualError(t, err, "cache conversion failed")
	require.Nil(t, reader)
	require.Nil(t, entry.Data)
	require.Nil(t, entry.releaseData)
}

func TestDiskCacheReusesRangeFileFromStart(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)

	err = cache.Update(ctx, &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{{
			Offset: 99,
			Size:   3,
			Data:   []byte("foo"),
		}},
	}, false)
	require.NoError(t, err)

	vector := &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{Offset: 99, Size: 3},
			{Offset: 99, Size: 3},
		},
	}
	defer vector.Release()

	require.NoError(t, cache.Read(ctx, vector))
	for i := range vector.Entries {
		require.True(t, vector.Entries[i].done)
		require.Equal(t, []byte("foo"), vector.Entries[i].Data)
	}
}

func TestDiskCacheFileCache(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	vector := IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{
				Offset: 0,
				Size:   3,
				Data:   []byte("foo"),
			},
			{
				Offset: 3,
				Size:   3,
				Data:   []byte("bar"),
			},
		},
	}

	data, err := io.ReadAll(newIOEntriesReader(ctx, vector.Entries))
	assert.Nil(t, err)
	err = cache.SetFile(ctx, vector.FilePath, func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(data)), nil
	})
	assert.Nil(t, err)

	readVector := &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{
				Offset: 0,
				Size:   2,
			},
			{
				Offset: 2,
				Size:   2,
			},
			{
				Offset: 4,
				Size:   2,
			},
		},
	}
	err = cache.Read(ctx, readVector)
	assert.Nil(t, err)
	assert.Equal(t, []byte("fo"), readVector.Entries[0].Data)
	assert.Equal(t, []byte("ob"), readVector.Entries[1].Data)
	assert.Equal(t, []byte("ar"), readVector.Entries[2].Data)
	readVector.Release()

}

func TestDiskCacheCanonicalizesFullFileKeys(t *testing.T) {
	ctx := context.Background()
	for _, filePath := range []string{"shared:/foo", "/foo"} {
		t.Run(filePath, func(t *testing.T) {
			cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
			require.NoError(t, err)
			defer cache.Close(ctx)

			err = cache.SetFile(ctx, filePath, func(context.Context) (io.ReadCloser, error) {
				return io.NopCloser(bytes.NewReader([]byte("foo"))), nil
			})
			require.NoError(t, err)

			vector := &IOVector{
				FilePath: "foo",
				Entries:  []IOEntry{{Offset: 0, Size: 3}},
			}
			defer vector.Release()
			require.NoError(t, cache.Read(ctx, vector))
			require.True(t, vector.Entries[0].done)
			require.Equal(t, []byte("foo"), vector.Entries[0].Data)

			require.NoError(t, cache.DeletePaths(ctx, []string{"foo"}))
			require.False(t, cache.cache.Contains(cache.wholePath("foo", 3)))
		})
	}
}

func TestDiskCacheRejectsInvalidFullFilePath(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)

	openReaderCalled := false
	err = cache.SetFile(ctx, "foo#bar", func(context.Context) (io.ReadCloser, error) {
		openReaderCalled = true
		return io.NopCloser(bytes.NewReader([]byte("foo"))), nil
	})
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrInvalidPath), "unexpected error: %v", err)
	require.False(t, openReaderCalled)
}

func TestDiskCacheDeletePathsRejectsInvalidListAtomically(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)

	err = cache.SetFile(ctx, "shared:/foo", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader([]byte("foo"))), nil
	})
	require.NoError(t, err)

	err = cache.DeletePaths(ctx, []string{"foo", "foo#bar"})
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrInvalidPath), "unexpected error: %v", err)

	vector := &IOVector{
		FilePath: "foo",
		Entries:  []IOEntry{{Offset: 0, Size: 3}},
	}
	defer vector.Release()
	require.NoError(t, cache.Read(ctx, vector))
	require.True(t, vector.Entries[0].done)
	require.Equal(t, []byte("foo"), vector.Entries[0].Data)
}

func TestDiskCacheStaleRepairReplacesCacheEntrySize(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(8192+4096-1), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)

	err = cache.SetFile(ctx, "foo", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader([]byte("foo"))), nil
	})
	require.NoError(t, err)

	stalePath := cache.wholePath("foo", 3)
	require.NoError(t, os.Remove(stalePath))
	require.True(t, cache.cache.Contains(stalePath))

	// A rewritten object of another size is another range file; the stale entry is
	// evicted first.
	err = cache.SetFile(ctx, "foo", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(bytes.Repeat([]byte("x"), 8192))), nil
	})
	require.NoError(t, err)
	fooPath := cache.wholePath("foo", 8192)
	require.True(t, cache.cache.Contains(fooPath))
	require.False(t, cache.cache.Contains(stalePath))

	err = cache.SetFile(ctx, "bar", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader([]byte("xy"))), nil
	})
	require.NoError(t, err)

	require.False(t, cache.cache.Contains(fooPath))
	require.True(t, cache.cache.Contains(cache.wholePath("bar", 2)))
}

func TestDiskCacheRemovesFileEvictedDuringWrite(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(4096), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)

	err = cache.SetFile(ctx, "foo", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(bytes.Repeat([]byte("x"), 8192))), nil
	})
	require.NoError(t, err)

	fooPath := cache.wholePath("foo", 8192)
	require.False(t, cache.cache.Contains(fooPath))
	cache.Flush(ctx)
	_, err = os.Stat(fooPath)
	require.True(t, os.IsNotExist(err))
}

func TestDiskCacheDirSize(t *testing.T) {
	ctx := context.Background()
	var counter perfcounter.CounterSet
	ctx = perfcounter.WithCounterSet(ctx, &counter)

	dir := t.TempDir()
	capacity := 1 << 20
	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(int64(capacity)), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	// Keep enough files to exercise repeated eviction, but avoid turning this
	// invariant check into an O(n^2) test: dirSize walks the whole cache
	// directory after every write.
	data := bytes.Repeat([]byte("a"), capacity/4)
	for i := 0; i < capacity/len(data)*2; i++ {
		err := cache.Update(ctx, &IOVector{
			FilePath: fmt.Sprintf("%v", i),
			Entries: []IOEntry{
				{
					Offset: 0,
					Size:   int64(len(data)),
					Data:   data,
				},
			},
		}, false)
		assert.Nil(t, err)
		cache.Flush(ctx)
		size := dirSize(dir)
		assert.LessOrEqual(t, size, capacity)
	}
	// Evict counter removed
}

func dirSize(path string) (ret int) {
	if err := filepath.WalkDir(path, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		stat, err := d.Info()
		if err != nil {
			return err
		}
		ret += int(fileSize(stat))
		return nil
	}); err != nil {
		panic(err)
	}
	return
}

func benchmarkDiskCacheWriteThenRead(
	b *testing.B,
	size int64,
) {
	b.Helper()

	b.SetBytes(size)
	data := bytes.Repeat([]byte("a"), int(size))

	dir := b.TempDir()
	ctx := context.Background()

	cache, err := NewDiskCache(
		ctx,
		dir,
		fscache.ConstCapacity(10<<30),
		nil,
		false,
		nil,
		"",
	)
	if err != nil {
		b.Fatal(err)
	}
	defer cache.Close(ctx)

	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		buf := make([]byte, 0, size)
		for pb.Next() {

			path := strconv.FormatInt(mrand.Int63(), 10)

			// update
			err := cache.Update(
				ctx,
				&IOVector{
					FilePath: path,
					Entries: []IOEntry{
						{
							Size: size,
							Data: data,
						},
					},
				},
				false,
			)
			if err != nil {
				b.Fatal(err)
			}

			// read
			vec := &IOVector{
				FilePath: path,
				Entries: []IOEntry{
					{
						Size: size,
						Data: buf,
					},
				},
			}
			err = cache.Read(
				ctx,
				vec,
			)
			if err != nil {
				b.Fatal(err)
			}
			if !bytes.Equal(vec.Entries[0].Data, data) {
				b.Fatal()
			}

		}
	})

}

func BenchmarkDiskCacheWriteThenRead(b *testing.B) {
	b.Run("4K", func(b *testing.B) {
		benchmarkDiskCacheWriteThenRead(b, 4096)
	})
	b.Run("1M", func(b *testing.B) {
		benchmarkDiskCacheWriteThenRead(b, 1<<20)
	})
	b.Run("16M", func(b *testing.B) {
		benchmarkDiskCacheWriteThenRead(b, 16<<20)
	})
}

func benchmarkDiskCacheReadRandomOffsetAtLargeFile(
	b *testing.B,
	fileSize int64,
	readSize int64,
) {
	b.Helper()

	b.SetBytes(readSize)
	data := make([]byte, fileSize)
	_, err := rand.Read(data)
	if err != nil {
		b.Fatal(err)
	}

	dir := b.TempDir()
	ctx := context.Background()

	cache, err := NewDiskCache(
		ctx,
		dir,
		fscache.ConstCapacity(8<<30),
		nil,
		false,
		nil,
		"",
	)
	if err != nil {
		b.Fatal(err)
	}
	defer cache.Close(ctx)

	err = cache.SetFile(ctx, "foo", func(ctx context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(data)), nil
	})
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		buf := make([]byte, 0, readSize)
		for pb.Next() {

			// read
			offset := mrand.Intn(int(fileSize - readSize))
			vec := &IOVector{
				FilePath: "foo",
				Entries: []IOEntry{
					{
						Offset: int64(offset),
						Size:   readSize,
						Data:   buf,
					},
				},
			}
			err = cache.Read(
				ctx,
				vec,
			)
			if err != nil {
				b.Fatal(err)
			}
			if !bytes.Equal(vec.Entries[0].Data, data[offset:offset+int(readSize)]) {
				b.Fatal()
			}

		}
	})

}

func BenchmarkDiskCacheReadRandomOffsetAtLargeFile(b *testing.B) {
	b.Run("4K", func(b *testing.B) {
		benchmarkDiskCacheReadRandomOffsetAtLargeFile(b, 1<<30, 4096)
	})
	b.Run("1M", func(b *testing.B) {
		benchmarkDiskCacheReadRandomOffsetAtLargeFile(b, 1<<30, 1<<20)
	})
	b.Run("16M", func(b *testing.B) {
		benchmarkDiskCacheReadRandomOffsetAtLargeFile(b, 1<<30, 16<<20)
	})
}

func BenchmarkDiskCacheMultipleIOEntries(b *testing.B) {
	dir := b.TempDir()
	ctx := context.Background()

	cache, err := NewDiskCache(
		ctx,
		dir,
		fscache.ConstCapacity(8<<30),
		nil,
		false,
		nil,
		"",
	)
	if err != nil {
		b.Fatal(err)
	}
	defer cache.Close(ctx)

	err = cache.SetFile(ctx, "foo", func(ctx context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(bytes.Repeat([]byte("a"), 1<<20))), nil
	})
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var entries []IOEntry
		for i := 0; i < 64; i++ {
			entries = append(entries, IOEntry{
				Offset: 0,
				Size:   4096,
			})
		}
		var counter perfcounter.CounterSet
		ctx := perfcounter.WithCounterSet(ctx, &counter)
		vec := &IOVector{
			FilePath: "foo",
			Entries:  entries,
		}
		err := cache.Read(ctx, vec)
		if err != nil {
			b.Fatal(err)
		}
		// OpenFullFile counter removed
		vec.Release()
	}
}

func TestDiskCacheClearFiles(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	// write garbage temp file
	err := os.WriteFile(
		filepath.Join(dir, "a"+cacheFileTempSuffix),
		[]byte("foo"),
		0644,
	)
	assert.Nil(t, err)
	err = os.Chtimes(
		filepath.Join(dir, "a"+cacheFileTempSuffix),
		time.Now().Add(-time.Hour*24),
		time.Now().Add(-time.Hour*24),
	)
	assert.Nil(t, err)

	// write garbage file
	err = os.WriteFile(
		filepath.Join(dir, "foo"),
		[]byte("foo"),
		0644,
	)
	assert.Nil(t, err)

	files, err := filepath.Glob(filepath.Join(dir, "*"))
	assert.Nil(t, err)
	numFiles := len(files)

	_, err = NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)

	files, err = filepath.Glob(filepath.Join(dir, "*"))
	assert.Nil(t, err)
	if len(files) != numFiles-2 {
		t.Fatalf("got %v", files)
	}

}

func TestDiskCacheBadWrite(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	written, err := cache.writeFileTo(
		ctx,
		filepath.Join(dir, "foo"),
		func(ctx context.Context) (io.ReadCloser, error) {
			// bad reader
			return nil, io.ErrUnexpectedEOF
		},
		false, notWholeFill, nil, diskFillTarget{d: cache, diskPath: filepath.Join(dir, "foo")}, readMark{},
	)
	assert.ErrorIs(t, err, io.ErrUnexpectedEOF)
	if written {
		t.Fatal()
	}

	// ensure no temp files
	files, err := filepath.Glob(filepath.Join(dir, "*"))
	assert.Nil(t, err)
	assert.Empty(t, files)

}

func TestDiskCacheGlobalSizeHint(t *testing.T) {
	ctx := context.Background()

	dir := t.TempDir()
	cache, err := NewDiskCache(
		context.Background(),
		dir,
		fscache.ConstCapacity(1<<20),
		nil,
		false,
		nil,
		"test",
	)
	assert.Nil(t, err)
	defer cache.Close(ctx)

	ch := make(chan int64, 1)
	cache.Evict(ctx, ch)
	n := <-ch
	if n > 1<<20 {
		t.Fatalf("got %v", n)
	}

	// shrink
	GlobalDiskCacheSizeHint.Store(1 << 10)
	defer GlobalDiskCacheSizeHint.Store(0)
	cache.Evict(ctx, ch)
	n = <-ch
	if n > 1<<10 {
		t.Fatalf("got %v", n)
	}

	// shrink
	GlobalDiskCacheSizeHint.Store(1 << 9)
	defer GlobalDiskCacheSizeHint.Store(0)
	ret := EvictDiskCaches(ctx)
	if ret["test"] > 1<<9 {
		t.Fatalf("got %v", ret)
	}

}

func TestDiskCacheReadUsesMemoryCacheReservation(t *testing.T) {
	ctx := context.Background()
	const request = 10
	capacity := int64(1 << 20)
	memoryCache := NewMemCache(func() int64 { return capacity }, nil, nil, "")
	defer memoryCache.Close(ctx)
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, memoryCache, "")
	require.NoError(t, err)
	defer cache.Close(ctx)

	capacity = int64(memoryCache.BackingSize(request))
	seed := NewBytes(make([]byte, request, capacity))
	_, err = memoryCache.cache.Set(ctx, fscache.CacheKey{Path: "seed", Sz: request}, seed)
	require.NoError(t, err)
	seed.Release()
	require.Equal(t, int64(memoryCache.BackingSize(request)), memoryCache.cache.Used())

	err = cache.Update(ctx, &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{{
			Offset: 0,
			Size:   3,
			Data:   []byte("foo"),
		}},
	}, false)
	require.Nil(t, err)

	vec := &IOVector{
		FilePath: "foo",
		Entries: []IOEntry{{
			Offset: 0,
			Size:   3,
			ToCacheData: func(ctx context.Context, _ io.Reader, _ []byte, allocator CacheDataAllocator) (fscache.Data, error) {
				require.Same(t, memoryCache, allocator)
				return allocator.AllocateCacheDataWithHint(ctx, request, malloc.NoClear), nil
			},
		}},
	}
	err = cache.Read(ctx, vec)
	require.NoError(t, err)
	require.True(t, vec.Entries[0].done)
	data := vec.Entries[0].CachedData.(*Bytes)
	require.Same(t, memoryCache.allocator.owner, data.CacheDataOwner())
	require.Zero(t, memoryCache.cache.Used())
	require.Equal(t, int64(memoryCache.BackingSize(request)), memoryCache.reservedBytes.Load())
	vec.Release()
	memoryCache.Flush(ctx)
	require.Zero(t, memoryCache.reservedBytes.Load())
}

// Files found on disk at startup are older than a file indexed while the cache loads.
func TestDiskCacheLoadIndexesFilesAsOldest(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(12<<10), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	old := cache.wholePath("old", 8<<10)
	require.NoError(t, os.MkdirAll(filepath.Dir(old), 0755))
	require.NoError(t, os.WriteFile(old, make([]byte, 8<<10), 0644))

	live := cache.wholePath("live", 8<<10)
	cache.cache.Set(ctx, live, struct{}{}, 8<<10)
	cache.loadCache(ctx)
	require.True(t, cache.cache.Contains(live))
	require.False(t, cache.cache.Contains(old))
}

// wholePath returns the range file of the whole object at filePath, of size bytes.
func (d *DiskCache) wholePath(filePath string, size int64) string {
	return d.pathForIOEntry(filePath, IOEntry{Size: size})
}

// objectFiles returns the range files on disk of the object at filePath.
func (d *DiskCache) objectFiles(filePath string) []string {
	filePath = pathpkg.Clean(filePath)
	files, _ := filepath.Glob(filepath.Join(d.path, diskCacheShard(filePath), "[0-9]*-[0-9]*"+toOSPath(filePath)+cacheFileSuffix))
	return files
}

// fileDownload returns the download of the file fill of the disk cache holding the bytes
// [lo, hi) of the object at objectPath, and where its bytes start in the object.
func (d *DiskCache) fileDownload(objectPath string, lo, hi int64) (*objectDownload, int64) {
	p, o := d.fileFill(objectPath, false, lo, hi)
	if p == nil {
		return nil, 0
	}
	return o, p.start
}

// The size of a range file is the prefix of the digits after its offset that equals the
// file's length, also when the object name starts with digits. Only files under the shard
// directory of their object are range files, also those of objects with nested paths.
func TestParseRangeFile(t *testing.T) {
	root := "cache"
	in := func(object, name string) string {
		return filepath.Join(root, diskCacheShard(object), name+cacheFileSuffix)
	}
	require.NotEqual(t, diskCacheShard("123obj"), diskCacheShard("23obj"))

	key, offset, size, ok := parseRangeFile(root, in("123obj", "10-100123obj"), 100)
	require.True(t, ok)
	require.Equal(t, filepath.Join(root, diskCacheShard("123obj"), "123obj"+cacheFileSuffix), key)
	require.Equal(t, int64(10), offset)
	require.Equal(t, int64(100), size)

	key, _, size, ok = parseRangeFile(root, in("23obj", "10-100123obj"), 1001)
	require.True(t, ok)
	require.Equal(t, filepath.Join(root, diskCacheShard("23obj"), "23obj"+cacheFileSuffix), key)
	require.Equal(t, int64(1001), size)

	key, offset, size, ok = parseRangeFile(root, filepath.Join(root, diskCacheShard("dir/obj"), "10-7dir", "obj"+cacheFileSuffix), 7)
	require.True(t, ok)
	require.Equal(t, filepath.Join(root, diskCacheShard("dir/obj"), "dir", "obj"+cacheFileSuffix), key)
	require.Equal(t, int64(10), offset)
	require.Equal(t, int64(7), size)

	for name, path := range map[string]string{
		"no size prefix equals the length": in("123obj", "10-100123obj"),
		"another object's shard":           in("23obj", "10-7obj"),
		"whole file":                       in("full123obj", "full123obj"),
		"no object name":                   in("", "10-7"),
		"no offset":                        in("obj", "x-7obj"),
		"temporary file":                   filepath.Join(root, diskCacheShard("obj"), "10-7obj.tmp"),
		"nested path in another shard":     filepath.Join(root, diskCacheShard("obj"), "10-7dir", "obj"+cacheFileSuffix),
		"nested whole file":                filepath.Join(root, diskCacheShard("dir/obj"), "fulldir", "obj"+cacheFileSuffix),
		"outside a shard":                  filepath.Join(root, "10-7obj"+cacheFileSuffix),
	} {
		_, _, _, ok := parseRangeFile(root, path, 7)
		require.False(t, ok, name)
	}
}

func rangeTestData(n int) []byte {
	data := make([]byte, n)
	for i := range data {
		data[i] = byte(i*7 + 3)
	}
	return data
}

func readEntry(t *testing.T, cache *DiskCache, offset, size int64) IOEntry {
	vec := &IOVector{FilePath: "123obj", Entries: []IOEntry{{Offset: offset, Size: size}}}
	require.NoError(t, cache.Read(context.Background(), vec))
	return vec.Entries[0]
}

// A read inside a cached range is served from it; a read beyond it is not, nor is one
// after the range left the cache. A restarted cache finds the range again.
func TestDiskCacheReadWithinCachedRange(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	data := rangeTestData(200)
	require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "123obj", Entries: []IOEntry{
		{Offset: 100, Size: 100, Data: data[100:]},
		{Offset: 120, Size: 50, Data: data[120:170]},
	}}, false))

	e := readEntry(t, cache, 130, 30)
	require.True(t, e.done)
	require.Same(t, cache, e.fromCache)
	require.Equal(t, data[130:160], e.Data)
	e = readEntry(t, cache, 100, 100)
	require.True(t, e.done, "the exact range")
	require.Equal(t, data[100:], e.Data)
	require.False(t, readEntry(t, cache, 90, 20).done, "starts before the range")
	require.False(t, readEntry(t, cache, 190, 20).done, "ends after the range")

	cache.Close(ctx)
	cache, err = NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	e = readEntry(t, cache, 101, 10)
	require.True(t, e.done, "after restart")
	require.Equal(t, data[101:111], e.Data)

	cache.cache.Delete(ctx, cache.pathForIOEntry("123obj", IOEntry{Offset: 100, Size: 100}))
	e = readEntry(t, cache, 101, 10)
	require.False(t, e.done, "the range left the cache")
	e = readEntry(t, cache, 125, 10)
	require.True(t, e.done, "the other range still contains it")
	require.Equal(t, data[125:135], e.Data)
}

// A read passes over an indexed range file the cache no longer holds, forgets it, and is
// served by the next range that contains it.
func TestDiskCacheReadPassesOverStaleRange(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	data := rangeTestData(200)
	require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "123obj", Entries: []IOEntry{
		{Offset: 120, Size: 50, Data: data[120:170]},
	}}, false))
	stale := cache.pathForIOEntry("123obj", IOEntry{Offset: 125, Size: 10})
	require.NoError(t, os.WriteFile(stale, make([]byte, 10), 0644))
	cache.ranges.add(cache.path, stale)

	e := readEntry(t, cache, 126, 4)
	require.True(t, e.done)
	require.Equal(t, data[126:130], e.Data)
	cache.ranges.mu.Lock()
	_, indexed := cache.ranges.objectOf[stale]
	cache.ranges.mu.Unlock()
	require.False(t, indexed)
}

// covering and containing on the sorted snapshots return what a scan of every range of
// the object returns, as ranges are added and removed.
func TestCachedRangesMatchFullScan(t *testing.T) {
	scanCovering := func(ranges []cachedRange, offset, size int64) ([]cachedRange, bool) {
		var out []cachedRange
		for pos, end := offset, offset+size; pos < end; {
			var best cachedRange
			ok := false
			for _, r := range ranges {
				if r.offset <= pos && pos < r.offset+r.size && (!ok || r.offset+r.size > best.offset+best.size) {
					best, ok = r, true
				}
			}
			if !ok {
				return nil, false
			}
			out = append(out, best)
			pos = best.offset + best.size
		}
		return out, true
	}
	ends := func(rs []cachedRange) []int64 {
		out := make([]int64, 0, len(rs))
		for _, r := range rs {
			out = append(out, r.offset+r.size)
		}
		return out
	}
	rnd := mrand.New(mrand.NewSource(1))
	for round := 0; round < 50; round++ {
		var c cachedRanges
		var all []cachedRange
		for i := 0; i < 1+rnd.Intn(40); i++ {
			r := cachedRange{diskPath: strconv.Itoa(i), offset: int64(rnd.Intn(1000)), size: int64(1 + rnd.Intn(200))}
			all = append(all, r)
			c.mu.Lock()
			if c.objectOf == nil {
				c.objectOf = make(map[string]string)
			}
			c.objectOf[r.diskPath] = "k"
			c.byObject.Store("k", c.snapshot("k").with(r))
			c.mu.Unlock()
		}
		check := func() {
			for q := 0; q < 200; q++ {
				offset, size := int64(rnd.Intn(1200)), int64(1+rnd.Intn(300))
				got, ok := c.covering("k", offset, size)
				want, wantOK := scanCovering(all, offset, size)
				require.Equal(t, wantOK, ok)
				// Ties between ranges reaching as far may pick either; the ends are the same.
				require.Equal(t, ends(want), ends(got))
				gotC, okC := c.containing("k", offset, size, nil)
				wantC, wantOKC := smallestContaining(all, offset, size, nil)
				require.Equal(t, wantOKC, okC)
				require.Equal(t, wantC.size, gotC.size)
			}
		}
		check()
		for len(all) > 0 && rnd.Intn(3) > 0 {
			i := rnd.Intn(len(all))
			c.remove(all[i].diskPath)
			all = slices.Delete(all, i, i+1)
			check()
		}
	}
}

// heldStorage reads the first read's bytes at once and returns them when released.
type heldStorage struct {
	ObjectStorage
	once     sync.Once
	read     chan struct{}
	released chan struct{}
}

func (h *heldStorage) Read(ctx context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	r, err := h.ObjectStorage.Read(ctx, key, min, max)
	if err != nil {
		return nil, err
	}
	held := false
	h.once.Do(func() { held = true })
	if !held {
		return r, nil
	}
	defer r.Close()
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	close(h.read)
	select {
	case <-h.released:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return io.NopCloser(bytes.NewReader(data)), nil
}

// A read that skips disk cache reads is not served from cached prefetch ranges.
func TestPrefetchRangeSkipsDiskCacheReads(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 4096, 64<<10)
	require.Equal(t, data[5000:5002], readRangeOf(t, fs, 0, 5000, 2))
	require.Equal(t, int64(1), storage.reads.Load())
	require.Equal(t, data[5010:5012], readRangeOf(t, fs, SkipDiskCacheReads, 5010, 2))
	require.Equal(t, int64(2), storage.reads.Load(), "read from object storage")
}

// A probe of a few columns reads exactly their entries, never the whole object, even when
// the entries lie far apart; policy 0 downloads the whole object.
func TestExactColumnsReadNeverDownloadsTheObject(t *testing.T) {
	for _, policy := range []Policy{ExactColumnsRead, 0} {
		fs, storage, data := newPrefetchRangeFS(t, 4096, 300<<10)
		vec := &IOVector{FilePath: "object", Policy: policy | SkipMemoryCache,
			Entries: []IOEntry{{Offset: 100, Size: 10}, {Offset: 200 << 10, Size: 10}}}
		require.NoError(t, fs.Read(context.Background(), vec))
		flushPublishers(fs)
		require.Equal(t, data[100:110], vec.Entries[0].Data)
		require.Equal(t, data[200<<10:200<<10+10], vec.Entries[1].Data)
		vec.Release()
		if policy == ExactColumnsRead {
			require.Equal(t, int64(2), storage.reads.Load())
			require.Equal(t, int64(20), storage.bytes.Load(), "only the entries")
		} else {
			require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))), "policy 0 caches the whole object")
		}
	}
}

// An entry read before a delete is never found by a read after the delete returns: a
// delete during its publication waits for the publication to end and removes the entry.
func TestDeleteWaitsForPublicationOfEarlierRead(t *testing.T) {
	for _, async := range []bool{false, true} {
		for _, claimed := range []bool{false, true} {
			fs, _, data := newPrefetchRangeFS(t, 64<<10, 300<<10)
			d := fs.diskCache
			ctx := context.Background()
			key := d.objectKey("object")
			entry := IOEntry{Offset: 100, Size: 10, Data: slices.Clone(data[100:110]), readDeletes: d.readDeletes()}

			deleted := make(chan struct{})
			var deleteErr error
			deleteObject := func() { defer close(deleted); deleteErr = d.DeletePaths(ctx, []string{"object"}) }
			var published, seenAfterDelete atomic.Bool
			if claimed {
				d.hooks.afterPublicationClaim = func() {
					go deleteObject()
					select {
					case <-deleted:
					case <-time.After(100 * time.Millisecond):
					}
				}
				d.hooks.afterFilePublication = func() {
					published.Store(true)
					select {
					case <-deleted:
						if _, cached := d.ranges.covering(key, 100, 10); cached {
							seenAfterDelete.Store(true)
						}
					default:
					}
				}
			} else {
				d.hooks.beforeFilePublication = func() {
					counted := d.deletes.now()
					go deleteObject()
					for d.deletes.now() == counted {
						time.Sleep(time.Millisecond)
					}
				}
			}
			require.NoError(t, d.Update(ctx, &IOVector{FilePath: "object", Entries: []IOEntry{entry}}, async))
			d.Flush(ctx)
			<-deleted
			require.NoError(t, deleteErr)

			if claimed {
				require.True(t, published.Load(), "async %v", async)
			}
			require.False(t, seenAfterDelete.Load(), "a read after the delete found the entry, async %v", async)
			_, cached := d.ranges.covering(key, 100, 10)
			require.False(t, cached, "async %v claimed %v", async, claimed)
		}
	}
}

func readTestEntry(t *testing.T, ctx context.Context, cache *DiskCache, path string, size int64) ([]byte, bool) {
	t.Helper()
	vec := &IOVector{FilePath: path, Entries: []IOEntry{{Offset: 0, Size: size}}}
	require.NoError(t, cache.Read(ctx, vec))
	defer vec.Release()
	if !vec.Entries[0].done {
		return nil, false
	}
	return append([]byte(nil), vec.Entries[0].Data...), true
}

// A delete of a cached object drops and closes the files kept open for it; the cache's
// close closes the others. Reads of an entry at once share its kept file and read its
// bytes while its object is deleted and cached again.
func TestDiskCacheKeptFileClosedOnDeleteAndClose(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	for _, p := range []string{"foo", "bar"} {
		require.NoError(t, cache.Update(ctx, &IOVector{FilePath: p, Entries: []IOEntry{{Offset: 0, Size: 3, Data: []byte(p)}}}, false))
		_, ok := readTestEntry(t, ctx, cache, p, 3)
		require.True(t, ok)
	}
	keptOf := func(p string) *openCacheFile {
		cache.files.mu.Lock()
		defer cache.files.mu.Unlock()
		return cache.files.m[cache.pathForIOEntry(p, IOEntry{Offset: 0, Size: 3})]
	}
	foo, bar := keptOf("foo"), keptOf("bar")
	require.NotNil(t, foo)
	require.NotNil(t, bar)

	require.NoError(t, cache.DeletePaths(ctx, []string{"foo"}))
	require.Nil(t, keptOf("foo"))
	_, err = foo.file.Stat()
	require.ErrorIs(t, err, os.ErrClosed)
	_, ok := readTestEntry(t, ctx, cache, "foo", 3)
	require.False(t, ok, "a deleted entry is not read through its kept file")

	data := []byte("0123456789abcdef")
	update := func() {
		require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "baz", Entries: []IOEntry{{Offset: 0, Size: 16, Data: data}}}, false))
	}
	update()
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				vec := &IOVector{FilePath: "baz", Entries: []IOEntry{{Offset: 0, Size: 16}}}
				if err := cache.Read(ctx, vec); err != nil {
					t.Errorf("read: %v", err)
					return
				}
				if vec.Entries[0].done && !bytes.Equal(vec.Entries[0].Data, data) {
					t.Errorf("read %q, want %q", vec.Entries[0].Data, data)
				}
				vec.Release()
			}
		}()
	}
	for i := 0; i < 50; i++ {
		require.NoError(t, cache.DeletePaths(ctx, []string{"baz"}))
		update()
	}
	close(stop)
	wg.Wait()

	cache.Close(ctx)
	_, err = bar.file.Stat()
	require.ErrorIs(t, err, os.ErrClosed)
}

// Past the most files kept, the least recently taken file no read holds is closed for a
// new one; with every file held, a new file is not kept. A file opened before a change of
// its path is not kept. disk-open-files-max bounds the files kept open by the disk cache
// and by the index-file cache, DefaultDiskOpenFilesMax when unset; 0 keeps no file open
// and notes no path missing.
func TestOpenFilePoolLimit(t *testing.T) {
	dir := t.TempDir()
	open := func(name string) *os.File {
		f, err := os.Create(filepath.Join(dir, name))
		require.NoError(t, err)
		return f
	}
	closed := func(f *os.File) bool {
		_, err := f.Stat()
		return errors.Is(err, os.ErrClosed)
	}
	o := openFilePool{limit: 2}
	a := o.keep("a", o.seq("a"), open("a"))
	b := o.keep("b", o.seq("b"), open("b"))
	o.release(a)
	o.release(b)
	require.NotNil(t, o.take("a"))
	o.release(a)

	c := o.keep("c", o.seq("c"), open("c"))
	require.NotNil(t, c)
	require.True(t, closed(b.file), "the least recently taken file is closed")
	require.Nil(t, o.m["b"])

	require.NotNil(t, o.take("a"))
	d := open("d")
	defer d.Close()
	require.Nil(t, o.keep("d", o.seq("d"), d), "with every file held, none is kept")

	seq := o.seq("e")
	o.changed("e")
	e := open("e")
	defer e.Close()
	o.release(c)
	require.Nil(t, o.keep("e", seq, e), "a file opened before its path changed is not kept")

	// A dropped file a read holds is closed when the read releases it.
	o.closeAll()
	require.True(t, closed(c.file))
	require.False(t, closed(a.file))
	o.release(a)
	require.True(t, closed(a.file))

	for _, limit := range []int{3, 0} {
		config := indexCacheConfig(t, 64<<20, 64<<20)
		require.Equal(t, DefaultDiskOpenFilesMax, config.diskOpenFilesMax())
		config.DiskOpenFilesMax = &limit
		caches := newTestFileServiceCaches(t, config)
		if limit != 0 {
			require.Equal(t, limit, caches.disk.files.limit)
			require.Equal(t, limit, caches.index.disk.files.limit)
			continue
		}
		cache := caches.disk
		ctx := context.Background()
		_, ok := readTestEntry(t, ctx, cache, "foo", 3)
		require.False(t, ok)
		require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "foo", Entries: []IOEntry{{Offset: 0, Size: 3, Data: []byte("abc")}}}, false))
		got, ok := readTestEntry(t, ctx, cache, "foo", 3)
		require.True(t, ok)
		require.Equal(t, []byte("abc"), got)
		cache.files.mu.Lock()
		require.Empty(t, cache.files.m)
		require.Empty(t, cache.files.missing)
		cache.files.mu.Unlock()
	}
}

// A read that finds no file at a path notes it missing and does not open it again; a write
// of the path forgets that, so the next read finds the file.
func TestDiskCacheMissingPathForgottenOnWrite(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	diskPath := cache.pathForIOEntry("foo", IOEntry{Offset: 0, Size: 3})
	_, ok := readTestEntry(t, ctx, cache, "foo", 3)
	require.False(t, ok)
	cache.files.mu.Lock()
	require.Contains(t, cache.files.missing, diskPath)
	cache.files.mu.Unlock()

	require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "foo", Entries: []IOEntry{{Offset: 0, Size: 3, Data: []byte("abc")}}}, false))
	got, ok := readTestEntry(t, ctx, cache, "foo", 3)
	require.True(t, ok)
	require.Equal(t, []byte("abc"), got)
}

func TestColumnReadPropagatesJoinedPrefetchSourceError(t *testing.T) {
	for _, tc := range []struct {
		name       string
		fileBacked bool
	}{
		{name: "memory_prefetch"},
		{name: "file_prefetch", fileBacked: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			t.Cleanup(cancel)
			fs, storage, _ := newPrefetchRangeFS(t, 0, 300<<10)
			<-fs.diskCache.load.done
			r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
			if tc.fileBacked {
				fs.diskCache.memoryPrefetchMax = r.Size - 1
			}
			s := &failOnceStorage{ObjectStorage: storage, fail: r.Offset, gate: make(chan struct{})}
			fs.server.raw = s
			release := sync.OnceFunc(func() { close(s.gate) })
			t.Cleanup(release)
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r},
			}))
			require.Eventually(t, s.failed.Load, 5*time.Second, time.Millisecond, "prefetch entered storage")

			finished := make(chan struct{})
			defer func() {
				release()
				cancel()
				<-finished
			}()
			result := make(chan error, 1)
			go func() {
				defer close(finished)
				_, err := readColumnRange(ctx, fs, r)
				result <- err
			}()
			diskPath := fs.diskCache.objectKey("object")
			require.Eventually(t, func() bool {
				if !tc.fileBacked {
					return memoryWaiters(fs, diskPath) == 1
				}
				o, _ := fs.diskCache.fileDownload(diskPath, r.Offset, r.Offset+r.Size)
				if o == nil {
					return false
				}
				o.mu.Lock()
				defer o.mu.Unlock()
				return len(o.waiters) == 1
			}, 5*time.Second, time.Millisecond, "the foreground read joined the stalled prefetch")
			release()
			err := <-result
			waitPrefetcherIdle(t, fs)
			require.ErrorContains(t, err, "probe: storage failed",
				"the joined read should receive the source error (successful fallback opens: %d)", storage.reads.Load())
			require.Zero(t, storage.reads.Load())
		})
	}
}

func TestIndexObjectReadRejectsEmptyRange(t *testing.T) {
	for _, name := range []string{"index_cache_miss", "index_cache_hit", "disk_cache"} {
		cached := name == "index_cache_hit"
		t.Run(name, func(t *testing.T) {
			if name == "disk_cache" {
				fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
				<-fs.diskCache.load.done
				vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: 10}}}
				defer vec.Release()
				err := fs.diskCache.Read(context.Background(), vec)
				require.True(t, moerr.IsMoErrCode(err, moerr.ErrEmptyRange), "got %v", err)
				return
			}
			fs, storage, data := newIndexObjectFS(t, true)
			if cached {
				cacheIndexObject(t, fs, "object", data)
				require.Equal(t, data[100:110], readIndexObjectRange(t, fs, IndexFileObject, 100, 10))
				require.Zero(t, storage.reads.Load())
			}
			vector := &IOVector{
				FilePath: "object",
				Policy:   IndexFileObject | SkipMemoryCache | SkipFullFilePreloads,
				Entries:  []IOEntry{{Offset: 100, Size: 0}},
			}
			defer vector.Release()
			err := fs.Read(context.Background(), vector)
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrEmptyRange), "expected ErrEmptyRange, got %v", err)
		})
	}
}

func TestRangeIndexSurvivesOlderEviction(t *testing.T) {
	for _, tc := range []struct {
		name               string
		readDuringEviction bool
	}{
		{name: "evict_without_read"},
		{name: "evict_with_read", readDuringEviction: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
			d := fs.diskCache
			<-d.load.done
			r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r},
			}))
			waitPrefetcherIdle(t, fs)
			path := d.pathForIOEntry("object", r.entry())
			key := d.objectKey("object")
			require.True(t, d.cache.Contains(path))
			require.FileExists(t, path)

			d.cache.ForceEvict(ctx, d.cache.Used())
			if tc.readDuringEviction {
				vector := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads, Entries: []IOEntry{r.entry()}}
				require.NoError(t, d.Read(ctx, vector))
				if vector.Entries[0].done {
					require.Equal(t, data[r.Offset:r.Offset+r.Size], vector.Entries[0].Data)
				}
				vector.Release()
			}
			d.Flush(ctx)
			if !d.cache.Contains(path) {
				// A completed eviction need not keep the range. Refill it to
				// establish the same warm postcondition as a successful reindex.
				require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
					FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r},
				}))
				waitPrefetcherIdle(t, fs)
			}
			require.True(t, d.cache.Contains(path))
			require.FileExists(t, path)
			want := PrefetchRange{Offset: r.Offset + 1, Size: r.Size - 2}
			_, indexed := d.ranges.containing(key, want.Offset, want.Size, nil)
			before := storage.reads.Load()
			got, err := readColumnRange(ctx, fs, want)
			require.NoError(t, err)
			require.Equal(t, data[want.Offset:want.Offset+want.Size], got)
			require.Equal(t, before, storage.reads.Load(),
				"a warm subrange should not open storage (range indexed before read: %t)", indexed)
		})
	}
}

// A read of a known size is served from a warm index-file entry. A read to the end is read
// from storage, so a truncated entry file cannot shorten it; the entries after it are
// still served from the index-file cache.
func TestIndexObjectReadToEndReadsWholeTail(t *testing.T) {
	for _, tc := range []struct {
		name      string
		readToEnd bool
		truncate  bool
		mixed     bool
	}{
		{name: "known_size"},
		{name: "read_to_end", readToEnd: true},
		{name: "read_to_end_truncated_entry", readToEnd: true, truncate: true},
		{name: "read_to_end_then_known_size", mixed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			if tc.mixed {
				vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
					Entries: []IOEntry{{Offset: 0, Size: -1}, {Offset: 100, Size: 10}}}
				defer vec.Release()
				require.NoError(t, fs.readIndexObject(context.Background(), vec))
				require.False(t, vec.Entries[0].done)
				require.True(t, vec.Entries[1].done)
				require.Equal(t, data[100:110], vec.Entries[1].Data)
				return
			}
			if tc.truncate {
				diskPath, ok := fs.indexObjectCached("object")
				require.True(t, ok)
				require.NoError(t, os.Truncate(filepath.Join(diskPath, IndexObjectFile), 1024))
			}
			offset := int64(1000)
			size := int64(len(data)) - offset
			if tc.readToEnd {
				size = -1
			}
			got := readIndexObjectRange(t, fs, IndexFileObject|SkipFullFilePreloads, offset, size)
			require.Equal(t, data[offset:], got, "the read returns the whole object tail")
			if !tc.readToEnd {
				require.Zero(t, storage.reads.Load(), "a warm index-file entry serves the read")
			}
		})
	}
}

func TestIndexObjectReadRefreshesIdleTTL(t *testing.T) {
	for _, tc := range []struct {
		name     string
		sinceUse time.Duration
	}{
		{name: "outside_touch_interval", sinceUse: 2 * indexObjectTouch},
		{name: "inside_touch_interval", sinceUse: time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			require.Equal(t, data[10:20], readIndexObjectRange(t, fs, IndexFileObject, 10, 10))
			path, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			c := fs.indexFileCache
			c.idleTTL = 2 * time.Second

			// Age the previous use in both places without waiting for the touch interval.
			lastRead := time.Now().Add(-tc.sinceUse)
			require.NoError(t, os.Chtimes(path, lastRead, lastRead))
			fs.indexObjects.mu.Lock()
			fs.indexObjects.touched[path] = lastRead
			fs.indexObjects.mu.Unlock()

			require.Equal(t, data[20:30], readIndexObjectRange(t, fs, IndexFileObject, 20, 10))
			require.Zero(t, storage.reads.Load())
			removed, err := c.collectIdle(ctx, time.Now().Add(1500*time.Millisecond))
			require.NoError(t, err)
			require.Zero(t, removed, "idle collection should keep the index object that was just read")
		})
	}
}

// A read that streams an entry gets its reader while the body is still being read.
func TestReadCloserReturnsBeforeReadingBody(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
	}{
		{name: "bypass_disk_reads", policy: SkipDiskCacheReads},
		{name: "eligible_for_range_sharing"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newPrefetchRangeFS(t, 0, 1024)
			// A streamed entry past the bytes a prefetch object buffers opens its own reader.
			fs.diskCache.memoryPrefetchMax = 32
			<-fs.diskCache.load.done
			storage := &stallingStorage{ObjectStorage: fs.server.raw, stalled: make(chan struct{}), release: make(chan struct{})}
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.release) })
			readDone := make(chan struct{})
			defer func() {
				unblock()
				<-readDone
			}()
			type result struct {
				reader io.ReadCloser
				err    error
			}
			results := make(chan result, 1)
			go func() {
				defer close(readDone)
				var reader io.ReadCloser
				vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | tc.policy,
					Entries: []IOEntry{{Size: 64, ReadCloserForRead: &reader}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				results <- result{reader: reader, err: err}
			}()

			var got result
			select {
			case got = <-results:
			case <-ctx.Done():
				t.Fatal("Read waited for the remote body before handing the ReadCloser to the caller")
			}
			require.NoError(t, got.err)
			require.NotNil(t, got.reader)
			defer got.reader.Close()
			unblock()
			content, err := io.ReadAll(got.reader)
			require.NoError(t, err)
			require.Equal(t, data[:64], content)
		})
	}
}

// A Delete waits for no publication I/O of a put, only for its publishing rename.
func TestIndexFileCacheDeleteReturnsWhilePutIsStalled(t *testing.T) {
	for _, protectedPut := range []bool{false, true} {
		name := "ordinary_put"
		if protectedPut {
			name = "put_with_delete_mark"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			fs, _, data := newIndexObjectFS(t, true)
			cache := fs.indexFileCache
			blocked := make(chan struct{})
			resume := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(resume) })
			defer unblock()
			cache.disk.hooks.beforeFilePublication = func() {
				close(blocked)
				<-resume
			}
			putDone := make(chan error, 1)
			go func() {
				var err error
				if protectedPut {
					_, err = cache.PutUnlessDeleted(ctx, "entry", cache.DeleteMark(), recoveryFiles(data[:64]))
				} else {
					_, err = cache.Put(ctx, "entry", recoveryFiles(data[:64]))
				}
				putDone <- err
			}()
			defer func() {
				unblock()
				require.NoError(t, <-putDone)
			}()
			select {
			case <-blocked:
			case <-time.After(5 * time.Second):
				t.Fatal("put never reached publication")
			}

			deleteCtx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
			defer cancel()
			deleted := make(chan error, 1)
			go func() { deleted <- cache.Delete(deleteCtx, "entry") }()
			select {
			case err := <-deleted:
				require.ErrorIs(t, err, context.DeadlineExceeded)
			case <-time.After(time.Second):
				unblock()
				require.ErrorIs(t, <-deleted, context.DeadlineExceeded)
				t.Fatal("Delete stayed blocked past its deadline until the put resumed")
			}
		})
	}
}

// A delete is seen by the reads begun before it and only for its object; once the
// remembered deletes are forgotten, every read begun before then counts as deleted.
func TestObjectDeletesSince(t *testing.T) {
	var o objectDeletes
	start := o.now()
	require.False(t, o.since("a", start))
	o.deleted("a")
	require.True(t, o.since("a", start))
	require.False(t, o.since("b", start))
	require.False(t, o.since("a", o.now()))

	before := o.now()
	for i := range objectDeletesMax {
		o.deleted(strconv.Itoa(i))
	}
	o.deleted("c")
	require.True(t, o.since("b", before), "the deletes since are forgotten")
	require.False(t, o.since("b", o.now()))
}
