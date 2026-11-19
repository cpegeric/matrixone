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
	"fmt"
	"io"
	"os"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/stretchr/testify/require"
)

func readMetaEntry(t *testing.T, cache *DiskCache, policy Policy, offset, size int64) IOEntry {
	vec := &IOVector{FilePath: "123obj", Policy: policy, Entries: []IOEntry{{Offset: offset, Size: size}}}
	require.NoError(t, cache.Read(context.Background(), vec))
	return vec.Entries[0]
}

func waitDiskCacheLoaded(_ *testing.T, cache *DiskCache) {
	<-cache.load.done
}

// The entry of a meta read is kept in the meta layer, where reads of either kind find it,
// a restart keeps it, and a delete of its object removes it.
func TestDiskCacheMetaLayer(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	cache, err := NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	data := rangeTestData(200)
	require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "123obj", Policy: ObjectMetaRead,
		Entries: []IOEntry{{Offset: 100, Size: 100, Data: data[100:]}}}, false))
	metaPath := cache.metaPathForIOEntry("123obj", IOEntry{Offset: 100, Size: 100})
	_, err = os.Stat(metaPath)
	require.NoError(t, err)
	_, err = os.Stat(cache.pathForIOEntry("123obj", IOEntry{Offset: 100, Size: 100}))
	require.True(t, os.IsNotExist(err), "nothing in the data layer")
	require.NotZero(t, cache.cache.ClassUsed(metaClass))

	e := readMetaEntry(t, cache, ObjectMetaRead, 100, 100)
	require.True(t, e.done)
	require.Equal(t, data[100:], e.Data)
	e = readMetaEntry(t, cache, 0, 120, 30)
	require.True(t, e.done, "a data read finds the meta file")
	require.Equal(t, data[120:150], e.Data)

	cache.Close(ctx)
	cache, err = NewDiskCache(ctx, dir, fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	waitDiskCacheLoaded(t, cache)
	require.NotZero(t, cache.cache.ClassUsed(metaClass), "a restart keeps the file in the meta layer")
	e = readMetaEntry(t, cache, ObjectMetaRead, 110, 10)
	require.True(t, e.done)
	require.Equal(t, data[110:120], e.Data)

	require.NoError(t, cache.DeletePaths(ctx, []string{"123obj"}))
	_, err = os.Stat(metaPath)
	require.True(t, os.IsNotExist(err), "a delete of the object removes its meta file")
	require.Zero(t, cache.cache.ClassUsed(metaClass))
}

// A meta read of an object cached whole is served from the whole file and writes nothing.
func TestDiskCacheMetaReadOfWholeObject(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	data := rangeTestData(200)
	written, err := cache.writeFile(ctx, cache.pathForFile("123obj"), func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(data)), nil
	})
	require.NoError(t, err)
	require.True(t, written)
	vec := &IOVector{FilePath: "123obj", Policy: ObjectMetaRead, Entries: []IOEntry{{Offset: 150, Size: 50}}}
	require.NoError(t, cache.Read(ctx, vec))
	require.True(t, vec.Entries[0].done)
	require.Equal(t, data[150:], vec.Entries[0].Data)
	require.NoError(t, cache.Update(ctx, vec, false))
	require.Zero(t, cache.cache.ClassUsed(metaClass))
}

// Data files are evicted before a meta file idle less than R times as long as the oldest
// data file, and an idle meta file ages out.
func TestDiskCacheMetaLayerEviction(t *testing.T) {
	ctx := context.Background()
	// latency × bandwidth of 100 KiB: a meta file of a block stays 26 times longer idle.
	costs := rangeReadCosts.get()
	rangeReadCosts.set(0.001, 100<<20)
	t.Cleanup(func() { rangeReadCosts.set(costs.latency, costs.bytesPerS) })
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	waitDiskCacheLoaded(t, cache)
	meta := rangeTestData(100)
	require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "123obj", Policy: ObjectMetaRead,
		Entries: []IOEntry{{Offset: 0, Size: 100, Data: meta}}}, false))
	metaPath := cache.metaPathForIOEntry("123obj", IOEntry{Offset: 0, Size: 100})
	data := rangeTestData(256 << 10)
	update := func(i int) {
		require.NoError(t, cache.Update(ctx, &IOVector{FilePath: fmt.Sprintf("%dobj", i),
			Entries: []IOEntry{{Offset: 0, Size: int64(len(data)), Data: data}}}, false))
	}
	for i := range 20 {
		update(i)
	}
	require.True(t, cache.cache.Contains(metaPath), "data is evicted first")
	require.LessOrEqual(t, cache.cache.Used(), int64(1<<20))
	for i := 20; i < 500; i++ {
		update(i)
	}
	require.False(t, cache.cache.Contains(metaPath), "an idle meta file ages out")
}

// A meta file whose uses a cache above serves stays while data files churn, when those
// uses are told to the disk cache.
func TestDiskCacheMetaTouchKeepsMetaFile(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	defer cache.Close(ctx)
	waitDiskCacheLoaded(t, cache)
	require.NoError(t, cache.Update(ctx, &IOVector{FilePath: "123obj", Policy: ObjectMetaRead,
		Entries: []IOEntry{{Offset: 0, Size: 100, Data: rangeTestData(100)}}}, false))
	metaPath := cache.metaPathForIOEntry("123obj", IOEntry{Offset: 0, Size: 100})
	data := rangeTestData(256 << 10)
	for i := range 500 {
		if i%20 == 0 {
			cache.touch(metaPath)
		}
		require.NoError(t, cache.Update(ctx, &IOVector{FilePath: fmt.Sprintf("%dobj", i),
			Entries: []IOEntry{{Offset: 0, Size: int64(len(data)), Data: data}}}, false))
	}
	require.True(t, cache.cache.Contains(metaPath))
}

type metaTouchRecorder struct {
	FileService
	touched []string
}

func (r *metaTouchRecorder) TouchMeta(filePath string, offset, size int64) {
	r.touched = append(r.touched, fmt.Sprintf("%s@%d+%d", filePath, offset, size))
}

// A use told to a FileServices or a sub path reaches the file service under it.
func TestTouchMetaReachesUpstream(t *testing.T) {
	ctx := context.Background()
	mem, err := NewMemoryFS("shared", DisabledCacheConfig, nil)
	require.NoError(t, err)
	defer mem.Close(ctx)
	r := &metaTouchRecorder{FileService: mem}
	services, err := NewFileServices("shared", r)
	require.NoError(t, err)
	TouchMeta(services, "obj", 1, 2)
	TouchMeta(SubPath(r, "sub"), "obj", 3, 4)
	TouchMeta(mem, "obj", 5, 6)
	require.Equal(t, []string{"obj@1+2", "shared:sub/obj@3+4"}, r.touched)
}
