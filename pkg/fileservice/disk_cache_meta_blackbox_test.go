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
	"io/fs"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/util/toml"
	"github.com/stretchr/testify/require"
)

// The tests below check the contracts of docs/design/20261119-disk-cache-meta-layer.md
// through the file service: reads, writes, deletes and restarts, counting the reads of
// object storage.

const (
	metaReadPolicy = ObjectMetaRead | SkipFullFilePreloads | ExactRead | SkipMemoryCache
	dataReadPolicy = SkipFullFilePreloads | ExactRead | SkipMemoryCache
)

type metaLayerFS struct {
	*S3FS
	storage *countingStorage
	disk    string
}

// newMetaLayerFS opens a file service on bucket with a disk cache of capacity bytes at disk.
func newMetaLayerFS(t *testing.T, bucket, disk string, capacity int64) *metaLayerFS {
	ctx := context.Background()
	s, err := NewS3FS(ctx, ObjectStorageArguments{Name: "s3", Endpoint: "disk", Bucket: bucket}, CacheConfig{
		DiskPath: ptrTo(disk), DiskCapacity: ptrTo(toml.ByteSize(capacity)),
		MemoryCapacity: ptrTo[toml.ByteSize](0),
	}, nil, false, false)
	require.NoError(t, err)
	s.SetAsyncUpdate(false)
	<-s.diskCache.load.done
	storage := &countingStorage{ObjectStorage: s.storage}
	s.storage = storage
	return &metaLayerFS{S3FS: s, storage: storage, disk: disk}
}

func (m *metaLayerFS) write(t *testing.T, name string, data []byte, policy Policy) {
	require.NoError(t, m.Write(context.Background(), IOVector{FilePath: name, Policy: policy,
		Entries: []IOEntry{{Size: int64(len(data)), Data: data}}}))
}

// read reads size bytes of name at offset under policy, and returns them and the reads of
// object storage it made.
func (m *metaLayerFS) read(t *testing.T, name string, policy Policy, offset, size int64) ([]byte, int64) {
	before := m.storage.reads.Load()
	vec := &IOVector{FilePath: name, Policy: policy, Entries: []IOEntry{{Offset: offset, Size: size}}}
	require.NoError(t, m.Read(context.Background(), vec))
	data := slices.Clone(vec.Entries[0].Data)
	vec.Release()
	return data, m.storage.reads.Load() - before
}

// files returns the cache files of the meta layer and of the data layer.
func (m *metaLayerFS) files(t *testing.T) (meta, data int) {
	metaDir := filepath.Join(m.disk, metaCacheDir) + string(filepath.Separator)
	require.NoError(t, filepath.WalkDir(m.disk, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, cacheFileSuffix) {
			return err
		}
		if strings.HasPrefix(path, metaDir) {
			meta++
		} else {
			data++
		}
		return nil
	}))
	return
}

func setRangeReadCosts(t *testing.T, latency, bytesPerS float64) {
	saved := rangeReadCosts.get()
	rangeReadCosts.set(latency, bytesPerS)
	t.Cleanup(func() { rangeReadCosts.set(saved.latency, saved.bytesPerS) })
}

// A meta read caches its bytes in the meta layer, which later meta reads and data reads
// of those bytes are served from.
func TestBlackBoxMetaReadCachedInMetaLayer(t *testing.T) {
	m := newMetaLayerFS(t, t.TempDir(), t.TempDir(), 64<<20)
	defer m.Close(context.Background())
	data := rangeTestData(200 << 10)
	m.write(t, "obj", data, SkipDiskCache|SkipMemoryCache)

	got, reads := m.read(t, "obj", metaReadPolicy, 199<<10, 1<<10)
	require.Equal(t, data[199<<10:], got)
	require.Equal(t, int64(1), reads)
	meta, files := m.files(t)
	require.Equal(t, 1, meta)
	require.Zero(t, files, "nothing in the data layer")

	got, reads = m.read(t, "obj", metaReadPolicy, 199<<10, 1<<10)
	require.Equal(t, data[199<<10:], got)
	require.Zero(t, reads, "a meta read finds the meta layer")
	got, reads = m.read(t, "obj", dataReadPolicy, (199<<10)+10, 100)
	require.Equal(t, data[(199<<10)+10:(199<<10)+110], got)
	require.Zero(t, reads, "a data read finds the meta layer")
}

// A meta read of bytes the data layer holds, in a whole object or a range of data, is
// served from it and writes nothing.
func TestBlackBoxMetaReadServedByDataLayer(t *testing.T) {
	m := newMetaLayerFS(t, t.TempDir(), t.TempDir(), 64<<20)
	defer m.Close(context.Background())
	data := rangeTestData(200 << 10)
	m.write(t, "whole", data, 0)
	m.write(t, "ranged", data, SkipDiskCache|SkipMemoryCache)
	_, reads := m.read(t, "ranged", dataReadPolicy, 190<<10, 10<<10)
	require.Equal(t, int64(1), reads)
	meta, files := m.files(t)
	require.Zero(t, meta)
	require.Equal(t, 2, files)

	for _, name := range []string{"whole", "ranged"} {
		got, reads := m.read(t, name, metaReadPolicy, 199<<10, 1<<10)
		require.Equal(t, data[199<<10:], got, name)
		require.Zero(t, reads, name)
	}
	meta, files = m.files(t)
	require.Zero(t, meta, "a meta read served from the data layer writes nothing")
	require.Equal(t, 2, files)
}

// A restart keeps a meta file in the meta layer: it is served without object storage and
// outlives the data files that churn after it.
func TestBlackBoxMetaLayerSurvivesRestart(t *testing.T) {
	ctx := context.Background()
	bucket, disk := t.TempDir(), t.TempDir()
	m := newMetaLayerFS(t, bucket, disk, 1<<20)
	data := rangeTestData(200 << 10)
	m.write(t, "obj", data, SkipDiskCache|SkipMemoryCache)
	_, reads := m.read(t, "obj", metaReadPolicy, 199<<10, 1<<10)
	require.Equal(t, int64(1), reads)
	m.Close(ctx)

	m = newMetaLayerFS(t, bucket, disk, 1<<20)
	got, reads := m.read(t, "obj", metaReadPolicy, 199<<10, 1<<10)
	require.Equal(t, data[199<<10:], got)
	require.Zero(t, reads)
	m.Close(ctx)

	// No read of the meta file after this restart: the cache has it from its layer alone.
	m = newMetaLayerFS(t, bucket, disk, 1<<20)
	defer m.Close(ctx)
	setRangeReadCosts(t, 0.0002, 100<<20)
	block := rangeTestData(64 << 10)
	for i := range 64 {
		m.write(t, fmt.Sprintf("data%d", i), block, 0)
	}
	meta, _ := m.files(t)
	require.Equal(t, 1, meta, "the data files are evicted before the meta file")
	for i := 64; i < 600; i++ {
		m.write(t, fmt.Sprintf("data%d", i), block, 0)
	}
	meta, _ = m.files(t)
	require.Zero(t, meta, "the meta file counts in the cache and ages out")
}

// A delete of an object removes its files in both layers: an object written again under its
// name is read anew by meta and data reads.
func TestBlackBoxDeleteRemovesBothLayers(t *testing.T) {
	ctx := context.Background()
	m := newMetaLayerFS(t, t.TempDir(), t.TempDir(), 64<<20)
	defer m.Close(ctx)
	old := rangeTestData(200 << 10)
	m.write(t, "obj", old, SkipDiskCache|SkipMemoryCache)
	m.read(t, "obj", metaReadPolicy, 199<<10, 1<<10)
	m.read(t, "obj", dataReadPolicy, 0, 10<<10)
	meta, files := m.files(t)
	require.Equal(t, 1, meta)
	require.Equal(t, 1, files)

	require.NoError(t, m.Delete(ctx, "obj"))
	meta, files = m.files(t)
	require.Zero(t, meta)
	require.Zero(t, files)
	err := m.Read(ctx, &IOVector{FilePath: "obj", Policy: metaReadPolicy, Entries: []IOEntry{{Offset: 0, Size: 1}}})
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "%v", err)

	renewed := make([]byte, len(old))
	for i := range renewed {
		renewed[i] = ^old[i]
	}
	m.write(t, "obj", renewed, SkipDiskCache|SkipMemoryCache)
	got, reads := m.read(t, "obj", metaReadPolicy, 199<<10, 1<<10)
	require.Equal(t, renewed[199<<10:], got)
	require.Equal(t, int64(1), reads)
	got, reads = m.read(t, "obj", dataReadPolicy, 0, 10<<10)
	require.Equal(t, renewed[:10<<10], got)
	require.Equal(t, int64(1), reads)
}

// metaKeptUnderChurn caches a meta file of size bytes, writes count data files of 64 KiB
// into a disk cache of 1 MiB, and reports whether the meta file is kept.
func metaKeptUnderChurn(t *testing.T, latency float64, size int64, count int, touch bool) bool {
	ctx := context.Background()
	m := newMetaLayerFS(t, t.TempDir(), t.TempDir(), 1<<20)
	defer m.Close(ctx)
	data := rangeTestData(200 << 10)
	m.write(t, "obj", data, SkipDiskCache|SkipMemoryCache)
	offset := int64(len(data)) - size
	m.read(t, "obj", metaReadPolicy, offset, size)
	setRangeReadCosts(t, latency, 100<<20)
	block := rangeTestData(64 << 10)
	for i := range count {
		if touch && i%8 == 0 {
			m.TouchMeta("obj", offset, size)
		}
		m.write(t, fmt.Sprintf("data%d", i), block, 0)
	}
	meta, _ := m.files(t)
	return meta == 1
}

// Of the least recently used meta and data files, the meta file goes first only when idle
// more than R times as long, R = 1 + latency × bandwidth / its size: with no latency the
// layers age as one LRU, with latency a small meta file outlives the data files written
// after it, and a large one less long.
func TestBlackBoxMetaEvictionFollowsMissCost(t *testing.T) {
	// The cache holds 16 data files: a meta file is kept while fewer than about 16 × R are
	// written after it.
	require.False(t, metaKeptUnderChurn(t, 0, 1<<10, 64, false), "R = 1: one LRU")
	require.True(t, metaKeptUnderChurn(t, 0.0002, 1<<10, 64, false), "R ≈ 21")
	require.False(t, metaKeptUnderChurn(t, 0.0002, 1<<10, 600, false), "an idle meta file ages out")
	require.False(t, metaKeptUnderChurn(t, 0.0002, 100<<10, 64, false), "R ≈ 1.2 for a large meta file")
}

// A use told by TouchMeta, as for a read the memory meta cache served, keeps a meta file
// that ages out without it.
func TestBlackBoxTouchMetaKeepsMetaFile(t *testing.T) {
	require.False(t, metaKeptUnderChurn(t, 0, 1<<10, 64, false))
	require.True(t, metaKeptUnderChurn(t, 0, 1<<10, 64, true))
}

// Neither layer has a share of its own: meta files alone fill the whole capacity, and data
// files written after them take it back.
func TestBlackBoxLayersShareCapacity(t *testing.T) {
	ctx := context.Background()
	m := newMetaLayerFS(t, t.TempDir(), t.TempDir(), 1<<20)
	defer m.Close(ctx)
	setRangeReadCosts(t, 0, 100<<20)
	data := rangeTestData(64 << 10)
	for i := range 15 {
		name := fmt.Sprintf("obj%d", i)
		m.write(t, name, data, SkipDiskCache|SkipMemoryCache)
		m.read(t, name, metaReadPolicy, 0, int64(len(data)))
	}
	meta, files := m.files(t)
	require.Equal(t, 15, meta, "metas alone take the whole capacity")
	require.Zero(t, files)
	for i := range 15 {
		_, reads := m.read(t, fmt.Sprintf("obj%d", i), metaReadPolicy, 0, int64(len(data)))
		require.Zero(t, reads)
	}
	setRangeReadCosts(t, 0, 100<<20)
	for i := range 16 {
		m.write(t, fmt.Sprintf("data%d", i), data, 0)
	}
	meta, files = m.files(t)
	require.Zero(t, meta, "data alone take the whole capacity")
	require.Equal(t, 16, files)
}
