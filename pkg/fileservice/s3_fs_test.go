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
	"encoding/csv"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/defines"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/matrixorigin/matrixone/pkg/logutil"
	"github.com/matrixorigin/matrixone/pkg/perfcounter"
	metric "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
	"github.com/matrixorigin/matrixone/pkg/util/toml"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

type objectStorageReadRange struct {
	key string
	min *int64
	max *int64
}

type testObjectCopyStorage struct {
	dummyObjectStorage
	src     ObjectStorage
	srcKey  string
	dstKey  string
	exists  bool
	err     error
	copyErr error
	copies  int
}

func (s *testObjectCopyStorage) Exists(context.Context, string) (bool, error) {
	return s.exists, s.err
}

func (s *testObjectCopyStorage) CopyObject(
	_ context.Context,
	src ObjectStorage,
	srcKey string,
	dstKey string,
) (bool, error) {
	s.copies++
	s.src = src
	s.srcKey = srcKey
	s.dstKey = dstKey
	return true, s.copyErr
}

func TestS3FSCopyObject(t *testing.T) {
	ctx := context.Background()
	srcStorage := &testObjectCopyStorage{}
	dstStorage := &testObjectCopyStorage{}
	src := &S3FS{name: "src", keyPrefix: "cluster", server: newObjectStorageServer(srcStorage, 1024), rawStorage: srcStorage}
	dst := &S3FS{name: "dst", keyPrefix: "fixture", server: newObjectStorageServer(dstStorage, 1024), rawStorage: dstStorage}

	copied, err := dst.CopyObject(ctx, src, "objects/a", "backup/objects/a")
	require.NoError(t, err)
	require.True(t, copied)
	require.Same(t, srcStorage, dstStorage.src)
	require.Equal(t, "cluster/objects/a", dstStorage.srcKey)
	require.Equal(t, "fixture/backup/objects/a", dstStorage.dstKey)

	sub := SubPath(dst, "table")
	copied, err = sub.(ObjectCopier).CopyObject(ctx, src, "objects/a", "objects/a")
	require.NoError(t, err)
	require.True(t, copied)
	require.Equal(t, "fixture/table/objects/a", dstStorage.dstKey)

	srcServices, err := NewFileServices("src", src)
	require.NoError(t, err)
	dstServices, err := NewFileServices("dst", dst)
	require.NoError(t, err)
	copied, err = dstServices.CopyObject(ctx, srcServices, "objects/a", "objects/b")
	require.NoError(t, err)
	require.True(t, copied)
	require.Equal(t, "cluster/objects/a", dstStorage.srcKey)
	require.Equal(t, "fixture/objects/b", dstStorage.dstKey)

	noCopier := &S3FS{name: "dst", server: newObjectStorageServer(dummyObjectStorage{}, 1024), rawStorage: dummyObjectStorage{}}
	copied, err = noCopier.CopyObject(ctx, src, "objects/a", "objects/b")
	require.NoError(t, err)
	require.False(t, copied)
	_, err = noCopier.CopyObject(ctx, src, "", "objects/b")
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "got %v", err)
	_, err = noCopier.CopyObject(ctx, src, "objects/a", "")
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "got %v", err)

	dstStorage.exists = true
	_, err = dst.CopyObject(ctx, src, "objects/a", "objects/b")
	require.Error(t, err)
	dstStorage.exists = false
	dstStorage.err = errors.New("exists failed")
	_, err = dst.CopyObject(ctx, src, "objects/a", "objects/b")
	require.ErrorContains(t, err, "exists failed")
	dstStorage.err = nil
	dstStorage.copyErr = errors.New("copy failed")
	_, err = dst.CopyObject(ctx, src, "objects/a", "objects/b")
	require.ErrorContains(t, err, "copy failed")
	dstStorage.copyErr = context.DeadlineExceeded
	canceledCtx, cancel := context.WithCancel(ctx)
	cancel()
	copies := dstStorage.copies
	_, err = dst.CopyObject(canceledCtx, src, "objects/a", "objects/b")
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, copies, dstStorage.copies)

	_, err = dst.CopyObject(ctx, src, "~~", "objects/b")
	require.Error(t, err)
	_, err = dst.CopyObject(ctx, src, "objects/a", "~~")
	require.Error(t, err)
	_, err = dst.CopyObject(ctx, src, "", "objects/b")
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "got %v", err)
	_, err = dst.CopyObject(ctx, src, "objects/a", "")
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "got %v", err)
}

func TestObjectCopyCapabilityFallbacks(t *testing.T) {
	ctx := context.Background()
	plain := dummyFileService{name: "plain"}
	services, err := NewFileServices("plain", plain)
	require.NoError(t, err)

	// A FileServices destination without the optional copy capability must let
	// callers fall back to streaming instead of treating it as an error.
	copied, err := services.CopyObject(ctx, plain, "source", "destination")
	require.NoError(t, err)
	require.False(t, copied)

	_, err = services.CopyObject(ctx, plain, "source", "~~")
	require.Error(t, err)
	_, err = services.CopyObject(ctx, plain, "source", "missing:destination")
	require.Error(t, err)

	sub := SubPath(plain, "prefix").(ObjectCopier)
	copied, err = sub.CopyObject(ctx, plain, "source", "destination")
	require.NoError(t, err)
	require.False(t, copied)
	_, err = sub.CopyObject(ctx, plain, "source", "~~")
	require.Error(t, err)

	source, _, err := resolveS3CopySource(plain, "source")
	require.NoError(t, err)
	require.Nil(t, source)
	_, _, err = resolveS3CopySource(SubPath(plain, "prefix"), "~~")
	require.Error(t, err)
	_, _, err = resolveS3CopySource(services, "missing:source")
	require.Error(t, err)
}

func TestIsS3BackedFileServiceResolvesWrappers(t *testing.T) {
	s3 := &S3FS{name: "archive"}
	local := dummyFileService{name: "local"}
	services, err := NewFileServices("local", local, s3)
	require.NoError(t, err)

	require.True(t, IsS3BackedFileService(services, "archive:input.arrow"))
	require.True(t, IsS3BackedFileService(SubPath(s3, "tenant"), "input.arrow"))
	require.False(t, IsS3BackedFileService(services, "local:input.arrow"))
	require.False(t, IsS3BackedFileService(services, "missing:input.arrow"))
	require.False(t, IsS3BackedFileService(nil, "archive:input.arrow"))
}

func TestObjectCopyRejectsIncompatibleEndpoints(t *testing.T) {
	copied, err := (&AwsSDKv2{endpoint: "https://s3-b.example.com"}).CopyObject(
		context.Background(), &AwsSDKv2{endpoint: "https://s3-a.example.com"}, "src", "dst",
	)
	require.NoError(t, err)
	require.False(t, copied)

	copied, err = (&MinioSDK{endpoint: "minio-b:9000"}).CopyObject(
		context.Background(), &MinioSDK{endpoint: "minio-a:9000"}, "src", "dst",
	)
	require.NoError(t, err)
	require.False(t, copied)

	copied, err = (&AliyunSDK{endpoint: "oss-b.example.com"}).CopyObject(
		context.Background(), &AliyunSDK{endpoint: "oss-a.example.com"}, "src", "dst",
	)
	require.NoError(t, err)
	require.False(t, copied)
}

func TestObjectCopyRejectsDifferentCredentialDomains(t *testing.T) {
	left := newObjectStorageCopyCredentialDomain("left-id", "left-secret")
	right := newObjectStorageCopyCredentialDomain("right-id", "right-secret")

	copied, err := (&AwsSDKv2{
		endpoint: "https://s3.example.com", copyCredentialDomain: left,
	}).CopyObject(context.Background(), &AwsSDKv2{
		endpoint: "https://s3.example.com", copyCredentialDomain: right,
	}, "src", "dst")
	require.NoError(t, err)
	require.False(t, copied)

	copied, err = (&MinioSDK{
		endpoint: "minio.example.com", copyCredentialDomain: left,
	}).CopyObject(context.Background(), &MinioSDK{
		endpoint: "minio.example.com", copyCredentialDomain: right,
	}, "src", "dst")
	require.NoError(t, err)
	require.False(t, copied)

	copied, err = (&AliyunSDK{
		endpoint: "oss.example.com", copyCredentialDomain: left,
	}).CopyObject(context.Background(), &AliyunSDK{
		endpoint: "oss.example.com", copyCredentialDomain: right,
	}, "src", "dst")
	require.NoError(t, err)
	require.False(t, copied)

	copied, err = (&QCloudSDK{copyCredentialDomain: left}).CopyObject(
		context.Background(), &QCloudSDK{copyCredentialDomain: right}, "src", "dst",
	)
	require.NoError(t, err)
	require.False(t, copied)
}

func TestObjectStorageCopyCredentialDomain(t *testing.T) {
	first := newObjectStorageCopyCredentialDomain("id", "secret", "token")
	require.True(t, first.matches(newObjectStorageCopyCredentialDomain("id", "secret", "token")))
	require.False(t, first.matches(newObjectStorageCopyCredentialDomain("id", "other", "token")))
	require.False(t, first.matches(newObjectStorageCopyCredentialDomain("", "", "")))
}

type readRangeRecordingObjectStorage struct {
	ObjectStorage
	mu    sync.Mutex
	reads []objectStorageReadRange
}

type blockingReadObjectStorage struct {
	ObjectStorage
	readStarted chan struct{}
	releaseRead chan struct{}
	readCount   atomic.Int64
}

type generatedRangeObjectStorage struct {
	dummyObjectStorage
	mu    sync.Mutex
	reads []objectStorageReadRange
}

type controlledRangeObjectStorage struct {
	dummyObjectStorage
	data            []byte
	followerStarted chan struct{}
	releaseFollower chan struct{}
	followerOnce    sync.Once
	readCount       atomic.Int64
}

type heldWriter struct {
	once    sync.Once
	started chan struct{}
	release chan struct{}
}

type blockingDataCache struct {
	memoryCacheBacking
	updateStarted chan struct{}
	releaseUpdate chan struct{}
	updateCount   atomic.Int64
}

func (b *blockingReadObjectStorage) Read(ctx context.Context, key string, min *int64, max *int64) (io.ReadCloser, error) {
	if b.readCount.Add(1) == 1 {
		close(b.readStarted)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-b.releaseRead:
		}
	}
	return b.ObjectStorage.Read(ctx, key, min, max)
}

func (b *blockingDataCache) Set(ctx context.Context, key fscache.CacheKey, data fscache.Data) (bool, error) {
	if b.updateCount.Add(1) == 1 {
		close(b.updateStarted)
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-b.releaseUpdate:
		}
	}
	return b.memoryCacheBacking.Set(ctx, key, data)
}

func (g *generatedRangeObjectStorage) Read(
	_ context.Context,
	key string,
	min *int64,
	max *int64,
) (io.ReadCloser, error) {
	g.mu.Lock()
	g.reads = append(g.reads, objectStorageReadRange{
		key: key,
		min: cloneInt64Pointer(min),
		max: cloneInt64Pointer(max),
	})
	g.mu.Unlock()
	if min == nil || max == nil || *max < *min {
		return nil, errors.New("expected a bounded range read")
	}
	return io.NopCloser(bytes.NewReader(make([]byte, *max-*min))), nil
}

func (g *generatedRangeObjectStorage) readRanges() []objectStorageReadRange {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]objectStorageReadRange(nil), g.reads...)
}

func (c *controlledRangeObjectStorage) Read(
	ctx context.Context,
	key string,
	min *int64,
	max *int64,
) (io.ReadCloser, error) {
	if c.readCount.Add(1) > 1 {
		c.followerOnce.Do(func() { close(c.followerStarted) })
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-c.releaseFollower:
		}
	}
	start := int64(0)
	if min != nil {
		start = *min
	}
	end := int64(len(c.data))
	if max != nil {
		end = *max
	}
	if start < 0 || end < start || end > int64(len(c.data)) {
		return nil, errors.New("expected a valid bounded range read")
	}
	return io.NopCloser(bytes.NewReader(c.data[start:end])), nil
}

func (w *heldWriter) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.started) })
	<-w.release
	return len(p), nil
}

func (r *readRangeRecordingObjectStorage) Read(ctx context.Context, key string, min *int64, max *int64) (io.ReadCloser, error) {
	r.mu.Lock()
	r.reads = append(r.reads, objectStorageReadRange{
		key: key,
		min: cloneInt64Pointer(min),
		max: cloneInt64Pointer(max),
	})
	r.mu.Unlock()
	return r.ObjectStorage.Read(ctx, key, min, max)
}

func (r *readRangeRecordingObjectStorage) lastRead() (objectStorageReadRange, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.reads) == 0 {
		return objectStorageReadRange{}, false
	}
	return r.reads[len(r.reads)-1], true
}

func cloneInt64Pointer(v *int64) *int64 {
	if v == nil {
		return nil
	}
	ret := *v
	return &ret
}

func TestS3FS(
	t *testing.T,
) {
	t.Run("default policy", func(t *testing.T) {
		testS3FS(t, 0)
	})
	t.Run("skip full file preloads", func(t *testing.T) {
		testS3FS(t, SkipFullFilePreloads)
	})
}

func testS3FS(
	t *testing.T,
	policy Policy,
) {

	getArgs := func(t *testing.T, name string) ObjectStorageArguments {
		return ObjectStorageArguments{
			Name:      name,
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		}
	}

	t.Run("file service", func(t *testing.T) {
		testFileService(t, policy, func(name string) FileService {
			ctx := context.Background()
			fs, err := NewS3FS(
				ctx,
				getArgs(t, name),
				DisabledCacheConfig,
				nil,
				true,
				false,
			)
			assert.Nil(t, err)

			// to test continuation
			switch storage := fs.server.raw.(type) {
			case *AwsSDKv2:
				storage.listMaxKeys = 5
			}

			return fs
		})
	})

	t.Run("list root", func(t *testing.T) {
		ctx := context.Background()
		fs, err := NewS3FS(
			ctx,
			getArgs(t, "test"),
			DisabledCacheConfig,
			nil,
			true,
			false,
		)
		assert.Nil(t, err)
		var counterSet, counterSet2 perfcounter.CounterSet
		ctx = perfcounter.WithCounterSet(ctx, &counterSet)
		ctx = perfcounter.WithCounterSet(ctx, &counterSet2)

		err = fs.Write(ctx, IOVector{
			FilePath: "foo",
			Entries: []IOEntry{
				{
					Data: []byte("foo"),
					Size: 3,
				},
			},
		})
		assert.Nil(t, err)
		// The disk-backed test object store is not a physical S3 boundary, so it
		// must not publish S3 accepted-byte facts.
		assert.Zero(t, counterSet.FileService.S3WriteSize.Load())
		assert.Zero(t, counterSet2.FileService.S3WriteSize.Load())

		entries, err := SortedList(fs.List(ctx, ""))
		assert.Nil(t, err)

		assert.True(t, len(entries) > 0)
		assert.True(t, counterSet.FileService.S3.List.Load() > 0)
		assert.True(t, counterSet2.FileService.S3.List.Load() > 0)
	})

	t.Run("mem caching file service", func(t *testing.T) {
		testCachingFileService(t, func() CachingFileService {
			ctx := context.Background()
			fs, err := NewS3FS(
				ctx,
				getArgs(t, "test"),
				CacheConfig{
					MemoryCapacity: ptrTo[toml.ByteSize](128 * 1024),
				},
				nil,
				false,
				false,
			)
			assert.Nil(t, err)
			return fs
		})
	})

	t.Run("disk caching file service", func(t *testing.T) {
		testCachingFileService(t, func() CachingFileService {
			ctx := context.Background()
			fs, err := NewS3FS(
				ctx,
				getArgs(t, "name"),
				CacheConfig{
					MemoryCapacity: ptrTo[toml.ByteSize](1),
					DiskCapacity:   ptrTo[toml.ByteSize](128 * 1024),
					DiskPath:       ptrTo(t.TempDir()),
				},
				nil,
				false,
				false,
			)
			assert.Nil(t, err)
			return fs
		})
	})

	t.Run("mem and disk caching file service", func(t *testing.T) {
		testCachingFileService(t, func() CachingFileService {
			ctx := context.Background()
			fs, err := NewS3FS(
				ctx,
				getArgs(t, "test"),
				CacheConfig{
					MemoryCapacity: ptrTo[toml.ByteSize](128 * 1024),
					DiskCapacity:   ptrTo[toml.ByteSize](128 * 1024),
					DiskPath:       ptrTo(t.TempDir()),
				},
				nil,
				false,
				false,
			)
			assert.Nil(t, err)
			return fs
		})
	})

}

func TestDynamicS3(t *testing.T) {
	ctx := WithSQLFileAccess(context.Background(), "sys", "moadmin")
	testFileService(t, 0, func(name string) FileService {
		buf := new(strings.Builder)
		w := csv.NewWriter(buf)
		err := w.Write([]string{
			"s3",
			"disk",
			"",
			t.TempDir(),
			"",
			"",
			time.Now().Format("2006-01-02.15:04:05.000000"),
			name,
		})
		assert.Nil(t, err)
		w.Flush()
		fs, path, err := GetForETL(ctx, nil, JoinPath(
			buf.String(),
			"foo/bar/baz",
		))
		assert.Nil(t, err)
		assert.Equal(t, path, "foo/bar/baz")
		return fs
	})
}

func TestDynamicS3Opts(t *testing.T) {
	ctx := WithSQLFileAccess(context.Background(), "sys", "moadmin")
	testFileService(t, 0, func(name string) FileService {
		buf := new(strings.Builder)
		w := csv.NewWriter(buf)
		err := w.Write([]string{
			"s3-opts",
			"endpoint=disk",
			"bucket=" + t.TempDir(),
			"prefix=" + time.Now().Format("2006-01-02.15:04:05.000000"),
			"name=" + name,
		})
		assert.Nil(t, err)
		w.Flush()
		fs, path, err := GetForETL(ctx, nil, JoinPath(
			buf.String(),
			"foo/bar/baz",
		))
		assert.Nil(t, err)
		assert.Equal(t, path, "foo/bar/baz")
		return fs
	})
}

func TestDynamicS3NoKey(t *testing.T) {
	ctx := WithParallelMode(WithSQLFileAccess(context.Background(), "sys", "moadmin"), ParallelForce)
	prefix := "etl-prefix"
	buf := new(strings.Builder)
	w := csv.NewWriter(buf)
	err := w.Write([]string{
		"s3-no-key",
		"disk",
		"",
		t.TempDir(),
		prefix,
		"s3-nokey",
	})
	assert.Nil(t, err)
	w.Flush()

	fs, path, err := GetForETL(ctx, nil, JoinPath(buf.String(), "foo.txt"))
	assert.Nil(t, err)
	assert.Equal(t, "foo.txt", path)
	s3fs, ok := fs.(*S3FS)
	assert.True(t, ok)
	assert.Equal(t, ParallelForce, s3fs.parallelMode)
	assert.Equal(t, prefix, s3fs.keyPrefix)
	assert.IsType(t, &diskObjectStorage{}, baseStorage(t, s3fs))
}

func TestDynamicMinio(t *testing.T) {
	ctx := WithParallelMode(WithSQLFileAccess(context.Background(), "sys", "moadmin"), ParallelForce)
	buf := new(strings.Builder)
	w := csv.NewWriter(buf)
	err := w.Write([]string{
		"minio",
		"http://127.0.0.1:9000",
		"us-east-1",
		"bucket",
		"ak",
		"sk",
		"pref",
		"minio-etl",
	})
	assert.Nil(t, err)
	w.Flush()

	fs, path, err := GetForETL(ctx, nil, JoinPath(buf.String(), "bar.txt"))
	assert.Nil(t, err)
	assert.Equal(t, "bar.txt", path)
	s3fs, ok := fs.(*S3FS)
	assert.True(t, ok)
	assert.Equal(t, ParallelForce, s3fs.parallelMode)
	assert.Equal(t, "pref", s3fs.keyPrefix)
	assert.IsType(t, &MinioSDK{}, baseStorage(t, s3fs))
}

func baseStorage(t *testing.T, fs *S3FS) ObjectStorage {
	return fs.server.raw
}

func BenchmarkS3FS(b *testing.B) {
	cacheDir := b.TempDir()
	b.ResetTimer()

	ctx := context.Background()
	benchmarkFileService(ctx, b, func() FileService {
		fs, err := NewS3FS(
			ctx,
			ObjectStorageArguments{
				Name:      "bench",
				Endpoint:  "disk",
				IsMinio:   false,
				Region:    "",
				Bucket:    b.TempDir(),
				KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
				KeyID:     "",
				KeySecret: "",
			},
			CacheConfig{
				DiskPath: ptrTo(cacheDir),
			},
			nil,
			true,
			false,
		)
		assert.Nil(b, err)
		return fs
	})
}

func TestS3FSWithSubPath(t *testing.T) {
	testFileService(t, 0, func(name string) FileService {
		ctx := context.Background()
		fs, err := NewS3FS(
			ctx,
			ObjectStorageArguments{
				Name:      name,
				Endpoint:  "disk",
				Bucket:    t.TempDir(),
				KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
			},
			DisabledCacheConfig,
			nil,
			true,
			false,
		)
		assert.Nil(t, err)
		return SubPath(fs, "foo/")
	})
}

func BenchmarkS3ConcurrentRead(b *testing.B) {
	ctx := context.Background()

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "bench",
			Endpoint:  "disk",
			Bucket:    b.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		DisabledCacheConfig,
		nil,
		true,
		false,
	)
	if err != nil {
		b.Fatal(err)
	}
	if fs == nil {
		b.Fatal(err)
	}

	vector := IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{
				Size: 3,
				Data: []byte("foo"),
			},
		},
	}
	err = fs.Write(ctx, vector)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		sem := make(chan struct{}, 128)
		for pb.Next() {
			sem <- struct{}{}
			go func() {
				defer func() {
					<-sem
				}()
				err := fs.Read(ctx, &IOVector{
					FilePath: "foo",
					Entries: []IOEntry{
						{
							Size: 3,
						},
					},
				})
				if err != nil {
					panic(err)
				}
			}()
		}
		for i := 0; i < cap(sem); i++ {
			sem <- struct{}{}
		}
	})

}

func TestSequentialS3Read(t *testing.T) {
	ctx := context.Background()

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "bench",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		DisabledCacheConfig,
		nil,
		true,
		false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if fs == nil {
		t.Fatal(err)
	}

	vector := IOVector{
		FilePath: "foo",
		Entries: []IOEntry{
			{
				Size: 3,
				Data: []byte("foo"),
			},
		},
	}
	err = fs.Write(ctx, vector)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 128; i++ {
		err := fs.Read(ctx, &IOVector{
			FilePath: "foo",
			Entries: []IOEntry{
				{
					Size: 3,
				},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
	}

}

func TestS3PrefetchFile(t *testing.T) {
	ctx := context.Background()
	var pcSet perfcounter.CounterSet
	ctx = perfcounter.WithCounterSet(ctx, &pcSet)

	cacheDir := t.TempDir()
	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(cacheDir),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	assert.Nil(t, err)
	defer fs.Close(ctx)

	data := bytes.Repeat([]byte("abcd"), 2<<20)

	// write file
	err = fs.Write(ctx, IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: int64(len(data)),
				Data: data,
			},
		},
		Policy: SkipDiskCache | SkipMemoryCache,
	})
	assert.Nil(t, err)

	// preload
	err = fs.PrefetchFile(ctx, "foo/bar")
	assert.Nil(t, err)
	err = fs.PrefetchFile(ctx, "foo/bar")
	assert.Nil(t, err)
	fs.FlushCache(ctx)

	// read
	// Exercise representative cache reads without repeatedly rereading the
	// whole object. The old 1000-point ramp performed about 4 GiB of reads for
	// an 8 MiB fixture while checking the same disk-cache hit invariant.
	readSizes := []int{
		1,
		4 << 10,
		_ReadCoalesceSize,
		1 << 20,
		len(data) / 2,
		len(data) - 1,
		len(data),
	}
	lastHit := int64(0)
	for _, size := range readSizes {
		vec := &IOVector{
			FilePath: "foo/bar",
			Entries: []IOEntry{
				{
					Size: int64(size),
				},
			},
		}
		err = fs.Read(ctx, vec)
		assert.Nil(t, err)
		assert.Equal(t, data[:size], vec.Entries[0].Data)
		assert.Equal(t, lastHit+1, pcSet.FileService.Cache.Disk.Hit.Load())
		vec.Release()
		lastHit++
	}

}

func TestS3PrefetchFileDoesNotWaitForAsyncDiskFinalize(t *testing.T) {
	ctx := context.Background()
	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 20),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)
	require.NoError(t, fs.Write(ctx, IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{{
			Size: 11,
			Data: []byte("hello world"),
		}},
		Policy: SkipDiskCache | SkipMemoryCache,
	}))

	syncStarted, unblock := installBlockedDiskCacheFileSync(fs.diskCache)
	t.Cleanup(func() {
		unblock()
		fs.Close(ctx)
	})

	prefetchDone := make(chan error, 1)
	go func() {
		prefetchDone <- fs.PrefetchFile(ctx, "foo/bar")
	}()
	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async prefetch finalizer did not start")
	}
	select {
	case err := <-prefetchDone:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		unblock()
		t.Fatalf("prefetch waited for disk finalization: %v", <-prefetchDone)
	}

	diskPath := fs.diskCache.wholePath("foo/bar", 11)
	_, err = os.Stat(diskPath)
	require.ErrorIs(t, err, os.ErrNotExist)

	unblock()
	flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
	defer cancel()
	fs.FlushCache(flushCtx)
	require.NoError(t, flushCtx.Err())
	data, err := os.ReadFile(diskPath)
	require.NoError(t, err)
	require.Equal(t, []byte("hello world"), data)
}

func TestS3FSWriteDiskCacheFinalizeMode(t *testing.T) {
	for _, test := range []struct {
		name  string
		async bool
	}{
		{name: "async", async: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := context.Background()
			fs, err := NewS3FS(
				ctx,
				ObjectStorageArguments{
					Name:      "s3",
					Endpoint:  "disk",
					Bucket:    t.TempDir(),
					KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
				},
				CacheConfig{
					DiskPath:     ptrTo(t.TempDir()),
					DiskCapacity: ptrTo[toml.ByteSize](1 << 20),
				},
				nil,
				false,
				false,
			)
			require.NoError(t, err)

			syncStarted, unblock := installBlockedDiskCacheFileSync(fs.diskCache)
			t.Cleanup(func() {
				unblock()
				fs.Close(ctx)
			})

			writeDone := make(chan error, 1)
			go func() {
				writeDone <- fs.Write(ctx, IOVector{
					FilePath: "foo/bar",
					Entries: []IOEntry{{
						Size: 11,
						Data: []byte("hello world"),
					}},
				})
			}()
			select {
			case <-syncStarted:
			case <-time.After(diskCacheLifecycleTestTimeout):
				t.Fatal("disk-cache finalization did not start")
			}

			diskPath := fs.diskCache.wholePath("foo/bar", 11)
			select {
			case err := <-writeDone:
				require.NoError(t, err)
			case <-time.After(diskCacheLifecycleTestTimeout):
				unblock()
				t.Fatalf("write waited for disk finalization: %v", <-writeDone)
			}
			_, err = os.Stat(diskPath)
			require.ErrorIs(t, err, os.ErrNotExist)

			unblock()
			flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
			defer cancel()
			fs.FlushCache(flushCtx)
			require.NoError(t, flushCtx.Err())
			data, err := os.ReadFile(diskPath)
			require.NoError(t, err)
			require.Equal(t, []byte("hello world"), data)
		})
	}
}

func TestS3FSFullObjectDiskCacheFillDoesNotRetainWholeObjectBuffer(t *testing.T) {
	ctx := context.Background()
	var pcSet perfcounter.CounterSet
	ctx = perfcounter.WithCounterSet(ctx, &pcSet)

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	assert.Nil(t, err)
	defer fs.Close(ctx)

	data := bytes.Repeat([]byte("abcd"), 1<<18)
	err = fs.Write(ctx, IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: int64(len(data)),
				Data: data,
			},
		},
		Policy: SkipDiskCache | SkipMemoryCache,
	})
	assert.Nil(t, err)

	vec := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Offset: 12345,
				Size:   7,
			},
		},
	}
	err = fs.Read(ctx, vec)
	assert.Nil(t, err)
	assert.Equal(t, data[12345:12352], vec.Entries[0].Data)
	assert.Less(t, cap(vec.Entries[0].Data), len(data)/16)
	assert.Equal(t, int64(1), pcSet.FileService.S3.Get.Load())
	vec.Release()

	// A read in async-update mode guarantees cache admission, not publication.
	// Wait on the explicit barrier before asserting that a later read hits disk.
	flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
	defer cancel()
	fs.FlushCache(flushCtx)
	require.NoError(t, flushCtx.Err())

	hitVec := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Offset: 54321,
				Size:   11,
			},
		},
	}
	err = fs.Read(ctx, hitVec)
	assert.Nil(t, err)
	assert.Equal(t, data[54321:54332], hitVec.Entries[0].Data)
	assert.Equal(t, int64(1), pcSet.FileService.S3.Get.Load())
	assert.Equal(t, int64(1), pcSet.FileService.Cache.Disk.Hit.Load())
	hitVec.Release()
}

func TestS3FSFullObjectDiskCacheFillDoesNotReturnWholeObjectDataWithCachedData(t *testing.T) {
	ctx := context.Background()
	var pcSet perfcounter.CounterSet
	ctx = perfcounter.WithCounterSet(ctx, &pcSet)

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	assert.Nil(t, err)
	defer fs.Close(ctx)

	data := bytes.Repeat([]byte("abcd"), 1<<18)
	err = fs.Write(ctx, IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: int64(len(data)),
				Data: data,
			},
		},
		Policy: SkipDiskCache | SkipMemoryCache,
	})
	assert.Nil(t, err)

	vec := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size:        int64(len(data)),
				ToCacheData: CacheOriginalData,
			},
		},
	}
	err = fs.Read(ctx, vec)
	assert.Nil(t, err)
	assert.NotNil(t, vec.Entries[0].CachedData)
	if vec.Entries[0].CachedData != nil {
		assert.Equal(t, data, vec.Entries[0].CachedData.Bytes())
	}
	assert.Nil(t, vec.Entries[0].Data)
	assert.Equal(t, int64(1), pcSet.FileService.S3.Get.Load())
	vec.Release()
}

// The full-object default lets later reads reuse one disk-cache artifact.
// Without a disk cache, the default must not expand a bounded entry.
func TestS3FSReadPolicyPreservesFullObjectDefault(t *testing.T) {
	for _, tc := range []struct {
		name         string
		policy       Policy
		disk, memory bool
		noCache      bool
		min          int64
		max          *int64
	}{
		{name: "default full object", disk: true, min: 0},
		{name: "explicit range", disk: true, policy: SkipFullFilePreloads, min: 123, max: ptrTo(int64(130))},
		{name: "zero cache capacities", min: 123, max: ptrTo(int64(130))},
		{name: "memory cache only", memory: true, min: 123, max: ptrTo(int64(130))},
		{name: "cache initialization disabled", disk: true, noCache: true, min: 123, max: ptrTo(int64(130))},
		{name: "disk writes disabled", disk: true, policy: SkipDiskCacheWrites, min: 123, max: ptrTo(int64(130))},
		{name: "disk reads disabled", disk: true, policy: SkipDiskCacheReads, min: 123, max: ptrTo(int64(130))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			config := CacheConfig{
				MemoryCapacity: ptrTo[toml.ByteSize](0),
				DiskCapacity:   ptrTo[toml.ByteSize](0),
				DiskPath:       ptrTo(t.TempDir()),
			}
			if tc.disk {
				config.DiskCapacity = ptrTo[toml.ByteSize](1 << 20)
			}
			if tc.memory {
				config.MemoryCapacity = ptrTo[toml.ByteSize](1 << 20)
			}
			fs, err := NewS3FS(
				ctx,
				ObjectStorageArguments{
					Name: "s3", Endpoint: "disk", Bucket: t.TempDir(),
				},
				config,
				nil, tc.noCache, false,
			)
			require.NoError(t, err)
			t.Cleanup(func() { fs.Close(ctx) })

			data := bytes.Repeat([]byte("abcd"), 64)
			require.NoError(t, fs.Write(ctx, IOVector{
				FilePath: "foo/bar",
				Entries:  []IOEntry{{Size: int64(len(data)), Data: data}},
				Policy:   SkipDiskCache | SkipMemoryCache,
			}))

			recorder := &readRangeRecordingObjectStorage{ObjectStorage: fs.server.raw}
			fs.server.raw = recorder
			vec := &IOVector{
				FilePath: "foo/bar",
				Policy:   tc.policy,
				Entries:  []IOEntry{{Offset: 123, Size: 7}},
			}
			t.Cleanup(vec.Release)
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[123:130], vec.Entries[0].Data)
			require.Equal(t, tc.policy, vec.Policy, "range selection must not mutate caller policy")

			read, ok := recorder.lastRead()
			require.True(t, ok)
			require.NotNil(t, read.min)
			require.Equal(t, tc.min, *read.min)
			if tc.max == nil {
				require.Nil(t, read.max)
			} else {
				require.Equal(t, *tc.max, *read.max)
			}
		})
	}
}

func TestS3FSRangeReadSkipsFullObjectDiskCacheUpdate(t *testing.T) {
	ctx := context.Background()
	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	assert.Nil(t, err)
	defer fs.Close(ctx)

	data := bytes.Repeat([]byte("abcd"), 1<<10)
	err = fs.Write(ctx, IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: int64(len(data)),
				Data: data,
			},
		},
		Policy: SkipDiskCache | SkipMemoryCache,
	})
	assert.Nil(t, err)

	fullObjectVec := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: int64(len(data)),
			},
		},
	}
	doneMerge, waitMerge := fs.ioMerger.Merge(fs.readMergeKey(fullObjectVec), maxIOWaitDuration)
	assert.NotNil(t, doneMerge)
	assert.Nil(t, waitMerge)
	releasedMerge := false
	releaseMerge := func() {
		if !releasedMerge {
			doneMerge()
			releasedMerge = true
		}
	}
	defer releaseMerge()

	type readResult struct {
		data []byte
		err  error
	}
	readDone := make(chan readResult, 1)
	go func() {
		vec := &IOVector{
			FilePath: "foo/bar",
			Entries: []IOEntry{
				{
					Offset: 123,
					Size:   7,
				},
			},
		}
		err := fs.Read(ctx, vec)
		defer vec.Release()
		if err != nil {
			readDone <- readResult{err: err}
			return
		}
		readDone <- readResult{data: append([]byte(nil), vec.Entries[0].Data...)}
	}()

	select {
	case result := <-readDone:
		assert.Nil(t, result.err)
		assert.Equal(t, data[123:130], result.data)
	case <-time.After(2 * time.Second):
		releaseMerge()
		result := <-readDone
		t.Fatalf("range read waited for full-object io merge: %v", result.err)
	}
}

func TestS3FSRangeReadSkipsFullObjectIOMergeBeforeDiskCacheUpdate(t *testing.T) {
	ctx := context.Background()
	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	assert.Nil(t, err)
	defer fs.Close(ctx)

	data := bytes.Repeat([]byte("abcd"), 1<<10)
	err = fs.Write(ctx, IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: int64(len(data)),
				Data: data,
			},
		},
		Policy: SkipDiskCache | SkipMemoryCache,
	})
	assert.Nil(t, err)

	recorder := &readRangeRecordingObjectStorage{
		ObjectStorage: fs.server.raw,
	}
	fs.server.raw = recorder

	fullObjectVec := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: int64(len(data)),
			},
		},
	}
	doneMerge, waitMerge := fs.ioMerger.Merge(fs.readMergeKey(fullObjectVec), maxIOWaitDuration)
	assert.NotNil(t, doneMerge)
	assert.Nil(t, waitMerge)
	releasedMerge := false
	releaseMerge := func() {
		if !releasedMerge {
			doneMerge()
			releasedMerge = true
		}
	}
	defer releaseMerge()

	type readResult struct {
		data []byte
		err  error
	}
	readDone := make(chan readResult, 1)
	go func() {
		vec := &IOVector{
			FilePath: "foo/bar",
			Entries: []IOEntry{
				{
					Offset: 123,
					Size:   7,
				},
			},
		}
		err := fs.Read(ctx, vec)
		defer vec.Release()
		if err != nil {
			readDone <- readResult{err: err}
			return
		}
		readDone <- readResult{data: append([]byte(nil), vec.Entries[0].Data...)}
	}()

	select {
	case result := <-readDone:
		assert.Nil(t, result.err)
		assert.Equal(t, data[123:130], result.data)
		assert.True(t, fs.ioMerger.IsMerging(fs.readMergeKey(fullObjectVec)))
		read, ok := recorder.lastRead()
		assert.True(t, ok)
		if assert.NotNil(t, read.min) {
			assert.Equal(t, int64(123), *read.min)
		}
		if assert.NotNil(t, read.max) {
			assert.Equal(t, int64(130), *read.max)
		}
	case <-time.After(2 * time.Second):
		releaseMerge()
		result := <-readDone
		t.Fatalf("range read waited for full-object io merge before disk cache update: %v", result.err)
	}
}

func TestS3FSExpensiveRangeWaitsForFullObjectMergeUntilContextDeadline(t *testing.T) {
	ctx := context.Background()
	originalShortWait := shortIOWaitDuration
	shortIOWaitDuration = time.Millisecond
	t.Cleanup(func() {
		shortIOWaitDuration = originalShortWait
	})

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)

	storage := &blockingReadObjectStorage{
		ObjectStorage: fs.server.raw,
		readStarted:   make(chan struct{}),
		releaseRead:   make(chan struct{}),
	}
	fs.server.raw = storage
	vector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{Offset: 0, Size: 1 << 20},
			{Offset: 32 << 20, Size: 1 << 20},
		},
	}
	doneMerge, waitMerge := fs.ioMerger.Merge(fs.readMergeKey(vector), maxIOWaitDuration)
	require.NotNil(t, doneMerge)
	require.Nil(t, waitMerge)
	releaseMerge := sync.OnceFunc(doneMerge)
	releaseRead := sync.OnceFunc(func() { close(storage.releaseRead) })
	readCtx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	t.Cleanup(func() {
		cancel()
		releaseRead()
		releaseMerge()
		vector.Release()
		fs.Close(ctx)
	})

	err = fs.Read(readCtx, vector)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Equal(t, int64(0), storage.readCount.Load())
}

func TestS3FSExpensiveRangeHasBoundedPerEntryFallback(t *testing.T) {
	ctx := context.Background()
	originalShortWait := shortIOWaitDuration
	originalMaxWait := maxIOWaitDuration
	shortIOWaitDuration = time.Millisecond
	maxIOWaitDuration = 5 * time.Millisecond
	t.Cleanup(func() {
		shortIOWaitDuration = originalShortWait
		maxIOWaitDuration = originalMaxWait
	})

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)

	storage := &generatedRangeObjectStorage{}
	fs.server.raw = storage
	vector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{Offset: 0, Size: 1},
			{Offset: 9 << 20, Size: 1},
		},
	}
	doneMerge, waitMerge := fs.ioMerger.Merge(fs.readMergeKey(vector), maxIOWaitDuration)
	require.NotNil(t, doneMerge)
	require.Nil(t, waitMerge)
	releaseMerge := sync.OnceFunc(doneMerge)
	t.Cleanup(func() {
		releaseMerge()
		vector.Release()
		fs.Close(ctx)
	})

	readDone := make(chan error, 1)
	go func() {
		readDone <- fs.Read(context.Background(), vector)
	}()
	select {
	case err := <-readDone:
		require.NoError(t, err)
	case <-time.After(time.Second):
		releaseMerge()
		select {
		case <-readDone:
		case <-time.After(time.Second):
			t.Fatal("follower did not terminate after releasing the merge leader")
		}
		t.Fatal("follower with no caller deadline did not reach a bounded fallback")
	}

	reads := storage.readRanges()
	require.Len(t, reads, 2)
	require.Equal(t, int64(0), *reads[0].min)
	require.Equal(t, int64(1), *reads[0].max)
	require.Equal(t, int64(9<<20), *reads[1].min)
	require.Equal(t, int64(9<<20)+1, *reads[1].max)
}

func TestS3FSCacheFillDoesNotWaitForNonProducingLeader(t *testing.T) {
	ctx := context.Background()
	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(ctx) })

	data := make([]byte, 1025)
	data[0] = 1
	data[len(data)-1] = 2
	storage := &controlledRangeObjectStorage{
		data:            data,
		followerStarted: make(chan struct{}),
		releaseFollower: make(chan struct{}),
	}
	releaseFollower := sync.OnceFunc(func() { close(storage.releaseFollower) })
	fs.server.raw = storage
	fs.rawStorage = storage
	t.Cleanup(releaseFollower)

	writer := &heldWriter{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	releaseWriter := sync.OnceFunc(func() { close(writer.release) })
	leaderVector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{{
			Offset:        0,
			Size:          1,
			WriterForRead: writer,
		}},
	}
	followerVector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{Offset: 0, Size: 1},
			{Offset: int64(len(data) - 1), Size: 1},
		},
	}
	t.Cleanup(followerVector.Release)
	require.False(t, fs.readMergeKey(leaderVector).CacheFill)
	require.True(t, fs.readMergeKey(followerVector).CacheFill)
	require.NotEqual(t, fs.readMergeKey(leaderVector), fs.readMergeKey(followerVector))

	leaderDone := make(chan error, 1)
	go func() { leaderDone <- fs.Read(ctx, leaderVector) }()
	t.Cleanup(func() {
		releaseWriter()
		select {
		case err := <-leaderDone:
			require.NoError(t, err)
		case <-time.After(5 * time.Second):
			t.Error("non-cache-producing leader did not terminate")
		}
	})
	select {
	case <-writer.started:
	case <-time.After(5 * time.Second):
		t.Fatal("non-cache-producing leader did not reach its writer")
	}
	require.False(t, fs.ioMerger.IsMerging(fs.readMergeKey(leaderVector)))

	followerDone := make(chan error, 1)
	go func() { followerDone <- fs.Read(ctx, followerVector) }()
	select {
	case <-storage.followerStarted:
	case err := <-followerDone:
		require.NoError(t, err)
		t.Fatal("cache-producing follower completed before the controlled storage read was observed")
	case <-time.After(5 * time.Second):
		releaseWriter()
		releaseFollower()
		t.Fatal("cache-producing follower did not start storage read while non-cache-producing leader was blocked")
	}
	releaseFollower()
	select {
	case err := <-followerDone:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		releaseWriter()
		t.Fatal("cache-producing follower did not terminate after storage release")
	}
	require.Equal(t, []byte{1}, followerVector.Entries[0].Data)
	require.Equal(t, []byte{2}, followerVector.Entries[1].Data)
}

func TestS3FSRangeMergeWaitHasBoundedFallback(t *testing.T) {
	ctx := context.Background()
	originalMaxWait := maxIOWaitDuration
	maxIOWaitDuration = 5 * time.Millisecond
	t.Cleanup(func() { maxIOWaitDuration = originalMaxWait })

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(ctx) })
	generatedStorage := &generatedRangeObjectStorage{}
	storage := &blockingReadObjectStorage{
		ObjectStorage: generatedStorage,
		readStarted:   make(chan struct{}),
		releaseRead:   make(chan struct{}),
	}
	fs.server.raw = storage
	vector := &IOVector{
		FilePath: "foo/bar",
		Policy:   SkipFullFilePreloads | SkipDiskCacheWrites,
		Entries:  []IOEntry{{Offset: 0, Size: 1}},
	}
	t.Cleanup(vector.Release)
	finishMerge, waitMerge := fs.ioMerger.Merge(fs.readMergeKey(vector), maxIOWaitDuration)
	require.NotNil(t, finishMerge)
	require.Nil(t, waitMerge)
	releaseMerge := sync.OnceFunc(finishMerge)
	t.Cleanup(releaseMerge)
	releaseRead := sync.OnceFunc(func() { close(storage.releaseRead) })
	t.Cleanup(releaseRead)

	readCtx := WithEventLogger(context.Background())
	waiterLogger := readCtx.Value(EventLoggerKey).(*eventLogger)
	readDone := make(chan error, 1)
	readExited := make(chan struct{})
	go func() {
		defer close(readExited)
		readDone <- fs.Read(readCtx, vector)
	}()
	t.Cleanup(func() {
		releaseRead()
		releaseMerge()
		select {
		case <-readExited:
		case <-time.After(5 * time.Second):
			t.Error("range follower goroutine did not terminate")
		}
	})
	require.Eventually(t, func() bool {
		waiterLogger.mu.Lock()
		defer waiterLogger.mu.Unlock()
		for _, ev := range *waiterLogger.events {
			if ev.ev == str_ioMerger_Merge_wait {
				return true
			}
		}
		return false
	}, 5*time.Second, time.Millisecond,
		"range follower did not join the held merge generation")
	select {
	case <-storage.readStarted:
		require.True(t, fs.ioMerger.IsMerging(fs.readMergeKey(vector)),
			"the timed-out follower must bypass, rather than replace, the held generation")
	case err := <-readDone:
		require.NoError(t, err)
		t.Fatal("range follower completed before its fallback storage read was observed")
	case <-time.After(5 * time.Second):
		releaseMerge()
		releaseRead()
		t.Fatal("range follower did not bypass the timed-out merge generation")
	}
	releaseRead()
	select {
	case err := <-readDone:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		releaseMerge()
		t.Fatal("range follower did not terminate after its fallback storage read was released")
	}
	reads := generatedStorage.readRanges()
	require.Len(t, reads, 1)
	require.Equal(t, int64(0), *reads[0].min)
	require.Equal(t, int64(1), *reads[0].max)
}

func TestS3FSSingleEntryFallbackCoalescesExactRange(t *testing.T) {
	ctx := context.Background()
	originalShortWait := shortIOWaitDuration
	shortIOWaitDuration = time.Millisecond
	t.Cleanup(func() { shortIOWaitDuration = originalShortWait })

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			MemoryCapacity: ptrTo[toml.ByteSize](1 << 20),
			DiskPath:       ptrTo(t.TempDir()),
			DiskCapacity:   ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)
	require.NotNil(t, fs.memCache)
	t.Cleanup(func() { fs.Close(ctx) })

	rangeStorage := &generatedRangeObjectStorage{}
	storage := &blockingReadObjectStorage{
		ObjectStorage: rangeStorage,
		readStarted:   make(chan struct{}),
		releaseRead:   make(chan struct{}),
	}
	fs.server.raw = storage

	newVector := func() *IOVector {
		return &IOVector{
			FilePath: "foo/bar",
			Entries: []IOEntry{{
				Offset:      123,
				Size:        7,
				ToCacheData: CacheOriginalData,
			}},
		}
	}
	fullObjectVector := newVector()
	fullObjectDone, fullObjectWait := fs.ioMerger.Merge(
		fs.readMergeKey(fullObjectVector), maxIOWaitDuration,
	)
	require.NotNil(t, fullObjectDone)
	require.Nil(t, fullObjectWait)
	releaseFullObject := sync.OnceFunc(fullObjectDone)
	releaseStorage := sync.OnceFunc(func() { close(storage.releaseRead) })
	t.Cleanup(func() {
		releaseStorage()
		releaseFullObject()
	})

	leaderVector := newVector()
	leaderDone := make(chan error, 1)
	go func() { leaderDone <- fs.Read(ctx, leaderVector) }()
	select {
	case <-storage.readStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("single-range leader did not reach object storage")
	}

	const followers = 16
	followerCtx, cancelFollowers := context.WithTimeout(ctx, 100*time.Millisecond)
	defer cancelFollowers()
	followerDone := make(chan error, followers)
	for range followers {
		go func() {
			vector := newVector()
			defer vector.Release()
			followerDone <- fs.Read(followerCtx, vector)
		}()
	}
	for range followers {
		select {
		case err := <-followerDone:
			require.ErrorIs(t, err, context.DeadlineExceeded)
		case <-time.After(5 * time.Second):
			t.Fatal("single-range follower did not honor caller cancellation")
		}
	}
	require.Equal(t, int64(1), storage.readCount.Load())

	releaseStorage()
	select {
	case err := <-leaderDone:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("single-range leader did not terminate")
	}
	require.Equal(t, []byte(make([]byte, 7)), leaderVector.Entries[0].Data)
	leaderVector.Release()

	cachedVector := newVector()
	require.NoError(t, fs.Read(ctx, cachedVector))
	require.NotNil(t, cachedVector.Entries[0].CachedData)
	require.Equal(t, []byte(make([]byte, 7)), cachedVector.Entries[0].CachedData.Bytes())
	cachedVector.Release()
	require.Equal(t, int64(1), storage.readCount.Load())
	require.Len(t, rangeStorage.readRanges(), 1)
}

func TestS3FSSingleEntryRangeMergeKeyValidation(t *testing.T) {
	fs := &S3FS{memCache: &MemCache{}}
	cacheableEntry := func(offset int64) IOEntry {
		return IOEntry{
			Offset:      offset,
			Size:        7,
			ToCacheData: CacheOriginalData,
		}
	}

	t.Run("skips completed entries", func(t *testing.T) {
		completed := cacheableEntry(10)
		completed.done = true
		vector := &IOVector{
			FilePath: "foo/bar",
			Entries:  []IOEntry{completed, cacheableEntry(123)},
		}

		key, ok := fs.singleEntryRangeMergeKey(vector)
		require.True(t, ok)
		require.Equal(t, IOMergeKey{
			Path:      "foo/bar",
			Offset:    123,
			End:       130,
			CacheFill: true,
		}, key)
	})

	t.Run("rejects an already completed vector", func(t *testing.T) {
		completed := cacheableEntry(10)
		completed.done = true

		_, ok := fs.singleEntryRangeMergeKey(&IOVector{
			FilePath: "foo/bar",
			Entries:  []IOEntry{completed},
		})
		require.False(t, ok)
	})

	t.Run("rejects multiple unfinished entries", func(t *testing.T) {
		_, ok := fs.singleEntryRangeMergeKey(&IOVector{
			FilePath: "foo/bar",
			Entries: []IOEntry{
				cacheableEntry(10),
				cacheableEntry(20),
			},
		})
		require.False(t, ok)
	})
}

func TestS3FSSingleEntryFallbackDoesNotMergeDifferentRanges(t *testing.T) {
	ctx := context.Background()
	originalShortWait := shortIOWaitDuration
	shortIOWaitDuration = time.Millisecond
	t.Cleanup(func() { shortIOWaitDuration = originalShortWait })

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			MemoryCapacity: ptrTo[toml.ByteSize](1 << 20),
			DiskPath:       ptrTo(t.TempDir()),
			DiskCapacity:   ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(ctx) })

	rangeStorage := &generatedRangeObjectStorage{}
	storage := &blockingReadObjectStorage{
		ObjectStorage: rangeStorage,
		readStarted:   make(chan struct{}),
		releaseRead:   make(chan struct{}),
	}
	fs.server.raw = storage

	newVector := func(offset int64) *IOVector {
		return &IOVector{
			FilePath: "foo/bar",
			Entries: []IOEntry{{
				Offset:      offset,
				Size:        7,
				ToCacheData: CacheOriginalData,
			}},
		}
	}
	fullObjectDone, fullObjectWait := fs.ioMerger.Merge(
		fs.readMergeKey(newVector(123)), maxIOWaitDuration,
	)
	require.NotNil(t, fullObjectDone)
	require.Nil(t, fullObjectWait)
	releaseFullObject := sync.OnceFunc(fullObjectDone)
	releaseStorage := sync.OnceFunc(func() { close(storage.releaseRead) })
	t.Cleanup(func() {
		releaseStorage()
		releaseFullObject()
	})

	first := newVector(123)
	firstDone := make(chan error, 1)
	go func() { firstDone <- fs.Read(ctx, first) }()
	select {
	case <-storage.readStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("first exact range did not reach object storage")
	}

	second := newVector(456)
	secondDone := make(chan error, 1)
	go func() { secondDone <- fs.Read(ctx, second) }()
	select {
	case err := <-secondDone:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("different exact range waited for an unrelated leader")
	}
	require.Equal(t, int64(2), storage.readCount.Load())

	releaseStorage()
	select {
	case err := <-firstDone:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("first exact range did not terminate")
	}
	first.Release()
	second.Release()
}

func TestS3FSRangeReadSkipsPrefetchFullObjectIOMerge(t *testing.T) {
	ctx := context.Background()
	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 30),
		},
		nil,
		false,
		false,
	)
	assert.Nil(t, err)
	defer fs.Close(ctx)

	data := bytes.Repeat([]byte("abcd"), 1<<10)
	err = fs.Write(ctx, IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: int64(len(data)),
				Data: data,
			},
		},
		Policy: SkipDiskCache | SkipMemoryCache,
	})
	assert.Nil(t, err)

	doneMerge, waitMerge := fs.ioMerger.Merge(IOMergeKey{
		Path:       "foo/bar",
		FullObject: true,
		CacheFill:  true,
	}, maxIOWaitDuration)
	assert.NotNil(t, doneMerge)
	assert.Nil(t, waitMerge)
	releasedMerge := false
	releaseMerge := func() {
		if !releasedMerge {
			doneMerge()
			releasedMerge = true
		}
	}
	defer releaseMerge()

	type readResult struct {
		data []byte
		err  error
	}
	readDone := make(chan readResult, 1)
	go func() {
		vec := &IOVector{
			FilePath: "foo/bar",
			Entries: []IOEntry{
				{
					Offset: 5,
					Size:   9,
				},
			},
		}
		err := fs.Read(ctx, vec)
		defer vec.Release()
		if err != nil {
			readDone <- readResult{err: err}
			return
		}
		readDone <- readResult{data: append([]byte(nil), vec.Entries[0].Data...)}
	}()

	select {
	case result := <-readDone:
		assert.Nil(t, result.err)
		assert.Equal(t, data[5:14], result.data)
	case <-time.After(2 * time.Second):
		releaseMerge()
		result := <-readDone
		t.Fatalf("range read waited for prefetch full-object io merge: %v", result.err)
	}
}

func TestS3FSReadFullObjectToDiskCacheStreamingDoesNotOpenReaderWhenCacheExists(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	err = cache.SetFile(ctx, "foo/bar", func(context.Context) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader([]byte("hello world"))), nil
	})
	assert.Nil(t, err)

	fs := &S3FS{
		diskCache: cache,
	}
	vector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Offset: 1,
				Size:   4,
			},
		},
	}

	getReaderCalls := 0
	done, err := fs.readFullObjectToDiskCacheStreaming(
		ctx,
		vector,
		"foo/bar",
		getterSource(func(context.Context, *int64, *int64) (io.ReadCloser, error) {
			getReaderCalls++
			return io.NopCloser(bytes.NewReader([]byte("backend"))), nil
		}),
	)
	assert.Nil(t, err)
	assert.False(t, done)
	assert.Zero(t, getReaderCalls)
}

func TestS3FSReadFullObjectToDiskCacheStreamingDoesNotWaitForAsyncFinalize(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)

	syncStarted, unblock := installBlockedDiskCacheFileSync(cache)
	t.Cleanup(func() {
		unblock()
		cache.Close(ctx)
	})

	s3 := &S3FS{
		diskCache: cache,
	}
	vector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{{
			Offset: 1,
			Size:   4,
		}},
	}
	type result struct {
		done bool
		err  error
	}
	readDone := make(chan result, 1)
	go func() {
		done, err := s3.readFullObjectToDiskCacheStreaming(
			ctx,
			vector,
			"foo/bar",
			getterSource(func(context.Context, *int64, *int64) (io.ReadCloser, error) {
				return io.NopCloser(bytes.NewReader([]byte("hello world"))), nil
			}),
		)
		readDone <- result{done: done, err: err}
	}()

	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async full-object finalizer did not start")
	}
	select {
	case result := <-readDone:
		require.NoError(t, result.err)
		require.True(t, result.done)
	case <-time.After(diskCacheLifecycleTestTimeout):
		unblock()
		result := <-readDone
		t.Fatalf("full-object read waited for disk finalization: %v", result.err)
	}
	require.Equal(t, []byte("ello"), vector.Entries[0].Data)
	// The streaming helper fills Data but leaves done unchanged for its caller,
	// matching the existing full-object streaming contract.
	require.False(t, vector.Entries[0].done)

	unblock()
	flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
	defer cancel()
	cache.Flush(flushCtx)
	require.NoError(t, flushCtx.Err())
}

func TestS3FSReadIOMergerReleaseMatchesDiskFinalizeMode(t *testing.T) {
	for _, test := range []struct {
		name  string
		async bool
	}{
		{name: "async", async: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := context.Background()
			fs, err := NewS3FS(
				ctx,
				ObjectStorageArguments{
					Name:      "s3",
					Endpoint:  "disk",
					Bucket:    t.TempDir(),
					KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
				},
				CacheConfig{
					DiskPath:     ptrTo(t.TempDir()),
					DiskCapacity: ptrTo[toml.ByteSize](1 << 20),
				},
				nil,
				false,
				false,
			)
			require.NoError(t, err)
			require.NoError(t, fs.Write(ctx, IOVector{
				FilePath: "foo/bar",
				Entries: []IOEntry{{
					Size: 11,
					Data: []byte("hello world"),
				}},
				Policy: SkipDiskCache | SkipMemoryCache,
			}))

			syncStarted, unblock := installBlockedDiskCacheFileSync(fs.diskCache)
			t.Cleanup(func() {
				unblock()
				fs.Close(ctx)
			})

			vector := &IOVector{
				FilePath: "foo/bar",
				Entries:  []IOEntry{{Offset: 1, Size: 4}},
			}
			t.Cleanup(vector.Release)
			mergeKey := fs.readMergeKey(vector)
			readDone := make(chan error, 1)
			go func() { readDone <- fs.Read(ctx, vector) }()
			select {
			case <-syncStarted:
			case <-time.After(diskCacheLifecycleTestTimeout):
				t.Fatal("full-object disk-cache finalization did not start")
			}

			select {
			case err := <-readDone:
				require.NoError(t, err)
			case <-time.After(diskCacheLifecycleTestTimeout):
				unblock()
				t.Fatalf("full-object read waited for disk finalization: %v", <-readDone)
			}
			require.False(t, fs.ioMerger.IsMerging(mergeKey))

			unblock()
			require.Equal(t, []byte("ello"), vector.Entries[0].Data)
			flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
			defer cancel()
			fs.FlushCache(flushCtx)
			require.NoError(t, flushCtx.Err())
			require.False(t, fs.ioMerger.IsMerging(mergeKey))
		})
	}
}

func TestS3FSRangeFollowerDoesNotWaitForAsyncDiskFinalize(t *testing.T) {
	originalShortWait := shortIOWaitDuration
	shortIOWaitDuration = 5 * time.Millisecond
	t.Cleanup(func() { shortIOWaitDuration = originalShortWait })

	var pcSet perfcounter.CounterSet
	ctx := perfcounter.WithCounterSet(context.Background(), &pcSet)
	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    t.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			DiskPath:     ptrTo(t.TempDir()),
			DiskCapacity: ptrTo[toml.ByteSize](1 << 20),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)
	require.NoError(t, fs.Write(ctx, IOVector{
		FilePath: "foo/bar",
		Entries:  []IOEntry{{Size: 11, Data: []byte("hello world")}},
		Policy:   SkipDiskCache | SkipMemoryCache,
	}))

	syncStarted, unblock := installBlockedDiskCacheFileSync(fs.diskCache)
	t.Cleanup(func() {
		unblock()
		fs.Close(ctx)
	})

	newVector := func() *IOVector {
		return &IOVector{
			FilePath: "foo/bar",
			Entries:  []IOEntry{{Offset: 1, Size: 4}},
			Policy:   SkipFullFilePreloads,
		}
	}
	leader := newVector()
	defer leader.Release()
	require.NoError(t, fs.Read(ctx, leader))
	require.Equal(t, []byte("ello"), leader.Entries[0].Data)
	select {
	case <-syncStarted:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("async range finalizer did not start")
	}

	follower := newVector()
	defer follower.Release()
	followerDone := make(chan error, 1)
	go func() { followerDone <- fs.Read(ctx, follower) }()
	select {
	case err := <-followerDone:
		require.NoError(t, err)
	case <-time.After(diskCacheLifecycleTestTimeout):
		unblock()
		<-followerDone
		t.Fatal("range follower waited for async disk finalization")
	}
	require.Equal(t, []byte("ello"), follower.Entries[0].Data)
	require.Equal(t, int64(2), pcSet.FileService.S3.Get.Load())
	diskPath := fs.diskCache.pathForIOEntry("foo/bar", follower.Entries[0])

	unblock()
	// Flush waits for the async finalizer to publish the cache file.  Keep a
	// generous bounded deadline here: the finalizer is deliberately blocked
	// above, and a loaded CI worker may need more than one scheduler quantum
	// after it is released.  The test still fails fast on a genuinely stuck
	// finalizer while avoiding a false timeout caused by CI contention.
	flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
	defer cancel()
	fs.FlushCache(flushCtx)
	require.NoError(t, flushCtx.Err())
	require.FileExists(t, diskPath)
}

func TestS3FSReadFullObjectToDiskCacheStreamingReturnsReaderError(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	fs := &S3FS{
		diskCache: cache,
	}
	vector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Offset: 1,
				Size:   3,
			},
		},
	}
	readerErr := errors.New("reader failed")

	done, err := fs.readFullObjectToDiskCacheStreaming(
		ctx,
		vector,
		"foo/bar",
		getterSource(func(context.Context, *int64, *int64) (io.ReadCloser, error) {
			return &errorReadCloser{
				reader: bytes.NewReader([]byte("hello world")),
				err:    readerErr,
			}, nil
		}),
	)

	// The read has its bytes before the reader fails; the fill caches nothing.
	assert.True(t, done)
	assert.NoError(t, err)
	assert.Equal(t, []byte("ell"), vector.Entries[0].Data)
	cache.Flush(ctx)
	assert.Empty(t, cache.objectFiles("foo/bar"))
}

// streamFill runs readFullObjectToDiskCacheStreaming of foo/bar bytes 1-3 in the
// background with an async disk cache, reading the object from getReader.
func streamFill(
	t *testing.T,
	ctx context.Context,
	getReader func(context.Context, *int64, *int64) (io.ReadCloser, error),
) (*DiskCache, *IOVector, chan error) {
	cache, err := NewDiskCache(context.Background(), t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	t.Cleanup(func() { cache.Close(context.Background()) })
	s3 := &S3FS{diskCache: cache}
	vector := &IOVector{FilePath: "foo/bar", Entries: []IOEntry{{Offset: 1, Size: 3}}}
	done := make(chan error, 1)
	go func() {
		ok, err := s3.readFullObjectToDiskCacheStreaming(ctx, vector, "foo/bar", getterSource(getReader))
		if err == nil && !ok {
			err = errors.New("not done")
		}
		done <- err
	}()
	return cache, vector, done
}

func waitResult(t *testing.T, done chan error) error {
	select {
	case err := <-done:
		return err
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("read did not return")
		return nil
	}
}

// A reader whose ranges are captured returns while the rest of the object still
// downloads, and the download then caches the whole object.
func TestS3FSReadFullObjectToDiskCacheStreamingReturnsOnceCaptured(t *testing.T) {
	ctx := context.Background()
	pr, pw := io.Pipe()
	release := make(chan struct{})
	go func() {
		_, _ = pw.Write([]byte("hello"))
		<-release
		_, _ = pw.Write([]byte(" world"))
		_ = pw.Close()
	}()
	cache, vector, done := streamFill(t, ctx, func(context.Context, *int64, *int64) (io.ReadCloser, error) {
		return pr, nil
	})

	require.NoError(t, waitResult(t, done))
	require.Equal(t, []byte("ell"), vector.Entries[0].Data)

	close(release)
	flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
	defer cancel()
	cache.Flush(flushCtx)
	require.True(t, cache.cache.Contains(cache.wholePath("foo/bar", 11)))
}

// A download that fails after the ranges were captured does not fail the read, and
// caches nothing.
func TestS3FSReadFullObjectToDiskCacheStreamingErrorAfterCapture(t *testing.T) {
	ctx := context.Background()
	cache, vector, done := streamFill(t, ctx, func(context.Context, *int64, *int64) (io.ReadCloser, error) {
		return &errorReadCloser{reader: bytes.NewReader([]byte("hello world")), err: errors.New("reader failed")}, nil
	})

	require.NoError(t, waitResult(t, done))
	require.Equal(t, []byte("ell"), vector.Entries[0].Data)
	flushCtx, cancel := context.WithTimeout(ctx, diskCacheLifecycleTestTimeout)
	defer cancel()
	cache.Flush(flushCtx)
	require.Empty(t, cache.objectFiles("foo/bar"))
}

// Cancelling the read before its ranges are captured cancels the download.
func TestS3FSReadFullObjectToDiskCacheStreamingCancelBeforeCapture(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	pr, pw := io.Pipe()
	go func() { _, _ = pw.Write([]byte("h")) }()
	cache, _, done := streamFill(t, ctx, func(fillCtx context.Context, _ *int64, _ *int64) (io.ReadCloser, error) {
		go func() {
			<-fillCtx.Done()
			_ = pw.CloseWithError(fillCtx.Err())
		}()
		return pr, nil
	})

	cancel()
	require.ErrorIs(t, waitResult(t, done), context.Canceled)
	require.Empty(t, cache.objectFiles("foo/bar"))
}

// With every background slot taken, a reader waits for its download as before.
func TestS3FSReadFullObjectToDiskCacheStreamingWaitsWhenBackgroundFull(t *testing.T) {
	require.Eventually(t, func() bool { return len(backgroundFills) == 0 },
		diskCacheLifecycleTestTimeout, time.Millisecond, "fills of earlier tests released their slots")
	for i := 0; i < cap(backgroundFills); i++ {
		backgroundFills <- struct{}{}
	}
	t.Cleanup(func() {
		for i := 0; i < cap(backgroundFills); i++ {
			<-backgroundFills
		}
	})
	ctx := context.Background()
	pr, pw := io.Pipe()
	release := make(chan struct{})
	go func() {
		_, _ = pw.Write([]byte("hello"))
		<-release
		_, _ = pw.Write([]byte(" world"))
		_ = pw.Close()
	}()
	_, vector, done := streamFill(t, ctx, func(context.Context, *int64, *int64) (io.ReadCloser, error) {
		return pr, nil
	})

	select {
	case err := <-done:
		t.Fatalf("read returned before its download ended: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	require.NoError(t, waitResult(t, done))
	require.Equal(t, []byte("ell"), vector.Entries[0].Data)
}

func TestS3FSReadFullObjectToDiskCacheStreamingReadToEnd(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	fs := &S3FS{
		diskCache: cache,
	}
	vector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Offset: 6,
				Size:   -1,
			},
		},
	}

	done, err := fs.readFullObjectToDiskCacheStreaming(
		ctx,
		vector,
		"foo/bar",
		getterSource(func(context.Context, *int64, *int64) (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader([]byte("hello world"))), nil
		}),
	)

	assert.True(t, done)
	assert.Nil(t, err)
	assert.False(t, vector.Entries[0].done)
	assert.Equal(t, []byte("world"), vector.Entries[0].Data)
	assert.Equal(t, int64(5), vector.Entries[0].Size)
}

func TestS3FSReadFullObjectToDiskCacheStreamingUnexpectedEOF(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	fs := &S3FS{
		diskCache: cache,
	}
	vector := &IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Offset: 6,
				Size:   20,
			},
		},
	}

	done, err := fs.readFullObjectToDiskCacheStreaming(
		ctx,
		vector,
		"foo/bar",
		getterSource(func(context.Context, *int64, *int64) (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader([]byte("hello world"))), nil
		}),
	)

	assert.True(t, done)
	assert.True(t, moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF))
	assert.False(t, vector.Entries[0].done)
	assert.Nil(t, vector.Entries[0].Data)
	assert.Equal(t, int64(20), vector.Entries[0].Size)
}

func TestS3FSShouldStreamFullObjectToDiskCacheExclusions(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	assert.Nil(t, err)
	defer cache.Close(ctx)

	fs := &S3FS{
		diskCache: cache,
	}

	assert.False(t, fs.shouldStreamFullObjectToDiskCache(&IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size: 0,
			},
		},
	}))

	buf := new(bytes.Buffer)
	assert.False(t, fs.shouldStreamFullObjectToDiskCache(&IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size:          3,
				WriterForRead: buf,
			},
		},
	}))

	var reader io.ReadCloser
	assert.False(t, fs.shouldStreamFullObjectToDiskCache(&IOVector{
		FilePath: "foo/bar",
		Entries: []IOEntry{
			{
				Size:              3,
				ReadCloserForRead: &reader,
			},
		},
	}))
}

func TestS3FSWillCacheFullObject(t *testing.T) {
	ctx := context.Background()
	cache, err := NewDiskCache(ctx, t.TempDir(), fscache.ConstCapacity(1<<20), nil, false, nil, "")
	require.NoError(t, err)
	t.Cleanup(func() { cache.Close(ctx) })
	fs := &S3FS{diskCache: cache}

	writer := new(bytes.Buffer)
	var reader io.ReadCloser
	tests := []struct {
		name   string
		vector *IOVector
		want   bool
	}{
		{
			name:   "ordinary entry",
			vector: &IOVector{Entries: []IOEntry{{Size: 1}}},
			want:   true,
		},
		{
			name:   "writer only",
			vector: &IOVector{Entries: []IOEntry{{Size: 1, WriterForRead: writer}}},
		},
		{
			name:   "read closer only",
			vector: &IOVector{Entries: []IOEntry{{Size: 1, ReadCloserForRead: &reader}}},
		},
		{
			name: "mixed ordinary and writer",
			vector: &IOVector{Entries: []IOEntry{
				{Size: 1, WriterForRead: writer},
				{Offset: 1, Size: 1},
			}},
			want: true,
		},
		{
			name: "zero-sized entry prevents publication",
			vector: &IOVector{Entries: []IOEntry{
				{Size: 1},
				{Offset: 1, Size: 0},
			}},
		},
		{
			name: "disk cache writes disabled",
			vector: &IOVector{
				Policy:  SkipDiskCacheWrites,
				Entries: []IOEntry{{Size: 1}},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			require.Equal(t, test.want, fs.willCacheFullObject(test.vector))
		})
	}
}

type errorReadCloser struct {
	reader *bytes.Reader
	err    error
}

func (e *errorReadCloser) Read(p []byte) (int, error) {
	n, err := e.reader.Read(p)
	if err != nil {
		return n, e.err
	}
	return n, nil
}

func (e *errorReadCloser) Close() error {
	return nil
}

type S3CredentialTestCase struct {
	Skip bool
	ObjectStorageArguments
}

var s3CredentialTestCases = func() []S3CredentialTestCase {
	content, err := os.ReadFile("s3_fs_test_new.xml")
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		panic(err)
	}
	var spec struct {
		XMLName xml.Name               `xml:"Spec"`
		Cases   []S3CredentialTestCase `xml:"Case"`
	}
	err = xml.Unmarshal(content, &spec)
	if err != nil {
		panic(err)
	}
	return spec.Cases
}()

func TestNewS3FSFromSpec(t *testing.T) {
	if len(s3CredentialTestCases) == 0 {
		t.Skip("no case")
	}

	for _, kase := range s3CredentialTestCases {
		if kase.Skip {
			continue
		}

		t.Run(kase.Name, func(t *testing.T) {

			ctx := context.Background()
			fs, err := NewS3FS(
				ctx,
				kase.ObjectStorageArguments,
				DisabledCacheConfig,
				nil,
				true,
				false,
			)
			assert.Nil(t, err)
			_ = fs

		})

		t.Run(kase.Name+" bad bucket", func(t *testing.T) {

			args := kase.ObjectStorageArguments
			args.Bucket = args.Bucket + "foobarbaz"
			ctx := context.Background()
			_, err := NewS3FS(
				ctx,
				args,
				DisabledCacheConfig,
				nil,
				true,
				false,
			)
			if err == nil {
				t.Fatal("should fail")
			}

		})
	}

}

func TestNewS3NoDefaultCredential(t *testing.T) {
	ctx := context.Background()
	_, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Endpoint: "aliyuncs.com",
		},
		DisabledCacheConfig,
		nil,
		true,
		true,
	)
	assert.True(t, moerr.IsMoErrCode(err, moerr.ErrInvalidInput))
	assert.True(t, strings.Contains(err.Error(), "no valid credentials"))
}

func TestS3FSIOMerger(t *testing.T) {
	ctx := context.Background()

	fs, err := NewS3FS(
		context.Background(),
		ObjectStorageArguments{
			Endpoint: "disk",
			Bucket:   t.TempDir(),
		},
		CacheConfig{
			MemoryCapacity: ptrTo(toml.ByteSize(1 << 20)),
		},
		nil,
		false,
		false,
	)
	require.NoError(t, err)

	err = fs.Write(ctx, IOVector{
		FilePath: "foo",
		Policy:   SkipDiskCache, // skip disk cache
		Entries: []IOEntry{
			{
				Data: []byte("foo"),
				Size: 3,
			},
		},
	})
	require.NoError(t, err)

	cache := &blockingDataCache{
		memoryCacheBacking: fs.memCache.cache,
		updateStarted:      make(chan struct{}),
		releaseUpdate:      make(chan struct{}),
	}
	fs.memCache.cache = cache
	storage := &blockingReadObjectStorage{
		ObjectStorage: fs.server.raw,
		readStarted:   make(chan struct{}),
		releaseRead:   make(chan struct{}),
	}
	fs.server.raw = storage
	readCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	releaseRead := sync.OnceFunc(func() { close(storage.releaseRead) })
	releaseUpdate := sync.OnceFunc(func() { close(cache.releaseUpdate) })
	var wg sync.WaitGroup
	t.Cleanup(func() {
		cancel()
		releaseRead()
		releaseUpdate()
		wg.Wait()
		fs.Close(ctx)
	})

	newReadVector := func() *IOVector {
		return &IOVector{
			FilePath: "foo",
			Policy:   SkipFullFilePreloads,
			Entries: []IOEntry{
				{
					Size:        3,
					ToCacheData: CacheOriginalData,
				},
			},
		}
	}
	read := func(ctx context.Context) error {
		vec := newReadVector()
		defer vec.Release()
		if err := fs.Read(ctx, vec); err != nil {
			return err
		}
		if !bytes.Equal([]byte("foo"), vec.Entries[0].CachedData.Bytes()) {
			return errors.New("unexpected read data")
		}
		return nil
	}

	results := make(chan error, 2)
	startRead := func(ctx context.Context) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results <- read(ctx)
		}()
	}
	waitForSignal := func(ch <-chan struct{}, name string) {
		t.Helper()
		select {
		case <-ch:
		case <-readCtx.Done():
			t.Fatalf("timed out waiting for %s: %v", name, readCtx.Err())
		}
	}

	startRead(readCtx)
	waitForSignal(storage.readStarted, "object read")
	releaseRead()
	waitForSignal(cache.updateStarted, "cache update")

	// The cache update is deliberately blocked. The merge must still be active
	// at this point, otherwise a waiter can become a second merge leader before
	// the first leader's cache contents are visible. This assertion is
	// deterministically false with defer done() at the Merge call site.
	require.True(t, fs.ioMerger.IsMerging(fs.readMergeKey(newReadVector())))

	waiterCtx := WithEventLogger(readCtx)
	startRead(waiterCtx)
	waiterLogger := waiterCtx.Value(EventLoggerKey).(*eventLogger)
	require.Eventually(t, func() bool {
		waiterLogger.mu.Lock()
		defer waiterLogger.mu.Unlock()
		for _, ev := range *waiterLogger.events {
			if ev.ev == str_ioMerger_Merge_wait {
				return true
			}
		}
		return false
	}, 5*time.Second, time.Millisecond)

	releaseUpdate()
	require.NoError(t, <-results)
	require.NoError(t, <-results)
	require.Equal(t, int64(1), storage.readCount.Load())
}

func TestNewS3FSCacheInitializationFailureRollsBackMemoryCache(t *testing.T) {
	ctx := context.Background()
	cachePath := t.TempDir() + "/not-a-directory"
	require.NoError(t, os.WriteFile(cachePath, nil, 0o644))
	name := t.Name()

	fs, err := NewS3FS(
		ctx,
		ObjectStorageArguments{
			Name:     name,
			Endpoint: "disk",
			Bucket:   t.TempDir(),
		},
		CacheConfig{
			MemoryCapacity: ptrTo(toml.ByteSize(1 << 20)),
			DiskCapacity:   ptrTo(toml.ByteSize(1 << 20)),
			DiskPath:       &cachePath,
		},
		nil,
		false,
		false,
	)
	require.Nil(t, fs)
	require.Error(t, err)

	registered := false
	allMemoryCaches.Range(func(_, value any) bool {
		if value.(memoryCacheRegistration).name == name {
			registered = true
			return false
		}
		return true
	})
	require.False(t, registered)
}

func BenchmarkS3FSAllocateCacheData(b *testing.B) {
	ctx := context.Background()

	fs, err := NewS3FS(
		context.Background(),
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    b.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			MemoryCapacity: ptrTo[toml.ByteSize](128 * 1024),
		},
		nil,
		false,
		false,
	)
	assert.Nil(b, err)
	defer fs.Close(ctx)

	benchmarkFileServiceAllocateCacheData(b, fs.AllocateCacheData, 42, 1)
}

func BenchmarkS3FSAllocateCacheDataHighCardinality(b *testing.B) {
	ctx := context.Background()

	fs, err := NewS3FS(
		context.Background(),
		ObjectStorageArguments{
			Name:      "s3",
			Endpoint:  "disk",
			Bucket:    b.TempDir(),
			KeyPrefix: time.Now().Format("2006-01-02.15:04:05.000000"),
		},
		CacheConfig{
			MemoryCapacity: ptrTo[toml.ByteSize](128 * 1024),
		},
		nil,
		false,
		false,
	)
	assert.Nil(b, err)
	defer fs.Close(ctx)

	benchmarkFileServiceAllocateCacheData(b, fs.AllocateCacheData, 1, 1024)
}

func TestS3FSFromSpecs(t *testing.T) {

	for _, args := range objectStorageArgumentsForTest("test", t) {

		t.Run(args.Name, func(t *testing.T) {
			csPrefix := args.Name == "cs-s3"
			fixturePrefix := args.KeyPrefix
			if csPrefix {
				prepareCITestStorage(t, args)
			}

			testFileServiceWithContext(t, 0, func(ctx context.Context, name string) FileService {
				args.Name = name
				if csPrefix {
					args.KeyPrefix = path.Join(fixturePrefix, time.Now().Format("2006-01-02.15:04:05.000000"))
				} else {
					args.KeyPrefix = time.Now().Format("2006-01-02.15:04:05.000000")
				}

				fs, err := NewS3FS(
					ctx,
					args,
					DisabledCacheConfig,
					nil,
					true,
					false,
				)
				if err != nil {
					logutil.Error("S3FS errror",
						zap.Any("args", args),
						zap.Any("error", err),
					)
					t.Fatal(err)
				}

				return fs
			})

		})

	}

}

func TestS3FSReadWrapperOwnership(t *testing.T) {
	t.Run("stream-retry", func(t *testing.T) {
		var opens int
		var closes [2]int
		closeErr := errors.New("provider close failed")
		source := &readTestObjectStorage{open: func(ctx context.Context, key string, min, max *int64, expected *ObjectIdentity) (io.ReadCloser, error) {
			require.Equal(t, true, ctx.Value(httpTraceKey{}))
			require.Equal(t, "key", key)
			require.Nil(t, expected)
			return newRetryableReader(func(offset int64) (io.ReadCloser, error) {
				i := opens
				opens++
				require.Equal(t, int64(0), offset)
				if i == 0 {
					return &readCloser{r: readerFunc(func([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }), closeFunc: func() error { closes[0]++; return nil }}, nil
				}
				return &readCloser{r: strings.NewReader("abc"), closeFunc: func() error { closes[1]++; return closeErr }}, nil
			}, 0, func(err error) bool { return errors.Is(err, io.ErrUnexpectedEOF) })
		}}
		fs := &S3FS{name: "test", server: newObjectStorageServer(source, minRequestRoom), rawStorage: source, ioMerger: NewIOMerger()}
		var reader io.ReadCloser
		vector := IOVector{FilePath: "test:key", Policy: SkipAllCache | SkipFullFilePreloads, Entries: []IOEntry{{Size: 3, ReadCloserForRead: &reader}}}
		t.Cleanup(vector.Release)
		require.NoError(t, fs.Read(t.Context(), &vector))
		require.NotNil(t, reader)
		t.Cleanup(func() { _ = reader.Close() })
		data, err := io.ReadAll(reader)
		require.NoError(t, err)
		require.Equal(t, "abc", string(data))
		require.Equal(t, 2, opens)
		require.Zero(t, heldRoom(fs.server))
		require.ErrorIs(t, reader.Close(), closeErr)
		require.ErrorIs(t, reader.Close(), closeErr)
		require.Equal(t, [2]int{1, 1}, closes)
	})
	t.Run("conditional-lease", func(t *testing.T) {
		var closes int
		closeErr := error(ErrObjectChanged)
		source := &readTestObjectStorage{open: func(ctx context.Context, key string, min, max *int64, expected *ObjectIdentity) (io.ReadCloser, error) {
			require.Equal(t, true, ctx.Value(httpTraceKey{}))
			require.Equal(t, "key", key)
			require.Equal(t, int64(1), *min)
			require.Equal(t, int64(4), *max)
			require.Equal(t, &ObjectIdentity{ETag: "version1", Size: 8}, expected)
			return &readCloser{r: strings.NewReader("abc"), closeFunc: func() error { closes++; return closeErr }}, nil
		}}
		fs := &S3FS{name: "test", server: newObjectStorageServer(source, minRequestRoom), rawStorage: source}
		reader := NewLeasedRangeReader(fs).(ConditionalLeasedRangeReader)
		identity, err := reader.StatIdentity(t.Context(), "test:key")
		require.NoError(t, err)
		admission := new(testRangeAdmission)
		lease, err := reader.ReadRangeLeaseWithIdentity(t.Context(), "test:key", 1, 3, identity, admission)
		require.Nil(t, lease)
		require.ErrorIs(t, err, ErrObjectChanged)
		require.Zero(t, admission.committed.Load())
		require.Equal(t, int64(3), admission.aborted.Load())
		require.Zero(t, heldRoom(fs.server))
		require.Equal(t, 1, closes)
		closeErr = nil
		lease, err = reader.ReadRangeLeaseWithIdentity(t.Context(), "test:key", 1, 3, identity, admission)
		require.NoError(t, err)
		t.Cleanup(lease.Release)
		require.Equal(t, "abc", string(lease.Bytes()))
		require.Equal(t, int64(3), admission.committed.Load())
		require.Equal(t, 2, closes)
		lease.Release()
		require.Equal(t, int64(3), admission.released.Load())
		require.Zero(t, heldRoom(fs.server))
	})
}

// With every background slot taken, cancelling a reader that waits for its download
// cancels the download and returns.
func TestS3FSReadFullObjectToDiskCacheStreamingCancelWhileWaiting(t *testing.T) {
	require.Eventually(t, func() bool { return len(backgroundFills) == 0 },
		diskCacheLifecycleTestTimeout, time.Millisecond, "fills of earlier tests released their slots")
	for i := 0; i < cap(backgroundFills); i++ {
		backgroundFills <- struct{}{}
	}
	t.Cleanup(func() {
		for i := 0; i < cap(backgroundFills); i++ {
			<-backgroundFills
		}
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pr, pw := io.Pipe()
	go func() { _, _ = pw.Write([]byte("hello")) }()
	cache, _, done := streamFill(t, ctx, func(fillCtx context.Context, _ *int64, _ *int64) (io.ReadCloser, error) {
		go func() {
			<-fillCtx.Done()
			_ = pw.CloseWithError(fillCtx.Err())
		}()
		return pr, nil
	})

	select {
	case err := <-done:
		t.Fatalf("read returned before its download ended: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	cancel()
	require.Error(t, waitResult(t, done))
	require.Empty(t, cache.objectFiles("foo/bar"))
}

// With every background slot taken, a download that fails after the reader captured its
// bytes does not fail the read.
func TestS3FSReadFullObjectToDiskCacheStreamingErrorAfterCaptureWhenBackgroundFull(t *testing.T) {
	require.Eventually(t, func() bool { return len(backgroundFills) == 0 },
		diskCacheLifecycleTestTimeout, time.Millisecond, "fills of earlier tests released their slots")
	for i := 0; i < cap(backgroundFills); i++ {
		backgroundFills <- struct{}{}
	}
	t.Cleanup(func() {
		for i := 0; i < cap(backgroundFills); i++ {
			<-backgroundFills
		}
	})
	pr, pw := io.Pipe()
	go func() { _, _ = pw.Write([]byte("hello")) }()
	_, vector, done := streamFill(t, context.Background(), func(context.Context, *int64, *int64) (io.ReadCloser, error) {
		return pr, nil
	})

	select {
	case err := <-done:
		t.Fatalf("read returned before its download ended: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	require.NoError(t, pw.CloseWithError(errors.New("source failed")))
	require.NoError(t, waitResult(t, done))
	require.Equal(t, []byte("ell"), vector.Entries[0].Data)
}

// A download continuing after its reader returned keeps the reader's deadline and ends
// when the disk cache closes.
func TestS3FSReadFullObjectToDiskCacheStreamingBackgroundEndsOnClose(t *testing.T) {
	require.Eventually(t, func() bool { return len(backgroundFills) == 0 },
		diskCacheLifecycleTestTimeout, time.Millisecond, "fills of earlier tests released their slots")
	ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
	defer cancel()
	deadline, _ := ctx.Deadline()
	pr, pw := io.Pipe()
	go func() { _, _ = pw.Write([]byte("hello")) }()
	fillDeadline := make(chan time.Time, 1)
	fillDone := make(chan struct{})
	cache, vector, done := streamFill(t, ctx, func(fillCtx context.Context, _ *int64, _ *int64) (io.ReadCloser, error) {
		d, _ := fillCtx.Deadline()
		fillDeadline <- d
		go func() {
			<-fillCtx.Done()
			close(fillDone)
			_ = pw.CloseWithError(fillCtx.Err())
		}()
		return pr, nil
	})

	require.NoError(t, waitResult(t, done))
	require.Equal(t, []byte("ell"), vector.Entries[0].Data)
	require.Equal(t, deadline, <-fillDeadline)
	cache.Close(ctx)
	select {
	case <-fillDone:
	case <-time.After(diskCacheLifecycleTestTimeout):
		t.Fatal("the download outlived the disk cache")
	}
}

func TestS3FSOrdinarySparseRangeRead(t *testing.T) {
	for _, tc := range []struct {
		name    string
		offsets []int64
		reads   int
	}{
		{"sparse", []int64{0, 16 << 20}, 2},
		{"adjacent", []int64{0, 4096}, 1},
		{"exact span threshold", []int64{0, minExpensiveRangeReadSpan - 4096}, 1},
		{"above span threshold", []int64{0, minExpensiveRangeReadSpan - 4095}, 2},
		{"out of order", []int64{16 << 20, 0}, 1},
		{"four sparse", []int64{0, 16 << 20, 32 << 20, 48 << 20}, 4},
		{"five sparse", []int64{0, 16 << 20, 32 << 20, 48 << 20, 64 << 20}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
			require.NoError(t, err)
			t.Cleanup(func() { fs.Close(context.Background()) })
			storage := &generatedRangeObjectStorage{}
			fs.server.raw = storage
			vector := &IOVector{FilePath: "foo", Policy: SkipAllCache}
			for _, offset := range tc.offsets {
				vector.Entries = append(vector.Entries, IOEntry{Offset: offset, Size: 4096})
			}
			t.Cleanup(vector.Release)
			require.NoError(t, fs.Read(ctx, vector))
			reads := storage.readRanges()
			require.Len(t, reads, tc.reads)
			if tc.reads > 1 {
				for i, read := range reads {
					require.Equal(t, tc.offsets[i], *read.min)
					require.Equal(t, tc.offsets[i]+4096, *read.max)
				}
			}
			for _, entry := range vector.Entries {
				require.Equal(t, make([]byte, 4096), entry.Data)
			}
		})
	}
}

var errTestHDFSSeek = errors.New("test HDFS seek beyond EOF")

// hdfsSeekReader models the bounds check in colinmarc/hdfs FileReader.Seek.
type hdfsSeekReader struct {
	size, offset int64
}

func (r *hdfsSeekReader) Seek(offset int64, whence int) (int64, error) {
	if whence != io.SeekStart || offset < 0 || offset > r.size {
		return r.offset, errTestHDFSSeek
	}
	r.offset = offset
	return offset, nil
}

func (r *hdfsSeekReader) Read(p []byte) (int, error) {
	if r.offset == r.size {
		return 0, io.EOF
	}
	n := int(min(int64(len(p)), r.size-r.offset))
	clear(p[:n])
	r.offset += int64(n)
	return n, nil
}

type hdfsSeekStorage struct {
	dummyObjectStorage
	size   int64
	ranges [][2]int64
	closes int
}

func (s *hdfsSeekStorage) Read(_ context.Context, key string, min, max *int64) (io.ReadCloser, error) {
	s.ranges = append(s.ranges, [2]int64{*min, *max})
	reader, err := readObjectRange(&hdfsSeekReader{size: s.size}, key, min, max)
	if err != nil {
		s.closes++ // HDFS.Read closes its opened file on an error.
		return nil, err
	}
	return &readCloser{r: reader, closeFunc: func() error { s.closes++; return nil }}, nil
}

func TestS3FSOrdinarySparseHDFSSeekBounds(t *testing.T) {
	const far = int64(16 << 20)
	for _, tc := range []struct {
		name       string
		hdfs       bool
		size       int64
		offsets    [2]int64
		wantRanges [][2]int64
		wantErr    error
		short      bool
	}{
		{"later beyond EOF", true, 8192, [2]int64{0, far}, [][2]int64{{0, far + 4096}}, nil, true},
		{"first beyond EOF", true, 8192, [2]int64{far, 2 * far}, [][2]int64{{far, 2*far + 4096}}, errTestHDFSSeek, false},
		{"first at EOF", true, 8192, [2]int64{8192, far}, [][2]int64{{8192, far + 4096}}, nil, true},
		{"success", true, far + 4096, [2]int64{0, far}, [][2]int64{{0, far + 4096}}, nil, false},
		{"non-HDFS control", false, 8192, [2]int64{0, far}, [][2]int64{{0, 4096}, {far, far + 4096}}, errTestHDFSSeek, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
			require.NoError(t, err)
			t.Cleanup(func() { fs.Close(context.Background()) })
			storage := &hdfsSeekStorage{size: tc.size}
			fs.server.raw = storage
			if tc.hdfs {
				fs.rawStorage = &HDFS{}
			} else {
				fs.rawStorage = storage
			}
			cache := &publicationVectorCache{}
			vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Caches: []IOVectorCache{cache}, Entries: []IOEntry{
				{Offset: tc.offsets[0], Size: 4096}, {Offset: tc.offsets[1], Size: 4096},
			}}
			t.Cleanup(vector.Release)
			err = fs.Read(ctx, vector)
			switch {
			case tc.short:
				require.True(t, moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF), "%v", err)
			case tc.wantErr != nil:
				require.ErrorIs(t, err, tc.wantErr)
			default:
				require.NoError(t, err)
				for _, entry := range vector.Entries {
					require.Equal(t, make([]byte, 4096), entry.Data)
				}
			}
			require.Equal(t, tc.wantRanges, storage.ranges)
			require.Equal(t, len(tc.wantRanges), storage.closes)
			if err != nil {
				require.Zero(t, cache.updates.Load())
			} else {
				require.EqualValues(t, 1, cache.updates.Load())
			}
		})
	}
	t.Run("partial cache with two misses", func(t *testing.T) {
		ctx := t.Context()
		fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
		require.NoError(t, err)
		t.Cleanup(func() { fs.Close(context.Background()) })
		storage := &hdfsSeekStorage{size: 8192}
		fs.server.raw, fs.rawStorage = storage, &HDFS{}
		hit := &sparseFirstEntryCache{}
		miss := &publicationVectorCache{}
		vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Caches: []IOVectorCache{hit, miss}, Entries: []IOEntry{
			{Offset: 0, Size: 4096}, {Offset: 4096, Size: 4096}, {Offset: far, Size: 4096},
		}}
		t.Cleanup(vector.Release)
		err = fs.Read(ctx, vector)
		require.True(t, moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF), "%v", err)
		require.Equal(t, []byte("cached"), vector.Entries[0].Data)
		require.Equal(t, [][2]int64{{4096, far + 4096}}, storage.ranges)
		require.Equal(t, 1, storage.closes)
		require.EqualValues(t, 1, hit.reads.Load())
		require.EqualValues(t, 1, miss.reads.Load())
		require.Zero(t, hit.updates.Load())
		require.Zero(t, miss.updates.Load())
	})
}

type sparseFailureStorage struct {
	dummyObjectStorage
	calls int
	err   error
}

func (s *sparseFailureStorage) Read(_ context.Context, _ string, min, max *int64) (io.ReadCloser, error) {
	s.calls++
	if s.calls == 2 {
		return nil, s.err
	}
	return io.NopCloser(bytes.NewReader(make([]byte, *max-*min))), nil
}

type rangeStatusError struct{}

func (rangeStatusError) Error() string       { return "range beyond object" }
func (rangeStatusError) HTTPStatusCode() int { return 416 }

func TestS3FSSparseRangeFailure(t *testing.T) {
	ctx := t.Context()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	storage := &sparseFailureStorage{err: rangeStatusError{}}
	fs.server.raw = storage
	cache := &publicationVectorCache{}
	vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Caches: []IOVectorCache{cache}, Entries: []IOEntry{
		{Offset: 0, Size: 4096}, {Offset: 16 << 20, Size: 4096},
	}}
	t.Cleanup(vector.Release)
	err = fs.Read(ctx, vector)
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF), "%v", err)
	require.Equal(t, 2, storage.calls)
	require.Zero(t, cache.updates.Load())
}

func TestS3FSSparseFailureDoesNotPopulateMemoryCache(t *testing.T) {
	ctx := t.Context()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()},
		CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](1 << 20)}, nil, false, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	storage := &sparseFailureStorage{err: rangeStatusError{}}
	fs.server.raw = storage
	entry := func(offset int64) IOEntry {
		return IOEntry{Offset: offset, Size: 4096, ToCacheData: CacheOriginalData}
	}
	vector := &IOVector{FilePath: "foo", Entries: []IOEntry{entry(0), entry(16 << 20)}}
	t.Cleanup(vector.Release)
	require.Error(t, fs.Read(ctx, vector))
	cached := &IOVector{FilePath: "foo", Entries: []IOEntry{entry(0)}}
	t.Cleanup(cached.Release)
	require.NoError(t, fs.ReadCache(ctx, cached))
	require.False(t, cached.allDone())
}

func TestS3FSSparseFailureDoesNotPopulateDiskCache(t *testing.T) {
	ctx := t.Context()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()},
		CacheConfig{DiskCapacity: ptrTo[toml.ByteSize](1 << 20), DiskPath: ptrTo(t.TempDir())}, nil, false, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	fs.diskCache.prefetchRange = prefetchRangeConfig{}
	fs.server.raw = &sparseFailureStorage{err: rangeStatusError{}}
	entry := func(offset int64) IOEntry { return IOEntry{Offset: offset, Size: 4096, ToCacheData: CacheOriginalData} }
	vector := &IOVector{FilePath: "foo", Policy: SkipFullFilePreloads, Entries: []IOEntry{entry(0), entry(16 << 20)}}
	t.Cleanup(vector.Release)
	require.Error(t, fs.Read(ctx, vector))
	cached := &IOVector{FilePath: "foo", Policy: SkipFullFilePreloads, Entries: []IOEntry{entry(0)}}
	t.Cleanup(cached.Release)
	require.NoError(t, readCache(ctx, fs.diskCache, cached))
	require.False(t, cached.allDone())
}

func TestS3FSSparsePreservesConverterError(t *testing.T) {
	ctx := t.Context()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	fs.server.raw = &generatedRangeObjectStorage{}
	vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Entries: []IOEntry{
		{Offset: 0, Size: 4096},
		{Offset: 16 << 20, Size: 4096, ToCacheData: func(context.Context, io.Reader, []byte, CacheDataAllocator) (fscache.Data, error) {
			return nil, rangeStatusError{}
		}},
	}}
	t.Cleanup(vector.Release)
	err = fs.Read(ctx, vector)
	status, ok := objectStorageStatusCode(err)
	require.True(t, ok, "%v", err)
	require.Equal(t, 416, status)
}

type sparsePatternReader struct{ value byte }

func (r sparsePatternReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = r.value
	}
	return len(p), nil
}

type sparseTrackingStorage struct {
	dummyObjectStorage
	// mu guards the counters, which a read's prefetch object may update after the read.
	mu                  sync.Mutex
	opens, closes       int
	offsets             []int64
	setFileEvents       int
	bytes               int64
	firstErr, secondErr error
	shortSecond         bool
	bodyErrSecond       bool
	cancel              context.CancelFunc
}

func (s *sparseTrackingStorage) Read(ctx context.Context, _ string, min, max *int64) (io.ReadCloser, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.opens == 0 {
		s.setFileEvents = countLoggedEvents(ctx, str_disk_cache_setfile_begin)
	}
	s.opens++
	s.offsets = append(s.offsets, *min)
	if s.opens == 1 && s.firstErr != nil {
		return nil, s.firstErr
	}
	if s.opens == 2 && s.secondErr != nil {
		return nil, s.secondErr
	}
	n := *max - *min
	if s.opens == 2 && s.shortSecond {
		n--
	}
	s.bytes += n
	var body io.Reader = io.LimitReader(sparsePatternReader{value: byte(*min/(16<<20) + 1)}, n)
	if s.opens == 2 && s.bodyErrSecond {
		body = readerFunc(func([]byte) (int, error) { return 0, rangeStatusError{} })
	}
	return &readCloser{r: body, closeFunc: func() error {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.closes++
		if s.cancel != nil {
			s.cancel()
			s.cancel = nil
		}
		return nil
	}}, nil
}

// counts returns the opens, closes and opened offsets so far.
func (s *sparseTrackingStorage) counts() (opens, closes int, offsets []int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.opens, s.closes, slices.Clone(s.offsets)
}

type sparseReleaseCountingData struct {
	fscache.Data
	releases *int
}

func (d *sparseReleaseCountingData) Release() { *d.releases++; d.Data.Release() }

func TestS3FSSparseReadLifecycle(t *testing.T) {
	for _, tc := range []struct {
		name                string
		firstErr, secondErr error
		short, cancel       bool
		opens, closes       int
	}{
		{name: "first 416", firstErr: rangeStatusError{}, opens: 1},
		{name: "second 416", secondErr: rangeStatusError{}, opens: 2, closes: 1},
		{name: "short body", short: true, opens: 2, closes: 2},
		{name: "cancel between opens", cancel: true, opens: 1, closes: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
			require.NoError(t, err)
			t.Cleanup(func() { fs.Close(context.Background()) })
			storage := &sparseTrackingStorage{firstErr: tc.firstErr, secondErr: tc.secondErr, shortSecond: tc.short}
			if tc.cancel {
				storage.cancel = cancel
			}
			fs.server.raw = storage
			releases := 0
			convert := func(ctx context.Context, r io.Reader, data []byte, allocator CacheDataAllocator) (fscache.Data, error) {
				d, err := CacheOriginalData(ctx, r, data, allocator)
				if err != nil {
					return nil, err
				}
				return &sparseReleaseCountingData{Data: d, releases: &releases}, nil
			}
			vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Entries: []IOEntry{
				{Offset: 0, Size: 4096, ToCacheData: convert}, {Offset: 16 << 20, Size: 4096},
			}}
			err = fs.Read(ctx, vector)
			require.Error(t, err)
			if tc.cancel {
				require.ErrorIs(t, err, context.Canceled)
			} else if tc.firstErr != nil {
				status, ok := objectStorageStatusCode(err)
				require.True(t, ok)
				require.Equal(t, 416, status)
			} else {
				require.True(t, moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF), "%v", err)
			}
			require.Equal(t, tc.opens, storage.opens)
			require.Equal(t, tc.closes, storage.closes)
			vector.ReleaseReadResultOnError()
			if tc.firstErr == nil {
				require.Equal(t, 1, releases)
			} else {
				require.Zero(t, releases)
			}
		})
	}
}

type sparseFirstEntryCache struct{ publicationVectorCache }

func (c *sparseFirstEntryCache) Read(_ context.Context, vector *IOVector) error {
	c.reads.Add(1)
	vector.Entries[0].Data = []byte("cached")
	vector.Entries[0].done = true
	return nil
}

func TestS3FSSparsePartialCacheAndStreaming(t *testing.T) {
	ctx := t.Context()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	t.Run("two misses in a larger vector", func(t *testing.T) {
		storage := &sparseTrackingStorage{}
		fs.server.raw = storage
		cache := &callbackVectorCache{onRead: func(vector *IOVector) {
			for pos := range vector.Entries[:4] {
				vector.Entries[pos].Data = []byte("cached")
				vector.Entries[pos].done = true
			}
		}}
		// Cached entries include a streaming destination and a descending offset.
		// The decision uses only the two buffered misses.
		var cachedOutput bytes.Buffer
		vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Caches: []IOVectorCache{cache}, Entries: []IOEntry{
			{Offset: 64 << 20, Size: 4096},
			{Offset: 0, Size: 4096, WriterForRead: &cachedOutput},
			{Offset: 4096, Size: 4096},
			{Offset: 8192, Size: 4096},
			{Offset: 16 << 20, Size: 4096},
			{Offset: 32 << 20, Size: 4096},
		}}
		t.Cleanup(vector.Release)
		require.NoError(t, fs.Read(ctx, vector))
		for _, entry := range vector.Entries[:4] {
			require.Equal(t, []byte("cached"), entry.Data)
		}
		require.Equal(t, bytes.Repeat([]byte{2}, 4096), vector.Entries[4].Data)
		require.Equal(t, bytes.Repeat([]byte{3}, 4096), vector.Entries[5].Data)
		require.Equal(t, 2, storage.opens)
		require.Equal(t, 2, storage.closes)
		require.Equal(t, int64(8192), storage.bytes)
	})
	t.Run("one remaining miss", func(t *testing.T) {
		storage := &sparseTrackingStorage{}
		fs.server.raw = storage
		vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Caches: []IOVectorCache{&sparseFirstEntryCache{}}, Entries: []IOEntry{
			{Offset: 0, Size: 4096}, {Offset: 16 << 20, Size: 4096},
		}}
		t.Cleanup(vector.Release)
		require.NoError(t, fs.Read(ctx, vector))
		require.Equal(t, 1, storage.opens)
		require.Equal(t, bytes.Repeat([]byte{2}, 4096), vector.Entries[1].Data)
		require.Equal(t, int64(4096), storage.bytes)
	})
	t.Run("success updates original vector", func(t *testing.T) {
		storage := &sparseTrackingStorage{}
		fs.server.raw = storage
		cache := &publicationVectorCache{}
		buf := make([]byte, 4096)
		vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Caches: []IOVectorCache{cache}, Entries: []IOEntry{
			{Offset: 0, Size: 4096, Data: buf}, {Offset: 16 << 20, Size: 4096},
		}}
		t.Cleanup(vector.Release)
		require.NoError(t, fs.Read(ctx, vector))
		require.Same(t, vector, cache.last)
		require.Equal(t, int64(1), cache.updates.Load())
		require.Same(t, &buf[0], &vector.Entries[0].Data[0])
	})
	t.Run("mixed read closer", func(t *testing.T) {
		storage := &sparseTrackingStorage{}
		fs.server.raw = storage
		var reader io.ReadCloser
		vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Entries: []IOEntry{
			{Offset: 0, Size: 4096}, {Offset: 16 << 20, Size: 4096, ReadCloserForRead: &reader},
		}}
		t.Cleanup(vector.Release)
		require.NoError(t, fs.Read(ctx, vector))
		require.Equal(t, 1, storage.opens)
		data, err := io.ReadAll(reader)
		require.NoError(t, err)
		require.NoError(t, reader.Close())
		require.Len(t, data, 4096)
		require.Equal(t, 1, storage.closes)
	})
	t.Run("mixed writer", func(t *testing.T) {
		storage := &sparseTrackingStorage{}
		fs.server.raw = storage
		var output bytes.Buffer
		vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Entries: []IOEntry{
			{Offset: 0, Size: 4096}, {Offset: 16 << 20, Size: 4096, WriterForRead: &output},
		}}
		t.Cleanup(vector.Release)
		require.NoError(t, fs.Read(ctx, vector))
		require.Equal(t, 1, storage.opens)
		require.Equal(t, 1, storage.closes)
		require.Equal(t, 4096, output.Len())
	})
}

type sparseFailingReadCache struct{ publicationVectorCache }

func (*sparseFailingReadCache) Read(context.Context, *IOVector) error {
	return errors.New("cache read failed")
}

func TestS3FSFailedCacheReadDoesNotUpdateEarlierCache(t *testing.T) {
	ctx := t.Context()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	first := &publicationVectorCache{}
	vector := &IOVector{FilePath: "foo", Policy: SkipAllCache, Caches: []IOVectorCache{first, &sparseFailingReadCache{}}, Entries: []IOEntry{{Offset: 0, Size: 4096}}}
	require.ErrorContains(t, fs.Read(ctx, vector), "cache read failed")
	require.Zero(t, first.updates.Load())
}

type sparseFailingUpdateCache struct{ publicationVectorCache }

func (c *sparseFailingUpdateCache) Update(ctx context.Context, vector *IOVector, async bool) error {
	_ = c.publicationVectorCache.Update(ctx, vector, async)
	return errors.New("cache update failed")
}

func TestS3FSCacheUpdateErrorDoesNotAffectReadOrOtherUpdates(t *testing.T) {
	ctx := t.Context()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()},
		CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](1 << 20)}, nil, false, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	fs.server.raw = &generatedRangeObjectStorage{}
	first := &publicationVectorCache{}
	failing := &sparseFailingUpdateCache{}
	vector := &IOVector{FilePath: "foo", Policy: SkipFullFilePreloads, Caches: []IOVectorCache{first, failing}, Entries: []IOEntry{
		{Offset: 0, Size: 4096, ToCacheData: CacheOriginalData},
	}}
	t.Cleanup(vector.Release)
	require.NoError(t, fs.Read(ctx, vector))
	require.Equal(t, make([]byte, 4096), vector.Entries[0].Data)
	require.Equal(t, int64(1), failing.updates.Load())
	require.Equal(t, int64(1), first.updates.Load())

	cached := &IOVector{FilePath: "foo", Entries: []IOEntry{{Offset: 0, Size: 4096, ToCacheData: CacheOriginalData}}}
	t.Cleanup(cached.Release)
	require.NoError(t, fs.ReadCache(ctx, cached))
	require.True(t, cached.allDone())
}

func TestS3FSFullCacheHitUpdatesOnlyEarlierMisses(t *testing.T) {
	ctx := t.Context()
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	first := &publicationVectorCache{}
	hit := &sparseFirstEntryCache{}
	last := &publicationVectorCache{}
	vector := &IOVector{FilePath: "foo", Caches: []IOVectorCache{first, hit, last}, Entries: []IOEntry{{Offset: 0, Size: 6}}}
	t.Cleanup(vector.Release)
	require.NoError(t, fs.Read(ctx, vector))
	require.Equal(t, []byte("cached"), vector.Entries[0].Data)
	require.Equal(t, int64(1), first.updates.Load())
	require.Zero(t, hit.updates.Load())
	require.Zero(t, last.reads.Load())
	require.Zero(t, last.updates.Load())
}

// callbackVectorCache exposes cache read and update boundaries to lifecycle tests.
type callbackVectorCache struct {
	publicationVectorCache
	onRead   func(*IOVector)
	onUpdate func()
}

func (c *callbackVectorCache) Read(_ context.Context, v *IOVector) error {
	if c.onRead != nil {
		c.onRead(v)
	}
	return nil
}

func (c *callbackVectorCache) Update(_ context.Context, _ *IOVector, _ bool) error {
	if c.onUpdate != nil {
		c.onUpdate()
	}
	return nil
}

func TestS3FSCacheReadUnwindDoesNotPublish(t *testing.T) {
	for _, cause := range []string{"panic", "goexit"} {
		t.Run(cause, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "unwind", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
			require.NoError(t, err)
			t.Cleanup(func() { fs.Close(context.Background()) })
			first := &publicationVectorCache{}
			entered := make(chan struct{})
			second := &callbackVectorCache{onRead: func(v *IOVector) {
				close(entered)
				v.Entries[0].Data = []byte("partial")
				v.Entries[0].done = true
				if cause == "panic" {
					panic("cache read panic")
				}
				runtime.Goexit()
			}}
			vector := &IOVector{FilePath: "foo", Caches: []IOVectorCache{first, second}, Entries: []IOEntry{{Size: 7}, {Offset: 7, Size: 1}}}
			result := make(chan any, 1)
			finished := make(chan struct{})
			returned := make(chan error, 1)
			go func() {
				defer close(finished)
				defer func() { result <- recover() }()
				returned <- fs.Read(ctx, vector)
			}()
			t.Cleanup(func() { cancel(); <-finished; vector.Release() })
			select {
			case got := <-result:
				if cause == "panic" {
					require.Equal(t, "cache read panic", got)
				} else {
					require.Nil(t, got)
				}
			case <-ctx.Done():
				t.Fatal("read did not unwind")
			}
			<-finished
			select {
			case <-entered:
			default:
				t.Fatal("target cache callback was not reached")
			}
			select {
			case err := <-returned:
				t.Fatalf("Read returned normally: %v", err)
			default:
			}
			require.Zero(t, first.updates.Load())
		})
	}
}

func TestS3FSCacheUpdateUnwindRunsRemainingUpdates(t *testing.T) {
	for _, cause := range []string{"panic", "goexit"} {
		t.Run(cause, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "unwind", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, false, false)
			require.NoError(t, err)
			t.Cleanup(func() { fs.Close(context.Background()) })
			fs.server.raw = &generatedRangeObjectStorage{}
			calls := make(chan string, 6)
			entered := make(chan struct{})
			var key IOMergeKey
			mk := func(name string) *callbackVectorCache {
				return &callbackVectorCache{onUpdate: func() {
					if !fs.ioMerger.IsMerging(key) {
						calls <- "merge-already-released"
					}
					calls <- name
					if name == "C" {
						close(entered)
						if cause == "panic" {
							panic("cache update panic")
						}
						runtime.Goexit()
					}
				}}
			}
			vector := &IOVector{FilePath: "foo", Policy: SkipFullFilePreloads, Caches: []IOVectorCache{mk("A"), mk("B"), mk("C")}, Entries: []IOEntry{{Size: 4096}}}
			key = fs.readMergeKey(vector)
			result := make(chan any, 1)
			finished := make(chan struct{})
			returned := make(chan error, 1)
			go func() {
				defer close(finished)
				defer func() { result <- recover() }()
				returned <- fs.Read(ctx, vector)
			}()
			t.Cleanup(func() { cancel(); <-finished; vector.Release() })
			select {
			case got := <-result:
				if cause == "panic" {
					require.Equal(t, "cache update panic", got)
				} else {
					require.Nil(t, got)
				}
			case <-ctx.Done():
				t.Fatal("read did not unwind")
			}
			<-finished
			select {
			case <-entered:
			default:
				t.Fatal("target cache callback was not reached")
			}
			select {
			case err := <-returned:
				t.Fatalf("Read returned normally: %v", err)
			default:
			}
			close(calls)
			var got []string
			for call := range calls {
				got = append(got, call)
			}
			require.Equal(t, []string{"C", "B", "A"}, got)
			require.False(t, fs.ioMerger.IsMerging(key))
		})
	}
}

func TestS3FSCacheUpdatesPreserveRetryOrder(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var eventsMu sync.Mutex
	var events []string
	record := func(event string) { eventsMu.Lock(); defer eventsMu.Unlock(); events = append(events, event) }
	ctx = WithEventLogger(ctx)
	logger := ctx.Value(EventLoggerKey).(*eventLogger)
	var fs *S3FS
	postSet := func(fscache.CacheKey, fscache.Data) {
		record("memory")
	}
	fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "retry", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](1 << 20), DiskCapacity: ptrTo[toml.ByteSize](1 << 20), DiskPath: ptrTo(t.TempDir()), CacheCallbacks: CacheCallbacks{PostSet: []CacheCallbackFunc{postSet}}}, nil, false, false)
	require.NoError(t, err)
	t.Cleanup(func() { fs.Close(context.Background()) })
	fs.diskCache.prefetchRange = prefetchRangeConfig{}
	fs.server.raw = &generatedRangeObjectStorage{}
	order := make(chan []string, 1)
	first := &callbackVectorCache{onUpdate: func() {
		record("vector1")
		logger.mu.Lock()
		defer logger.mu.Unlock()
		var got []string
		for _, ev := range *logger.events {
			switch ev.ev {
			case str_update_disk_cache_Caches_begin:
				got = append(got, "disk")
			case str_update_memory_cache_begin:
				got = append(got, "memory")
			case str_update_vector_Caches_begin:
				got = append(got, "vector")
			}
		}
		order <- got
	}}
	vector := &IOVector{FilePath: "foo", Policy: SkipFullFilePreloads, Caches: []IOVectorCache{first, &callbackVectorCache{onUpdate: func() { record("vector2") }}}, Entries: []IOEntry{{Size: 4096, ToCacheData: CacheOriginalData}}}
	t.Cleanup(vector.Release)
	done, _ := fs.ioMerger.merge(fs.readMergeKey(vector))
	require.NotNil(t, done)
	release := sync.OnceFunc(done)
	result := make(chan error, 1)
	finished := make(chan struct{})
	go func() { defer close(finished); result <- fs.Read(ctx, vector) }()
	t.Cleanup(func() { release(); cancel(); <-finished })
	for {
		logger.mu.Lock()
		found := false
		if logger.events != nil {
			for _, ev := range *logger.events {
				if ev.ev == str_ioMerger_Merge_wait {
					found = true
					break
				}
			}
		}
		logger.mu.Unlock()
		if found {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("read did not join seeded generation")
		default:
			runtime.Gosched()
		}
	}
	release()
	select {
	case err := <-result:
		require.NoError(t, err)
	case <-ctx.Done():
		t.Fatal("read did not finish")
	}
	<-finished
	eventsMu.Lock()
	gotEvents := append([]string(nil), events...)
	eventsMu.Unlock()
	require.Equal(t, []string{"memory", "vector2", "vector1"}, gotEvents)
	fs.diskCache.Flush(ctx)
	for _, cache := range []IOVectorCache{fs.memCache, fs.diskCache} {
		cached := &IOVector{FilePath: "foo", Entries: []IOEntry{{Size: 4096, ToCacheData: CacheOriginalData}}}
		t.Cleanup(cached.Release)
		require.NoError(t, cache.Read(ctx, cached))
		require.True(t, cached.allDone())
		require.Equal(t, make([]byte, 4096), cached.Entries[0].CachedData.Bytes())
	}
	select {
	case got := <-order:
		require.Equal(t, []string{"disk", "memory", "disk", "memory", "vector", "vector"}, got)
	case <-ctx.Done():
		t.Fatal("missing cache update")
	}
}

// Exercise the disk-cache fallback when the fill a read joined stops.
func TestS3FSSparseDiskFallbackErrors(t *testing.T) {
	for _, mode := range []string{"fill stopped"} {
		for _, tc := range []struct {
			name                    string
			first, second           int64
			firstErr, secondErr     error
			short, cancel, hit      bool
			stream, writer, convert bool
			sameOffset, bodyError   bool
			unexpected, ok          bool
			opens, closes           int
			offsets                 []int64
		}{
			{name: "first 416", first: 9 << 20, second: 25 << 20, firstErr: rangeStatusError{}, opens: 1},
			{name: "later 416", first: 9 << 20, second: 25 << 20, secondErr: rangeStatusError{}, unexpected: true, opens: 2, closes: 1},
			{name: "lower 416", first: 25 << 20, second: 9 << 20, secondErr: rangeStatusError{}, opens: 2, closes: 1},
			{name: "cached first 416", first: 9 << 20, second: 25 << 20, firstErr: rangeStatusError{}, hit: true, opens: 1},
			{name: "cached later 416", first: 9 << 20, second: 25 << 20, secondErr: rangeStatusError{}, hit: true, unexpected: true, opens: 2, closes: 1},
			{name: "same offset 416", first: 9 << 20, second: 9 << 20, secondErr: rangeStatusError{}, sameOffset: true, opens: 2, closes: 1},
			// A source that ends short leaves the read to read the range itself.
			{name: "short", first: 9 << 20, second: 25 << 20, short: true, ok: true, opens: 3, closes: 3, offsets: []int64{9 << 20, 25 << 20, 25 << 20}},
			{name: "body 416", first: 9 << 20, second: 25 << 20, bodyError: true, unexpected: true, opens: 2, closes: 2},
			{name: "cancel", first: 9 << 20, second: 25 << 20, cancel: true, opens: 1, closes: 1},
			{name: "converter 416", first: 9 << 20, second: 25 << 20, convert: true, opens: 2, closes: 2},
			{name: "stream first", first: 9 << 20, second: 25 << 20, stream: true, secondErr: rangeStatusError{}, opens: 2, closes: 1},
			{name: "writer first", first: 9 << 20, second: 25 << 20, writer: true, secondErr: rangeStatusError{}, opens: 2, closes: 1},
		} {
			t.Run(mode+"/"+tc.name, func(t *testing.T) {
				ctx, cancel := context.WithCancel(WithEventLogger(t.Context()))
				defer cancel()
				// The download ends with no bytes once the read joined it.
				disk, pw, _ := downloading(t)
				download := disk.download(disk.objectKey("obj"))
				go func() {
					for ctx.Err() == nil {
						download.mu.Lock()
						joined := len(download.waiters) > 0
						download.mu.Unlock()
						if joined {
							_ = pw.Close()
							return
						}
						time.Sleep(time.Millisecond)
					}
				}()
				fs, err := NewS3FS(ctx, ObjectStorageArguments{Name: "sparse", Endpoint: "disk", Bucket: t.TempDir()}, CacheConfig{}, nil, true, false)
				require.NoError(t, err)
				disk.prefetchRange, disk.memoryPrefetchMax = prefetchRangeConfig{}, 0
				fs.diskCache = disk
				t.Cleanup(func() { fs.diskCache = nil; fs.Close(context.Background()) })
				storage := &sparseTrackingStorage{firstErr: tc.firstErr, secondErr: tc.secondErr, shortSecond: tc.short, bodyErrSecond: tc.bodyError}
				if tc.cancel {
					storage.cancel = cancel
				}
				fs.server.raw = storage
				cache := &publicationVectorCache{}
				hitCache := &sparseFirstEntryCache{}
				first := IOEntry{Offset: tc.first, Size: 4096}
				releases := 0
				if !tc.stream && !tc.writer {
					first.ToCacheData = func(ctx context.Context, r io.Reader, data []byte, allocator CacheDataAllocator) (fscache.Data, error) {
						d, err := CacheOriginalData(ctx, r, data, allocator)
						if err != nil {
							return nil, err
						}
						return &sparseReleaseCountingData{Data: d, releases: &releases}, nil
					}
				}
				var stream io.ReadCloser
				t.Cleanup(func() {
					if stream != nil {
						_ = stream.Close()
					}
				})
				if tc.stream {
					first.ReadCloserForRead = &stream
				}
				var output bytes.Buffer
				if tc.writer {
					first.WriterForRead = &output
				}
				second := IOEntry{Offset: tc.second, Size: 4096}
				if tc.convert {
					second.ToCacheData = func(context.Context, io.Reader, []byte, CacheDataAllocator) (fscache.Data, error) {
						return nil, rangeStatusError{}
					}
				}
				vector := &IOVector{FilePath: "obj", Caches: []IOVectorCache{cache}, Entries: []IOEntry{first, second}}
				if tc.hit {
					vector.Caches = []IOVectorCache{hitCache, cache}
					vector.Entries = append([]IOEntry{{Offset: 0, Size: 4096}}, vector.Entries...)
				}
				if tc.sameOffset {
					vector.Entries = append(vector.Entries, IOEntry{Offset: 25 << 20, Size: 4096})
				}
				t.Cleanup(vector.ReleaseReadResultOnError)
				before := registeredPrefetchObjects(disk)
				err = fs.Read(ctx, vector)
				if stream != nil {
					closeErr := stream.Close()
					stream = nil
					require.NoError(t, closeErr)
				}
				// A read past disk-prefetch-memory-max reads its ranges through prefetch
				// objects of its own, which end after the read returns.
				wantOpens, wantCloses := tc.opens, tc.closes
				if tc.cancel {
					// The first range's object closes its reader, canceling the read, before
					// or after the second range is opened; once the prefetch objects the read
					// started end, every opened reader is closed.
					require.Eventually(t, func() bool {
						opens, closes, _ := storage.counts()
						return opens >= 1 && opens == closes && registeredPrefetchObjects(disk).subsetOf(before)
					}, 10*time.Second, time.Millisecond)
					wantOpens, _, _ = storage.counts()
					wantCloses = wantOpens
					if err == nil {
						require.Equal(t, 2, wantOpens, "a read that got all its ranges opened both")
					}
				}
				require.Eventually(t, func() bool {
					opens, closes, _ := storage.counts()
					return opens == wantOpens && closes == wantCloses
				}, 10*time.Second, time.Millisecond)
				opens, closes, offsets := storage.counts()
				unexpected := tc.unexpected
				if tc.sameOffset && moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF) {
					// The second entry shared the first's prefetch object, so the next read
					// of storage is the 416 past it.
					unexpected, tc.offsets = true, []int64{9 << 20, 25 << 20}
				}
				switch {
				case tc.ok:
					require.NoError(t, err)
				case tc.cancel:
					if err != nil {
						require.ErrorIs(t, err, context.Canceled)
					}
				case unexpected:
					require.True(t, moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF), "%v", err)
				default:
					status, ok := objectStorageStatusCode(err)
					require.True(t, ok, "%v", err)
					require.Equal(t, 416, status)
				}
				require.True(t, vector.fillStopped)
				require.Equal(t, wantOpens, opens)
				require.Equal(t, wantCloses, closes)
				wantOffsets := []int64{tc.first}
				if wantOpens == 2 {
					wantOffsets = append(wantOffsets, tc.second)
				}
				if tc.offsets != nil {
					wantOffsets = tc.offsets
				}
				require.Equal(t, wantOffsets, offsets)
				if err == nil {
					// A read that got all its ranges updates its caller's cache.
					require.EqualValues(t, 1, cache.updates.Load())
				} else {
					require.Zero(t, cache.updates.Load())
				}
				require.Zero(t, hitCache.updates.Load())
				vector.ReleaseReadResultOnError()
				if tc.firstErr == nil && !tc.stream && !tc.writer {
					require.Equal(t, 1, releases)
				} else {
					require.Zero(t, releases)
				}
			})
		}
	}
}

// prefetchObjectSet is a set of prefetch objects, registered or waiting.
type prefetchObjectSet map[any]struct{}

// registeredPrefetchObjects returns the prefetch objects d holds, registered or waiting for
// their resources.
func registeredPrefetchObjects(d *DiskCache) prefetchObjectSet {
	f := &d.prefetchObjects
	f.mu.Lock()
	defer f.mu.Unlock()
	set := prefetchObjectSet{}
	for _, objects := range f.m {
		for _, o := range objects {
			set[o] = struct{}{}
		}
	}
	for o := range f.waiting {
		set[o] = struct{}{}
	}
	return set
}

func (s prefetchObjectSet) subsetOf(t prefetchObjectSet) bool {
	for o := range s {
		if _, ok := t[o]; !ok {
			return false
		}
	}
	return true
}

// newConnCountingServer returns an object storage stub that answers every HEAD with a
// 3-byte object, and the count of its open connections.
func newConnCountingServer(t *testing.T) (*httptest.Server, *atomic.Int64) {
	var open atomic.Int64
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "3")
		w.Header().Set("ETag", `"etag"`)
		w.Header().Set("Last-Modified", "Wed, 02 Sep 2026 03:04:05 GMT")
		w.WriteHeader(http.StatusOK)
	}))
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		switch state {
		case http.StateNew:
			open.Add(1)
		case http.StateClosed, http.StateHijacked:
			open.Add(-1)
		}
	}
	server.Start()
	t.Cleanup(server.Close)
	return server, &open
}

func newStubS3FS(t *testing.T, endpoint string, minio bool) *S3FS {
	fs, err := NewS3FS(context.Background(), ObjectStorageArguments{
		Name:               "s3",
		Endpoint:           endpoint,
		Region:             "us-east-1",
		Bucket:             "bucket",
		KeyID:              "id",
		KeySecret:          "secret",
		IsMinio:            minio,
		NoBucketValidation: true,
	}, DisabledCacheConfig, nil, true, true)
	require.NoError(t, err)
	return fs
}

// A closed S3FS leaves no idle connection open, well before the idle timeout; a second
// Close does nothing. Control: an S3FS left open keeps its idle connection.
func TestS3FSCloseReleasesIdleConnections(t *testing.T) {
	for _, minio := range []bool{false, true} {
		name := "aws"
		if minio {
			name = "minio"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			server, open := newConnCountingServer(t)

			control := newStubS3FS(t, server.URL, minio)
			defer control.Close(ctx)
			_, err := control.StatFile(ctx, "control")
			require.NoError(t, err)
			require.Eventually(t, func() bool { return open.Load() == 1 }, time.Second, time.Millisecond)

			fs := newStubS3FS(t, server.URL, minio)
			_, err = fs.StatFile(ctx, "object")
			require.NoError(t, err)
			require.Eventually(t, func() bool { return open.Load() == 2 }, time.Second, time.Millisecond)

			fs.Close(ctx)
			require.Eventually(t, func() bool { return open.Load() == 1 }, time.Second, time.Millisecond,
				"the closed file service leaves an idle connection open")
			fs.Close(ctx)

			time.Sleep(100 * time.Millisecond)
			require.Equal(t, int64(1), open.Load(), "control: the open file service keeps its idle connection")
		})
	}
}

type closeCountingETLFS struct {
	ETLFileService
	name   string
	closed atomic.Int64
}

func (c *closeCountingETLFS) Name() string { return c.name }

func (c *closeCountingETLFS) Close(context.Context) { c.closed.Add(1) }

// CloseOwnedETL closes a file service GetForETL created for a local path or a path with
// service arguments, and leaves the named service GetForETL returns open.

// heldTailStorage returns the first head bytes of a read at once and the rest once
// proceed is closed.
type heldTailStorage struct {
	ObjectStorage
	head    int
	proceed chan struct{}
}

func (s *heldTailStorage) Read(ctx context.Context, path string, min, max *int64) (io.ReadCloser, error) {
	r, err := s.ObjectStorage.Read(ctx, path, min, max)
	if err != nil {
		return nil, err
	}
	return &heldTailReader{ReadCloser: r, left: s.head, proceed: s.proceed, ctx: ctx}, nil
}

type heldTailReader struct {
	io.ReadCloser
	left    int
	proceed chan struct{}
	ctx     context.Context
}

func (r *heldTailReader) Read(p []byte) (int, error) {
	if r.left <= 0 {
		select {
		case <-r.proceed:
		case <-r.ctx.Done():
			return 0, r.ctx.Err()
		}
		return r.ReadCloser.Read(p)
	}
	n, err := r.ReadCloser.Read(p[:min(len(p), r.left)])
	r.left -= n
	return n, err
}

// firstWriteWriter signals first once it is written to.
type firstWriteWriter struct {
	mu    sync.Mutex
	buf   bytes.Buffer
	once  sync.Once
	first chan struct{}
}

func (w *firstWriteWriter) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.first) })
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(p)
}

func (w *firstWriteWriter) bytes() []byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return bytes.Clone(w.buf.Bytes())
}

func TestStreamedRangeReadWritesBeforeRangeArrives(t *testing.T) {
	const size = minDiskPrefetchMemoryMax + 1<<20
	for _, tc := range []struct {
		name   string
		writer bool
	}{
		{name: "read_closer"},
		{name: "writer_for_read", writer: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, size)
			fs.diskCache.memoryPrefetchMax = minDiskPrefetchMemoryMax
			<-fs.diskCache.load.done
			storage := &heldTailStorage{ObjectStorage: fs.server.raw, head: 64 << 10, proceed: make(chan struct{})}
			fs.server.raw = storage
			release := sync.OnceFunc(func() { close(storage.proceed) })
			defer release()
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()

			w := &firstWriteWriter{first: make(chan struct{})}
			var rc io.ReadCloser
			entry := IOEntry{Offset: 0, Size: size}
			if tc.writer {
				entry.WriterForRead = w
			} else {
				entry.ReadCloserForRead = &rc
			}
			vector := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache, Entries: []IOEntry{entry}}
			done := make(chan error, 1)
			copied := make(chan error, 1)
			go func() {
				err := fs.Read(ctx, vector)
				if err == nil && rc != nil {
					go func() {
						defer rc.Close()
						_, err := io.Copy(w, rc)
						copied <- err
					}()
				} else {
					copied <- nil
				}
				done <- err
			}()
			if tc.writer {
				// The range is read before the writer runs.
				require.Never(t, func() bool {
					select {
					case <-w.first:
						return true
					default:
						return false
					}
				}, 100*time.Millisecond, time.Millisecond)
			} else {
				select {
				case <-w.first:
				case <-time.After(10 * time.Second):
					t.Fatal("the streamed entry got no bytes before the rest of its range arrived")
				}
			}
			release()
			require.NoError(t, <-done)
			require.NoError(t, <-copied)
			require.Equal(t, data, w.bytes())
		})
	}
}

// openProbeStreams opens n streams of size bytes each of the object, from offset 0 on, and
// returns them open.
func openProbeStreams(t *testing.T, fs *S3FS, n int, size int64) []io.ReadCloser {
	streams := make([]io.ReadCloser, n)
	for i := range streams {
		var rc io.ReadCloser
		vec := &IOVector{FilePath: "object", Policy: SkipAllCache,
			Entries: []IOEntry{{Offset: int64(i) * size, Size: size, ReadCloserForRead: &rc}}}
		require.NoError(t, fs.Read(context.Background(), vec))
		require.NotNil(t, rc)
		streams[i] = rc
	}
	return streams
}

// A caller that keeps open streams whose buffers fill the memory budget and then reads a
// column range whose flight needs a buffer gets its bytes promptly. Control: the same read
// with no stream open returns, and the streams read their bytes.
func TestStreamProbeOpenStreamsDoNotBlockFlights(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	const size = 1 << 10
	fs.server.memoryLimit = 2 * flightBufSize(size)

	control := PrefetchRange{Offset: 200 << 10, Size: size}
	got, err := probeRead(fs, control)
	require.NoError(t, err)
	require.Equal(t, data[control.Offset:control.Offset+control.Size], got)

	streams := openProbeStreams(t, fs, 2, size)
	defer func() {
		for _, s := range streams {
			_ = s.Close()
		}
	}()
	r := PrefetchRange{Offset: 250 << 10, Size: size}
	start := time.Now()
	got, err = probeRead(fs, r)
	require.NoError(t, err, "a read waited for the buffers of streams its caller holds open")
	require.Less(t, time.Since(start), time.Second)
	require.Equal(t, data[r.Offset:r.Offset+r.Size], got)

	for i, s := range streams {
		b, err := io.ReadAll(s)
		require.NoError(t, err)
		require.Equal(t, data[i*size:(i+1)*size], b)
	}
}

func probeStreamBytes(fs *S3FS) int64 { return heldMemory(fs.diskCache) }

// Streams take buffers only while their bytes fit the memory budget when they open; the
// others read into their temporary files, and every stream reads its bytes. Control: once
// they are closed, a stream takes a buffer again.
func TestStreamProbeStreamBytesWithinBound(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	const size = 1 << 10
	buf := flightBufSize(size)
	bound := 2 * buf
	fs.server.memoryLimit = bound

	streams := openProbeStreams(t, fs, 6, size)
	require.Equal(t, bound, probeStreamBytes(fs), "the streams that fit take buffers")
	memory := 0
	for _, s := range streams {
		if s.(*streamReader).f.buf != nil {
			memory++
		}
	}
	require.Equal(t, 2, memory)
	for i, s := range streams {
		b, err := io.ReadAll(s)
		require.NoError(t, err)
		require.Equal(t, data[i*size:(i+1)*size], b)
		require.LessOrEqual(t, probeStreamBytes(fs), bound)
		require.NoError(t, s.Close())
	}
	require.Zero(t, probeStreamBytes(fs))

	again := openProbeStreams(t, fs, 1, size)
	require.Equal(t, buf, probeStreamBytes(fs))
	require.NoError(t, again[0].Close())
	require.Zero(t, probeStreamBytes(fs))
}

func TestExactReadSkipsGapBeforeTail(t *testing.T) {
	for _, tc := range []struct {
		name     string
		tailSize int64
	}{
		{name: "known_size", tailSize: 16},
		{name: "read_to_end", tailSize: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 0, 1024)
			vector := &IOVector{
				FilePath: "object",
				Policy:   ExactColumnsRead | SkipAllCache,
				Entries:  []IOEntry{{Offset: 0, Size: 16}, {Offset: 1008, Size: tc.tailSize}},
			}
			defer vector.Release()
			require.NoError(t, fs.Read(context.Background(), vector))
			require.Equal(t, data[:16], vector.Entries[0].Data)
			require.Equal(t, data[1008:], vector.Entries[1].Data)
			storage.mu.Lock()
			defer storage.mu.Unlock()
			require.Equal(t, []int64{0, 1008}, storage.starts, "exact reads should open each entry without reading the intervening bytes")
		})
	}
}

type idlePartStorage struct {
	ObjectStorage
	opened   atomic.Bool
	canceled chan struct{}
	proceed  chan struct{}
}

func (s *idlePartStorage) Read(ctx context.Context, path string, min, max *int64) (io.ReadCloser, error) {
	if min != nil && *min == 0 && s.opened.CompareAndSwap(false, true) {
		return io.NopCloser(&slowCancelReader{ctx: ctx, canceled: s.canceled, proceed: s.proceed}), nil
	}
	return s.ObjectStorage.Read(ctx, path, min, max)
}

func TestJoinedWholeReadFallsBackAfterPartIdleTimeout(t *testing.T) {
	shortIdle(t, 50*time.Millisecond)
	for _, tc := range []struct {
		name  string
		parts int
	}{
		{name: "single_stream", parts: 1},
		{name: "split_download", parts: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: tc.parts, min: 1}, 256<<10)
			<-fs.diskCache.load.done
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			storage := &idlePartStorage{
				ObjectStorage: fs.server.raw,
				canceled:      make(chan struct{}),
				proceed:       make(chan struct{}),
			}
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.proceed) })
			prefetchDone := make(chan struct{})
			go func() {
				defer close(prefetchDone)
				_ = fs.PrefetchFile(ctx, "object")
			}()
			defer func() { cancel(); unblock(); <-prefetchDone }()
			select {
			case <-storage.canceled:
			case <-ctx.Done():
				t.Fatal("prefetch part did not hit its idle timeout")
			}
			download := fs.diskCache.download(fs.diskCache.objectKey("object"))
			require.NotNil(t, download)
			vector := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache, Entries: []IOEntry{{Offset: 0, Size: 16}}}
			readDone := make(chan struct{})
			var readErr error
			go func() {
				defer close(readDone)
				readErr = fs.Read(ctx, vector)
			}()
			defer func() { cancel(); unblock(); <-readDone; vector.Release() }()
			require.Eventually(t, func() bool {
				download.idle.mu.Lock()
				defer download.idle.mu.Unlock()
				return download.idle.waiting > 0
			}, time.Second, time.Millisecond, "foreground read should join before the stalled part returns")
			unblock()
			select {
			case <-readDone:
				require.NoError(t, readErr, "a late joiner should fall back to its exact bytes after a prefetch idle timeout")
				require.Equal(t, data[:16], vector.Entries[0].Data)
			case <-ctx.Done():
				t.Fatal("joined read failed to return")
			}
		})
	}
}

func TestJoinedColumnReadReleasesBufferOnDecoderPanic(t *testing.T) {
	for _, tc := range []struct {
		name   string
		panics bool
	}{
		{name: "decoder_returns_error"},
		{name: "decoder_panics", panics: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 4096)
			<-fs.diskCache.load.done
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			publishing := make(chan struct{})
			resume := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(resume) })
			defer unblock()
			fs.diskCache.hooks.beforeFilePublication = func() {
				close(publishing)
				<-resume
			}
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: 128, Ranges: []PrefetchRange{{Size: 128}},
			}))
			select {
			case <-publishing:
			case <-ctx.Done():
				t.Fatal("prefetch did not reach publication")
			}
			decoderError := errors.New("decoder rejected input")
			vector := &IOVector{
				FilePath: "object",
				Policy:   ColumnPrefetch | SkipFullFilePreloads | SkipMemoryCache,
				Entries: []IOEntry{{Offset: 8, Size: 8, ToCacheData: func(context.Context, io.Reader, []byte, CacheDataAllocator) (fscache.Data, error) {
					if tc.panics {
						panic(decoderError)
					}
					return nil, decoderError
				}}},
			}
			defer vector.Release()
			if tc.panics {
				require.PanicsWithValue(t, decoderError, func() { _ = fs.Read(ctx, vector) })
			} else {
				require.ErrorIs(t, fs.Read(ctx, vector), decoderError)
			}
			unblock()
			waitFillsIdle(t, fs.diskCache)
			fs.diskCache.prefetchObjects.reads.Wait()
			flights := &fs.diskCache.prefetchObjects
			flights.mu.Lock()
			charged := heldMemory(fs.diskCache)
			flights.mu.Unlock()
			require.Zero(t, charged, "a finished prefetch must return its buffer charge after the reading decoder exits")
		})
	}
}

func TestReadStartedSplitHonorsConfiguredPartCount(t *testing.T) {
	for _, tc := range []struct {
		name   string
		offset int64
	}{
		{name: "read_starts_at_object_start"},
		{name: "read_starts_at_interior_block", offset: 64 << 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 0, 256<<10)
			<-fs.diskCache.load.done
			fs.diskCache.objectSplit = (&CacheConfig{
				DiskPrefetchObjectSplit:    ptrTo(4),
				DiskPrefetchObjectSplitMin: ptrTo[toml.ByteSize](0),
			}).diskObjectSplit()
			fs.diskCache.storePrefetchedSize(fs.diskCache.objectKey("object"), int64(len(data)), fs.diskCache.deletes.now())
			vector := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: tc.offset, Size: 16}}}
			defer vector.Release()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.Read(ctx, vector))
			flushPublishers(fs)
			require.Equal(t, data[tc.offset:tc.offset+16], vector.Entries[0].Data)
			require.EqualValues(t, 4, storage.reads.Load(), "the known-size object fits four aligned parts, including a boundary at the read's first needed block")
		})
	}
}

func TestFileRangeReadUsesCachePublishedWhileAdmitted(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "read_after_publication"},
		{name: "read_admitted_before_publication", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const size = minDiskPrefetchMemoryMax + 64
			fs, storage, data := newPrefetchRangeFS(t, 0, size)
			fs.diskCache.memoryPrefetchMax = minDiskPrefetchMemoryMax
			<-fs.diskCache.load.done
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			creating := make(chan struct{})
			proceed := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(proceed) })
			defer unblock()
			fs.diskCache.hooks.beforeCacheTempFileCreate = sync.OnceFunc(func() {
				close(creating)
				<-proceed
			})
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: size, Ranges: []PrefetchRange{{Size: size}},
			}))
			select {
			case <-creating:
			case <-ctx.Done():
				t.Fatal("background range did not acquire the cache path")
			}
			if !tc.overlap {
				unblock()
				waitFillsIdle(t, fs.diskCache)
			}
			vector := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache, Entries: []IOEntry{{Size: size}}}
			readDone := make(chan struct{})
			var readErr error
			go func() {
				defer close(readDone)
				readErr = fs.Read(ctx, vector)
			}()
			defer func() { cancel(); unblock(); <-readDone; vector.Release() }()
			if tc.overlap {
				// The read joins the background range, so it does not end before it.
				select {
				case <-readDone:
					t.Fatal("the foreground read ended before the background range it joins")
				case <-time.After(100 * time.Millisecond):
				}
				unblock()
			}
			select {
			case <-readDone:
				require.NoError(t, readErr)
			case <-ctx.Done():
				t.Fatal("foreground read did not finish")
			}
			require.True(t, bytes.Equal(data, vector.Entries[0].Data))
			require.EqualValues(t, 1, storage.reads.Load(), "publication of the same range should avoid a second storage read")
		})
	}
}

func TestIndexPrefetchPreservesConcurrentDiskCacheFill(t *testing.T) {
	for _, tc := range []struct {
		name           string
		prefetchPolicy Policy
	}{
		{name: "prefetch_into_disk_cache"},
		{name: "prefetch_into_index_cache", prefetchPolicy: IndexFileObject},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			gated := newBlackBoxStorage(storage, func(_ string, offset int64) bool { return offset == 0 })
			fs.server.raw = gated
			unblock := sync.OnceFunc(func() { close(gated.gate) })
			prefetchDone := make(chan struct{})
			var prefetchErr error
			go func() {
				defer close(prefetchDone)
				prefetchErr = fs.PrefetchFile(WithFileServicePolicy(ctx, tc.prefetchPolicy), "object")
			}()
			defer func() { unblock(); <-prefetchDone }()
			require.Eventually(t, func() bool {
				active, _, _ := gated.counts("object")
				return active == 1
			}, time.Second, time.Millisecond, "whole prefetch should have opened storage")
			// The first exact read bypasses disk-cache reads but allows its writes.
			vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipDiskCacheReads | SkipMemoryCache,
				Entries: []IOEntry{{Offset: 128, Size: 16}}}
			require.NoError(t, fs.Read(context.Background(), vec))
			require.Equal(t, data[128:144], vec.Entries[0].Data)
			vec.Release()
			unblock()
			select {
			case <-prefetchDone:
				require.NoError(t, prefetchErr)
				flushPublishers(fs)
			case <-ctx.Done():
				t.Fatal("whole prefetch did not finish")
			}
			_, indexed := fs.indexObjectCached("object")
			require.Equal(t, tc.prefetchPolicy == IndexFileObject, indexed)
			before := storage.reads.Load()
			require.Equal(t, data[128:144], readIndexObjectRange(t, fs, ExactColumnsRead, 128, 16))
			require.Equal(t, before, storage.reads.Load(), "the normal read should find the whole object or its exact range in the disk cache")
		})
	}
}

func TestDeleteInvalidatesPrefetchedRangeForNestedPath(t *testing.T) {
	for _, tc := range []struct {
		name string
		path string
	}{
		{name: "baseline_flat_object_path", path: "flat-object"},
		{name: "nested_object_path", path: "prefix/nested-object"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const size = 4 << 10
			fs, storage, _ := newPrefetchRangeFS(t, 0, size)
			<-fs.diskCache.load.done
			ctx := context.Background()
			write := func(value byte) {
				t.Helper()
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: tc.path, Policy: SkipAllCache,
					Entries: []IOEntry{{Size: size, Data: bytes.Repeat([]byte{value}, size)}}}))
			}
			read := func() byte {
				t.Helper()
				vec := &IOVector{FilePath: tc.path, Policy: ExactColumnsRead | SkipMemoryCache | SkipDiskCacheWrites,
					Entries: []IOEntry{{Size: size}}}
				defer vec.Release()
				require.NoError(t, fs.Read(ctx, vec))
				require.Len(t, vec.Entries[0].Data, size)
				return vec.Entries[0].Data[0]
			}

			write(1)
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: tc.path, Size: size, Ranges: []PrefetchRange{{Size: size}},
			}))
			require.Eventually(t, func() bool { return !fs.diskCache.fillsUnderWay(tc.path) }, time.Second, time.Millisecond)
			require.EqualValues(t, 1, storage.reads.Load(), "the prefetch should read the former object once")
			require.Equal(t, byte(1), read())
			require.EqualValues(t, 1, storage.reads.Load(), "the former data should be read from the prefetched cache range")

			require.NoError(t, fs.Delete(ctx, tc.path))
			write(2)
			require.Equal(t, byte(2), read(), "a read after delete and recreate should return the new object")
		})
	}
}

func TestAsyncWholeFillDoesNotPublishAfterDeleteMark(t *testing.T) {
	for _, tc := range []struct {
		name   string
		delete bool
	}{
		{name: "baseline_no_delete"},
		{name: "delete_marks_object_before_publication", delete: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
			d := fs.diskCache
			<-d.load.done
			reached, release := make(chan struct{}), make(chan struct{})
			allowPublish := sync.OnceFunc(func() { close(release) })
			defer allowPublish()
			d.hooks.beforeFilePublication = func() {
				close(reached)
				<-release
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 64 << 10}))
			select {
			case <-reached:
			case <-ctx.Done():
				t.Fatal("whole-object fill did not reach the publication gate", ctx.Err())
			}

			diskPath := d.objectKey("object")
			if tc.delete {
				mark := d.deletes.now()
				deleteCtx, cancelDelete := context.WithCancel(ctx)
				defer cancelDelete()
				deleted := make(chan error, 1)
				go func() { deleted <- fs.Delete(deleteCtx, "object") }()
				require.Eventually(t, func() bool {
					return d.deletes.since(diskPath, mark)
				}, time.Second, time.Millisecond, "the source delete completed and the cache delete was counted")
				_, err := fs.StatFile(ctx, "object")
				require.Error(t, err, "object storage no longer holds the object")
				require.Never(t, func() bool { return len(deleted) > 0 }, 20*time.Millisecond, time.Millisecond,
					"the delete waits for the publication under way")
				allowPublish()
				require.NoError(t, <-deleted)
			}
			allowPublish()
			d.Flush(ctx)
			require.NoError(t, ctx.Err())
			if !tc.delete {
				require.FileExists(t, d.wholePath("object", 64<<10), "the staged whole object normally publishes")
				return
			}
			require.NoFileExists(t, d.wholePath("object", 64<<10), "the old whole object must not appear once its delete has been counted")
		})
	}
}

func TestSyncWholeFillDoesNotPublishAfterDeleteMark(t *testing.T) {
	for _, tc := range []struct {
		name   string
		delete bool
	}{
		{name: "baseline_no_delete"},
		{name: "delete_during_publication", delete: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
			d := fs.diskCache
			<-d.load.done
			reached, release := make(chan struct{}), make(chan struct{})
			allowPublish := sync.OnceFunc(func() { close(release) })
			defer allowPublish()
			d.hooks.afterPublicationClaim = func() {
				close(reached)
				<-release
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: 64 << 10}))
			select {
			case <-reached:
			case <-ctx.Done():
				t.Fatal("whole-object fill did not reach the publication claim", ctx.Err())
			}

			diskPath := d.objectKey("object")
			deleted := make(chan error, 1)
			if tc.delete {
				mark := d.deletes.now()
				deleteCtx, cancelDelete := context.WithCancel(ctx)
				defer cancelDelete()
				go func() { deleted <- fs.Delete(deleteCtx, "object") }()
				// The delete is counted and waits behind the held publication; its caller
				// gives up, and the publisher still runs its removal.
				require.Eventually(t, func() bool { return d.deletes.since(diskPath, mark) },
					diskCacheLifecycleTestTimeout, time.Millisecond)
				cancelDelete()
				require.ErrorIs(t, <-deleted, context.Canceled)
			}
			allowPublish()
			require.Eventually(t, func() bool { return !fs.diskCache.fillsUnderWay("object") }, time.Second, time.Millisecond)
			d.Flush(ctx)
			if !tc.delete {
				require.FileExists(t, d.wholePath("object", 64<<10))
				return
			}
			require.NoFileExists(t, d.wholePath("object", 64<<10), "a delete counted during the publication of the whole object removes it")
		})
	}
}

func TestDeleteInvalidatesCachedEntryForNestedPath(t *testing.T) {
	for _, path := range []string{"flat-object", "prefix/nested-object"} {
		t.Run(path, func(t *testing.T) {
			const size = 4 << 10
			fs, storage, _ := newPrefetchRangeFS(t, 0, size)
			<-fs.diskCache.load.done
			ctx := context.Background()
			write := func(value byte) {
				t.Helper()
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: path, Policy: SkipAllCache,
					Entries: []IOEntry{{Size: size, Data: bytes.Repeat([]byte{value}, size)}}}))
			}
			read := func() byte {
				t.Helper()
				vec := &IOVector{FilePath: path, Policy: SkipMemoryCache | SkipFullFilePreloads,
					Entries: []IOEntry{{Offset: 16, Size: 64}}}
				defer vec.Release()
				require.NoError(t, fs.Read(ctx, vec))
				return vec.Entries[0].Data[0]
			}

			write(1)
			require.Equal(t, byte(1), read())
			flushPublishers(fs)
			reads := storage.reads.Load()
			require.Equal(t, byte(1), read())
			require.Equal(t, reads, storage.reads.Load(), "the entry is read from the disk cache")

			require.NoError(t, fs.Delete(ctx, path))
			write(2)
			require.Equal(t, byte(2), read())
		})
	}
}

func TestPrefetchBufferReleasedBeforeDecode(t *testing.T) {
	for _, blockDecode := range []bool{false, true} {
		name := "completed_decode"
		if blockDecode {
			name = "blocked_decode_after_copy"
		}
		t.Run(name, func(t *testing.T) {
			const budget = 64 << 10
			const size = 4 << 10
			fs, _, data := newPrefetchRangeFS(t, 0, 64<<10)
			d := fs.diskCache
			<-d.load.done
			d.server.memoryLimit = 2 * budget
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "bystander", Policy: SkipAllCache,
				Entries: []IOEntry{{Size: size, Data: data[:size]}}}))
			storage := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return key == fs.pathToKey("object") })
			fs.server.raw = storage
			resumeStorage := sync.OnceFunc(func() { close(storage.gate) })
			defer resumeStorage()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: size,
				Ranges: []PrefetchRange{{Size: size}}}))
			require.Eventually(t, func() bool {
				cur, _, _ := storage.counts(fs.pathToKey("object"))
				return cur == 1
			}, time.Second, time.Millisecond)

			decoding := make(chan struct{})
			resumeDecode := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(resumeDecode) })
			if !blockDecode {
				unblock()
			}
			vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache,
				Entries: []IOEntry{{Size: size, ToCacheData: func(ctx context.Context, r io.Reader, data []byte, alloc CacheDataAllocator) (fscache.Data, error) {
					close(decoding)
					<-resumeDecode
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
				unblock()
				resumeStorage()
				<-done
				require.NoError(t, readErr)
			}()
			require.Eventually(t, func() bool {
				return memoryWaiters(fs, d.objectKey("object")) == 1
			}, time.Second, time.Millisecond, "the read should be joined to the prefetch")
			resumeStorage()
			select {
			case <-decoding:
			case <-ctx.Done():
				t.Fatal("the reader did not reach the decoder")
			}
			waitPrefetcherIdle(t, fs)
			d.prefetchObjects.reads.Wait()
			if !blockDecode {
				<-done
			}

			// The rest of the budget is held, so the bystander gets a buffer only once the
			// prefetch released its own.
			holdForegroundBuffer(t, fs, budget)
			other := &IOVector{FilePath: "bystander", Policy: ExactColumnsRead | SkipMemoryCache | SkipDiskCacheWrites,
				Entries: []IOEntry{{Size: size}}}
			defer other.Release()
			otherDone := make(chan struct{})
			var otherErr error
			go func() {
				defer close(otherDone)
				otherErr = fs.Read(ctx, other)
			}()
			defer func() {
				unblock()
				<-otherDone
				require.NoError(t, otherErr)
				require.Equal(t, data[:size], other.Entries[0].Data)
			}()
			var charged int64
			require.Eventually(t, func() bool {
				select {
				case <-otherDone:
					return true
				default:
				}
				f := &d.prefetchObjects
				f.mu.Lock()
				defer f.mu.Unlock()
				for o := range f.waiting {
					if target := o.target(); target.filePath == "bystander" && target.state == objectWaitingRoom {
						charged = heldMemory(d)
						return true
					}
				}
				return false
			}, time.Second, time.Millisecond)
			require.LessOrEqual(t, charged, int64(budget), "the bystander should finish while the decoder runs on its own copy")
		})
	}
}

func TestIndexPrefetchWithDirectDiskDownload(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "after_direct_disk_download"},
		{name: "during_direct_disk_download", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				fs, storage, data := newIndexObjectFS(t, true)
				publishing := make(chan struct{})
				publish := make(chan struct{})
				release := sync.OnceFunc(func() { close(publish) })
				defer release()
				// Only the first publication, that of the disk cache, waits.
				var first atomic.Bool
				fs.diskCache.hooks.beforeFilePublication = func() {
					if first.CompareAndSwap(false, true) {
						close(publishing)
						<-publish
					}
				}

				diskPrefetch := make(chan error, 1)
				go func() { diskPrefetch <- fs.PrefetchFile(ctx, "object") }()
				<-publishing
				if !tc.overlap {
					release()
					require.NoError(t, <-diskPrefetch)
				}

				require.NoError(t, fs.PrefetchObject(WithFileServicePolicy(ctx, IndexFileObject), ObjectPrefetch{
					FilePath: "object", Size: int64(len(data)),
				}))
				if tc.overlap {
					// Let the index prefetch reach its wait or complete before publishing
					// the other tier. A worker that defers the fill can then proceed.
					synctest.Wait()
					release()
					require.NoError(t, <-diskPrefetch)
				}
				waitPrefetcherIdle(t, fs)

				require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))))
				require.True(t, fs.indexObjectHeld("object"),
					"index prefetch must warm its own tier after the direct disk fill (backend reads: %d)", storage.reads.Load())
				require.EqualValues(t, 2, storage.reads.Load())
			})
		})
	}
}

func TestIndexRangePrefetchHonorsCacheWritePolicy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		ranged bool
		policy Policy
	}{
		{name: "whole_prefetch_skips_all_cache_writes", policy: SkipCacheWrites},
		{name: "whole_prefetch_with_default_writes"},
		{name: "range_prefetch_skips_all_cache_writes", ranged: true, policy: SkipCacheWrites},
		{name: "range_prefetch_with_default_writes", ranged: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			ctx := WithFileServicePolicy(context.Background(), IndexFileObject|tc.policy)
			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
			if tc.ranged {
				req.Size = 4096
				req.Ranges = []PrefetchRange{{Offset: 64 << 10, Size: req.Size}}
			}
			require.NoError(t, fs.PrefetchObject(ctx, req))
			waitPrefetcherIdle(t, fs)
			require.Zero(t, fs.diskCache.cache.Used(),
				"index-object bytes must stay out of the table-data tier (backend reads: %d)", storage.reads.Load())
			if tc.policy.Any(SkipCacheWrites) {
				require.False(t, fs.indexObjectHeld("object"), "skipping all cache writes must leave both tiers empty")
			} else if !tc.ranged {
				require.True(t, fs.indexObjectHeld("object"), "a whole prefetch with writes enabled warms the index tier")
			}
		})
	}
}

func TestIndexObjectIdleTTLStartsAfterReadReleasesHold(t *testing.T) {
	for _, tc := range []struct {
		name     string
		readTime time.Duration
	}{
		{name: "read_finishes_before_idle_ttl"},
		{name: "read_outlasts_idle_ttl", readTime: 3 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				fs, storage, data := newIndexObjectFS(t, true)
				cacheIndexObject(t, fs, "object", data)
				cache := fs.indexFileCache
				cache.idleTTL = 2 * time.Second
				ctx := context.Background()
				var received bytes.Buffer
				vec := &IOVector{
					FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
					Entries: []IOEntry{{
						Offset: 100, Size: 64, Data: make([]byte, 64),
						WriterForRead: readShareWriterFunc(func(data []byte) (int, error) {
							// Advance virtual time while the cached entry is held by the read.
							time.Sleep(tc.readTime)
							return received.Write(data)
						}),
					}},
				}
				defer vec.Release()
				require.NoError(t, fs.Read(ctx, vec))
				require.Equal(t, data[100:164], received.Bytes())
				require.Zero(t, storage.reads.Load())
				removed, err := cache.collectIdle(ctx, time.Now())
				require.NoError(t, err)
				require.Zero(t, removed, "an index object just released by a successful read is not idle")
			})
		})
	}
}

func TestExactReadDoesNotWidenRange(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
	}{
		{name: "exact_columns_read", policy: ExactColumnsRead},
		{name: "exact_read_on_its_own", policy: ExactRead},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, false)
			var offset atomic.Int64
			tracked := newBlackBoxStorage(storage, func(_ string, from int64) bool {
				offset.Store(from)
				return false
			})
			fs.server.raw = tracked
			require.Equal(t, data[100:164], readIndexObjectRange(t, fs, tc.policy, 100, 64))
			require.EqualValues(t, 1, storage.reads.Load())
			require.EqualValues(t, 100, offset.Load(), "an exact read opens storage at the requested offset")
		})
	}
}

func TestIndexReadKeepsDownloadTierAcrossEntries(t *testing.T) {
	for _, tc := range []struct {
		name              string
		startDiskPrefetch bool
	}{
		{name: "index_publication_only"},
		{name: "table_download_replaces_index_download", startDiskPrefetch: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				fs, _, data := newIndexObjectFS(t, true)
				indexCtx := WithFileServicePolicy(ctx, IndexFileObject)

				indexPublishing, allowIndexPublish := make(chan struct{}), make(chan struct{})
				releaseIndex := sync.OnceFunc(func() { close(allowIndexPublish) })
				defer releaseIndex()
				fs.diskCache.hooks.beforeFilePublication = sync.OnceFunc(func() {
					close(indexPublishing)
					<-allowIndexPublish
				})
				diskFailure := errors.New("table download source failed")
				probe := &probeStorage{
					ObjectStorage: fs.server.raw, at: 0, gate: make(chan struct{}),
					answer: func(r io.ReadCloser) (io.ReadCloser, error) {
						_ = r.Close()
						return nil, diskFailure
					},
				}
				probe.hit.Store(true) // Leave the initial index download unchanged.
				fs.server.raw = probe
				releaseDisk := sync.OnceFunc(func() { close(probe.gate) })
				defer releaseDisk()

				indexPrefetch := make(chan error, 1)
				go func() { indexPrefetch <- fs.PrefetchFile(indexCtx, "object") }()
				<-indexPublishing

				firstEntry, resumeRead := make(chan struct{}), make(chan struct{})
				releaseRead := sync.OnceFunc(func() { close(resumeRead) })
				defer releaseRead()
				var received bytes.Buffer
				vec := &IOVector{
					FilePath: "object", Policy: IndexFileObject | SkipDiskCacheReads | SkipMemoryCache,
					Entries: []IOEntry{
						{Offset: 0, Size: 64, Data: make([]byte, 64), WriterForRead: readShareWriterFunc(func(b []byte) (int, error) {
							close(firstEntry)
							<-resumeRead
							return received.Write(b)
						})},
						{Offset: 64, Size: 64, Data: make([]byte, 64)},
					},
				}
				result := make(chan error, 1)
				go func() { result <- fs.Read(ctx, vec) }()
				<-firstEntry
				releaseIndex()
				require.NoError(t, <-indexPrefetch)
				flushPublishers(fs)
				require.True(t, fs.indexObjectHeld("object"))

				diskPrefetch := make(chan error, 1)
				if tc.startDiskPrefetch {
					probe.hit.Store(false)
					go func() { diskPrefetch <- fs.PrefetchFile(ctx, "object") }()
					synctest.Wait()
					require.True(t, probe.hit.Load(), "the replacement table-data download reached storage")
				}
				releaseRead()
				synctest.Wait()
				if tc.startDiskPrefetch {
					releaseDisk()
					require.ErrorIs(t, <-diskPrefetch, diskFailure)
				}
				readErr := <-result
				defer vec.Release()
				require.NoError(t, readErr, "a read skipping the table-data tier should use the published index entry")
				require.Equal(t, data[:64], received.Bytes())
				require.Equal(t, data[64:128], vec.Entries[1].Data)
			})
		})
	}
}

func TestNegativeConcurrencyIsRejected(t *testing.T) {
	noProbe := func(context.Context, string) (string, error) { return "", nil }
	for _, tc := range []struct {
		concurrency int64
		ok          bool
	}{{0, true}, {1, true}, {minRequestRoom - 1, true}, {minRequestRoom, true}, {1024, true}, {-1, false}} {
		args := ObjectStorageArguments{Endpoint: "disk", Bucket: t.TempDir(), Concurrency: tc.concurrency}
		err := args.validateWithRegionProbe(noProbe)
		if tc.ok {
			require.NoError(t, err, "concurrency=%d", tc.concurrency)
		} else {
			require.ErrorContains(t, err, fmt.Sprintf("concurrency must not be negative, got %d", tc.concurrency))
		}
		fs, err := NewS3FS(context.Background(), ObjectStorageArguments{Name: "s3", Endpoint: "disk", Bucket: t.TempDir(), Concurrency: tc.concurrency},
			CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](0)}, nil, false, true)
		if tc.ok {
			require.NoError(t, err, "concurrency=%d", tc.concurrency)
			fs.Close(context.Background())
		} else {
			require.ErrorContains(t, err, "concurrency must not be negative")
		}
	}
}

func TestDownloadConnsAreHalfTheStorageConcurrency(t *testing.T) {
	for _, tc := range []struct {
		conns       int
		concurrency int64
		want        int
	}{
		{0, 0, maxConnsPerHost / 2},
		{1, 0, 32},
		{0, 1, 32},
		{2, 3, 32},
		{64, 0, 32},
		{256, 0, 128},
		{0, 64, 32},
		{64, 64, 32},
		{256, 100, 50},
		{64, 1 << 20, 32},
		{4096, 0, 512},
	} {
		ctx := context.Background()
		fs, err := NewS3FS(ctx, ObjectStorageArguments{
			Name: "s3", Endpoint: "disk", Bucket: t.TempDir(),
			Concurrency: tc.concurrency, MaxConnsPerHost: tc.conns,
		}, CacheConfig{DiskPath: ptrTo(t.TempDir()), DiskCapacity: ptrTo[toml.ByteSize](64 << 20), MemoryCapacity: ptrTo[toml.ByteSize](0)}, nil, false, false)
		require.NoError(t, err)
		require.Equal(t, tc.want, fs.server.backgroundMax, "conns %d concurrency %d", tc.conns, tc.concurrency)
		require.Equal(t, 2*tc.want, fs.server.capacity, "conns %d concurrency %d", tc.conns, tc.concurrency)
		fs.Close(ctx)
	}
}

func newStorageSlotFS(t *testing.T, concurrency int64, conns int, paths []string, data []byte, stalled func(string) bool) (*S3FS, *blackBoxStorage) {
	ctx := context.Background()
	config := indexCacheConfig(t, 64<<20, 64<<20)
	config.IndexFileCacheEnabled = false
	config.MemoryCapacity = DisabledCacheConfig.MemoryCapacity
	fs, err := NewS3FS(ctx, ObjectStorageArguments{
		Name: "s3", Endpoint: "disk", Bucket: t.TempDir(),
		Concurrency: concurrency, MaxConnsPerHost: conns,
	}, config, nil, false, false)
	require.NoError(t, err)
	<-fs.diskCache.load.done
	for _, path := range paths {
		require.NoError(t, fs.Write(ctx, IOVector{
			FilePath: path, Policy: SkipAllCache,
			Entries: []IOEntry{{Size: int64(len(data)), Data: data}},
		}))
	}
	// Stall prefetch opens after the server admits them.
	gated := newBlackBoxStorage(fs.server.raw, func(key string, _ int64) bool { return stalled(key) })
	fs.server.raw = gated
	return fs, gated
}

func requireForegroundReads(t *testing.T, fs *S3FS, gated *blackBoxStorage, data []byte, policies ...Policy) {
	for _, policy := range policies {
		readCtx, cancel := context.WithTimeout(context.Background(), time.Second)
		vec := &IOVector{FilePath: "foreground", Policy: policy, Entries: []IOEntry{{Offset: 64, Size: 16}}}
		readErr := fs.Read(readCtx, vec)
		_, _, opens := gated.counts("foreground")
		require.NoError(t, readErr, "policy %v, foreground opens: %d", policy, opens)
		require.Equal(t, data[64:80], vec.Entries[0].Data)
		vec.Release()
		cancel()
	}
}

// Two stalled whole-object prefetches leave storage slots and download connections for
// foreground reads, cache-filling ones included.
func TestPrefetchLeavesStorageSlotForForegroundRead(t *testing.T) {
	for _, tc := range []struct {
		name        string
		concurrency int64
	}{
		{name: "default_storage_concurrency"},
		{name: "storage_concurrency_64", concurrency: minRequestRoom},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				data := bytes.Repeat([]byte("data"), 64<<10)
				fs, gated := newStorageSlotFS(t, tc.concurrency, 64, []string{"prefetch-a", "prefetch-b", "foreground"}, data,
					func(key string) bool { return key == "prefetch-a" || key == "prefetch-b" })
				defer fs.Close(ctx)
				defer close(gated.gate)
				for _, path := range []string{"prefetch-a", "prefetch-b"} {
					require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: path, Size: int64(len(data))}))
				}
				synctest.Wait()
				first, _, _ := gated.counts("prefetch-a")
				second, _, _ := gated.counts("prefetch-b")
				require.Positive(t, first+second, "at least one prefetch must still be in flight")
				requireForegroundReads(t, fs, gated, data, SkipAllCache|ExactColumnsRead, SkipMemoryCache)
			})
		})
	}
}

// Stalled column prefetches holding every connection downloads may hold leave storage
// slots for a foreground read when the storage concurrency is below the connections per host.
func TestStalledDownloadsLeaveStorageConcurrencyForForegroundRead(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx := context.Background()
		data := bytes.Repeat([]byte("data"), 64<<10)
		paths := make([]string, 0, 33)
		paths = append(paths, "foreground")
		for i := range 32 {
			paths = append(paths, fmt.Sprintf("prefetch-%d", i))
		}
		fs, gated := newStorageSlotFS(t, minRequestRoom, 0, paths, data,
			func(key string) bool { return strings.HasPrefix(key, "prefetch-") })
		defer fs.Close(ctx)
		defer close(gated.gate)
		for _, path := range paths[1:] {
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: path, Size: int64(len(data)), Ranges: columnRanges(8, 20<<10, 1<<10), Scan: &PrefetchScan{},
			}))
		}
		synctest.Wait()
		inFlight := 0
		for _, path := range paths[1:] {
			cur, _, _ := gated.counts(path)
			inFlight += cur
		}
		require.EqualValues(t, fs.server.backgroundMax, inFlight, "stalled downloads hold every background room")
		requireForegroundReads(t, fs, gated, data, SkipAllCache|ExactColumnsRead)
	})
}

func waitReviewPrefetchIdle(t *testing.T, fs *S3FS) {
	t.Helper()
	waitFillsIdle(t, fs.diskCache)
}

func TestPrefetchDiskTierAfterIndex(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "after_index_download"},
		{name: "during_index_download", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			fs, _, data := newIndexObjectFS(t, true)
			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
			fs.server.raw = storage
			resume := sync.OnceFunc(func() { close(storage.gate) })
			defer resume()

			require.NoError(t, fs.PrefetchObject(WithFileServicePolicy(ctx, IndexFileObject), req))
			require.Eventually(t, func() bool {
				current, _, _ := storage.counts(fs.pathToKey(req.FilePath))
				return current == 1
			}, 5*time.Second, time.Millisecond)
			if !tc.overlap {
				resume()
				waitReviewPrefetchIdle(t, fs)
			}
			require.NoError(t, fs.PrefetchObject(ctx, req))
			resume()
			waitReviewPrefetchIdle(t, fs)

			require.True(t, fs.indexObjectHeld(req.FilePath), "the index prefetch should fill the index-file cache")
			_, _, reads := storage.counts(fs.pathToKey(req.FilePath))
			require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath(req.FilePath, int64(len(data)))),
				"the disk prefetch should fill the disk cache (storage opens: %d)", reads)
			require.Equal(t, 2, reads)
		})
	}
}

func TestPrefetchBothQueuedTiers(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "separate_queue_entries"},
		{name: "same_object_queued_in_both_tiers", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			fs, storage, data := newIndexObjectFS(t, true)
			resume := busyFills(t, fs)

			req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
			indexReq := req
			if !tc.overlap {
				indexReq.FilePath = "another-object"
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: indexReq.FilePath, Policy: SkipAllCache,
					Entries: []IOEntry{{Size: req.Size, Data: data}}}))
			}
			require.NoError(t, fs.PrefetchObject(ctx, req))
			require.True(t, fs.diskCache.isQueued(req.FilePath))
			require.NoError(t, fs.PrefetchObject(WithFileServicePolicy(ctx, IndexFileObject), indexReq))
			require.True(t, fs.diskCache.isQueued(indexReq.FilePath))
			resume()
			waitReviewPrefetchIdle(t, fs)

			require.True(t, fs.indexObjectHeld(indexReq.FilePath), "the index prefetch should fill the index-file cache")
			require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath(req.FilePath, int64(len(data)))),
				"the disk prefetch should fill the disk cache (storage opens: %d)", storage.reads.Load())
			require.EqualValues(t, 2, storage.reads.Load())
		})
	}
}

func TestIndexPrefetchAfterAsyncDiskDownload(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "after_disk_publication"},
		{name: "while_disk_publication_is_pending", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				fs, _, data := newIndexObjectFS(t, true)
				storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
				fs.server.raw = storage
				resumeStorage := sync.OnceFunc(func() { close(storage.gate) })
				defer resumeStorage()
				publicationEntered, publish := make(chan struct{}), make(chan struct{})
				resumePublication := sync.OnceFunc(func() { close(publish) })
				defer resumePublication()
				// Only the first publication waits.
				var first atomic.Bool
				fs.diskCache.hooks.beforeFilePublication = func() {
					if first.CompareAndSwap(false, true) {
						close(publicationEntered)
						<-publish
					}
				}

				req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
				indexCtx := WithFileServicePolicy(ctx, IndexFileObject)
				require.NoError(t, fs.PrefetchObject(ctx, req))
				require.Eventually(t, func() bool {
					current, _, _ := storage.counts(fs.pathToKey(req.FilePath))
					return current == 1
				}, 5*time.Second, time.Millisecond)
				if tc.overlap {
					require.NoError(t, fs.PrefetchObject(indexCtx, req))
				}
				resumeStorage()
				select {
				case <-publicationEntered:
				case <-time.After(5 * time.Second):
					t.Fatal("the disk cache did not reach publication")
				}
				// Let the deferred prefetch reach its wait or complete before resuming publication.
				synctest.Wait()
				resumePublication()
				waitReviewPrefetchIdle(t, fs)
				fs.diskCache.Flush(ctx)
				require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath(req.FilePath, int64(len(data)))))
				if !tc.overlap {
					require.NoError(t, fs.PrefetchObject(indexCtx, req))
					waitReviewPrefetchIdle(t, fs)
					fs.diskCache.Flush(ctx)
				}

				_, _, reads := storage.counts(fs.pathToKey(req.FilePath))
				require.True(t, fs.indexObjectHeld(req.FilePath),
					"the index prefetch should survive the disk cache's asynchronous publication (storage opens: %d)", reads)
				require.Equal(t, 2, reads)
			})
		})
	}
}

func TestQueuedRangeMergeChargesOnlyUniqueBytes(t *testing.T) {
	for _, tc := range []struct {
		name         string
		secondOffset int64
	}{
		{name: "adjacent_requests", secondOffset: 128},
		{name: "overlapping_requests", secondOffset: 64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const total = 192
			ctx := context.Background()
			fs, storage, _ := newIndexObjectFS(t, false)
			resume := busyFills(t, fs)

			for _, r := range []PrefetchRange{{Size: 128}, {Offset: tc.secondOffset, Size: total - tc.secondOffset}} {
				require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}))
			}
			resume()
			waitReviewPrefetchIdle(t, fs)

			d := fs.diskCache
			key := d.objectKey("object")
			_, prefixCached := d.ranges.covering(key, 0, 128)
			require.True(t, prefixCached)
			_, unionCached := d.ranges.covering(key, 0, total)
			require.True(t, unionCached, "all %d unique bytes are cached; cached ranges: %+v", total, d.ranges.of(key))
			require.EqualValues(t, 2, storage.reads.Load(), "the bytes the first fill does not hold get a fill of their own")
		})
	}
}

func TestBatchDeleteInvalidatesOtherIndexObjects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		readOnly bool
	}{
		{name: "both_index_entries_writable"},
		{name: "first_index_shard_read_only", readOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnly && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			fs, _, data := newIndexObjectFS(t, true)
			const first, second = "first-object", "object"
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: first, Policy: SkipAllCache,
				Entries: []IOEntry{{Size: int64(len(data)), Data: data}}}))
			cacheIndexObject(t, fs, first, data)
			cacheIndexObject(t, fs, second, data)
			firstDir, firstCached := fs.indexObjectCached(first)
			secondDir, secondCached := fs.indexObjectCached(second)
			require.True(t, firstCached && secondCached)
			firstShard := filepath.Dir(firstDir)
			require.NotEqual(t, firstShard, filepath.Dir(secondDir))
			if tc.readOnly {
				require.NoError(t, os.Chmod(firstShard, 0o500))
				defer func() { require.NoError(t, os.Chmod(firstShard, 0o755)) }()
			}

			err := fs.Delete(ctx, first, second)
			if tc.readOnly {
				require.ErrorIs(t, err, os.ErrPermission)
			} else {
				require.NoError(t, err)
			}
			_, err = fs.StatFile(ctx, second)
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "the second object is deleted from storage: %v", err)
			vec := &IOVector{FilePath: second, Policy: IndexFileObject | ExactColumnsRead,
				Entries: []IOEntry{{Size: 16}}}
			defer vec.Release()
			err = fs.Read(ctx, vec)
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound),
				"the second object's index entry should be invalidated independently: read returned %d bytes, error %v", len(vec.Entries[0].Data), err)
		})
	}
}

func TestBatchDeleteRemovesOtherDiskCacheObjects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		readOnly bool
	}{
		{name: "both_shards_writable"},
		{name: "first_shard_read_only", readOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnly && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			fs, _, data := newPrefetchRangeFS(t, 4096, 300<<10)
			const first, second = "first-object", "object"
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: first, Policy: SkipDiskCache | SkipMemoryCache,
				Entries: []IOEntry{{Size: int64(len(data)), Data: data}}}))
			read := func(file string, policy Policy, offset int64) error {
				vec := &IOVector{FilePath: file, Policy: policy | SkipMemoryCache,
					Entries: []IOEntry{{Offset: offset, Size: 10}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				fs.diskCache.prefetchObjects.reads.Wait()
				flushPublishers(fs)
				return err
			}
			for _, file := range []string{first, second} {
				require.NoError(t, read(file, SkipFullFilePreloads, 100))
				require.NoError(t, read(file, 0, 200<<10))
			}
			firstPath, secondPath := fs.diskCache.wholePath(first, int64(len(data))), fs.diskCache.wholePath(second, int64(len(data)))
			require.True(t, fs.diskCache.cache.Contains(firstPath) && fs.diskCache.cache.Contains(secondPath))
			require.NotEmpty(t, fs.diskCache.ranges.of(fs.diskCache.objectKey(second)))
			firstShard := filepath.Dir(firstPath)
			require.NotEqual(t, firstShard, filepath.Dir(secondPath))
			if tc.readOnly {
				require.NoError(t, os.Chmod(firstShard, 0o500))
				defer func() { require.NoError(t, os.Chmod(firstShard, 0o755)) }()
			}

			err := fs.Delete(ctx, first, second)
			if tc.readOnly {
				require.ErrorIs(t, err, os.ErrPermission)
			} else {
				require.NoError(t, err)
			}
			for _, policy := range []Policy{0, SkipFullFilePreloads} {
				err = read(second, policy, 100)
				require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "policy %v: %v", policy, err)
			}
			require.False(t, fs.diskCache.cache.Contains(secondPath), "the second object's whole-object entry is removed")
			require.Empty(t, fs.diskCache.ranges.of(fs.diskCache.objectKey(second)), "the second object's range files are removed")
		})
	}
}

func TestLocalFSBatchDeleteInvalidatesDeletedFiles(t *testing.T) {
	for _, tc := range []struct {
		name     string
		readOnly bool
	}{
		{name: "both_directories_writable"},
		{name: "second_directory_read_only", readOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnly && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			root := t.TempDir()
			fs, err := NewLocalFS(ctx, "test", root, CacheConfig{MemoryCapacity: ptrTo[toml.ByteSize](1 << 20)}, nil)
			require.NoError(t, err)
			t.Cleanup(func() { fs.Close(ctx) })
			const first, second = "a/first", "b/second"
			read := func(file string) ([]byte, error) {
				vec := &IOVector{FilePath: file, Entries: []IOEntry{{Size: 3, ToCacheData: CacheOriginalData}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				return append([]byte(nil), vec.Entries[0].Data...), err
			}
			for _, file := range []string{first, second} {
				require.NoError(t, fs.Write(ctx, IOVector{FilePath: file, Entries: []IOEntry{{Size: 3, Data: []byte("abc")}}}))
				data, err := read(file)
				require.NoError(t, err)
				require.Equal(t, "abc", string(data))
			}
			if tc.readOnly {
				dir := filepath.Join(root, "b")
				require.NoError(t, os.Chmod(dir, 0o500))
				defer func() { require.NoError(t, os.Chmod(dir, 0o755)) }()
			}

			err = fs.Delete(ctx, first, second)
			if tc.readOnly {
				require.ErrorIs(t, err, os.ErrPermission)
			} else {
				require.NoError(t, err)
			}
			data, err := read(first)
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "the deleted first file is not read from the cache: read %q, error %v", data, err)
		})
	}
}

func TestCachingReadDoesNotJoinUncachedFlight(t *testing.T) {
	for _, tc := range []struct {
		name       string
		skipLookup bool
	}{
		{name: "baseline_direct_caching_read", skipLookup: true},
		{name: "caching_read_with_shared_lookup"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const offset, size = 100, 8
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, storage, data := newPrefetchRangeFS(t, 0, 1024)
			<-fs.diskCache.load.done
			failedOpen := &failOnceStorage{ObjectStorage: storage, fail: offset, gate: make(chan struct{})}
			fs.server.raw = failedOpen
			release := sync.OnceFunc(func() { close(failedOpen.gate) })
			defer release()
			type result struct {
				data string
				err  error
			}
			start := func(policy Policy) <-chan result {
				done := make(chan result, 1)
				go func() {
					vector := &IOVector{FilePath: "object", Policy: policy | ExactColumnsRead | SkipMemoryCache,
						Entries: []IOEntry{{Offset: offset, Size: size}}}
					defer vector.Release()
					err := fs.Read(ctx, vector)
					done <- result{data: string(vector.Entries[0].Data), err: err}
				}()
				return done
			}

			path := fs.diskCache.objectKey("object")
			first := start(SkipDiskCacheWrites)
			require.Eventually(t, func() bool { return failedOpen.failed.Load() && memoryWaiters(fs, path) == 1 },
				time.Second, time.Millisecond, "the uncached flight must be waiting for its first storage open")

			var policy Policy
			if tc.skipLookup {
				policy = SkipDiskCacheReads
			}
			second := start(policy)
			require.Eventually(t, func() bool { return len(second) == 1 || memoryWaiters(fs, path) == 2 },
				time.Second, time.Millisecond, "the second read must either complete or join the stalled flight")
			release()
			require.ErrorContains(t, (<-first).err, "probe: storage failed")
			got := <-second
			require.NoError(t, got.err, "the caching read should open storage independently of the non-caching flight")
			require.Equal(t, string(data[offset:offset+size]), got.data)
			require.EqualValues(t, 1, storage.reads.Load(), "the independent storage open succeeds")
		})
	}
}

func TestAsyncDiskPrefetchSurvivesIndexDownload(t *testing.T) {
	for _, tc := range []struct {
		name    string
		overlap bool
	}{
		{name: "baseline_index_published_first"},
		{name: "disk_prefetch_during_index_download", overlap: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				fs, _, data := newIndexObjectFS(t, true)
				storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
				fs.server.raw = storage
				resume := sync.OnceFunc(func() { close(storage.gate) })
				defer resume()

				req := ObjectPrefetch{FilePath: "object", Size: int64(len(data))}
				indexCtx := WithFileServicePolicy(ctx, IndexFileObject)
				require.NoError(t, fs.PrefetchObject(indexCtx, req))
				synctest.Wait()
				current, _, attempts := storage.counts("object")
				require.Equal(t, 1, current, "the index download holds the object path")
				require.Equal(t, 1, attempts)
				if !tc.overlap {
					resume()
					waitReviewPrefetchIdle(t, fs)
					fs.diskCache.Flush(ctx)
				}

				require.NoError(t, fs.PrefetchObject(ctx, req))
				synctest.Wait()
				resume()
				waitReviewPrefetchIdle(t, fs)
				fs.diskCache.Flush(ctx)

				require.True(t, fs.indexObjectHeld("object"))
				_, _, attempts = storage.counts("object")
				require.True(t, fs.diskCache.cache.Contains(fs.diskCache.wholePath("object", int64(len(data)))),
					"the disk cache stayed empty after %d storage open(s), despite a prefetch for each cache", attempts)
				require.Equal(t, 2, attempts)
			})
		})
	}
}

func TestStartupLoadPreservesRefilledFile(t *testing.T) {
	for _, tc := range []struct {
		name       string
		deletePath string
	}{
		{name: "baseline_unrelated_delete", deletePath: "unrelated"},
		{name: "delete_and_refill_same_object", deletePath: "object"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			d := newLifecycleTestDiskCache(t)
			path := d.wholePath("object", 3)
			require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
			require.NoError(t, os.WriteFile(path, []byte("old"), 0o644))

			found, resume, loaded := make(chan struct{}), make(chan struct{}), make(chan struct{})
			unblock := sync.OnceFunc(func() { close(resume) })
			d.hooks.beforeLoadFile = func(file string) {
				if file == path {
					close(found)
					<-resume
				}
			}
			go func() {
				defer close(loaded)
				d.loadCache(ctx)
			}()
			defer func() { unblock(); <-loaded }()
			select {
			case <-found:
			case <-ctx.Done():
				t.Fatal("startup load did not find the old file")
			}

			require.NoError(t, d.DeletePaths(ctx, []string{tc.deletePath}))
			want := "old"
			if tc.deletePath == "object" {
				want = "new"
				require.NoError(t, d.SetFile(ctx, "object", func(context.Context) (io.ReadCloser, error) {
					return io.NopCloser(strings.NewReader(want)), nil
				}))
				got, err := os.ReadFile(path)
				require.NoError(t, err)
				require.Equal(t, want, string(got), "refill published before the loader resumes")
			}

			unblock()
			<-loaded
			got, err := os.ReadFile(path)
			require.NoError(t, err, "startup load removed the current cache file")
			require.Equal(t, want, string(got))
			require.True(t, d.cache.Contains(path))
		})
	}
}

func TestDeleteRetryInvalidatesRange(t *testing.T) {
	for _, tc := range []struct {
		name          string
		readOnlyShard bool
	}{
		{name: "baseline_writable_shard"},
		{name: "first_delete_cannot_unlink_range", readOnlyShard: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnlyShard && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			fs, _, data := newPrefetchRangeFS(t, 0, 1024)
			<-fs.diskCache.load.done
			const offset, size = 100, 4
			rangeEntry := IOEntry{Offset: offset, Size: size}
			require.Equal(t, data[offset:offset+size], readRangeOf(t, fs, ExactRead, offset, size))
			path := fs.diskCache.pathForIOEntry("object", rangeEntry)
			require.FileExists(t, path)
			if tc.readOnlyShard {
				parent := filepath.Dir(path)
				require.NoError(t, os.Chmod(parent, 0o555))
				t.Cleanup(func() { require.NoError(t, os.Chmod(parent, 0o755)) })
				require.ErrorIs(t, fs.Delete(ctx, "object"), os.ErrPermission)
				require.FileExists(t, path)
				require.NoError(t, os.Chmod(parent, 0o755))
			} else {
				require.NoError(t, fs.Delete(ctx, "object"))
			}
			require.NoError(t, fs.Delete(ctx, "object"), "the delete retry succeeds")
			replacement := make([]byte, len(data))
			copy(replacement[offset:], "new!")
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Policy: SkipAllCache,
				Entries: []IOEntry{{Size: int64(len(replacement)), Data: replacement}}}))
			got := readRangeOf(t, fs, ExactRead, offset, size)
			require.Equal(t, "new!", string(got), "a successful delete retry must invalidate the old cached range")
		})
	}
}

func TestIndexPrefetchRepairsNonFileData(t *testing.T) {
	for _, tc := range []struct {
		name        string
		isDirectory bool
	}{
		{name: "baseline_missing_data"},
		{name: "data_replaced_by_directory", isDirectory: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			path, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			name := filepath.Join(path, IndexObjectFile)
			require.NoError(t, os.Remove(name))
			if tc.isDirectory {
				require.NoError(t, os.Mkdir(name, 0o755))
			}
			ctx := WithFileServicePolicy(context.Background(), IndexFileObject)
			require.NoError(t, fs.PrefetchFile(ctx, "object"))
			require.EqualValues(t, 1, storage.reads.Load(), "prefetch must replace the invalid index-cache entry from storage")
			require.Equal(t, data[100:104], readIndexObjectRange(t, fs, IndexFileObject, 100, 4))
			require.EqualValues(t, 1, storage.reads.Load(), "the repaired entry serves the next read")
		})
	}
}

func TestDropCachedRangeKeepsRangeUntilUnlinked(t *testing.T) {
	for _, tc := range []struct {
		name          string
		readOnlyShard bool
	}{
		{name: "baseline_writable_shard"},
		{name: "drop_cannot_unlink_range", readOnlyShard: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnlyShard && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			fs, _, data := newPrefetchRangeFS(t, 0, 1024)
			<-fs.diskCache.load.done
			const offset, size = 100, 4
			require.Equal(t, data[offset:offset+size], readRangeOf(t, fs, ExactRead, offset, size))
			path := fs.diskCache.pathForIOEntry("object", IOEntry{Offset: offset, Size: size})
			require.NoError(t, os.Truncate(path, 1))
			drop := func() {
				f, err := os.Open(path)
				require.NoError(t, err)
				fs.diskCache.dropCachedRange(fs.diskCache.objectKey("object"), path, f)
				require.NoError(t, f.Close())
				fs.diskCache.Flush(ctx)
			}
			if tc.readOnlyShard {
				parent := filepath.Dir(path)
				require.NoError(t, os.Chmod(parent, 0o555))
				t.Cleanup(func() { require.NoError(t, os.Chmod(parent, 0o755)) })
				drop()
				require.FileExists(t, path)
				require.True(t, fs.diskCache.cache.Contains(path), "a range not unlinked stays indexed")
				require.NoError(t, os.Chmod(parent, 0o755))
			} else {
				drop()
				require.False(t, fs.diskCache.cache.Contains(path))
			}
			require.NoError(t, fs.Delete(ctx, "object"))
			require.NoFileExists(t, path, "a delete after a failed drop must remove the range file")
			require.False(t, fs.diskCache.cache.Contains(path))
		})
	}
}

func TestIndexReadRepairsNonFileData(t *testing.T) {
	for _, tc := range []struct {
		name        string
		isDirectory bool
	}{
		{name: "baseline_missing_data"},
		{name: "data_replaced_by_directory", isDirectory: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, storage, data := newIndexObjectFS(t, true)
			cacheIndexObject(t, fs, "object", data)
			path, ok := fs.indexObjectCached("object")
			require.True(t, ok)
			name := filepath.Join(path, IndexObjectFile)
			require.NoError(t, os.Remove(name))
			if tc.isDirectory {
				require.NoError(t, os.Mkdir(name, 0o755))
			}
			require.Equal(t, data[100:104], readIndexObjectRange(t, fs, IndexFileObject, 100, 4))
			require.EqualValues(t, 1, storage.reads.Load())
			require.True(t, fs.indexObjectHeld("object"), "the read must replace the invalid index-cache entry")
			require.Equal(t, data[100:104], readIndexObjectRange(t, fs, IndexFileObject, 100, 4))
			require.EqualValues(t, 1, storage.reads.Load(), "the repaired entry serves the next read")
		})
	}
}

// probeStorage holds the first read from offset at until gate is closed, then answers it
// through answer, given the reader of object storage; it counts the other reads.
type probeStorage struct {
	ObjectStorage
	at     int64
	gate   chan struct{}
	answer func(io.ReadCloser) (io.ReadCloser, error)
	hit    atomic.Bool
	others atomic.Int64
}

func (s *probeStorage) Read(ctx context.Context, key string, from, to *int64) (io.ReadCloser, error) {
	if from != nil && *from == s.at && s.hit.CompareAndSwap(false, true) {
		select {
		case <-s.gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		r, err := s.ObjectStorage.Read(ctx, key, from, to)
		if err != nil {
			return nil, err
		}
		return s.answer(r)
	}
	s.others.Add(1)
	return s.ObjectStorage.Read(ctx, key, from, to)
}

type readCloser2 struct {
	io.Reader
	io.Closer
}

// failAfter returns a reader of n bytes of r that then fails with err.
func failAfter(n int64, err error) func(io.ReadCloser) (io.ReadCloser, error) {
	return func(r io.ReadCloser) (io.ReadCloser, error) {
		return readCloser2{Reader: io.MultiReader(io.LimitReader(r, n), failingReader{err}), Closer: r}, nil
	}
}

// shortBy returns a reader of r without its last n bytes.
func shortBy(r PrefetchRange, n int64) func(io.ReadCloser) (io.ReadCloser, error) {
	return func(rc io.ReadCloser) (io.ReadCloser, error) {
		return readCloser2{Reader: io.LimitReader(rc, r.Size-n), Closer: rc}, nil
	}
}

type failingReader struct{ err error }

func (r failingReader) Read([]byte) (int, error) { return 0, r.err }

// joinedColumnRead prefetches r of the object, held in s, and returns what a column read of
// r that joins the prefetch returns once s answers it.
func joinedColumnRead(t *testing.T, fileBacked bool, r PrefetchRange, answer func(io.ReadCloser) (io.ReadCloser, error)) ([]byte, []byte, error, *probeStorage) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
	<-fs.diskCache.load.done
	if fileBacked {
		fs.diskCache.memoryPrefetchMax = r.Size - 1
	}
	s := &probeStorage{ObjectStorage: storage, at: r.Offset, gate: make(chan struct{}), answer: answer}
	fs.server.raw = s
	release := sync.OnceFunc(func() { close(s.gate) })
	t.Cleanup(release)
	require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}))
	require.Eventually(t, s.hit.Load, 5*time.Second, time.Millisecond)
	result := make(chan error, 1)
	var got []byte
	go func() {
		b, err := readColumnRange(ctx, fs, r)
		got = b
		result <- err
	}()
	diskPath := fs.diskCache.objectKey("object")
	require.Eventually(t, func() bool {
		if !fileBacked {
			return memoryWaiters(fs, diskPath) == 1
		}
		o, _ := fs.diskCache.fileDownload(diskPath, r.Offset, r.Offset+r.Size)
		if o == nil {
			return false
		}
		o.mu.Lock()
		defer o.mu.Unlock()
		return len(o.waiters) == 1
	}, 5*time.Second, time.Millisecond, "the read joins the prefetch")
	release()
	err := <-result
	waitPrefetcherIdle(t, fs)
	return got, data[r.Offset : r.Offset+r.Size], err, s
}

var joinedKinds = []struct {
	name       string
	fileBacked bool
}{{name: "memory"}, {name: "file", fileBacked: true}}

// BC2: a read whose joined prefetch object fails with an unexpected end of its source gets
// the error a read of its own reports.
func TestJoinedReadReportsUnexpectedEOFAsOwnRead(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	for _, k := range joinedKinds {
		t.Run(k.name, func(t *testing.T) {
			_, _, err, _ := joinedColumnRead(t, k.fileBacked, r, failAfter(100, io.ErrUnexpectedEOF))
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrUnexpectedEOF), "got %v", err)
		})
	}
}

// BC1: a read whose joined prefetch object finds the object gone fails as its own read
// would, without reading object storage again.
func TestJoinedReadFailsOnMissingObject(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	for _, k := range joinedKinds {
		t.Run(k.name, func(t *testing.T) {
			_, _, err, s := joinedColumnRead(t, k.fileBacked, r, func(rc io.ReadCloser) (io.ReadCloser, error) {
				_ = rc.Close()
				return nil, moerr.NewFileNotFoundNoCtx("object")
			})
			require.True(t, moerr.IsMoErrCode(err, moerr.ErrFileNotFound), "got %v", err)
			require.Zero(t, s.others.Load(), "the read does not read the object again")
		})
	}
}

// BC3: a read whose joined prefetch object's source ends short of its bytes reads them
// itself.
func TestJoinedReadOfShortSourceReadsItsBytes(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	for _, k := range joinedKinds {
		for _, short := range []int64{100, r.Size} {
			t.Run(k.name+"/"+map[bool]string{true: "empty", false: "short"}[short == r.Size], func(t *testing.T) {
				got, want, err, s := joinedColumnRead(t, k.fileBacked, r, shortBy(r, short))
				require.NoError(t, err)
				require.True(t, bytes.Equal(want, got))
				require.EqualValues(t, 1, s.others.Load(), "the read reads its bytes itself")
			})
		}
	}
}

// BC4: a read that started a whole-object download and cannot read the bytes the download
// wrote, from the download's file, reads them elsewhere.
func TestStreamedReadFallsBackOnUnreadableDownloadFile(t *testing.T) {
	fs, _, data := newPrefetchRangeFS(t, 0, 300<<10)
	d := fs.diskCache
	<-d.load.done
	storage := newBlackBoxStorage(fs.server.raw, func(_ string, offset int64) bool { return offset == 0 })
	fs.server.raw = storage
	release := sync.OnceFunc(func() { close(storage.gate) })
	t.Cleanup(release)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: 1000, Size: 4000}}}
	defer vec.Release()
	result := make(chan error, 1)
	go func() { result <- fs.Read(ctx, vec) }()
	diskPath := d.objectKey("object")
	var download *objectDownload
	require.Eventually(t, func() bool {
		if download = d.download(diskPath); download == nil {
			return false
		}
		download.mu.Lock()
		defer download.mu.Unlock()
		return len(download.waiters) == 1
	}, 5*time.Second, time.Millisecond, "the read waits for its download")
	download.mu.Lock()
	require.NoError(t, download.file.Close())
	download.mu.Unlock()
	release()
	require.NoError(t, <-result)
	require.True(t, bytes.Equal(data[1000:5000], vec.Entries[0].Data))
}

// stalledWholeDownload starts a whole-object download of the object whose source sends
// nothing until the returned function is called.
func stalledWholeDownload(t *testing.T, fs *S3FS) (*objectDownload, func()) {
	d := fs.diskCache
	<-d.load.done
	storage := newBlackBoxStorage(fs.server.raw, func(_ string, offset int64) bool { return offset == 0 })
	fs.server.raw = storage
	release := sync.OnceFunc(func() { close(storage.gate) })
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = fs.PrefetchFile(context.Background(), "object")
	}()
	t.Cleanup(func() {
		release()
		<-done
	})
	var download *objectDownload
	require.Eventually(t, func() bool {
		download = d.download(d.objectKey("object"))
		return download != nil
	}, 5*time.Second, time.Millisecond)
	return download, release
}

// BC12: a read of the disk cache whose context ends while it waits for a download returns
// its context's error.
func TestDownloadWaitEndsWithReadContext(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
	download, _ := stalledWholeDownload(t, fs)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Size: 4096}}}
	defer vec.Release()
	result := make(chan error, 1)
	go func() { result <- fs.diskCache.Read(ctx, vec) }()
	require.Eventually(t, func() bool {
		download.mu.Lock()
		defer download.mu.Unlock()
		return len(download.waiters) == 1
	}, 5*time.Second, time.Millisecond)
	cancel()
	require.ErrorIs(t, <-result, context.Canceled)
}

// BC19: a prefetch whose object is deleted while it downloads counts as ended, not failed.
func TestPrefetchOfDeletedObjectEnds(t *testing.T) {
	for _, whole := range []bool{false, true} {
		t.Run(map[bool]string{true: "whole", false: "ranges"}[whole], func(t *testing.T) {
			fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
			<-fs.diskCache.load.done
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
			fs.server.raw = storage
			t.Cleanup(func() { close(storage.gate) })
			failed := promtestutil.ToFloat64(metric.FSPrefetchFailedCounter)
			ended := promtestutil.ToFloat64(metric.FSPrefetchEndedCounter)
			req := ObjectPrefetch{FilePath: "object", Size: 300 << 10}
			if !whole {
				req.Ranges = []PrefetchRange{{Offset: 4096, Size: 4096}}
			}
			require.NoError(t, fs.PrefetchObject(context.Background(), req))
			key := fs.pathToKey("object")
			require.Eventually(t, func() bool {
				cur, _, _ := storage.counts(key)
				return cur == 1
			}, 5*time.Second, time.Millisecond)
			require.NoError(t, fs.Delete(context.Background(), "object"))
			waitPrefetcherIdle(t, fs)
			require.Equal(t, failed, promtestutil.ToFloat64(metric.FSPrefetchFailedCounter))
			require.Equal(t, ended+1, promtestutil.ToFloat64(metric.FSPrefetchEndedCounter))
		})
	}
}

// BC6: an entry of no bytes is an empty range for the disk cache too.
func TestDiskCacheReadOfEmptyEntryIsEmptyRange(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
	<-fs.diskCache.load.done
	vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: 10}}}
	defer vec.Release()
	err := fs.diskCache.Read(context.Background(), vec)
	require.True(t, moerr.IsMoErrCode(err, moerr.ErrEmptyRange), "got %v", err)
}

// BC7: the index-file cache serves every entry it can, also those after one read to the
// end of the object.
func TestIndexObjectServesEntriesAfterReadToEnd(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	cacheIndexObject(t, fs, "object", data)
	vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
		Entries: []IOEntry{{Offset: 0, Size: -1}, {Offset: 100, Size: 10}}}
	defer vec.Release()
	require.NoError(t, fs.readIndexObject(context.Background(), vec))
	require.False(t, vec.Entries[0].done)
	require.True(t, vec.Entries[1].done)
	require.Equal(t, data[100:110], vec.Entries[1].Data)
}

// BC8: the bytes a read finds cached once a prefetch object it joined ended are read from
// cached ranges that hold them together, or from the cached whole object.
func TestCachedBytesAcrossRangesAndWholeObject(t *testing.T) {
	const part = 64 << 10
	ctx := context.Background()
	t.Run("ranges", func(t *testing.T) {
		fs, _, data := newPrefetchRangeFS(t, 0, 4*part)
		d := fs.diskCache
		<-d.load.done
		require.NoError(t, d.Update(ctx, &IOVector{FilePath: "object", Entries: []IOEntry{
			{Size: part, Data: data[:part]},
			{Offset: part, Size: part, Data: data[part : 2*part]},
		}}, false))
		dst := make([]byte, 32)
		require.True(t, d.readCachedBytes(ctx, "object", part-16, dst))
		require.Equal(t, data[part-16:part+16], dst)
	})
	t.Run("whole", func(t *testing.T) {
		fs, _, data := newPrefetchRangeFS(t, 0, 4*part)
		d := fs.diskCache
		<-d.load.done
		require.NoError(t, d.setFileContent(ctx, "object", data, false))
		dst := make([]byte, 32)
		require.True(t, d.readCachedBytes(ctx, "object", part-16, dst))
		require.Equal(t, data[part-16:part+16], dst)
		require.False(t, d.readCachedBytes(ctx, "object", int64(len(data))-16, dst), "bytes past the object's end")
		require.True(t, d.cache.Contains(d.wholePath("object", int64(len(data)))), "a read past its end keeps the whole object")
	})
}

// BC17: a read whose whole-object fill was skipped, as for an object cached meanwhile,
// reads the disk cache again before object storage.
func TestSkippedWholeFillReadsCacheAgain(t *testing.T) {
	fs, storage, data := newPrefetchRangeFS(t, 0, 64<<10)
	d := fs.diskCache
	<-d.load.done
	ctx := context.Background()
	require.NoError(t, d.setFileContent(ctx, "object", data, false))
	vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: 1000, Size: 100}}}
	defer vec.Release()
	require.NoError(t, fs.read(ctx, vec, false, false))
	require.Equal(t, data[1000:1100], vec.Entries[0].Data)
	require.Zero(t, storage.reads.Load(), "the cached object serves the read")
}

// BC16: an exact read joins a queued prefetch of ranges holding its bytes, as a column
// read does.
func TestExactReadJoinsQueuedRanges(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 300<<10)
	<-fs.diskCache.load.done
	busyFills(t, fs)
	storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool { return true })
	fs.server.raw = storage
	t.Cleanup(func() { close(storage.gate) })
	r := PrefetchRange{Offset: 4096, Size: 4096}
	require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: 300 << 10, Ranges: []PrefetchRange{r}}))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead, Entries: []IOEntry{{Offset: r.Offset, Size: r.Size}}}
	require.True(t, fs.joinFills(ctx, vec, fs.sourcesFor(ctx, vec)))
}

// BC15: a read made under a download's context joins no prefetch object.
func TestReadOfDownloadJoinsNoPrefetchObject(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	for _, policy := range []Policy{ColumnPrefetch | SkipFullFilePreloads, SkipFullFilePreloads} {
		t.Run(map[bool]string{true: "column", false: "range"}[policy.Any(ColumnPrefetch)], func(t *testing.T) {
			fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
			<-fs.diskCache.load.done
			s := &probeStorage{ObjectStorage: storage, at: r.Offset, gate: make(chan struct{}),
				answer: func(rc io.ReadCloser) (io.ReadCloser, error) { return rc, nil }}
			fs.server.raw = s
			t.Cleanup(sync.OnceFunc(func() { close(s.gate) }))
			require.NoError(t, fs.PrefetchObject(context.Background(), ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}))
			require.Eventually(t, s.hit.Load, 5*time.Second, time.Millisecond)
			ctx, cancel := context.WithTimeout(withDownloadStarted(context.Background(), func() {}), 5*time.Second)
			defer cancel()
			vec := &IOVector{FilePath: "object", Policy: policy | SkipMemoryCache, Entries: []IOEntry{{Offset: r.Offset, Size: r.Size}}}
			defer vec.Release()
			require.NoError(t, fs.Read(ctx, vec))
			require.Equal(t, data[r.Offset:r.Offset+r.Size], vec.Entries[0].Data)
			require.EqualValues(t, 1, s.others.Load(), "the read reads its bytes itself")
		})
	}
}

// BC13: a fill of an index object that skips every cache write fills neither the
// index-file cache nor the disk cache.
func TestIndexObjectFillSkippingCacheWrites(t *testing.T) {
	fs, _, _ := newIndexObjectFS(t, true)
	ctx := WithFileServicePolicy(context.Background(), IndexFileObject|SkipCacheWrites)
	require.NoError(t, fs.PrefetchFile(ctx, "object"))
	fs.diskCache.prefetchObjects.reads.Wait()
	_, cached := fs.indexObjectCached("object")
	require.False(t, cached)
	require.Zero(t, fs.diskCache.cache.Used())
	require.True(t, FullFilePreloadSkipped(ctx, fs, IndexFileObject|SkipCacheWrites, "object"))
}

// BC14: a read of an index object that skips disk-cache reads joins its download into the
// index-file cache.
func TestIndexObjectReadSkippingDiskReadsJoinsIndexDownload(t *testing.T) {
	fs, _, data := newIndexObjectFS(t, true)
	d := fs.diskCache
	storage := newBlackBoxStorage(fs.server.raw, func(_ string, offset int64) bool { return offset == 0 })
	fs.server.raw = storage
	release := sync.OnceFunc(func() { close(storage.gate) })
	t.Cleanup(release)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	filled := make(chan error, 1)
	go func() { filled <- fs.PrefetchFile(WithFileServicePolicy(ctx, IndexFileObject), "object") }()
	var download *objectDownload
	require.Eventually(t, func() bool {
		download = d.download(d.objectKey("object"))
		return download != nil
	}, 5*time.Second, time.Millisecond)
	require.NotNil(t, d.wholeFill(d.objectKey("object"), true), "the download is of the index-file cache")
	vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipDiskCacheReads | SkipMemoryCache,
		Entries: []IOEntry{{Offset: 100, Size: 4096}}}
	defer vec.Release()
	result := make(chan error, 1)
	go func() { result <- fs.Read(ctx, vec) }()
	require.Eventually(t, func() bool {
		download.mu.Lock()
		defer download.mu.Unlock()
		return len(download.waiters) == 1
	}, 5*time.Second, time.Millisecond, "the read joins the download")
	release()
	require.NoError(t, <-result)
	require.NoError(t, <-filled)
	require.Equal(t, data[100:4196], vec.Entries[0].Data)
	_, _, total := storage.counts(fs.pathToKey("object"))
	require.Equal(t, 1, total, "the download alone reads the object")
}

// BC10: a read that caches nothing joins any prefetch object holding its bytes, also one
// that caches nothing.
func TestUncachingReadJoinsUncachingFlight(t *testing.T) {
	r := PrefetchRange{Offset: 40 << 10, Size: 4 << 10}
	fs, storage, data := newPrefetchRangeFS(t, 0, 300<<10)
	<-fs.diskCache.load.done
	s := &probeStorage{ObjectStorage: storage, at: r.Offset, gate: make(chan struct{}),
		answer: func(rc io.ReadCloser) (io.ReadCloser, error) { return rc, nil }}
	fs.server.raw = s
	release := sync.OnceFunc(func() { close(s.gate) })
	t.Cleanup(release)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	first := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache, Entries: []IOEntry{{Offset: r.Offset, Size: r.Size}}}
	defer first.Release()
	firstDone := make(chan error, 1)
	go func() { firstDone <- fs.Read(ctx, first) }()
	diskPath := fs.diskCache.objectKey("object")
	require.Eventually(t, s.hit.Load, 5*time.Second, time.Millisecond)
	var out bytes.Buffer
	second := &IOVector{FilePath: "object", Policy: ColumnPrefetch | SkipFullFilePreloads | SkipDiskCacheWrites | SkipMemoryCache,
		Entries: []IOEntry{{Offset: r.Offset, Size: r.Size, WriterForRead: &out}}}
	defer second.Release()
	secondDone := make(chan error, 1)
	go func() { secondDone <- fs.Read(ctx, second) }()
	require.Eventually(t, func() bool { return memoryWaiters(fs, diskPath) == 2 }, 5*time.Second, time.Millisecond,
		"the second read joins the first one's flight")
	release()
	require.NoError(t, <-firstDone)
	require.NoError(t, <-secondDone)
	require.Equal(t, data[r.Offset:r.Offset+r.Size], out.Bytes())
	require.Zero(t, s.others.Load())
}

// BC11: a read of the disk cache joins the first live FilePrefetchObject registered
// holding its bytes, never a stopped one.
func TestFileDownloadPicksLiveObject(t *testing.T) {
	fs, _, _ := newPrefetchRangeFS(t, 0, 64<<10)
	d := fs.diskCache
	<-d.load.done
	diskPath := d.objectKey("object")
	tempPath := t.TempDir() + "/download"
	require.NoError(t, os.WriteFile(tempPath, make([]byte, 64<<10), 0600))
	object := func(written int64, stopped bool) *filePrefetchObject {
		p := newFilePrefetchObject(context.Background(), 0, 64<<10)
		o := newObjectDownload(tempPath, nil)
		t.Cleanup(o.retire)
		o.begin(0)
		if written > 0 {
			o.advance(0, written)
		}
		p.download = o
		if stopped {
			p.cancelRead()
		}
		return p
	}
	slow, fast, stoppedFast := object(0, false), object(64<<10, false), object(64<<10, true)
	f := &d.prefetchObjects
	f.mu.Lock()
	f.registerLocked(diskPath, stoppedFast)
	f.registerLocked(diskPath, slow)
	f.registerLocked(diskPath, fast)
	f.mu.Unlock()
	got, _ := d.fileDownload(diskPath, 100, 200)
	require.Same(t, slow.download, got)
	f.mu.Lock()
	f.forgetLocked(diskPath, slow)
	f.mu.Unlock()
	got, _ = d.fileDownload(diskPath, 100, 200)
	require.Same(t, fast.download, got)
	f.mu.Lock()
	f.forgetLocked(diskPath, fast)
	f.mu.Unlock()
	got, _ = d.fileDownload(diskPath, 100, 200)
	require.Nil(t, got)
}

// tailHeldStorage returns the first 64 KiB of every read at once and holds the rest until
// gate is closed.
type tailHeldStorage struct {
	ObjectStorage
	gate  chan struct{}
	opens atomic.Int64
}

func (s *tailHeldStorage) Read(ctx context.Context, key string, from, to *int64) (io.ReadCloser, error) {
	s.opens.Add(1)
	r, err := s.ObjectStorage.Read(ctx, key, from, to)
	if err != nil {
		return nil, err
	}
	return readCloser2{Reader: io.MultiReader(io.LimitReader(r, 64<<10), &heldTail{ctx: ctx, gate: s.gate, r: r}), Closer: r}, nil
}

type heldTail struct {
	ctx  context.Context
	gate chan struct{}
	r    io.Reader
	open bool
}

func (h *heldTail) Read(b []byte) (int, error) {
	if !h.open {
		select {
		case <-h.gate:
		case <-h.ctx.Done():
			return 0, h.ctx.Err()
		}
		h.open = true
	}
	return h.r.Read(b)
}

// firstWriteSignal closes first on its first write. It has no ReadFrom, so a copy to it
// writes each chunk.
type firstWriteSignal struct {
	mu    sync.Mutex
	buf   []byte
	first chan struct{}
	once  sync.Once
}

func (w *firstWriteSignal) Write(b []byte) (int, error) {
	w.once.Do(func() { close(w.first) })
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, b...)
	return len(b), nil
}

func (w *firstWriteSignal) Bytes() []byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf
}

// A streamed entry larger than a prefetch object buffers opens its own reader: a reader
// gets its first bytes before a running range download that holds them sends its last, a
// writer gets its range once it is read; one of at most that many bytes joins the download.
func TestStreamedEntryPastPrefetchBufferReadsItself(t *testing.T) {
	const max = minDiskPrefetchMemoryMax
	for _, tc := range []struct {
		name   string
		size   int64
		reader bool
		policy Policy
	}{
		{name: "writer", size: max + 1<<20},
		{name: "writer_skip_disk_reads", size: max + 1<<20, policy: SkipDiskCacheReads},
		{name: "reader", size: max + 1<<20, reader: true},
		{name: "reader_skip_disk_reads", size: max + 1<<20, reader: true, policy: SkipDiskCacheReads},
		{name: "writer_joins_at_most_buffer", size: 128 << 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, max+2<<20)
			d := fs.diskCache
			d.memoryPrefetchMax = max
			d.objectSplit.parts = 1
			<-d.load.done
			storage := &tailHeldStorage{ObjectStorage: fs.server.raw, gate: make(chan struct{})}
			fs.server.raw = storage
			release := sync.OnceFunc(func() { close(storage.gate) })
			t.Cleanup(release)
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			r := PrefetchRange{Size: max + 1<<20}
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: r.Size, Ranges: []PrefetchRange{r}}))
			diskPath := d.objectKey("object")
			var download *objectDownload
			require.Eventually(t, func() bool {
				download, _ = d.fileDownload(diskPath, 0, r.Size)
				return download != nil
			}, 60*time.Second, time.Millisecond)

			entry := IOEntry{Size: tc.size}
			writer := &firstWriteSignal{first: make(chan struct{})}
			var stream io.ReadCloser
			if tc.reader {
				entry.ReadCloserForRead = &stream
			} else {
				entry.WriterForRead = writer
			}
			vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache | tc.policy, Entries: []IOEntry{entry}}
			defer vec.Release()
			result := make(chan error, 1)
			go func() { result <- fs.Read(ctx, vec) }()
			if tc.size <= max {
				require.Eventually(t, func() bool {
					download.mu.Lock()
					defer download.mu.Unlock()
					return len(download.waiters) == 1
				}, 60*time.Second, time.Millisecond, "the read joins the download")
				release()
				require.NoError(t, <-result)
				require.Equal(t, data[:tc.size], writer.Bytes())
				require.EqualValues(t, 1, storage.opens.Load())
				return
			}
			if tc.reader {
				require.NoError(t, <-result)
				var first [1]byte
				_, err := io.ReadFull(stream, first[:])
				require.NoError(t, err, "the reader returns bytes before the download's tail")
				require.Equal(t, data[0], first[0])
				release()
				rest, err := io.ReadAll(stream)
				require.NoError(t, err)
				require.Equal(t, data[1:tc.size], rest)
				require.NoError(t, stream.Close())
				return
			}
			require.Never(t, func() bool {
				select {
				case <-writer.first:
					return true
				default:
					return false
				}
			}, 100*time.Millisecond, time.Millisecond, "the writer runs once its range is read")
			release()
			require.NoError(t, <-result)
			require.Equal(t, data[:tc.size], writer.Bytes())
		})
	}
}

// A large exact read of bytes a stalled file prefetch holds joins it: it waits, opens no
// source of its own, and returns the bytes once the prefetch goes on.
func TestLargeReadJoinsStalledFilePrefetch(t *testing.T) {
	const size = 9 << 20
	for _, tc := range []struct {
		name   string
		policy Policy
	}{
		{name: "baseline_skip_cache_writes", policy: SkipDiskCacheWrites},
		{name: "cache_writes_enabled"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, _, data := newPrefetchRangeFS(t, 0, 2*size)
			d := fs.diskCache
			d.memoryPrefetchMax = minDiskPrefetchMemoryMax
			d.objectSplit.parts = 1
			<-d.load.done

			var opens atomic.Int64
			storage := newBlackBoxStorage(fs.server.raw, func(string, int64) bool {
				return opens.Add(1) == 1
			})
			fs.server.raw = storage
			release := sync.OnceFunc(func() { close(storage.gate) })
			defer release()

			ctx := context.Background()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: size, Ranges: []PrefetchRange{{Size: size}},
			}))
			require.Eventually(t, func() bool { return opens.Load() == 1 }, 5*time.Second, time.Millisecond)
			download, _ := d.fileDownload(d.objectKey("object"), 0, size)
			require.NotNil(t, download)

			readCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()
			vec := &IOVector{
				FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache | tc.policy,
				Entries: []IOEntry{{Size: size}},
			}
			defer vec.Release()
			done := make(chan error, 1)
			go func() { done <- fs.Read(readCtx, vec) }()
			require.Never(t, func() bool { return len(done) > 0 }, 200*time.Millisecond, time.Millisecond,
				"the read waits for the prefetch it joined")
			require.EqualValues(t, 1, opens.Load(), "the read opens no source of its own")
			release()
			require.NoError(t, <-done)
			require.True(t, bytes.Equal(data[:size], vec.Entries[0].Data), "the exact range should be returned")
			require.EqualValues(t, 1, opens.Load())
		})
	}
}

func TestJoinedFileRangeWaitsForCaller(t *testing.T) {
	const (
		size = 192 << 10
		wait = 100 * time.Millisecond
	)
	for _, tc := range []struct {
		name    string
		ioBound time.Duration
	}{
		{name: "baseline_long_io_wait", ioBound: 5 * time.Second},
		{name: "short_io_wait", ioBound: wait},
	} {
		t.Run(tc.name, func(t *testing.T) {
			originalBound := maxIOWaitDuration
			maxIOWaitDuration = tc.ioBound
			defer func() { maxIOWaitDuration = originalBound }()
			fs, _, data := newPrefetchRangeFS(t, 0, 256<<10)
			d := fs.diskCache
			d.memoryPrefetchMax = 128 << 10
			d.objectSplit.parts = 1
			<-d.load.done
			storage := newBlackBoxStorage(fs.server.raw, func(_ string, start int64) bool { return start == 0 })
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.gate) })
			defer unblock()
			key := fs.pathToKey("object")
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{
				FilePath: "object", Size: size, Ranges: []PrefetchRange{{Size: size}},
			}))
			require.Eventually(t, func() bool {
				cur, _, _ := storage.counts(key)
				return cur == 1
			}, time.Second, time.Millisecond)
			download, _ := d.fileDownload(d.objectKey("object"), 16, 80)
			require.NotNil(t, download)

			var dst bytes.Buffer
			vec := &IOVector{FilePath: "object", Policy: SkipFullFilePreloads | SkipMemoryCache | SkipDiskCacheWrites,
				Entries: []IOEntry{{Offset: 16, Size: 64, WriterForRead: &dst}}}
			defer vec.Release()
			result := make(chan error, 1)
			readDone := make(chan struct{})
			go func() {
				defer close(readDone)
				result <- fs.Read(ctx, vec)
			}()
			defer func() {
				unblock()
				<-readDone
			}()
			require.Eventually(t, func() bool {
				download.mu.Lock()
				defer download.mu.Unlock()
				return len(download.waiters) == 1
			}, time.Second, time.Millisecond, "the read joins the file-backed range")
			select {
			case err := <-result:
				_, _, total := storage.counts(key)
				t.Fatalf("read returned before its joined range sent bytes: err=%v, source opens=%d", err, total)
			case <-time.After(3 * wait):
			}
			unblock()
			require.NoError(t, <-result)
			require.Equal(t, data[16:80], dst.Bytes())
			_, _, total := storage.counts(key)
			require.Equal(t, 1, total, "the joined range alone supplies the bytes")
		})
	}
}

func TestUnwritableIndexCacheKeepsTableCacheSeparate(t *testing.T) {
	const size = 9 << 20
	for _, tc := range []struct {
		name     string
		readOnly bool
	}{
		{name: "baseline_writable_index_cache"},
		{name: "unwritable_index_cache", readOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.readOnly && !permissionsEnforced(t) {
				t.Skip("filesystem permissions are not enforced")
			}
			ctx := context.Background()
			config := indexCacheConfig(t, 64<<20, 64<<20)
			config.MemoryCapacity = ptrTo[toml.ByteSize](0)
			config.DiskPrefetchMemoryMax = ptrTo[toml.ByteSize](minDiskPrefetchMemoryMax)
			fs, err := NewS3FS(ctx, ObjectStorageArguments{
				Name: defines.SharedFileServiceName, Endpoint: "disk", Bucket: t.TempDir(),
			}, config, nil, false, false)
			require.NoError(t, err)
			defer fs.Close(ctx)
			<-fs.diskCache.load.done
			<-fs.indexFileCache.disk.load.done

			data := bytes.Repeat([]byte{0x59}, size)
			require.NoError(t, fs.Write(ctx, IOVector{FilePath: "object", Policy: SkipAllCache,
				Entries: []IOEntry{{Size: size, Data: data}}}))
			if tc.readOnly {
				path := fs.indexFileCache.disk.path
				require.NoError(t, os.Chmod(path, 0o500))
				defer func() { require.NoError(t, os.Chmod(path, 0o755)) }()
			}
			vec := &IOVector{FilePath: "object", Policy: IndexFileObject | SkipMemoryCache,
				Entries: []IOEntry{{Size: size}}}
			defer vec.Release()
			require.NoError(t, fs.Read(ctx, vec))
			require.True(t, bytes.Equal(data, vec.Entries[0].Data), "the index read should return the whole object")
			fs.diskCache.prefetchObjects.reads.Wait()
			require.Zero(t, fs.diskCache.cache.Used(), "an index-object read should take no space in the table-data cache")
		})
	}
}

type reviewFailOnceWriter struct {
	err   error
	calls int
}

func (w *reviewFailOnceWriter) Write(data []byte) (int, error) {
	w.calls++
	if w.calls == 1 {
		return 0, w.err
	}
	return len(data), nil
}

func TestSplitCacheReadReturnsWriterError(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
	}{
		{name: "baseline_source_read", policy: SkipDiskCache},
		{name: "read_across_two_cached_ranges"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const part = 64 << 10
			fs, storage, data := newPrefetchRangeFS(t, 0, 4*part)
			d := fs.diskCache
			<-d.load.done
			ctx := context.Background()
			require.NoError(t, d.Update(ctx, &IOVector{FilePath: "object", Entries: []IOEntry{
				{Size: part, Data: data[:part]},
				{Offset: part, Size: part, Data: data[part : 2*part]},
			}}, false))
			const offset, size = part - 16, 32
			rs, ok := d.ranges.covering(d.objectKey("object"), offset, size)
			require.True(t, ok)
			require.Len(t, rs, 2)
			wantErr := errors.New("review: caller's writer rejected output")
			writer := &reviewFailOnceWriter{err: wantErr}
			vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipMemoryCache | tc.policy,
				Entries: []IOEntry{{Offset: offset, Size: size, WriterForRead: writer}}}
			defer vec.Release()
			err := fs.Read(ctx, vec)
			require.ErrorIs(t, err, wantErr, "a caller writer failure should be returned; writer calls=%d, source opens=%d", writer.calls, storage.reads.Load())
			require.Equal(t, 1, writer.calls)
		})
	}
}

// wantSources is the table of the shared sources a read may take its bytes from, by its
// policy, whether it is a read of a download and whether it streams an entry larger than
// a prefetch object buffers.
func wantSources(s *S3FS, p Policy, download, streamsLarge bool) readSources {
	// BC15: the reads of a download join no prefetch object. A large streamed entry opens
	// its own reader.
	disk := s.diskCache != nil && !p.Any(SkipDiskCacheReads) && !download && !streamsLarge
	// BC13: index-file cache reads are gated by SkipCacheReads, its fills by
	// SkipCacheWrites. The bytes of an index object are cached in the index-file cache
	// only, never in the disk cache.
	index := p.Any(IndexFileObject) && s.cachesIndexObjects()
	tier := diskTier
	switch {
	case index && p&SkipCacheWrites != SkipCacheWrites:
		tier = indexTier
	case index, p.Any(SkipDiskCacheWrites):
		tier = noTier
	}
	return readSources{
		indexRead: index && p&SkipCacheReads != SkipCacheReads,
		tier:      tier,
		flights:   disk && p.Any(ColumnPrefetch, ExactRead),
		window:    disk && tier == diskTier,
		// BC10: a read that caches its bytes joins only objects that cache theirs.
		caches: p.CacheIOEntry() && tier == diskTier,
		fetch:  disk,
	}
}

func TestSourcesForTable(t *testing.T) {
	fs, _, _ := newIndexObjectFS(t, true)
	for p := Policy(0); p < IndexFileObject<<1; p++ {
		for _, download := range []bool{false, true} {
			for _, streams := range []int64{0, 1, fs.diskCache.memoryPrefetchMax + 1} {
				ctx := context.Background()
				if download {
					ctx = withDownloadStarted(ctx, func() {})
				}
				vec := &IOVector{FilePath: "object", Policy: p, Entries: []IOEntry{{Size: 1}}}
				if streams > 0 {
					vec.Entries[0].Size, vec.Entries[0].WriterForRead = streams, new(bytes.Buffer)
				}
				require.Equal(t, wantSources(fs, p, download, streams > fs.diskCache.memoryPrefetchMax), fs.sourcesFor(ctx, vec),
					"policy %b, download %v, streams %v", p, download, streams)
			}
		}
	}
}

func TestCloseCancelsUnclaimedPrefetchPublication(t *testing.T) {
	for _, tc := range []struct {
		name string
		size int64
	}{
		{name: "ordinary_async_entry", size: 64 << 10},
		{name: "file_backed_range", size: 144 << 10},
		{name: "memory_backed_range", size: 64 << 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newPrefetchRangeFS(t, 0, 256<<10)
			d := fs.diskCache
			d.memoryPrefetchMax = 128 << 10
			<-d.load.done

			atPublication := make(chan struct{})
			resume := make(chan struct{})
			release := sync.OnceFunc(func() { close(resume) })
			defer release()
			d.hooks.beforeFilePublication = func() {
				close(atPublication)
				<-resume
			}
			if tc.name == "ordinary_async_entry" {
				vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Size: tc.size, Data: data[:tc.size]}}}
				require.NoError(t, d.Update(ctx, vec, true))
			} else {
				require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: tc.size,
					Ranges: []PrefetchRange{{Size: tc.size}}}))
			}
			select {
			case <-atPublication:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			canceled := d.pub.ctx.Done()
			if tc.name != "ordinary_async_entry" {
				objects := &d.prefetchObjects
				objects.mu.Lock()
				active := slices.Clone(objects.m[d.objectKey("object")])
				objects.mu.Unlock()
				require.Len(t, active, 1)
				switch flight := active[0].(type) {
				case *memoryPrefetchObject:
					canceled = flight.ctx.Done()
				case *filePrefetchObject:
					canceled = flight.ctx.Done()
				default:
					t.Fatalf("unexpected range owner %T", flight)
				}
			}

			closed := make(chan struct{})
			go func() {
				d.Close(ctx)
				close(closed)
			}()
			select {
			case <-canceled:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			release()
			select {
			case <-closed:
				require.NoError(t, ctx.Err())
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}

			path := d.pathForIOEntry("object", IOEntry{Size: tc.size})
			require.NoFileExists(t, path, "close canceled the flight before it could claim publication")
		})
	}
}

func TestSplitReadReservesFirstConnectionForNeededPart(t *testing.T) {
	for _, offset := range []int64{0, 128 << 10} {
		name := "needed_prefix"
		if offset > 0 {
			name = "needed_suffix"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newObjectSplitFS(t, objectSplitConfig{parts: 2, min: 128 << 10}, 300<<10)
			d := fs.diskCache
			<-d.load.done
			limitRoom(t, fs.server, 1, 1)

			opened := make(chan int64, 2)
			storage := newBlackBoxStorage(fs.server.raw, func(_ string, start int64) bool {
				opened <- start
				return true
			})
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.gate) })
			defer unblock()

			result := make(chan error, 1)
			go func() {
				vec := &IOVector{FilePath: "object", Entries: []IOEntry{{Offset: offset, Size: 16}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				if err == nil && !bytes.Equal(data[offset:offset+16], vec.Entries[0].Data) {
					err = fmt.Errorf("incorrect bytes at offset %d", offset)
				}
				result <- err
			}()

			var first int64
			select {
			case first = <-opened:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			require.Equal(t, 1, heldRoom(fs.server))
			unblock()
			require.NoError(t, <-result)
			flushPublishers(fs)
			require.Zero(t, heldRoom(fs.server))
			require.Equal(t, offset, first, "with room for one the read's part must start first")
		})
	}
}

// An exact read of bytes a stalled file-backed fill holds joins it and waits for its
// bytes; one that skips disk-cache reads cannot read what the fill caches and reads its
// bytes itself.
func TestFileRangeReadJoinsStalledFileFill(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy Policy
		joins  bool
	}{
		{name: "bypass_disk_reads", policy: SkipDiskCacheReads},
		{name: "can_read_disk_cache", joins: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newPrefetchRangeFS(t, 0, 256<<10)
			d := fs.diskCache
			d.memoryPrefetchMax = 128 << 10
			<-d.load.done

			open := make(chan struct{})
			opened := sync.OnceFunc(func() { close(open) })
			storage := newBlackBoxStorage(fs.server.raw, func(_ string, offset int64) bool {
				if offset != 0 {
					return false
				}
				opened()
				return true
			})
			fs.server.raw = storage
			unblock := sync.OnceFunc(func() { close(storage.gate) })
			defer unblock()
			const size = 192 << 10
			require.NoError(t, fs.PrefetchObject(ctx, ObjectPrefetch{FilePath: "object", Size: size,
				Ranges: []PrefetchRange{{Size: size}}}))
			select {
			case <-open:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			download, _ := d.fileDownload(d.objectKey("object"), 16, 80)
			require.NotNil(t, download)

			vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead | SkipDiskCacheWrites | tc.policy,
				Entries: []IOEntry{{Offset: 16, Size: 64}}}
			defer vec.Release()
			readDone := make(chan error, 1)
			readFinished := make(chan struct{})
			go func() {
				defer close(readFinished)
				readDone <- fs.Read(ctx, vec)
			}()
			defer func() {
				unblock()
				<-readFinished
			}()
			var queuedWaiters int
			require.Eventually(t, func() bool {
				select {
				case <-readFinished:
					return true
				default:
				}
				download.mu.Lock()
				queuedWaiters = len(download.waiters)
				download.mu.Unlock()
				return queuedWaiters > 0
			}, 4*time.Second, time.Millisecond)

			unblock()
			require.NoError(t, <-readDone)
			require.Equal(t, data[16:80], vec.Entries[0].Data)
			require.Equal(t, tc.joins, queuedWaiters > 0, "the read waits on the file fill it joined")
		})
	}
}

func TestFlightBudgetDoesNotOverflow(t *testing.T) {
	for _, tc := range []struct {
		name  string
		limit toml.ByteSize
	}{
		{name: "default_sized_limit", limit: DefaultDiskPrefetchMemoryMax},
		{name: "large_limit", limit: 1 << 57},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prior := memoryAvailable
			memoryAvailable = func() (uint64, bool) { return 1 << 30, true }
			t.Cleanup(func() { memoryAvailable = prior })
			cfg := CacheConfig{DiskPrefetchMemoryMax: &tc.limit}
			fs, _, _ := newPrefetchRangeFS(t, 0, 1)
			fs.diskCache.memoryPrefetchMax = cfg.diskPrefetchMemoryMax()
			require.EqualValues(t, (uint64(1)<<30)/100*5, fs.server.memoryBudget(), "the measured ceiling should constrain an oversized configured ceiling")
		})
	}
}

// Overlapping exact reads larger than disk-prefetch-memory-max share one file-backed
// prefetch object (decision 2026-11-23).
func TestExactReadsShareLargeRange(t *testing.T) {
	for _, tc := range []struct {
		name string
		size int64
	}{
		{name: "memory_sized_range", size: 64 << 10},
		{name: "file_sized_range", size: 128<<10 + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fs, _, data := newPrefetchRangeFS(t, 0, 256<<10)
			fs.diskCache.memoryPrefetchMax = 128 << 10
			<-fs.diskCache.load.done

			// Exact column reads skip the grid. Use distinct start offsets so the
			// identical-request I/O merger leaves deduplication to the range owner.
			started := make(chan struct{}, 2)
			gate := newBlackBoxStorage(fs.server.raw, func(string, int64) bool {
				select {
				case started <- struct{}{}:
				default:
				}
				return true
			})
			fs.server.raw = gate
			unblock := sync.OnceFunc(func() { close(gate.gate) })
			defer unblock()
			read := func(offset, size int64, result chan<- error) {
				vec := &IOVector{FilePath: "object", Policy: ExactColumnsRead,
					Entries: []IOEntry{{Offset: offset, Size: size}}}
				defer vec.Release()
				err := fs.Read(ctx, vec)
				if err == nil && !bytes.Equal(data[offset:offset+size], vec.Entries[0].Data) {
					err = fmt.Errorf("incorrect bytes at offset %d", offset)
				}
				result <- err
			}
			results := make(chan error, 2)
			go read(0, tc.size, results)
			select {
			case <-started:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			go read(1, tc.size-1, results)
			// If the second reader opens its own range we have the counterexample.
			// Otherwise, observe it waiting for bytes on the registered range.
			require.Eventually(t, func() bool {
				if _, _, reads := gate.counts("object"); reads == 2 {
					return true
				}
				objects := &fs.diskCache.prefetchObjects
				objects.mu.Lock()
				defer objects.mu.Unlock()
				for _, object := range objects.m[fs.diskCache.objectKey("object")] {
					var waiters int
					switch flight := object.(type) {
					case *memoryPrefetchObject:
						flight.mu.Lock()
						waiters = len(flight.waiters)
						flight.mu.Unlock()
					case *filePrefetchObject:
						if download := flight.joinableDownload(); download != nil {
							download.mu.Lock()
							waiters = len(download.waiters)
							download.mu.Unlock()
						}
					}
					if waiters == 2 {
						return true
					}
				}
				return false
			}, 4*time.Second, time.Millisecond)
			unblock()
			for range 2 {
				require.NoError(t, <-results)
			}
			_, _, reads := gate.counts("object")
			require.Equal(t, 1, reads, "overlapping exact reads should share a registered download")
		})
	}
}

// CloseOwnedETL closes a file service GetForETL created for a local path or a path with
// service arguments, and leaves the named service GetForETL returns open.
func TestCloseOwnedETL(t *testing.T) {
	ctx := WithSQLFileAccess(context.Background(), "sys", "moadmin")
	shared := &closeCountingETLFS{name: "etl"}
	services, err := NewFileServices("etl", shared)
	require.NoError(t, err)

	got, _, err := GetForETL(ctx, services, "etl:/a.csv")
	require.NoError(t, err)
	require.Same(t, shared, got)
	CloseOwnedETL(ctx, "etl:/a.csv", got)
	require.Zero(t, shared.closed.Load(), "the named service is shared")

	for _, path := range []string{
		"/tmp/a.csv",
		"s3-opts,endpoint=http://127.0.0.1:1,region=r,bucket=b,key=k,secret=s:/a.csv",
	} {
		owned := &closeCountingETLFS{name: "owned"}
		CloseOwnedETL(ctx, path, owned)
		require.Equal(t, int64(1), owned.closed.Load(), path)
	}

	CloseOwnedETL(ctx, "/tmp/a.csv", nil)
}
