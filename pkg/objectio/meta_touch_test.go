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

package objectio

import (
	"context"
	"fmt"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/common/mpool"
	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/fileservice"
	"github.com/matrixorigin/matrixone/pkg/fileservice/fscache"
	"github.com/stretchr/testify/require"
)

type touchRecorder struct {
	fileservice.FileService
	touched []string
}

func (r *touchRecorder) TouchMeta(filePath string, offset, size int64) {
	r.touched = append(r.touched, fmt.Sprintf("%s@%d+%d", filePath, offset, size))
}

// A hit of a meta or bloom filter in metaCache tells the file service of a use of its
// entry, at most once an interval; a load that reads it does not.
func TestMetaCacheHitTouchesEntry(t *testing.T) {
	ctx := context.Background()
	base, err := fileservice.NewMemoryFS("meta-touch", fileservice.DisabledCacheConfig, nil)
	require.NoError(t, err)
	t.Cleanup(func() { base.Close(ctx) })
	mp := mpool.MustNewZero()
	bat := batch.NewWithSize(1)
	bat.Vecs[0] = vector.NewVec(types.T_int64.ToType())
	t.Cleanup(func() { bat.Clean(mp); mpool.DeleteMPool(mp) })
	require.NoError(t, vector.AppendFixed(bat.Vecs[0], int64(42), false, mp))
	bat.SetRowCount(1)
	id := NewObjectid()
	name := BuildObjectNameWithObjectID(&id)
	writer, err := NewObjectWriter(name, base, 0, []uint16{0}, nil)
	require.NoError(t, err)
	_, err = writer.Write(bat)
	require.NoError(t, err)
	require.NoError(t, writer.WriteBF(0, 0, []byte("bloom"), 0))
	blocks, err := writer.WriteEnd(ctx)
	require.NoError(t, err)
	location := BuildLocation(name, blocks[0].GetExtent(), 1, 0)

	oldCache := metaCache
	metaCache = newMetaCache(fscache.ConstCapacity(1 << 20))
	t.Cleanup(func() { metaCache = oldCache })
	fs := &touchRecorder{FileService: base}

	meta, err := FastLoadObjectMeta(ctx, &location, false, fs)
	require.NoError(t, err)
	require.Empty(t, fs.touched, "a load that reads the meta")
	_, err = FastLoadObjectMeta(ctx, &location, false, fs)
	require.NoError(t, err)
	_, err = FastLoadObjectMeta(ctx, &location, false, fs)
	require.NoError(t, err)
	extent := location.Extent()
	require.Equal(t, []string{fmt.Sprintf("%s@%d+%d", name.String(), extent.Offset(), extent.Length())}, fs.touched,
		"one use an interval")

	fs.touched = nil
	_, err = FastLoadBF(ctx, location, false, fs)
	require.NoError(t, err)
	require.Empty(t, fs.touched)
	_, err = FastLoadBF(ctx, location, false, fs)
	require.NoError(t, err)
	bf := meta.MustDataMeta().BlockHeader().BFExtent()
	require.Equal(t, []string{fmt.Sprintf("%s@%d+%d", name.String(), bf.Offset(), bf.Length())}, fs.touched)
}
