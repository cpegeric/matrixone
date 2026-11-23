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

package readutil

import (
	"context"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/matrixorigin/matrixone/pkg/common/mpool"
	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/fileservice"
	"github.com/matrixorigin/matrixone/pkg/objectio"
)

// writeTypedObject writes an object of 2 blocks whose column at position p has type
// colTypes[p] and seqnum seqs[p], and returns its first block and each column's range.
func writeTypedObject(t *testing.T, fs fileservice.FileService, colTypes []types.T, seqs []uint16) (*objectio.BlockInfo, map[uint16]fileservice.PrefetchRange) {
	mp := mpool.MustNewZero()
	t.Cleanup(func() { mpool.DeleteMPool(mp) })
	ctx := context.Background()
	const rows = 200
	rnd := rand.New(rand.NewSource(1))
	id := objectio.NewObjectid()
	name := objectio.BuildObjectNameWithObjectID(&id)
	writer, err := objectio.NewObjectWriter(name, fs, 0, seqs, nil)
	require.NoError(t, err)
	for range 2 {
		bat := batch.NewWithSize(len(colTypes))
		for p, typ := range colTypes {
			v := vector.NewVec(typ.ToType(), mp)
			for range rows {
				switch typ {
				case types.T_int64:
					require.NoError(t, vector.AppendFixed(v, rnd.Int63(), false, mp))
				case types.T_varchar:
					require.NoError(t, vector.AppendBytes(v, []byte(randomString(rnd, 300)), false, mp))
				case types.T_array_float32:
					f := make([]float32, 64)
					for i := range f {
						f[i] = rnd.Float32()
					}
					require.NoError(t, vector.AppendArray(v, f, false, mp))
				}
			}
			bat.Vecs[p] = v
		}
		bat.SetRowCount(rows)
		_, err := writer.Write(bat)
		require.NoError(t, err)
		bat.Clean(mp)
	}
	colMetas := make([]objectio.ColumnMeta, len(colTypes))
	for i := range colMetas {
		colMetas[i] = objectio.BuildObjectColumnMeta()
	}
	writer.WriteObjectMeta(ctx, 2*rows, colMetas)
	blocks, err := writer.WriteEnd(ctx)
	require.NoError(t, err)
	loc := objectio.BuildLocation(name, blocks[0].GetExtent(), rows, 0)
	blk := &objectio.BlockInfo{BlockID: *objectio.BuildObjectBlockid(name, 0), MetaLoc: objectio.ObjectLocation(loc)}
	meta, err := objectio.FastLoadObjectMeta(ctx, &loc, false, fs)
	require.NoError(t, err)
	extents := make(map[uint16]fileservice.PrefetchRange, len(seqs))
	for _, seq := range seqs {
		ext := meta.MustDataMeta().MustGetColumn(seq).Location()
		extents[seq] = fileservice.PrefetchRange{Offset: int64(ext.Offset()), Size: int64(ext.Length())}
	}
	return blk, extents
}

func randomString(rnd *rand.Rand, n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte('a' + rnd.Intn(26))
	}
	return string(b)
}

// Each column a scan reads is one range. A range may be read across the column before
// it when neither that column nor the one before it is big, it is not big, and the
// column between holds at most the bytes of the smaller of the two; columns are taken
// in their order in the object, whatever their seqnums.
func TestColumnRangesMarkGapsOfOneColumn(t *testing.T) {
	fs, err := fileservice.NewMemoryFS("mem", fileservice.DisabledCacheConfig, nil)
	require.NoError(t, err)
	colTypes := []types.T{
		types.T_int64, types.T_int64, types.T_int64, types.T_int64, types.T_int64,
		types.T_array_float32, types.T_int64, types.T_varchar, types.T_int64, types.T_int64,
	}
	seqs := []uint16{9, 3, 7, 0, 5, 1, 8, 2, 6, 4}
	blk, ext := writeTypedObject(t, fs, colTypes, seqs)
	col := func(p int) fileservice.PrefetchRange { return ext[seqs[p]] }
	for p := 1; p < len(seqs); p++ {
		require.Equal(t, col(p-1).Offset+col(p-1).Size, col(p).Offset, "the columns lie in their order")
	}
	require.Greater(t, col(7).Size, col(6).Size, "the varchar column holds more bytes than its neighbors")
	across := func(p int) fileservice.PrefetchRange {
		r := col(p)
		r.Across = col(p - 1).Size
		return r
	}
	for _, tc := range []struct {
		name      string
		positions []int
		want      []fileservice.PrefetchRange
	}{
		{"gaps of one column", []int{0, 2, 4}, []fileservice.PrefetchRange{col(0), across(2), across(4)}},
		{"the first two columns", []int{0, 1}, []fileservice.PrefetchRange{col(0), col(1)}},
		{"a gap of a vector column", []int{4, 6}, []fileservice.PrefetchRange{across(4), col(6)}},
		{"a vector column read", []int{3, 5}, []fileservice.PrefetchRange{across(3), col(5)}},
		{"a gap of more bytes than its neighbors", []int{6, 8}, []fileservice.PrefetchRange{col(6), col(8)}},
		{"after a column of more bytes than the gap", []int{7, 9}, []fileservice.PrefetchRange{col(7), across(9)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			read := make([]uint16, len(tc.positions))
			var want int64
			for i, p := range tc.positions {
				read[i] = seqs[p]
				want += col(p).Size
			}
			ranges, bytes, recorded, err := ColumnRanges(context.Background(), fs, blk, read)
			require.NoError(t, err)
			require.True(t, recorded)
			require.Equal(t, tc.want, ranges)
			require.Equal(t, want, bytes)
		})
	}
}
