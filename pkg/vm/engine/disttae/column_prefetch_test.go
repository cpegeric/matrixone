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

package disttae

import (
	"context"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/common/mpool"
	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/fileservice"
	"github.com/matrixorigin/matrixone/pkg/objectio"
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	"github.com/matrixorigin/matrixone/pkg/vm/engine"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/disttae/logtailreplay"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/readutil"
	"github.com/stretchr/testify/require"
)

type columnPrefetchFS struct {
	fileservice.FileService
	reqs []fileservice.ObjectPrefetch
}

func (fs *columnPrefetchFS) PrefetchObject(_ context.Context, req fileservice.ObjectPrefetch) error {
	fs.reqs = append(fs.reqs, req)
	return nil
}

// newColumnPrefetchSource writes one object of 3 blocks, a small int32 column 0 and a large
// vector column 1, recording their ranges in the object or not, and returns a data source
// over its blocks.
func newColumnPrefetchSource(t *testing.T, recordColumns bool) (*LocalDisttaeDataSource, *columnPrefetchFS, objectio.ObjectStats) {
	mp := mpool.MustNewZero()
	t.Cleanup(func() { mpool.DeleteMPool(mp) })
	ctx := context.Background()
	mem, err := fileservice.NewMemoryFS("mem", fileservice.DisabledCacheConfig, nil)
	require.NoError(t, err)
	fs := &columnPrefetchFS{FileService: mem}

	id := objectio.NewObjectid()
	name := objectio.BuildObjectNameWithObjectID(&id)
	writer, err := objectio.NewObjectWriter(name, mem, 0, []uint16{0, 1}, nil)
	require.NoError(t, err)
	const rows = 200
	for blk := 0; blk < 3; blk++ {
		bat := batch.NewWithSize(2)
		bat.Vecs[0] = vector.NewVec(types.T_int32.ToType(), mp)
		bat.Vecs[1] = vector.NewVec(types.T_array_float32.ToType(), mp)
		for row := 0; row < rows; row++ {
			require.NoError(t, vector.AppendFixed(bat.Vecs[0], int32(blk*rows+row), false, mp))
			emb := make([]float32, 64)
			for i := range emb {
				emb[i] = float32(blk*rows + row + i)
			}
			require.NoError(t, vector.AppendArray(bat.Vecs[1], emb, false, mp))
		}
		bat.SetRowCount(rows)
		_, err := writer.Write(bat)
		require.NoError(t, err)
		bat.Clean(mp)
	}
	if recordColumns {
		writer.WriteObjectMeta(ctx, 3*rows, []objectio.ColumnMeta{objectio.BuildObjectColumnMeta(), objectio.BuildObjectColumnMeta()})
	}
	blocks, err := writer.WriteEnd(ctx)
	require.NoError(t, err)
	stats := writer.GetObjectStats()

	pState := logtailreplay.NewPartitionState("", true, 0, false)
	pState.UpdateDuration(types.BuildTS(0, 0), types.MaxTs())
	require.NoError(t, pState.HandleObjectEntry(ctx, nil, objectio.ObjectEntry{ObjectStats: stats, CreateTime: types.BuildTS(1, 0)}, false))

	data := readutil.NewBlockListRelationData(0)
	for i, b := range blocks {
		loc := objectio.BuildLocation(name, b.GetExtent(), rows, uint16(i))
		block := objectio.BlockInfo{BlockID: *objectio.BuildObjectBlockid(name, uint16(i)), MetaLoc: objectio.ObjectLocation(loc)}
		data.AppendBlockInfo(&block)
	}
	ls := &LocalDisttaeDataSource{fs: fs, pState: pState, rangeSlice: data.GetBlockInfoSlice(), iteratePhase: engine.Persisted}
	return ls, fs, stats
}

// A column-prefetch scan downloads the ranges of the columns it reads, however much of the
// object they are, and never the whole object; other scans keep whole objects.
func TestColumnPrefetchRangesOrWholeObject(t *testing.T) {
	for _, tc := range []struct {
		name     string
		policy   fileservice.Policy
		seqNums  []uint16
		ranges   int // ranges of the one request; -1 means the whole object; -2 no request
		bare     bool
		overHalf bool // the columns read are at least half of the object
	}{
		{name: "small column", policy: fileservice.SkipFullFilePreloads | fileservice.ColumnPrefetch, seqNums: []uint16{0}, ranges: 1},
		{name: "column over half of the object", policy: fileservice.SkipFullFilePreloads | fileservice.ColumnPrefetch, seqNums: []uint16{1}, ranges: 1, overHalf: true},
		{name: "all columns", policy: fileservice.SkipFullFilePreloads | fileservice.ColumnPrefetch, seqNums: []uint16{0, 1}, ranges: 2},
		{name: "columns not in the object", policy: fileservice.SkipFullFilePreloads | fileservice.ColumnPrefetch, seqNums: []uint16{7}, ranges: -2},
		{name: "column ranges not recorded", policy: fileservice.SkipFullFilePreloads | fileservice.ColumnPrefetch, seqNums: []uint16{0}, ranges: -2, bare: true},
		{name: "scan", seqNums: []uint16{0}, ranges: -1},
		{name: "whole preloads skipped", policy: fileservice.SkipFullFilePreloads, seqNums: []uint16{0}, ranges: -2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ls, fs, stats := newColumnPrefetchSource(t, !tc.bare)
			ctx := fileservice.WithFileServicePolicy(context.Background(), tc.policy)
			ls.batchPrefetch(ctx, tc.seqNums)
			if tc.ranges == -2 {
				require.Empty(t, fs.reqs)
				return
			}
			require.Len(t, fs.reqs, 1, "an object is asked for once")
			req := fs.reqs[0]
			require.Equal(t, stats.ObjectName().String(), req.FilePath)
			if tc.ranges == -1 {
				require.Empty(t, req.Ranges)
				require.Equal(t, int64(stats.Size()), req.Size)
				require.Equal(t, 3, req.Reads)
				require.Equal(t, 3, req.Blocks)
				return
			}
			require.Len(t, req.Ranges, tc.ranges)
			var size int64
			for _, r := range req.Ranges {
				size += r.Size
			}
			require.Equal(t, size, req.Size)
			require.Less(t, req.Size, int64(stats.Size()), "the columns not read are not downloaded")
			if tc.overHalf {
				require.GreaterOrEqual(t, req.Size*2, int64(stats.Size()))
			}
		})
	}
}

func TestSmallLocalScanPrefetchesColumnsOnReach(t *testing.T) {
	for _, small := range []bool{false, true} {
		name := "four_blocks"
		if small {
			name = "three_blocks"
		}
		t.Run(name, func(t *testing.T) {
			ls, fs, stats := newColumnPrefetchSource(t, true)
			defer ls.Close()
			ctx := context.Background()
			if !small {
				other, otherFS, otherStats := newColumnPrefetchSource(t, true)
				defer other.Close()
				// Add a readable block from a second object to exceed the local window cutoff.
				vec := &fileservice.IOVector{FilePath: otherStats.ObjectName().String(), Entries: []fileservice.IOEntry{{Size: -1}}}
				defer vec.Release()
				require.NoError(t, otherFS.Read(ctx, vec))
				require.NoError(t, fs.Write(ctx, *vec))
				ls.rangeSlice.AppendBlockInfo(other.rangeSlice.Get(0))
			}
			ls.table = &txnTable{db: &txnDatabase{}}
			ls.SetOrderBy(nil) // Apply the same four-block cutoff as NewLocalDataSource.
			ctx = fileservice.WithFileServicePolicy(ctx, fileservice.ExactColumnsRead|fileservice.ColumnPrefetch)
			blk, state, err := ls.Next(ctx, []string{"key"}, nil, []uint16{0}, 0, &readutil.MemPKFilter{}, nil, nil)
			require.NoError(t, err)
			require.Equal(t, engine.Persisted, state)
			require.NotNil(t, blk)
			var requests int
			for _, req := range fs.reqs {
				if req.FilePath == stats.ObjectName().String() {
					require.Len(t, req.Ranges, 1)
					requests++
				}
			}
			require.Positive(t, requests, "entering a narrow scan's first object should request its column range")
		})
	}
}

func TestSmallLocalScanPrefetchesColumnsAhead(t *testing.T) {
	for _, columns := range []bool{false, true} {
		name := "whole_object"
		if columns {
			name = "column_ranges"
		}
		t.Run(name, func(t *testing.T) {
			ls, fs, stats := newColumnPrefetchSource(t, true)
			defer ls.Close()
			ls.SetOrderBy(nil)
			var policy fileservice.Policy
			if columns {
				policy = fileservice.SkipFullFilePreloads | fileservice.ColumnPrefetch
			}
			ls.batchPrefetch(fileservice.WithFileServicePolicy(context.Background(), policy), []uint16{0})
			var requests int
			for _, req := range fs.reqs {
				if req.FilePath == stats.ObjectName().String() {
					require.Len(t, req.Ranges, 1)
					requests++
				}
			}
			if columns {
				require.Equal(t, 1, requests, "a three-block scan prefetches its column ranges ahead")
			} else {
				require.Zero(t, requests, "a three-block scan does not prefetch whole objects")
			}
		})
	}
}

func TestOrderedLocalScanSkipsColumnPrefetch(t *testing.T) {
	for _, tc := range []struct {
		name  string
		limit uint64
	}{
		{name: "unlimited_ordered_scan"},
		{name: "ordered_scan_with_limit", limit: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ls, fs, _ := newColumnPrefetchSource(t, true)
			defer ls.Close()
			typ := types.T_int32.ToType()
			ls.table = &txnTable{db: &txnDatabase{}, tableDef: &plan.TableDef{
				Cols: []*plan.ColDef{{Name: "k", Typ: plan.Type{Id: int32(typ.Oid)}, Seqnum: 0}},
			}}
			ls.Limit = tc.limit
			ls.SetOrderBy([]*plan.OrderBySpec{{Expr: &plan.Expr{
				Typ:  plan.Type{Id: int32(typ.Oid)},
				Expr: &plan.Expr_Col{Col: &plan.ColRef{Name: "k"}},
			}}})
			ctx := fileservice.WithFileServicePolicy(context.Background(), fileservice.ColumnPrefetch|fileservice.SkipFullFilePreloads)
			mp := mpool.MustNewZero()
			defer mpool.DeleteMPool(mp)
			out := batch.NewWithSize(1)
			defer out.Clean(mp)
			block, state, err := ls.Next(ctx, []string{"k"}, []types.Type{typ}, []uint16{0}, 0, &readutil.MemPKFilter{}, mp, out)
			require.NoError(t, err)
			require.Equal(t, engine.Persisted, state)
			require.NotNil(t, block)
			require.Empty(t, fs.reqs, "an ordered scan should issue zero column prefetches, got %d", len(fs.reqs))
		})
	}
}
