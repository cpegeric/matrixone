// Copyright 2022 Matrix Origin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mergedelete

import (
	"bytes"
	"context"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/vm/process"

	"github.com/matrixorigin/matrixone/pkg/catalog"
	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/objectio"
	"github.com/matrixorigin/matrixone/pkg/sql/colexec"
	"github.com/matrixorigin/matrixone/pkg/testutil"
	"github.com/matrixorigin/matrixone/pkg/vm"
	"github.com/matrixorigin/matrixone/pkg/vm/engine"
	"github.com/stretchr/testify/require"
)

type mockRelation struct {
	engine.Relation
	result *batch.Batch
}

func (e *mockRelation) Delete(ctx context.Context, b *batch.Batch, attrName string) error {
	e.result = b
	return nil
}

func TestString(t *testing.T) {
	buf := new(bytes.Buffer)
	arg := new(MergeDelete)
	arg.String(buf)
}

func TestOpType(t *testing.T) {
	arg := new(MergeDelete)
	require.Equal(t, arg.OpType(), vm.MergeDelete)
}

func TestMergeDelete(t *testing.T) {
	proc := testutil.NewProc(t)
	proc.Ctx = context.TODO()
	metaLocBat0 := &batch.Batch{
		Attrs: []string{
			catalog.BlockMetaOffset,
		},
		Vecs: []*vector.Vector{
			testutil.MakeInt64Vector([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, nil),
		},
	}
	bytes, err := metaLocBat0.MarshalBinary()
	require.Nil(t, err)

	vcu32, _ := vector.NewConstFixed(types.T_uint32.ToType(), uint32(15), 1, proc.GetMPool())
	batch1 := &batch.Batch{
		Attrs: []string{
			catalog.BlockMeta_Delete_ID,
			catalog.BlockMeta_DeltaLoc,
			catalog.BlockMeta_Type,
			catalog.BlockMeta_Partition,
			catalog.BlockMeta_Deletes_Length,
		},
		Vecs: []*vector.Vector{
			testutil.MakeTextVector([]string{"mock_block_id0"}, nil),
			testutil.MakeTextVector([]string{string(bytes)}, nil),
			testutil.MakeInt8Vector([]int8{0}, nil),
			testutil.MakeInt32Vector([]int32{0}, nil),
			vcu32,
		},
	}
	batch1.SetRowCount(1)
	uuid1 := objectio.NewSegmentid()
	blkId1 := objectio.NewBlockid(uuid1, 0, 0)
	metaLocBat1 := &batch.Batch{
		Attrs: []string{
			catalog.Row_ID,
		},
		Vecs: []*vector.Vector{
			testutil.MakeRowIdVector([]types.Rowid{
				objectio.NewRowid(blkId1, 0),
				objectio.NewRowid(blkId1, 1),
				objectio.NewRowid(blkId1, 2),
				objectio.NewRowid(blkId1, 3),
				objectio.NewRowid(blkId1, 4),
				objectio.NewRowid(blkId1, 5),
				objectio.NewRowid(blkId1, 6),
				objectio.NewRowid(blkId1, 7),
				objectio.NewRowid(blkId1, 8),
				objectio.NewRowid(blkId1, 9),
				objectio.NewRowid(blkId1, 10),
				objectio.NewRowid(blkId1, 11),
				objectio.NewRowid(blkId1, 12),
				objectio.NewRowid(blkId1, 13),
				objectio.NewRowid(blkId1, 14),
			}, nil),
		},
	}
	bytes1, err := metaLocBat1.MarshalBinary()
	require.Nil(t, err)

	metaLocBat2 := &batch.Batch{
		Attrs: []string{
			catalog.BlockMetaOffset,
		},
		Vecs: []*vector.Vector{
			testutil.MakeInt64Vector([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, nil),
		},
	}
	bytes2, err := metaLocBat2.MarshalBinary()
	require.Nil(t, err)

	metaLocBat3 := &batch.Batch{
		Attrs: []string{
			catalog.BlockMeta_DeltaLoc,
		},
		Vecs: []*vector.Vector{
			testutil.MakeTextVector([]string{"d:magic:15"}, nil),
		},
	}
	bytes3, err := metaLocBat3.MarshalBinary()
	require.Nil(t, err)

	vcu32_2, _ := vector.NewConstFixed(types.T_uint32.ToType(), uint32(45), 3, proc.GetMPool())
	batch2 := &batch.Batch{
		Attrs: []string{
			catalog.BlockMeta_Delete_ID,
			catalog.BlockMeta_DeltaLoc,
			catalog.BlockMeta_Type,
			catalog.BlockMeta_Partition,
			catalog.BlockMeta_Deletes_Length,
		},
		Vecs: []*vector.Vector{
			testutil.MakeTextVector([]string{"mock_block_id1", "mock_block_id2", "mock_block_id3"}, nil),
			testutil.MakeTextVector([]string{string(bytes1), string(bytes2), string(bytes3)}, nil),
			testutil.MakeInt8Vector([]int8{0, 1, 2}, nil),
			testutil.MakeInt32Vector([]int32{0, 0, 0}, nil),
			vcu32_2,
		},
	}
	batch2.SetRowCount(3)

	argument1 := MergeDelete{
		ctr: container{
			delSource: &mockRelation{},
			bat:       &batch.Batch{},
		},
		AddAffectedRows: true,
		OperatorBase: vm.OperatorBase{
			OperatorInfo: vm.OperatorInfo{
				Idx:     0,
				IsFirst: false,
				IsLast:  false,
			},
		},
	}

	// require.NoError(t, argument1.Prepare(proc))
	argument1.OpAnalyzer = process.NewAnalyzer(0, false, false, "mergedelete")
	resetChildren(&argument1, batch1)
	_, err = vm.Exec(&argument1, proc)
	require.NoError(t, err)
	require.Equal(t, uint64(15), argument1.GetAffectedRows())

	argument1.Reset(proc, false, err)
	resetChildren(&argument1, batch2)
	_, err = vm.Exec(&argument1, proc)
	require.NoError(t, err)
	require.Equal(t, uint64(60), argument1.GetAffectedRows())

	argument1.ctr.affectedRows = 0
	argument1.Reset(proc, false, err)
	resetChildren(&argument1, nil)
	_, err = vm.Exec(&argument1, proc)
	require.NoError(t, err)
	require.Equal(t, uint64(0), argument1.GetAffectedRows())

	argument2 := MergeDelete{
		ctr: container{
			delSource:    &mockRelation{},
			bat:          &batch.Batch{},
			affectedRows: 0,
		},
		AddAffectedRows: true,
		OperatorBase: vm.OperatorBase{
			OperatorInfo: vm.OperatorInfo{
				Idx:     0,
				IsFirst: false,
				IsLast:  false,
			},
		},
	}

	argument2.Reset(proc, false, err)
	resetChildren(&argument2, batch2)
	argument2.OpAnalyzer = process.NewAnalyzer(0, false, false, "mergedelete")
	_, err = vm.Exec(&argument2, proc)
	require.NoError(t, err)
	require.Equal(t, uint64(45), argument2.GetAffectedRows())

	// free resource
	argument1.Free(proc, false, nil)
	metaLocBat0.Clean(proc.GetMPool())
	metaLocBat1.Clean(proc.GetMPool())
	metaLocBat2.Clean(proc.GetMPool())
	metaLocBat3.Clean(proc.GetMPool())
	batch1.Clean(proc.GetMPool())
	batch2.Clean(proc.GetMPool())
	require.Equal(t, int64(0), proc.GetMPool().CurrNB())
}

func resetChildren(arg *MergeDelete, bat *batch.Batch) {
	op := colexec.NewMockOperator().WithBatchs([]*batch.Batch{bat})
	arg.Children = nil
	arg.AppendChild(op)
}
