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

package intersectall

import (
	"testing"

	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/sql/colexec"
	"github.com/matrixorigin/matrixone/pkg/testutil"
	"github.com/matrixorigin/matrixone/pkg/vm"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
	"github.com/stretchr/testify/require"
)

type intersectAllTestCase struct {
	proc *process.Process
	arg  *IntersectAll
}

func TestIntersectAll(t *testing.T) {
	proc := testutil.NewProcess(t)
	// [2 rows + 2 row, 3 columns] minus [1 row + 2 rows, 3 columns]
	/*
		{1, 2, 3}				{1, 2, 3}	  {1, 2, 3}
		{1, 2, 3} intersect all {4, 5, 6} ==> {1, 2, 3}
		{3, 4, 5}				{1, 2, 3}
		{3, 4, 5}
	*/
	var end vm.CallResult
	c := newIntersectAllTestCase(proc)

	setProcForTest(proc, c.arg)
	err := c.arg.Prepare(c.proc)
	require.NoError(t, err)
	cnt := 0
	for {
		end, err = vm.Exec(c.arg, c.proc)
		require.NoError(t, err)
		result := end.Batch
		if result != nil && !result.IsEmpty() {
			cnt += result.RowCount()
			require.Equal(t, 3, len(result.Vecs))
		} else {
			break
		}
	}
	require.Equal(t, 2, cnt) // 1 row

	for _, child := range c.arg.Children {
		child.Reset(proc, false, nil)
	}
	c.arg.Reset(c.proc, false, nil)

	setProcForTest(proc, c.arg)
	err = c.arg.Prepare(c.proc)
	require.NoError(t, err)
	cnt = 0
	for {
		end, err = vm.Exec(c.arg, c.proc)
		require.NoError(t, err)
		result := end.Batch
		if result != nil && !result.IsEmpty() {
			cnt += result.RowCount()
			require.Equal(t, 3, len(result.Vecs))
		} else {
			break
		}
	}
	require.Equal(t, 2, cnt) // 1 row

	for _, child := range c.arg.Children {
		child.Reset(proc, false, nil)
		child.Free(proc, false, nil)
	}
	c.arg.Reset(c.proc, false, nil)
	c.arg.Free(c.proc, false, nil)
	c.proc.Free()
	require.Equal(t, int64(0), c.proc.Mp().CurrNB())
}

func newIntersectAllTestCase(proc *process.Process) intersectAllTestCase {
	arg := new(IntersectAll)
	arg.OperatorBase.OperatorInfo = vm.OperatorInfo{
		Idx:     0,
		IsFirst: false,
		IsLast:  false,
	}
	return intersectAllTestCase{
		proc: proc,
		arg:  arg,
	}
}

func setProcForTest(proc *process.Process, intersetAll *IntersectAll) {
	for _, child := range intersetAll.Children {
		child.Free(proc, false, nil)
	}
	intersetAll.Children = nil
	leftBatches := []*batch.Batch{
		testutil.NewBatchWithVectors(
			[]*vector.Vector{
				testutil.NewVector(2, types.T_int64.ToType(), proc.Mp(), false, []int64{1, 1}),
				testutil.NewVector(2, types.T_int64.ToType(), proc.Mp(), false, []int64{2, 2}),
				testutil.NewVector(2, types.T_int64.ToType(), proc.Mp(), false, []int64{3, 3}),
			}, nil),
		testutil.NewBatchWithVectors(
			[]*vector.Vector{
				testutil.NewVector(2, types.T_int64.ToType(), proc.Mp(), false, []int64{3, 3}),
				testutil.NewVector(2, types.T_int64.ToType(), proc.Mp(), false, []int64{4, 4}),
				testutil.NewVector(2, types.T_int64.ToType(), proc.Mp(), false, []int64{5, 5}),
			}, nil),
	}

	rightBatches := []*batch.Batch{
		testutil.NewBatchWithVectors(
			[]*vector.Vector{
				testutil.NewVector(1, types.T_int64.ToType(), proc.Mp(), false, []int64{1, 1}),
				testutil.NewVector(1, types.T_int64.ToType(), proc.Mp(), false, []int64{2, 2}),
				testutil.NewVector(1, types.T_int64.ToType(), proc.Mp(), false, []int64{3, 3}),
			}, nil),
		testutil.NewBatchWithVectors(
			[]*vector.Vector{
				testutil.NewVector(1, types.T_int64.ToType(), proc.Mp(), false, []int64{4}),
				testutil.NewVector(1, types.T_int64.ToType(), proc.Mp(), false, []int64{5}),
				testutil.NewVector(1, types.T_int64.ToType(), proc.Mp(), false, []int64{6}),
			}, nil),
	}

	leftChild := colexec.NewMockOperator().WithBatchs(leftBatches)
	rightChild := colexec.NewMockOperator().WithBatchs(rightBatches)
	intersetAll.AppendChild(leftChild)
	intersetAll.AppendChild(rightChild)
}
