// Copyright 2021 Matrix Origin
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

package loopjoin

import (
	"github.com/matrixorigin/matrixone/pkg/common/mpool"
	"github.com/matrixorigin/matrixone/pkg/common/reuse"
	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	"github.com/matrixorigin/matrixone/pkg/sql/colexec"
	"github.com/matrixorigin/matrixone/pkg/vm"
	"github.com/matrixorigin/matrixone/pkg/vm/message"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
)

var _ vm.Operator = new(LoopJoin)

const (
	Build = iota
	Probe
	End
)

const (
	LoopInner = iota
	LoopAnti
	LoopLeft
	LoopMark
	LoopSemi
	LoopSingle
)

type container struct {
	state    int
	probeIdx int
	batIdx   int
	inbat    *batch.Batch
	rbat     *batch.Batch
	joinBat  *batch.Batch
	expr     colexec.ExpressionExecutor
	cfs      []func(*vector.Vector, *vector.Vector, int64, int) error
	mp       *message.JoinMap
}

type LoopJoin struct {
	ctr        container
	Typs       []types.Type
	Cond       *plan.Expr
	Result     []colexec.ResultPos
	JoinMapTag int32
	JoinType   int
	MarkPos    int

	vm.OperatorBase
}

func (loopJoin *LoopJoin) GetOperatorBase() *vm.OperatorBase {
	return &loopJoin.OperatorBase
}

func init() {
	reuse.CreatePool[LoopJoin](
		func() *LoopJoin {
			return &LoopJoin{}
		},
		func(a *LoopJoin) {
			*a = LoopJoin{}
		},
		reuse.DefaultOptions[LoopJoin]().
			WithEnableChecker(),
	)
}

func (loopJoin LoopJoin) TypeName() string {
	return opName
}

func NewArgument() *LoopJoin {
	return reuse.Alloc[LoopJoin](nil)
}

func (loopJoin *LoopJoin) Release() {
	if loopJoin != nil {
		reuse.Free[LoopJoin](loopJoin, nil)
	}
}

func (loopJoin *LoopJoin) Reset(proc *process.Process, pipelineFailed bool, err error) {
	ctr := &loopJoin.ctr

	ctr.resetExprExecutor()
	ctr.cleanHashMap()
	ctr.state = Build
	ctr.inbat = nil
}

func (loopJoin *LoopJoin) Free(proc *process.Process, pipelineFailed bool, err error) {
	ctr := &loopJoin.ctr

	ctr.cleanBatch(proc.Mp())
	ctr.cleanExprExecutor()

}

func (loopJoin *LoopJoin) ExecProjection(proc *process.Process, input *batch.Batch) (*batch.Batch, error) {
	return input, nil
}

func (ctr *container) cleanBatch(mp *mpool.MPool) {
	if ctr.rbat != nil {
		ctr.rbat.Clean(mp)
		ctr.rbat = nil
	}
	if ctr.joinBat != nil {
		ctr.joinBat.Clean(mp)
		ctr.joinBat = nil
	}
}

func (ctr *container) resetExprExecutor() {
	if ctr.expr != nil {
		ctr.expr.ResetForNextQuery()
	}
}

func (ctr *container) cleanExprExecutor() {
	if ctr.expr != nil {
		ctr.expr.Free()
		ctr.expr = nil
	}
	ctr.expr = nil
}

func (ctr *container) cleanHashMap() {
	if ctr.mp != nil {
		ctr.mp.Free()
		ctr.mp = nil
	}
}
