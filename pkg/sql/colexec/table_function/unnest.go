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

package table_function

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/bytejson"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	"github.com/matrixorigin/matrixone/pkg/sql/colexec"
	plan2 "github.com/matrixorigin/matrixone/pkg/sql/plan"
	"github.com/matrixorigin/matrixone/pkg/vm"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
)

const (
	unnestMode      = "both"
	unnestRecursive = false
)

func genFilterMap(filters []string) map[string]struct{} {
	if filters == nil {
		return defaultFilterMap
	}
	filterMap := make(map[string]struct{}, len(filters))
	for _, f := range filters {
		filterMap[f] = struct{}{}
	}
	return filterMap
}

type unnestParam struct {
	FilterMap map[string]struct{} `json:"filterMap"`
	ColName   string              `json:"colName"`
}

var (
	unnestDeniedFilters = []string{"col", "seq"}
	defaultFilterMap    = map[string]struct{}{
		"key":   {},
		"path":  {},
		"index": {},
		"value": {},
		"this":  {},
	}
)

type unnestState struct {
	inited bool
	param  unnestParam
	path   bytejson.Path
	outer  bool

	called bool
	// holding one call batch, unnestState owns it.
	batch *batch.Batch
}

func (u *unnestState) end(tf *TableFunction, proc *process.Process) error {
	return nil
}

func (u *unnestState) reset(tf *TableFunction, proc *process.Process) {
	if u.batch != nil {
		u.batch.CleanOnlyData()
	}
	u.called = false
}

func (u *unnestState) call(tf *TableFunction, proc *process.Process) (vm.CallResult, error) {
	var res vm.CallResult
	if u.called {
		return res, nil
	}
	res.Batch = u.batch
	u.called = true
	return res, nil
}

func (u *unnestState) free(tf *TableFunction, proc *process.Process, pipelineFailed bool, err error) {
	if u.batch != nil {
		u.batch.Clean(proc.Mp())
	}
}

func unnestPrepare(proc *process.Process, arg *TableFunction) (tvfState, error) {
	st := &unnestState{}
	st.param.ColName = string(arg.Params)
	if len(st.param.ColName) == 0 {
		st.param.ColName = "UNNEST_DEFAULT"
	}
	var filters []string
	for i := range arg.Attrs {
		denied := false
		for j := range unnestDeniedFilters {
			if arg.Attrs[i] == unnestDeniedFilters[j] {
				denied = true
				break
			}
		}
		if !denied {
			filters = append(filters, arg.Attrs[i])
		}
	}
	st.param.FilterMap = genFilterMap(filters)
	if len(arg.Args) < 1 || len(arg.Args) > 3 {
		return nil, moerr.NewInvalidInput(proc.Ctx, "unnest: argument number must be 1, 2 or 3")
	}
	if len(arg.Args) == 1 {
		vType := types.T_varchar.ToType()
		bType := types.T_bool.ToType()
		arg.Args = append(arg.Args, &plan.Expr{Typ: plan2.MakePlan2Type(&vType), Expr: &plan.Expr_Lit{Lit: &plan2.Const{Value: &plan.Literal_Sval{Sval: "$"}}}})
		arg.Args = append(arg.Args, &plan.Expr{Typ: plan2.MakePlan2Type(&bType), Expr: &plan.Expr_Lit{Lit: &plan2.Const{Value: &plan.Literal_Bval{Bval: false}}}})
	} else if len(arg.Args) == 2 {
		bType := types.T_bool.ToType()
		arg.Args = append(arg.Args, &plan.Expr{Typ: plan2.MakePlan2Type(&bType), Expr: &plan.Expr_Lit{Lit: &plan2.Const{Value: &plan.Literal_Bval{Bval: false}}}})
	}
	dt, err := json.Marshal(st.param)
	if err != nil {
		return nil, err
	}
	arg.Params = dt
	arg.ctr.executorsForArgs, err = colexec.NewExpressionExecutorsFromPlanExpressions(proc, arg.Args)
	arg.ctr.argVecs = make([]*vector.Vector, len(arg.Args))
	return st, err
}

// start calling tvf on nthRow and put the result in u.batch.  Note that current unnest impl will
// always return one batch per nthRow.
func (u *unnestState) start(tf *TableFunction, proc *process.Process, nthRow int, analyzer process.Analyzer) error {
	var err error
	jsonVec := tf.ctr.argVecs[0]

	if !u.inited {
		// do some typecheck craziness.  This really should have been done in prepare.
		if jsonVec.GetType().Oid != types.T_json && jsonVec.GetType().Oid != types.T_varchar {
			return moerr.NewInvalidInput(proc.Ctx, fmt.Sprintf("unnest: first argument must be json or string, but got %s", jsonVec.GetType().String()))
		}

		pathVec := tf.ctr.argVecs[1]
		if pathVec.GetType().Oid != types.T_varchar {
			return moerr.NewInvalidInput(proc.Ctx, fmt.Sprintf("unnest: second argument must be string, but got %s", pathVec.GetType().String()))
		}

		outerVec := tf.ctr.argVecs[2]
		if outerVec.GetType().Oid != types.T_bool {
			return moerr.NewInvalidInput(proc.Ctx, fmt.Sprintf("unnest: third argument must be bool, but got %s", outerVec.GetType().String()))
		}

		if !pathVec.IsConst() || !outerVec.IsConst() {
			return moerr.NewInvalidInput(proc.Ctx, "unnest: second and third arguments must be scalar")
		}

		if u.path, err = types.ParseStringToPath(pathVec.UnsafeGetStringAt(0)); err != nil {
			return err
		}

		u.outer = vector.MustFixedColWithTypeCheck[bool](outerVec)[0]
		u.batch = tf.createResultBatch()
		u.inited = true
	}

	u.called = false
	// clean up the batch
	u.batch.CleanOnlyData()
	for i := range u.batch.Vecs {
		u.batch.Vecs[i].SetClass(vector.FLAT)
	}

	switch jsonVec.GetType().Oid {
	case types.T_json:
		err = handle(u.batch, jsonVec, nthRow, &u.path, u.outer, &u.param, tf, proc, parseJson)
	case types.T_varchar:
		err = handle(u.batch, jsonVec, nthRow, &u.path, u.outer, &u.param, tf, proc, parseStr)
	default:
		panic("unreachable")
	}
	return err
}

func handle(bat *batch.Batch, jsonVec *vector.Vector, nthRow int,
	path *bytejson.Path, outer bool, param *unnestParam, arg *TableFunction,
	proc *process.Process, fn func(dt []byte) (bytejson.ByteJson, error)) error {
	// nthRow is the row number in the input batch, const batch is handled correctly in GetBytesAt
	if jsonVec.GetNulls().Contains(uint64(nthRow)) {
		return nil
	}
	json, err := fn(jsonVec.GetBytesAt(nthRow))
	if err != nil {
		return err
	}
	//
	ures, thiscnt, err := json.Unnest(path, outer, unnestRecursive, unnestMode, param.FilterMap)
	if err != nil {
		return err
	}

	err = makeBatch(bat, ures, param, arg, thiscnt, proc)
	if err != nil {
		return err
	}
	bat.SetRowCount(len(ures))
	return nil
}

func makeBatch(bat *batch.Batch, ures []bytejson.UnnestResult, param *unnestParam, arg *TableFunction, thiscnt int, proc *process.Process) error {
	for i := 0; i < len(ures); i++ {
		for j := 0; j < len(arg.Attrs); j++ {
			vec := bat.GetVector(int32(j))
			var err error
			switch arg.Attrs[j] {
			case "col":
				err = vector.AppendBytes(vec, []byte(param.ColName), false, proc.Mp())
			case "seq":
				err = vector.AppendFixed(vec, int32(i), false, proc.Mp())
			case "index":
				val, ok := ures[i][arg.Attrs[j]]
				if !ok || val == nil {
					err = vector.AppendFixed(vec, int32(0), true, proc.Mp())
				} else {
					intVal, _ := strconv.ParseInt(string(val), 10, 32)
					err = vector.AppendFixed(vec, int32(intVal), false, proc.Mp())
				}
			case "key", "path", "value":
				val, ok := ures[i][arg.Attrs[j]]
				err = vector.AppendBytes(vec, val, !ok || val == nil, proc.Mp())
			case "this":
				if thiscnt == 1 {
					if i == 0 {
						val, ok := ures[i][arg.Attrs[j]]
						err = vector.AppendBytes(vec, val, !ok || val == nil, proc.Mp())
						vec.SetClass(vector.CONSTANT)
						vec.SetLength(len(ures))
					}
				} else {
					val, ok := ures[i][arg.Attrs[j]]
					err = vector.AppendBytes(vec, val, !ok || val == nil, proc.Mp())
				}
			default:
				err = moerr.NewInvalidArg(proc.Ctx, "unnest: invalid column name:%s", arg.Attrs[j])
			}
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func parseJson(dt []byte) (bytejson.ByteJson, error) {
	ret := types.DecodeJson(dt)
	return ret, nil
}
func parseStr(dt []byte) (bytejson.ByteJson, error) {
	return types.ParseSliceToByteJson(dt)
}
