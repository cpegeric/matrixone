// Copyright 2021 - 2022 Matrix Origin
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

package function

import (
	"encoding/json"
	"os"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/vectorindex"
	"github.com/matrixorigin/matrixone/pkg/vectorindex/hnsw"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
)

func hnswCdcUpdate(ivecs []*vector.Vector, result vector.FunctionResultWrapper, proc *process.Process, length int, selectList *FunctionSelectList) error {

	if len(ivecs) != 4 {
		return moerr.NewInvalidInput(proc.Ctx, "number of arguments != 4")
	}

	os.Stderr.WriteString("hnsCdcUpdate START\n")
	dbVec := vector.GenerateFunctionStrParameter(ivecs[0])
	tblVec := vector.GenerateFunctionStrParameter(ivecs[1])
	dimVec := vector.GenerateFunctionFixedTypeParameter[int32](ivecs[2])
	cdcVec := vector.GenerateFunctionStrParameter(ivecs[3])

	for i := uint64(0); i < uint64(length); i++ {
		dbname, isnull := dbVec.GetStrValue(i)
		if isnull {
			return moerr.NewInvalidInput(proc.Ctx, "dbname is null")
		}

		tblname, isnull := tblVec.GetStrValue(i)
		if isnull {
			return moerr.NewInvalidInput(proc.Ctx, "table name is null")

		}

		dim, isnull := dimVec.GetValue(i)
		if isnull {
			return moerr.NewInvalidInput(proc.Ctx, "dimension is null")
		}

		cdcstr, isnull := cdcVec.GetStrValue(i)
		if isnull {
			return moerr.NewInvalidInput(proc.Ctx, "cdc is null")
		}

		var cdc vectorindex.VectorIndexCdc[float32]
		err := json.Unmarshal([]byte(cdcstr), &cdc)
		if err != nil {
			return moerr.NewInvalidInput(proc.Ctx, "cdc is not json object")
		}
		// hnsw sync
		//os.Stderr.WriteString(fmt.Sprintf("db=%s, table=%s, dim=%d, json=%s\n", dbname, tblname, dim, cdcstr))
		err = hnsw.CdcSync(proc, string(dbname), string(tblname), dim, &cdc)
		if err != nil {
			return err
		}
	}

	os.Stderr.WriteString("hnsCdcUpdate END\n")
	return nil
}
