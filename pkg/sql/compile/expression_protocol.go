// Copyright 2026 Matrix Origin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compile

import (
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	plan2 "github.com/matrixorigin/matrixone/pkg/sql/plan"
	"github.com/matrixorigin/matrixone/pkg/versionchecker"
	"github.com/matrixorigin/matrixone/pkg/vm/engine"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
)

// remoteWorkersSupportProtocol delegates to versionchecker.SupportProtocol; kept as the in-package
// name the existing constrain*/validate* callers already use.
func remoteWorkersSupportProtocol(proc *process.Process, workers engine.Nodes, minimum int64) (bool, error) {
	return versionchecker.SupportProtocol(proc, workers, minimum)
}

// constrainRemoteExpressionWorkers keeps a query whose remote expressions require a corrected
// semantics contract on a single CN while a rolling cluster still contains workers below that
// contract's MORPC version. It asks the shared destination-check registry for the highest version
// this query needs -- the same declarative entries the send boundary fails closed on -- so no
// per-feature list or switch lives here. The highest floor governs because degrading is
// all-or-nothing (ONECN), and the probe runs once per compile with no per-row execution cost.
func (c *Compile) constrainRemoteExpressionWorkers(qry *plan.Query) error {
	if c.execType != plan2.ExecTypeAP_MULTICN {
		return nil
	}
	floor, _ := versionchecker.MaxRequiredVersion(c.proc, qry)
	if floor == 0 {
		return nil
	}
	supported, err := remoteWorkersSupportProtocol(c.proc, c.cnList, floor)
	if err != nil {
		return err
	}
	if supported {
		return nil
	}
	c.execType = plan2.ExecTypeAP_ONECN
	c.cnList, err = c.scheduleQueryWorkers()
	return err
}
