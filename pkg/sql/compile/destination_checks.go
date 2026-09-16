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
	"github.com/matrixorigin/matrixone/pkg/defines"
	"github.com/matrixorigin/matrixone/pkg/pb/pipeline"
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	"github.com/matrixorigin/matrixone/pkg/versionchecker"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
)

// init registers every remote destination version-check this package owns, in one place, so the
// send path (encodeRemoteScope / encodeScope) just loops versionchecker.RunAll. A new version-gated
// wire feature adds one entry here plus its detector, rather than a bespoke validator + a hand-added
// call in the encoders. Detectors live here (not in versionchecker) because they read sql-layer
// types the generic registry must not import; strict-write and group-concat register as Custom
// because they are not plain fail-closed gates (they also degrade / self-detect).
func init() {
	for _, c := range []versionchecker.Check{
		{MinVer: defines.MORPCVersion70, Needs: convBasesNeeded, Name: "row-dependent CONV bases"},
		{MinVer: defines.MORPCVersion71, Needs: integerDomainNeeded, Name: "checked integer arithmetic"},
		{MinVer: defines.MORPCVersion72, Needs: ipFunctionNeeded, Name: "corrected IP function semantics"},
		{Custom: validateStrictWriteDestination},
		{Custom: validateGroupConcatTimeZoneDestination},
	} {
		versionchecker.Register(c)
	}
}

// convBasesNeeded reports whether the pipeline carries row-dependent CONV bases remote expressions.
func convBasesNeeded(_ *process.Process, p *pipeline.Pipeline) bool {
	features, err := plan.RequiredRemoteExpressionFeatures(p)
	return err == nil && features.RowDependentConvBases
}

// integerDomainNeeded reports whether the pipeline carries checked-integer-arithmetic remote
// expressions. It re-scans the pipeline -- the shared cost of a generic registry.
func integerDomainNeeded(_ *process.Process, p *pipeline.Pipeline) bool {
	features, err := plan.RequiredRemoteExpressionFeatures(p)
	return err == nil && features.IntegerArithmeticDomains
}

// ipFunctionNeeded reports whether the pipeline carries corrected-IP-function remote expressions.
func ipFunctionNeeded(_ *process.Process, p *pipeline.Pipeline) bool {
	features, err := plan.RequiredRemoteExpressionFeatures(p)
	return err == nil && features.IPFunctionSemantics
}
