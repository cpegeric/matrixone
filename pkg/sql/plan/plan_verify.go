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

package plan

import (
	"github.com/matrixorigin/matrixone/pkg/planverify"
)

// planVerifyEnabled reports whether the plan_verifier session variable is on.
//
// OFF is the default and the production setting: a violation FAILS THE QUERY, which is what
// a test run wants and what a served workload must never do. Reading the variable (rather
// than a build tag) is what lets a whole BVT run be verified with one `set global
// plan_verifier = 1`, which is where the interesting plans are -- thousands of real queries
// nobody hand-wrote.
//
// Any error resolving it means off. A verifier that fails closed would turn its own
// plumbing problems into query failures, which is precisely the risk this feature must not
// add to the serving path.
func (builder *QueryBuilder) planVerifyEnabled() bool {
	if builder == nil || builder.compCtx == nil {
		return false
	}
	val, err := builder.compCtx.ResolveVariable("plan_verifier", true, false)
	if err != nil {
		return false
	}
	switch v := val.(type) {
	case int64:
		return v != 0
	case int8:
		return v != 0
	case int:
		return v != 0
	}
	return false
}

// verifyPlan runs the structural checks when the variable is on, and is a single boolean
// test otherwise.
//
// It is called at two points with different rule sets, because remapAllColRefs changes what
// a column reference means: before it, RelPos is a binding tag that can be traced to the
// node producing it; after it, RelPos is a child index. See pkg/planverify.
func (builder *QueryBuilder) verifyPlan(stage planverify.Stage) error {
	if !builder.planVerifyEnabled() {
		return nil
	}
	return planverify.Verify(builder.qry, stage)
}
