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
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	"github.com/matrixorigin/matrixone/pkg/planverify"
)

// brokenQuery is a JOIN carrying a FilterList: an expression on a field compile never reads,
// which is the shipped bug pkg/planverify exists for.
func brokenQuery() *plan.Query {
	scan := &plan.Node{NodeType: plan.Node_TABLE_SCAN, BindingTags: []int32{1}}
	join := &plan.Node{
		NodeType:    plan.Node_JOIN,
		JoinType:    plan.Node_INNER,
		Children:    []int32{0},
		BindingTags: []int32{2},
		FilterList: []*plan.Expr{{Expr: &plan.Expr_Col{
			Col: &plan.ColRef{RelPos: 1, ColPos: 0}}}},
	}
	return &plan.Query{Nodes: []*plan.Node{scan, join}, Steps: []int32{1}}
}

func builderWithVerifier(t *testing.T, value any) *QueryBuilder {
	t.Helper()
	mock := NewMockCompilerContext(false)
	mock.ResolveVariableFunc = func(varName string, _, _ bool) (interface{}, error) {
		if varName == "plan_verifier" {
			return value, nil
		}
		return nil, nil
	}
	b := NewQueryBuilder(plan.Query_SELECT, mock, false, true)
	b.qry = brokenQuery()
	return b
}

// TestVerifyPlan_OffByDefault: the contract that keeps this feature off the serving path.
// A malformed plan must pass silently when the variable is unset or 0 -- verification is
// opt-in, and a violation fails the query, which a served workload must never risk.
func TestVerifyPlan_OffByDefault(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
	}{
		{"zero int64", int64(0)},
		{"zero int8", int8(0)},
		{"unset", nil},
		{"unexpected type", "1"}, // a string must not be read as on
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := builderWithVerifier(t, tc.value)
			require.False(t, b.planVerifyEnabled())
			require.NoError(t, b.verifyPlan(planverify.PreRemap),
				"a broken plan must pass while the verifier is off")
		})
	}
}

// TestVerifyPlan_ResolveErrorMeansOff: the verifier must fail OPEN. If reading the variable
// errors, its own plumbing problem must not become a query failure.
func TestVerifyPlan_ResolveErrorMeansOff(t *testing.T) {
	mock := NewMockCompilerContext(false)
	mock.ResolveVariableFunc = func(string, bool, bool) (interface{}, error) {
		return nil, moerr.NewInternalErrorNoCtx("variable subsystem unavailable")
	}
	b := NewQueryBuilder(plan.Query_SELECT, mock, false, true)
	b.qry = brokenQuery()

	require.False(t, b.planVerifyEnabled())
	require.NoError(t, b.verifyPlan(planverify.PreRemap))
}

// TestVerifyPlan_OnCatchesTheShippedBug closes the loop the unit tests in pkg/planverify
// cannot: that the session variable actually reaches the rules and that a violation is
// returned as an error, rather than logged and dropped.
func TestVerifyPlan_OnCatchesTheShippedBug(t *testing.T) {
	for _, value := range []any{int64(1), int8(1), 1} {
		b := builderWithVerifier(t, value)
		require.True(t, b.planVerifyEnabled())

		err := b.verifyPlan(planverify.PreRemap)
		require.Error(t, err, "plan_verifier=%v must catch a JOIN FilterList", value)
		require.Contains(t, err.Error(), "[field-honored]")
		require.Contains(t, err.Error(), "silently dropped")
	}
}

// TestVerifyPlan_OnAcceptsAWellFormedPlan: the same switch, on, must leave a correct plan
// alone -- otherwise BVT could never run with it enabled.
func TestVerifyPlan_OnAcceptsAWellFormedPlan(t *testing.T) {
	b := builderWithVerifier(t, int64(1))
	b.qry.Nodes[1].FilterList = nil
	require.NoError(t, b.verifyPlan(planverify.PreRemap))
	require.NoError(t, b.verifyPlan(planverify.PostRemap))
}
