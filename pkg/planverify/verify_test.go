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

package planverify

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/matrixorigin/matrixone/pkg/pb/plan"
)

// The fixtures below construct the broken plan directly rather than running a rewrite that
// produces one. That is deliberate: a rule proven against a plan built in three lines keeps
// working after the rewrite that once produced it is fixed, refactored or deleted, whereas
// a fixture that depends on a live rewrite stops testing the rule the moment the rewrite
// stops emitting that shape.

func col(tag, pos int32) *plan.Expr {
	return &plan.Expr{Expr: &plan.Expr_Col{Col: &plan.ColRef{RelPos: tag, ColPos: pos}}}
}

func fn(name string, args ...*plan.Expr) *plan.Expr {
	return &plan.Expr{Expr: &plan.Expr_F{F: &plan.Function{
		Func: &plan.ObjectRef{ObjName: name}, Args: args,
	}}}
}

// scanJoinQuery builds the shape every case here starts from:
//
//	Step -> JOIN(tag 3) -> [ SCAN(tag 1), SCAN(tag 2) ]
func scanJoinQuery() *plan.Query {
	left := &plan.Node{NodeType: plan.Node_TABLE_SCAN, BindingTags: []int32{1}}
	right := &plan.Node{NodeType: plan.Node_TABLE_SCAN, BindingTags: []int32{2}}
	join := &plan.Node{
		NodeType:    plan.Node_JOIN,
		JoinType:    plan.Node_INNER,
		Children:    []int32{0, 1},
		BindingTags: []int32{3},
		OnList:      []*plan.Expr{fn("=", col(1, 0), col(2, 0))},
	}
	return &plan.Query{Nodes: []*plan.Node{left, right, join}, Steps: []int32{2}}
}

func requireClean(t *testing.T, q *plan.Query, stage Stage) {
	t.Helper()
	require.NoError(t, Verify(q, stage), "a well-formed plan must pass")
}

func requireViolation(t *testing.T, q *plan.Query, stage Stage, rule string) string {
	t.Helper()
	err := Verify(q, stage)
	require.Error(t, err, "expected rule %q to fire", rule)
	require.Contains(t, err.Error(), "["+rule+"]", "wrong rule fired: %v", err)
	return err.Error()
}

// TestVerify_CleanPlanPasses guards against the failure mode that makes a verifier useless:
// rules that fire on healthy plans get switched off, and then nothing is checked at all.
func TestVerify_CleanPlanPasses(t *testing.T) {
	requireClean(t, scanJoinQuery(), PreRemap)
	requireClean(t, scanJoinQuery(), PostRemap)
	require.NoError(t, Verify(nil, PreRemap), "no plan is not a violation")
	require.NoError(t, Verify(&plan.Query{}, PreRemap), "an empty plan is not a violation")
}

// TestFieldHonored_JoinFilterList is the shipped bug: a predicate parked on a field the
// executor never reads. EXPLAIN printed it, review passed it, and the rows it should have
// excluded came back.
func TestFieldHonored_JoinFilterList(t *testing.T) {
	q := scanJoinQuery()
	q.Nodes[2].FilterList = []*plan.Expr{fn(">", col(3, 1), col(3, 0))}

	msg := requireViolation(t, q, PreRemap, "field-honored")
	require.Contains(t, msg, "node 2")
	require.Contains(t, msg, "FilterList")
	require.Contains(t, msg, "silently dropped")

	// The same predicate on a FILTER node -- where compile does apply it -- is correct.
	fixed := scanJoinQuery()
	filter := &plan.Node{
		NodeType:   plan.Node_FILTER,
		Children:   []int32{2},
		FilterList: []*plan.Expr{fn(">", col(3, 1), col(3, 0))},
	}
	fixed.Nodes = append(fixed.Nodes, filter)
	fixed.Steps = []int32{3}
	requireClean(t, fixed, PreRemap)
}

// TestSortHasKeys: a SORT that sorts by nothing.
func TestSortHasKeys(t *testing.T) {
	q := scanJoinQuery()
	q.Nodes = append(q.Nodes, &plan.Node{NodeType: plan.Node_SORT, Children: []int32{2}})
	q.Steps = []int32{3}

	msg := requireViolation(t, q, PreRemap, "sort-has-keys")
	require.Contains(t, msg, "node 3")

	q.Nodes[3].OrderBy = []*plan.OrderBySpec{{Expr: col(3, 0), Flag: plan.OrderBySpec_DESC}}
	requireClean(t, q, PreRemap)
}

// TestColRefResolvable: a reference to a binding produced nowhere below the node.
func TestColRefResolvable(t *testing.T) {
	q := scanJoinQuery()
	// tag 99 exists nowhere in the plan.
	q.Nodes[2].ProjectList = []*plan.Expr{col(99, 0)}

	msg := requireViolation(t, q, PreRemap, "colref-resolvable")
	require.Contains(t, msg, "tag 99")
	require.Contains(t, msg, "ProjectList")

	// Post-remap a ColRef's RelPos is a child index, not a tag, so the rule must not run.
	require.NoError(t, Verify(q, PostRemap), "colref-resolvable is pre-remap only")

	// A reference to either input, or to the join's own binding, resolves.
	q.Nodes[2].ProjectList = []*plan.Expr{col(1, 0), col(2, 0), col(3, 0)}
	requireClean(t, q, PreRemap)
}

// TestColRefResolvable_NestedAndOrderBy: the walk has to look inside function arguments,
// lists and window specs, and at every expression-bearing field -- not just ProjectList.
// The original bug lived in a field nobody thought to check.
func TestColRefResolvable_NestedAndOrderBy(t *testing.T) {
	for _, tc := range []struct {
		name  string
		apply func(n *plan.Node)
		field string
	}{
		{"nested in a function", func(n *plan.Node) {
			n.ProjectList = []*plan.Expr{fn("round", fn("+", col(99, 0), col(1, 0)))}
		}, "ProjectList"},
		{"inside a list", func(n *plan.Node) {
			n.ProjectList = []*plan.Expr{{Expr: &plan.Expr_List{
				List: &plan.ExprList{List: []*plan.Expr{col(99, 1)}}}}}
		}, "ProjectList"},
		{"in OnList", func(n *plan.Node) {
			n.OnList = append(n.OnList, fn("=", col(1, 0), col(99, 0)))
		}, "OnList"},
		{"in a window spec", func(n *plan.Node) {
			n.WinSpecList = []*plan.Expr{{Expr: &plan.Expr_W{W: &plan.WindowSpec{
				WindowFunc:  fn("rank"),
				PartitionBy: []*plan.Expr{col(99, 0)},
			}}}}
		}, "WinSpecList"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q := scanJoinQuery()
			tc.apply(q.Nodes[2])
			msg := requireViolation(t, q, PreRemap, "colref-resolvable")
			require.Contains(t, msg, tc.field)
		})
	}
}

// TestDagReachable covers the arena's failure modes: a dangling child, a nil node, a cycle,
// and a step root that does not exist.
func TestDagReachable(t *testing.T) {
	t.Run("dangling child", func(t *testing.T) {
		q := scanJoinQuery()
		q.Nodes[2].Children = []int32{0, 42}
		msg := requireViolation(t, q, PreRemap, "dag-reachable")
		require.Contains(t, msg, "child id 42 does not exist")
	})

	t.Run("nil node", func(t *testing.T) {
		q := scanJoinQuery()
		q.Nodes[1] = nil
		msg := requireViolation(t, q, PreRemap, "dag-reachable")
		require.Contains(t, msg, "nil node")
	})

	t.Run("cycle", func(t *testing.T) {
		q := scanJoinQuery()
		q.Nodes[0].Children = []int32{2} // scan points back at the join
		msg := requireViolation(t, q, PreRemap, "dag-reachable")
		require.Contains(t, msg, "cycle")
	})

	t.Run("step out of range", func(t *testing.T) {
		q := scanJoinQuery()
		q.Steps = []int32{7}
		msg := requireViolation(t, q, PreRemap, "dag-reachable")
		require.Contains(t, msg, "out of range")
	})
}

// TestVerify_IgnoresAbandonedArenaNodes: plan.Nodes is append-only, so a successful rewrite
// leaves its pre-rewrite nodes behind, unreferenced and often malformed by design. Checking
// them would fail valid plans -- a mistake made once already, by a view-definition walk that
// scanned the arena and refused good views because of an orphan it found there.
func TestVerify_IgnoresAbandonedArenaNodes(t *testing.T) {
	q := scanJoinQuery()
	abandoned := &plan.Node{
		NodeType:   plan.Node_JOIN,
		FilterList: []*plan.Expr{fn(">", col(3, 1), col(3, 0))}, // would violate field-honored
		Children:   []int32{999},                                // and dag-reachable
	}
	q.Nodes = append(q.Nodes, abandoned) // in the arena, referenced by nothing

	require.NoError(t, Verify(q, PreRemap),
		"a node no step can reach is not part of the plan and must not be checked")
}

// TestVerify_ReportsEveryViolation: one bad rewrite usually breaks several things at once.
// Reporting them together turns one debugging round into one fix.
func TestVerify_ReportsEveryViolation(t *testing.T) {
	q := scanJoinQuery()
	q.Nodes[2].FilterList = []*plan.Expr{fn(">", col(3, 1), col(3, 0))}
	q.Nodes[2].ProjectList = []*plan.Expr{col(99, 0)}
	q.Nodes = append(q.Nodes, &plan.Node{NodeType: plan.Node_SORT, Children: []int32{2}})
	q.Steps = []int32{3}

	err := Verify(q, PreRemap)
	require.Error(t, err)
	msg := err.Error()
	for _, rule := range []string{"field-honored", "colref-resolvable", "sort-has-keys"} {
		require.Contains(t, msg, "["+rule+"]", "every violated rule must be reported")
	}
	require.Contains(t, msg, "3 violation(s)")
	require.True(t, strings.HasPrefix(msg, "plan verification failed (pre-remap)"))
}

// TestVerify_SelectedRules: a caller may run one rule, which is what makes a rule debuggable
// in isolation when it is suspected of a false positive.
func TestVerify_SelectedRules(t *testing.T) {
	q := scanJoinQuery()
	q.Nodes[2].FilterList = []*plan.Expr{fn(">", col(3, 1), col(3, 0))}
	q.Nodes[2].ProjectList = []*plan.Expr{col(99, 0)}

	err := Verify(q, PreRemap, SortHasKeys{})
	require.NoError(t, err, "only the named rule runs")

	err = Verify(q, PreRemap, FieldHonored{})
	require.Error(t, err)
	require.NotContains(t, err.Error(), "colref-resolvable")
}
