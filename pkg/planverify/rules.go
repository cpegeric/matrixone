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
	"fmt"

	"github.com/matrixorigin/matrixone/pkg/pb/plan"
)

// ---------------------------------------------------------------------------------------
// field-honored
// ---------------------------------------------------------------------------------------

// ignoredField names a (node type, field) pair that the EXECUTOR DOES NOT READ. Setting one
// is always a bug: the value is silently discarded, and because EXPLAIN prints several of
// these fields the plan looks correct while behaving as though the value were absent.
type ignoredField struct {
	nodeType plan.Node_NodeType
	field    string
	// why records how the pair was established, because this table is only trustworthy if
	// every entry was traced through compile rather than assumed.
	why string
}

// ignoredFields is a DENYLIST, deliberately, not an allowlist of honored fields.
//
// An allowlist would be the stronger check and cannot be written honestly: it would require
// proving, for every node type and every field, that compile reads it -- and a wrong entry
// there rejects valid plans. A denylist only ever grows by evidence: each entry is a pair
// somebody traced to the point where the executor ignores it. It catches less and is never
// wrong, which is the right trade for a check that fails the query.
var ignoredFields = []ignoredField{
	{
		nodeType: plan.Node_JOIN,
		field:    "FilterList",
		why: "compileJoin reads OnList only; no planner path sets a JOIN FilterList and " +
			"compile never applies one, though EXPLAIN prints it as `Filter Cond`",
	},
}

// FieldHonored reports expressions parked on a field the executor ignores.
//
// The bug behind it: a lifted MATCH predicate was attached to Node_JOIN.FilterList. EXPLAIN
// showed `Filter Cond: (score > 0.5)` on the join, the plan read as correct in review, and
// the predicate was never evaluated -- the query returned rows it excluded. It survived a
// suite of tests because every one of them asserted `score > 0`, which every matching row
// satisfies, so a never-applied predicate was indistinguishable from a correct one.
type FieldHonored struct{}

func (FieldHonored) Name() string    { return "field-honored" }
func (FieldHonored) Stages() []Stage { return []Stage{PreRemap, PostRemap} }

func (r FieldHonored) Check(c *Ctx) []Violation {
	var out []Violation
	for _, id := range c.Nodes() {
		node := c.Node(id)
		if node == nil {
			continue
		}
		for _, ig := range ignoredFields {
			if node.NodeType != ig.nodeType {
				continue
			}
			if n := fieldLen(node, ig.field); n > 0 {
				out = append(out, Violation{
					Rule:   r.Name(),
					NodeID: id,
					Field:  fmt.Sprintf("%s.%s", node.NodeType, ig.field),
					Detail: fmt.Sprintf("%d expression(s) set on a field the executor ignores, "+
						"so they are silently dropped: %s", n, ig.why),
				})
			}
		}
	}
	return out
}

func fieldLen(node *plan.Node, field string) int {
	switch field {
	case "FilterList":
		return len(node.FilterList)
	case "OnList":
		return len(node.OnList)
	case "ProjectList":
		return len(node.ProjectList)
	case "GroupBy":
		return len(node.GroupBy)
	case "AggList":
		return len(node.AggList)
	case "OrderBy":
		return len(node.OrderBy)
	}
	return 0
}

// ---------------------------------------------------------------------------------------
// sort-has-keys
// ---------------------------------------------------------------------------------------

// SortHasKeys reports a SORT node with nothing to sort by.
//
// The bug behind it: a rewrite built its ordering keys from two lists, and a stream that
// belonged to neither produced a SORT with an empty OrderBy. It sorted by nothing --
// buffering every row, and spilling under memory pressure, to return them in the order they
// arrived -- while the same query through another path came back ranked by relevance. A
// keyless sort is never intentional: it is either missing keys or a node that should not
// have been built.
type SortHasKeys struct{}

func (SortHasKeys) Name() string    { return "sort-has-keys" }
func (SortHasKeys) Stages() []Stage { return []Stage{PreRemap, PostRemap} }

func (r SortHasKeys) Check(c *Ctx) []Violation {
	var out []Violation
	for _, id := range c.Nodes() {
		node := c.Node(id)
		if node == nil || node.NodeType != plan.Node_SORT {
			continue
		}
		if len(node.OrderBy) == 0 {
			out = append(out, Violation{
				Rule:   r.Name(),
				NodeID: id,
				Field:  "OrderBy",
				Detail: "SORT with no keys: it buffers every row to order them by nothing. " +
					"Either the keys were not attached or the node should not exist",
			})
		}
	}
	return out
}

// ---------------------------------------------------------------------------------------
// dag-reachable
// ---------------------------------------------------------------------------------------

// DagReachable reports structural damage to the node graph: a child id that does not exist,
// a nil node, or a cycle.
//
// It runs before the rules that walk children so their traversals are known to terminate,
// and it catches the failure mode of an append-only arena -- a rewrite that repoints a
// child at a node it also abandoned, leaving a dangling or self-referential edge.
type DagReachable struct{}

func (DagReachable) Name() string    { return "dag-reachable" }
func (DagReachable) Stages() []Stage { return []Stage{PreRemap, PostRemap} }

func (r DagReachable) Check(c *Ctx) []Violation {
	var out []Violation
	for _, step := range c.Query.Steps {
		if step < 0 || int(step) >= len(c.Query.Nodes) {
			out = append(out, Violation{
				Rule:   r.Name(),
				NodeID: step,
				Field:  "Steps",
				Detail: fmt.Sprintf("step root %d is out of range (arena holds %d nodes)",
					step, len(c.Query.Nodes)),
			})
		}
	}

	const (
		white = 0
		grey  = 1
		black = 2
	)
	color := make(map[int32]int8, len(c.Query.Nodes))
	var walk func(id int32, from int32)
	walk = func(id int32, from int32) {
		if id < 0 || int(id) >= len(c.Query.Nodes) {
			out = append(out, Violation{
				Rule:   r.Name(),
				NodeID: from,
				Field:  "Children",
				Detail: fmt.Sprintf("child id %d does not exist (arena holds %d nodes)",
					id, len(c.Query.Nodes)),
			})
			return
		}
		switch color[id] {
		case grey:
			out = append(out, Violation{
				Rule:   r.Name(),
				NodeID: id,
				Field:  "Children",
				Detail: fmt.Sprintf("cycle: node %d is reachable from itself (via node %d)", id, from),
			})
			return
		case black:
			return
		}
		color[id] = grey
		if node := c.Query.Nodes[id]; node == nil {
			out = append(out, Violation{
				Rule:   r.Name(),
				NodeID: id,
				Detail: "nil node reachable from Steps",
			})
		} else {
			for _, child := range node.Children {
				walk(child, id)
			}
		}
		color[id] = black
	}
	for _, step := range c.Query.Steps {
		if step >= 0 && int(step) < len(c.Query.Nodes) {
			walk(step, -1)
		}
	}
	return out
}

// ---------------------------------------------------------------------------------------
// colref-resolvable
// ---------------------------------------------------------------------------------------

// ColRefResolvable reports a column reference whose binding tag is produced nowhere at or
// below the node holding it.
//
// A rewrite that detaches a node, or publishes a replacement expression to the wrong
// consumer, leaves references pointing at a binding that is no longer in scope. Depending
// on the path that surfaces either as a remap error much later, with no trace of which
// rewrite caused it, or as a column read from the wrong place.
//
// PRE-REMAP ONLY. After remapAllColRefs a ColRef's RelPos is a child index rather than a
// binding tag, so the same test would be meaningless there.
type ColRefResolvable struct{}

func (ColRefResolvable) Name() string    { return "colref-resolvable" }
func (ColRefResolvable) Stages() []Stage { return []Stage{PreRemap} }

func (r ColRefResolvable) Check(c *Ctx) []Violation {
	var out []Violation
	for _, id := range c.Nodes() {
		node := c.Node(id)
		if node == nil {
			continue
		}
		// A node's own tags count: a scan's ProjectList refers to the scan's binding.
		visible := make(map[int32]bool, 8)
		for _, tag := range c.TagsBelow(id) {
			visible[tag] = true
		}
		for _, site := range ExprSites(node) {
			for _, col := range colRefsOf(site.Expr) {
				if visible[col.RelPos] {
					continue
				}
				// A correlated reference legitimately names a binding from an enclosing
				// query, which is not below this node. Those carry a non-zero depth.
				if col.RelPos < 0 {
					continue
				}
				out = append(out, Violation{
					Rule:   r.Name(),
					NodeID: id,
					Field:  site.Field,
					Detail: fmt.Sprintf("column (tag %d, pos %d) is not produced at or below "+
						"this node, so it cannot be resolved here", col.RelPos, col.ColPos),
				})
			}
		}
	}
	return out
}

// colRefsOf collects every ColRef in expr, descending through functions and lists. Subquery
// expressions are skipped: their references belong to the subquery's own scope, which this
// walk does not model.
func colRefsOf(expr *plan.Expr) []*plan.ColRef {
	var out []*plan.ColRef
	var walk func(e *plan.Expr)
	walk = func(e *plan.Expr) {
		if e == nil {
			return
		}
		switch t := e.Expr.(type) {
		case *plan.Expr_Col:
			if t.Col != nil {
				out = append(out, t.Col)
			}
		case *plan.Expr_F:
			if t.F != nil {
				for _, arg := range t.F.Args {
					walk(arg)
				}
			}
		case *plan.Expr_List:
			if t.List != nil {
				for _, item := range t.List.List {
					walk(item)
				}
			}
		case *plan.Expr_W:
			if t.W != nil {
				walk(t.W.WindowFunc)
				for _, p := range t.W.PartitionBy {
					walk(p)
				}
				for _, o := range t.W.OrderBy {
					if o != nil {
						walk(o.Expr)
					}
				}
			}
		}
	}
	walk(expr)
	return out
}
