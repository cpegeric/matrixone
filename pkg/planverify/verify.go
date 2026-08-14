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

// Package planverify checks structural invariants of a query plan immediately after the
// rewrites that build it.
//
// It exists because plan bugs are silent. A malformed plan does not crash and does not
// return an error -- it returns WRONG ROWS, and only a test whose expected output happens
// to disagree will notice. Three examples from one month of index work, each of which this
// package turns into a loud failure at the exact rewrite that caused it:
//
//   - A predicate was attached to Node_JOIN.FilterList. Nothing in the planner sets that
//     field and nothing in compile reads it, but EXPLAIN prints it, so the plan LOOKED
//     right while the predicate was never applied. Rows the predicate excluded came back.
//   - A SORT node was built with no keys at all: it buffered rows to sort them by nothing
//     and returned them in join order, while the identical query on another path came back
//     ranked.
//   - A rewrite left a column reference whose binding tag no longer existed below the node
//     holding it.
//
// Scope, stated plainly: these are STRUCTURAL invariants. A plan can satisfy every rule
// here and still be semantically wrong -- reporting one search term's relevance for a
// different term produced a perfectly well-formed plan. Discriminating tests remain the
// only oracle for that. What this package buys is that the structural class stops reaching
// review, so tests can be about semantics.
package planverify

import (
	"fmt"
	"sort"
	"strings"

	"github.com/matrixorigin/matrixone/pkg/pb/plan"
)

// Stage is when a check runs. The two differ because remapAllColRefs rewrites every column
// reference: before it, a ColRef names a binding tag and can be traced to the node that
// produced it; after it, positions are relative to a child's output and the earlier rule
// would report false violations.
type Stage int

const (
	// PreRemap runs immediately after applyIndices, while binding tags still identify the
	// rewrite that produced each column. Violations point at the culprit rewrite.
	PreRemap Stage = iota
	// PostRemap runs after remapAllColRefs, catching damage done by remapping itself.
	PostRemap
)

func (s Stage) String() string {
	if s == PostRemap {
		return "post-remap"
	}
	return "pre-remap"
}

// Violation is one broken invariant, named so the message says which rule fired, where, and
// what to do about it rather than leaving the reader to infer it from a node dump.
type Violation struct {
	Rule   string
	NodeID int32
	Field  string
	Detail string
}

func (v Violation) String() string {
	loc := fmt.Sprintf("node %d", v.NodeID)
	if v.Field != "" {
		loc += "." + v.Field
	}
	return fmt.Sprintf("[%s] %s: %s", v.Rule, loc, v.Detail)
}

// Rule is one invariant. Keeping them separate objects (rather than one big walk) means a
// rule can be added, tested and reasoned about on its own, and the report says which one
// fired.
type Rule interface {
	Name() string
	// Stages reports the stages this rule is meaningful at.
	Stages() []Stage
	Check(c *Ctx) []Violation
}

// Ctx is the plan under check plus the helpers every rule needs. Rules must not mutate it.
type Ctx struct {
	Query *plan.Query
	Stage Stage

	reachable  []int32
	tagsBelow  map[int32][]int32
	visitGuard map[int32]bool
}

// Nodes returns the node ids reachable from Query.Steps, in deterministic order.
//
// Reachability matters: plan.Nodes is an APPEND-ONLY ARENA. A successful rewrite leaves its
// pre-rewrite nodes in it, unreferenced, and those abandoned nodes are frequently malformed
// by design (half-rewritten, dangling children). Checking them would report violations for
// plan fragments that no longer exist -- a mistake already made once, in a view-definition
// walk that scanned the arena and refused valid views because of an orphan it found there.
func (c *Ctx) Nodes() []int32 {
	if c.reachable != nil {
		return c.reachable
	}
	seen := make(map[int32]bool, len(c.Query.Nodes))
	var order []int32
	var walk func(id int32)
	walk = func(id int32) {
		if id < 0 || int(id) >= len(c.Query.Nodes) || seen[id] {
			return
		}
		seen[id] = true
		order = append(order, id)
		node := c.Query.Nodes[id]
		if node == nil {
			return
		}
		for _, child := range node.Children {
			walk(child)
		}
	}
	for _, step := range c.Query.Steps {
		walk(step)
	}
	c.reachable = order
	return order
}

// Node returns the node by id, or nil when the id is out of range.
func (c *Ctx) Node(id int32) *plan.Node {
	if id < 0 || int(id) >= len(c.Query.Nodes) {
		return nil
	}
	return c.Query.Nodes[id]
}

// TagsBelow returns every binding tag produced at or below nodeID. A ColRef whose tag is
// absent from this set cannot be resolved by that node at execution.
func (c *Ctx) TagsBelow(nodeID int32) []int32 {
	if c.tagsBelow == nil {
		c.tagsBelow = make(map[int32][]int32, len(c.Query.Nodes))
	}
	if tags, ok := c.tagsBelow[nodeID]; ok {
		return tags
	}
	if c.visitGuard == nil {
		c.visitGuard = make(map[int32]bool)
	}
	if c.visitGuard[nodeID] {
		return nil // cycle: dag-reachable reports it; do not spin here
	}
	c.visitGuard[nodeID] = true
	defer delete(c.visitGuard, nodeID)

	node := c.Node(nodeID)
	if node == nil {
		return nil
	}
	set := make(map[int32]bool, 4)
	for _, tag := range node.BindingTags {
		set[tag] = true
	}
	for _, child := range node.Children {
		for _, tag := range c.TagsBelow(child) {
			set[tag] = true
		}
	}
	tags := make([]int32, 0, len(set))
	for tag := range set {
		tags = append(tags, tag)
	}
	sort.Slice(tags, func(i, j int) bool { return tags[i] < tags[j] })
	c.tagsBelow[nodeID] = tags
	return tags
}

// ExprSite is one expression-bearing field of a node, carrying the FIELD NAME so a
// violation can say `Node_JOIN.FilterList` rather than "some expression".
type ExprSite struct {
	Field string
	Expr  *plan.Expr
}

// ExprSites returns every expression a node carries, with its field name.
//
// The list is deliberately exhaustive rather than "the fields I expected to matter": the
// bug this package was written for lived in a field nobody thought to look at.
func ExprSites(node *plan.Node) []ExprSite {
	if node == nil {
		return nil
	}
	var sites []ExprSite
	add := func(field string, exprs ...*plan.Expr) {
		for _, e := range exprs {
			if e != nil {
				sites = append(sites, ExprSite{Field: field, Expr: e})
			}
		}
	}
	add("ProjectList", node.ProjectList...)
	add("FilterList", node.FilterList...)
	add("OnList", node.OnList...)
	add("GroupBy", node.GroupBy...)
	add("AggList", node.AggList...)
	add("WinSpecList", node.WinSpecList...)
	add("BlockFilterList", node.BlockFilterList...)
	add("TblFuncExprList", node.TblFuncExprList...)
	add("FillVal", node.FillVal...)
	add("OnUpdateExprs", node.OnUpdateExprs...)
	add("Limit", node.Limit)
	add("Offset", node.Offset)
	for i := range node.OrderBy {
		if node.OrderBy[i] != nil {
			add("OrderBy", node.OrderBy[i].Expr)
		}
	}
	return sites
}

// Verify runs the rules that apply at this stage and returns every violation as one error.
//
// All rules run even after the first failure: a single rewrite usually breaks an invariant
// in several places at once, and reporting them together turns one debugging round into
// one fix instead of several.
func Verify(query *plan.Query, stage Stage, rules ...Rule) error {
	if query == nil {
		return nil
	}
	if len(rules) == 0 {
		rules = DefaultRules()
	}
	ctx := &Ctx{Query: query, Stage: stage}

	var found []Violation
	for _, rule := range rules {
		if !appliesAt(rule, stage) {
			continue
		}
		found = append(found, rule.Check(ctx)...)
	}
	if len(found) == 0 {
		return nil
	}

	var b strings.Builder
	fmt.Fprintf(&b, "plan verification failed (%s): %d violation(s)", stage, len(found))
	for _, v := range found {
		b.WriteString("\n  ")
		b.WriteString(v.String())
	}
	return fmt.Errorf("%s", b.String())
}

func appliesAt(rule Rule, stage Stage) bool {
	for _, s := range rule.Stages() {
		if s == stage {
			return true
		}
	}
	return false
}

// DefaultRules is the set enabled by the hooks. Every rule here either has a shipped bug
// behind it or is a precondition another rule depends on.
func DefaultRules() []Rule {
	return []Rule{
		FieldHonored{},
		SortHasKeys{},
		DagReachable{},
		ColRefResolvable{},
	}
}
