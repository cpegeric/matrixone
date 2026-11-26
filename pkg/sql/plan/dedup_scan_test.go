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

	"github.com/matrixorigin/matrixone/pkg/catalog"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/fileservice"
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	pb "github.com/matrixorigin/matrixone/pkg/pb/plan"
	"github.com/stretchr/testify/require"
)

// The scan a DEDUP join checks keys against reads column ranges, for the primary key and
// unique indexes of INSERT, UPDATE and REPLACE.
func TestDedupTargetScanIsNarrow(t *testing.T) {
	for _, sql := range []string{
		"insert into constraint_test.emp values (1, 'a', 'b', 1, '2020-01-01', 1.0, 1.0, 1)",
		"insert into constraint_test.emp select * from constraint_test.emp",
		"update constraint_test.emp set empno = empno + 1",
		"update constraint_test.emp set ename = 'x'",
		"update ignore constraint_test.emp a join constraint_test.emp b on a.empno = b.empno set a.ename = 'x', b.job = 'y'",
		"replace into constraint_test.emp values (1, 'a', 'b', 1, '2020-01-01', 1.0, 1.0, 1)",
	} {
		t.Run(sql, func(t *testing.T) {
			p, err := runOneStmt(NewMockOptimizer(false, newPlanTestProcess(t)), t, sql)
			require.NoError(t, err)
			query := p.GetQuery()
			dedups := 0
			for _, n := range query.Nodes {
				if n.NodeType != plan.Node_JOIN || n.JoinType != plan.Node_DEDUP {
					continue
				}
				dedups++
				narrow := false
				for _, c := range n.Children {
					if query.Nodes[c].NodeType == plan.Node_TABLE_SCAN && readsColumnRanges(query.Nodes[c]) {
						narrow = true
					}
				}
				require.True(t, narrow, "DEDUP join on %s has a narrow target scan", n.DedupColName)
			}
			require.NotZero(t, dedups)
		})
	}
}

// A scan reading only some of a table's visible columns prefetches and reads just those
// columns; a scan of every visible column, hidden columns aside, reads whole objects.
func TestScanOfSomeColumnsReadsColumnRanges(t *testing.T) {
	for _, tc := range []struct {
		sql     string
		columns bool
	}{
		{"select id, payload from constraint_test.docs_ft", true},
		{"select id from constraint_test.docs_ft where payload > 1", true},
		{"select count(*) from constraint_test.docs_ft", true},
		{"select id, body, payload, embedding from constraint_test.docs_ft", false},
		{"select * from constraint_test.docs_ft", false},
		{"select * from constraint_test.emp", false},
	} {
		t.Run(tc.sql, func(t *testing.T) {
			p, err := runOneStmt(NewMockOptimizer(false, newPlanTestProcess(t)), t, tc.sql)
			require.NoError(t, err)
			scans := 0
			for _, n := range p.GetQuery().Nodes {
				if n.NodeType != plan.Node_TABLE_SCAN {
					continue
				}
				scans++
				got := fileservice.Policy(n.ReadPolicy)
				require.Equal(t, tc.columns, got.Any(fileservice.ColumnPrefetch), "policy %b", got)
				require.Equal(t, tc.columns, got.Any(fileservice.SkipFullFilePreloads), "policy %b", got)
			}
			require.NotZero(t, scans)
		})
	}
}

func TestOneVisibleColumnKeepsWholeObjectPolicy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		column string
	}{
		{name: "visible_column", column: "id"},
		{name: "only_hidden_row_id", column: catalog.Row_ID},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opt := newCheckAlterOptimizer(t, "create table one_visible_column (id bigint primary key)")
			opt.ctxt.isDml = true // keep the row ID in the mock catalog as the engine does
			table := opt.ctxt.tables["one_visible_column"]
			if _, ok := table.Name2ColIndex[catalog.Row_ID]; !ok {
				if table.Name2ColIndex == nil {
					table.Name2ColIndex = map[string]int32{"id": 0}
				}
				table.Name2ColIndex[catalog.Row_ID] = int32(len(table.Cols))
				table.Cols = append(table.Cols, &pb.ColDef{
					Name: catalog.Row_ID, Hidden: true, Typ: pb.Type{Id: int32(types.T_Rowid)},
				})
			}
			p, err := runOneStmt(opt, t, "select "+tc.column+" from one_visible_column")
			require.NoError(t, err)
			scans := 0
			for _, node := range p.GetQuery().Nodes {
				if node.NodeType != pb.Node_TABLE_SCAN {
					continue
				}
				scans++
				require.Len(t, node.TableDef.Cols, 1)
				require.Equal(t, tc.column, node.TableDef.Cols[0].Name)
				require.False(t, fileservice.Policy(node.ReadPolicy).Any(fileservice.ColumnPrefetch, fileservice.SkipFullFilePreloads),
					"a table with one visible column should keep whole-object prefetch")
			}
			require.Equal(t, 1, scans)
		})
	}
}
