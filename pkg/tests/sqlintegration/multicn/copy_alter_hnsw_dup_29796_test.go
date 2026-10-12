// Copyright 2026 Matrix Origin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package multicn

import (
	"context"
	"database/sql"
	"fmt"
	"testing"
	"time"

	_ "github.com/go-sql-driver/mysql"
	"github.com/matrixorigin/matrixone/pkg/clusterservice"
	"github.com/matrixorigin/matrixone/pkg/embed"
	"github.com/matrixorigin/matrixone/pkg/pb/metadata"
	"github.com/matrixorigin/matrixone/pkg/sql/plan"
	"github.com/stretchr/testify/require"
)

// #29796: a COPY ALTER that adds a column and an HNSW index must not fail with a
// native usearch "Duplicate keys" error on distinct primary keys. The build's
// source scan (SELECT ... FROM <renamed replacement> CROSS APPLY hnsw_create)
// must deliver each row once to the single coordinator builder. When the scan is
// forced onto multiple CNs, a remote reader must not serve a stale/duplicate copy
// of the in-transaction renamed source.
func TestIssue29796CopyAlterHnswNoDuplicate(t *testing.T) {
	cluster, err := embed.StartTestCluster(embed.WithCNCount(2))
	if cluster != nil {
		t.Cleanup(func() { require.NoError(t, cluster.Close()) })
	}
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	cn, err := cluster.GetCNService(0)
	require.NoError(t, err)
	inventory := clusterservice.GetMOCluster(cn.ServiceID())
	refresher, ok := inventory.(clusterservice.AuthoritativeRefresher)
	require.True(t, ok)
	require.Eventually(t, func() bool {
		if refresher.Refresh(ctx) != nil {
			return false
		}
		count := 0
		inventory.GetCNService(clusterservice.NewSelector(), func(metadata.CNService) bool {
			count++
			return true
		})
		return count == 2
	}, 30*time.Second, 100*time.Millisecond)

	oldForce := plan.GetForceScanOnMultiCN()
	plan.SetForceScanOnMultiCN(true)
	t.Cleanup(func() { plan.SetForceScanOnMultiCN(oldForce) })

	db, err := sql.Open("mysql", fmt.Sprintf("dump:111@tcp(127.0.0.1:%d)/", cn.GetServiceConfig().CN.Frontend.Port))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })
	conn, err := db.Conn(ctx)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, conn.Close()) })

	exec := func(statement string) {
		t.Helper()
		_, err := conn.ExecContext(ctx, statement)
		require.NoError(t, err, statement)
	}

	const schema = "issue_29796"
	for _, statement := range []string{
		"create database " + schema,
		"use " + schema,
		"set experimental_hnsw_index=1",
		"create table vectors(id bigint primary key, embedding vecf32(3))",
		"insert into vectors values(1,'[1,0,0]'),(2,'[0,1,0]'),(3,'[0,0,1]')",
	} {
		exec(statement)
	}
	exec("select mo_ctl('dn','flush','" + schema + ".vectors')")

	// The COPY ALTER: add a column AND an HNSW index in one statement. The
	// internal build must preserve the three distinct keys, not report a
	// native duplicate.
	exec("alter table vectors add column note int, add index h_embedding using hnsw(embedding) op_type 'vector_l2_ops'")

	rows, err := conn.QueryContext(ctx, "select id from vectors order by l2_distance(embedding,'[1,0,0]') limit 1")
	require.NoError(t, err)
	defer rows.Close()
	var id int64
	require.True(t, rows.Next())
	require.NoError(t, rows.Scan(&id))
	require.NoError(t, rows.Err())
	require.Equal(t, int64(1), id, "nearest to [1,0,0] must be id 1")
}
