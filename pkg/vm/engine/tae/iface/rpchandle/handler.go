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

package rpchandle

import (
	"context"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/cmd_util"

	apipb "github.com/matrixorigin/matrixone/pkg/pb/api"
	"github.com/matrixorigin/matrixone/pkg/pb/timestamp"
	"github.com/matrixorigin/matrixone/pkg/pb/txn"
)

type Handler interface {
	HandleCommit(
		ctx context.Context,
		meta txn.TxnMeta,
		response *txn.TxnResponse,
		commitRequests *txn.TxnCommitRequest,
	) (timestamp.Timestamp, error)

	HandleRollback(
		ctx context.Context,
		meta txn.TxnMeta,
	) error

	HandleCommitting(
		ctx context.Context,
		meta txn.TxnMeta,
	) error

	HandlePrepare(
		ctx context.Context,
		meta txn.TxnMeta,
	) (
		timestamp.Timestamp,
		error,
	)

	HandleStartRecovery(
		ctx context.Context,
		ch chan txn.TxnMeta,
	)

	HandleClose(ctx context.Context) error

	HandleDestroy(ctx context.Context) error

	HandleGetLogTail(
		ctx context.Context,
		meta txn.TxnMeta,
		req *apipb.SyncLogTailReq,
		resp *apipb.SyncLogTailResp,
	) (func(), error)

	HandlePreCommitWrite(
		ctx context.Context,
		meta txn.TxnMeta,
		req *apipb.PrecommitWriteCmd,
		resp *apipb.TNStringResponse,
	) error

	HandleFlushTable(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.FlushTable,
		resp *apipb.SyncLogTailResp,
	) (func(), error)

	HandleCommitMerge(
		ctx context.Context,
		meta txn.TxnMeta,
		req *apipb.MergeCommitEntry,
		resp *apipb.TNStringResponse,
	) error

	HandleForceCheckpoint(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.Checkpoint,
		resp *apipb.SyncLogTailResp,
	) (func(), error)

	HandleForceGlobalCheckpoint(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.Checkpoint,
		resp *apipb.SyncLogTailResp,
	) (func(), error)
	HandleInspectTN(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.InspectTN,
		resp *cmd_util.InspectResp,
	) (func(), error)

	HandleAddFaultPoint(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.FaultPoint,
		resp *apipb.SyncLogTailResp,
	) (func(), error)

	HandleBackup(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.Checkpoint,
		resp *apipb.SyncLogTailResp,
	) (func(), error)

	HandleTraceSpan(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.TraceSpan,
		resp *apipb.SyncLogTailResp,
	) (func(), error)

	HandleStorageUsage(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.StorageUsageReq,
		resp *cmd_util.StorageUsageResp_V3,
	) (func(), error)

	HandleSnapshotRead(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.SnapshotReadReq,
		resp *cmd_util.SnapshotReadResp,
	) (func(), error)

	HandleInterceptCommit(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.InterceptCommit,
		resp *apipb.SyncLogTailResp,
	) (func(), error)
	HandleDiskCleaner(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.DiskCleaner,
		resp *apipb.SyncLogTailResp,
	) (cb func(), err error)

	HandleGetLatestCheckpoint(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.Checkpoint,
		resp *apipb.CheckpointResp,
	) (cb func(), err error)

	HandleGetChangedTableList(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.GetChangedTableListReq,
		resp *cmd_util.GetChangedTableListResp,
	) (func(), error)

	HandleFaultInject(
		ctx context.Context,
		meta txn.TxnMeta,
		req *cmd_util.FaultInjectReq,
		resp *apipb.TNStringResponse,
	) (cb func(), err error)
}
