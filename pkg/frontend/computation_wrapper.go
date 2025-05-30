// Copyright 2021 Matrix Origin
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

package frontend

import (
	"bytes"
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/mohae/deepcopy"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/common/runtime"
	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/container/vector"
	"github.com/matrixorigin/matrixone/pkg/defines"
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	"github.com/matrixorigin/matrixone/pkg/pb/timestamp"
	"github.com/matrixorigin/matrixone/pkg/perfcounter"
	"github.com/matrixorigin/matrixone/pkg/sql/compile"
	"github.com/matrixorigin/matrixone/pkg/sql/models"
	"github.com/matrixorigin/matrixone/pkg/sql/parsers/tree"
	plan2 "github.com/matrixorigin/matrixone/pkg/sql/plan"
	"github.com/matrixorigin/matrixone/pkg/sql/util"
	"github.com/matrixorigin/matrixone/pkg/txn/storage/memorystorage"
	util2 "github.com/matrixorigin/matrixone/pkg/util"
	"github.com/matrixorigin/matrixone/pkg/util/trace"
	"github.com/matrixorigin/matrixone/pkg/util/trace/impl/motrace/statistic"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/disttae"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/disttae/cache"
	"github.com/matrixorigin/matrixone/pkg/vm/process"
)

var (
	_ ComputationWrapper = &TxnComputationWrapper{}
)

type TxnComputationWrapper struct {
	stmt      tree.Statement
	plan      *plan2.Plan
	proc      *process.Process
	ses       FeSession
	compile   *compile.Compile
	runResult *util2.RunResult

	ifIsExeccute bool
	uuid         uuid.UUID
	//holds values of params in the PREPARE
	paramVals []any

	explainBuffer *bytes.Buffer
	binaryPrepare bool
	prepareName   string
}

func InitTxnComputationWrapper(
	ses FeSession,
	stmt tree.Statement,
	proc *process.Process,
) *TxnComputationWrapper {
	uuid, _ := uuid.NewV7()
	return &TxnComputationWrapper{
		stmt: stmt,
		proc: proc,
		ses:  ses,
		uuid: uuid,
	}
}

func (cwft *TxnComputationWrapper) BinaryExecute() (bool, string) {
	return cwft.binaryPrepare, cwft.prepareName
}

func (cwft *TxnComputationWrapper) Plan() *plan.Plan {
	return cwft.plan
}

func (cwft *TxnComputationWrapper) ResetPlanAndStmt(stmt tree.Statement) {
	cwft.plan = nil
	cwft.freeStmt()
	cwft.stmt = stmt
}

func (cwft *TxnComputationWrapper) GetAst() tree.Statement {
	return cwft.stmt
}

func (cwft *TxnComputationWrapper) Free() {
	cwft.freeStmt()
	cwft.Clear()
}

func (cwft *TxnComputationWrapper) freeStmt() {
	if cwft.stmt != nil {
		if !cwft.ifIsExeccute {
			cwft.stmt.Free()
			cwft.stmt = nil
		}
	}
}

func (cwft *TxnComputationWrapper) Clear() {
	cwft.plan = nil
	cwft.proc = nil
	cwft.ses = nil
	cwft.compile = nil
	cwft.runResult = nil
	cwft.paramVals = nil
	cwft.prepareName = ""
	cwft.binaryPrepare = false
}

func (cwft *TxnComputationWrapper) ParamVals() []any {
	return cwft.paramVals
}

func (cwft *TxnComputationWrapper) GetProcess() *process.Process {
	return cwft.proc
}

func (cwft *TxnComputationWrapper) GetColumns(ctx context.Context) ([]interface{}, error) {
	var err error
	cols := plan2.GetResultColumnsFromPlan(cwft.plan)
	switch cwft.GetAst().(type) {
	case *tree.ShowColumns:
		if len(cols) == 7 {
			cols = []*plan2.ColDef{
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Field"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Type"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Null"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Key"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Default"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Extra"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Comment"},
			}
		} else {
			cols = []*plan2.ColDef{
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Field"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Type"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Collation"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Null"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Key"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Default"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Extra"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Privileges"},
				{Typ: plan2.Type{Id: int32(types.T_char)}, Name: "Comment"},
			}
		}
	}
	columns := make([]interface{}, len(cols))
	for i, col := range cols {
		c, err := colDef2MysqlColumn(ctx, col)
		if err != nil {
			return nil, err
		}
		columns[i] = c
	}
	return columns, err
}

func (cwft *TxnComputationWrapper) GetServerStatus() uint16 {
	return uint16(cwft.ses.GetTxnHandler().GetServerStatus())
}

func checkResultQueryPrivilege(proc *process.Process, p *plan.Plan, reqCtx context.Context, sid string, ses *Session) (statistic.StatsArray, error) {
	var ids []string
	var err error
	var stats statistic.StatsArray
	stats.Reset()

	if ids, err = isResultQuery(proc, p); err != nil || ids == nil {
		return stats, err
	}
	return checkPrivilege(sid, ids, reqCtx, ses)
}

// Compile build logical plan and then build physical plan `Compile` object
func (cwft *TxnComputationWrapper) Compile(any any, fill func(*batch.Batch, *perfcounter.CounterSet) error) (interface{}, error) {
	var originSQL string
	var span trace.Span
	var err error

	execCtx := any.(*ExecCtx)
	execCtx.reqCtx, span = trace.Start(execCtx.reqCtx, "TxnComputationWrapper.Compile",
		trace.WithKind(trace.SpanKindStatement))
	defer span.End(trace.WithStatementExtra(cwft.ses.GetTxnId(), cwft.ses.GetStmtId(), cwft.ses.GetSqlOfStmt()))

	defer RecordStatementTxnID(execCtx.reqCtx, cwft.ses)
	if cwft.ses.GetTxnHandler().HasTempEngine() {
		updateTempStorageInCtx(execCtx, cwft.proc, cwft.ses.GetTxnHandler().GetTempStorage())
	}
	stats := statistic.StatsInfoFromContext(execCtx.reqCtx)

	cacheHit := cwft.plan != nil
	if !cacheHit {
		cwft.plan, err = buildPlan(execCtx.reqCtx, cwft.ses, cwft.ses.GetTxnCompileCtx(), cwft.stmt)
		if err != nil {
			return nil, err
		}
	}

	if cwft.ses != nil && cwft.ses.GetTenantInfo() != nil && !cwft.ses.IsBackgroundSession() {
		var accId uint32
		accId, err = defines.GetAccountId(execCtx.reqCtx)
		if err != nil {
			return nil, err
		}
		cwft.ses.SetAccountId(accId)

		// the content of prepare sql don't need to authenticate when execute stmt
		if !execCtx.input.isBinaryProtExecute {
			authStats, err := authenticateCanExecuteStatementAndPlan(execCtx.reqCtx, cwft.ses.(*Session), cwft.stmt, cwft.plan)
			if err != nil {
				return nil, err
			}
			// record permission statistics.
			stats.PermissionAuth.Add(&authStats)
		}
	}

	if !cwft.ses.IsBackgroundSession() {
		cwft.ses.SetPlan(cwft.plan)
		authStats, err := checkResultQueryPrivilege(cwft.proc, cwft.plan, execCtx.reqCtx, cwft.ses.GetService(), cwft.ses.(*Session))
		if err != nil {
			return nil, err
		}
		stats.PermissionAuth.Add(&authStats)
	}

	if _, isTextProtExecute := cwft.stmt.(*tree.Execute); isTextProtExecute || execCtx.input.isBinaryProtExecute {
		var retComp *compile.Compile
		var plan *plan.Plan
		var stmt tree.Statement
		var sql string
		if isTextProtExecute {
			executePlan := cwft.plan.GetDcl().GetExecute()
			retComp, plan, stmt, sql, err = initExecuteStmtParam(execCtx, cwft.ses.(*Session), cwft, executePlan, executePlan.GetName())
			if err != nil {
				return nil, err
			}
			authStats, err := checkResultQueryPrivilege(cwft.proc, plan, execCtx.reqCtx, cwft.ses.GetService(), cwft.ses.(*Session))
			if err != nil {
				return nil, err
			}
			stats.PermissionAuth.Add(&authStats)

			cwft.plan = plan
			cwft.stmt.Free()
			// reset plan & stmt
			cwft.stmt = stmt
		} else {
			// binary protocol execute
			retComp, _, _, sql, err = initExecuteStmtParam(execCtx, cwft.ses.(*Session), cwft, nil, execCtx.input.stmtName)
			if err != nil {
				return nil, err
			}
		}
		originSQL = sql
		cwft.ifIsExeccute = true

		// reset some special stmt for execute statement
		switch cwft.stmt.(type) {
		case *tree.ShowTableStatus:
			cwft.ses.SetShowStmtType(ShowTableStatus)
			cwft.ses.SetData(nil)
		case *tree.SetVar, *tree.ShowVariables, *tree.ShowErrors, *tree.ShowWarnings,
			*tree.CreateAccount, *tree.AlterAccount, *tree.DropAccount:
			return nil, nil
		}

		if retComp == nil {
			cwft.compile, err = createCompile(execCtx, cwft.ses, cwft.proc, cwft.ses.GetSql(), cwft.stmt, cwft.plan, fill, false)
			if err != nil {
				return nil, err
			}
			cwft.compile.SetOriginSQL(originSQL)
		} else {
			// retComp
			cwft.proc.ReplaceTopCtx(execCtx.reqCtx)
			retComp.Reset(cwft.proc, getStatementStartAt(execCtx.reqCtx), fill, cwft.ses.GetSql())
			cwft.compile = retComp
		}

		//check privilege
		/* prepare not need check privilege
		   err = authenticateUserCanExecutePrepareOrExecute(requestCtx, cwft.ses, prepareStmt.PrepareStmt, newPlan)
		   if err != nil {
		   	return nil, err
		   }
		*/
	} else {
		cwft.compile, err = createCompile(execCtx, cwft.ses, cwft.proc, execCtx.sqlOfStmt, cwft.stmt, cwft.plan, fill, false)
		if err != nil {
			return nil, err
		}
	}

	return cwft.compile, err
}

func updateTempStorageInCtx(execCtx *ExecCtx, proc *process.Process, tempStorage *memorystorage.Storage) {
	if execCtx != nil && execCtx.reqCtx != nil {
		execCtx.reqCtx = attachValue(execCtx.reqCtx, defines.TemporaryTN{}, tempStorage)
		proc.ReplaceTopCtx(execCtx.reqCtx)
	}
}

func (cwft *TxnComputationWrapper) RecordExecPlan(ctx context.Context, phyPlan *models.PhyPlan) error {
	if stm := cwft.ses.GetStmtInfo(); stm != nil {
		waitActiveCost := time.Duration(0)
		if handler := cwft.ses.GetTxnHandler(); handler.InActiveTxn() {
			txn := handler.GetTxn()
			if txn != nil {
				waitActiveCost = txn.GetWaitActiveCost()
			}
		}
		stm.SetSerializableExecPlan(NewJsonPlanHandler(ctx, stm, cwft.ses, cwft.plan, phyPlan, WithWaitActiveCost(waitActiveCost)))
	}
	return nil
}

// RecordCompoundStmt Check if it is a compound statement, What is a compound statement?
func (cwft *TxnComputationWrapper) RecordCompoundStmt(ctx context.Context, statsBytes statistic.StatsArray) error {
	if stm := cwft.ses.GetStmtInfo(); stm != nil {
		// Check if it is a compound statement, What is a compound statement?
		jsonHandle := &jsonPlanHandler{
			jsonBytes:  sqlQueryIgnoreExecPlan,
			statsBytes: statsBytes,
		}
		stm.SetSerializableExecPlan(jsonHandle)
	}
	return nil
}

func (cwft *TxnComputationWrapper) StatsCompositeSubStmtResource(ctx context.Context) (statsByte statistic.StatsArray) {
	waitActiveCost := time.Duration(0)
	if handler := cwft.ses.GetTxnHandler(); handler.InActiveTxn() {
		txn := handler.GetTxn()
		if txn != nil {
			waitActiveCost = txn.GetWaitActiveCost()
		}
	}

	h := NewMarshalPlanHandlerCompositeSubStmt(ctx, cwft.plan, WithWaitActiveCost(waitActiveCost))
	statsByte, _ = h.Stats(ctx, cwft.ses)
	return statsByte
}

func (cwft *TxnComputationWrapper) SetExplainBuffer(buf *bytes.Buffer) {
	cwft.explainBuffer = buf
}

func (cwft *TxnComputationWrapper) GetUUID() []byte {
	return cwft.uuid[:]
}

func (cwft *TxnComputationWrapper) Run(ts uint64) (*util2.RunResult, error) {
	runResult, err := cwft.compile.Run(ts)
	cwft.compile.Release()
	cwft.runResult = runResult
	cwft.compile = nil
	return runResult, err
}

func (cwft *TxnComputationWrapper) GetLoadTag() bool {
	return cwft.plan.GetQuery().GetLoadTag()
}

func appendStatementAt(ctx context.Context, value time.Time) context.Context {
	return context.WithValue(ctx, defines.StartTS{}, value)
}

func getStatementStartAt(ctx context.Context) time.Time {
	v := ctx.Value(defines.StartTS{})
	if v == nil {
		return time.Now()
	}
	return v.(time.Time)
}

func CheckTableDefChange(catalogCache *cache.CatalogCache, tblKey *cache.TableChangeQuery) bool {
	return catalogCache.HasNewerVersion(tblKey)
}

// initExecuteStmtParam replaces the plan of the EXECUTE by the plan generated by
// the PREPARE and setups the params for the plan.
func initExecuteStmtParam(execCtx *ExecCtx, ses *Session, cwft *TxnComputationWrapper, execPlan *plan.Execute, stmtName string) (*compile.Compile, *plan.Plan, tree.Statement, string, error) {
	reqCtx := execCtx.reqCtx
	if execPlan != nil { // binary protocol, don't have to buildplan, execPlan is nil
		stmtName = execPlan.GetName()
	}
	prepareStmt, err := ses.GetPrepareStmt(reqCtx, stmtName)
	if err != nil {
		return nil, nil, nil, "", err
	}
	originSQL := prepareStmt.Sql
	preparePlan := prepareStmt.PreparePlan.GetDcl().GetPrepare()

	// TODO check if schema change, obj.Obj is zero all the time in 0.6
	eng := ses.proc.Base.SessionInfo.StorageEngine
	catalogCache := eng.(*disttae.Engine).GetLatestCatalogCache()

	var change bool
	for _, obj := range preparePlan.GetSchemas() {
		accountId := ses.GetAccountId()
		if ShouldSwitchToSysAccount(obj.SchemaName, obj.ObjName) {
			accountId = uint32(sysAccountID)
		}
		tblKey := &cache.TableChangeQuery{
			AccountId:  accountId,
			DatabaseId: uint64(obj.Db),
			Name:       obj.ObjName,
			Version:    uint32(obj.Server),
			TableId:    uint64(obj.Obj),
			Ts:         prepareStmt.Ts,
		}

		change = CheckTableDefChange(catalogCache, tblKey)
		if change {
			break
		}
	}

	// rebuild and recompile
	if change {
		originPrepareStmt := &tree.PrepareStmt{
			Name: tree.Identifier(prepareStmt.Name),
			Stmt: prepareStmt.PrepareStmt,
		}
		newPlan, err := buildPlan(reqCtx, ses, ses.GetTxnCompileCtx(), originPrepareStmt)
		if err != nil {
			return nil, nil, nil, "", err
		}

		if prepareStmt.compile != nil {
			prepareStmt.compile.FreeOperator()
			prepareStmt.compile.SetIsPrepare(false)
			prepareStmt.compile.Release()
			prepareStmt.compile = nil // set nil and recompile later
		}

		preparePlan = newPlan.GetDcl().GetPrepare()
		if _, ok := preparePlan.Plan.Plan.(*plan.Plan_Query); ok {
			//only DQL & DML will pre compile
			comp, err := createCompile(execCtx, ses, ses.proc, originSQL, prepareStmt.PrepareStmt, preparePlan.Plan, ses.GetOutputCallback(execCtx), true)
			if err != nil {
				if !moerr.IsMoErrCode(err, moerr.ErrCantCompileForPrepare) {
					return nil, nil, nil, "", err
				}
			}
			// do not save ap query now()
			if comp != nil && !comp.IsTpQuery() {
				comp.SetIsPrepare(false)
				comp.Release()
				comp = nil
			}
			prepareStmt.compile = comp

		}
		prepareStmt.PreparePlan = newPlan

		prepareStmt.Ts = timestamp.Timestamp{PhysicalTime: time.Now().Unix()}
	}

	numParams := len(preparePlan.ParamTypes)
	if prepareStmt.params != nil && prepareStmt.params.Length() > 0 { // use binary protocol
		if prepareStmt.params.Length() != numParams {
			return nil, nil, nil, originSQL, moerr.NewInvalidInput(reqCtx, "Incorrect arguments to EXECUTE")
		}
		cwft.proc.SetPrepareParams(prepareStmt.params)
	} else if execPlan != nil && len(execPlan.Args) > 0 {
		if len(execPlan.Args) != numParams {
			return nil, nil, nil, originSQL, moerr.NewInvalidInput(reqCtx, "Incorrect arguments to EXECUTE")
		}
		params := vector.NewVec(types.T_text.ToType())
		paramVals := make([]any, numParams)
		for i, arg := range execPlan.Args {
			exprImpl := arg.Expr.(*plan.Expr_V)
			param, err := cwft.proc.GetResolveVariableFunc()(exprImpl.V.Name, exprImpl.V.System, exprImpl.V.Global)
			if err != nil {
				return nil, nil, nil, originSQL, err
			}
			err = util.AppendAnyToStringVector(cwft.proc, param, params)
			if err != nil {
				return nil, nil, nil, originSQL, err
			}
			paramVals[i] = param
		}
		cwft.proc.SetPrepareParams(params)
		cwft.paramVals = paramVals
	} else {
		if numParams > 0 {
			return nil, nil, nil, originSQL, moerr.NewInvalidInput(reqCtx, "Incorrect arguments to EXECUTE")
		}
	}
	return prepareStmt.compile, preparePlan.Plan, prepareStmt.PrepareStmt, originSQL, nil
}

func createCompile(
	execCtx *ExecCtx,
	ses FeSession,
	proc *process.Process,
	originSQL string,
	stmt tree.Statement,
	plan *plan2.Plan,
	fill func(*batch.Batch, *perfcounter.CounterSet) error,
	isPrepare bool,
) (retCompile *compile.Compile, err error) {

	addr := ""
	pu := getPu(ses.GetService())
	if len(pu.ClusterNodes) > 0 {
		addr = pu.ClusterNodes[0].Addr
	}
	proc.ReplaceTopCtx(execCtx.reqCtx)
	proc.Base.FileService = pu.FileService

	var tenant string
	tInfo := ses.GetTenantInfo()
	if tInfo != nil {
		tenant = tInfo.GetTenant()
	}

	stats := statistic.StatsInfoFromContext(execCtx.reqCtx)
	stats.CompileStart()
	crs := new(perfcounter.CounterSet)
	execCtx.reqCtx = perfcounter.AttachCompilePlanMarkKey(execCtx.reqCtx, crs)
	defer func() {
		stats.AddCompileS3Request(statistic.S3Request{
			List:      crs.FileService.S3.List.Load(),
			Head:      crs.FileService.S3.Head.Load(),
			Put:       crs.FileService.S3.Put.Load(),
			Get:       crs.FileService.S3.Get.Load(),
			Delete:    crs.FileService.S3.Delete.Load(),
			DeleteMul: crs.FileService.S3.DeleteMulti.Load(),
		})
		stats.CompileEnd()
	}()

	defer func() {
		if err != nil && retCompile != nil {
			retCompile.SetIsPrepare(false)
			retCompile.Release()
			retCompile = nil
		}
	}()
	retCompile = compile.NewCompile(
		addr,
		ses.GetDatabaseName(),
		ses.GetSql(),
		tenant,
		ses.GetUserName(),
		ses.GetTxnHandler().GetStorage(),
		proc,
		stmt,
		ses.GetIsInternal(),
		deepcopy.Copy(ses.getCNLabels()).(map[string]string),
		getStatementStartAt(execCtx.reqCtx),
	)
	retCompile.SetIsPrepare(isPrepare)
	retCompile.SetBuildPlanFunc(func(ctx context.Context) (*plan2.Plan, error) {
		// No permission verification is required when retry execute buildPlan
		plan, err := buildPlan(ctx, ses, ses.GetTxnCompileCtx(), stmt)
		if err != nil {
			return nil, err
		}
		if plan.IsPrepare {
			_, _, err = plan2.ResetPreparePlan(ses.GetTxnCompileCtx(), plan)
		}
		return plan, err
	})

	if _, ok := stmt.(*tree.ExplainAnalyze); ok {
		fill = func(bat *batch.Batch, crs *perfcounter.CounterSet) error { return nil }
	}

	if _, ok := stmt.(*tree.ExplainPhyPlan); ok {
		fill = func(bat *batch.Batch, crs *perfcounter.CounterSet) error { return nil }
	}

	err = retCompile.Compile(execCtx.reqCtx, plan, fill)
	if err != nil {
		return
	}
	// check if it is necessary to initialize the temporary engine
	if !ses.GetTxnHandler().HasTempEngine() && retCompile.NeedInitTempEngine() {
		// 0. init memory-non-dist storage
		err = ses.GetTxnHandler().CreateTempStorage(runtime.ServiceRuntime(ses.GetService()).Clock())
		if err != nil {
			return
		}

		// temporary storage is passed through Ctx
		updateTempStorageInCtx(execCtx, proc, ses.GetTxnHandler().GetTempStorage())

		// 1. init memory-non-dist engine
		ses.GetTxnHandler().CreateTempEngine()
		tempEngine := ses.GetTxnHandler().GetTempEngine()

		// 2. bind the temporary engine to the session and txnHandler
		retCompile.SetTempEngine(tempEngine, ses.GetTxnHandler().GetTempStorage())

		// 3. init temp-db to store temporary relations
		txnOp2 := ses.GetTxnHandler().GetTxn()
		err = tempEngine.Create(execCtx.reqCtx, defines.TEMPORARY_DBNAME, txnOp2)
		if err != nil {
			return
		}
	}
	retCompile.SetOriginSQL(originSQL)
	return
}
