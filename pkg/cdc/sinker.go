// Copyright 2024 Matrix Origin
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

package cdc

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/matrixorigin/matrixone/pkg/catalog"
	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/common/mpool"
	"github.com/matrixorigin/matrixone/pkg/container/batch"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/logutil"
	"github.com/matrixorigin/matrixone/pkg/objectio"
	"github.com/matrixorigin/matrixone/pkg/pb/plan"
	v2 "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
	"github.com/matrixorigin/mysql"
)

const (
	// sqlBufReserved leave 5 bytes for mysql driver
	sqlBufReserved         = 5
	sqlPrintLen            = 200
	fakeSql                = "fakeSql"
	createTable            = "create table"
	createTableIfNotExists = "create table if not exists"
)

var (
	begin    = []byte("begin")
	commit   = []byte("commit")
	rollback = []byte("rollback")
	dummy    = []byte("")
)

var NewSinker = func(
	sinkUri UriInfo,
	accountId uint64,
	taskId string,
	dbTblInfo *DbTableInfo,
	watermarkUpdater *CDCWatermarkUpdater,
	tableDef *plan.TableDef,
	retryTimes int,
	retryDuration time.Duration,
	ar *ActiveRoutine,
	maxSqlLength uint64,
	sendSqlTimeout string,
) (Sinker, error) {
	//TODO: remove console
	if sinkUri.SinkTyp == CDCSinkType_Console {
		return NewConsoleSinker(dbTblInfo, watermarkUpdater), nil
	}

	var (
		err  error
		sink Sink

		doRecord bool
	)

	if tableDef != nil {
		doRecord, _ = objectio.CDCRecordTxnInjected(tableDef.DbName, tableDef.Name)
	}

	if sink, err = NewMysqlSink(
		sinkUri.User, sinkUri.Password, sinkUri.Ip, sinkUri.Port,
		retryTimes, retryDuration, sendSqlTimeout, doRecord); err != nil {
		return nil, err
	}

	ctx := context.Background()
	padding := strings.Repeat(" ", sqlBufReserved)
	// create db
	err = sink.Send(ctx, ar, []byte(padding+fmt.Sprintf("CREATE DATABASE IF NOT EXISTS `%s`", dbTblInfo.SinkDbName)), false)
	if err != nil {
		return nil, err
	}
	// use db
	err = sink.Send(ctx, ar, []byte(padding+fmt.Sprintf("use `%s`", dbTblInfo.SinkDbName)), false)
	if err != nil {
		return nil, err
	}
	// possibly need to drop table first
	if dbTblInfo.IdChanged {
		err = sink.Send(ctx, ar, []byte(padding+fmt.Sprintf("DROP TABLE IF EXISTS `%s`", dbTblInfo.SinkTblName)), false)
		if err != nil {
			return nil, err
		}
		dbTblInfo.IdChanged = false
	}
	// create table
	createSql := strings.TrimSpace(dbTblInfo.SourceCreateSql)
	if len(createSql) < len(createTableIfNotExists) || !strings.EqualFold(createSql[:len(createTableIfNotExists)], createTableIfNotExists) {
		createSql = createTableIfNotExists + createSql[len(createTable):]
	}
	tableStart := len(createTableIfNotExists)
	tableEnd := strings.Index(createSql, "(")
	newTablePart := ""
	if dbTblInfo.SinkDbName != "" {
		newTablePart = dbTblInfo.SinkDbName + "." + dbTblInfo.SinkTblName
	} else {
		newTablePart = dbTblInfo.SinkTblName
	}
	createSql = createSql[:tableStart] + " " + newTablePart + createSql[tableEnd:]
	err = sink.Send(ctx, ar, []byte(padding+createSql), false)
	if err != nil {
		return nil, err
	}

	return NewMysqlSinker(
		sink,
		accountId,
		taskId,
		dbTblInfo,
		watermarkUpdater,
		tableDef,
		ar,
		maxSqlLength,
		sinkUri.SinkTyp == CDCSinkType_MO,
	), nil
}

var _ Sinker = new(consoleSinker)

type consoleSinker struct {
	dbTblInfo        *DbTableInfo
	watermarkUpdater *CDCWatermarkUpdater
}

func NewConsoleSinker(
	dbTblInfo *DbTableInfo,
	watermarkUpdater *CDCWatermarkUpdater,
) Sinker {
	return &consoleSinker{
		dbTblInfo:        dbTblInfo,
		watermarkUpdater: watermarkUpdater,
	}
}

func (s *consoleSinker) Run(_ context.Context, _ *ActiveRoutine) {}

func (s *consoleSinker) Sink(ctx context.Context, data *DecoderOutput) {
	logutil.Info("====console sinker====")

	logutil.Infof("output type %s", data.outputTyp)
	switch data.outputTyp {
	case OutputTypeSnapshot:
		if data.checkpointBat != nil && data.checkpointBat.RowCount() > 0 {
			//FIXME: only test here
			logutil.Info("checkpoint")
			//logutil.Info(data.checkpointBat.String())
		}
	case OutputTypeTail:
		if data.insertAtmBatch != nil && data.insertAtmBatch.Rows.Len() > 0 {
			//FIXME: only test here
			wantedColCnt := len(data.insertAtmBatch.Batches[0].Vecs) - 2
			row := make([]any, wantedColCnt)
			wantedColIndice := make([]int, wantedColCnt)
			for i := 0; i < wantedColCnt; i++ {
				wantedColIndice[i] = i
			}

			iter := data.insertAtmBatch.GetRowIterator()
			for iter.Next() {
				_ = iter.Row(ctx, row)
				logutil.Infof("insert %v", row)
			}
			iter.Close()
		}
	}
}

func (s *consoleSinker) SendBegin() {}

func (s *consoleSinker) SendCommit() {}

func (s *consoleSinker) SendRollback() {}

func (s *consoleSinker) SendDummy() {}

func (s *consoleSinker) Error() error {
	return nil
}

func (s *consoleSinker) ClearError() {}

func (s *consoleSinker) Reset() {}

func (s *consoleSinker) Close() {}

var _ Sinker = new(mysqlSinker)

type mysqlSinker struct {
	mysql Sink
	// account id of the cdc task
	accountId uint64
	// task id of the cdc task
	taskId           string
	dbTblInfo        *DbTableInfo
	watermarkUpdater *CDCWatermarkUpdater
	ar               *ActiveRoutine

	// buf of sql statement
	sqlBufs      [2][]byte
	curBufIdx    int
	sqlBuf       []byte
	sqlBufSendCh chan []byte

	// prefix of sql statement, e.g. `insert into xx values ...`
	insertPrefix []byte
	deletePrefix []byte
	// prefix of sql statement with ts, e.g. `/* [fromTs, toTs) */ insert into xx values ...`
	tsInsertPrefix []byte
	tsDeletePrefix []byte
	// suffix of sql statement, e.g. `;` or `);`
	insertSuffix []byte
	deleteSuffix []byte

	// buf of row data from batch, e.g. values part of insert statement `insert into xx values (a),(b),(c)`
	// or `where ... in ... ` part of delete statement `delete from xx where pk in ((a),(b),(c))`
	rowBuf []byte
	// prefix of row buffer, e.g. `(`
	insertRowPrefix []byte
	deleteRowPrefix []byte
	// separator of col buffer, e.g. `,` or `and`
	insertColSeparator []byte
	deleteColSeparator []byte
	// suffix of row buffer, e.g. `)`
	insertRowSuffix []byte
	deleteRowSuffix []byte
	// separator of row buffer, e.g. `,` or `or`
	insertRowSeparator []byte
	deleteRowSeparator []byte

	// only contains user defined column types, no mo meta cols
	insertTypes []*types.Type
	// only contains pk columns
	deleteTypes []*types.Type
	// used for delete multi-col pk
	pkColNames []string

	// for collect row data, allocate only once
	insertRow []any
	deleteRow []any

	// insert or delete of last record, used for combine inserts and deletes
	preRowType RowType
	// the length of all completed sql statement in sqlBuf
	preSqlBufLen int

	err  atomic.Value
	isMO bool
}

var NewMysqlSinker = func(
	mysql Sink,
	accountId uint64,
	taskId string,
	dbTblInfo *DbTableInfo,
	watermarkUpdater *CDCWatermarkUpdater,
	tableDef *plan.TableDef,
	ar *ActiveRoutine,
	maxSqlLength uint64,
	isMO bool,
) Sinker {
	s := &mysqlSinker{
		mysql:            mysql,
		accountId:        accountId,
		taskId:           taskId,
		dbTblInfo:        dbTblInfo,
		watermarkUpdater: watermarkUpdater,
		ar:               ar,
	}
	var maxAllowedPacket uint64
	_ = mysql.(*mysqlSink).conn.QueryRow("SELECT @@max_allowed_packet").Scan(&maxAllowedPacket)
	maxAllowedPacket = min(maxAllowedPacket, maxSqlLength)
	logutil.Infof("cdc mysqlSinker(%v) maxAllowedPacket = %d", s.dbTblInfo, maxAllowedPacket)

	// sqlBuf
	s.sqlBufs[0] = make([]byte, sqlBufReserved, maxAllowedPacket)
	s.sqlBufs[1] = make([]byte, sqlBufReserved, maxAllowedPacket)
	s.curBufIdx = 0
	s.sqlBuf = s.sqlBufs[s.curBufIdx]
	s.sqlBufSendCh = make(chan []byte)

	s.rowBuf = make([]byte, 0, 1024)

	// prefix and suffix
	s.insertPrefix = []byte(fmt.Sprintf("REPLACE INTO `%s`.`%s` VALUES ", s.dbTblInfo.SinkDbName, s.dbTblInfo.SinkTblName))
	s.insertSuffix = []byte(";")
	s.insertRowPrefix = []byte("(")
	s.insertColSeparator = []byte(",")
	s.insertRowSuffix = []byte(")")
	s.insertRowSeparator = []byte(",")
	if isMO && len(tableDef.Pkey.Names) > 1 {
		//                                     deleteRowSeparator
		//      |<- deletePrefix  ->| 			       v
		// e.g. delete from t1 where pk1=a1 and pk2=a2 or pk1=b1 and pk2=b2 or pk1=c1 and pk2=c2 ...;
		//                                   ^
		//                            deleteColSeparator
		s.deletePrefix = []byte(fmt.Sprintf("DELETE FROM `%s`.`%s` WHERE ", s.dbTblInfo.SinkDbName, s.dbTblInfo.SinkTblName))
		s.deleteSuffix = []byte(";")
		s.deleteRowPrefix = []byte("")
		s.deleteColSeparator = []byte(" and ")
		s.deleteRowSuffix = []byte("")
		s.deleteRowSeparator = []byte(" or ")
	} else {
		//                                     	    deleteRowSeparator
		//      | <-------- deletePrefix --------> |       v
		// e.g. delete from t1 where (pk1, pk2) in ((a1,a2),(b1,b2),(c1,c2) ...);
		//                                             ^
		//                                      deleteColSeparator
		s.deletePrefix = []byte(fmt.Sprintf("DELETE FROM `%s`.`%s` WHERE %s IN (", s.dbTblInfo.SinkDbName, s.dbTblInfo.SinkTblName, genPrimaryKeyStr(tableDef)))
		s.deleteSuffix = []byte(");")
		s.deleteRowPrefix = []byte("(")
		s.deleteColSeparator = []byte(",")
		s.deleteRowSuffix = []byte(")")
		s.deleteRowSeparator = []byte(",")
	}
	s.tsInsertPrefix = make([]byte, 0, 1024)
	s.tsDeletePrefix = make([]byte, 0, 1024)

	// types
	for _, col := range tableDef.Cols {
		// skip internal columns
		if _, ok := catalog.InternalColumns[col.Name]; ok {
			continue
		}

		s.insertTypes = append(s.insertTypes, &types.Type{
			Oid:   types.T(col.Typ.Id),
			Width: col.Typ.Width,
			Scale: col.Typ.Scale,
		})
	}
	for _, name := range tableDef.Pkey.Names {
		s.pkColNames = append(s.pkColNames, name)
		col := tableDef.Cols[tableDef.Name2ColIndex[name]]
		s.deleteTypes = append(s.deleteTypes, &types.Type{
			Oid:   types.T(col.Typ.Id),
			Width: col.Typ.Width,
			Scale: col.Typ.Scale,
		})
	}

	// rows
	s.insertRow = make([]any, len(s.insertTypes))
	s.deleteRow = make([]any, 1)

	// pre
	s.preRowType = NoOp
	s.preSqlBufLen = sqlBufReserved

	// reset err
	s.ClearError()

	s.isMO = isMO
	return s
}

func (s *mysqlSinker) Run(ctx context.Context, ar *ActiveRoutine) {
	logutil.Infof("cdc mysqlSinker(%v).Run: start", s.dbTblInfo)
	defer func() {
		logutil.Infof("cdc mysqlSinker(%v).Run: end", s.dbTblInfo)
	}()

	for sqlBuffer := range s.sqlBufSendCh {
		// have error, skip
		if s.Error() != nil {
			continue
		}

		if bytes.Equal(sqlBuffer, dummy) {
			// dummy sql, do nothing
		} else if bytes.Equal(sqlBuffer, begin) {
			if err := s.mysql.SendBegin(ctx); err != nil {
				logutil.Errorf("cdc mysqlSinker(%v) SendBegin, err: %v", s.dbTblInfo, err)
				// record error
				s.SetError(err)
			}
		} else if bytes.Equal(sqlBuffer, commit) {
			if err := s.mysql.SendCommit(ctx); err != nil {
				logutil.Errorf("cdc mysqlSinker(%v) SendCommit, err: %v", s.dbTblInfo, err)
				// record error
				s.SetError(err)
			}
		} else if bytes.Equal(sqlBuffer, rollback) {
			if err := s.mysql.SendRollback(ctx); err != nil {
				logutil.Errorf("cdc mysqlSinker(%v) SendRollback, err: %v", s.dbTblInfo, err)
				// record error
				s.SetError(err)
			}
		} else {
			if err := s.mysql.Send(ctx, ar, sqlBuffer, true); err != nil {
				logutil.Errorf("cdc mysqlSinker(%v) send sql failed, err: %v, sql: %s", s.dbTblInfo, err, sqlBuffer[sqlBufReserved:])
				// record error
				s.SetError(err)
			}
		}
	}
}

func (s *mysqlSinker) Sink(ctx context.Context, data *DecoderOutput) {
	key := WatermarkKey{
		AccountId: s.accountId,
		TaskId:    s.taskId,
		DBName:    s.dbTblInfo.SourceDbName,
		TableName: s.dbTblInfo.SourceTblName,
	}
	watermark, err := s.watermarkUpdater.GetFromCache(ctx, &key)
	if err != nil {
		logutil.Error(
			"CDC-MySQLSinker-GetWatermarkFailed",
			zap.String("info", s.dbTblInfo.String()),
			zap.String("key", key.String()),
			zap.Error(err),
		)
		return
	}

	if data.toTs.LE(&watermark) {
		logutil.Error(
			"CDC-MySQLSinker-UnexpectedWatermark",
			zap.String("info", s.dbTblInfo.String()),
			zap.String("to-ts", data.toTs.ToString()),
			zap.String("watermark", watermark.ToString()),
			zap.String("key", key.String()),
		)
		return
	}

	if data.noMoreData {
		// complete sql statement
		if s.isNonEmptyInsertStmt() {
			s.sqlBuf = appendBytes(s.sqlBuf, s.insertSuffix)
			s.preSqlBufLen = len(s.sqlBuf)
		}
		if s.isNonEmptyDeleteStmt() {
			s.sqlBuf = appendBytes(s.sqlBuf, s.deleteSuffix)
			s.preSqlBufLen = len(s.sqlBuf)
		}

		// output the left sql
		if s.preSqlBufLen > sqlBufReserved {
			s.sqlBufSendCh <- s.sqlBuf[:s.preSqlBufLen]
			s.curBufIdx ^= 1
			s.sqlBuf = s.sqlBufs[s.curBufIdx]
		}

		// reset status
		s.preSqlBufLen = sqlBufReserved
		s.sqlBuf = s.sqlBuf[:s.preSqlBufLen]
		s.preRowType = NoOp
		return
	}

	start := time.Now()
	defer func() {
		v2.CdcSinkDurationHistogram.Observe(time.Since(start).Seconds())
	}()

	tsPrefix := fmt.Sprintf("/* [%s, %s) */", data.fromTs.ToString(), data.toTs.ToString())
	s.tsInsertPrefix = s.tsInsertPrefix[:0]
	s.tsInsertPrefix = append(s.tsInsertPrefix, []byte(tsPrefix)...)
	s.tsInsertPrefix = append(s.tsInsertPrefix, s.insertPrefix...)
	s.tsDeletePrefix = s.tsDeletePrefix[:0]
	s.tsDeletePrefix = append(s.tsDeletePrefix, []byte(tsPrefix)...)
	s.tsDeletePrefix = append(s.tsDeletePrefix, s.deletePrefix...)

	if data.outputTyp == OutputTypeSnapshot {
		s.sinkSnapshot(ctx, data.checkpointBat)
	} else if data.outputTyp == OutputTypeTail {
		s.sinkTail(ctx, data.insertAtmBatch, data.deleteAtmBatch)
	}
}

func (s *mysqlSinker) SendBegin() {
	s.sqlBufSendCh <- begin
}

func (s *mysqlSinker) SendCommit() {
	s.sqlBufSendCh <- commit
}

func (s *mysqlSinker) SendRollback() {
	s.sqlBufSendCh <- rollback
}

func (s *mysqlSinker) SendDummy() {
	s.sqlBufSendCh <- dummy
}

func (s *mysqlSinker) Error() error {
	if errPtr := s.err.Load().(*error); *errPtr != nil {
		if moErr, ok := (*errPtr).(*moerr.Error); !ok {
			return moerr.ConvertGoError(context.Background(), *errPtr)
		} else {
			if moErr == nil {
				return nil
			}
			return moErr
		}
	}
	return nil
}

func (s *mysqlSinker) SetError(err error) {
	s.err.Store(&err)
}

func (s *mysqlSinker) ClearError() {
	var err *moerr.Error
	s.SetError(err)
}

func (s *mysqlSinker) Reset() {
	s.sqlBufs[0] = s.sqlBufs[0][:sqlBufReserved]
	s.sqlBufs[1] = s.sqlBufs[1][:sqlBufReserved]
	s.curBufIdx = 0
	s.sqlBuf = s.sqlBufs[s.curBufIdx]
	s.preRowType = NoOp
	s.preSqlBufLen = sqlBufReserved
	s.ClearError()
}

func (s *mysqlSinker) Close() {
	// stop Run goroutine
	close(s.sqlBufSendCh)
	s.mysql.Close()
	s.sqlBufs[0] = nil
	s.sqlBufs[1] = nil
	s.sqlBuf = nil
	s.rowBuf = nil
	s.insertPrefix = nil
	s.deletePrefix = nil
	s.tsInsertPrefix = nil
	s.tsDeletePrefix = nil
	s.insertTypes = nil
	s.deleteTypes = nil
	s.insertRow = nil
	s.deleteRow = nil
}

func (s *mysqlSinker) sinkSnapshot(ctx context.Context, bat *batch.Batch) {
	var err error

	// if last row is not insert row, means this is the first snapshot batch
	if s.preRowType != InsertRow {
		s.sqlBuf = append(s.sqlBuf[:sqlBufReserved], s.tsInsertPrefix...)
		s.preRowType = InsertRow
	}

	for i := 0; i < batchRowCount(bat); i++ {
		// step1: get row from the batch
		if err = extractRowFromEveryVector(ctx, bat, i, s.insertRow); err != nil {
			s.SetError(err)
			return
		}

		// step2: transform rows into sql parts
		if err = s.getInsertRowBuf(ctx); err != nil {
			s.SetError(err)
			return
		}

		// step3: append to sqlBuf, send sql if sqlBuf is full
		if err = s.appendSqlBuf(InsertRow); err != nil {
			s.SetError(err)
			return
		}
	}
}

// insertBatch and deleteBatch is sorted by ts
// for the same ts, delete first, then insert
func (s *mysqlSinker) sinkTail(ctx context.Context, insertBatch, deleteBatch *AtomicBatch) {
	var err error

	insertIter := insertBatch.GetRowIterator().(*atomicBatchRowIter)
	deleteIter := deleteBatch.GetRowIterator().(*atomicBatchRowIter)
	defer func() {
		insertIter.Close()
		deleteIter.Close()
	}()

	// output sql until one iterator reach the end
	insertIterHasNext, deleteIterHasNext := insertIter.Next(), deleteIter.Next()

	// output the rest of insert iterator
	for insertIterHasNext {
		if err = s.sinkInsert(ctx, insertIter); err != nil {
			s.SetError(err)
			return
		}
		// get next item
		insertIterHasNext = insertIter.Next()
	}

	// output the rest of delete iterator
	for deleteIterHasNext {
		if err = s.sinkDelete(ctx, deleteIter); err != nil {
			s.SetError(err)
			return
		}
		// get next item
		deleteIterHasNext = deleteIter.Next()
	}
	s.tryFlushSqlBuf()
}

func (s *mysqlSinker) sinkInsert(ctx context.Context, insertIter *atomicBatchRowIter) (err error) {
	// if last row is not insert row, need complete the last sql first
	if s.preRowType != InsertRow {
		if s.isNonEmptyDeleteStmt() {
			s.sqlBuf = appendBytes(s.sqlBuf, s.deleteSuffix)
			s.preSqlBufLen = len(s.sqlBuf)
		}
		s.sqlBuf = append(s.sqlBuf[:s.preSqlBufLen], s.tsInsertPrefix...)
		s.preRowType = InsertRow
	}

	// step1: get row from the batch
	if err = insertIter.Row(ctx, s.insertRow); err != nil {
		return
	}

	// step2: transform rows into sql parts
	if err = s.getInsertRowBuf(ctx); err != nil {
		return
	}

	// step3: append to sqlBuf
	if err = s.appendSqlBuf(InsertRow); err != nil {
		return
	}

	return
}

func (s *mysqlSinker) sinkDelete(ctx context.Context, deleteIter *atomicBatchRowIter) (err error) {
	// if last row is not insert row, need complete the last sql first
	if s.preRowType != DeleteRow {
		if s.isNonEmptyInsertStmt() {
			s.sqlBuf = appendBytes(s.sqlBuf, s.insertSuffix)
			s.preSqlBufLen = len(s.sqlBuf)
		}
		s.sqlBuf = append(s.sqlBuf[:s.preSqlBufLen], s.tsDeletePrefix...)
		s.preRowType = DeleteRow
	}

	// step1: get row from the batch
	if err = deleteIter.Row(ctx, s.deleteRow); err != nil {
		return
	}

	// step2: transform rows into sql parts
	if err = s.getDeleteRowBuf(ctx); err != nil {
		return
	}

	// step3: append to sqlBuf
	if err = s.appendSqlBuf(DeleteRow); err != nil {
		return
	}

	return
}

func (s *mysqlSinker) tryFlushSqlBuf() (err error) {
	if s.isNonEmptyInsertStmt() {
		s.sqlBuf = appendBytes(s.sqlBuf, s.insertSuffix)
		s.preSqlBufLen = len(s.sqlBuf)
	}
	if s.isNonEmptyDeleteStmt() {
		s.sqlBuf = appendBytes(s.sqlBuf, s.deleteSuffix)
		s.preSqlBufLen = len(s.sqlBuf)
	}
	// send it to downstream
	s.sqlBufSendCh <- s.sqlBuf[:s.preSqlBufLen]
	s.curBufIdx ^= 1
	s.sqlBuf = s.sqlBufs[s.curBufIdx]

	s.preSqlBufLen = sqlBufReserved
	s.sqlBuf = s.sqlBuf[:s.preSqlBufLen]
	s.preRowType = NoOp
	return
}

// appendSqlBuf appends rowBuf to sqlBuf if not exceed its cap
// otherwise, send sql to downstream first, then reset sqlBuf and append
func (s *mysqlSinker) appendSqlBuf(rowType RowType) (err error) {
	suffixLen := len(s.insertSuffix)
	if rowType == DeleteRow {
		suffixLen = len(s.deleteSuffix)
	}

	// if s.sqlBuf has no enough space
	if len(s.sqlBuf)+len(s.rowBuf)+suffixLen > cap(s.sqlBuf) {
		// complete sql statement
		if s.isNonEmptyInsertStmt() {
			s.sqlBuf = appendBytes(s.sqlBuf, s.insertSuffix)
			s.preSqlBufLen = len(s.sqlBuf)
		}
		if s.isNonEmptyDeleteStmt() {
			s.sqlBuf = appendBytes(s.sqlBuf, s.deleteSuffix)
			s.preSqlBufLen = len(s.sqlBuf)
		}

		// send it to downstream
		s.sqlBufSendCh <- s.sqlBuf[:s.preSqlBufLen]
		s.curBufIdx ^= 1
		s.sqlBuf = s.sqlBufs[s.curBufIdx]

		// reset s.sqlBuf
		s.preSqlBufLen = sqlBufReserved
		if rowType == InsertRow {
			s.sqlBuf = append(s.sqlBuf[:s.preSqlBufLen], s.tsInsertPrefix...)
		} else {
			s.sqlBuf = append(s.sqlBuf[:s.preSqlBufLen], s.tsDeletePrefix...)
		}
	}

	// append bytes
	if s.isNonEmptyInsertStmt() {
		s.sqlBuf = appendBytes(s.sqlBuf, s.insertRowSeparator)
	}
	if s.isNonEmptyDeleteStmt() {
		s.sqlBuf = appendBytes(s.sqlBuf, s.deleteRowSeparator)
	}
	s.sqlBuf = append(s.sqlBuf, s.rowBuf...)
	return
}

func (s *mysqlSinker) isNonEmptyDeleteStmt() bool {
	return s.preRowType == DeleteRow && len(s.sqlBuf)-s.preSqlBufLen > len(s.tsDeletePrefix)
}

func (s *mysqlSinker) isNonEmptyInsertStmt() bool {
	return s.preRowType == InsertRow && len(s.sqlBuf)-s.preSqlBufLen > len(s.tsInsertPrefix)
}

// getInsertRowBuf convert insert row to string
func (s *mysqlSinker) getInsertRowBuf(ctx context.Context) (err error) {
	s.rowBuf = appendBytes(s.rowBuf[:0], s.insertRowPrefix)
	for i := 0; i < len(s.insertRow); i++ {
		if i != 0 {
			s.rowBuf = appendBytes(s.rowBuf, s.insertColSeparator)
		}
		//transform column into text values
		if s.rowBuf, err = convertColIntoSql(ctx, s.insertRow[i], s.insertTypes[i], s.rowBuf); err != nil {
			return
		}
	}
	s.rowBuf = appendBytes(s.rowBuf, s.insertRowSuffix)
	return
}

var unpackWithSchema = types.UnpackWithSchema

// getDeleteRowBuf convert delete row to string
func (s *mysqlSinker) getDeleteRowBuf(ctx context.Context) (err error) {
	s.rowBuf = appendBytes(s.rowBuf[:0], s.deleteRowPrefix)

	if len(s.deleteTypes) == 1 {
		// single column pk
		// transform column into text values
		if s.rowBuf, err = convertColIntoSql(ctx, s.deleteRow[0], s.deleteTypes[0], s.rowBuf); err != nil {
			return
		}
	} else {
		// composite pk
		var pkTuple types.Tuple
		if pkTuple, _, err = unpackWithSchema(s.deleteRow[0].([]byte)); err != nil {
			return
		}
		for i, pkEle := range pkTuple {
			if i > 0 {
				s.rowBuf = appendBytes(s.rowBuf, s.deleteColSeparator)
			}
			//transform column into text values
			if s.isMO {
				s.rowBuf = appendBytes(s.rowBuf, []byte(s.pkColNames[i]+"="))
			}
			if s.rowBuf, err = convertColIntoSql(ctx, pkEle, s.deleteTypes[i], s.rowBuf); err != nil {
				return
			}
		}
	}

	s.rowBuf = appendBytes(s.rowBuf, s.deleteRowSuffix)
	return
}

var _ Sink = new(mysqlSink)

type mysqlSink struct {
	conn           *sql.DB
	tx             *sql.Tx
	user, password string
	ip             string
	port           int

	retryTimes    int
	retryDuration time.Duration
	timeout       string

	debugTxnRecorder struct {
		doRecord bool
		txnSQL   []string
		sqlBytes int
	}
}

var NewMysqlSink = func(
	user, password string,
	ip string, port int,
	retryTimes int,
	retryDuration time.Duration,
	timeout string,
	doRecord bool,
) (Sink, error) {
	ret := &mysqlSink{
		user:          user,
		password:      password,
		ip:            ip,
		port:          port,
		retryTimes:    retryTimes,
		retryDuration: retryDuration,
		timeout:       timeout,
	}

	ret.debugTxnRecorder.doRecord = doRecord

	err := ret.connect()
	return ret, err
}

func (s *mysqlSink) recordTxnSQL(sqlBuf []byte) {
	if !s.debugTxnRecorder.doRecord {
		return
	}

	s.debugTxnRecorder.sqlBytes += len(sqlBuf)
	s.debugTxnRecorder.txnSQL = append(
		s.debugTxnRecorder.txnSQL, string(sqlBuf[sqlBufReserved:]))
}

func (s *mysqlSink) infoRecordedTxnSQLs(err error) {
	if !s.debugTxnRecorder.doRecord {
		return
	}

	if len(s.debugTxnRecorder.txnSQL) == 0 {
		return
	}

	if s.debugTxnRecorder.sqlBytes <= mpool.MB {
		buf := bytes.Buffer{}
		for _, sqlStr := range s.debugTxnRecorder.txnSQL {
			buf.WriteString(sqlStr)
			buf.WriteString("; ")
		}

		logutil.Info("CDC-RECORDED-TXN",
			zap.Error(err),
			zap.String("details", buf.String()))
	}

	s.resetRecordedTxn()
}

func (s *mysqlSink) resetRecordedTxn() {
	if !s.debugTxnRecorder.doRecord {
		return
	}
	s.debugTxnRecorder.sqlBytes = 0
	s.debugTxnRecorder.txnSQL = s.debugTxnRecorder.txnSQL[:0]
}

// Send must leave 5 bytes at the head of sqlBuf
func (s *mysqlSink) Send(ctx context.Context, ar *ActiveRoutine, sqlBuf []byte, needRetry bool) error {
	reuseQueryArg := sql.NamedArg{
		Name:  mysql.ReuseQueryBuf,
		Value: sqlBuf,
	}

	s.recordTxnSQL(sqlBuf)

	f := func() (err error) {
		if s.tx != nil {
			_, err = s.tx.Exec(fakeSql, reuseQueryArg)
		} else {
			_, err = s.conn.Exec(fakeSql, reuseQueryArg)
		}

		if err != nil {

			s.infoRecordedTxnSQLs(err)

			logutil.Errorf("cdc mysqlSink Send failed, err: %v, sql: %s", err, sqlBuf[sqlBufReserved:min(len(sqlBuf), sqlPrintLen)])
			//logutil.Errorf("cdc mysqlSink Send failed, err: %v, sql: %s", err, sqlBuf[sqlBufReserved:])
		}
		//logutil.Infof("cdc mysqlSink Send success, sql: %s", sqlBuf[sqlBufReserved:])
		return
	}

	if !needRetry {
		return f()
	}
	return s.retry(ctx, ar, f)
}

func (s *mysqlSink) SendBegin(ctx context.Context) (err error) {
	s.resetRecordedTxn()
	s.tx, err = s.conn.BeginTx(ctx, nil)
	return err
}

func (s *mysqlSink) SendCommit(_ context.Context) error {
	s.resetRecordedTxn()
	defer func() {
		s.tx = nil
	}()
	return s.tx.Commit()
}

func (s *mysqlSink) SendRollback(_ context.Context) error {
	s.resetRecordedTxn()
	defer func() {
		s.tx = nil
	}()
	return s.tx.Rollback()
}

func (s *mysqlSink) Close() {
	if s.conn != nil {
		_ = s.conn.Close()
		s.conn = nil
	}

	s.debugTxnRecorder.txnSQL = nil
}

func (s *mysqlSink) connect() (err error) {
	s.conn, err = OpenDbConn(s.user, s.password, s.ip, s.port, s.timeout)
	return err
}

func (s *mysqlSink) retry(ctx context.Context, ar *ActiveRoutine, fn func() error) (err error) {
	needRetry := func(retry int, startTime time.Time) bool {
		// retryTimes == -1 means retry forever
		// do not exceed retryTimes and retryDuration
		return (s.retryTimes == -1 || retry < s.retryTimes) && time.Since(startTime) < s.retryDuration
	}
	for retry, startTime := 0, time.Now(); needRetry(retry, startTime); retry++ {
		select {
		case <-ctx.Done():
			return
		case <-ar.Pause:
			return
		case <-ar.Cancel:
			return
		default:
		}

		start := time.Now()
		err = fn()
		v2.CdcSendSqlDurationHistogram.Observe(time.Since(start).Seconds())

		// return if success
		if err == nil {
			return
		}

		logutil.Errorf("cdc mysqlSink retry failed, err: %v", err)
		v2.CdcMysqlSinkErrorCounter.Inc()
		time.Sleep(time.Second)
	}
	return moerr.NewInternalError(ctx, "cdc mysqlSink retry exceed retryTimes or retryDuration")
}

//type matrixoneSink struct {
//}
//
//func (*matrixoneSink) Send(ctx context.Context, data *DecoderOutput) error {
//	return nil
//}

func genPrimaryKeyStr(tableDef *plan.TableDef) string {
	buf := strings.Builder{}
	buf.WriteByte('(')
	for i, pkName := range tableDef.Pkey.Names {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(pkName)
	}
	buf.WriteByte(')')
	return buf.String()
}
