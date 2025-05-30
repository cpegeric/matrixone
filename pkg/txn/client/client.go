// Copyright 2022 Matrix Origin
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

package client

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"math"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"github.com/matrixorigin/matrixone/pkg/common/log"
	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/common/runtime"
	"github.com/matrixorigin/matrixone/pkg/common/stopper"
	"github.com/matrixorigin/matrixone/pkg/lockservice"
	"github.com/matrixorigin/matrixone/pkg/pb/timestamp"
	"github.com/matrixorigin/matrixone/pkg/pb/txn"
	"github.com/matrixorigin/matrixone/pkg/txn/clock"
	"github.com/matrixorigin/matrixone/pkg/txn/rpc"
	"github.com/matrixorigin/matrixone/pkg/txn/util"
	v2 "github.com/matrixorigin/matrixone/pkg/util/metric/v2"
	"go.uber.org/ratelimit"
	"go.uber.org/zap"
)

// WithTxnIDGenerator setup txn id generator
func WithTxnIDGenerator(generator TxnIDGenerator) TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.generator = generator
	}
}

// WithLockService setup lock service
func WithLockService(lockService lockservice.LockService) TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.lockService = lockService
	}
}

// WithEnableSacrificingFreshness sacrifice freshness to reduce the waiting time for transaction start,
// which will help to improve the latency of the transaction, but will sacrifice some data freshness.
//
// In Push mode, TN will bba to push Logtail to CN, if we need to guarantee the freshness of data, then
// we need to use the current latest time as the start time of the transaction, this will ensure that
// enough logtail is collected before the transaction read/write starts, but this will have some delayed
// waiting time.
//
// But if we can accept to sacrifice some data freshness, we can optimize this waiting time, we just need to
// use the latest logtail timestamp received + 1 as the transaction start timestamp, so we can not wait.

// When making this optimization, there are some scenarios where data consistency must be guaranteed, such as
// a database connection in a session where the latter transaction must be able to read the data committed by
// the previous transaction, then it is necessary to maintain a Session-level transaction last commit time, and
// the start time of the next transaction cannot be less than this value.
//
// If we need to ensure that all the transactions on a CN can read the writes of the previous committed
// transaction, then we can use WithEnableCNBasedConsistency to turn on.
func WithEnableSacrificingFreshness() TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.enableSacrificingFreshness = true
	}
}

// WithTimestampWaiter setup timestamp waiter to get the latest applied committed timestamp from logtail.
func WithTimestampWaiter(waiter TimestampWaiter) TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.timestampWaiter = waiter
	}
}

// WithEnableCNBasedConsistency let all transactions on a CN see writes committed by other
// transactions before them. When this feature is enabled, the client maintains a CN-Based
// commit timestamp, and when opening a new transaction, it adjusts the transaction's snapshot
// timestamp to at least >= lastCommitTimestamp, so that it can see the writes of the previously
// committed transaction
func WithEnableCNBasedConsistency() TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.enableCNBasedConsistency = true
	}
}

// WithEnableRefreshExpression in RC mode, in the event of a conflict, the later transaction needs
// to see the latest data after the previous transaction commits. At this time we need to re-read
// the data, re-read the latest data, and re-compute the expression.
func WithEnableRefreshExpression() TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.enableRefreshExpression = true
	}
}

// WithEnableLeakCheck enable txn leak check. Used to found any txn is not committed or rolled back.
func WithEnableLeakCheck(
	maxActiveAges time.Duration,
	leakHandleFunc func([]ActiveTxn)) TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.leakChecker = newLeakCheck(maxActiveAges, leakHandleFunc)
	}
}

// WithTxnLimit flow control of transaction creation, maximum number of transactions per second
func WithTxnLimit(n int) TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.limiter = ratelimit.New(n, ratelimit.Per(time.Second))
	}
}

// WithMaxActiveTxn is the count of max active txn in current cn.  If reached max value, the txn is
// added to a FIFO queue. Default is unlimited.
func WithMaxActiveTxn(n int) TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.maxActiveTxn = n
	}
}

// WithNormalStateNoWait sets the normalStateNoWait value of txnClient.
func WithNormalStateNoWait(t bool) TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.normalStateNoWait = t
	}
}

func WithTxnOpenedCallback(callbacks []func(op TxnOperator)) TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.txnOpenedCallbacks = callbacks
	}
}

func WithCheckDup() TxnClientCreateOption {
	return func(tc *txnClient) {
		tc.enableCheckDup = true
	}
}

var _ TxnClient = (*txnClient)(nil)

type status int

const (
	paused        = status(0)
	normal status = status(1)
)

type txnClient struct {
	sid                        string
	stopper                    *stopper.Stopper
	logger                     *log.MOLogger
	clock                      clock.Clock
	sender                     rpc.TxnSender
	generator                  TxnIDGenerator
	lockService                lockservice.LockService
	timestampWaiter            TimestampWaiter
	leakChecker                *leakChecker
	limiter                    ratelimit.Limiter
	maxActiveTxn               int
	enableCheckDup             bool
	enableCNBasedConsistency   bool
	enableSacrificingFreshness bool
	enableRefreshExpression    bool
	txnOpenedCallbacks         []func(TxnOperator)

	// normalStateNoWait is used to control if wait for the txn client's
	// state to be normal. If it is false, which is default value, wait
	// until the txn client's state to be normal; otherwise, if it is true,
	// do not wait, and just return an error.
	normalStateNoWait bool

	atomic struct {
		// we maintain a CN-based last commit timestamp to ensure that
		// a txn with that CN can see previous writes.
		// FIXME(fagongzi): this is a remedial solution to disable the
		// cn-based commit ts when the session-level last commit ts have
		// been processed.
		latestCommitTS atomic.Pointer[timestamp.Timestamp]
		// just for bvt testing
		forceSyncCommitTimes atomic.Uint64
	}

	mu struct {
		sync.RWMutex
		// cond is used to control if we can create new txn and notify
		// if the state is changed.
		cond *sync.Cond
		// indicate whether the CN can provide service normally.
		state status
		// user active txns
		users int
		// all active txns
		activeTxns map[string]*txnOperator
		// FIFO queue for ready to active txn
		waitActiveTxns            []*txnOperator
		waitMarkAllActiveAbortedC chan struct{}
	}

	abortC chan time.Time
}

func (client *txnClient) GetState() TxnState {
	client.mu.Lock()
	defer client.mu.Unlock()
	at := make([]string, 0, len(client.mu.activeTxns))
	for k := range client.mu.activeTxns {
		at = append(at, hex.EncodeToString([]byte(k)))
	}
	wt := make([]string, 0, len(client.mu.waitActiveTxns))
	for _, v := range client.mu.waitActiveTxns {
		wt = append(wt, hex.EncodeToString(v.reset.txnID))
	}
	return TxnState{
		State:          int(client.mu.state),
		Users:          client.mu.users,
		ActiveTxns:     at,
		WaitActiveTxns: wt,
		LatestTS:       client.timestampWaiter.LatestTS(),
	}
}

// NewTxnClient create a txn client with TxnSender and Options
func NewTxnClient(
	sid string,
	sender rpc.TxnSender,
	options ...TxnClientCreateOption,
) TxnClient {
	c := &txnClient{
		sid:    sid,
		logger: util.GetLogger(sid),
		clock:  runtime.ServiceRuntime(sid).Clock(),
		sender: sender,
		abortC: make(chan time.Time, 1),
	}
	c.stopper = stopper.NewStopper("txn-client", stopper.WithLogger(c.logger.RawLogger()))
	c.mu.state = paused
	c.mu.cond = sync.NewCond(&c.mu)
	c.mu.activeTxns = make(map[string]*txnOperator, 100000)
	for _, opt := range options {
		opt(c)
	}
	c.adjust()
	c.startLeakChecker()
	if err := c.stopper.RunTask(c.handleMarkActiveTxnAborted); err != nil {
		panic(err)
	}
	return c
}

func (client *txnClient) adjust() {
	if client.generator == nil {
		client.generator = newUUIDTxnIDGenerator(client.sid)
	}
	if runtime.ServiceRuntime(client.sid).Clock() == nil {
		panic("txn clock not set")
	}
	if client.limiter == nil {
		client.limiter = ratelimit.NewUnlimited()
	}
	if client.maxActiveTxn == 0 {
		client.maxActiveTxn = math.MaxInt
	}
}

func (client *txnClient) New(
	ctx context.Context,
	minTS timestamp.Timestamp,
	options ...TxnOption,
) (TxnOperator, error) {
	op := newTxnOperator(
		client.sid,
		client.clock,
		client.sender,
		client.newTxnMeta(),
		client.getTxnOptions(options)...,
	)
	return client.doCreateTxn(
		ctx,
		op,
		minTS,
	)
}

func (client *txnClient) RestartTxn(
	ctx context.Context,
	txnOp TxnOperator,
	minTS timestamp.Timestamp,
	options ...TxnOption,
) (TxnOperator, error) {
	op := txnOp.(*txnOperator)
	op.init(
		client.newTxnMeta(),
		client.getTxnOptions(options)...,
	)
	return client.doCreateTxn(
		ctx,
		op,
		minTS,
	)
}

func (client *txnClient) doCreateTxn(
	ctx context.Context,
	op *txnOperator,
	minTS timestamp.Timestamp,
) (TxnOperator, error) {
	start := time.Now()
	defer func() {
		v2.TxnCreateTotalDurationHistogram.Observe(time.Since(start).Seconds())
	}()

	// we take a token from the limiter to control the number of transactions created per second.
	client.limiter.Take()

	op.timestampWaiter = client.timestampWaiter
	op.AppendEventCallback(
		ClosedEvent,
		client.updateLastCommitTS,
		client.closeTxn,
	)

	if err := client.openTxn(op); err != nil {
		return nil, err
	}

	for _, cb := range client.txnOpenedCallbacks {
		cb(op)
	}

	ts := client.determineTxnSnapshot(minTS)
	if !op.opts.skipWaitPushClient {
		if err := op.UpdateSnapshot(ctx, ts); err != nil {
			_ = op.Rollback(ctx)
			return nil, errors.Join(err, moerr.NewTxnError(ctx, "update txn snapshot"))
		}
	}

	util.LogTxnSnapshotTimestamp(
		client.logger,
		minTS,
		ts,
	)

	if err := op.waitActive(ctx); err != nil {
		_ = op.Rollback(ctx)
		return nil, errors.Join(err, moerr.NewTxnError(ctx, "wait active"))
	}
	return op, nil
}

func (client *txnClient) NewWithSnapshot(
	snapshot txn.CNTxnSnapshot,
) (TxnOperator, error) {
	op := newTxnOperatorWithSnapshot(
		client.logger,
		client.sender,
		snapshot,
	)
	op.timestampWaiter = client.timestampWaiter
	return op, nil
}

func (client *txnClient) Close() error {
	client.stopper.Stop()
	if client.leakChecker != nil {
		client.leakChecker.close()
	}
	return client.sender.Close()
}

func (client *txnClient) MinTimestamp() timestamp.Timestamp {
	client.mu.RLock()
	defer client.mu.RUnlock()

	min := timestamp.Timestamp{}
	for _, op := range client.mu.activeTxns {
		if min.IsEmpty() ||
			op.Txn().SnapshotTS.Less(min) {
			min = op.Txn().SnapshotTS
		}
	}
	return min
}

func (client *txnClient) WaitLogTailAppliedAt(
	ctx context.Context,
	ts timestamp.Timestamp) (timestamp.Timestamp, error) {
	if client.timestampWaiter == nil {
		return timestamp.Timestamp{}, nil
	}
	return client.timestampWaiter.GetTimestamp(ctx, ts)
}

func (client *txnClient) getTxnIsolation() txn.TxnIsolation {
	if v, ok := runtime.ServiceRuntime(client.sid).GetGlobalVariables(runtime.TxnIsolation); ok {
		return v.(txn.TxnIsolation)
	}
	return txn.TxnIsolation_RC
}

func (client *txnClient) getTxnMode() txn.TxnMode {
	if v, ok := runtime.ServiceRuntime(client.sid).GetGlobalVariables(runtime.TxnMode); ok {
		return v.(txn.TxnMode)
	}
	return txn.TxnMode_Pessimistic
}

func (client *txnClient) updateLastCommitTS(event TxnEvent) {
	if event.Txn.CommitTS.IsEmpty() {
		return
	}

	var old *timestamp.Timestamp
	new := &event.Txn.CommitTS
	for {
		old = client.atomic.latestCommitTS.Load()
		if old != nil && old.GreaterEq(event.Txn.CommitTS) {
			return
		}

		if client.atomic.latestCommitTS.CompareAndSwap(old, new) {
			return
		}
	}
}

// determineTxnSnapshot assuming we determine the timestamp to be ts, the final timestamp
// returned will be ts+1. This is because we need to see the submitted data for ts, and the
// timestamp for all things is ts+1.
func (client *txnClient) determineTxnSnapshot(minTS timestamp.Timestamp) timestamp.Timestamp {
	start := time.Now()
	defer func() {
		v2.TxnDetermineSnapshotDurationHistogram.Observe(time.Since(start).Seconds())
	}()

	// always use the current ts as txn's snapshot ts is enableSacrificingFreshness
	if !client.enableSacrificingFreshness {
		// TODO: Consider how to handle clock offsets. If use Clock-SI, can use the current
		// time minus the maximum clock offset as the transaction's snapshotTimestamp to avoid
		// conflicts due to clock uncertainty.
		now, _ := client.clock.Now()
		minTS = now
	} else if client.enableCNBasedConsistency {
		minTS = client.adjustTimestamp(minTS)
	}

	return minTS
}

func (client *txnClient) adjustTimestamp(ts timestamp.Timestamp) timestamp.Timestamp {
	v := client.atomic.latestCommitTS.Load()
	if v != nil && v.Greater(ts) {
		return *v
	}
	return ts
}

func (client *txnClient) GetLatestCommitTS() timestamp.Timestamp {
	return client.adjustTimestamp(timestamp.Timestamp{})
}

func (client *txnClient) SyncLatestCommitTS(ts timestamp.Timestamp) {
	client.updateLastCommitTS(TxnEvent{Txn: txn.TxnMeta{CommitTS: ts}})
	if client.timestampWaiter != nil {
		ctx, cancel := context.WithTimeoutCause(context.Background(), time.Minute*5, moerr.CauseSyncLatestCommitT)
		defer cancel()
		for {
			_, err := client.timestampWaiter.GetTimestamp(ctx, ts)
			if err == nil {
				break
			}
			err = moerr.AttachCause(ctx, err)
			client.logger.Fatal("wait latest commit ts failed", zap.Error(err))
		}
	}
	client.atomic.forceSyncCommitTimes.Add(1)
}

func (client *txnClient) GetSyncLatestCommitTSTimes() uint64 {
	return client.atomic.forceSyncCommitTimes.Load()
}

func (client *txnClient) openTxn(op *txnOperator) error {
	client.mu.Lock()
	defer func() {
		v2.TxnActiveQueueSizeGauge.Set(float64(len(client.mu.activeTxns)))
		v2.TxnWaitActiveQueueSizeGauge.Set(float64(len(client.mu.waitActiveTxns)))
		client.mu.Unlock()
	}()

	client.waitMarkAllActiveAbortedLocked()

	if !op.opts.skipWaitPushClient {
		for client.mu.state == paused {
			if client.normalStateNoWait {
				return moerr.NewInternalErrorNoCtx("cn service is not ready, retry later")
			}

			if op.opts.options.WaitPausedDisabled() {
				return moerr.NewInvalidStateNoCtx("txn client is in pause state")
			}

			client.logger.Warn("txn client is in pause state, wait for it to be ready",
				zap.String("txn ID", hex.EncodeToString(op.reset.txnID)))
			// Wait until the txn client's state changed to normal, and it will probably take
			// no more than 5 seconds in theory.
			client.mu.cond.Wait()
			client.logger.Warn("txn client is in ready state",
				zap.String("txn ID", hex.EncodeToString(op.reset.txnID)))
		}
	}

	if !op.opts.options.UserTxn() ||
		client.mu.users < client.maxActiveTxn {
		client.addActiveTxnLocked(op)
		return nil
	}
	op.reset.waiter = newWaiter(timestamp.Timestamp{})
	op.reset.waiter.ref()
	client.mu.waitActiveTxns = append(client.mu.waitActiveTxns, op)
	return nil
}

func (client *txnClient) closeTxn(event TxnEvent) {
	txn := event.Txn

	client.mu.Lock()
	defer func() {
		v2.TxnActiveQueueSizeGauge.Set(float64(len(client.mu.activeTxns)))
		v2.TxnWaitActiveQueueSizeGauge.Set(float64(len(client.mu.waitActiveTxns)))
		client.mu.Unlock()
	}()

	if moerr.IsMoErrCode(event.Err, moerr.ErrCannotCommitOnInvalidCN) {
		client.markAllActiveTxnAborted()
	}

	key := string(txn.ID)
	op, ok := client.mu.activeTxns[key]
	if ok {
		v2.TxnLifeCycleDurationHistogram.Observe(time.Since(op.reset.createAt).Seconds())

		delete(client.mu.activeTxns, key)
		client.removeFromLeakCheck(txn.ID)
		if !op.opts.options.UserTxn() {
			return
		}
		client.mu.users--
		if client.mu.users < 0 {
			panic("BUG: user txns < 0")
		}
		if len(client.mu.waitActiveTxns) > 0 {
			newCanAdded := client.maxActiveTxn - client.mu.users
			for i := 0; i < newCanAdded; i++ {
				op := client.fetchWaitActiveOpLocked()
				if op == nil {
					return
				}
				client.addActiveTxnLocked(op)
				op.notifyActive()
			}
		}
	} else if ok = client.removeFromWaitActiveLocked(txn.ID); ok {
		client.removeFromLeakCheck(txn.ID)
	} else {
		client.logger.Warn("txn closed",
			zap.String("txn ID", hex.EncodeToString(txn.ID)),
			zap.String("stack", string(debug.Stack())))
	}
}

func (client *txnClient) addActiveTxnLocked(op *txnOperator) {
	if op.opts.options.UserTxn() {
		client.mu.users++
	}
	client.mu.activeTxns[string(op.reset.txnID)] = op
	client.addToLeakCheck(op)
}

func (client *txnClient) fetchWaitActiveOpLocked() *txnOperator {
	if len(client.mu.waitActiveTxns) == 0 {
		return nil
	}
	op := client.mu.waitActiveTxns[0]
	client.mu.waitActiveTxns = append(client.mu.waitActiveTxns[:0], client.mu.waitActiveTxns[1:]...)
	return op
}

func (client *txnClient) Pause() {
	client.mu.Lock()
	defer client.mu.Unlock()

	client.logger.Info("txn client status changed to paused")
	client.mu.state = paused
}

func (client *txnClient) Resume() {
	client.mu.Lock()
	defer client.mu.Unlock()

	client.logger.Info("txn client status changed to normal")
	client.mu.state = normal

	// Notify all waiting transactions to goon with the opening operation.
	if !client.normalStateNoWait {
		client.mu.cond.Broadcast()
	}
}

func (client *txnClient) startLeakChecker() {
	if client.leakChecker != nil {
		client.leakChecker.start()
	}
}

func (client *txnClient) addToLeakCheck(op *txnOperator) {
	if client.leakChecker != nil {
		client.leakChecker.txnOpened(op, op.reset.txnID, op.opts.options)
	}
}

func (client *txnClient) removeFromLeakCheck(id []byte) {
	if client.leakChecker != nil {
		client.leakChecker.txnClosed(id)
	}
}

func (client *txnClient) IterTxns(fn func(TxnOverview) bool) {
	ops := client.getAllTxnOperators()

	for _, op := range ops {
		if !fn(op.GetOverview()) {
			return
		}
	}
}

func (client *txnClient) getAllTxnOperators() []*txnOperator {
	client.mu.RLock()
	defer client.mu.RUnlock()

	ops := make([]*txnOperator, 0, len(client.mu.activeTxns)+len(client.mu.waitActiveTxns))
	for _, op := range client.mu.activeTxns {
		ops = append(ops, op)
	}
	ops = append(ops, client.mu.waitActiveTxns...)
	return ops
}

func (client *txnClient) newTxnMeta() txn.TxnMeta {
	txnMeta := txn.TxnMeta{}
	txnMeta.ID = client.generator.Generate()
	txnMeta.Mode = client.getTxnMode()
	txnMeta.Isolation = client.getTxnIsolation()
	if client.lockService != nil {
		txnMeta.LockService = client.lockService.GetServiceID()
	}
	return txnMeta
}

func (client *txnClient) getTxnOptions(
	options []TxnOption,
) []TxnOption {
	options = append(options,
		WithTxnCNCoordinator(),
		WithTxnLockService(client.lockService))
	if client.enableCheckDup {
		options = append(options, WithTxnEnableCheckDup())
	}
	return options
}

func (client *txnClient) markAllActiveTxnAborted() {
	select {
	case client.abortC <- time.Now():
	default:
	}
}

func (client *txnClient) handleMarkActiveTxnAborted(
	ctx context.Context,
) {
	defer client.logger.InfoAction("mark active txn aborted task")()

	for {
		select {
		case <-ctx.Done():
			return
		case from := <-client.abortC:
			fn := func() {
				client.mu.Lock()
				client.mu.waitMarkAllActiveAbortedC = make(chan struct{})
				ops := make([]*txnOperator, 0, len(client.mu.activeTxns))
				for _, op := range client.mu.activeTxns {
					if op.reset.createAt.Before(from) {
						ops = append(ops, op)
					}
				}
				client.mu.Unlock()

				for _, op := range ops {
					op.addFlag(AbortedFlag)
				}

				client.mu.Lock()
				close(client.mu.waitMarkAllActiveAbortedC)
				client.mu.waitMarkAllActiveAbortedC = nil
				client.mu.Unlock()
			}
			fn()

			if err := client.lockService.(lockservice.ResumeLockService).Resume(); err != nil {
				client.logger.Error(
					"resume lock service failed",
					zap.Error(err),
				)
			}
		}
	}
}

func (client *txnClient) removeFromWaitActiveLocked(txnID []byte) bool {
	var ok bool
	values := client.mu.waitActiveTxns[:0]
	for _, op := range client.mu.waitActiveTxns {
		if bytes.Equal(op.reset.txnID, txnID) {
			ok = true
			continue
		}
		values = append(values, op)
	}
	client.mu.waitActiveTxns = values
	return ok
}

func (client *txnClient) waitMarkAllActiveAbortedLocked() {
	if client.mu.waitMarkAllActiveAbortedC != nil {
		c := client.mu.waitMarkAllActiveAbortedC
		client.mu.Unlock()
		<-c
		client.mu.Lock()
	}
}
