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

package batchstoredriver

import (
	"context"
	"sync"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/common/mpool"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/tae/logstore/driver"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/tae/logstore/driver/entry"
	storeEntry "github.com/matrixorigin/matrixone/pkg/vm/engine/tae/logstore/entry"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/tae/testutils"
	"github.com/panjf2000/ants/v2"
	"github.com/stretchr/testify/assert"
)

const (
	Modulename = "batchstoredriver"
)

func initEnv(t *testing.T) *baseStore {
	dir := testutils.InitTestEnv(Modulename, t)
	name := "mock"
	cfg := &StoreCfg{
		RotateChecker: NewMaxSizeRotateChecker(int(mpool.KB) * 3),
	}
	s, err := NewBaseStore(dir, name, cfg)
	assert.NoError(t, err)
	return s
}

func restartStore(s *baseStore, t *testing.T) *baseStore {
	err := s.Close()
	assert.NoError(t, err)
	maxlsn := s.GetDSN()
	// for ver,lsns:=range s.addrs{
	// 	logutil.Infof("v%d lsn%v",ver,lsns.Intervals)
	// }
	cfg := &StoreCfg{
		RotateChecker: NewMaxSizeRotateChecker(int(mpool.KB) * 3),
	}
	s, err = NewBaseStore(s.dir, s.name, cfg)
	assert.NoError(t, err)
	tempLsn := uint64(0)
	err = s.Replay(
		context.Background(),
		func(e *entry.Entry) driver.ReplayEntryState {
			if e.DSN < tempLsn {
				panic(moerr.NewInternalErrorNoCtxf("logic error %d<%d", e.DSN, tempLsn))
			}
			tempLsn = e.DSN
			_, err = s.Read(e.DSN)
			assert.NoError(t, err)
			// logutil.Infof("lsn is %d",e.DSN)
			return driver.RE_Nomal
		},
		func() driver.ReplayMode {
			return driver.ReplayMode_ReplayForWrite
		},
		nil,
	)
	assert.NoError(t, err)
	assert.Equal(t, maxlsn, s.GetDSN())
	assert.Equal(t, maxlsn, s.synced)
	assert.Equal(t, maxlsn, s.syncing)
	// for ver,lsns:=range s.addrs{
	// 	logutil.Infof("v%d lsn%v",ver,lsns.Intervals)
	// }
	return s
}

func concurrentAppendReadCheckpoint(s *baseStore, t *testing.T) {
	entryCnt := 100
	entries := make([]*entry.Entry, 0)
	for i := 0; i < entryCnt; i++ {
		e := entry.MockEntry()
		entries = append(entries, e)
	}

	wg := sync.WaitGroup{}
	worker, _ := ants.NewPool(10)
	defer worker.Release()
	appendfn := func(i int) func() {
		return func() {
			e := entries[i]
			err := s.Append(e)
			assert.NoError(t, err)
			lsn := s.GetDSN()
			assert.GreaterOrEqual(t, lsn, e.DSN)
			wg.Done()
		}
	}

	readfn := func(i int) func() {
		return func() {
			e := entries[i]
			assert.NoError(t, e.WaitDone())
			e2, err := s.Read(e.DSN)
			assert.NoError(t, err)
			assert.Equal(t, e2.Entry.GetInfo().(*storeEntry.Info).GroupLSN, e.Info.GroupLSN)
			e2.Entry.Free()
			wg.Done()
		}
	}

	truncatefn := func(i int) func() {
		return func() {
			e := entries[i]
			err := s.Truncate(e.DSN)
			assert.NoError(t, err)
			e.Entry.Free()
			lsn, err := s.GetTruncated()
			assert.NoError(t, err)
			assert.GreaterOrEqual(t, lsn, e.DSN)
			wg.Done()
		}
	}

	for i := range entries {
		wg.Add(1)
		_ = worker.Submit(appendfn(i))
		wg.Add(1)
		_ = worker.Submit(readfn(i))
	}
	wg.Wait()
	for i := range entries {
		wg.Add(1)
		_ = worker.Submit(truncatefn(i))
	}
	wg.Wait()
}

func TestDriver(t *testing.T) {
	defer testutils.AfterTest(t)()
	s := initEnv(t)
	concurrentAppendReadCheckpoint(s, t)
	s = restartStore(s, t)
	concurrentAppendReadCheckpoint(s, t)
	s.Close()
}
