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

package db

import (
	"sync"

	"github.com/matrixorigin/matrixone/pkg/objectio"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/tae/common"
	"github.com/matrixorigin/matrixone/pkg/vm/engine/tae/tasks"
)

func ScopeConflictCheck(oldScope, newScope *common.ID) (err error) {
	if oldScope.TableID != newScope.TableID {
		return
	}
	if !oldScope.SegmentID().Eq(*newScope.SegmentID()) &&
		!objectio.IsEmptySegid(oldScope.SegmentID()) &&
		!objectio.IsEmptySegid(newScope.SegmentID()) {
		return
	}
	if oldScope.BlockID != newScope.BlockID &&
		!objectio.IsEmptyBlkid(&oldScope.BlockID) &&
		!objectio.IsEmptyBlkid(&newScope.BlockID) {
		return
	}
	return tasks.ErrScheduleScopeConflict
}

type asyncJobDispatcher struct {
	sync.RWMutex
	*tasks.BaseDispatcher
	actives map[common.ID]struct{}
}

func newAsyncJobDispatcher() *asyncJobDispatcher {
	return &asyncJobDispatcher{
		actives:        make(map[common.ID]struct{}),
		BaseDispatcher: tasks.NewBaseDispatcher(),
	}
}

func (dispatcher *asyncJobDispatcher) checkConflictLocked(scopes []common.ID) (err error) {
	for active := range dispatcher.actives {
		for _, scope := range scopes {
			if err = ScopeConflictCheck(&active, &scope); err != nil {
				return
			}
		}
	}
	return
}

func (dispatcher *asyncJobDispatcher) TryDispatch(task tasks.Task) (err error) {
	mscoped := task.(tasks.MScopedTask)
	scopes := mscoped.Scopes()
	if len(scopes) == 0 {
		dispatcher.Dispatch(task)
		return
	}
	dispatcher.Lock()
	if err = dispatcher.checkConflictLocked(scopes); err != nil {
		dispatcher.Unlock()
		return
	}
	for _, scope := range scopes {
		dispatcher.actives[scope] = struct{}{}
	}
	task.AddObserver(dispatcher)
	dispatcher.Unlock()
	dispatcher.Dispatch(task)
	return
}

func (dispatcher *asyncJobDispatcher) OnExecDone(v any) {
	task := v.(tasks.MScopedTask)
	scopes := task.Scopes()
	dispatcher.Lock()
	for _, scope := range scopes {
		delete(dispatcher.actives, scope)
	}
	dispatcher.Unlock()
}
