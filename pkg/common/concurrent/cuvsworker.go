//go:build gpu

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

package concurrent

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/logutil"
	"github.com/rapidsai/cuvs/go"
	"go.uber.org/zap"
)

// CuvsTask represents a task to be executed by the CuvsWorker.
type CuvsTask struct {
	ID uint64
	Fn func(res *cuvs.Resource) (any, error)
}

// CuvsTaskResult holds the result of a CuvsTask execution.
type CuvsTaskResult struct {
	ID     uint64
	Result any
	Error  error
}

// CuvsTaskResultStore manages the storage and retrieval of CuvsTaskResults.
type CuvsTaskResultStore struct {
	results    map[uint64]*CuvsTaskResult
	resultCond *sync.Cond
	mu         sync.Mutex
	nextJobID  uint64
	stopCh     chan struct{} // New field
	stopped    atomic.Bool   // New field
}

// NewCuvsTaskResultStore creates a new CuvsTaskResultStore.
func NewCuvsTaskResultStore() *CuvsTaskResultStore {
	s := &CuvsTaskResultStore{
		results:   make(map[uint64]*CuvsTaskResult),
		nextJobID: 0,                   // Start job IDs from 0
		stopCh:    make(chan struct{}), // Initialize
		stopped:   atomic.Bool{},       // Initialize
	}
	s.resultCond = sync.NewCond(&s.mu)
	return s
}

// Store saves a CuvsTaskResult in the store and signals any waiting goroutines.
func (s *CuvsTaskResultStore) Store(result *CuvsTaskResult) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.results[result.ID] = result
	s.resultCond.Broadcast()
}

// Wait blocks until the result for the given jobID is available and returns it.
// The result is removed from the internal map after being retrieved.
func (s *CuvsTaskResultStore) Wait(jobID uint64) (*CuvsTaskResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for {
		if result, ok := s.results[jobID]; ok {
			delete(s.results, jobID) // Clean up the map
			return result, nil
		}
		// If the store is stopped and result is not found, return error
		if s.stopped.Load() {
			return nil, moerr.NewInternalErrorNoCtx("CuvsTaskResultStore stopped before result was available")
		}
		s.resultCond.Wait() // This will block and release the lock, then re-acquire
	}
}

// GetNextJobID atomically increments and returns a new unique job ID.
func (s *CuvsTaskResultStore) GetNextJobID() uint64 {
	return atomic.AddUint64(&s.nextJobID, 1)
}

// Stop signals the CuvsTaskResultStore to stop processing new waits.
func (s *CuvsTaskResultStore) Stop() {
	close(s.stopCh)
	s.stopped.Store(true)
	// Broadcast to unblock any waiting goroutines so they can check the stopped flag.
	s.resultCond.Broadcast()
}

// CuvsWorker runs tasks in a dedicated OS thread with a CUDA context.
type CuvsWorker struct {
	tasks                chan *CuvsTask
	stopCh               chan struct{}
	wg                   sync.WaitGroup
	stopped              atomic.Bool // Indicates if the worker has been stopped
	*CuvsTaskResultStore             // Embed the result store
	nthread              int
}

// NewCuvsWorker creates a new CuvsWorker.
func NewCuvsWorker(nthread int) *CuvsWorker {
	return &CuvsWorker{
		tasks:               make(chan *CuvsTask, nthread),
		stopCh:              make(chan struct{}),
		stopped:             atomic.Bool{}, // Initialize to false
		CuvsTaskResultStore: NewCuvsTaskResultStore(),
		nthread:             nthread,
	}
}

// Start begins the worker's execution loop.
func (w *CuvsWorker) Start(initFn func(res *cuvs.Resource) error) {
	w.wg.Add(1)
	go w.run(initFn)
}

// Stop signals the worker to terminate.
func (w *CuvsWorker) Stop() {
	if !w.stopped.Load() {
		w.stopped.Store(true)
		close(w.stopCh) // Signal run() to stop.
		close(w.tasks)  // Close tasks channel here.
	}
	w.wg.Wait()
	w.CuvsTaskResultStore.Stop() // Signal the result store to stop
}

// Submit sends a task to the worker.
func (w *CuvsWorker) Submit(fn func(res *cuvs.Resource) (any, error)) (uint64, error) {
	if w.stopped.Load() {
		return 0, moerr.NewInternalErrorNoCtx("cannot submit task: worker is stopped")
	}
	jobID := w.GetNextJobID()
	task := &CuvsTask{
		ID: jobID,
		Fn: fn,
	}
	w.tasks <- task
	return jobID, nil
}

func (w *CuvsWorker) workerLoop(wg *sync.WaitGroup) {
	defer wg.Done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Each worker gets its own stream and resource, which can be considered
	// a "child" resource that has access to the parent's context.
	stream, err := cuvs.NewCudaStream()
	if err != nil {
		logutil.Error("failed to create cuda stream in worker", zap.Error(err))
		return
	}
	defer stream.Close()

	resource, err := cuvs.NewResource(stream)
	if err != nil {
		logutil.Error("failed to create cuvs resource in worker", zap.Error(err))
		return
	}
	defer resource.Close()
	defer runtime.KeepAlive(resource)

	for {
		select {
		case task, ok := <-w.tasks:
			if !ok { // tasks channel closed
				return // No more tasks, and channel is closed. Exit.
			}
			result, err := task.Fn(&resource)
			cuvsResult := &CuvsTaskResult{
				ID:     task.ID,
				Result: result,
				Error:  err,
			}
			w.CuvsTaskResultStore.Store(cuvsResult)
		case <-w.stopCh:
			// stopCh signaled. Drain remaining tasks from w.tasks then exit.
			for {
				select {
				case task, ok := <-w.tasks:
					if !ok { // tasks channel closed during drain
						return // Channel closed, no more tasks. Exit.
					}
					result, err := task.Fn(&resource)
					cuvsResult := &CuvsTaskResult{
						ID:     task.ID,
						Result: result,
						Error:  err,
					}
					w.CuvsTaskResultStore.Store(cuvsResult)
				default:
					return // All tasks drained, or channel is empty.
				}
			}
		}
	}
}

func (w *CuvsWorker) run(initFn func(res *cuvs.Resource) error) {
	defer w.wg.Done()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Create a parent resource to run the one-time init function.
	// Data initialized here is available in the CUDA context for child resources.
	parentStream, err := cuvs.NewCudaStream()
	if err != nil {
		logutil.Fatal("failed to create parent cuda stream", zap.Error(err))
	}
	defer parentStream.Close()

	parentResource, err := cuvs.NewResource(parentStream)
	if err != nil {
		logutil.Fatal("failed to create parent cuvs resource", zap.Error(err))
	}
	defer parentResource.Close()

	// Execute initFn once.
	if initFn != nil {
		if err := initFn(&parentResource); err != nil {
			logutil.Fatal("failed to initialize cuvs resource with provided function", zap.Error(err))
		}
	}

	if w.nthread == 1 {
		// Special case: nthread is 1, process tasks directly in this goroutine
		for {
			select {
			case task, ok := <-w.tasks:
				if !ok { // tasks channel closed
					return // Channel closed, no more tasks. Exit.
				}
				result, err := task.Fn(&parentResource)
				cuvsResult := &CuvsTaskResult{
					ID:     task.ID,
					Result: result,
					Error:  err,
				}
				w.CuvsTaskResultStore.Store(cuvsResult)
			case <-w.stopCh:
				// Drain the tasks channel before exiting
				for {
					select {
					case task, ok := <-w.tasks:
						if !ok { // tasks channel closed during drain
							return
						}
						result, err := task.Fn(&parentResource)
						cuvsResult := &CuvsTaskResult{
							ID:     task.ID,
							Result: result,
							Error:  err,
						}
						w.CuvsTaskResultStore.Store(cuvsResult)
					default:
						return
					}
				}
			}
		}
	} else {
		// General case: nthread > 1, create worker goroutines
		var workerWg sync.WaitGroup
		workerWg.Add(w.nthread)
		for i := 0; i < w.nthread; i++ {
			go w.workerLoop(&workerWg)
		}

		// Wait for stop signal
		<-w.stopCh

		// Signal workers to stop and wait for them to finish.
		workerWg.Wait()
	}
}


// Wait blocks until the result for the given jobID is available and returns it.
// The result is removed from the internal map after being retrieved.
func (w *CuvsWorker) Wait(jobID uint64) (*CuvsTaskResult, error) {
	return w.CuvsTaskResultStore.Wait(jobID)
}
