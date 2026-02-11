//go:build gpu

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

package brute_force

import (
	//	"fmt"

	"github.com/matrixorigin/matrixone/pkg/common/concurrent"
	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/vectorindex"
	"github.com/matrixorigin/matrixone/pkg/vectorindex/cache"
	"github.com/matrixorigin/matrixone/pkg/vectorindex/metric"
	"github.com/matrixorigin/matrixone/pkg/vectorindex/sqlexec"

	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/brute_force"
)

type GpuBruteForceIndex[T cuvs.TensorNumberType] struct {
	Dataset     *cuvs.Tensor[T]
	Index       *brute_force.BruteForceIndex
	Metric      cuvs.Distance
	Dimension   uint
	Count       uint
	ElementSize uint
	Worker      *concurrent.CuvsWorker
}

var _ cache.VectorIndexSearchIf = &GpuBruteForceIndex[float32]{}

func NewBruteForceIndex[T types.RealNumbers](dataset [][]T,
	dimension uint,
	m metric.MetricType,
	elemsz uint,
	nthread uint) (cache.VectorIndexSearchIf, error) {

	switch dset := any(dataset).(type) {
	case [][]float64:
		return NewCpuBruteForceIndex[T](dataset, dimension, m, elemsz)
	case [][]float32:
		return NewCpuBruteForceIndex[float32](dset, dimension, m, elemsz)
		//return NewGpuBruteForceIndex[float32](dset, dimension, m, elemsz, nthread)
	default:
		return nil, moerr.NewInternalErrorNoCtx("type not supported for BruteForceIndex")
	}

}

func NewGpuBruteForceIndex[T cuvs.TensorNumberType](dataset [][]T,
	dimension uint,
	m metric.MetricType,
	elemsz uint,
	nthread uint) (cache.VectorIndexSearchIf, error) {

	idx := &GpuBruteForceIndex[T]{}
	// Create CuvsWorker
	worker := concurrent.NewCuvsWorker(nthread) // Assuming 1 thread for now
	idx.Worker = worker                         // Only assign, don't start here

	tensor, err := cuvs.NewTensor(dataset)
	if err != nil {
		return nil, err
	}
	idx.Dataset = &tensor
	idx.Metric = metric.MetricTypeToCuvsMetric[m]
	idx.Dimension = dimension
	idx.Count = uint(len(dataset))

	idx.ElementSize = elemsz
	return idx, nil

}

func (idx *GpuBruteForceIndex[T]) Load(sqlproc *sqlexec.SqlProcess) (err error) {
	// Define initFn
	initFn := func(resource *cuvs.Resource) error {
		// Transfer dataset to device
		if _, err = idx.Dataset.ToDevice(resource); err != nil {
			return err
		}

		idx.Index, err = brute_force.CreateIndex()
		if err != nil {
			return err
		}

		err = brute_force.BuildIndex[T](*resource, idx.Dataset, idx.Metric, 0, idx.Index)
		if err != nil {
			return err
		}

		if err = resource.Sync(); err != nil {
			return err
		}
		return nil
	}

	// Define stopFn
	stopFn := func(resource *cuvs.Resource) error {
		if idx.Index != nil {
			idx.Index.Close()
			idx.Index = nil // Clear to prevent double close
		}
		if idx.Dataset != nil {
			idx.Dataset.Close()
			idx.Dataset = nil // Clear to prevent double close
		}
		return nil
	}

	// Start the worker with initFn and stopFn
	idx.Worker.Start(initFn, stopFn)

	return nil // No direct error from Load itself now, it's handled by initFn if any.
}

func (idx *GpuBruteForceIndex[T]) Search(proc *sqlexec.SqlProcess, _queries any, rt vectorindex.RuntimeConfig) (retkeys any, retdistances []float64, err error) {
	queriesvec, ok := _queries.([][]T)
	if !ok {
		return nil, nil, moerr.NewInternalErrorNoCtx("queries type invalid")
	}

	queries, err := cuvs.NewTensor(queriesvec)
	if err != nil {
		return nil, nil, err
	}
	defer queries.Close() // Close the host-side tensor

	// Submit the GPU operations as a task to the CuvsWorker
	jobID, err := idx.Worker.Submit(func(resource *cuvs.Resource) (any, error) {
		// All GPU operations using 'resource' provided by CuvsWorker
		neighbors, err := cuvs.NewTensorOnDevice[int64](resource, []int64{int64(len(queriesvec)), int64(rt.Limit)})
		if err != nil {
			return nil, err
		}
		defer neighbors.Close()

		distances, err := cuvs.NewTensorOnDevice[float32](resource, []int64{int64(len(queriesvec)), int64(rt.Limit)})
		if err != nil {
			return nil, err
		}
		defer distances.Close()

		if _, err = queries.ToDevice(resource); err != nil {
			return nil, err
		}

		err = brute_force.SearchIndex(*resource, idx.Index, &queries, &neighbors, &distances)
		if err != nil {
			return nil, err
		}

		if _, err = neighbors.ToHost(resource); err != nil {
			return nil, err
		}

		if _, err = distances.ToHost(resource); err != nil {
			return nil, err
		}

		if err = resource.Sync(); err != nil {
			return nil, err
		}

		// Collect results to pass back
		neighborsSlice, err := neighbors.Slice()
		if err != nil {
			return nil, err
		}

		distancesSlice, err := distances.Slice()
		if err != nil {
			return nil, err
		}

		// Return a custom struct or map to hold both slices
		return struct {
			Neighbors [][]int64
			Distances [][]float32
		}{
			Neighbors: neighborsSlice,
			Distances: distancesSlice,
		}, nil
	})
	if err != nil {
		return nil, nil, err
	}

	// Wait for the task to complete
	resultCuvsTask, err := idx.Worker.Wait(jobID)
	if err != nil {
		return nil, nil, err
	}
	if resultCuvsTask.Error != nil {
		return nil, nil, resultCuvsTask.Error
	}

	// Unpack the result
	res := resultCuvsTask.Result.(struct {
		Neighbors [][]int64
		Distances [][]float32
	})
	neighborsSlice := res.Neighbors
	distancesSlice := res.Distances

	retdistances = make([]float64, len(distancesSlice)*int(rt.Limit))
	for i := range distancesSlice {
		for j, dist := range distancesSlice[i] {
			retdistances[i*int(rt.Limit)+j] = float64(dist)
		}
	}

	keys := make([]int64, len(neighborsSlice)*int(rt.Limit))
	for i := range neighborsSlice {
		for j, key := range neighborsSlice[i] {
			keys[i*int(rt.Limit)+j] = int64(key)
		}
	}
	retkeys = keys
	return
}

func (idx *GpuBruteForceIndex[T]) UpdateConfig(sif cache.VectorIndexSearchIf) error {
	return nil
}

func (idx *GpuBruteForceIndex[T]) Destroy() {
	if idx.Worker != nil {
		idx.Worker.Stop() // This will trigger the stopFn
	}
}
