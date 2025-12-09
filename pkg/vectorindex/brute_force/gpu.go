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
	"fmt"
	"os"

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
	Dataset     [][]T
	Metric      cuvs.Distance
	Dimension   uint
	Count       uint
	ElementSize uint
}

var _ cache.VectorIndexSearchIf = &GpuBruteForceIndex[float32]{}

func NewBruteForceIndex[T types.RealNumbers](dataset [][]T,
	dimension uint,
	m metric.MetricType,
	elemsz uint) (cache.VectorIndexSearchIf, error) {

	// return NewCpuBruteForceIndex[T](dataset, dimension, m, elemsz)

	switch dset := any(dataset).(type) {
	case [][]float64:
		return NewCpuBruteForceIndex[T](dataset, dimension, m, elemsz)
	case [][]float32:
		os.Stderr.WriteString(fmt.Sprintf("center set %d\n", len(dset)))
		idx := &GpuBruteForceIndex[float32]{}
		idx.Dataset = dset
		ok := false
		idx.Metric, ok = metric.MetricTypeToCuvsMetric[m]
		if !ok {
			return nil, moerr.NewInternalErrorNoCtx("metric not supported for BruteForceIndex")
		}
		idx.Dimension = dimension
		idx.Count = uint(len(dset))
		idx.ElementSize = elemsz
		return idx, nil
	default:
		return nil, moerr.NewInternalErrorNoCtx("type not supported for BruteForceIndex")
	}

}

func (idx *GpuBruteForceIndex[T]) Load(sqlproc *sqlexec.SqlProcess) (err error) {
	os.Stderr.WriteString("load brute force index\n")
	return
}

func (idx *GpuBruteForceIndex[T]) Search(proc *sqlexec.SqlProcess, _queries any, rt vectorindex.RuntimeConfig) (retkeys any, retdistances []float64, err error) {
	queriesvec, ok := _queries.([][]T)
	if !ok {
		return nil, nil, moerr.NewInternalErrorNoCtx("queries type invalid")
	}

	os.Stderr.WriteString(fmt.Sprintf("probe set %d\n", len(queriesvec)))
	os.Stderr.WriteString("brute force index search start\n")

	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return
	}
	defer resource.Close()

	dataset, err := cuvs.NewTensor(idx.Dataset)
	if err != nil {
		return
	}
	defer dataset.Close()

	index, err := brute_force.CreateIndex()
	if err != nil {
		return
	}
	defer index.Close()

	queries, err := cuvs.NewTensor(queriesvec)
	if err != nil {
		return
	}
	defer queries.Close()

	neighbors, err := cuvs.NewTensorOnDevice[int64](&resource, []int64{int64(len(queriesvec)), int64(rt.Limit)})
	if err != nil {
		return
	}
	defer neighbors.Close()

	distances, err := cuvs.NewTensorOnDevice[float32](&resource, []int64{int64(len(queriesvec)), int64(rt.Limit)})
	if err != nil {
		return
	}
	defer distances.Close()

	if _, err = dataset.ToDevice(&resource); err != nil {
		return
	}

	if err = resource.Sync(); err != nil {
		return
	}

	err = brute_force.BuildIndex(resource, &dataset, idx.Metric, 2.0, index)
	if err != nil {
		os.Stderr.WriteString(fmt.Sprintf("BruteForceIndex: build index failed %v\n", err))
		os.Stderr.WriteString(fmt.Sprintf("BruteForceIndex: build index failed centers %v\n", idx.Dataset))
		return
	}

	if err = resource.Sync(); err != nil {
		return
	}
	os.Stderr.WriteString("built brute force index\n")

	if _, err = queries.ToDevice(&resource); err != nil {
		return
	}

	os.Stderr.WriteString("brute force index search Runing....\n")
	err = brute_force.SearchIndex(resource, *index, &queries, &neighbors, &distances)
	if err != nil {
		return
	}
	os.Stderr.WriteString("brute force index search finished Runing....\n")

	if _, err = neighbors.ToHost(&resource); err != nil {
		return
	}
	os.Stderr.WriteString("brute force index search neighbour to host done....\n")

	if _, err = distances.ToHost(&resource); err != nil {
		return
	}
	os.Stderr.WriteString("brute force index search distances to host done....\n")

	if err = resource.Sync(); err != nil {
		return
	}

	os.Stderr.WriteString("brute force index search return result....\n")
	neighborsSlice, err := neighbors.Slice()
	if err != nil {
		return
	}

	distancesSlice, err := distances.Slice()
	if err != nil {
		return
	}

	//fmt.Printf("flattened %v\n", flatten)
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
	os.Stderr.WriteString("brute force index search RETURN NOW....\n")
	return
}

func (idx *GpuBruteForceIndex[T]) UpdateConfig(sif cache.VectorIndexSearchIf) error {
	return nil
}

func (idx *GpuBruteForceIndex[T]) Destroy() {
	os.Stderr.WriteString("destroy brute fore index START\n")
	os.Stderr.WriteString("destroy brute fore END\n")
}
