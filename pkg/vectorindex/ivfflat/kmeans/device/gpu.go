//go:build gpu

// Copyright 2023 Matrix Origin
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

package device

import (
	//"os"

	"context"
	"runtime"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/matrixorigin/matrixone/pkg/vectorindex/ivfflat/kmeans"
	"github.com/matrixorigin/matrixone/pkg/vectorindex/ivfflat/kmeans/elkans"
	"github.com/matrixorigin/matrixone/pkg/vectorindex/metric"
	"github.com/matrixorigin/matrixone/pkg/common/concurrent"
	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/ivf_flat"
)

type GpuClusterer[T cuvs.TensorNumberType] struct {
	indexParams *ivf_flat.IndexParams
	nlist       int
	dim         int
	vectors     [][]T
	worker      *concurrent.CuvsWorker
}

func (c *GpuClusterer[T]) InitCentroids(ctx context.Context) error {
	return nil
}

func (c *GpuClusterer[T]) Cluster(ctx context.Context) (any, error) {
	jobID, err := c.worker.Submit(func(resource *cuvs.Resource) (any, error) {
		dataset, err := cuvs.NewTensor(c.vectors)
		if err != nil {
			return nil, err
		}
		defer dataset.Close()

		index, err := ivf_flat.CreateIndex[T](c.indexParams)
		if err != nil {
			return nil, err
		}
		defer index.Close()

		if _, err := dataset.ToDevice(resource); err != nil {
			return nil, err
		}

		centers, err := cuvs.NewTensorOnDevice[T](resource, []int64{int64(c.nlist), int64(c.dim)})
		if err != nil {
			return nil, err
		}
		defer centers.Close()

		if err := ivf_flat.BuildIndex(*resource, c.indexParams, &dataset, index); err != nil {
			return nil, err
		}

		if err := resource.Sync(); err != nil {
			return nil, err
		}

		if err := ivf_flat.GetCenters(index, &centers); err != nil {
			return nil, err
		}

		if _, err := centers.ToHost(resource); err != nil {
			return nil, err
		}

		if err := resource.Sync(); err != nil {
			return nil, err
		}

		result, err := centers.Slice()
		if err != nil {
			return nil, err
		}

		runtime.KeepAlive(index)
		runtime.KeepAlive(dataset)
		runtime.KeepAlive(centers)
		runtime.KeepAlive(c)
		return result, nil
	})
	if err != nil {
		return nil, err
	}
	result, err := c.worker.Wait(jobID)
	if err != nil {
		return nil, err
	}
	if result.Error != nil {
		return nil, result.Error
	}
	return result.Result, nil
}

func (c *GpuClusterer[T]) SSE() (float64, error) {
	return 0, nil
}

func (c *GpuClusterer[T]) Close() error {
	if c.indexParams != nil {
		c.indexParams.Close()
	}
	if c.worker != nil {
		c.worker.Stop()
	}
	return nil
}

func resolveCuvsDistanceForDense(distance metric.MetricType) cuvs.Distance {
	switch distance {
	case metric.Metric_L2sqDistance:
		return cuvs.DistanceL2
	case metric.Metric_L2Distance:
		return cuvs.DistanceL2
	case metric.Metric_InnerProduct:
		return cuvs.DistanceL2
	case metric.Metric_CosineDistance:
		return cuvs.DistanceL2
	case metric.Metric_L1Distance:
		return cuvs.DistanceL2
	default:
		return cuvs.DistanceL2
	}
}

func NewKMeans[T types.RealNumbers](vectors [][]T, clusterCnt,
	maxIterations int, deltaThreshold float64,
	distanceType metric.MetricType, initType kmeans.InitType,
	spherical bool,
	nworker int) (kmeans.Clusterer, error) {

	switch vecs := any(vectors).(type) {
	case [][]float32:

		c := &GpuClusterer[float32]{}
		c.nlist = clusterCnt
		if len(vectors) == 0 {
			return nil, moerr.NewInternalErrorNoCtx("empty dataset")
		}
		c.vectors = vecs
		c.dim = len(vecs[0])

		// GPU - nworker is 1
		c.worker = concurrent.NewCuvsWorker(uint(1))

		indexParams, err := ivf_flat.CreateIndexParams()
		if err != nil {
			return nil, err
		}
		indexParams.SetNLists(uint32(clusterCnt))
		indexParams.SetMetric(resolveCuvsDistanceForDense(distanceType))
		indexParams.SetKMeansNIters(uint32(maxIterations))
		indexParams.SetKMeansTrainsetFraction(1) // train all sample
		c.indexParams = indexParams
		c.worker.Start(nil, nil)
		return c, nil
	default:
		return elkans.NewKMeans(vectors, clusterCnt, maxIterations, deltaThreshold, distanceType, initType, spherical, nworker)

	}
}
