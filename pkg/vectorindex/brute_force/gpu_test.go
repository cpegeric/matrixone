//go:build gpu

package brute_force

import (
	//"fmt"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/vectorindex"
	"github.com/matrixorigin/matrixone/pkg/vectorindex/metric"
	"github.com/stretchr/testify/require"
)

func TestGpuBruteForce(t *testing.T) {

	dimension := uint(128)
	ncpu := uint(1)
	limit := uint(1)
	elemsz := uint(4) // float32

	dataset := make([][]float32, 400)
	for i := range dataset {
		dataset[i] = make([]float32, dimension)
		for j := 0; j < int(dimension); j++ {
			dataset[i][j] = rand.Float32()
		}
	}

	query := make([][]float32, 8192)
	for i := range query {
		query[i] = make([]float32, dimension)
		for j := 0; j < int(dimension); j++ {
			query[i][j] = rand.Float32()
		}
	}

	idx, err := NewBruteForceIndex[float32](dataset, dimension, metric.Metric_L2sqDistance, elemsz)
	require.NoError(t, err)
	defer idx.Destroy()

	err = idx.Load(nil)
	require.NoError(t, err)

	rt := vectorindex.RuntimeConfig{Limit: limit, NThreads: ncpu}

	var wg sync.WaitGroup

	for n := 0; n < 4; n++ {

		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				keys, distances, err := idx.Search(nil, query, rt)
				require.NoError(t, err)

				var _ = keys
				var _ = distances
				/*
					keys_i64, ok := keys.([]int64)
					require.Equal(t, ok, true)

					for j, key := range keys_i64 {
						require.Equal(t, key, int64(j))
						require.Equal(t, distances[j], float64(0))
					}
				*/
				// fmt.Printf("keys %v, dist %v\n", keys, distances)
			}
		}()
	}

	wg.Wait()

}
