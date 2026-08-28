// Copyright 2026 Matrix Origin
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

//go:build amd64 && go1.26 && goexperiment.simd

package metric

import (
	"math"
	"math/rand"
	"testing"

	"github.com/matrixorigin/matrixone/pkg/container/types"
	"github.com/stretchr/testify/require"
)

// scaleF8 lifts unit-norm embedding components (sigma ~ 1/sqrt(dim)) out of the
// E4M3 subnormal range, where spacing is absolute and relative error explodes.
// It is a fixed constant, not trained state, and it cancels out of the ranking:
// scaling both sides scales every distance by scale^2.
const scaleF8 = 32

func randF8Vectors(t testing.TB, dim int, seed int64) (a, b []types.Float8, fa, fb []float32) {
	r := rand.New(rand.NewSource(seed))
	sigma := 1.0 / math.Sqrt(float64(dim))
	a = make([]types.Float8, dim)
	b = make([]types.Float8, dim)
	fa = make([]float32, dim)
	fb = make([]float32, dim)
	for i := 0; i < dim; i++ {
		x := float32(r.NormFloat64() * sigma * scaleF8)
		y := float32(r.NormFloat64() * sigma * scaleF8)
		a[i] = types.Float8FromFloat32(x)
		b[i] = types.Float8FromFloat32(y)
		fa[i] = a[i].ToFloat32() // compare against what E4M3 actually stores
		fb[i] = b[i].ToFloat32()
	}
	return
}

// The SIMD kernel must agree with the scalar one, which decodes exactly.
func TestL2sqF8SIMDMatchesScalar(t *testing.T) {
	for _, dim := range []int{1, 3, 4, 7, 8, 31, 32, 33, 128, 768, 1000} {
		a, b, _, _ := randF8Vectors(t, dim, int64(dim))
		want, err := l2sqF8Scalar(a, b)
		require.NoError(t, err)
		got, err := l2sqF8AVX2(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, want, got, 1e-5, "dim=%d", dim)
	}
}

// And both must agree with an f32 computation over the decoded values: the only
// error is the encoding, not the arithmetic.
func TestL2sqF8MatchesFloat32OverDecodedValues(t *testing.T) {
	a, b, fa, fb := randF8Vectors(t, 768, 42)
	var want float64
	for i := range fa {
		d := float64(fa[i] - fb[i])
		want += d * d
	}
	scalar, err := l2sqF8Scalar(a, b)
	require.NoError(t, err)
	simd, err := l2sqF8AVX2(a, b)
	require.NoError(t, err)
	require.InEpsilon(t, want, scalar, 1e-5)
	require.InEpsilon(t, want, simd, 1e-5)
}

func TestL2sqF8RejectsDimensionMismatch(t *testing.T) {
	a, _, _, _ := randF8Vectors(t, 8, 1)
	b, _, _, _ := randF8Vectors(t, 9, 2)
	_, err := l2sqF8Scalar(a, b)
	require.Error(t, err)
	_, err = l2sqF8AVX2(a, b)
	require.Error(t, err)
}

// How much distance error does E4M3 encoding introduce, against exact f32?
func TestL2sqF8EncodingErrorVsFloat32(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	dim := 768
	sigma := 1.0 / math.Sqrt(float64(dim))
	worst := 0.0
	for trial := 0; trial < 200; trial++ {
		a8 := make([]types.Float8, dim)
		b8 := make([]types.Float8, dim)
		var exact float64
		for i := 0; i < dim; i++ {
			x := r.NormFloat64() * sigma
			y := r.NormFloat64() * sigma
			a8[i] = types.Float8FromFloat32(float32(x * scaleF8))
			b8[i] = types.Float8FromFloat32(float32(y * scaleF8))
			d := x - y
			exact += d * d
		}
		got, err := l2sqF8Scalar(a8, b8)
		require.NoError(t, err)
		got /= scaleF8 * scaleF8 // undo the fixed scale
		if rel := math.Abs(got-exact) / exact; rel > worst {
			worst = rel
		}
	}
	t.Logf("E4M3 (scale %d) squared-L2 error vs exact f32, dim=%d: worst %.3f%% over 200 trials",
		scaleF8, dim, worst*100)
	require.Less(t, worst, 0.05, "distance error should stay well under the per-component 6%%")
}

func BenchmarkL2sqF8Scalar(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.SetBytes(int64(2 * narrowBenchDim))
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l2sqF8Scalar(a, b)
	}
}

func BenchmarkL2sqF8AVX2(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.SetBytes(int64(2 * narrowBenchDim))
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l2sqF8AVX2(a, b)
	}
}

func TestL2sqF8LUTMatchesScalar(t *testing.T) {
	for _, dim := range []int{1, 7, 33, 768, 1024} {
		a, b, _, _ := randF8Vectors(t, dim, int64(dim))
		want, err := l2sqF8Scalar(a, b)
		require.NoError(t, err)
		got, err := l2sqF8LUT(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, want, got, 1e-6, "dim=%d", dim)
	}
}

func BenchmarkL2sqF8LUT(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.SetBytes(int64(2 * narrowBenchDim))
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l2sqF8LUT(a, b)
	}
}

func BenchmarkL2sqF8LUTFlat(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.SetBytes(int64(2 * narrowBenchDim))
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l2sqF8LUTFlat(a, b)
	}
}

func TestL2sqF8AVX2x2MatchesScalar(t *testing.T) {
	for _, dim := range []int{1, 7, 33, 64, 65, 768, 1024} {
		a, b, _, _ := randF8Vectors(t, dim, int64(dim))
		want, err := l2sqF8Scalar(a, b)
		require.NoError(t, err)
		got, err := l2sqF8AVX2x2(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, want, got, 1e-5, "dim=%d", dim)
	}
}

func BenchmarkL2sqF8AVX2x2(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.SetBytes(int64(2 * narrowBenchDim))
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l2sqF8AVX2x2(a, b)
	}
}

func TestL2sqF8AVX512MatchesScalar(t *testing.T) {
	for _, dim := range []int{1, 7, 33, 64, 65, 128, 768, 1024} {
		a, b, _, _ := randF8Vectors(t, dim, int64(dim))
		want, err := l2sqF8Scalar(a, b)
		require.NoError(t, err)
		got, err := l2sqF8AVX512(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, want, got, 1e-5, "dim=%d", dim)
	}
}

func BenchmarkL2sqF8AVX512(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.SetBytes(int64(2 * narrowBenchDim))
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l2sqF8AVX512(a, b)
	}
}

func TestL2sqF8AVX512x2MatchesScalar(t *testing.T) {
	for _, dim := range []int{1, 7, 33, 64, 65, 128, 129, 768, 1024} {
		a, b, _, _ := randF8Vectors(t, dim, int64(dim))
		want, err := l2sqF8Scalar(a, b)
		require.NoError(t, err)
		got, err := l2sqF8AVX512x2(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, want, got, 1e-5, "dim=%d", dim)
	}
}

func BenchmarkL2sqF8AVX512x2(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.SetBytes(int64(2 * narrowBenchDim))
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l2sqF8AVX512x2(a, b)
	}
}
