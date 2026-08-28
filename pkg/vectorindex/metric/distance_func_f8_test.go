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
	"fmt"
	"math"
	"math/rand"
	"sort"
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

// Every metric is checked against a scalar computation over the decoded values:
// the only error is the encoding, never the arithmetic or the lane handling.
func TestF8AllMetricsMatchScalar(t *testing.T) {
	for _, dim := range []int{1, 7, 33, 63, 64, 65, 768, 1024} {
		a, b, fa, fb := randF8Vectors(t, dim, int64(dim))

		var wantIP, wantL1, wantDot, wantNA, wantNB float64
		for i := range fa {
			wantIP += float64(fa[i]) * float64(fb[i])
			wantL1 += math.Abs(float64(fa[i]) - float64(fb[i]))
			wantDot += float64(fa[i]) * float64(fb[i])
			wantNA += float64(fa[i]) * float64(fa[i])
			wantNB += float64(fb[i]) * float64(fb[i])
		}

		gotIP, err := innerProductF8AVX512(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, wantIP, gotIP, 1e-4, "innerProduct dim=%d", dim)

		gotL1, err := l1DistanceF8AVX512(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, wantL1, gotL1, 1e-4, "l1 dim=%d", dim)

		gotCos, err := cosineDistanceF8AVX512(a, b)
		require.NoError(t, err)
		wantCos := 1 - wantDot/(math.Sqrt(wantNA)*math.Sqrt(wantNB))
		require.InDelta(t, wantCos, gotCos, 1e-4, "cosine dim=%d", dim)
	}
}

func TestF8MetricsRejectDimensionMismatch(t *testing.T) {
	a, _, _, _ := randF8Vectors(t, 8, 1)
	b, _, _, _ := randF8Vectors(t, 9, 2)
	_, err := innerProductF8AVX512(a, b)
	require.Error(t, err)
	_, err = l1DistanceF8AVX512(a, b)
	require.Error(t, err)
	_, err = cosineDistanceF8AVX512(a, b)
	require.Error(t, err)
}

// A zero vector has no direction, so cosine must report rather than divide by zero.
func TestF8CosineRejectsZeroNorm(t *testing.T) {
	zero := make([]types.Float8, 64)
	a, _, _, _ := randF8Vectors(t, 64, 3)
	_, err := cosineDistanceF8AVX512(a, zero)
	require.Error(t, err)

	got, err := cosineDistanceF8AVX512(nil, nil)
	require.NoError(t, err, "empty input is defined as 0, matching the other formats")
	require.Equal(t, float64(0), got)
}

func BenchmarkInnerProductF8AVX512(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = innerProductF8AVX512(a, b)
	}
}

func BenchmarkL1DistanceF8AVX512(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l1DistanceF8AVX512(a, b)
	}
}

func BenchmarkCosineDistanceF8AVX512(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = cosineDistanceF8AVX512(a, b)
	}
}

func TestF8UnrolledMetricsMatchScalar(t *testing.T) {
	for _, dim := range []int{1, 7, 33, 64, 65, 127, 128, 129, 768, 1024} {
		a, b, fa, fb := randF8Vectors(t, dim, int64(dim))
		var wantIP, wantL1 float64
		for i := range fa {
			wantIP += float64(fa[i]) * float64(fb[i])
			wantL1 += math.Abs(float64(fa[i]) - float64(fb[i]))
		}
		gotIP, err := innerProductF8AVX512x2(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, wantIP, gotIP, 1e-4, "innerProduct x2 dim=%d", dim)
		gotL1, err := l1DistanceF8AVX512x2(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, wantL1, gotL1, 1e-4, "l1 x2 dim=%d", dim)
	}
}

func BenchmarkInnerProductF8AVX512x2(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = innerProductF8AVX512x2(a, b)
	}
}

func BenchmarkL1DistanceF8AVX512x2(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l1DistanceF8AVX512x2(a, b)
	}
}

func TestF8LUTMetricsMatchSIMD(t *testing.T) {
	for _, dim := range []int{1, 7, 33, 64, 65, 768, 1024} {
		a, b, _, _ := randF8Vectors(t, dim, int64(dim))

		wantIP, err := innerProductF8AVX512x2(a, b)
		require.NoError(t, err)
		gotIP, err := innerProductF8LUT(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, wantIP, gotIP, 1e-4, "innerProduct dim=%d", dim)

		wantL1, err := l1DistanceF8AVX512x2(a, b)
		require.NoError(t, err)
		gotL1, err := l1DistanceF8LUT(a, b)
		require.NoError(t, err)
		require.InEpsilon(t, wantL1, gotL1, 1e-4, "l1 dim=%d", dim)

		wantCos, err := cosineDistanceF8AVX512(a, b)
		require.NoError(t, err)
		gotCos, err := cosineDistanceF8LUT(a, b)
		require.NoError(t, err)
		require.InDelta(t, wantCos, gotCos, 1e-4, "cosine dim=%d", dim)
	}
}

func BenchmarkInnerProductF8LUT(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = innerProductF8LUT(a, b)
	}
}

func BenchmarkL1DistanceF8LUT(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = l1DistanceF8LUT(a, b)
	}
}

func BenchmarkCosineDistanceF8LUT(bch *testing.B) {
	a, b, _, _ := randF8Vectors(bch, narrowBenchDim, 1)
	bch.ResetTimer()
	for i := 0; i < bch.N; i++ {
		_, _ = cosineDistanceF8LUT(a, b)
	}
}

// Tier choice is dimension-dependent for the existing narrow types -- AVX2 beats
// AVX512 below roughly dim 512 -- so the f8 tiers are swept rather than assumed
// from a single width.
func BenchmarkL2sqF8ByDim(bch *testing.B) {
	for _, dim := range []int{128, 256, 384, 512, 768, 1024} {
		a, b, _, _ := randF8Vectors(bch, dim, int64(dim))
		bch.Run(fmt.Sprintf("dim%d/AVX2x2", dim), func(bch *testing.B) {
			for i := 0; i < bch.N; i++ {
				_, _ = l2sqF8AVX2x2(a, b)
			}
		})
		bch.Run(fmt.Sprintf("dim%d/AVX512x2", dim), func(bch *testing.B) {
			for i := 0; i < bch.N; i++ {
				_, _ = l2sqF8AVX512x2(a, b)
			}
		})
		bch.Run(fmt.Sprintf("dim%d/LUT", dim), func(bch *testing.B) {
			for i := 0; i < bch.N; i++ {
				_, _ = l2sqF8LUT(a, b)
			}
		})
	}
}

// Score fidelity against exact float32, per metric, plus the thing that actually
// decides search quality: whether the RANKING survives. A distance can be several
// percent off and cost nothing if the order is unchanged.
func TestF8AccuracyVsFloat32(t *testing.T) {
	const (
		dim        = 768
		candidates = 500
		queries    = 50
		topK       = 10
	)
	r := rand.New(rand.NewSource(11))
	sigma := 1.0 / math.Sqrt(float64(dim))

	// Exact f32 vectors, and their E4M3 encodings under the fixed scale.
	mk := func() ([]float32, []types.Float8) {
		f := make([]float32, dim)
		q := make([]types.Float8, dim)
		for i := range f {
			f[i] = float32(r.NormFloat64() * sigma)
			q[i] = types.Float8FromFloat32(f[i] * scaleF8)
		}
		return f, q
	}
	baseF := make([][]float32, candidates)
	base8 := make([][]types.Float8, candidates)
	for i := range baseF {
		baseF[i], base8[i] = mk()
	}

	exactL2 := func(a, b []float32) float64 {
		var s float64
		for i := range a {
			d := float64(a[i]) - float64(b[i])
			s += d * d
		}
		return s
	}

	var worstL2, worstIP, worstCos, typicalIP float64
	hits, total := 0, 0

	for qi := 0; qi < queries; qi++ {
		qF, q8 := mk()

		type scored struct {
			id   int
			dist float64
		}
		exact := make([]scored, candidates)
		approx := make([]scored, candidates)
		for c := 0; c < candidates; c++ {
			e := exactL2(qF, baseF[c])
			a, err := l2sqF8AVX512x2(q8, base8[c])
			require.NoError(t, err)
			a /= scaleF8 * scaleF8
			exact[c] = scored{c, e}
			approx[c] = scored{c, a}
			if e > 0 {
				if rel := math.Abs(a-e) / e; rel > worstL2 {
					worstL2 = rel
				}
			}
		}

		// Metric-level fidelity on the first candidate of each query.
		var eIP, eNA, eNB float64
		for i := range qF {
			eIP += float64(qF[i]) * float64(baseF[0][i])
			eNA += float64(qF[i]) * float64(qF[i])
			eNB += float64(baseF[0][i]) * float64(baseF[0][i])
		}
		aIP, err := innerProductF8AVX512x2(q8, base8[0])
		require.NoError(t, err)
		aIP /= scaleF8 * scaleF8
		// Absolute, not relative: the inner product of two independent random
		// vectors is near zero, so a relative figure divides by ~0 and reports
		// hundreds of percent for an error that is numerically tiny.
		if d := math.Abs(aIP - eIP); d > worstIP {
			worstIP = d
		}
		if m := math.Abs(eIP); m > typicalIP {
			typicalIP = m
		}
		eCos := 1 - eIP/(math.Sqrt(eNA)*math.Sqrt(eNB))
		aCos, err := cosineDistanceF8AVX512(q8, base8[0])
		require.NoError(t, err)
		if d := math.Abs(aCos - eCos); d > worstCos {
			worstCos = d
		}

		sort.Slice(exact, func(i, j int) bool { return exact[i].dist < exact[j].dist })
		sort.Slice(approx, func(i, j int) bool { return approx[i].dist < approx[j].dist })
		truth := map[int]bool{}
		for i := 0; i < topK; i++ {
			truth[exact[i].id] = true
		}
		for i := 0; i < topK; i++ {
			if truth[approx[i].id] {
				hits++
			}
			total++
		}
	}

	recall := float64(hits) / float64(total)
	t.Logf("dim=%d, %d queries x %d candidates", dim, queries, candidates)
	t.Logf("  L2sq     worst relative error %.3f%%", worstL2*100)
	t.Logf("  innerProd worst ABSOLUTE error %.5f (largest |exact| seen %.5f)", worstIP, typicalIP)
	t.Logf("  cosine   worst absolute error %.5f  (scale is [0,2])", worstCos)
	t.Logf("  rank agreement         recall@%d = %.4f", topK, recall)
	require.Greater(t, recall, 0.90, "E4M3 must not reorder the top-K materially")
}

// E4M3 versus int8 at the same one byte per element, on the same data and the
// same harness. The two spend their byte differently: int8 spreads 256 uniform
// levels across a trained [min,max], so its step is absolute; E4M3 spends bits on
// an exponent, so its step is relative and gets finer near zero.
//
// Which wins depends entirely on how tight the trained bounds are, and that is
// exactly what the quantizer buys -- and what makes it fragile when the data
// drifts outside the range it was trained on.
func TestF8VsInt8Accuracy(t *testing.T) {
	const (
		dim        = 768
		candidates = 500
		queries    = 50
		topK       = 10
	)
	r := rand.New(rand.NewSource(11))
	sigma := 1.0 / math.Sqrt(float64(dim))

	// Train int8 bounds on the data, the way the real quantizer does.
	sample := make([]float32, 0, dim*64)
	for i := 0; i < dim*64; i++ {
		sample = append(sample, float32(r.NormFloat64()*sigma))
	}
	lo, hi := sample[0], sample[0]
	for _, v := range sample {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	mul := 255.0 / float64(hi-lo)
	add := -float64(lo)*mul - 128
	encInt8 := func(v float32) int8 {
		q := math.Round(float64(v)*mul + add)
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		return int8(q)
	}
	decInt8 := func(q int8) float64 { return (float64(q) - add) / mul }

	mk := func() ([]float32, []types.Float8, []int8) {
		f := make([]float32, dim)
		q8 := make([]types.Float8, dim)
		qi := make([]int8, dim)
		for i := range f {
			f[i] = float32(r.NormFloat64() * sigma)
			q8[i] = types.Float8FromFloat32(f[i] * scaleF8)
			qi[i] = encInt8(f[i])
		}
		return f, q8, qi
	}
	baseF := make([][]float32, candidates)
	base8 := make([][]types.Float8, candidates)
	baseI := make([][]int8, candidates)
	for i := range baseF {
		baseF[i], base8[i], baseI[i] = mk()
	}

	exactL2 := func(a, b []float32) float64 {
		var s float64
		for i := range a {
			d := float64(a[i]) - float64(b[i])
			s += d * d
		}
		return s
	}
	int8L2 := func(a, b []int8) float64 {
		var s float64
		for i := range a {
			d := decInt8(a[i]) - decInt8(b[i])
			s += d * d
		}
		return s
	}

	var worst8, worstI float64
	hits8, hitsI, total := 0, 0, 0

	for qi := 0; qi < queries; qi++ {
		qF, q8, qI := mk()
		type scored struct {
			id   int
			dist float64
		}
		exact := make([]scored, candidates)
		app8 := make([]scored, candidates)
		appI := make([]scored, candidates)
		for c := 0; c < candidates; c++ {
			e := exactL2(qF, baseF[c])
			a8, err := l2sqF8AVX512x2(q8, base8[c])
			require.NoError(t, err)
			a8 /= scaleF8 * scaleF8
			aI := int8L2(qI, baseI[c])
			exact[c] = scored{c, e}
			app8[c] = scored{c, a8}
			appI[c] = scored{c, aI}
			if e > 0 {
				if d := math.Abs(a8-e) / e; d > worst8 {
					worst8 = d
				}
				if d := math.Abs(aI-e) / e; d > worstI {
					worstI = d
				}
			}
		}
		sort.Slice(exact, func(i, j int) bool { return exact[i].dist < exact[j].dist })
		sort.Slice(app8, func(i, j int) bool { return app8[i].dist < app8[j].dist })
		sort.Slice(appI, func(i, j int) bool { return appI[i].dist < appI[j].dist })
		truth := map[int]bool{}
		for i := 0; i < topK; i++ {
			truth[exact[i].id] = true
		}
		for i := 0; i < topK; i++ {
			if truth[app8[i].id] {
				hits8++
			}
			if truth[appI[i].id] {
				hitsI++
			}
			total++
		}
	}

	t.Logf("dim=%d, %d queries x %d candidates, both at 1 byte/element", dim, queries, candidates)
	t.Logf("  int8 trained bounds [%.4f, %.4f] -> step %.6f", lo, hi, 1/mul)
	t.Logf("  E4M3  worst L2 relative error %.3f%%   recall@%d %.4f", worst8*100, topK, float64(hits8)/float64(total))
	t.Logf("  int8  worst L2 relative error %.3f%%   recall@%d %.4f", worstI*100, topK, float64(hitsI)/float64(total))
}
