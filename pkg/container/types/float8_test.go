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

package types

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

// E4M3 has 256 encodings, so correctness can be established exhaustively rather
// than sampled: every non-NaN encoding must survive widening and re-narrowing.
func TestFloat8RoundTripsEveryEncoding(t *testing.T) {
	for i := 0; i < 256; i++ {
		b := uint8(i)
		if b&0x7f == f8NaNBits { // the single NaN encoding, both signs
			require.True(t, math.IsNaN(float64(Float8(b).ToFloat32())), "0x%02x should be NaN", b)
			continue
		}
		f := Float8(b).ToFloat32()
		require.False(t, math.IsInf(float64(f), 0), "0x%02x decoded to Inf; E4M3 has none", b)
		back := Float8FromFloat32(f)
		require.Equal(t, b, uint8(back), "0x%02x -> %v -> 0x%02x", b, f, uint8(back))
	}
}

func TestFloat8Boundaries(t *testing.T) {
	require.Equal(t, float32(448), Float8(f8MaxBits).ToFloat32(), "max finite")
	require.Equal(t, float32(0.001953125), Float8(0x01).ToFloat32(), "min subnormal 2^-9")
	require.Equal(t, float32(0.015625), Float8(0x08).ToFloat32(), "min normal 2^-6")
	require.Equal(t, float32(1), Float8(0x38).ToFloat32(), "1.0")

	// Saturation, not overflow to infinity.
	require.Equal(t, f8MaxBits, uint8(Float8FromFloat32(1e30)), "large saturates")
	require.Equal(t, uint8(0x80)|f8MaxBits, uint8(Float8FromFloat32(-1e30)), "negative saturates")
	require.Equal(t, f8MaxBits, uint8(Float8FromFloat32(float32(math.Inf(1)))), "+Inf saturates")

	// Signed zero and underflow.
	require.Equal(t, uint8(0x00), uint8(Float8FromFloat32(0)))
	require.Equal(t, uint8(0x80), uint8(Float8FromFloat32(float32(math.Copysign(0, -1)))))
	require.Equal(t, uint8(0x00), uint8(Float8FromFloat32(1e-12)), "underflows to zero")
}

// Ties must go to the even mantissa, not away from zero.
func TestFloat8RoundsTiesToEven(t *testing.T) {
	// Between 1.0 (mant 000) and 1.125 (mant 001) the midpoint is 1.0625.
	require.Equal(t, uint8(0x38), uint8(Float8FromFloat32(1.0625)), "tie rounds down to even mantissa 000")
	// Between 1.125 (001) and 1.25 (010) the midpoint is 1.1875 -> even is 010.
	require.Equal(t, uint8(0x3a), uint8(Float8FromFloat32(1.1875)), "tie rounds up to even mantissa 010")
	// Clear non-ties either side.
	require.Equal(t, uint8(0x38), uint8(Float8FromFloat32(1.05)))
	require.Equal(t, uint8(0x39), uint8(Float8FromFloat32(1.10)))
}

// E4M3 splits into two precision regimes, and the boundary matters for embeddings.
//
//   - Normal (|x| >= 2^-6 = 0.015625): relative spacing, error within a half-ulp
//     of 3 mantissa bits, i.e. 6.25%.
//   - Subnormal (2^-9 <= |x| < 2^-6): ABSOLUTE spacing of 2^-9, so relative error
//     grows without bound as values approach zero -- up to 50% at the low end.
//
// A unit-norm 768-dimension embedding has components with sigma ~ 1/sqrt(768) ~
// 0.036, which puts roughly a third of them in the subnormal regime. That is a
// property of the format, not a defect, but it is the reason FP8 embeddings are
// scaled before encoding rather than stored raw.
func TestFloat8PrecisionRegimes(t *testing.T) {
	normalWorst := 0.0
	for x := 0.015625; x < 400; x *= 1.0009 {
		got := float64(Float8FromFloat32(float32(x)).ToFloat32())
		if rel := math.Abs(got-x) / x; rel > normalWorst {
			normalWorst = rel
		}
	}
	require.Less(t, normalWorst, 0.0625,
		"normal-range relative error %.4f exceeds half-ulp of 3 mantissa bits", normalWorst)
	t.Logf("normal    [2^-6, 400): worst relative error %.2f%%", normalWorst*100)

	subWorst := 0.0
	for x := 0.001953125; x < 0.015625; x *= 1.0009 {
		got := float64(Float8FromFloat32(float32(x)).ToFloat32())
		if rel := math.Abs(got-x) / x; rel > subWorst {
			subWorst = rel
		}
	}
	t.Logf("subnormal [2^-9, 2^-6): worst relative error %.2f%%", subWorst*100)
	require.Greater(t, subWorst, 0.0625,
		"subnormals are expected to be coarser than normals; if not, the encoder is wrong")
}

// Scaling a vector into the normal range before encoding removes the subnormal
// penalty entirely. The scale is a fixed constant, not trained state, so it costs
// none of the quantizer machinery that motivated FP8 in the first place.
func TestFloat8ScalingLiftsEmbeddingsOutOfSubnormals(t *testing.T) {
	// Components of a unit-norm 768-dim embedding, +-3 sigma around 1/sqrt(768).
	sigma := 1.0 / math.Sqrt(768)
	measure := func(scale float64) (worst float64, subnormal int, total int) {
		for k := -300; k <= 300; k++ {
			x := sigma * float64(k) / 100
			if math.Abs(x) < 1e-9 {
				continue
			}
			total++
			if math.Abs(x*scale) < 0.015625 {
				subnormal++
			}
			got := float64(Float8FromFloat32(float32(x*scale)).ToFloat32()) / scale
			if rel := math.Abs(got-x) / math.Abs(x); rel > worst {
				worst = rel
			}
		}
		return
	}
	rawWorst, rawSub, total := measure(1)
	sclWorst, sclSub, _ := measure(32)
	t.Logf("scale  1: %d/%d components subnormal, worst relative error %.1f%%", rawSub, total, rawWorst*100)
	t.Logf("scale 32: %d/%d components subnormal, worst relative error %.1f%%", sclSub, total, sclWorst*100)
	require.Less(t, sclWorst, rawWorst, "scaling into the normal range must reduce worst-case error")
}
