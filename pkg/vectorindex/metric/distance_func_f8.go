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

package metric

import (
	"math"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/container/types"
)

// f8LUT decodes every E4M3 encoding once at init. The format has 256 values, so
// the table is 1 KiB and stays resident in L1 -- a single indexed load replaces
// the shift/mask/multiply chain that bit-twiddling decode needs per element.
var f8LUT = func() (t [256]float32) {
	for i := range t {
		t[i] = types.Float8(i).ToFloat32()
	}
	return
}()

// l2sqF8LUTFlat is the same table lookup with no unrolling, so the two isolate
// what unrolling contributes on top of the decode method.
func l2sqF8LUTFlat(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	var s float32
	for i := range a {
		d := f8LUT[a[i]] - f8LUT[b[i]]
		s += d * d
	}
	return float64(s), nil
}

// l2sqF8LUT trades arithmetic for an L1 table lookup, unrolled by four.
func l2sqF8LUT(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	var s0, s1, s2, s3 float32
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		d0 := f8LUT[a[i]] - f8LUT[b[i]]
		d1 := f8LUT[a[i+1]] - f8LUT[b[i+1]]
		d2 := f8LUT[a[i+2]] - f8LUT[b[i+2]]
		d3 := f8LUT[a[i+3]] - f8LUT[b[i+3]]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < n; i++ {
		d := f8LUT[a[i]] - f8LUT[b[i]]
		s0 += d * d
	}
	return float64((s0 + s1) + (s2 + s3)), nil
}

// l2sqF8Scalar is the portable kernel, unrolled by four to give the scheduler
// independent accumulator chains. It decodes exactly, NaN included, so it is also
// the oracle the SIMD path is tested against.
func l2sqF8Scalar(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	var s0, s1, s2, s3 float32
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		d0 := a[i].ToFloat32() - b[i].ToFloat32()
		d1 := a[i+1].ToFloat32() - b[i+1].ToFloat32()
		d2 := a[i+2].ToFloat32() - b[i+2].ToFloat32()
		d3 := a[i+3].ToFloat32() - b[i+3].ToFloat32()
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < n; i++ {
		d := a[i].ToFloat32() - b[i].ToFloat32()
		s0 += d * d
	}
	return float64((s0 + s1) + (s2 + s3)), nil
}

// The remaining metrics via the lookup table, unrolled by four to match the L2
// variant. These exist to answer whether the table beats in-register decoding
// once the metric does more arithmetic per element: the table's cost is fixed at
// two L1 loads regardless of metric, while the SIMD decode is paid per lane and
// then competes with the metric's own FMAs for issue slots.

func innerProductF8LUT(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	var s0, s1, s2, s3 float32
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		s0 += f8LUT[a[i]] * f8LUT[b[i]]
		s1 += f8LUT[a[i+1]] * f8LUT[b[i+1]]
		s2 += f8LUT[a[i+2]] * f8LUT[b[i+2]]
		s3 += f8LUT[a[i+3]] * f8LUT[b[i+3]]
	}
	for ; i < n; i++ {
		s0 += f8LUT[a[i]] * f8LUT[b[i]]
	}
	return float64((s0 + s1) + (s2 + s3)), nil
}

func l1DistanceF8LUT(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	abs := func(v float32) float32 {
		if v < 0 {
			return -v
		}
		return v
	}
	var s0, s1, s2, s3 float32
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		s0 += abs(f8LUT[a[i]] - f8LUT[b[i]])
		s1 += abs(f8LUT[a[i+1]] - f8LUT[b[i+1]])
		s2 += abs(f8LUT[a[i+2]] - f8LUT[b[i+2]])
		s3 += abs(f8LUT[a[i+3]] - f8LUT[b[i+3]])
	}
	for ; i < n; i++ {
		s0 += abs(f8LUT[a[i]] - f8LUT[b[i]])
	}
	return float64((s0 + s1) + (s2 + s3)), nil
}

func cosineDistanceF8LUT(a, b []types.Float8) (float64, error) {
	if len(a) == 0 {
		return 0, nil
	}
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	var d0, d1, na0, na1, nb0, nb1 float32
	n := len(a)
	i := 0
	for ; i <= n-2; i += 2 {
		av0, bv0 := f8LUT[a[i]], f8LUT[b[i]]
		av1, bv1 := f8LUT[a[i+1]], f8LUT[b[i+1]]
		d0 += av0 * bv0
		d1 += av1 * bv1
		na0 += av0 * av0
		na1 += av1 * av1
		nb0 += bv0 * bv0
		nb1 += bv1 * bv1
	}
	for ; i < n; i++ {
		av, bv := f8LUT[a[i]], f8LUT[b[i]]
		d0 += av * bv
		na0 += av * av
		nb0 += bv * bv
	}
	sdot, sna, snb := d0+d1, na0+na1, nb0+nb1
	if sna == 0 || snb == 0 {
		return 0, moerr.NewInternalErrorNoCtx("cosine distance with zero-norm vector")
	}
	return 1 - float64(sdot)/(math.Sqrt(float64(sna))*math.Sqrt(float64(snb))), nil
}
