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
	"simd/archsimd"
	"unsafe"

	"github.com/matrixorigin/matrixone/pkg/common/moerr"
	"github.com/matrixorigin/matrixone/pkg/container/types"
)

// E4M3 -> float32 without branches, the same magic-multiply trick the f16 kernels
// use: place the exponent+mantissa where float32 expects them, then multiply by
// 2^(127-7) to rebias. Subnormals renormalise for free in that multiply.
//
// NaN is not patched. The only E4M3 NaN encoding is S.1111.111, which decodes
// here to +-480 instead. Entry values come from a saturating cast that cannot
// produce NaN from a finite input, and the scalar kernels below handle it
// exactly; this fast path trades that corner for two fewer ops per lane.
var f8Magic = math.Float32frombits(uint32(127+120) << 23) // 2^120

func f8fast(b types.Float8) float32 {
	o := uint32(b&0x7f) << 20
	of := math.Float32frombits(o) * f8Magic
	return math.Float32frombits(math.Float32bits(of) | uint32(b&0x80)<<24)
}

// f8AsU32 reinterprets a byte slice as uint32 lanes, four E4M3 values per lane.
func f8AsU32(s []types.Float8) []uint32 {
	if len(s) < 4 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&s[0])), len(s)/4)
}

// f8decX8 decodes the byte at position k of every lane.
func f8decX8(u archsimd.Uint32x8, k uint8, mFF, m7f, m80 archsimd.Uint32x8, magic archsimd.Float32x8) archsimd.Float32x8 {
	// No byte mask: 0x7f and 0x80 are both subsets of 0xff, so masking the byte
	// out first is dead work -- the two field masks below already isolate it.
	b := u.ShiftAllRight(uint64(8 * k))
	of := b.And(m7f).ShiftAllLeft(20).AsFloat32x8().Mul(magic)
	return of.AsUint32x8().Or(b.And(m80).ShiftAllLeft(24)).AsFloat32x8()
}

func f8DecodeConstsX8() (mFF, m7f, m80 archsimd.Uint32x8, magic archsimd.Float32x8) {
	mFF = archsimd.BroadcastUint32x8(0xff)
	m7f = archsimd.BroadcastUint32x8(0x7f)
	m80 = archsimd.BroadcastUint32x8(0x80)
	magic = archsimd.BroadcastFloat32x8(f8Magic)
	return
}

// l2sqF8AVX2 consumes 32 E4M3 values per iteration: 8 uint32 lanes x 4 bytes.
func l2sqF8AVX2(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX8()
	acc0, acc1 := archsimd.Float32x8{}, archsimd.Float32x8{}
	acc2, acc3 := archsimd.Float32x8{}, archsimd.Float32x8{}
	np, j := len(au), 0
	for ; j <= np-8; j += 8 {
		ua := archsimd.LoadUint32x8Slice(au[j : j+8])
		ub := archsimd.LoadUint32x8Slice(bu[j : j+8])
		d0 := f8decX8(ua, 0, mFF, m7f, m80, magic).Sub(f8decX8(ub, 0, mFF, m7f, m80, magic))
		d1 := f8decX8(ua, 1, mFF, m7f, m80, magic).Sub(f8decX8(ub, 1, mFF, m7f, m80, magic))
		d2 := f8decX8(ua, 2, mFF, m7f, m80, magic).Sub(f8decX8(ub, 2, mFF, m7f, m80, magic))
		d3 := f8decX8(ua, 3, mFF, m7f, m80, magic).Sub(f8decX8(ub, 3, mFF, m7f, m80, magic))
		acc0 = d0.MulAdd(d0, acc0)
		acc1 = d1.MulAdd(d1, acc1)
		acc2 = d2.MulAdd(d2, acc2)
		acc3 = d3.MulAdd(d3, acc3)
	}
	sum := sumF32x8(acc0.Add(acc1).Add(acc2.Add(acc3)))
	for i := j * 4; i < n; i++ {
		d := f8fast(a[i]) - f8fast(b[i])
		sum += d * d
	}
	return float64(sum), nil
}

// l2sqF8AVX2x2 unrolls the SIMD loop twice, 64 E4M3 values per iteration. Each
// load feeds eight independent decode chains, so the single-load version leaves
// the ALU waiting on its own dependencies; a second load doubles the work in
// flight without needing more registers than the accumulators already use.
func l2sqF8AVX2x2(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX8()
	a0, a1, a2, a3 := archsimd.Float32x8{}, archsimd.Float32x8{}, archsimd.Float32x8{}, archsimd.Float32x8{}
	b0, b1, b2, b3 := archsimd.Float32x8{}, archsimd.Float32x8{}, archsimd.Float32x8{}, archsimd.Float32x8{}
	np, j := len(au), 0
	for ; j <= np-16; j += 16 {
		ua0 := archsimd.LoadUint32x8Slice(au[j : j+8])
		ub0 := archsimd.LoadUint32x8Slice(bu[j : j+8])
		ua1 := archsimd.LoadUint32x8Slice(au[j+8 : j+16])
		ub1 := archsimd.LoadUint32x8Slice(bu[j+8 : j+16])

		d0 := f8decX8(ua0, 0, mFF, m7f, m80, magic).Sub(f8decX8(ub0, 0, mFF, m7f, m80, magic))
		d1 := f8decX8(ua0, 1, mFF, m7f, m80, magic).Sub(f8decX8(ub0, 1, mFF, m7f, m80, magic))
		d2 := f8decX8(ua0, 2, mFF, m7f, m80, magic).Sub(f8decX8(ub0, 2, mFF, m7f, m80, magic))
		d3 := f8decX8(ua0, 3, mFF, m7f, m80, magic).Sub(f8decX8(ub0, 3, mFF, m7f, m80, magic))
		e0 := f8decX8(ua1, 0, mFF, m7f, m80, magic).Sub(f8decX8(ub1, 0, mFF, m7f, m80, magic))
		e1 := f8decX8(ua1, 1, mFF, m7f, m80, magic).Sub(f8decX8(ub1, 1, mFF, m7f, m80, magic))
		e2 := f8decX8(ua1, 2, mFF, m7f, m80, magic).Sub(f8decX8(ub1, 2, mFF, m7f, m80, magic))
		e3 := f8decX8(ua1, 3, mFF, m7f, m80, magic).Sub(f8decX8(ub1, 3, mFF, m7f, m80, magic))

		a0 = d0.MulAdd(d0, a0)
		a1 = d1.MulAdd(d1, a1)
		a2 = d2.MulAdd(d2, a2)
		a3 = d3.MulAdd(d3, a3)
		b0 = e0.MulAdd(e0, b0)
		b1 = e1.MulAdd(e1, b1)
		b2 = e2.MulAdd(e2, b2)
		b3 = e3.MulAdd(e3, b3)
	}
	for ; j <= np-8; j += 8 {
		ua := archsimd.LoadUint32x8Slice(au[j : j+8])
		ub := archsimd.LoadUint32x8Slice(bu[j : j+8])
		d0 := f8decX8(ua, 0, mFF, m7f, m80, magic).Sub(f8decX8(ub, 0, mFF, m7f, m80, magic))
		d1 := f8decX8(ua, 1, mFF, m7f, m80, magic).Sub(f8decX8(ub, 1, mFF, m7f, m80, magic))
		d2 := f8decX8(ua, 2, mFF, m7f, m80, magic).Sub(f8decX8(ub, 2, mFF, m7f, m80, magic))
		d3 := f8decX8(ua, 3, mFF, m7f, m80, magic).Sub(f8decX8(ub, 3, mFF, m7f, m80, magic))
		a0 = d0.MulAdd(d0, a0)
		a1 = d1.MulAdd(d1, a1)
		a2 = d2.MulAdd(d2, a2)
		a3 = d3.MulAdd(d3, a3)
	}
	sum := sumF32x8(a0.Add(a1).Add(a2.Add(a3)).Add(b0.Add(b1).Add(b2.Add(b3))))
	for i := j * 4; i < n; i++ {
		d := f8fast(a[i]) - f8fast(b[i])
		sum += d * d
	}
	return float64(sum), nil
}

// ---------------------------------------------------------------------------
// AVX-512: 16 uint32 lanes per load, so 64 E4M3 values per iteration.

func f8decX16(u archsimd.Uint32x16, k uint8, mFF, m7f, m80 archsimd.Uint32x16, magic archsimd.Float32x16) archsimd.Float32x16 {
	b := u.ShiftAllRight(uint64(8 * k))
	of := b.And(m7f).ShiftAllLeft(20).AsFloat32x16().Mul(magic)
	return of.AsUint32x16().Or(b.And(m80).ShiftAllLeft(24)).AsFloat32x16()
}

func f8DecodeConstsX16() (mFF, m7f, m80 archsimd.Uint32x16, magic archsimd.Float32x16) {
	mFF = archsimd.BroadcastUint32x16(0xff)
	m7f = archsimd.BroadcastUint32x16(0x7f)
	m80 = archsimd.BroadcastUint32x16(0x80)
	magic = archsimd.BroadcastFloat32x16(f8Magic)
	return
}

func l2sqF8AVX512(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX16()
	acc0, acc1 := archsimd.Float32x16{}, archsimd.Float32x16{}
	acc2, acc3 := archsimd.Float32x16{}, archsimd.Float32x16{}
	np, j := len(au), 0
	for ; j <= np-16; j += 16 {
		ua := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub := archsimd.LoadUint32x16Slice(bu[j : j+16])
		d0 := f8decX16(ua, 0, mFF, m7f, m80, magic).Sub(f8decX16(ub, 0, mFF, m7f, m80, magic))
		d1 := f8decX16(ua, 1, mFF, m7f, m80, magic).Sub(f8decX16(ub, 1, mFF, m7f, m80, magic))
		d2 := f8decX16(ua, 2, mFF, m7f, m80, magic).Sub(f8decX16(ub, 2, mFF, m7f, m80, magic))
		d3 := f8decX16(ua, 3, mFF, m7f, m80, magic).Sub(f8decX16(ub, 3, mFF, m7f, m80, magic))
		acc0 = d0.MulAdd(d0, acc0)
		acc1 = d1.MulAdd(d1, acc1)
		acc2 = d2.MulAdd(d2, acc2)
		acc3 = d3.MulAdd(d3, acc3)
	}
	sum := sumF32x16(acc0.Add(acc1).Add(acc2.Add(acc3)))
	for i := j * 4; i < n; i++ {
		d := f8fast(a[i]) - f8fast(b[i])
		sum += d * d
	}
	return float64(sum), nil
}

// l2sqF8AVX512x2 combines both levers: 512-bit lanes and a twice-unrolled loop,
// 128 E4M3 values per iteration. This is the fastest arrangement the archsimd
// surface allows for this format -- there is no widening load and no FP8
// conversion instruction, so every value still costs a shift, two masks, a
// multiply and an or.
func l2sqF8AVX512x2(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX16()
	a0, a1, a2, a3 := archsimd.Float32x16{}, archsimd.Float32x16{}, archsimd.Float32x16{}, archsimd.Float32x16{}
	b0, b1, b2, b3 := archsimd.Float32x16{}, archsimd.Float32x16{}, archsimd.Float32x16{}, archsimd.Float32x16{}
	np, j := len(au), 0
	for ; j <= np-32; j += 32 {
		ua0 := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub0 := archsimd.LoadUint32x16Slice(bu[j : j+16])
		ua1 := archsimd.LoadUint32x16Slice(au[j+16 : j+32])
		ub1 := archsimd.LoadUint32x16Slice(bu[j+16 : j+32])

		d0 := f8decX16(ua0, 0, mFF, m7f, m80, magic).Sub(f8decX16(ub0, 0, mFF, m7f, m80, magic))
		d1 := f8decX16(ua0, 1, mFF, m7f, m80, magic).Sub(f8decX16(ub0, 1, mFF, m7f, m80, magic))
		d2 := f8decX16(ua0, 2, mFF, m7f, m80, magic).Sub(f8decX16(ub0, 2, mFF, m7f, m80, magic))
		d3 := f8decX16(ua0, 3, mFF, m7f, m80, magic).Sub(f8decX16(ub0, 3, mFF, m7f, m80, magic))
		e0 := f8decX16(ua1, 0, mFF, m7f, m80, magic).Sub(f8decX16(ub1, 0, mFF, m7f, m80, magic))
		e1 := f8decX16(ua1, 1, mFF, m7f, m80, magic).Sub(f8decX16(ub1, 1, mFF, m7f, m80, magic))
		e2 := f8decX16(ua1, 2, mFF, m7f, m80, magic).Sub(f8decX16(ub1, 2, mFF, m7f, m80, magic))
		e3 := f8decX16(ua1, 3, mFF, m7f, m80, magic).Sub(f8decX16(ub1, 3, mFF, m7f, m80, magic))

		a0 = d0.MulAdd(d0, a0)
		a1 = d1.MulAdd(d1, a1)
		a2 = d2.MulAdd(d2, a2)
		a3 = d3.MulAdd(d3, a3)
		b0 = e0.MulAdd(e0, b0)
		b1 = e1.MulAdd(e1, b1)
		b2 = e2.MulAdd(e2, b2)
		b3 = e3.MulAdd(e3, b3)
	}
	for ; j <= np-16; j += 16 {
		ua := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub := archsimd.LoadUint32x16Slice(bu[j : j+16])
		d0 := f8decX16(ua, 0, mFF, m7f, m80, magic).Sub(f8decX16(ub, 0, mFF, m7f, m80, magic))
		d1 := f8decX16(ua, 1, mFF, m7f, m80, magic).Sub(f8decX16(ub, 1, mFF, m7f, m80, magic))
		d2 := f8decX16(ua, 2, mFF, m7f, m80, magic).Sub(f8decX16(ub, 2, mFF, m7f, m80, magic))
		d3 := f8decX16(ua, 3, mFF, m7f, m80, magic).Sub(f8decX16(ub, 3, mFF, m7f, m80, magic))
		a0 = d0.MulAdd(d0, a0)
		a1 = d1.MulAdd(d1, a1)
		a2 = d2.MulAdd(d2, a2)
		a3 = d3.MulAdd(d3, a3)
	}
	sum := sumF32x16(a0.Add(a1).Add(a2.Add(a3)).Add(b0.Add(b1).Add(b2.Add(b3))))
	for i := j * 4; i < n; i++ {
		d := f8fast(a[i]) - f8fast(b[i])
		sum += d * d
	}
	return float64(sum), nil
}

// ---------------------------------------------------------------------------
// The remaining metrics, all on the AVX-512 shape that won for L2. Each decodes
// four byte positions per lane and keeps one accumulator per position, so the
// decode cost per element is identical across metrics and the comparison against
// the other formats is apples to apples.

func innerProductF8AVX512(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX16()
	acc0, acc1 := archsimd.Float32x16{}, archsimd.Float32x16{}
	acc2, acc3 := archsimd.Float32x16{}, archsimd.Float32x16{}
	np, j := len(au), 0
	for ; j <= np-16; j += 16 {
		ua := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub := archsimd.LoadUint32x16Slice(bu[j : j+16])
		acc0 = f8decX16(ua, 0, mFF, m7f, m80, magic).MulAdd(f8decX16(ub, 0, mFF, m7f, m80, magic), acc0)
		acc1 = f8decX16(ua, 1, mFF, m7f, m80, magic).MulAdd(f8decX16(ub, 1, mFF, m7f, m80, magic), acc1)
		acc2 = f8decX16(ua, 2, mFF, m7f, m80, magic).MulAdd(f8decX16(ub, 2, mFF, m7f, m80, magic), acc2)
		acc3 = f8decX16(ua, 3, mFF, m7f, m80, magic).MulAdd(f8decX16(ub, 3, mFF, m7f, m80, magic), acc3)
	}
	sum := sumF32x16(acc0.Add(acc1).Add(acc2.Add(acc3)))
	for i := j * 4; i < n; i++ {
		sum += f8fast(a[i]) * f8fast(b[i])
	}
	return float64(sum), nil
}

func l1DistanceF8AVX512(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX16()
	absMask := archsimd.BroadcastUint32x16(0x7fffffff)
	acc0, acc1 := archsimd.Float32x16{}, archsimd.Float32x16{}
	acc2, acc3 := archsimd.Float32x16{}, archsimd.Float32x16{}
	abs := func(v archsimd.Float32x16) archsimd.Float32x16 {
		return v.AsUint32x16().And(absMask).AsFloat32x16()
	}
	np, j := len(au), 0
	for ; j <= np-16; j += 16 {
		ua := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub := archsimd.LoadUint32x16Slice(bu[j : j+16])
		acc0 = acc0.Add(abs(f8decX16(ua, 0, mFF, m7f, m80, magic).Sub(f8decX16(ub, 0, mFF, m7f, m80, magic))))
		acc1 = acc1.Add(abs(f8decX16(ua, 1, mFF, m7f, m80, magic).Sub(f8decX16(ub, 1, mFF, m7f, m80, magic))))
		acc2 = acc2.Add(abs(f8decX16(ua, 2, mFF, m7f, m80, magic).Sub(f8decX16(ub, 2, mFF, m7f, m80, magic))))
		acc3 = acc3.Add(abs(f8decX16(ua, 3, mFF, m7f, m80, magic).Sub(f8decX16(ub, 3, mFF, m7f, m80, magic))))
	}
	sum := sumF32x16(acc0.Add(acc1).Add(acc2.Add(acc3)))
	for i := j * 4; i < n; i++ {
		d := f8fast(a[i]) - f8fast(b[i])
		if d < 0 {
			d = -d
		}
		sum += d
	}
	return float64(sum), nil
}

// cosineDistanceF8AVX512 accumulates the dot product and both norms in one pass,
// so each element is decoded once rather than three times.
//
// Each quantity gets one accumulator per byte position rather than a single
// shared one. Sharing would make every accumulator a four-deep dependency chain
// per iteration -- three chains of FMAs all waiting on themselves -- which is the
// one thing the other metrics here avoid by construction.
func cosineDistanceF8AVX512(a, b []types.Float8) (float64, error) {
	if len(a) == 0 {
		return 0, nil
	}
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX16()
	var d0, d1, d2, d3 archsimd.Float32x16
	var na0, na1, na2, na3 archsimd.Float32x16
	var nb0, nb1, nb2, nb3 archsimd.Float32x16
	np, j := len(au), 0
	for ; j <= np-16; j += 16 {
		ua := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub := archsimd.LoadUint32x16Slice(bu[j : j+16])

		av0 := f8decX16(ua, 0, mFF, m7f, m80, magic)
		bv0 := f8decX16(ub, 0, mFF, m7f, m80, magic)
		av1 := f8decX16(ua, 1, mFF, m7f, m80, magic)
		bv1 := f8decX16(ub, 1, mFF, m7f, m80, magic)
		av2 := f8decX16(ua, 2, mFF, m7f, m80, magic)
		bv2 := f8decX16(ub, 2, mFF, m7f, m80, magic)
		av3 := f8decX16(ua, 3, mFF, m7f, m80, magic)
		bv3 := f8decX16(ub, 3, mFF, m7f, m80, magic)

		d0 = av0.MulAdd(bv0, d0)
		d1 = av1.MulAdd(bv1, d1)
		d2 = av2.MulAdd(bv2, d2)
		d3 = av3.MulAdd(bv3, d3)
		na0 = av0.MulAdd(av0, na0)
		na1 = av1.MulAdd(av1, na1)
		na2 = av2.MulAdd(av2, na2)
		na3 = av3.MulAdd(av3, na3)
		nb0 = bv0.MulAdd(bv0, nb0)
		nb1 = bv1.MulAdd(bv1, nb1)
		nb2 = bv2.MulAdd(bv2, nb2)
		nb3 = bv3.MulAdd(bv3, nb3)
	}
	sdot := sumF32x16(d0.Add(d1).Add(d2.Add(d3)))
	sna := sumF32x16(na0.Add(na1).Add(na2.Add(na3)))
	snb := sumF32x16(nb0.Add(nb1).Add(nb2.Add(nb3)))
	for i := j * 4; i < n; i++ {
		av, bv := f8fast(a[i]), f8fast(b[i])
		sdot += av * bv
		sna += av * av
		snb += bv * bv
	}
	if sna == 0 || snb == 0 {
		return 0, moerr.NewInternalErrorNoCtx("cosine distance with zero-norm vector")
	}
	return 1 - float64(sdot)/(math.Sqrt(float64(sna))*math.Sqrt(float64(snb))), nil
}

// innerProductF8AVX512x2 and l1DistanceF8AVX512x2 unroll the load twice, matching
// the arrangement that was fastest for L2: two loads per iteration feeding eight
// independent accumulator chains, so the decode's own latency is overlapped
// rather than exposed.

func innerProductF8AVX512x2(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX16()
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Float32x16
	np, j := len(au), 0
	for ; j <= np-32; j += 32 {
		ua0 := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub0 := archsimd.LoadUint32x16Slice(bu[j : j+16])
		ua1 := archsimd.LoadUint32x16Slice(au[j+16 : j+32])
		ub1 := archsimd.LoadUint32x16Slice(bu[j+16 : j+32])
		a0 = f8decX16(ua0, 0, mFF, m7f, m80, magic).MulAdd(f8decX16(ub0, 0, mFF, m7f, m80, magic), a0)
		a1 = f8decX16(ua0, 1, mFF, m7f, m80, magic).MulAdd(f8decX16(ub0, 1, mFF, m7f, m80, magic), a1)
		a2 = f8decX16(ua0, 2, mFF, m7f, m80, magic).MulAdd(f8decX16(ub0, 2, mFF, m7f, m80, magic), a2)
		a3 = f8decX16(ua0, 3, mFF, m7f, m80, magic).MulAdd(f8decX16(ub0, 3, mFF, m7f, m80, magic), a3)
		b0 = f8decX16(ua1, 0, mFF, m7f, m80, magic).MulAdd(f8decX16(ub1, 0, mFF, m7f, m80, magic), b0)
		b1 = f8decX16(ua1, 1, mFF, m7f, m80, magic).MulAdd(f8decX16(ub1, 1, mFF, m7f, m80, magic), b1)
		b2 = f8decX16(ua1, 2, mFF, m7f, m80, magic).MulAdd(f8decX16(ub1, 2, mFF, m7f, m80, magic), b2)
		b3 = f8decX16(ua1, 3, mFF, m7f, m80, magic).MulAdd(f8decX16(ub1, 3, mFF, m7f, m80, magic), b3)
	}
	for ; j <= np-16; j += 16 {
		ua := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub := archsimd.LoadUint32x16Slice(bu[j : j+16])
		a0 = f8decX16(ua, 0, mFF, m7f, m80, magic).MulAdd(f8decX16(ub, 0, mFF, m7f, m80, magic), a0)
		a1 = f8decX16(ua, 1, mFF, m7f, m80, magic).MulAdd(f8decX16(ub, 1, mFF, m7f, m80, magic), a1)
		a2 = f8decX16(ua, 2, mFF, m7f, m80, magic).MulAdd(f8decX16(ub, 2, mFF, m7f, m80, magic), a2)
		a3 = f8decX16(ua, 3, mFF, m7f, m80, magic).MulAdd(f8decX16(ub, 3, mFF, m7f, m80, magic), a3)
	}
	sum := sumF32x16(a0.Add(a1).Add(a2.Add(a3)).Add(b0.Add(b1).Add(b2.Add(b3))))
	for i := j * 4; i < n; i++ {
		sum += f8fast(a[i]) * f8fast(b[i])
	}
	return float64(sum), nil
}

func l1DistanceF8AVX512x2(a, b []types.Float8) (float64, error) {
	if len(a) != len(b) {
		return 0, moerr.NewInternalErrorNoCtx("vector dimension not matched")
	}
	n := len(a)
	au, bu := f8AsU32(a), f8AsU32(b)
	mFF, m7f, m80, magic := f8DecodeConstsX16()
	absMask := archsimd.BroadcastUint32x16(0x7fffffff)
	abs := func(v archsimd.Float32x16) archsimd.Float32x16 {
		return v.AsUint32x16().And(absMask).AsFloat32x16()
	}
	dec := func(u, v archsimd.Uint32x16, k uint8) archsimd.Float32x16 {
		return abs(f8decX16(u, k, mFF, m7f, m80, magic).Sub(f8decX16(v, k, mFF, m7f, m80, magic)))
	}
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Float32x16
	np, j := len(au), 0
	for ; j <= np-32; j += 32 {
		ua0 := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub0 := archsimd.LoadUint32x16Slice(bu[j : j+16])
		ua1 := archsimd.LoadUint32x16Slice(au[j+16 : j+32])
		ub1 := archsimd.LoadUint32x16Slice(bu[j+16 : j+32])
		a0 = a0.Add(dec(ua0, ub0, 0))
		a1 = a1.Add(dec(ua0, ub0, 1))
		a2 = a2.Add(dec(ua0, ub0, 2))
		a3 = a3.Add(dec(ua0, ub0, 3))
		b0 = b0.Add(dec(ua1, ub1, 0))
		b1 = b1.Add(dec(ua1, ub1, 1))
		b2 = b2.Add(dec(ua1, ub1, 2))
		b3 = b3.Add(dec(ua1, ub1, 3))
	}
	for ; j <= np-16; j += 16 {
		ua := archsimd.LoadUint32x16Slice(au[j : j+16])
		ub := archsimd.LoadUint32x16Slice(bu[j : j+16])
		a0 = a0.Add(dec(ua, ub, 0))
		a1 = a1.Add(dec(ua, ub, 1))
		a2 = a2.Add(dec(ua, ub, 2))
		a3 = a3.Add(dec(ua, ub, 3))
	}
	sum := sumF32x16(a0.Add(a1).Add(a2.Add(a3)).Add(b0.Add(b1).Add(b2.Add(b3))))
	for i := j * 4; i < n; i++ {
		d := f8fast(a[i]) - f8fast(b[i])
		if d < 0 {
			d = -d
		}
		sum += d
	}
	return float64(sum), nil
}
