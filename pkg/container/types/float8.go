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

import "math"

// Float8 is the OCP FP8 E4M3 format: 1 sign bit, 4 exponent bits (bias 7), and
// 3 mantissa bits.
//
// It deliberately departs from IEEE 754 in two ways, both part of the E4M3
// specification:
//
//   - There is no infinity. The maximum exponent field is a normal exponent, so
//     the largest finite magnitude is 1.75 * 2^8 = 448.
//   - NaN has a single encoding, S.1111.111. Converting from float32 therefore
//     SATURATES rather than overflowing to infinity: anything above 448 becomes
//     448, which keeps a narrowing cast total.
//
// Representable magnitudes run from 2^-9 (smallest subnormal, ~0.00195) to 448.
// For a unit-norm 768-dimension embedding, components sit near 1/sqrt(768) ~
// 0.036 -- comfortably inside that window at both ends.
type Float8 uint8

const (
	// F8E4M3Max is the largest finite magnitude, 1.75 * 2^8.
	F8E4M3Max = float32(448)
	// f8MaxBits encodes +448: exponent field 1111, mantissa 110.
	f8MaxBits = uint8(0x7e)
	// f8NaNBits is the single NaN encoding, S.1111.111.
	f8NaNBits = uint8(0x7f)
)

// ToFloat32 widens an E4M3 value. Every E4M3 value is exactly representable in
// float32, so this direction is lossless.
func (f Float8) ToFloat32() float32 {
	return math.Float32frombits(f8bitsToF32bits(uint8(f)))
}

// Float8FromFloat32 narrows to E4M3 with round-to-nearest-even, saturating at
// +-448.
func Float8FromFloat32(f float32) Float8 {
	return Float8(f32bitsToF8bits(math.Float32bits(f)))
}

func f8bitsToF32bits(in uint8) uint32 {
	sign := uint32(in&0x80) << 24
	exp := uint32(in&0x78) >> 3
	coef := uint32(in & 0x07)

	if exp == 0x0f && coef == 0x07 {
		return sign | 0x7fc00000 // the one NaN encoding
	}

	if exp == 0 {
		if coef == 0 {
			return sign // signed zero
		}
		// Subnormal: renormalise into a float32 normal.
		exp++
		coef <<= 20
		for coef&0x00800000 == 0 {
			coef <<= 1
			exp--
		}
		coef &= 0x007fffff
		return sign | ((exp + (127 - 7)) << 23) | coef
	}

	return sign | ((exp + (127 - 7)) << 23) | (coef << 20)
}

// roundToNearestEven rounds v down to a multiple of 2^lsb, breaking ties toward
// the even multiple.
func roundToNearestEven(v uint32, lsb uint32) uint32 {
	if lsb == 0 {
		return v
	}
	mask := (uint32(1) << lsb) - 1
	rem := v & mask
	v &^= mask
	half := uint32(1) << (lsb - 1)
	if rem > half || (rem == half && (v>>lsb)&1 == 1) {
		v += uint32(1) << lsb
	}
	return v
}

func f32bitsToF8bits(u32 uint32) uint8 {
	sign := uint8((u32 >> 24) & 0x80)
	biased := (u32 >> 23) & 0xff
	mant := u32 & 0x007fffff

	if biased == 0xff {
		if mant != 0 {
			return sign | f8NaNBits
		}
		return sign | f8MaxBits // no infinity: saturate
	}
	if u32&0x7fffffff == 0 {
		return sign // signed zero
	}

	e8 := int32(biased) - 127 + 7 // exponent field for E4M3

	if e8 <= 0 {
		// Subnormal territory: value is m * 2^-9 for m in [0,7].
		full := mant | 0x00800000 // restore the implicit leading 1
		lsb := uint32(20 + (1 - e8))
		if lsb >= 32 {
			return sign // underflows to zero
		}
		m := roundToNearestEven(full, lsb) >> lsb
		if m >= 8 {
			return sign | (1 << 3) // rounded up into the smallest normal
		}
		return sign | uint8(m)
	}

	m := roundToNearestEven(mant, 20) >> 20
	if m >= 8 { // mantissa carried into the exponent
		m = 0
		e8++
	}
	if e8 > 15 || (e8 == 15 && m == 7) {
		// e8 == 15 is a normal exponent in E4M3, but mantissa 111 there is the
		// NaN encoding, so the largest finite value is 1.75 * 2^8.
		return sign | f8MaxBits
	}
	return sign | uint8(e8<<3) | uint8(m)
}

// Float8ToFloat32Slice widens a slice; the reverse of Float32ToFloat8Slice.
func Float8ToFloat32Slice(src []Float8) []float32 {
	out := make([]float32, len(src))
	for i := range src {
		out[i] = src[i].ToFloat32()
	}
	return out
}

// Float32ToFloat8Slice narrows a slice with round-to-nearest-even and saturation.
func Float32ToFloat8Slice(src []float32) []Float8 {
	out := make([]Float8, len(src))
	for i := range src {
		out[i] = Float8FromFloat32(src[i])
	}
	return out
}
