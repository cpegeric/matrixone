# FP8 (E4M3) Vector Column Type for IVFFLAT — Design Spec, Measured and Rejected

This document proposed `vecf8`, an 8-bit floating-point vector column type, and
its use as an IVFFLAT entry format. It was measured and rejected; see §0 for the
result and the recommendation that replaces it. The motivating property is not size — `vecint8`
already occupies the one-byte-per-element class — but that FP8 narrows through a
**cast** rather than a **trained quantizer**, which removes a large and
historically bug-prone body of machinery.

---

## 0. Status

**Measured and rejected for CPU IVFFLAT.** The format and its distance kernels were
built and benchmarked (`6aa855df28`); FP8 costs **6.6x float32** in the L2 kernel
after exhausting every optimisation the SIMD surface allows. The proposal below is
kept because its reasoning stands — the cast-vs-quantizer argument in §1 is sound
and still worth acting on — but the conclusion is that **bf16 already delivers it**.

### What was measured

Seven L2 kernels at dim 1024, against the existing formats:

| format | ns/op | vs f32 | bytes/elem |
|---|---|---|---|
| **bf16** | **39.9** | **0.94x — faster than f32** | 2 |
| f32 | 42.6 | 1.00x | 4 |
| int8 | 55.0 | 1.29x | 1 |
| f16 | 204.8 | 4.81x | 2 |
| **f8 AVX512 x2** | **280.5** | **6.58x** | 1 |
| f8 LUT x4 | 303.4 | 7.12x | 1 |
| f8 AVX512 | 345.5 | 8.11x | 1 |
| f8 AVX2 x2 | 401.9 | 9.43x | 1 |
| f8 AVX2 | 502.3 | 11.79x | 1 |
| f8 LUT flat | 612.8 | 14.38x | 1 |
| f8 scalar | 2379.0 | 55.85x | 1 |

The optimisation levers compound — 512-bit lanes 1.45x, loop unrolling 1.25x,
1.79x together over plain AVX2 — and the floor is still 6.6x.

### Why the floor is structural

x86 has **no FP8 conversion instruction**, the way f16 has `VCVTPH2PS`, and Go's
`simd/archsimd` exposes **no widening load** (`As*` are bit reinterprets, not
`VPMOVZXBD`-style conversions). So every value costs a shift, two masks, a multiply
and an or. int8 escapes all of it because sign-extending a byte to a dword is one
instruction — which is exactly why it runs at 55 ns and FP8 cannot.

The distance kernel was 28% of query CPU in profiling. At 6.6x that becomes ~185%,
roughly tripling query cost, and the memory saving cannot offset it: int8's 4x
footprint reduction produced **no** speedup on this path, so it is not memory-bound.

### Recommendation: bf16

bf16 delivers the entire motivation of §1 — narrower entries with **no quantizer**,
since it is already in the `CastSQL` family — while running *faster than float32*
at half the size, with ~0.4% relative error against E4M3's ~6%. It is already
implemented, already supported by all three index families, and needs no new type,
no parser change, and no kernels.

If the goal is "narrow IVFFLAT entries without the quantizer", the answer is
`quantization 'bf16'`, available today.

### Findings that outlive the verdict

1. **Raw normalized embeddings must not be stored in E4M3 unscaled.** Components of
   a unit-norm 768-dim vector sit near 1/sqrt(768) ~ 0.036, putting ~14% of them
   below 2^-6 into the subnormal regime, where spacing is absolute rather than
   relative; the smallest flush to zero. A fixed scale of 32 lifts them into the
   normal range: worst per-component error 100% -> 5.9%, squared-L2 error against
   exact f32 0.72%. The scale is a constant, not trained state, and cancels out of
   the ranking since scaling both sides scales every distance by scale^2.
2. **A branchy decode, not arithmetic, is what makes a scalar narrow kernel slow.**
   `Float8.ToFloat32` renormalises subnormals in a loop, so it branches per element.
   Replacing it with a 1 KiB L1-resident lookup table is 7.3x faster at identical
   unrolling — worth remembering for any future narrow type.
3. **Go 1.26 `simd/archsimd` has no float type below Float32.** Only
   `Float32x{4,8,16}` and `Float64x{2,4,8}`; f16 and bf16 are handled by loading raw
   bits as `Uint32xN` and decoding in-register. Any future sub-32-bit float format
   inherits that constraint.

---

## 1. Motivation: cast family vs quantizer family

MO already narrows vectors two different ways, and the split is visible in
`pkg/vectorindex/quantizer/quantizer.go`:

- **Cast family** — `CastSQL` emits `cast(<col> as <type>(dim))` for the float
  formats (float16, bf16, float32). No trained state.
- **Quantizer family** — `Int8EntrySQL` / `Int8EntrySQLFromBounds` emit an affine
  map `q(x) = x*mul + add` built from a trained `[min,max]`.

The quantizer family carries real weight. Its bounds must be trained on a sample
and stored in index metadata, then read back by every writer and by the query
encoder. `Int8EntrySQL`'s own doc comment spends fifteen lines on hazards the cast
family does not have: pinning the affine map to float32 arithmetic so a `vecf64`
base does not compute `q(x)` in f64 while the query encodes in f32, and formatting
`mul`/`add` with `%.9g` from the float32-narrowed value so the emitted literal
does not parse to an *adjacent* float32 and shift a bucket boundary by one.

That machinery has produced defects. #27732 was exactly this shape: the
synchronous-DML writer had no quantizer at all and projected the base column
verbatim into a narrow entries column, so a `vecf32(4)` landed as 16 raw bytes in
a `vecint8(4)`. Related: scalar-quantizer bounds trained on an unrepresentative
prefix saturate on sorted loads.

A cast type cannot have any of these. All three IVFFLAT entry writers (build,
ISCP, synchronous DML) emit the *same* cast expression, so they agree by
construction; there are no bounds to train, store, read back, or invalidate.

**This is the case for FP8. It is not a performance case** — see §8.

---

## 2. Format decision: E4M3

| | exponent | mantissa | significant bits | max finite | min subnormal |
|---|---|---|---|---|---|
| **E4M3** | 4 (bias 7) | 3 | 4 | 448 | 2⁻⁹ ≈ 0.00195 |
| E5M2 | 5 (bias 15) | 2 | 3 | 57344 | 2⁻¹⁶ |

**E5M2 is cheaper to decode.** It has the same 5-bit exponent and bias 15 as IEEE
float16, so an E5M2 value is bit-for-bit the top 8 bits of a float16 — the same
relationship bf16 has to float32. `E5M2 << 8` yields a valid float16, so the
existing `f16decX8` SIMD path could be reused with one extra shift.

**E4M3 is still the right choice.** E5M2 spends an exponent bit on dynamic range
that embeddings never use, and pays for it with the single mantissa bit that
decides recall. The decode cost is a handful of integer ops amortised across a
768-dimension dot product; the precision loss is permanent.

E4M3 is also the industry default for inference (NVIDIA Transformer Engine uses
E4M3 forward, E5M2 only for gradients).

### 2.1 Range is a non-issue for embeddings

For a unit-norm 768-dimension embedding, components are approximately Gaussian
with σ ≈ 1/√768 ≈ 0.036.

- **Upper**: components peak near 0.2, against E4M3's 448 — roughly 2000×
  headroom. Saturation is unreachable in practice.
- **Lower**: about 4% of components fall below the smallest subnormal 2⁻⁹ and
  flush to zero. Each contributes ≈ (0.002)² ≈ 4×10⁻⁶ to a squared distance of
  1.0 — negligible.

`vecf32` is a general column type, so a value above 448 is *representable in the
source*. The cast is therefore defined as **saturating** at ±448 (§5.2), not as
an error, matching how a narrowing cast behaves elsewhere.

---

## 3. Phase 0 — superseded

This section originally gated the work on a numpy recall simulation. That was the
wrong first question: building the distance kernels answered feasibility faster and
answered it on throughput, before recall ever became the deciding factor.

Recall was never measured for E4M3, and does not need to be — a format that costs
6.6x in the kernel cannot be justified by recall parity. The per-component error
figures in §0 are the only accuracy numbers gathered, and they are sound (5.9%
worst per component when scaled, 0.72% on squared L2), which suggests recall would
likely have been acceptable. That is precisely why measuring throughput first
mattered.

---

## 4. Non-goals

- **Not a GPU feature.** cuVS instantiates its indexes for `float`, `half`,
  `int8_t`, `uint8_t` only (verified across `cgo/cuvs`: 510/256/73/68 occurrences,
  zero for any fp8 spelling). CAGRA and IVF-PQ therefore cannot consume `vecf8`,
  and this proposal does nothing for the 88M-on-45GB-VRAM goal.
- **Not a throughput feature.** See §8.
- **Not FP4.** MXFP4/NVFP4 need block-shared scales, reintroducing quantizer-like
  state, and are blocked on cuVS support regardless.

---

## 5. Design

### 5.1 Type

- `types.T_array_float8 T = 230` — next free ID (226–229 are bf16, float16, int8,
  uint8)
- SQL name `vecf8`, following `vecf16` / `vecbf16` / `vecint8` / `vecuint8`

### 5.2 Conversion

New `pkg/container/types/float8.go`, mirroring `float16.go`:

- float32 → E4M3: round-to-nearest-even, **saturate at ±448**, flush subnormals
  below 2⁻⁹ to zero
- E4M3 → float32: shift into position, rebias exponent 7 → 127, handle E4M3's
  non-IEEE specials (no ±Inf; NaN is the single encoding `S.1111.111`)

### 5.3 Distance kernels

`pkg/vectorindex/metric/distance_func_narrow*.go`.

Go's `simd/archsimd` exposes only `Float32x{4,8,16}` and `Float64x{2,4,8}` as
float vectors — **no Float8, and no Float16 either**. The existing f16 kernels
already work around this by loading raw bits as `Uint32x8` lanes and decoding
in-register:

```go
au, bu := f16AsU32(a), f16AsU32(b)
ua := archsimd.LoadUint32x8Slice(au[j : j+8])
dE := f16decX8(ua.And(mLo), ...).Sub(f16decX8(ub.And(mLo), ...))
dO := f16decX8(ua.ShiftAllRight(16), ...).Sub(...)
acc0 = dE.MulAdd(dE, acc0)
```

E4M3 uses the same structure, unpacking **4 values per uint32 lane** instead of 2.
Required kernels: l2sq, innerProduct, l1Distance, cosineDistance — scalar first,
then AVX2/AVX512 tiers matching the existing narrow-type layout.

### 5.4 IVFFLAT integration

- `quantizer.go`: add float8 to the **`CastSQL`** branch. It must **not** reach
  the affine path — that is the entire point of the proposal.
- entries column type and `topnDistOf[T]` re-rank dispatch
- the three entry writers need no per-path work: all emit `CastSQL`

### 5.5 Rejection elsewhere

CAGRA, IVF-PQ, and the GPU paths must refuse `vecf8` with a clear error rather
than silently misbehave, since cuVS has no such dtype.

---

## 6. Code locations

Scope estimated from the two precedent commits.

**`f7c2a37d6f`** added vecbf16/vecf16/vecint8 as column types — **36 files**:

| area | files |
|---|---|
| format | `pkg/container/types/float16.go` (+287), `types.go` |
| vector | `pkg/container/vector/{vector,utils,tools}.go` |
| parser | `keywords.go`, `mysql_sql.y`, **regenerated `mysql_sql.go`** |
| functions | `func_{cast,compare,unary,binary}.go`, `list_builtIn.go`, `type_check.go` |
| frontend | `output.go`, `resultset.go`, `util.go` |
| plan | `build_util.go`, `make.go` |
| sort | `pkg/sort/sort.go` |
| BVT | `test/distributed/cases/array/array_vecnarrow.{sql,result}` |

**`1ac3c86fdf`** wired those types into index quantization — **343 files**, but
that covered all three index families plus GPU, CDC, and quantizer-training BVT.
Scoped to IVFFLAT with a cast-only type, the equivalent work is far smaller: no
metadata, no training, no per-writer changes.

---

## 7. Testing

- `float8_test.go`: round-trip, round-to-nearest-even at ties, saturation at
  ±448, subnormal flush, NaN encoding
- kernel tests: scalar vs SIMD equality per metric, mirroring
  `distance_func_narrow_*_test.go`
- BVT `array_vecf8`: cast, compare, output, load
- BVT `vector_ivfflat_f8`: build, DML, reindex, search — including DML *after*
  index creation, which is the shape that hid #27732
- negative BVT: CAGRA / IVF-PQ reject `vecf8`
- recall on wiki_all 1M, checked against the Phase 0 prediction

---

## 8. Risks and open questions

1. **Recall (blocking).** Gated by Phase 0.
2. **Parser regeneration.** The precedent shows a ~17k-line generated
   `mysql_sql.go` diff. Isolate it in its own commit so review can skip it.
3. **Type ID 230 touches serialization.** Additive, but confirm no persisted
   format assumes a contiguous or bounded type range.
4. **No throughput win is expected.** int8 — the same one byte per element —
   measured *no* speedup over float32 on the IVFFLAT path. Interleaved A/B
   (index rebuild ≈ 25 s, so rounds are cheap and paired):

   | round | float32 | int8 |
   |---|---|---|
   | 1 | 355.4 QPS @ 0.8941 | 238.9 QPS @ 0.8800 |
   | 2 | 374.0 QPS @ 0.8977 | 378.5 QPS @ 0.8777 |

   Narrower entries did not convert to throughput here, so FP8 should be argued
   on memory footprint and on deleting the quantizer — not on speed. (QPS on the
   development laptop also drifts downward under sustained load — identical code
   measured 461 → 382 → 354 — so anything under ~20% needs interleaved A/B with
   cooldown.)
5. **`vecf8` is CPU-only** for the foreseeable future, and IVFFLAT is the only
   index family that can use it.
6. **A third one-byte type erodes the width diagnostic.** Go's type system handles
   the aliasing fine — `BF16` and `Float16` are already two distinct named types
   over the same `uint16`, and `T_array_int8`/`T_array_uint8` are already two
   distinct one-byte array types, so dispatch is by Oid and nothing infers the
   element type from byte width. The `ArrayElement` constraint lists `BF16` and
   `Float16` as named types rather than `~uint16`, which is what keeps them
   separate; `Float8 uint8` must be added the same way and must not be spelled
   `~uint8`.

   The cost is diagnostic, not structural. #27732 was caught loudly because entry
   width did not match base width; a mis-encoding between two *same-width* types
   raises nothing and silently returns wrong distances. `vecf8` makes a third
   member of the one-byte class, so width distinguishes even less than before.
   Mitigation: the cast-only property means all three writers emit the same
   expression and cannot drift, and entry-level tests must compare **bytes**
   against an oracle rather than only checking dimensions.
