# Plan: Run `leet-triton/medium-lora_linear.py` on the Metal backend

> ## ✅ DONE (2026-07-14) — the verbatim, unmodified `medium-lora_linear.py` compiles and runs
> **bit-exact on the Metal backend**, single- and multi-program grids, at its native block shape
> (`BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, BLOCK_R=16`, `num_warps=4`), with `tl.swizzle2d`, masked
> loads + the `*` store mask, and non-multiple-of-8 dims. Committed in `ce92325`, `bd42283`,
> `4671b80`, `b66457d`. Test: `test_metal_lora_linear_verbatim`. All 8 walls closed
> (W2a runtime-K · W1 trans-B/fan-out/loop-acc · W2b recompute/multi-acc · W2c fused epilogue ·
> masks · multi-tile · multi-warp · masked-multi-warp). The sections below are the historical
> wall-by-wall record.


**Status:** `draft` — investigation complete, empirically validated, pending approval
**Generated:** 2026-07-14 · **Branch:** `metal-develop`
**Goal:** Take the fused LoRA-linear kernel from "fails at `convert-tritongpu-to-metal`" to "compiles + numerically correct on MPS."
**Primary file:** `third_party/metal/lib/Conversion/TritonGPUToMetal/TritonGPUToMetal.cpp` (7194 lines). All bare `:NNNN` line refs below are in this file.

---

## TL;DR

The kernel needs **four** distinct backend capabilities that don't exist yet. They are dependency-ordered and each is independently landable + testable (I isolated each with a minimal repro that fails today):

| # | Wall | First failure point | Needed by |
|---|------|---------------------|-----------|
| **W1** | `convert_layout` dot-operand preempt too narrow (can't peel through `tt.trans`, fan-out, or loop-carried acc) | classifier error `:6795` | as-written kernel |
| **W2a** | Dynamic / large **K-loop** dot (matchers require *static* trip count, ≤8 tiles) | `tt.dot` illegal `:7109` | as-written **and** decomposed |
| **W2b** | Two dots + two accumulators sharing one `scf.for` | matcher bail `:5712/:6144` | as-written kernel |
| **W2c** | Post-loop dot on a **loop-carried** operand + fused `scale*·+acc` epilogue | no matcher path `:6287` | as-written kernel |

**Two strategic tracks:**
- **Track B (recommended first — fastest correct result):** rewrite the kernel as 3 canonical matmuls, loading operands already-transposed via stride addressing (no `tl.trans`), intermediate `acc1` staged through DRAM. This needs **only W2a**. Gets a correct LoRA result on Metal in the least time and delivers the single highest-leverage matmul primitive (runtime-K dot) that every real matmul kernel wants.
- **Track A (support the kernel *as written*):** needs **W1 → W2a → W2b → W2c** in order. This is the "true fix" for the fused kernel; largest scope.

Recommendation: **land W2a first** (unblocks Track B and is a general win), then decide whether the fused kernel (Track A: W1/W2b/W2c) is worth the additional depth.

---

## Confirmed current baseline (what works today)

`pytest python/test/unit/test_metal_backend_dot_universal.py test_metal_backend_dot_f32_8x8.py` → **21 passed.** The supported `tt.dot` envelope is:

- exactly **one** `tl.dot`, **one** accumulator iter_arg;
- K-loop with a **static (constexpr) trip count**, **N ∈ [1,8] tiles** (so K ≤ 8·BK);
- operands **directly** from `tt.load` — no `tl.trans`, no fan-out to a second dot;
- result f32, rank-2, `shape % 8 == 0`, single use → one `tt.store`;
- `num_warps ∈ {1,4}` verified.

I verified the static-vs-runtime boundary directly: an otherwise-identical single-dot K-loop **compiles + is correct** with a constexpr `range(0, K_TILES)` but **fails to legalize `tt.dot`** with a runtime `range(0, K, BK)`. Repro files in the session scratchpad (`baseline.py` fails, `baseline2.py` passes).

The LoRA kernel violates this envelope on every axis at once: runtime K-loop (`medium-lora_linear.py:44`), two dots/two accumulators in the loop (`:52,:53`), a post-loop dot on a loop result (`:56`), and three `tl.trans` operands (`:52,:53,:56`).

---

## Blocker map (dependency-ordered)

### W1 — `convert_layout` dot-operand preempt is too narrow  *(the first error the as-written kernel hits)*  🟡 **trans-on-B DONE (2026-07-14)**

**Landed (transpose-on-B).** `tl.dot(a, tl.trans(b))` now compiles + runs on the runtime-K path,
with `b` kept in its natural `[N,K]` layout (no torch-side pre-transpose). Changes:
(1) `preprocessDotCvtChains` `peel` now accepts `cvt(trans(load))` (keeps the `tt.trans`, strips the cvt);
(2) `SimdgroupLoadDeviceStagedOp` gained a `transposed` UnitAttr (`MetalOps.td`) — the emitter writes the
gathered element to the **swapped** staging slot `sj*cols+si` (device address unchanged), so `simdgroup_load`
reads the transpose; (3) `tryRuntimeKLoopCanonicalDot` detects `dot.getB() == tt.trans(loadB)`, extracts the
N-origin from axis 0 (vs axis 1 for normal `[K,N]`), and emits the B staged load with the `transposed` attr and
`(N-origin, K-offset, N-stride)`; (4) the static canonical matcher bails on a `tt.trans` operand (no miscompile).
Tests: `test_metal_backend_dot_dynamic_k.py::test_dot_dynamic_k_transposed_b` (5 shapes), lit
`dot_dynamic_k_transposed_b.mlir` (asserts `{transposed}` on the B load). **257 pytest + 99 lit, zero regressions.**
**Full LoRA now runs on Metal via 3 trans-B matmuls with weights left as `[N,K]/[R,K]/[N,R]` — bit-exact.**
Constraint: the B tile's K dimension must be contiguous (`stride_bk == 1`), which holds for contiguous weights.

**Still open (needed only by the *fused* single-kernel LoRA, pair with W2b/W2c):** peeling a **fan-out** cvt
(one load feeding two dots) and a **loop-carried-accumulator** operand. Transpose-on-A is also deferred (LoRA
never needs it — every LoRA dot transposes the *second* operand).

**Original spec (for reference):**

**Symptom (verified):**
```
error: 'ttg.convert_layout' op ttg.convert_layout: broader staged-transpose
       deferred to L1d3 (rank≠2 or shape/elem-type change or non-blocked
       encoding or sizePerThread > 1)
```

**Root cause.** TritonGPU inserts a `ttg.convert_layout` (`#blocked → #dot_op<#mma>`) on each dot operand. The classifier walk `:6743–6800` accepts a cvt only if it's in-envelope (`sameShapeRank2BlockedPair && sizePerThreadAllOne`, predicate `:6778–6791`); a `#dot_op` destination is **not** a `BlockedEncodingAttr`, so it fails the "non-blocked encoding" arm and errors at `:6795`.

These cvts are *supposed* to be stripped earlier by `preprocessDotCvtChains` (`:6421`, peel helper `:6441–6453`) — but `peel` only fires when the cvt's source is a **plain single-use `tt.load`**, and it rewires A+B together-or-not-at-all (`:6457`). Every LoRA dot has an operand that isn't a plain load:
- `tl.dot(x, tl.trans(w), …)` / `tl.dot(x, tl.trans(a), …)` → B's source is `tt.trans` (`TransLowering` treats trans as identity passthrough `:1320–1329`) → not peeled.
- `x` is shared by **two** dots → its cvt is multi-use → not peeled.
- `tl.dot(acc1, tl.trans(b))` → A is a loop-carried accumulator (not a load) → not peeled.

**Fix.** Extend `preprocessDotCvtChains` to also peel:
1. **through `tt.trans`** between load and cvt — rewire the dot operand to the load and record a "transposed" flag so the downstream `simdgroup_load` reads Bᵀ via swapped row/col strides (the matmul track already extracts strides via `findStrideSplatSource`); no physical transpose needed;
2. **multi-use cvts** (fan-out) — peel per-use / clone the rewired operand;
3. **loop-carried-accumulator operands** — tolerate an operand that is an `scf.for` result rather than bailing the whole dot (the acc-init cvt case is already flagged as deferred at `:6412–6417`).

**Test.** New lit fixture `dot_operand_trans_peel.mlir` (`CHECK-NOT: ttg.convert_layout`, `CHECK: metal.simdgroup_load`), plus pytest: single-dot K-loop with `tl.dot(x, tl.trans(w))` compiles. **Spec already exists:** `.omc/specs/l1d3-broader-staged-transpose-dot-bridge.md` (Options A/B, ~6–10h estimate). **Size: M (2–4 days).**

### W2a — Dynamic / large K-loop dot  *(needed by BOTH tracks — highest leverage)*  ✅ **DONE (2026-07-14)**

**Landed.** New matcher `tryRuntimeKLoopCanonicalDot` (`TritonGPUToMetal.cpp`) matches the
canonical 3-iter_arg loop with a *runtime* upper bound (element-offset form, `step == BK`,
single-warp, unmasked, 8×8 f32) and emits a fresh `scf.for` stepping the K axis by 8 with a
single `simdgroup_matrix` accumulator iter_arg. Enabled by three `ModuleTranslation.cpp`
extensions: (1) `scf.for` with a `simdgroup_matrix` iter_arg (reuse the init op's temp),
(2) in-place `simdgroup_multiply_accumulate(vAcc, A, B, vAcc)` when C is the loop iter_arg
(Apple-family-9-safe), (3) `translateVarName` consults `_buffers` first (loop result). Plus the
`scf.for` legality guard now allows a `simdgroup_matrix` iter_arg. Tests: `test_metal_backend_dot_dynamic_k.py`
(8 cases, runtime K∈{8,16,64,128,256} + program-grid tiling), lit `dot_dynamic_k.mlir`
(asserts the loop is preserved, one MA inside). **252 pytest + 98 lit pass, zero regressions.**
**Track B proven:** a decomposed LoRA (3 runtime-K matmuls + torch combine) is numerically correct on Metal.

**Note (constrains W1 + Track B):** the `simdgroup` staged-load assumes **contiguous columns**
(`_stage_shared[c] = b[(row+si)*stride + col + sj]`, unit col stride). So "load B transposed via
swapped strides" does **not** work — Track B must pre-transpose weights to contiguous `[K,N]`
(a one-time prep), and W1 must realize the transpose physically (stage with a genuine column
gather), not by stride-swapping. First cut is single 8×8 tile / single-warp; multi-tile-per-program
and multi-warp runtime-K are follow-ups (larger shapes work today via an 8×8 program grid).

**Original spec (for reference):**

**Symptom (verified):** with a runtime loop bound, `error: failed to legalize operation 'tt.dot'` (op survives all matchers → illegal at `applyFullConversion` `:7109–7112`).

**Root cause.** All three matchers statically unroll and cap the trip count: `tryUnrollCanonical3IterArgDot` requires a static trip count with `N ∈ [1,8]` (`:5787–5792`); `tryUnrollKLoopDot` the same (`:6169–6174`). LoRA's `for k in range(0, K, BLOCK_SIZE_K)` has a **runtime** bound (K = `d_in`), and K is routinely > 8·BK.

**Fix.** Add a matcher that lowers a K-loop dot into an **`scf.for` of `metal.simdgroup_multiply_accumulate`** with a *runtime* trip count (the M/N tile grid stays as-is; only the K axis becomes a runtime loop instead of a static unroll). Reuse the existing `simdgroup_load/_multiply_accumulate/_store` emitters; the accumulator stays a single f32 iter_arg. Gate: masked K-tail handling for `K % BK != 0` (LoRA masks with `offs_k < K`).

**Test.** `test_metal_backend_dot_dynamic_k.py`: canonical single-dot matmul with runtime `K ∈ {64, 128, 512}`, `atol=1e-4` vs `torch.matmul`; lit fixture asserting `scf.for` (not unrolled) around `simdgroup_multiply_accumulate`. **Size: M–L (3–5 days).** This is the item to build first — it is the missing primitive behind essentially every non-toy matmul.

### W2b — Two dots + two accumulators in one K-loop  *(Track A)*  🟡 **recompute-from-IV single-dot DONE (2026-07-14)**

**Key discovery:** `medium-lora_linear.py`'s inner loop is the **recompute-from-IV** shape
(`offs_k = k + tl.arange(...)`, accumulators as the *only* iter_args, addresses rebuilt from the
induction var each iteration), NOT the pointer-advance shape W2a handles. New matcher
`tryRuntimeKLoopRecomputeDot` handles it: origins/strides are pulled from the in-loop loads
(with an axis-aware `findAxisStride` that avoids mistaking the K-offset's `splat(iv)` for a
stride — the generic helper's failure mode, which crashed erase with a dangling IV use), the
K axis is verified to be the induction var (`kAxisUsesInductionVar`), and a fresh step-8
`scf.for` carries the `simdgroup_matrix` accumulator. Composes with W1 (trans-B). First cut:
single dot / single 8×8 tile / single-warp / unmasked. Tests: `test_dot_dynamic_k_recompute_transposed_b`
(4 shapes), lit `dot_dynamic_k_recompute.mlir`. **261 pytest + 100 lit, zero regressions.**

**Multi-accumulator DONE (2026-07-14).** The real two-dot/two-accumulator loop
(`acc0 = dot(x, trans(w)); acc1 = dot(x, trans(a))` sharing `x`) now runs. Landed:
(a) the scf.for emitter's matrix-iter_arg branch generalised from 1 to **N** `simdgroup_matrix`
accumulators (each reuses its init temp; yields are in-place no-ops); (b) `tryRuntimeKLoopRecomputeMultiDot`
— an atomic rewriter that validates N dots sharing one A-operand load, emits one fresh `scf.for`
carrying N accumulators with a single shared staged A load per iteration, and N post-loop stores;
(c) `preprocessDotChains` restructured to group dots by enclosing loop and rewrite multi-dot loops
atomically (the per-dot worklist would dangle when one call erases a sibling dot); (d) the **fan-out**
fix in W1 — the preempt's `peel` no longer requires `hasOneUse`, so a shared operand's `convert_layout`
(fed to multiple dots) is rewired per-dot and dropped once fully unused. Tests:
`test_dot_dynamic_k_two_accumulators` (4 shapes), lit `dot_dynamic_k_multi_acc.mlir`.
**265 pytest + 101 lit, zero regressions.**

**Still open for the fully-fused kernel (dependency-ordered):**
1. **Masked loads** — LoRA masks every load (`offs_m<M & offs_k<K`, etc.); the matchers currently
   require unmasked. Needs a K-tail + M/N/R-edge masked staged load.
2. **W2c** (below) — the post-loop dot #3 + fused `scale*·+acc` epilogue + masked store.
3. **Multi-tile output** (N>8 per accumulator) — the matchers are single 8×8 tile; larger N/R
   currently need an 8×8 program grid.

**Original spec (for reference):**

**Symptom (verified):** minimal two-dot/two-accumulator loop (no transpose) fails at W1's classifier first; after W1 it would fail the dot matchers.

**Root cause.** The LoRA loop carries 5 iter_args `{x_ptrs, w_ptrs, a_ptrs, acc0, acc1}` with 2 dots. `tryUnrollCanonical3IterArgDot` requires **exactly 3** iter_args (`:5712`) and **exactly one** dot in the body (`if (dotInBody) return failure()` `:5741`, plus 3rd-load bail `:5739`); `tryUnrollKLoopDot` requires **exactly one** iter_arg (`:6144`).

**Fix.** Generalize the K-loop matcher to *n* accumulators / *m* dots sharing one loop: collect all `{dot_i, acc_i}` pairs, verify each acc_i is a distinct iter_arg yielded by dot_i, allow a shared operand (`x`) feeding multiple dots, and emit one `simdgroup_multiply_accumulate` chain per accumulator. Composes with W2a (runtime K).

**Test.** `test_metal_backend_dot_multi_acc.py`: the probe-B kernel (two dots, shared A operand, two accumulators, runtime K), both outputs correct. **Size: M (2–4 days).**

### W2c — Post-loop dot on a loop-carried operand + fused epilogue  *(Track A)*  ✅ **DONE, unmasked (2026-07-14)**

**Landed.** The fully-fused LoRA compute (`medium-lora_linear.py` minus masks) now runs on Metal.
`tryFusedLoRAEpilogue` matches the two-accumulator loop PLUS the post-loop
`acc0 += scale * tl.dot(acc1, tl.trans(b))`. Key trick: `sgmma(D,A,B)=A·B+D` and a fused store fold
the whole epilogue into simdgroup ops with **no simdgroup arithmetic** — `dot3 = acc1·trans(b)` is a
single `sgmma` on the loop's own accumulator, and a new `metal.simdgroup_fused_store(acc0, dot3, scale)`
op stages both matrices to threadgroup scratch and writes `acc0[c] + scale*dot3[c]` per lane (mask-ready).
Also: the preempt `peel` now strips a cvt on a **loop-carried accumulator** operand (dot3's A = `acc1`),
and `preprocessDotChains` runs a two-pass driver (multi-dot loops first — the fused rewriter consumes the
post-loop dot3 — then a fresh re-walk). Tests: `test_dot_dynamic_k_fused_lora` (4 shapes × scales), lit
`dot_dynamic_k_fused_lora.mlir`. **269 pytest + 102 lit, zero regressions.**

**Masks DONE (2026-07-14).** The fused LoRA now runs with **every load and the store masked**
(`tl.load(..., mask=(offs_row < ROW) & (offs_k < K), other=0.0)` + `tl.store(..., mask=(offs_m<M)&(offs_n<N))`),
so non-multiple-of-8 M/N/K/R are correct. Added a `metal.simdgroup_load_device_staged_masked` op
(coop-load with a per-lane `(gi<row_extent && gj<col_extent) ? … : 0` bounds check; extents pulled from
the mask via `extractMaskExtents`), an `emitStagedLoad` helper (masked-or-plain), and wired the fused
matcher's loads + `simdgroup_fused_store`'s `partial_extents`. Tests: `test_dot_dynamic_k_fused_lora_masked`
(4 shapes incl. M=6/N=5/R=3/K=20), lit `dot_dynamic_k_fused_lora_masked.mlir`. **273 pytest + 103 lit,
zero regressions.**

**Multi-tile output DONE (2026-07-14, register-bounded).** Each program now computes a
`(BM/8)×(BN/8)` grid of 8×8 output tiles. Both the single-dot recompute matcher and the fused
LoRA matcher emit the tile grid: one shared K-loop carries all tile accumulators (a `tileOrigin`
helper adds `tileIdx*8` to each base origin), the fused epilogue sums `dot3` over the R-tiles per
output tile, and the multi-tile epilogue chain skips the `ttg.convert_layout`s that appear between
`dot3→mulf→addf→store` at larger shapes (recursive dead-op cleanup on erase). Verified: fused LoRA
at `BLOCK 16/32`, `BR 8/16`, masked + unmasked, non-multiple-of-8 dims, bit-exact. A **register
guard** (`MT*NT + MT*RT ≤ 24`) bails grids too large to hold as live `simdgroup_matrix`
accumulators — the verbatim `BLOCK_M=64,N=128` = 144 tiles exceeds it and needs **multi-warp**
(each warp owning a sub-grid), the one genuinely-remaining architectural piece. Tests:
`test_dot_dynamic_k_{transposed_b,fused_lora,fused_lora_masked}_multitile`. **279 pytest + 103 lit,
zero regressions.**

**Multi-warp DONE (2026-07-14, unmasked).** The runtime-K single-dot and fused matchers now accept
`num_warps>1`, partitioning the output tile grid by **M-tile rows** across simdgroups (`sgid`). For the
fused path, M-only partitioning makes each warp self-contained (its output tiles' `dot3` uses only its own
`acc1` M-tiles — no cross-warp dependency), so per-warp accumulators stay bounded
(`mPerWarp*NT + mPerWarp*RT ≤ 24`). `simdgroup_fused_store` gained a per-warp `warp_index`, and the
`_fstore_base/_fstore_delta` epilogue scratch is now declared once and reused across all fused stores
(entry barrier) so threadgroup memory doesn't scale with tile count. **`medium-lora_linear.py`'s exact
block shape `BLOCK_M=64, BLOCK_N=128, BLOCK_R=16` now runs on Metal with `num_warps=8`, bit-exact.**
Tests: `test_dot_dynamic_k_{transposed_b,fused_lora}_multiwarp`. **285 pytest + 103 lit, zero regressions.**

**Bottom line:** `medium-lora_linear.py`'s full computation now runs on Metal at its verbatim block
shape (single-warp for small blocks, multi-warp for large). **The only remaining gap for the byte-for-byte
masked kernel is masked multi-warp** — the masked staged load lacks a per-warp buffer, so masked kernels
with `num_warps>1` bail (clean error). Masks work single-warp; multi-warp works unmasked; combining them
is the final mechanical step (add `warp_index` to `simdgroup_load_device_staged_masked` like the plain and
fused-store ops). (Minor, non-LoRA: masked multi-acc / pointer-advance matchers also still bail on masks.)

**Original spec (for reference):**

**Symptom (verified):** probe-C kernel (K-loop producing `acc1`, then `acc0 = tl.dot(acc1, b)` after the loop) fails.

**Root cause.** Dot #3 is **not** inside an `scf.for`, its A operand is a loop *result* (not a `tt.load`), and its result feeds `scale*· + acc0` arithmetic before the store. The inline `rewriteSingleDot` path requires operands directly from `tt.load` (`:6287–6289`) and result `hasOneUse → tt.store` (`:6293–6296`); no path matches.

**Fix.** Add a single-shot (non-loop) dot matcher whose operands may be arbitrary rank-2 f32 SSA values (loop results included) — stage them into threadgroup/simdgroup matrices — and whose result may feed elementwise ops (`arith.mulf`/`addf`) before a masked `tt.store`. This is the fused-epilogue generalization of the existing 8×8 single-dot path.

**Test.** `test_metal_backend_dot_postloop_fused.py`: probe-C kernel + a `scale*dot + bias` epilogue, correct vs reference. **Size: M–L (3–5 days).**

---

## Two tracks

### Track B — decompose the kernel (recommended first)

The LoRA math is three matmuls: `acc1 = x·Aᵀ` `[M,R]`, `acc0 = x·Wᵀ` `[M,N]`, `out = acc0 + scale·(acc1·Bᵀ)`. Rewrite as 3 canonical single-dot kernels (or one kernel with 3 sequential dots) that **load operands already transposed via stride addressing** — e.g. load `Wᵀ` as `[K,N]` with `W + offs_k[:,None]*stride_wk + offs_n[None,:]*stride_wn`, no `tl.trans`. Stage `acc1` through a DRAM tensor so launch 3 reads it as a plain load (not a loop-carried value).

Result: each dot is the *canonical* supported shape and **avoids W1, W2b, and W2c entirely** — the only missing piece is **W2a (runtime K)**. This is the shortest path to a correct LoRA result on Metal. Cost: the user's exact `.py` is not run verbatim (it's rewritten), and there are 3 launches + one intermediate DRAM buffer.

### Track A — support the fused kernel as written

Land **W1 → W2a → W2b → W2c**. After all four, `medium-lora_linear.py` compiles and runs unchanged. This is the complete fix but ~4× the work of Track B and touches the hardest parts of the backend (layout-conversion preempt + a general multi-dot matcher).

---

## Recommended sequencing

1. **W2a (dynamic-K dot)** — highest leverage; unblocks Track B and every real matmul. Ship with `test_metal_backend_dot_dynamic_k.py` + lit.
2. **Track B rewrite** — land a decomposed `lora_linear` variant + numeric test; **first correct LoRA result on Metal.** (Save the rewrite next to the original, e.g. `leet-triton/medium-lora_linear_metal.py`.)
3. **Decision gate:** is the fused kernel worth Track A? If yes:
4. **W1 (trans/fan-out/acc preempt)** → 5. **W2b (multi-acc loop)** → 6. **W2c (post-loop fused dot)**, each gated by its own pytest + lit and a full `test_metal_backend_*` regression sweep (zero regressions).

---

## Acceptance criteria

| ID | Invocation | Pass condition |
|----|-----------|----------------|
| AC-W2a | `pytest test_metal_backend_dot_dynamic_k.py` | runtime `K ∈ {64,128,512}`, `atol=1e-4` vs `torch.matmul`; lit shows non-unrolled `scf.for` around `simdgroup_multiply_accumulate` |
| AC-B | `pytest test_metal_backend_lora_decomposed.py` | decomposed LoRA vs `x@Wᵀ + scale*(x@Aᵀ)@Bᵀ`, `atol=1e-2, rtol=1e-2` |
| AC-W1 | `pytest` single-dot `tl.dot(x, tl.trans(w))` + `lit dot_operand_trans_peel.mlir` | compiles, correct; `CHECK-NOT: ttg.convert_layout` |
| AC-W2b | `pytest test_metal_backend_dot_multi_acc.py` | probe-B two-accumulator kernel, both outputs correct |
| AC-W2c | `pytest test_metal_backend_dot_postloop_fused.py` | probe-C + scaled epilogue correct |
| AC-A | `pytest test_metal_backend_lora_linear.py` (drives the original `solve()`) | fused kernel compiles + correct |
| AC-reg | full `pytest python/test/unit/test_metal_backend_*` | zero regressions vs baseline |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| **R1.** W1 transpose-via-stride-swap gets the simdgroup_load addressing wrong (silent numeric error, not a compile error). | Bit-tight numeric test with a non-symmetric matrix so a missed transpose is visible; lit-assert the swapped stride operands. |
| **R2.** Runtime-K masked tail (`K % BK != 0`) mis-handles the partial tile. | AC-W2a includes a non-multiple-of-BK K; reuse the proven masked-load path. |
| **R3.** W2b/W2c threadgroup-memory budget (multiple accumulators + staged operands vs Apple's ~32 KB). | Track the same budget concern flagged for `ac4-multiwarp`; prefer single-warp first; MSL aliasing/hoist if needed. |
| **R4.** Every wall touches `applyFullConversion` legality — a partial match can leave an illegal `tt.dot`. | Each wall lands with its own lit fixture + the full `test_metal_backend_*` regression sweep before "done". |
| **R5.** Native C++ changes require a rebuild before testing (per CLAUDE.md: `ninja -C build/cmake.macosx-11.0-arm64-cpython-3.12`). | Bake the rebuild into each wall's verification step. |

---

## Appendix — reproduction

Environment: `.pixi/envs/default/bin/python3` (torch+triton, MPS available). Run the as-written kernel:
```
TRITON_METAL_USE_MPS=1 .pixi/envs/default/bin/python3 <driver calling solve() on MPS tensors>
```
→ `RuntimeError: Metal backend: convert-tritongpu-to-metal failed` (W1 classifier error at `:6795`).

Isolation probes (each fails independently today): `scratchpad/isolate.py` — A = single-dot + `tl.trans`, B = two-dot/two-acc, C = post-loop dot. Baseline boundary: `scratchpad/baseline.py` (runtime K, **fails**) vs `scratchpad/baseline2.py` (static K, **passes**).
