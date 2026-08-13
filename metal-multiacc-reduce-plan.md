# Metal backend: multi-accumulator reduction (ILP unrolling)

**Status:** Phase 0 verification DONE 2026-07-13 — **VERDICT: PASS.**
**Phase 1 implementation DONE 2026-07-13 — landed & green.**
Spin-off from `metal-num-stages-pipelining-plan.md` (num_stages Phase 0 FAILED;
this one, discovered in the same spike, PASSES).

## Implementation (DONE 2026-07-13)
K=8 multi-accumulator reduce, gated at E>=8 (K=1 below → byte-identical to the
prior single-iter_arg emission). Changes:
- `TritonGPUToMetal.cpp` `lowerRank1Reduce`: the per-thread partial loop is now
  `scf.for(0, E, K)` with K f32/i32 iter_args (body refactored into a
  `buildPartial(kFlat, acc)` lambda, masked-`other` validation hoisted), plus a
  balanced tree-combine of the K results. The post-conversion safety guard was
  relaxed to allow any number of *scalar* f32/i32 iter_args.
- `ModuleTranslation.cpp` + `.h`: emitter learned multi-iter_arg `scf.for`
  (declare N temps, write all N back at `scf.yield`); `_scfForIterArgsMulti`
  map. **Bug fixed:** `translateValueOrVarName` now resolves a value via its own
  `_buffers` temp before translating its defining op — required for a specific
  result of a multi-result `scf.for` (else UNREACHABLE; hit only on the max
  path, since sum block_gt_tpb takes a different single-acc lowering).
- Lit: `rank1_reduce_addf_block{2048,4096_chain,16384}.mlir` updated to assert
  the K=8 structure (`step 8`, 8 iter_args, `CHECK-COUNT-8` combines).

Validation: reduce pytest 66/66 (sum+max f32 E=8/32/64 at K=8), Metal lit 97/97,
full metal suite 244 passed (0 regressions). MSL confirmed:
`for (int v12=0; v12<64; v12+=8)` + 8 accumulators + tree-combine.
Correctness: i32 bit-exact (associative), f32 within existing assert_close
tolerances. Perf win documented in Phase 0 below.

---

## 0. TL;DR

The Metal backend emits per-thread reduction partials as a **single serial
accumulator**: `float acc=0; for(i=lid; i<BLOCK; i+=256) acc+=x[i];`. At low
occupancy (the backend runs *all* reduces in a single threadgroup, `grid=(1,)`),
the bottleneck is the **serial FADD dependency chain**, not memory. Breaking it
with K independent accumulators (`a0..a7`, combined before the threadgroup tree
reduce) gives real speedups **at the sizes the backend actually generates**.

This is a *different* transform from `num_stages` (which was proven worthless):
it exploits instruction-level parallelism in the accumulation, not memory
prefetch.

## 1. Phase 0 data (amortized dispatch — the representative repeated-use case)

Hand-written MSL A/B mirroring the real kernel shape (single threadgroup, 256
threads, strided partial → threadgroup tree reduce). `compile_shader`, MPS,
amortized over 500 launches/iter (single-isolated-launch is distorted by a
~150µs cold-Event floor; the *amortized* trivial-kernel floor is only 3.4µs).
Script: `scratchpad/phase0_macc_floor.py` (also `phase0_macc_reduce.py`).

| BLOCK | E=iters/thr | base µs | macc8 µs | speedup |
|------:|------------:|--------:|---------:|--------:|
| 2048  | 8   | 5–6  | 3–5  | **1.24–1.70x** |
| 16384 | 64  | 8–12 | 4–5  | **1.81–2.96x** (tutorial02 autotune endpoint) |
| 65536 | 256 | ~26  | ~7   | **3.82x** (stable) |
| 262144| 1024| ~99  | ~23  | **4.1–4.4x** (stable) |

Small-BLOCK numbers are noisier (near the 3.4µs floor) but never dip below
1.24x. `macc4` ≈ 80–90% of `macc8`'s win; **K=8 is the sweet spot**.

**Contrast with num_stages:** memory prefetch was flat 0.95–1.02x everywhere.
Multi-acc is a genuine, monotonic win. The two findings came from the same
experiment (`phase0_bottleneck.py`): prefetch does nothing, breaking the FADD
chain does everything.

## 2. Why it works here specifically
- The Metal backend does **every reduce in one threadgroup** (`grid=(1,1,1)` in
  all reduce tests) → inherently **low occupancy** → the FADD chain is not
  hidden by other warps → ILP unrolling is exactly the right lever.
- The Metal/AIR compiler will **not** auto-reassociate FP reductions (illegal
  without fast-math, which `compile_shader` can't opt into per-kernel), so the
  serial chain survives to hardware unless we break it in source.

## 3. Correctness
- **Integer reduces: bit-exact.** Integer add/max/min are associative, so K-way
  reassociation is identical. The one `torch.equal` (bit-exact) test
  (`test_metal_backend_reduce_sum.py:80`) is i32 → unaffected.
- **Float reduces: within existing tolerances.** All float reduce tests use
  `torch.testing.assert_close(atol=1e-3..1e-5)`. K=8 reassociation changes
  rounding by ~ULP·log(N) — well inside those bounds (verify in Phase 1).
- **Combiner-generic:** valid for associative combiners (sum, max, min, prod).
  argmax/argmin (index-carrying) and any non-associative combiner must be
  **excluded** (fall back to single accumulator).

## 4. Proposed implementation
- **Site:** the per-thread partial-accumulation loop in the reduce lowering
  (`TritonGPUToMetal.cpp` — the Wall-14 bounded-unroll / E-loop that emits
  `for (v5=0; v5<E; ++v5) acc = acc + x[...]`). Emit K accumulators + a combine
  step before the existing threadgroup tree reduce. The tree-reduce stage is
  unchanged.
- **K:** default 8 (best in data); expose as an internal constant, not a user
  option.
- **Gate:** only when (a) the combiner is associative, (b) `E >= E_min`
  (~8–16; below that the chain is too short and K accumulators just add
  register pressure — see BLOCK=2048 giving only ~1.3x and the negative
  register-pressure trend at high K / tiny E), (c) element type is int or f32
  (skip until fp16/bf16 accumulation semantics are confirmed).
- **Numerics guard:** keep the combine order fixed/deterministic so results are
  reproducible across runs.

## 5. Risks
| Risk | Mitigation |
|---|---|
| Register pressure lowers occupancy at high K | K=8 cap; gate on E≥E_min; measure |
| Tiny reduces (E<8) regress | gate them out (fall back to single acc) |
| Non-associative combiner miscompiles | associativity gate; single-acc fallback |
| fp16/bf16 accumulation differs | restrict to int/f32 in v1 |
| Win only visible under amortized (repeated) dispatch | that IS the real case (softmax/iterative launch back-to-back on the MPS stream); document it |

## 6. Effort estimate
Localized codegen change (no new pass, no multi-iter_arg emitter rewrite):
**~2–4 days** incl. K/threshold tuning, associativity gate, and reduce-test
sweep (add a large-BLOCK perf assertion + confirm existing tolerances hold).
Much cheaper than the (abandoned) num_stages pipelining.

## 7. Open questions for Phase 1
- Exact `E_min` threshold (measure the regression boundary in-backend).
- Does the transform compose with the existing threadgroup tree reduce and the
  masked-tail handling (BLOCK>tpb `other=0.0`/`-inf` fill) without extra work?
- Could `metal::simd_sum()` (hardware SIMD-group reduction) replace part of the
  tree-reduce stage as a *complementary* win? (separate investigation)
