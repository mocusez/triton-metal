# Metal backend — `medium-speculative_decoding_verification.py` enablement plan

Status: ✅ **COMPLETE** (2026-07-14). The verbatim kernel COMPILES + RUNS +
matches a numpy reference BIT-EXACT (B=1..8, T=2..8, V=8..1024). All 6 walls
cleared across 4 commits on metal-develop:
`fdc60b6013` (W-A min-reduce) · `4fd61e86f8` (W-E cf.cond_br + scalar i32
load/store) · `380b03b2b2` (W-B rich rank-1 cone) · `96b0c0b608` (W-C tt.scan
cumsum + i1 control flow). Full Metal suite 319 passed, lit 109/109, zero
regression. Durable test: `test_metal_backend_speculative_decoding.py`.

## Progress log

- **Phase 1 / W-A (min-reduce) — DONE, regression-clean, uncommitted.**
  Added Metal dialect `minOp` (enum 14) + translator `min(T(a),T(b))` form +
  pre-pass `arith.minsi` classifier arm + `lowerRank1Reduce` `isMinI` wiring
  (si32 accumulator, identity `INT32_MAX`). Files: `MetalOps.td`, `MetalOps.cpp`
  (build+verify switches), `ModuleTranslation.cpp`, `TritonGPUToMetal.cpp`.
  Tests: pytest `test_reduce_min_rank1_i32` (6 BLOCK sizes) + lit
  `rank1_reduce_minsi_block32.mlir`. **Verified:** rank-1 reduce pytest 30 passed
  (was 24); Metal lit 107/107 (was 106); reduce/FA/adder regression 34 passed.
  The kernel now compiles PAST the `minsi` rejection.

- **Wall re-ordering discovered.** After W-A, the walls surface in a different
  order than the static inventory: cf.cond_br, then scalar i32 store, then
  scalar i32 load, then W-B (the rank-1 cone), then W-C (scan).

- **Increment 2 (commit `4fd61e86f8`) — DONE, regression-clean.** Cleared THREE
  walls at once:
  - **W-E `cf.cond_br`** (early-return guard). `--lift-cf-to-scf` does NOT fix
    it (a void early-return doesn't reconverge). New pre-conversion step
    `structureEarlyReturns` rewrites `cf.cond_br %c, ^ret(tt.return), ^cont` →
    `scf.if %c { } else { <cont> }`, KEEPING the original polarity (cont in the
    branched-to arm). Critical: do NOT negate the i1 — `arith.xori %c, true`
    mis-lowers to MSL bitwise `%c ^ -1` (always truthy; caught by the guard test
    firing wrong). Strict no-op unless the canonical shape matches.
  - **Scalar i32 store** over a multi-level scalar addptr chain
    (`out + b*(T+1) + idx`). `StoreLowering` gains a scalar-addptr branch using
    `accumulateScalarAddPtrOffsets` + `castToMemrefStorage` (i32→ui32 bridge).
  - **Scalar i32 load**, likewise. `ScalarLoadLowering` widened f32-only→f32/i32
    (ui32 storage, bridged back) and switched to `accumulateScalarAddPtrOffsets`
    (the old single-addptr path silently dropped outer offsets).
  - Tests: pytest `test_scalar_i32_load_store_guarded`; lit
    `early_return_guard.mlir`. Suite 313 passed (was 310); lit 108/108.

- **Increment 3 (commit `380b03b2b2`) — W-B DONE, regression-clean + numerically
  verified.** When the single-load Wall-11 walker fails, `lowerRank1Reduce` now
  falls back to the general per-element evaluator `evalRank1ValueAt` (shared with
  the rank-2 reduce), driven once per logical element in the scf.for+butterfly;
  `rank1ConeSupported` validates the cone up front. Three evaluator upgrades:
  (a) load handler adds the scalar tt.addptr base offset (`b*T*V + i*V`, was
  dropped); (b) masked loads guarded by an scf.if (GetElement INSIDE it → no OOB
  read when BLOCK≫V); (c) `arith.andi`/`ori` (i1). `rank1ConeSupported` matched.
  Verified NUMERICALLY (not just legalization): pytest
  `test_reduce_sum_computed_cone` + `test_reduce_min_idx_computed_cone` (masked,
  multi-program, BLOCK=1024>tpb — bit-exact min-idx, tight-tol sum). Negative
  fixture switched math.log→arith.remsi. Suite 318 passed, lit 108/108.

- **SOLE REMAINING WALL: W-C `tt.scan` (cumsum)**, line 85. `tl.cumsum(x, axis=0)`
  → `tt.scan` has ZERO backend infra. It produces a DISTRIBUTED rank-1 tensor
  (element i = Σ[0..i]) consumed by more tensor ops + a min-reduce — a new
  dataflow shape.

- **W-C SPIKE DONE (GO), scratchpad `scan_spike.py`.** Hand-written MSL parallel
  prefix-sum over the EXACT specdec layout (BLOCK=1024, tpb=128, spt=1, cyclic
  `pos=iv*128+tid`, E=8, masked pos>=V→0) matches `torch.cumsum` to f32
  accumulation error (~1e-7…6e-5) across V=8/100/777/1024/999, B=1–8 (multi-
  program). CRUCIALLY the distributed scan result feeds a FUSED follow-on
  (inverse-CDF min-index) bit-exact — proving the `scan→+running_sum→cmpf→
  select→min-reduce` dataflow. Algorithm proven:
    * per iv-block k: Hillis-Steele inclusive scan over tpb threads via a
      threadgroup buffer, DOUBLE-barriered (barrier between the read of
      `tg[tid-offset]` and the write of `tg[tid]`, and after);
    * carry = running total of prior iv-blocks (uniform, `tg[tpb-1]`);
    * `out[pos] = carry + pfx`; masked cols load 0 so padding stays flat.
    * barriers OUTSIDE divergent control (all threads reach them) — FA lesson.

  **Build architecture (for the real lowering):** a `ScanLowering` emitting the
  threadgroup-scratch Hillis-Steele + iv-block carry loop, writing each thread's
  E cumsum values to a threadgroup buffer `scanbuf[BLOCK]`. The scan RESULT is
  then a staged leaf: downstream tensor-op lowerings (the `+running_sum`, cmpf,
  select, min-reduce cones) read `scanbuf[pos]` per element via the
  `evalRank1ValueAt` staged-leaf hook (g_stagedLeaves-style) — W-D falls out for
  free once the scan writes a readable buffer. Multi-week build; spike de-risked
  the two unknowns (cyclic-layout prefix correctness + distributed-result reuse).

## Source of truth: the TTGIR

Full TTGIR dumped via `scratchpad/dump_ttgir.py`. Every reduce/scan in the
kernel is **rank-1** over `tensor<1024xf32|i32, #blocked>` with
`#blocked = sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]`
→ **tpb = 128, BLOCK = 1024 > tpb, spt = 1, E = BLOCK/tpb = 8**. This is the
existing `lowerRank1Reduce` **B2.3 spt1Direct** regime (Wall 8/14), NOT rank-2.

Grid = `(B,)`: each program processes ONE batch row; all "parallelism" is over
the vocab dim V (BLOCK_SIZE_V=1024). The algorithm is inherently sequential
across the outer `for i in range(T)` with a loop-carried `accepted_all` flag.

## Wall inventory (ordered by where the compile dies)

### W-A — min-reduce `tt.reduce(arith.minsi)`, i32, rank-1  ← FIRST FAILURE
Lines 90, 121 (`tl.min(v_idx_selected, axis=0)`). Pre-pass classifier
(`TritonGPUToMetal.cpp:8484`) rejects: *"reduce combine requires Session L3c
(future) — got arith.minsi"*.
- Metal dialect has `maxOp` (enum val 13) but **no `minOp`**.
- Scalar `arith.minsi/maxsi` (the `tl.minimum/maximum` clamps at lines 97-98,
  127-129) ALREADY lower via the translator value emitter
  (`ModuleTranslation.cpp:4654`). Only the min-**reduce** combine is missing.

**Fix (Phase 1, self-contained, independently testable + committable):**
1. `MetalOps.td:~380` — add `BinaryExpOperatorMin : I64EnumAttrCase<"minOp", 14>`
   to the `BinaryExpOperator` enum (+ MetalOps.cpp verify/fold switches if any
   enumerate cases exhaustively).
2. `ModuleTranslation.cpp translate(BinaryExpOp)` (~4834): add a `minOp`
   function-call arm emitting `min(ty(a), ty(b))`, mirroring the `maxOp` arm at
   ~4845 (the infix switch at ~4897 should `llvm_unreachable` for minOp too).
3. Pre-pass classifier (`:8475`-`:8484`): add an `arith.minsi` arm mirroring
   `arith.maxsi` (rank-1 gate, i32 width-32).
4. `lowerRank1Reduce` (~:2043): add `isMinI = isa<arith::MinSIOp>`; wire into the
   `isI32 && !(isAddI || isMaxI || isMinI)` guard, `emitCombine` (minOp branch),
   `buildIdentityVal` (identity = `INT32_MAX` for signed min), `storeTy = si32`.
   Optionally add f32 min (`arith.minnumf/minimumf`, identity `+FLT_MAX`) for
   symmetry — NOT needed by this kernel (its mins are all i32) but cheap.
5. Tests: lit `rank1_reduce_minsi_block32.mlir` (mirror the maxsi/maxf fixtures);
   pytest add `min` to `test_metal_backend_reduce_rank1.py`'s dtype×combiner grid.

### W-B — rich rank-1 reduce cone evaluator  ← next after W-A
The kernel's rank-1 reduces (`tl.sum`, `tl.min`) are over COMPUTED cones:
`select`, `cmpf`, `cmpi`, `andi`, `make_range`, `splat`, masked `load`, and the
SCAN result. The current rank-1 walker `walkBackThroughElementwiseChain`
(`:1831`) accepts ONLY `{addf,subf,mulf,divf,math.exp}` + a single-`splat`
binary + a `tt.load` terminator. It rejects `select`/`cmp`/`andi`/`make_range`
and any load-less cone. So even the f32 SUM reduces here fail today (they'd hit
"unsupported producer: arith.select").

The **rank-2** evaluators (`evalRank1ValueAt` :3033+, `evalRank2ConeAt` :3329+)
already handle make_range→per-element index, select, cmpi, cmpf, broadcast,
expand_dims, multi-tensor binary, masked loads, and staged leaves. W-B = give
the rank-1 reduce the same richness.

**Design decision (to make during Phase 2 spike):** either
(a) generalize `walkBackThroughElementwiseChain`/`emitScalarChain` to the full
op set with per-element index derivation (make_range → `iv*tpb + localTid` under
spt1Direct), or (b) route B2.3 spt1Direct computed cones through a rank-1
adaptation of `evalRank1ValueAt`. (b) reuses more, is preferred pending spike.
Must handle: make_range, splat, cmpf/cmpi (all predicates), andi/ori (i1),
select, addf/subf/mulf/divf, masked load (mask = `cmpi slt make_range splat`,
other = const), and a **staged-leaf hook** for the scan result (W-D).

### W-C — `tt.scan` (cumsum), rank-1 f32, addf  ← the centerpiece, highest risk
Lines 85, 116 (`tl.cumsum(x, axis=0)`). **Zero infrastructure** — `tt.scan` is
only named as a region-op to *skip* (`:8592`). Unlike reduce (tensor→scalar),
scan produces a **distributed rank-1 tensor** (element i = Σ elems[0..i]) that
downstream elementwise ops AND a reduce consume.

Required: a `ScanLowering` (or a `lowerRank1Scan` helper) that, for BLOCK=1024,
tpb=128, spt=1, E=8:
1. Each thread loads/derives its 8 owned elements (cyclic: `elem = iv*tpb + t`).
2. Per-thread local prefix over the 8 (serial).
3. Cross-thread exclusive-prefix of per-thread totals via threadgroup scratch +
   Hillis-Steele (log2(128)=7 steps, barriered) — reuse the butterfly scratch
   pattern from `lowerRank1Reduce`.
4. Add each thread's exclusive-prefix offset to its 8 local-prefix values.
5. **Materialize the result as a per-thread tensor** so downstream tensor ops
   read element (t,iv). This is the NEW dataflow shape: a cross-thread op whose
   result is a live distributed tensor, not a broadcast scalar.

Note the cyclic layout means element ordering across threads is
`t=0..127` at position `iv*128 + t`; the prefix must respect the **logical**
0..1023 order, i.e. within iv-block `iv`, positions `iv*128 .. iv*128+127`.
Careful index math (Blelloch over the flattened logical order) required.

**De-risk FIRST (Phase 2 spike, adder-Phase-0 style):** hand-write the
parallel-prefix MSL + a `torch.mps.compile_shader` driver; prove bit-exact vs
`torch.cumsum` for (BLOCK=1024, tpb=128) AND that the distributed result can be
re-read elementwise. No backend code until the spike is green.

### W-D — scan result as a reduce/elementwise leaf
After the scan: `total = splat(running_sum) + chunk_cumsum` → `cmpf oge` →
`andi v_mask` → `select` → **min-reduce**. So the scan's distributed tensor
feeds W-B's cone evaluator. The evaluator must treat a scan result as a
**staged leaf**: read back element (t,iv) from wherever the scan wrote it
(threadgroup scratch or per-thread registers). Mirrors adder Inc2.5 staged
leaves (`g_stagedLeaves` DenseMap). Depends on W-B + W-C.

### W-E — control-flow verification (likely mostly-supported; verify each)
- `cf.cond_br` early return (`if pid>=B: return`, `:0`→`^bb1/^bb2`) —
  **unstructured CFG**. Prior kernels used scf only. Verify it lowers; if not,
  add a `cf.cond_br`/`cf.br` translator arm or a canonicalize-to-scf.
- `scf.for iter_args(%accepted_all = i1)` — loop-carried **i1**. Multi-scalar
  iter_arg emitter `_scfForIterArgsMulti` handles f32/i32; verify/extend for i1.
- `scf.if -> i1` and `scf.if -> tensor<1024xf32>` (the `is_uniform` branch,
  yields a full tensor) — adder had tensor-yielding scf.if; likely OK, verify.
- `scf.for iter_args(f32, i32)` mixed-type — verify `_scfForIterArgsMulti`
  handles heterogeneous scalar types.

### W-F — `arith.andi` on i1 tensor as a standalone cone op
`%cond_32 = arith.andi %cond_31, %v_mask_29 : tensor<1024xi1>`. The rank-2
evaluator handles cmpi/cmpf/select but confirm `andi`/`ori` on i1 tensors are
in the cone op set; add if missing (folds into W-B).

## Phasing & sequencing

| Phase | Scope | Risk | Est. | Committable alone |
|-------|-------|------|------|-------------------|
| **1** | W-A min-reduce (dialect minOp + translator + classifier + lowering + tests) | Low | 0.5–1 d | ✅ yes |
| **2** | W-C scan **spike** (hand MSL parallel-prefix vs torch, no backend code) | Med | 1–2 d | n/a (spike) |
| **3** | W-B rich rank-1 cone evaluator (unblocks the f32 sum reduces + min-reduce cones) | Med-High | 3–6 d | partial (sum-over-computed-cone tests) |
| **4** | W-C scan lowering (build on Phase-2 spike) | High | 5–10 d | ✅ (cumsum unit tests) |
| **5** | W-D scan-result staged leaf + W-F andi | Med | 2–4 d | with Phase 4 |
| **6** | W-E control-flow verification + fixes | Low-Med | 1–3 d | — |
| **7** | End-to-end: verbatim kernel compiles + runs bit-exact; durable pytest | — | 1–2 d | ✅ |

**Total realistic: ~3–5 weeks.** This is comparable to or larger than the adder
project. The scan (W-C) is the dominant unknown; Phase 2 must de-risk it before
Phases 4-5 are committed.

## Verification protocol (per phase)
- Build: `ninja -C build/cmake.macosx-11.0-arm64-cpython-3.12` (native changes).
- Regression: full Metal pytest suite (`test_metal_backend_*.py`) must stay green
  (currently 301 passed) + Metal lit (`test/Dialect/Metal/**`, 106/106).
- New: per-phase lit fixture(s) + pytest, tight tolerance (1e-3 sum / 1e-5 max /
  exact int).
- End-to-end: `scratchpad/run_specdec.py` vs a numpy/torch reference solve.

## Start point
Phase 1 (min-reduce) begins now — low-risk, self-contained, independently
valuable (`tl.min` rank-1 is generally useful), and independently testable via a
`tl.min(tl.load(...), axis=0)` unit kernel that does NOT need the scan.
