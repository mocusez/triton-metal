# Metal backend: `num_stages` software pipelining + compile-option contract layer

**Status:** plan / not started
**Scope chosen:** full `num_stages` pipelining (most ambitious tier) + the honest
contract layer underneath it.

---

## 0. TL;DR and the one decision that dominates everything

`num_stages` is currently a **no-op that lands only in the cache key**
(`compiler.py:50` hash, never consumed by any pass — `make_ttgir`,
`compiler.py:148`, runs no pipeliner). Making it *actually* pipeline on Apple
GPUs is not a port of the CUDA pipeliner:

- The CUDA pipeliner overlaps `cp.async` global→shared copies across iterations
  using multi-buffered shared memory + commit/wait groups.
- **Apple GPUs have no `cp.async` analog.** MSL loads are synchronous; latency is
  hidden by *occupancy* (many resident simdgroups), not by software-managed async
  double-buffering.
- The only realizable form on Metal is **SSA-level software pipelining**
  (register / `iter_arg` rotation): issue iteration *i+k*'s load early in program
  order so its latency overlaps iteration *i*'s compute. This is what MLIR's
  `mlir::scf::pipelineForLoop` does natively, *without* async.

**Whether that helps at all on Apple silicon is unproven.** So Phase 0 is a
hard gate: hand-write a prefetched kernel, benchmark on MPS, and only build the
general machinery if there is a measurable win. If not, we ship the contract
layer (which is unconditionally correct) and `num_stages` becomes an
*explicitly-warned* no-op instead of a *silent* one.

---

## 1. Goal / non-goals

**Goal**
- No compile option is ever *silently* dropped. Every field of `MetalOptions` is
  classified `honored` / `ignored-with-warning` / `rejected`, enforced in
  `parse_options`.
- `num_stages > 1` produces a genuinely pipelined loop (prologue + rotated
  steady-state + epilogue) **iff** Phase 0 proves a win; otherwise it warns.

**Non-goals**
- No shared-memory multi-buffering / async-copy emulation (no HW primitive).
- No auto-tuning of `num_stages`; we honor what the user/`@triton.jit` passes.
- `warp_size` stays fixed at 32 (Apple SIMD width, `driver.py:318`) — not a knob.
- No fp8 / cooperative-grid / maxnreg codegen (those stay rejected-or-warned).

---

## 2. Current-state anchors (verified)

| Fact | Location |
|---|---|
| `parse_options` silently filters to known fields, drops the rest | `backend/compiler.py:115` |
| `num_stages` only in hash, consumed by no pass | `backend/compiler.py:50`, `:148` |
| `make_ttgir` runs no pipeliner/scheduling pass | `backend/compiler.py:148` |
| `scf.for` emitter supports **single f32/i32 iter_arg only** | `ModuleTranslation.cpp:808`, `:824` |
| Multi-iter_arg loop → explicit no-op (the gating gap) | `ModuleTranslation.cpp:894` |
| `scf.yield` writes back exactly one iter_arg temp | `ModuleTranslation.cpp:887` |
| Loads inside loops emit as synchronous MSL (`simdgroup_load`, strided) | `ModuleTranslation.cpp:561`, `:504` |
| `torch.mps.compile_shader(src)` takes source only — no compile-flags surface | `backend/driver.py:175` |

---

## 3. Architecture decision

Do the transform as an **MLIR pass on TTGIR (or the metal dialect) using
`mlir::scf::pipelineForLoop`**, not as ad-hoc text rewriting in the emitter.

- The pass restructures `scf.for` at the SSA level: it creates the prologue,
  emits the steady-state kernel with loads scheduled `num_stages-1` iterations
  ahead, rotates the prefetched values through **new `iter_args`**, and emits the
  epilogue. Output is still standard `scf.for`.
- Consequence: the emitter must learn to translate a **multi-iter_arg
  `scf.for`** (Phase 2). That is the real prerequisite and the bulk of the
  emitter work — pipelining without it cannot be printed.
- Keep it target-gated the same way `add_propagate_coalesced_layouts` is
  (`compiler.py:171`): a no-op unless the Metal pipeline invokes it, so upstream
  and other backends are untouched.

---

## 4. Phases

### Phase 0 — Feasibility & perf spike **(GATE) — DONE 2026-07-13. VERDICT: FAIL.**

Method: hand-wrote baseline vs register-prefetched MSL for a streaming f32
reduction (the *most* memory-latency-sensitive loop shape), compiled both via
`torch.mps.compile_shader`, A/B-benchmarked on MPS (N=64M floats / 268 MB)
across occupancy regimes. Scripts: `scratchpad/phase0_prefetch_bench.py`,
`scratchpad/phase0_bottleneck.py`.

Data (median ms, speedup vs `base`):

| threads | base | pf2 (ns=2) | pf4 (ns=4) | macc4 | macc8 |
|--------:|-----:|-----------:|-----------:|------:|------:|
| 256     | 58.4 | **1.00x**  | 0.98x      | 3.42x | 5.39x |
| 1024    | 15.5 | **0.99x**  | 1.00x      | 2.99x | 4.59x |
| 4096    | 4.43 | **0.99x**  | 0.97x      | 1.51x | 1.51x |
| 16384   | 3.34 | **0.95x**  | 0.97x      | 1.05x | 1.02x |

**Conclusion — register-rotation `num_stages` pipelining buys nothing on Apple MPS:**
1. Memory prefetch is flat **0.95–1.02x** at every occupancy level, incl. the
   low-occupancy latency-bound regime where it theoretically should help most.
   Deeper prefetch (`pf4`) trends *negative* (register pressure).
2. The real low-occupancy bottleneck is the **serial FADD accumulator chain**,
   not memory latency: breaking it with independent accumulators (`macc`) gives
   **3–5x**, while `macc4_pf` (prefetch on top) adds nothing over `macc4`.
3. The Metal/AIR compiler already schedules independent loads ahead, so
   source-level prefetch is redundant. The streaming reduction is the best case
   for prefetch; matmul k-loops (compute-bound) would benefit even less.

**Decision: STOP at Phase 1.** `num_stages>1` becomes an explicitly-warned
no-op, not a silent one. Do NOT build Phases 2–4 (multi-week, zero runtime
benefit, +register pressure, +cache-key churn).

**Spin-off finding (separate feature, not num_stages):** multi-accumulator
reduction gives 3–5x on low-occupancy reduce kernels. Worth its own spec — it
maps to reduction-unrolling / partial-accumulator codegen, not pipelining.

### Phase 1 — Compile-option contract layer **— DONE 2026-07-13.**

Landed in `third_party/metal/backend/compiler.py`: `_enforce_option_contract`
classifies all 19 `MetalOptions` fields (import-time assertion forbids silent
fall-through of future fields), called from `parse_options`. WARN (once/field,
deduped) on unhonorable non-default values — `num_stages`, `num_ctas`,
`enable_fp_fusion=False`, `launch_cooperative_grid`, `maxnreg`, `extern_libs`,
`max_num_imprecise_acc_default`, `instrumentation_mode`; REJECT `warp_size!=32`;
`TRITON_METAL_STRICT_OPTIONS=1` raises on unknown keys. Frontend-honored
`sanitize_overflow` correctly left quiet.

Tests: `python/test/unit/test_metal_backend_options_contract.py` (15, all pass).
Full suite **244 passed, no regressions**; the autotune smoke test that passes
`num_stages=3` now emits one honest warning instead of silently dropping it.

*Original design below (for reference).*

Replace the silent filter in `parse_options` with an explicit classification.

- Introduce a table in `compiler.py`:
  ```
  _OPTION_POLICY = {
     "num_warps": HONORED, "num_ctas": HONORED, "arch": HONORED,
     "default_dot_input_precision": HONORED, ...
     "num_stages": WARN_IF_NONDEFAULT,      # until/unless Phase 3 lands
     "enable_fp_fusion": WARN_IF_NONDEFAULT,
     "maxnreg": WARN_IF_NONDEFAULT, "extern_libs": WARN_IF_SET,
     "launch_cooperative_grid": WARN_IF_TRUE, ...
     "warp_size": REJECT_IF_NOT_32,
  }
  ```
- On a non-default value for a non-honored field, emit **one** `warnings.warn`
  (deduped) naming the field and that Metal ignores it. `REJECT` raises
  `ValueError` with a clear message.
- Unknown keys: keep dropping (Triton core passes some backends don't share) but
  gate behind `TRITON_METAL_STRICT_OPTIONS=1` to raise for debugging.
- Tests: `test_metal_options_contract.py` — assert warn/raise per policy.

This is the honest core and is worth landing on its own.

### Phase 2 — Emitter: general multi-`iter_arg` `scf.for`
Generalize `translate(scf::ForOp)` (`ModuleTranslation.cpp:808`) and
`translate(scf::YieldOp)` (`:873`) from 1 iter_arg to N.

- Declare one temp per iter_arg (`_scfForIterArgs[op] = [idx0, idx1, ...]`),
  each with its real element type via `typeToString`.
- Map every region iter-arg BlockArg **and** every `op.getResult(i)` to its temp.
- `scf.yield` writes back all N operands in order.
- Preserve byte-identical output for the existing single-iter_arg path (guard on
  N==1 to reuse the current code path, or prove the generalized path emits the
  same text — lit tests must not churn).
- Handle iter_arg element types beyond f32/i32 if pipelining introduces them
  (loaded vector/tile values may need a small struct or per-lane temp — resolve
  concretely from the Phase 0 spike's IR).
- Tests: extend lit (`test/Dialect/Metal/.../`) with a 2-iter_arg loop.

### Phase 3 — Pipelining pass
- New pass `TritonMetalPipelineLoops` (metal `lib/Dialect/Metal/Transforms/`),
  registered for the `triton-metal-opt` CLI and invoked from `make_ttgir` under a
  `metal:` gate.
- Use `mlir::scf::pipelineForLoop` with a `PipeliningOption`:
  - `getScheduleFn`: place `tt.load` / `metal.simdgroup_load` ops at stage 0 and
    their compute consumers at stage `num_stages-1` (load-ahead distance =
    `num_stages-1`).
  - No predication/async hooks (synchronous loads; the util's default
    prologue/epilogue peeling handles boundaries).
- Only fire when: loop trip count is static & > `num_stages`, body has a
  hoistable load feeding compute, and `num_stages > 1`. Otherwise leave the loop
  untouched (and Phase 1 already warned if it couldn't fire).
- Tests: lit checking the produced prologue + rotated steady-state IR.

### Phase 4 — Wire-through + end-to-end
- `make_ttgir` (`compiler.py:148`): add `passes.ttgpuir.add_metal_pipeline_loops(pm, options.num_stages)`
  after coalesce/remove-layout, gated so `num_stages==1` is a no-op.
- Flip `num_stages` in `_OPTION_POLICY` from `WARN` to `HONORED`.
- `hash()` already includes `num_stages` — keep it (correct cache-key behavior).
- Register the pybind/pass plumbing (`triton_metal.cc`) if the pass must be
  reachable from `ttgir_to_msl`.
- e2e pytest: `test_metal_num_stages_pipelines` — same kernel at `num_stages`
  1 vs 2 produces numerically identical results; MSL text shows the prologue
  (guard behind the Phase 0 verdict).

---

## 5. Testing strategy
- **Correctness first:** every pipelined kernel must be bit-for-bit equal to its
  `num_stages=1` counterpart (`_metal_oracle.py` / existing reduce tests).
- **lit** for the IR transform (Phase 2 emitter, Phase 3 pass) — no GPU needed.
- **pytest** on MPS for the numeric + perf e2e (Phase 0, Phase 4).
- Rebuild rule: Phases 2–4 touch native code → `ninja -C build/cmake.macosx-11.0-arm64-cpython-3.12`
  before pytest (per project memory; plain `make` picks the wrong Python).

---

## 6. Risks & fallbacks
| Risk | Mitigation |
|---|---|
| No perf win on Apple GPU (likely) | Phase 0 gate; fall back to warned no-op (Phase 1 still ships) |
| Register pressure from prefetch lowers occupancy → regression | measure occupancy in Phase 0; cap the loads we hoist |
| Multi-iter_arg emitter churns existing lit output | N==1 fast-path preserves byte-identical emission |
| `scf.pipelineForLoop` needs types the emitter can't print (vectors/tiles) | resolve from Phase 0 IR; extend `typeToString` or keep scalar-only |
| Loop bounds dynamic / trip count ≤ stages | pass declines to fire; Phase 1 warns |

## 7. Effort estimate
- Phase 0: 2–3 days (gate).
- Phase 1: 0.5–1 day (independent, high-value; land first).
- Phase 2: 3–5 days (emitter generalization + no-churn proof).
- Phase 3: 4–7 days (pass + schedule + lit).
- Phase 4: 2–3 days (wire + e2e).
- **Total if Phase 0 passes: ~3 weeks.** If Phase 0 fails: **~1 day** (Phase 1
  only), and `num_stages` is honestly documented as unsupported.
