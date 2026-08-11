# Metal backend — `medium-attention_with_sinks.py`

Plan for running `leet-triton/medium-attention_with_sinks.py` verbatim on the
Metal backend. Written 2026-08-11 against `metal-develop` @ `211a41b530`.

**STATUS: DONE (2026-08-11).** All phases landed; the kernel runs verbatim,
`<= 6.6e-07` against a float64 reference. What the plan did not anticipate is in
§5 — three of the four surprises came from Triton's own IR canonicalization, not
from the attention math.

Sibling docs: `metal-flash-attention-plan.md`,
`metal-sliding-window-attention-plan.md`. Read §1 of the latter first — the
failure mode it documents (a matcher that claims a kernel on op COUNTS and then
silently miscompiles it) is the reason this plan builds a *new* op instead of
widening `metal.flash_attention`.

---

## §0 Current status: hard compile error

```
RuntimeError: Metal backend: convert-tritongpu-to-metal failed
```

raised from `compiler.py:307 make_msl`, for every shape. Not a wrong answer — a
refusal. Measured on `M=64/d=16/sinks=4/win=32` and `M=256/d=16/sinks=4/win=128`.

`TRITON_METAL_FA_DEBUG=1` prints **nothing**, which pins the rejection to one of
the structural gates (1)–(7) of `tryFlashAttentionLoop`, before the template
verifier. The gate is (3):

```cpp
// (3) classify: dotA (Q@K^T) is the one whose B-operand cone has a tt.trans.
if (faConeHasTrans(dots[0].getB()) && !faConeHasTrans(dots[1].getB())) { ... }
else if (...) { ... } else return mlir::failure();
```

This kernel never calls `tl.trans`. It loads K already transposed —

```python
k_local = tl.load(K + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_km, ...)
```

— so the TTGIR has `tt.trans` count 0 and neither dot can be classified. After
the FA matcher declines, the loop falls through to the general path, whose cvt
classifier rejects 7 `ttg.convert_layout` ops with the L1d3 "broader
staged-transpose deferred" error. That is the error the user sees.

## §1 What the kernel actually computes

For query row `i` and key `j`, with `S = num_sinks`, `W = window_size`:

```
keep(i, j)  =  j <= i  AND  ( j < S  OR  j >= i - W + 1 )
out[i]      =  softmax_j( keep ? q_i·k_j * sm_scale : -inf ) @ V     (base 2)
```

Causal, with the first `S` tokens ("attention sinks") always visible plus a
one-sided sliding window of width `W`. `sm_scale = log2(e)/sqrt(d)` is computed
**on the host** and passed as an f32 kernel argument; the kernel uses `exp2`
throughout, so the `log2(e)` folds into the scale.

The kernel evaluates this in two phases, and the phase structure — not the
mathematical mask — is what the emitter must reproduce (see §3.3):

* **sink phase**, straight-line before the loop: keys `arange(0, BLOCK_S)`,
  predicate `s < S && s <= i`.
* **local phase**, `N_LOCAL_BLOCKS` blocks of `BLOCK_N`: keys
  `local_start + b*BLOCK_N + arange(0, BLOCK_N)` where
  `local_start = max(pid*BLOCK_M - W + 1, S)`, predicate
  `n < M && n <= i && n >= i - W + 1 && n >= S`.

`N_LOCAL_BLOCKS = cdiv(W + BLOCK_M - 1, BLOCK_N)` is a **host-side constexpr**.
The kernel is only correct when the host's `window_size` matches the
`N_LOCAL_BLOCKS` it was specialized for; a larger runtime `W` silently drops
keys. Reproducing the phase structure rather than the intended mask means the
emitted kernel agrees with the source kernel *including* on that edge, instead
of quietly disagreeing with it.

## §2 Why not widen `metal.flash_attention`

Every one of the following is a mismatch against the existing op, and each would
have to be threaded through the shared matcher + the shared emitter:

| # | Kernel | `metal.flash_attention` today |
|---|--------|-------------------------------|
| 1 | K loaded `[D, N]` by strides | requires a `tt.trans` on dot B |
| 2 | `exp2` + runtime f32 `sm_scale` arg | `exp` + emitter-computed `1/sqrt(d_head)` |
| 3 | sink block **outside** the loop feeding its iter_args | loop body is the whole attention; inits must be `0/0/-inf` constants |
| 4 | `m_i = where(row<M, -inf, 0)`, `l_i = where(row<M, 0, 1)` | constant inits |
| 5 | causal + sink + one-sided window | full, or symmetric band `abs(row-key) <= win` |
| 6 | 4 separate row-stride args, `d` a separate arg | one `d_model` serving as row stride *and* feature width |
| 7 | trip count is a constexpr; `N_LOCAL_BLOCKS == 1` unrolls the loop away | loop-anchored (`scf.for` required) |

Item 7 alone breaks the anchor: with `window_size=32, BLOCK_M=32, BLOCK_N=64`
the trip count is 1 and Triton deletes the `scf.for`, so a loop-anchored matcher
can never fire. Item 3 breaks the coverage gate, which is the safety property
that stops the existing matcher from over-claiming.

Widening the FA op to cover all seven means editing the code path that two
already-working kernels (`hard-mult_head_attention.py`,
`hard-sliding_window_self_attention.py`) depend on, for zero benefit to them.
**Decision: a new `metal.sink_attention` op with its own matcher and its own
emitter.** The existing FA path is not touched, so its regression risk is zero.

## §3 Design

### 3.1 The op

```tablegen
def SinkAttentionOp : Metal_Op<"sink_attention", [AttrSizedOperandSegments]> {
  let arguments = (ins
      Metal_MemRefType:$q, Metal_MemRefType:$k,
      Metal_MemRefType:$v, Metal_MemRefType:$out,
      Metal_MemRefType:$m,          // query row count == key count (ui32)
      Metal_MemRefType:$d_head,     // feature width (ui32)
      Metal_MemRefType:$scale,      // f32 logit scale, log2(e)/sqrt(d) folded in
      Metal_MemRefType:$stride_q, Metal_MemRefType:$stride_k,
      Metal_MemRefType:$stride_v, Metal_MemRefType:$stride_o,
      Optional<Metal_MemRefType>:$num_sinks,   // ui32, or $sinks_const
      Optional<Metal_MemRefType>:$window,      // ui32, or $window_const
      I64Attr:$bm,          // query rows per threadgroup, <= 32
      I64Attr:$bd,          // padded feature tile, >= d_head
      I64Attr:$bs,          // sink phase width (BLOCK_S)
      I64Attr:$local_len,   // N_LOCAL_BLOCKS * BLOCK_N
      OptionalAttr<I64Attr>:$sinks_const,
      OptionalAttr<I64Attr>:$window_const);
}
```

`num_sinks` / `window` are optional-with-constant-fallback for the same reason
`window` is on the FA op: Triton drops a kernel argument equal to 1 from the
signature and folds it into a `dense<1>` constant, leaving no buffer to point
at (`metal-sliding-window-attention-plan.md` §1b).

Threadgroup budget: `2*bm*bd + 2*bm` floats must fit 8192 (32 KiB). With
`bm = 32` that caps `bd` at 64 — the same practical head-width ceiling the FA op
already documents.

### 3.2 Emitter: one query row per lane, scalar inner product

The FA op stages Q/K^T/V/P tiles in threadgroup memory and runs both matmuls on
simdgroup hardware. That costs `3*bm*bd + 2*bd*bn + 2*bm*bn + 2*bm` floats, which
blows the 32 KiB budget at `bd = 32` for this kernel's `bn = 64`. It also needs
the key set to be a contiguous block range, which the two-phase mask is not.

So `metal.sink_attention` emits a **scalar** body: one query row per lane, keys
walked one at a time, online-softmax state updated per key.

```
threadgroup float _sa_qbuf[bm*bd];   // Q tile, staged once
threadgroup float _sa_obuf[bm*bd];   // O accumulator
threadgroup float _sa_rmax[bm], _sa_rsum[bm];

stage Q + zero obuf/rsum, rmax = -INFINITY;  threadgroup_barrier
q = lane; row = tgid.x*bm + q;  if (q < bm && row < M) {
  for (int s = 0; s < bs; ++s)  if (s < S && s <= row)  STEP(s);
  int lstart = max((int)(tgid.x*bm) - W + 1, S);
  for (int t = 0; t < local_len; ++t) { int n = lstart + t;
    if (n < M && n <= row && n >= row - W + 1 && n >= S)  STEP(n); }
  for (uint d = 0; d < dh; ++d) out[row*so + d] = _sa_obuf[q*bd + d] / _sa_rsum[q];
}

STEP(key):
  float a = 0; for (d < dh) a += _sa_qbuf[q*bd+d] * k[key*sk + d];
  float s = a * scale;
  float m_old = _sa_rmax[q], m_new = max(m_old, s);
  float sc = (m_old == m_new) ? 1.0f : exp2(m_old - m_new);
  float p  = exp2(s - m_new);
  _sa_rsum[q] = _sa_rsum[q]*sc + p;  _sa_rmax[q] = m_new;
  for (d < dh) _sa_obuf[q*bd+d] = _sa_obuf[q*bd+d]*sc + p*v[key*sv + d];
```

Notes that matter:

* After the single post-staging barrier every lane touches only its own row, so
  there is no further synchronization and no WAR hazard.
* The `(m_old == m_new) ? 1` guard is the same one the FA emitter carries: with
  a window a row can reach a key whose logit is `-inf`-adjacent, and
  `exp2(-inf - -inf)` is `NaN`.
* The epilogue divides plainly rather than testing `denom != 0`. A row with no
  visible key produces `0/0 = NaN` — which is exactly what the source kernel
  produces. Faithfulness beats politeness here.
* Cost is `2x` the FMA work of a block-max formulation (the accumulator is
  rescaled per key, not per block) and gives up simdgroup throughput entirely.
  This is a correctness-first body; a simdgroup upgrade is future work and is
  *not* in scope. Perf is measured in Phase 4 and recorded, not optimized.
* Numerics: per-key online softmax is not bit-identical to the source's
  per-block form (different summation order), but both are exact softmax up to
  rounding. Target `<= 1e-6` max abs error vs a float64 reference, same bar the
  sliding-window work landed at.

### 3.3 Matcher: a chain of merge steps, anchored on the store

`tryFlashAttentionLoop` anchors on `scf.for`. That cannot work here (§2 item 7),
so `trySinkAttention` anchors on the **unique `tt.store`** and walks backwards:

```
epilogue:   store(acc / lift(sum), mask=(row<M)&(d<dh))
            |
   walkMergeChain(acc, sum, max):
     - defined by an scf.for result   -> verify the body as ONE merge step whose
                                         key base is `local_start + iv*BN`,
                                         trip count NLB constant; recurse on the
                                         loop's init args
     - defined in the entry block     -> verify as ONE straight-line merge step;
                                         recurse on its inputs
     - the base inits                 -> acc == 0, sum == where(row<M, 0, 1),
                                         max == where(row<M, -inf, 0): STOP
```

A merge step is the role walk the FA template already does, minus the loop:

```
S      = dot(q, k_load)              # k_load is [BD, BNx], no tt.trans
Sm     = select(valid, S * splat(sm_scale), -inf)
m_new  = maxnumf(m_carried, reduce_max(Sm, 1))
alpha  = exp2(m_carried - m_new)
P      = exp2(Sm - lift(m_new))
sum'   = sum_carried * alpha + reduce_add(P, 1)
acc'   = dot(P, v_load, acc_carried * lift(alpha))
```

with `valid` a conjunction the matcher must classify term by term. The index
vocabulary is `Row | KeySink | KeyLocal | Feat`, and the accepted terms are

| step | required mask set (exactly) |
|------|------------------------------|
| sink | `row < M`, `s < S`, `s <= row` |
| local | `row < M`, `n < M`, `n <= row`, `n >= row - W + 1`, `n >= S` |

Then the chain as a whole must be: exactly one sink step, followed by local
steps whose key bases are `local_start + c*BN` for `c = 0 .. NLB-1` contiguous
(one `scf.for` with trip count NLB, or NLB unrolled straight-line steps, or a
mix). `local_len = NLB * BN`.

**Coverage.** The FA verifier's gate is "every op in the loop body is claimed".
That does not transfer, because the prologue shares the entry block with all the
addressing and program-id arithmetic. The gate here is instead: every op in the
**backward slice** of `{store value, store ptr, store mask, every loop init,
every loop yield operand}` must be claimed, plus the existing per-block gate for
any loop body. Same property (nothing unrecognized contributes to the result),
computed over the right region.

### 3.4 Where it runs in the pass

`runSinkAttentionMatcher(moduleOp)` goes immediately **after**
`runFlashAttentionMatcher` in `ConvertTritonGPUToMetalPass::runOnOperation`, for
the same reason: before `preprocessDotCvtChains` and the cvt legality walk, so
the dot-operand convert_layouts never reach the L1d3 reject. It DCEs the loop,
the prologue, and the epilogue exactly as `tryFlashAttentionLoop` does.

Ordering against the FA matcher matters: FA runs first and this kernel is
invisible to it (gate 3), so there is no contention. The reverse order would
also work today but is not relied upon.

## §4 Phases

Each phase ends with a green gate; nothing proceeds on a red one.

### Phase 0 — MSL spike (de-risk the emitter before the compiler work)

`metal-attention-with-sinks-phase0-spike.py`: the §3.2 body hand-written as MSL,
run through `torch.mps.compile_shader`, compared against a torch reference of
§1. Sweep `M ∈ {33, 64, 100, 256}`, `d ∈ {16, 32, 64}`, `S ∈ {0, 1, 4, 16}`,
`W ∈ {16, 32, 128}`.
**Gate:** max abs error `<= 1e-6` on every case.

### Phase 1 — op + emitter

`MetalOps.td`: `SinkAttentionOp` per §3.1. `ModuleTranslation.{h,cpp}`:
`translate(SinkAttentionOp)` emitting the Phase-0-validated body; register it in
the two op lists (`usesThreadgroupId` walk + the two dispatch switches at
`:348` / `:395` — the op needs `tgid`).
**Gate:** builds; a hand-written lit test in `test/Conversion/` renders the
expected MSL; `lit` suite green.

### Phase 2 — matcher, looped form

`trySinkAttention` + `SinkTemplate` per §3.3, handling the `scf.for` form
(`N_LOCAL_BLOCKS >= 2`). Wired in per §3.4.
**Gate:** `M=256, d=16, S=4, W=128` (NLB=3) runs and matches the reference;
`TRITON_METAL_SINK_DEBUG=1` explains any rejection.

### Phase 3 — matcher, unrolled form

Straight-line local steps (`N_LOCAL_BLOCKS == 1`, and any number of unrolled
copies).
**Gate:** `M=64, d=16, S=4, W=32` (NLB=1) runs and matches.

### Phase 4 — tests + regression

`python/test/unit/test_metal_backend_attention_with_sinks.py`: the kernel
verbatim (copied, as the sibling tests do), a config sweep, and a
**negative test** — a near-miss variant (e.g. the sink predicate dropped) must
NOT be claimed, i.e. must raise rather than silently compute full attention.
**Gate:** full `pytest python/test/unit` and `lit` at or above the recorded
baseline (pytest 667 / lit 388-of-389, the one failure pre-existing upstream);
`pixi run python leet-triton/medium-attention_with_sinks.py`-equivalent driver
green across the Phase-0 sweep.

## §5 Outcome — what actually happened

All five phases landed. Phases 2 and 3 merged: once the chain walk existed, the
unrolled form cost nothing extra, because a straight-line local step and a loop
step differ only in where the key base comes from.

Results:

* `leet-triton/medium-attention_with_sinks.py` runs verbatim. Max abs error vs a
  float64 reference over `M in {32,33,64,100,256} x d in {16,32,64} x
  (S,W) in {(4,32),(16,128),(2,16)}`: `<= 6.6e-07`.
* `python/test/unit/test_metal_backend_attention_with_sinks.py`: 53 tests,
  including the near-miss rejection and the `bd > 64` budget rejection.
* `test/Dialect/Metal/metal-translate/sink_attention_emit.mlir` pins the emitted
  MSL for both the buffer and the folded-constant spelling of the two counts.
* Regression: lit 387 passed / 1 failed (`TritonGPU/coalesce-propagate-reduce.mlir`,
  pre-existing upstream) of 390. pytest 737 passed across the metal test files run
  one process each; the only failing files are CUDA-only
  (`test_debug`, `test_debug_dump`, `test_debuginfo`, `test_knobs::test_nvidia_tool`,
  `test_link`), all pre-existing.

Four things the plan got wrong, all worth remembering:

1. **`BLOCK_S == BLOCK_D` makes the sink key and the feature index the SAME SSA
   value.** Triton CSEs `arange(BLOCK_S)` and `arange(BLOCK_D)` when both are 16,
   so a classifier keyed on the producer cannot tell `s < num_sinks` from
   `d < d_head`. The matcher classifies indices by the `tt.expand_dims` axis
   instead — and the axis can be attached at either end: address arithmetic lifts
   the index (`expand_dims(offs_n, 0)` then multiply), while a mask usually
   compares rank-1 values and lifts the i1 RESULT. Hence the `outer` axis
   threaded through `collectTags` / `classifyCmp`.

2. **Triton duplicates whole cones per layout instead of inserting a
   `convert_layout`.** At `d = 64` the running sum multiplies
   `exp2(subf(m_i_49, m_new))` and the accumulator multiplies
   `exp2(subf(m_i_48, cvt(m_new)))` — two distinct op chains computing the same
   value, right down to duplicated `select` and `cmpi` ops. SSA-identity
   comparison rejected a perfectly good kernel; `sameCone` compares structurally
   and claims BOTH sides, which the coverage gate requires anyway. Note that
   `arith.constant` cannot be compared by attribute dictionary there — the
   attribute embeds the tensor type, which is exactly what differs.

3. **A failed match attempt must roll back EVERYTHING it wrote.** Which phase a
   straight-line step belongs to is not knowable before matching it, so the walk
   tries one and retries. The first version restored only `claimed`/`why`, so a
   probe that had already pinned `BLOCK_S` to the local width poisoned the next
   step's width check — and the error surfaced two steps later as "two different
   local block widths", pointing at the wrong thing entirely.

4. **`window_size = 1` deletes the pattern the matcher looks for.** Triton's
   equal-to-1 specialization folds `pid*BM - 1 + 1` down to `pid*BM`, so
   `local_start` loses the `- W + 1` structure the window width was recovered
   from, and `row - W + 1` collapses to `row`. Both need an explicit
   folded-constant path, same as `metal.flash_attention`'s `window_const`.

Not revisited: the scalar body's performance (§3.2 is a correctness-first body by
design). No measurement was taken, so no claim is made.

## §6 Explicitly out of scope

* simdgroup acceleration of the sink body (scalar path only; see §3.2).
* `d > 64` — rejected by the threadgroup budget, same ceiling as the FA op.
* Multi-head / 2-D grid variants of this kernel.
* Any change to `metal.flash_attention` or its matcher.
* Making the general (non-fused) rank-2 dot path work — that is the L1d3
  staged-transpose backlog item and is not touched here.
