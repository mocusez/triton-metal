# Metal backend — multi-pass tile-loop reduce: design & risk assessment

Status: DESIGN (no code yet). Author target: Wall 18+.
Prereq context: Wall 17 Inc 1+2 shipped the re-emit computed-cone reduce
(`evalRank2ConeAt` / `evalRank1ValueAt`) for **device-rooted** computed cones.
This document covers what it takes to go further — to reduces whose inputs are
NOT re-derivable and to **chained** reduces (softmax) — and ultimately the leet
`adder_transformer` kernel.

## 1. The architecture today

`FuncOpLowering` (`TritonGPUToMetal.cpp:397`) lowers a Triton kernel body to
scalar Metal ops and, when `elemPerThread (E) > 1`, wraps the **entire** body in
ONE `scf.for(0, E, 1)` "tile loop" (`:502-525`). Each scalar op operates on the
per-`(thread, iv)` element; load/store recover their element index via
`emitPerIterIndex`.

`tt.reduce` axis=1 needs a whole ROW, which spans many threads × tile
iterations. The current rank-2 reduce body (`:3516+`) sidesteps the tile loop:
it **ignores the per-thread scalar** and re-reads the row straight from the
device `tt.load`, filling a threadgroup `rowBuf[M]` that is **hoisted ABOVE the
tile loop** (`:3640` "runs once"), then reads `rowBuf[outIdx]` inside the loop.

That works only because a device `tt.load` can be re-read from anywhere. For a
COMPUTED input the value lives in registers inside the tile loop, so:
- it does **not dominate** the hoisted staging point (SSA violation), and
- a single tile loop cannot express "finish all iterations → barrier → reduce →
  feed the result back into the same loop's later ops".

This is the single architectural blocker behind Increments 2.5 (per-row scalar
staging) and 3 (chained reduces): both need the tile loop **split into passes**.

## 2. Goal

Lower kernels containing reduces over computed tiles — including chained
reduces (max → exp(x−max) → sum → x/sum) — by splitting the single tile loop
into multiple passes separated by reduce barriers, materializing the tensors
that live across a barrier in threadgroup memory.

## 3. Design

### 3.1 Phase splitting (FuncOpLowering, pre-conversion)
Partition the tile-loop body into phases `P0 … Pk`. A `tt.reduce` is a barrier:
its input cone belongs to the current phase; its result is consumed in the next.
Emit:

```
stage buffers (threadgroup)                 // sized for live-across tensors
P0 tile loop:  scf.for(0,E){ compute …; stage live-across + reduce-input }
barrier; reduce#0 from stage → resultBuf0[M]; barrier
P1 tile loop:  scf.for(0,E){ unstage …; read resultBuf0; compute …; stage … }
barrier; reduce#1 …
…
Pk tile loop:  scf.for(0,E){ … ; store outputs }
```

### 3.2 Cross-phase liveness + materialization
For each barrier, the tensor SSA values defined ≤ this phase and used > this
phase must be threadgroup-materialized (registers don't survive a separate
loop). Rank-2 `[M,N]` → `stage[M*N]` written at `flatpos(thread,iv)=row*N+col`;
rank-1 `[M]` → `[M]`. Reuse buffers across non-overlapping lifetimes.

### 3.3 Two staging strategies (pick per value)
- **Stage the tile** (`[M,N]`, 32 KiB): unconditional, but expensive in
  threadgroup memory.
- **Stage only the irreducible roots** (`[M]` per-row values from scf.if /
  loop-carried state, e.g. adder's `d`) and **re-derive** the rest with the
  Wall-17 re-emit evaluator. Far cheaper (small `[M]` buffers) and reuses
  existing machinery. Preferred where the cone is re-derivable from small roots.

### 3.4 Chained reduces
Each reduce result is a `[M]` buffer; later phases/reduces read it. Reduce #2's
cone `exp(score − max[:,None])` reads `resultBuf_max[r]`; reduce #3 reads
`resultBuf_sum[r]`. This is just "an irreducible root that happens to be a prior
reduce result" — same mechanism as 3.3.

## 4. The hard part: adder's nesting + state

adder's three reduces sit INSIDE a **user `scf.for %pos`** that carries
`next_token` as **tensor iter_args** (autoregressive state), and `score` depends
on `d = scf.if(pos<31, load, next_token)`. So the real structure is:

```
[FuncOp tile loop] > [user pos loop, tensor iter_args] > [scf.if d]
                                                        > reduce#1,2,3 (chained)
                                                        > [scf.if argmax → next_token]
```

Phase-splitting must happen **inside the user pos loop body**, around each
reduce, while that loop carries per-row state across iterations. The irreducible
root `d` is itself loop-carried (`next_token`), so even staging `d[M]` requires
its converted per-thread scalar — which lives inside the (to-be-split) tile loop.

## 5. Memory budget

Threadgroup limit ≈ 32 KiB. adder's tile `[128×64] f32` = 32 KiB — at the limit
with ONE staged tile. Chained reduces need ≤1 live tile at a time IF buffers are
reused (score buffer reused for p, etc.) plus small `[M]` buffers (`d`, `max`,
`sum` = 512 B each). The "stage only roots + re-derive" strategy (3.3) keeps it
to small `[M]` buffers and fits comfortably; "stage the tile" is at the edge.

## 6. Risk assessment — HIGH

1. **Liveness/scheduling across nested control flow** (tile > pos > scf.if):
   correctly partitioning and computing live-across sets is the crux. HIGH.
2. **Restructuring inside a stateful user loop** with tensor iter_args: the pos
   loop's `next_token` and the `d` scf.if make this far harder than a flat body.
   HIGH.
3. **Blocked-coord flatpos for spt>1** (adder spt=[1,4]): stage/unstage indexing
   must be bit-exact or silent wrong results. MEDIUM (oracle exists).
4. **Doing this in dialect conversion** (op-by-op) is awkward; likely needs a
   dedicated pre-pass before TritonGPU→Metal. MEDIUM.
5. **Regression**: must not touch the 223 passing single-pass kernels; multi-pass
   path must trigger only for computed/chained reduces. MEDIUM.

## 7. Phased plan (each phase = its own wall, with probes + lit + pytest)

- **Phase A — flat single computed reduce → output.**
  Body = `out[r] = reduce(f(load), axis=1)` with `f` NON-re-derivable (force the
  stage-the-tile path). No outer loop. Validates: phase split of a flat body,
  stage `[M*N]`, reduce-from-stage, result→store. Smallest viable multi-pass.
- **Phase B — flat chained softmax.**
  `m=max(x); p=exp(x−m); s=sum(p); out=p/s` (+ `sum(p*v)`), no outer loop.
  Validates chained reduces + buffer reuse + result-buffer handoff. This is a
  REAL, broadly useful capability (any single-tile softmax).
- **Phase C — adder: reduces inside a stateful user loop + scf.if.**
  Adds: phase-split inside `scf.for pos`, tensor iter_arg `next_token`, `d`
  scf.if root. HIGHEST risk; may need the "stage roots + re-derive" strategy to
  avoid 32 KiB tiles inside the loop. This is the only phase that makes
  `adder_transformer` itself compile.

## 8. Recommendation

Build **Phase A then B** first: they deliver a genuinely useful capability
(computed-tile + flat softmax reductions) at moderate risk, and de-risk the
core machinery (phase split, staging, chained handoff). Treat **Phase C
(adder)** as a separate, explicitly high-risk follow-up: its nested stateful
control flow is near the worst case for this architecture, and a correct,
regression-safe implementation is a substantial multi-turn effort that may
ultimately need the "stage-roots + re-derive" hybrid rather than full tile
materialization. Re-evaluate Phase C after A/B land and the machinery is proven.

If the goal is strictly "make adder pass ASAP", the honest expectation is: A→B→C
across several turns, with C carrying real risk of needing design iteration.
