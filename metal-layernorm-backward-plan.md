# Layer-norm BACKWARD on Metal — plan

> **STATUS: ✅ DONE (2026-07-18), incl. verbatim kernel-2.** Backward runs
> end-to-end vs torch.autograd (fp32/fp16/bf16, non-pow2 N, M>GROUP contention).
> Commits d1f4aa002a (tensor masked atomic_add) + 69466d4d58 (backward e2e) +
> 7b2ffc06e0 (rank-2 axis=0 reduce → verbatim `_layer_norm_bwd_dwdb`). 393 pytest
> / 109 lit, zero regression. Gotcha: atomics force spt=1 → Stage-1 launches
> `num_warps=BLOCK/32` (E==1), caps N at 1024; Stage-2 (axis=0 reduce) runs at
> BLOCK_SIZE_N=128==tpb (E_out==1). Details in memory
> `metal-fp16-and-layernorm-status`.


Forward is done (fp32/fp16/bf16, E1/E>1, multi-program, non-pow2 N). Backward is
tutorial-05's two-kernel design. Probes (`scratchpad/bwd_spikes.py`) pin the scope:

| Piece | Probe | Status |
|---|---|---|
| Kernel-1 **dx** (two reduces → elementwise → masked store) | Spike A | ✅ **already works bit-exact** (1.2e-7) — 0 backend work |
| Kernel-1 **dw/db** accumulate (rank-1 tensor masked `atomic_add`) | Spike B | ❌ `tt.atomic_rmw` tensor form rejected (`:5329`) |
| Kernel-2 `_layer_norm_bwd_dwdb` (rank-2 **axis=0** reduce) | Spike C | ❌ axis=0 reduce deferred (`:9167`) |

## Key design decision: DROP the spin lock, use `atomic_add`

The verbatim kernel guards dw/db accumulation with a global spin lock
(`while tl.atomic_cas(Lock,0,1)==1: pass` … `tl.atomic_xchg(Lock,0)`). **This
cannot be ported faithfully to Apple GPUs**: Metal gives no independent
forward-progress guarantee across threadgroups, so a global spin lock can
deadlock (a spinning threadgroup can occupy the slot the lock-holder needs).
Even a perfect atomic_cas/xchg/scf.while lowering would produce a kernel that
*may hang* depending on M / GROUP_SIZE_M / occupancy — not shippable.

The lock exists only to serialize a read-modify-write accumulation into the
shared `_dw[lock_id]` buffer. `tl.atomic_add(DW, partial_dw, mask)` does exactly
that accumulation atomically and **lock-free** — no Lock, no Count, no
critical section, no barrier. Metal has native `atomic_fetch_add_explicit` on
`device atomic_float*` (already used by the scalar atomic path). Float-add
reorder makes it non-bit-deterministic but bit-close within fp tolerance (same
regime as the forward's fp16/bf16 tolerances). This is the canonical portable
rewrite. The two-stage structure and host driver stay unchanged.

→ Kernel-1 body becomes a **documented Metal variant** (verbatim dx; lock loop
replaced by two `atomic_add`s). Faithful spin-lock = explicit **non-goal**.

## Phases

**Phase 1 — rank-1 tensor masked `atomic_add`** (keystone; unblocks all of kernel-1
since dx already works). Extend `AtomicRmwLowering` (`:5316`): handle the rank-1
tensor fadd form with a real per-element mask. Model it on the **masked store**
(per-thread cone index + mask + value), emitting `atomic_fetch_add` instead of
`=` — NOT the scalar path's `localTid==0` guard (that's for per-program scalar
atomics; here each thread owns its own element). Test `test_metal_backend_atomic_add.py`
(scatter/histogram add). Zero-regression. Commit.

**Phase 2 — rank-2 axis=0 reduce → rank-1 [N]** (kernel-2). Implement the axis=0
direction (per-column sum of a 2D tile) + the loop-carried 2D-accumulator
reassociation (`sum_axis0(acc += tile) → s[n] += sum_axis0(tile)[n]`, addf
associative), lifting the `:9167` gate. Test standalone column-sum + full
`_layer_norm_bwd_dwdb`. Commit.

**Phase 3 — assemble kernel-1 + end-to-end parity**. Metal-variant kernel-1
(verbatim dx + atomic_add tail). Full `layer_norm` backward vs `torch.autograd`
(dx, dw, db) across fp32/fp16/bf16 and M/N — including **M > GROUP_SIZE_M** to
exercise real atomic contention. Extend `test_metal_backend_layer_norm.py`.
Update memory + this doc. Commit.

## Non-goals
- Faithful `tt.atomic_cas` / `tt.atomic_xchg` / `scf.while` spin-lock (hazard above).
- Layer-norm training benchmark harness.
