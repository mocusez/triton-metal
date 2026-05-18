"""Phase B regression: divergent-offset multi-load on 1D kernels.

Covers the bug surfaced by `.omc/specs/deep-interview-lmultiload-phase-a-diagnosis.md`
and partially fixed by `.omc/specs/deep-interview-lmultiload-phase-b-fix.md`.

Pre-fix failure mode (captured BEFORE the AddPtrLowering + emitLoadStoreIndex
edits): rank-1 `tt.load` / `tt.store` hardcoded `[[thread_position_in_grid]].x`
as the load index, discarding the actual offset arithmetic. As a result:

  * Pattern A (same base, divergent constant offsets): both
    `tl.load(ptr + offs)` and `tl.load(ptr + offs + K)` emitted the same MSL
    (`v0[id.x]`), so `out1` and `out2` were bit-identical (instead of
    out2 == ptr[offs + K]).
  * Pattern B (different bases, derived offsets `offs >> 1`): the `>> 1`
    shift was dropped — both loads read `v[id.x]` instead of `v[id.x >> 1]`.
  * Pattern C (same base, derived offset `offs >> 1`): same drop as B —
    the kernel stored `ptr[id.x]` instead of `ptr[id.x >> 1]`.

Post-fix status: Pattern A PASSES (the AddPtrLowering accumulator + new
`emitLoadStoreIndex` 1D path now thread constant divergent offsets through
the per-thread index). Patterns B and C remain BROKEN and are xfailed —
they need a change to `MakeRangeLowering`'s constant-0 placeholder
convention so per-thread arange values survive into the scalar IR. That
change is explicitly out of scope for Phase B (see "Non-Goals" in
`.omc/specs/deep-interview-lmultiload-phase-b-fix.md`) and is deferred to
a follow-up session.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not hasattr(libmetal, "launch_kernel_with_pipeline"):
    pytest.skip(
        "Metal runtime not compiled (non-Darwin build or Xcode CLT absent)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Pattern A: same base, divergent constant offsets (GLU shape).
# ---------------------------------------------------------------------------
@triton.jit
def _two_load_same_base_const_offset(in_ptr, out1, out2, K: tl.constexpr,
                                     BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x1 = tl.load(in_ptr + offs)
    x2 = tl.load(in_ptr + offs + K)
    tl.store(out1 + offs, x1)
    tl.store(out2 + offs, x2)


@pytest.mark.parametrize("BLOCK", [32, 128])
def test_pattern_A_same_base_divergent_const_offsets(BLOCK):
    K = BLOCK  # second half of the buffer
    N = 2 * BLOCK
    x = torch.arange(N, dtype=torch.float32)
    out1 = torch.zeros(BLOCK, dtype=torch.float32)
    out2 = torch.zeros(BLOCK, dtype=torch.float32)

    _two_load_same_base_const_offset[(1, 1, 1)](x, out1, out2, K=K, BLOCK=BLOCK)

    assert torch.equal(out1, x[:BLOCK]), (
        f"out1 mismatch: got {out1}, expected {x[:BLOCK]}"
    )
    assert torch.equal(out2, x[K:K + BLOCK]), (
        f"out2 mismatch (offset+K dropped?): got {out2}, "
        f"expected {x[K:K + BLOCK]}"
    )


# Pattern A multi-program variant (mirrors GLU's grid > 1 shape). The
# inner `tt.addptr(splat(in_ptr), offs)` has scalarized offset `pid*BLOCK`;
# the outer `tt.addptr(x1, K)` chains on top — without the AddPtrLowering
# accumulator the outer chain dropped the inner `pid*BLOCK`, and without
# the `emitLoadStoreIndex` rewrite the load reused `[[thread_position_in_
# grid]]` directly so multi-program kernels emitted the same `v0[id.x]`
# for every pid.
@triton.jit
def _two_load_multi_program(in_ptr, out1, out2, n_elements, K: tl.constexpr,
                            BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x1 = tl.load(in_ptr + offs, mask=mask, other=0.0)
    x2 = tl.load(in_ptr + offs + K, mask=mask, other=0.0)
    tl.store(out1 + offs, x1, mask=mask)
    tl.store(out2 + offs, x2, mask=mask)


@pytest.mark.parametrize("BLOCK,N_PROGS", [(64, 2), (128, 4)])
def test_pattern_A_multi_program(BLOCK, N_PROGS):
    n_elements = BLOCK * N_PROGS
    K = n_elements  # second half of a 2N buffer
    x = torch.arange(2 * n_elements, dtype=torch.float32)
    out1 = torch.zeros(n_elements, dtype=torch.float32)
    out2 = torch.zeros(n_elements, dtype=torch.float32)

    _two_load_multi_program[(N_PROGS, 1, 1)](
        x, out1, out2, n_elements, K=K, BLOCK=BLOCK
    )

    assert torch.equal(out1, x[:n_elements]), (
        f"multi-program out1 mismatch: got {out1}, expected {x[:n_elements]}"
    )
    assert torch.equal(out2, x[K:K + n_elements]), (
        f"multi-program out2 mismatch (pid*BLOCK or +K dropped?): "
        f"got {out2}, expected {x[K:K + n_elements]}"
    )


# ---------------------------------------------------------------------------
# Pattern B: different bases, same derived offsets via `>> 1` (vec_add
# shape with derived offsets).
# ---------------------------------------------------------------------------
@triton.jit
def _two_load_diff_base_derived_offset(A_ptr, B_ptr, out_ptr,
                                       BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    halved = offs >> 1
    a = tl.load(A_ptr + halved)
    b = tl.load(B_ptr + halved)
    tl.store(out_ptr + offs, a + b)


@pytest.mark.parametrize("BLOCK", [32, 128])
def test_pattern_B_diff_base_derived_offset(BLOCK):
    # halved offsets cover [0, BLOCK//2); buffers sized for that.
    A = torch.arange(BLOCK // 2, dtype=torch.float32)
    B = torch.arange(BLOCK // 2, dtype=torch.float32) * 10.0
    out = torch.zeros(BLOCK, dtype=torch.float32)

    _two_load_diff_base_derived_offset[(1, 1, 1)](A, B, out, BLOCK=BLOCK)

    idx = torch.arange(BLOCK) >> 1
    expected = A[idx] + B[idx]
    assert torch.equal(out, expected), (
        f"derived-offset multi-load mismatch (>>1 dropped?): "
        f"got {out}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Pattern C: same base, derived offsets (interleave shape — single load
# with `offs >> 1`, store at `offs`).
# ---------------------------------------------------------------------------
@triton.jit
def _single_load_derived_offset(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    halved = offs >> 1
    x = tl.load(in_ptr + halved)
    tl.store(out_ptr + offs, x)


@pytest.mark.parametrize("BLOCK", [32, 128])
def test_pattern_C_same_base_derived_offset(BLOCK):
    x = torch.arange(BLOCK // 2, dtype=torch.float32)
    out = torch.zeros(BLOCK, dtype=torch.float32)

    _single_load_derived_offset[(1, 1, 1)](x, out, BLOCK=BLOCK)

    idx = torch.arange(BLOCK) >> 1
    expected = x[idx]
    assert torch.equal(out, expected), (
        f"derived-offset single-load mismatch (>>1 dropped?): "
        f"got {out}, expected {expected}"
    )
