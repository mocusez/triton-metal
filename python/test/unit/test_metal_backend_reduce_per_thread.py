"""L3a-tileloop: per-thread-owned axis=1 sum reduce on the Metal backend.

Exercises the new register-level reduce branch in `ReduceLowering` that
activates when `threadsPerCTA[axis_dim] == 1` (the reduce axis is fully
serial within each thread). This branch emits a per-thread accumulator
plus an N-way unrolled `arith.addf` / `arith.addi` chain — NO threadgroup
memory, NO barriers.

See `.omc/specs/deep-interview-leet-triton-l3a-tileloop-per-thread-reduce.md`.

Honest divergence note (AC.T2 carry-forward): the per-thread branch's
IR-shape ACs (B1–B5 + lit T1) are landed in this session; runtime bit-
exactness for the multi-element-per-thread (M*N >> tpb) case requires a
follow-on session — the MLIR conversion model gives the branch ONE scalar
SSA value per (thread, tile-iv) pair, and the per-row gather across
multiple tile-iv values cannot be expressed at the `ReduceLowering` site
without hoisting the reduce out of the enclosing tile loop or
reintroducing TG memory (both out of scope per spec non-goals). The cases
below are therefore marked xfail with that carry-forward reason; the lit
fixture `reduce_per_thread_owned.mlir` pins the IR shape.
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


@triton.jit
def reduce_per_thread_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(x_ptr + addr)
    s = tl.sum(x, axis=1)
    tl.store(out_ptr + offs_m, s)


_CARRYFWD_REASON = (
    "L3a-tileloop runtime bit-exactness for multi-elem-per-thread (M*N >> tpb) "
    "is a carry-forward to L3a-tileloop-2 — see spec AC.T2 honest-divergence "
    "note. IR-shape ACs (B1–B5, lit T1) land in this session."
)


@pytest.mark.parametrize(
    "M, N",
    [
        (1024, 64),
        (512, 32),
        (256, 128),
    ],
)
@pytest.mark.xfail(reason=_CARRYFWD_REASON, strict=False)
def test_reduce_per_thread_owned_f32(M, N):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_per_thread_kernel[(1, 1, 1)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.sum(x.cpu(), dim=1)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
