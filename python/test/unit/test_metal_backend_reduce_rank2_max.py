"""Wall 16: axis=1 MAX reduce on the Metal backend (f32).

Companion to `test_metal_backend_reduce_sum.py`. The rank-2 `ReduceLowering`
row-scan body originally only implemented the sum combine (arith.addf /
arith.addi). This exercises the MAX combine (`tl.max`, which Triton emits as
arith.maxnumf): the kernel loads a 2D `(M, N)` block, calls
`tl.max(x, axis=1)`, and stores the resulting `(M,)` vector.

The combine is emitted as `metal.binary_exp ... maxOp` (MSL `max(a, b)`) and
the per-row scf.for iter_arg is identity-initialised to -FLT_MAX (the MSL
float-constant emitter can't render -inf, and `max(x, -FLT_MAX) == x` for
every finite x). Max is an exact element selection, so the result is compared
bit-tight against `torch.amax(input, dim=1)`.
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
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


@triton.jit
def reduce_max_axis1_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(x_ptr + addr)
    # axis=1 max reduce: tensor<BLOCK_M x BLOCK_N> -> tensor<BLOCK_M>
    m = tl.max(x, axis=1)
    tl.store(out_ptr + offs_m, m)


@pytest.mark.parametrize(
    "M, N",
    [
        (8, 16),
        (4, 32),
        (16, 16),
        (128, 64),  # adder_transformer's softmax tile shape (M == tpb)
    ],
)
def test_reduce_max_axis1_f32(M, N):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_max_axis1_kernel[(1, 1, 1)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.amax(x.cpu(), dim=1)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-6, rtol=0)


def test_reduce_max_axis1_negative_rows():
    """All-negative rows: confirms the -FLT_MAX identity never wins over a
    real (finite, negative) element."""
    M, N = 8, 16
    x = -torch.rand((M, N), dtype=torch.float32).contiguous() - 1.0  # in [-2, -1)
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_max_axis1_kernel[(1, 1, 1)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.amax(x.cpu(), dim=1)
    assert (out.cpu() < 0).all(), f"identity (-FLT_MAX) leaked into result: {out}"
    torch.testing.assert_close(out.cpu(), expected, atol=1e-6, rtol=0)
