"""Wall 17 Increment 1: rank-2 axis=1 reduce over a COMPUTED tile (f32).

The rank-2 `ReduceLowering` row-scan body originally required its source to be
a direct device `tt.load` (it reduces by re-reading `device[base + r*N + n]`).
Softmax-style kernels reduce register-resident COMPUTED tiles instead. This
exercises the `evalRank2ConeAt` cone evaluator, which re-derives each reduced
element as scalar Metal ops, recursing the pre-conversion tensor cone down to
its device-load / splat / constant leaves.

Increment 1 covers: device loads (single and MULTIPLE), f32 elementwise arith
(add/sub/mul/div/max — both operands may be tensors), and unary math (exp,
sqrt, ...). The softmax constructs in adder_transformer (broadcast of a per-row
scalar, tl.where, and chained reduce results) are later increments.
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
def _sum_affine(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rm, tl.sum(x * 2.0 + 1.0, axis=1))


@triton.jit
def _max_affine(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rm, tl.max(x * 2.0 + 1.0, axis=1))


@triton.jit
def _sum_exp(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rm, tl.sum(tl.exp(x), axis=1))


@triton.jit
def _sum_twoload(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    addr = rm[:, None] * N + rn[None, :]
    xa = tl.load(a_ptr + addr)
    xb = tl.load(b_ptr + addr)
    tl.store(out_ptr + rm, tl.sum(xa * xb, axis=1))


SHAPES = [(8, 16), (4, 32), (128, 64)]


@pytest.mark.parametrize("M, N", SHAPES)
def test_reduce_sum_affine_cone(M, N):
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    _sum_affine[(1, 1, 1)](x, out, M, N)
    torch.testing.assert_close(out.cpu(), (x * 2.0 + 1.0).sum(dim=1), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("M, N", SHAPES)
def test_reduce_max_affine_cone(M, N):
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    _max_affine[(1, 1, 1)](x, out, M, N)
    torch.testing.assert_close(out.cpu(), (x * 2.0 + 1.0).amax(dim=1), atol=1e-5, rtol=0)


@pytest.mark.parametrize("M, N", SHAPES)
def test_reduce_sum_exp_cone(M, N):
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    _sum_exp[(1, 1, 1)](x, out, M, N)
    torch.testing.assert_close(out.cpu(), torch.exp(x).sum(dim=1), atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("M, N", SHAPES)
def test_reduce_sum_two_load_product(M, N):
    """Two distinct device loads in one cone (q·k-style row dot)."""
    torch.manual_seed(0)
    a = torch.randn((M, N), dtype=torch.float32).contiguous()
    b = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    _sum_twoload[(1, 1, 1)](a, b, out, M, N)
    torch.testing.assert_close(out.cpu(), (a * b).sum(dim=1), atol=1e-3, rtol=1e-4)


# --- Increment 2: tt.where + per-column make_range broadcast + comparison ---
# (`tl.where(seq[None,:] <= K, x, fill)` — the softmax causal-mask shape, with
# the column index, the comparison, the select, and the broadcast all re-derived
# per element by the cone evaluator. The induction var feeds the cmpi directly,
# which exercises the translator's block-arg-safe comparison emission.)


@triton.jit
def _sum_where_col(x_ptr, out_ptr, K: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    masked = tl.where(rn[None, :] <= K, x, 0.0)
    tl.store(out_ptr + rm, tl.sum(masked, axis=1))


@triton.jit
def _max_where_col(x_ptr, out_ptr, K: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    masked = tl.where(rn[None, :] <= K, x, float("-inf"))
    tl.store(out_ptr + rm, tl.max(masked, axis=1))


@pytest.mark.parametrize("M, N", SHAPES)
def test_reduce_sum_where_col_mask(M, N):
    torch.manual_seed(0)
    K = N // 2
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    _sum_where_col[(1, 1, 1)](x, out, K, M, N)
    col = torch.arange(N)
    expected = torch.where(col[None, :] <= K, x, torch.zeros_like(x)).sum(dim=1)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("M, N", SHAPES)
def test_reduce_max_where_col_mask(M, N):
    torch.manual_seed(0)
    K = N // 2
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    _max_where_col[(1, 1, 1)](x, out, K, M, N)
    col = torch.arange(N)
    neg_inf = torch.full_like(x, float("-inf"))
    expected = torch.where(col[None, :] <= K, x, neg_inf).amax(dim=1)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=0)
