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


@triton.jit
def _rank2_sum_then_rank1_softmax_stats(q_ptr, k_ptr, m_ptr, l_ptr,
                                        M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    q = tl.load(q_ptr + rn)
    k = tl.load(k_ptr + rm[:, None] * N + rn[None, :])
    scores = tl.sum(q[None, :] * k, axis=1)
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - max_score)
    exp_sum = tl.sum(exp_scores, axis=0)
    tl.store(m_ptr, max_score)
    tl.store(l_ptr, exp_sum)


@pytest.mark.parametrize("M, N", [(8, 16), (32, 16), (32, 32)])
def test_rank2_sum_then_slice_rank1_softmax_stats(M, N):
    """A rank-2 reduce result keeps its slice layout into a rank-1 reduce."""
    torch.manual_seed(0x51CE + M * N)
    q = torch.randn(N, dtype=torch.float32, device="mps")
    k = torch.randn(M, N, dtype=torch.float32, device="mps")
    max_score = torch.empty((), dtype=torch.float32, device="mps")
    exp_sum = torch.empty((), dtype=torch.float32, device="mps")

    _rank2_sum_then_rank1_softmax_stats[(1,)](
        q, k, max_score, exp_sum, M=M, N=N
    )
    torch.mps.synchronize()

    scores = (q.cpu().double()[None, :] * k.cpu().double()).sum(dim=1)
    expected_max = scores.max()
    expected_sum = torch.exp(scores - expected_max).sum()
    torch.testing.assert_close(
        max_score.cpu().double(), expected_max, atol=2e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        exp_sum.cpu().double(), expected_sum, atol=2e-4, rtol=1e-4
    )


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


# --- Chained reduces, and a reduce result broadcast back into the tile -------
#
# A rank-2 softmax (`m = max(x); p = exp(x - m[:,None]); s = sum(p);
# out = p / s[:,None]`) needs a reduce result at TWO different indexings at
# once, and used to compile and return wrong numbers silently:
#
#  * Broadcast back into the 2D tile and materialised, element (r, n) wants
#    rowBuf[r]. The readback used the FLAT per-iteration index — for a 32x64
#    tile an index in [0, 2048) into a 32-slot buffer. It also picked the tile
#    geometry off the `tt.expand_dims` shim (`tensor<Mx1>`), whose shape says
#    nothing about what the consuming threads iterate.
#  * Consumed by ANOTHER reduce's cone, the row is the one that reduce is
#    filling (r == localTid), not the producer's tile position. Resolving it
#    through the producer's per-thread scalar (Inc-2.5 staging) cannot serve
#    both, so a consuming reduce now reads the producer's rowBuf directly.
#
# The rank-1 softmax (one program per row, `tl.max(row, axis=0)`) that the
# tutorials use never hit either path.


@triton.jit
def _rank2_softmax_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    idx = r[:, None] * N + c[None, :]
    x = tl.load(x_ptr + idx)
    m = tl.max(x, axis=1)
    p = tl.exp(x - m[:, None])
    s = tl.sum(p, axis=1)
    tl.store(o_ptr + idx, p / s[:, None])


@pytest.mark.parametrize("M, N", [(32, 64), (8, 16), (64, 32), (16, 128)])
def test_rank2_chained_softmax(M, N):
    torch.manual_seed(0x50F + M * N)
    x = torch.randn(M * N, dtype=torch.float32, device="mps")
    o = torch.zeros(M * N, dtype=torch.float32, device="mps")
    _rank2_softmax_kernel[(1,)](x, o, M, N)
    torch.mps.synchronize()
    ref = torch.softmax(x.cpu().double().reshape(M, N), dim=1).reshape(-1)
    err = (o.cpu().double() - ref).abs().max()
    assert err <= 2e-6, f"max err {err}"


@triton.jit
def _reduce_broadcast_back_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    idx = r[:, None] * N + c[None, :]
    x = tl.load(x_ptr + idx)
    m = tl.max(x, axis=1)
    tl.store(o_ptr + idx, x - m[:, None])      # 2D out: the row must be flat/N


@pytest.mark.parametrize("M, N", [(32, 64), (8, 16), (64, 32)])
def test_reduce_result_broadcast_into_tile(M, N):
    torch.manual_seed(0xBCA + M * N)
    x = torch.randn(M * N, dtype=torch.float32, device="mps")
    o = torch.zeros(M * N, dtype=torch.float32, device="mps")
    _reduce_broadcast_back_kernel[(1,)](x, o, M, N)
    torch.mps.synchronize()
    xc = x.cpu().double().reshape(M, N)
    ref = (xc - xc.max(1).values[:, None]).reshape(-1)
    err = (o.cpu().double() - ref).abs().max()
    assert err <= 2e-6, f"max err {err}"


@triton.jit
def _chained_sum_of_weighted_kernel(x_ptr, v_ptr, o_ptr,
                                    M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    idx = r[:, None] * N + c[None, :]
    x = tl.load(x_ptr + idx)
    v = tl.load(v_ptr + idx)
    m = tl.max(x, axis=1)
    p = tl.exp(x - m[:, None])
    s = tl.sum(p, axis=1)
    # A THIRD reduce over a cone reading both prior results — rank-1 output.
    tl.store(o_ptr + r, tl.sum(p * v, axis=1) / s)


@pytest.mark.parametrize("M, N", [(32, 64), (8, 16), (64, 32)])
def test_rank2_three_chained_reduces(M, N):
    torch.manual_seed(0x3C + M * N)
    x = torch.randn(M * N, dtype=torch.float32, device="mps")
    v = torch.randn(M * N, dtype=torch.float32, device="mps")
    o = torch.zeros(M, dtype=torch.float32, device="mps")
    _chained_sum_of_weighted_kernel[(1,)](x, v, o, M, N)
    torch.mps.synchronize()
    xc = x.cpu().double().reshape(M, N)
    vc = v.cpu().double().reshape(M, N)
    p = torch.exp(xc - xc.max(1).values[:, None])
    ref = (p * vc).sum(1) / p.sum(1)
    err = (o.cpu().double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-6, f"rel err {err}"
