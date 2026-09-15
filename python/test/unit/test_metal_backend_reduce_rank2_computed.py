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


@triton.jit
def _fft_like_remainder_twoload_reduce(
    x_ptr,
    real_ptr,
    imag_ptr,
    n_rows,
    n_cols,
    period,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    xr = tl.load(x_ptr + 2 * cols, mask=col_mask, other=0.0)
    xi = tl.load(x_ptr + 2 * cols + 1, mask=col_mask, other=0.0)
    phase_i = (rows[:, None] * cols[None, :]) % period
    phase = (phase_i.to(tl.float32) / period) * -6.283185307179586
    c = tl.cos(phase)
    s = tl.sin(phase)
    real = tl.sum(tl.where(mask, xr[None, :] * c - xi[None, :] * s, 0.0), axis=1)
    imag = tl.sum(tl.where(mask, xr[None, :] * s + xi[None, :] * c, 0.0), axis=1)

    tl.store(real_ptr + rows, real, mask=row_mask)
    tl.store(imag_ptr + rows, imag, mask=row_mask)


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


@pytest.mark.parametrize("n_rows,n_cols,block_m,block_n", [(17, 63, 32, 64), (65, 65, 32, 128)])
def test_reduce_sum_fft_like_remainder_two_loads(n_rows, n_cols, block_m, block_n):
    """DFT-shaped rank-2 cone with runtime tensor remainder and adjacent sums."""
    torch.manual_seed(0xD7F + n_rows * 31 + n_cols)
    source = torch.randn(n_cols, 2, dtype=torch.float32)
    real = torch.empty(n_rows, dtype=torch.float32, device="mps")
    imag = torch.empty(n_rows, dtype=torch.float32, device="mps")

    _fft_like_remainder_twoload_reduce[(triton.cdiv(n_rows, block_m),)](
        source.reshape(-1).to("mps"),
        real,
        imag,
        n_rows,
        n_cols,
        n_cols,
        BM=block_m,
        BN=block_n,
        num_warps=4,
    )
    torch.mps.synchronize()

    rows = torch.arange(n_rows, dtype=torch.int64)[:, None]
    cols = torch.arange(n_cols, dtype=torch.int64)[None, :]
    phase_i = torch.remainder(rows * cols, n_cols)
    phase = phase_i.to(torch.float64) / n_cols * -6.283185307179586
    xr = source[:, 0].to(torch.float64)[None, :]
    xi = source[:, 1].to(torch.float64)[None, :]
    expected_real = (xr * torch.cos(phase) - xi * torch.sin(phase)).sum(dim=1)
    expected_imag = (xr * torch.sin(phase) + xi * torch.cos(phase)).sum(dim=1)
    torch.testing.assert_close(real.cpu().double(), expected_real, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(imag.cpu().double(), expected_imag, atol=5e-4, rtol=5e-4)


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


# --- Leet categorical cross entropy ---------------------------------------
#
# This is the verbatim kernel shape from
# `python/test/unit/fixtures/metal_leet/medium-categorical_cross_entropy_loss.py`.  It combines three
# previously independent paths in one launch:
#
# * a masked rank-2 computed reduce whose `other=-inf` must survive cone replay,
# * an int64 label gather (PyTorch's standard class-index dtype), and
# * a second rank-1 reduce over the gathered logits, including BLOCK_N=128.


@triton.jit
def _categorical_cross_entropy_kernel(
    logits_ptr,
    labels_ptr,
    loss_ptr,
    n_rows,
    n_classes,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    class_mask = tl.arange(0, BLOCK_C) < n_classes

    offsets = n_classes * rows[:, None] + tl.arange(0, BLOCK_C)[None, :]
    mask = row_mask[:, None] & class_mask[None, :]
    logits = tl.load(logits_ptr + offsets, mask=mask, other=float("-inf"))

    result = tl.exp(logits)
    result = tl.where(row_mask, tl.sum(result, axis=1), 1.0)
    result = tl.sum(tl.log(result), axis=0)

    labels = tl.load(labels_ptr + rows, mask=row_mask, other=0)
    selected_offsets = n_classes * rows + labels
    selected = tl.load(logits_ptr + selected_offsets, mask=row_mask)
    result -= tl.sum(selected, axis=0)
    result /= n_rows
    tl.atomic_add(loss_ptr, result, sem="relaxed")


@pytest.mark.parametrize(
    "n_rows,n_classes",
    [
        (17, 7),       # BLOCK_N=128: small-class gather-reduce path
        (37, 9),       # non-power-of-two class mask with substantial padding
        (130, 17),     # non-power-of-two classes and multiple programs
        (5, 512),      # int64 label load with a small BLOCK_N
    ],
)
def test_categorical_cross_entropy(n_rows, n_classes):
    torch.manual_seed(0xCCE + n_rows * 1000 + n_classes)
    logits_cpu = torch.randn(n_rows, n_classes, dtype=torch.float32)
    labels_cpu = torch.randint(
        0, n_classes, (n_rows,), dtype=torch.int64
    )
    logits = logits_cpu.to("mps")
    labels = labels_cpu.to("mps")
    loss = torch.zeros(1, dtype=torch.float32, device="mps")

    block_c = triton.next_power_of_2(n_classes)
    block_n = 1024 // block_c
    _categorical_cross_entropy_kernel[(triton.cdiv(n_rows, block_n),)](
        logits,
        labels,
        loss,
        n_rows,
        n_classes,
        BLOCK_N=block_n,
        BLOCK_C=block_c,
    )
    torch.mps.synchronize()

    expected = torch.nn.functional.cross_entropy(logits_cpu, labels_cpu)
    torch.testing.assert_close(
        loss.cpu().squeeze(), expected, atol=2e-5, rtol=1e-5
    )


# --- Leet token embedding + LayerNorm -------------------------------------
#
# Verbatim kernel shape from `python/test/unit/fixtures/metal_leet/medium-token_embedding_layer.py`.
# It combines two rank-1 integer gathers, two rank-2 embedding gathers, and two
# chained axis=1 reductions whose results are broadcast back into a tiled 2D
# value.  The `position_ids[rows % T]` address also exercises signed remainder
# while replaying a rank-1 load inside the computed-reduce cone.


@triton.jit
def _token_embedding_layernorm_kernel(
    token_ids_ptr,
    position_ids_ptr,
    token_emb_ptr,
    position_emb_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    N,
    T,
    D,
    eps,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < N
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    mask2d = row_mask[:, None] & col_mask[None, :]

    tok_ids = tl.load(token_ids_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    pos_ids = tl.load(
        position_ids_ptr + rows % T, mask=row_mask, other=0
    ).to(tl.int64)
    emb_off = cols[None, :]
    tok = tl.load(
        token_emb_ptr + tok_ids[:, None] * D + emb_off,
        mask=mask2d,
        other=0.0,
    ).to(tl.float32)
    pos = tl.load(
        position_emb_ptr + pos_ids[:, None] * D + emb_off,
        mask=mask2d,
        other=0.0,
    ).to(tl.float32)
    summed = tok + pos

    mean = tl.sum(summed, axis=1) / D
    diff = tl.where(mask2d, summed - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    output = gamma[None, :] * diff * rstd[:, None] + beta[None, :]
    out_off = rows.to(tl.int64)[:, None] * D + emb_off
    tl.store(
        output_ptr + out_off,
        output.to(output_ptr.dtype.element_ty),
        mask=mask2d,
    )


@pytest.mark.parametrize(
    "B,T,V,P,D",
    [
        (1, 1, 8, 8, 32),
        (2, 5, 31, 16, 64),
        (3, 7, 23, 11, 33),
        (9, 8, 31, 16, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_token_embedding_layernorm_matches_torch(B, T, V, P, D, dtype):
    torch.manual_seed(0xEBD + B * 1009 + T * 101 + D)
    eps = 1e-5
    token_ids_cpu = torch.randint(0, V, (B, T), dtype=torch.int32)
    position_ids_cpu = torch.randint(0, P, (T,), dtype=torch.int32)
    token_emb_cpu = torch.randn(V, D, dtype=dtype)
    position_emb_cpu = torch.randn(P, D, dtype=dtype)
    gamma_cpu = torch.randn(D, dtype=dtype)
    beta_cpu = torch.randn(D, dtype=dtype)
    output = torch.empty(B, T, D, dtype=dtype, device="mps")

    block_d = triton.next_power_of_2(D)
    block_t = max(1, 4096 // block_d)
    grid = (triton.cdiv(B * T, block_t),)
    _token_embedding_layernorm_kernel[grid](
        token_ids_cpu.to("mps"),
        position_ids_cpu.to("mps"),
        token_emb_cpu.to("mps"),
        position_emb_cpu.to("mps"),
        gamma_cpu.to("mps"),
        beta_cpu.to("mps"),
        output,
        B * T,
        T,
        D,
        eps,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=8 if block_t * block_d >= 4096 else 4,
    )
    torch.mps.synchronize()

    gathered = (
        token_emb_cpu.float()[token_ids_cpu.long()]
        + position_emb_cpu.float()[position_ids_cpu.long()].unsqueeze(0)
    )
    expected = torch.nn.functional.layer_norm(
        gathered, (D,), gamma_cpu.float(), beta_cpu.float(), eps
    )
    atol = rtol = 1e-4 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(
        output.cpu().float(), expected, atol=atol, rtol=rtol
    )


# ---------------------------------------------------------------------------
# Unary math the rank-2 cone could not replay.
#
# `rank2ConeSupported` and `evalRank2ConeAt` carried their own copy of the
# admitted-op list, so abs/floor/ceil/exp2/log2 and a plain negation were
# missing here for the same reason as at rank 1. A `tl.sum(tl.sigmoid(x),
# axis=1)` aborted the process instead of reducing.
# ---------------------------------------------------------------------------


@triton.jit
def _sum_sigmoid(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rm, tl.sum(tl.sigmoid(x), axis=1))


@triton.jit
def _sum_floor(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rm, tl.sum(tl.floor(x), axis=1))


@triton.jit
def _max_abs(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rm, tl.max(tl.abs(x), axis=1))


@triton.jit
def _sum_neg_exp2(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rm, tl.sum(tl.exp2(-x), axis=1))


_RANK2_CONE_CASES = {
    "sigmoid": (_sum_sigmoid, lambda t: torch.sigmoid(t).sum(dim=1)),
    "floor": (_sum_floor, lambda t: torch.floor(t).sum(dim=1)),
    "abs": (_max_abs, lambda t: torch.abs(t).amax(dim=1)),
    "neg_exp2": (_sum_neg_exp2, lambda t: torch.exp2(-t).sum(dim=1)),
}


@pytest.mark.parametrize("case", sorted(_RANK2_CONE_CASES))
@pytest.mark.parametrize("M,N", [(4, 32), (8, 16), (2, 64)])
def test_rank2_unary_math_cone_matches_torch(case, M, N):
    kernel, ref = _RANK2_CONE_CASES[case]
    torch.manual_seed(M * 131 + N)
    x = torch.randn(M, N) * 2.0
    out = torch.zeros(M, device="mps")
    kernel[(1,)](x.to("mps"), out, M=M, N=N, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), ref(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("N, K, iterations, ties", [
    (1, 1, 1, False),
    (128, 4, 1, False),
    (257, 3, 3, False),
    (513, 17, 4, False),
    (129, 3, 2, True),
])
def test_kmeans_original_solve_centroids_and_labels(N, K, iterations, ties):
    """Exercise the unchanged assignment/count/copy sequence against CPU Lloyd steps."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "metal_leet" / "hard-k_means_clustering.py"
    spec = importlib.util.spec_from_file_location("leet_kmeans_clustering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = torch.Generator().manual_seed(2026 + K)
    points = torch.randn((2, N), generator=generator)
    centers = torch.randn((2, K), generator=generator)
    if ties:
        # Equal distance to centers 0 and 1, plus a permanently empty cluster.
        points.zero_()
        centers = torch.tensor([[-1.0, 1.0, 100.0], [0.0, 0.0, 100.0]])
    expected_centers = centers.clone()
    for _ in range(iterations):
        distances = ((points[:, None, :] - expected_centers[:, :, None]) ** 2).sum(0)
        expected_labels = distances.argmin(0)
        for cluster in range(K):
            selected = expected_labels == cluster
            if selected.any():
                expected_centers[:, cluster] = points[:, selected].mean(1)

    initial_x = torch.full((K + 3,), 12345.0, device="mps")
    initial_y = torch.full_like(initial_x, 12345.0)
    initial_x[:K] = centers[0].to("mps")
    initial_y[:K] = centers[1].to("mps")
    final_x = torch.full_like(initial_x, 12345.0)
    final_y = torch.full_like(initial_y, 12345.0)
    # The source uses final_* as atomic accumulators on its first iteration.
    final_x[:K] = 0
    final_y[:K] = 0
    labels = torch.full((N + 7,), -12345, dtype=torch.int32, device="mps")
    module.solve(points[0].to("mps"), points[1].to("mps"), labels,
                 initial_x, initial_y, final_x, final_y, N, K, iterations)
    torch.mps.synchronize()
    torch.testing.assert_close(labels[:N].cpu(), expected_labels.to(torch.int32), rtol=0, atol=0)
    for initial, final, expected in zip((initial_x, initial_y), (final_x, final_y), expected_centers):
        torch.testing.assert_close(final[:K].cpu(), expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(initial[:K].cpu(), expected, rtol=1e-4, atol=1e-5)
        assert torch.all(initial[K:].cpu() == 12345)
        assert torch.all(final[K:].cpu() == 12345)
    assert torch.all(labels[N:].cpu() == -12345)


@pytest.mark.parametrize("N, K", [(129, 1), (257, 3), (513, 17)])
def test_kmeans_postprocess_loop_counts_and_empty_clusters(N, K):
    """Count a comparison tile inside the source loop, then reset atomic sums."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "metal_leet" / "hard-k_means_clustering.py"
    spec = importlib.util.spec_from_file_location("leet_kmeans_postprocess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    labels = torch.arange(N, dtype=torch.int32) % max(K - 1, 1)
    counts = torch.bincount(labels.long(), minlength=K).float()
    initial_x = torch.full((K,), -99.0, device="mps")
    initial_y = torch.full_like(initial_x, -99.0)
    sum_x = (2.0 * counts).to("mps")
    sum_y = (-3.0 * counts).to("mps")
    module.postprocess_means[(1,)](
        labels.to("mps"), initial_x, initial_y, sum_x, sum_y,
        K, N, 128, triton.next_power_of_2(K),
    )
    torch.mps.synchronize()
    torch.testing.assert_close(initial_x.cpu(), torch.where(counts > 0, 2.0, -99.0), rtol=0, atol=0)
    torch.testing.assert_close(initial_y.cpu(), torch.where(counts > 0, -3.0, -99.0), rtol=0, atol=0)
    torch.testing.assert_close(sum_x.cpu(), torch.zeros(K), rtol=0, atol=0)
    torch.testing.assert_close(sum_y.cpu(), torch.zeros(K), rtol=0, atol=0)


@triton.jit
def _affine_range_axis0_count_kernel(
    labels_ptr,
    counts_out,
    trips_out,
    k,
    lo,
    hi,
    n,
    START: tl.constexpr,
    INC: tl.constexpr,
    STEP: tl.constexpr,
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
):
    off = tl.arange(START, START + BLOCK)
    bins = tl.arange(0, BK)
    counts = tl.zeros((BK,), tl.int32)
    trips = tl.zeros((BK,), tl.int32)
    for _ in range(lo, hi, STEP):
        mask = (off >= 0) & (off < n)
        idx = tl.load(labels_ptr + off, mask=mask, other=-1)
        eq = (idx[:, None] == bins[None, :]).to(tl.int32)
        counts += tl.sum(eq, axis=0)
        trips += tl.full((BK,), 1, tl.int32)
        off += INC
    active = bins < k
    tl.store(counts_out + bins, counts, mask=active)
    tl.store(trips_out + bins, trips, mask=active)


def _expected_affine_range_counts(labels, lo, hi, step, start, inc, block, k):
    chunks = []
    off = torch.arange(start, start + block, dtype=torch.int64)
    for _ in range(lo, hi, step):
        valid = (off >= 0) & (off < labels.numel())
        chunks.append(off[valid])
        off = off + inc
    idx = torch.cat(chunks) if chunks else torch.empty((0,), dtype=torch.int64)
    return torch.bincount(labels[idx].long(), minlength=k).to(torch.int32), len(chunks)


@pytest.mark.parametrize(
    "lo,hi,step,start,inc",
    [
        (5, 13, 2, 0, 128),
        (3, 11, 2, 384, -128),
    ],
    ids=["nonzero_lb_step2", "negative_increment"],
)
def test_loop_carried_affine_range_prepass_axis0_count(lo, hi, step, start, inc):
    block = 128
    k = 5
    bk = triton.next_power_of_2(k)
    n = 401
    labels_cpu = (torch.arange(n, dtype=torch.int32) * 3 + 1) % k
    expected_counts, expected_trips = _expected_affine_range_counts(
        labels_cpu, lo, hi, step, start, inc, block, k
    )

    labels = labels_cpu.to("mps")
    counts_out = torch.full((bk,), -777, dtype=torch.int32, device="mps")
    trips_out = torch.full((bk,), -888, dtype=torch.int32, device="mps")
    _affine_range_axis0_count_kernel[(1,)](
        labels,
        counts_out,
        trips_out,
        k,
        lo,
        hi,
        n,
        START=start,
        INC=inc,
        STEP=step,
        BLOCK=block,
        BK=bk,
        num_warps=4,
    )
    torch.mps.synchronize()

    torch.testing.assert_close(counts_out[:k].cpu(), expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(
        trips_out[:k].cpu(), torch.full((k,), expected_trips, dtype=torch.int32), rtol=0, atol=0
    )
    assert torch.all(counts_out[k:].cpu() == -777)
    assert torch.all(trips_out[k:].cpu() == -888)


# ---------------------------------------------------------------------------
# A rank-3 reduce whose rank-2 result is STORED directly.
#
# The staged rank-N reduce may relabel its result so that its fold and its own
# addressing agree. The `ttg.convert_layout` between that result and a rank-2
# store becomes a tile exchange, and the exchange used to publish by the result
# TYPE's `#ttg.slice` encoding instead of by the plan the producer actually
# used -- so each thread's sum landed in another element's slot. It was exact
# while the reduced axis stayed inside one register band, which is every shape
# the earlier sweeps had, because they all consumed the result by broadcasting
# it back to rank 3 instead of storing it.


@triton.jit
def _rank3_reduce_rank2_store_kernel(X, O, A: tl.constexpr, B: tl.constexpr,
                                     C: tl.constexpr, AXIS: tl.constexpr):
    a = tl.arange(0, A)[:, None, None]
    b = tl.arange(0, B)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    v = tl.load(X + a * (B * C) + b * C + c)
    s = tl.sum(v, axis=AXIS)
    if AXIS == 0:
        r0 = tl.arange(0, B)[:, None]
        r1 = tl.arange(0, C)[None, :]
        tl.store(O + r0 * C + r1, s)
    elif AXIS == 1:
        r0 = tl.arange(0, A)[:, None]
        r1 = tl.arange(0, C)[None, :]
        tl.store(O + r0 * C + r1, s)
    else:
        r0 = tl.arange(0, A)[:, None]
        r1 = tl.arange(0, B)[None, :]
        tl.store(O + r0 * B + r1, s)


@pytest.mark.parametrize(
    "shape,num_warps",
    [
        ((4, 8, 8), 1),      # eight elements per thread: the smallest failure
        ((4, 8, 16), 1),     # sixteen
        ((4, 8, 16), 2),     # eight, reached with two warps
        ((8, 8, 8), 1),
        ((8, 8, 8), 2),
        ((2, 8, 16), 1),
        ((2, 4, 8), 1),      # four: correct before the fix, still correct
        ((4, 4, 4), 2),      # one element per thread: the control
    ],
    ids=["4x8x8_w1", "4x8x16_w1", "4x8x16_w2", "8x8x8_w1", "8x8x8_w2",
         "2x8x16_w1", "2x4x8_w1_control", "4x4x4_w2_control"],
)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rank3_reduce_rank2_store_matches_torch(shape, num_warps, axis):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    a, b, c = shape
    torch.manual_seed(0xC0DE)
    x = torch.randn(a * b * c, dtype=torch.float32).contiguous()
    expected = x.reshape(a, b, c).sum(axis)
    out = torch.zeros(expected.numel(), dtype=torch.float32).contiguous()
    _rank3_reduce_rank2_store_kernel[(1, 1, 1)](
        x, out, A=a, B=b, C=c, AXIS=axis, num_warps=num_warps)
    torch.testing.assert_close(out.reshape(expected.shape), expected,
                               atol=2e-3, rtol=2e-3)


@triton.jit
def _rank3_permuted_reduce_kernel(X, O, A: tl.constexpr, B: tl.constexpr,
                                  C: tl.constexpr):
    a = tl.arange(0, A)[:, None, None]
    b = tl.arange(0, B)[None, :, None]
    c = tl.arange(0, C)[None, None, :]
    v = tl.load(X + a * (B * C) + b * C + c)
    s = tl.sum(tl.permute(v, (0, 2, 1)), axis=2)
    r0 = tl.arange(0, A)[:, None]
    r1 = tl.arange(0, C)[None, :]
    tl.store(O + r0 * C + r1, s)


@pytest.mark.parametrize(
    "shape,num_warps",
    [((4, 8, 8), 1), ((8, 8, 8), 2), ((2, 4, 8), 1)],
    ids=["4x8x8_w1", "8x8x8_w2", "2x4x8_w1"],
)
def test_rank3_permuted_reduce_is_rejected_not_wrong(shape, num_warps,
                                                     tmp_path):
    # The staged publish addresses each element by its position in the ORIGINAL
    # tile and labels it by the tile's layout; a `tl.permute` in between breaks
    # that pairing, and above one register band the fold summed a scattered set
    # of elements. A stored `tl.permute` at rank >= 3 was already refused by
    # name, so the reduced form is refused too rather than being right only
    # where the two labellings happen to coincide.
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    a, b, c = shape
    script = tmp_path / "rank3_permuted_reduce.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        spec = importlib.util.spec_from_file_location(
            "rank2_child", {str(Path(__file__).resolve())!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = GPUTarget("metal", 80, 32)

        for attempt in range(3):
            try:
                triton.compile(ASTSource(
                    module._rank3_permuted_reduce_kernel,
                    signature={{"X": "*fp32", "O": "*fp32"}},
                    constexprs={{"A": {a}, "B": {b}, "C": {c}}}),
                    target=target, options={{"num_warps": {num_warps}}})
            except RuntimeError:
                pass
            else:
                raise AssertionError("a permuted rank-3 reduce compiled")
        # The unpermuted neighbour is supported and stays so.
        triton.compile(ASTSource(
            module._rank3_reduce_rank2_store_kernel,
            signature={{"X": "*fp32", "O": "*fp32"}},
            constexprs={{"A": {a}, "B": {b}, "C": {c}, "AXIS": 1}}),
            target=target, options={{"num_warps": {num_warps}}})
        print("RECOVERED")
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, timeout=180,
                            env={**os.environ,
                                 "TRITON_CACHE_DIR": str(tmp_path / "cache")})
    output = result.stdout + result.stderr
    needle = "rank >= 3 reduce over a permuted tile is not implemented"
    assert result.returncode == 0, output
    assert output.count(needle) == 3, output
    assert "RECOVERED" in output
    assert "failed to legalize" not in output


# The rank-3 defect above came from sweeping a producer only through the
# consumer its implementation was built against. Sweeping every staged producer
# through the consumers a user writes (store, one elementwise op, broadcast back,
# consumed twice) found the rank-2 families clean across E = 1..16 --
# `.omc/fuzz/consumer_shapes.py`, 132 cells, 123 OK and 9 named refusals for the
# documented axis-extent limit. These pin the widest geometry of that sweep so
# the blind spot cannot reopen one rank down.


@triton.jit
def _rank2_reduce_consumer_kernel(X, O, M: tl.constexpr, N: tl.constexpr,
                                  AXIS: tl.constexpr, HOW: tl.constexpr):
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    v = tl.load(X + r * N + c)
    s = tl.sum(v, axis=AXIS)
    if HOW == "broadcast":
        b = s[None, :] if AXIS == 0 else s[:, None]
        tl.store(O + r * N + c, v - b)
    elif HOW == "arith":
        if AXIS == 0:
            tl.store(O + tl.arange(0, N), s * 2.0 + 1.0)
        else:
            tl.store(O + tl.arange(0, M), s * 2.0 + 1.0)
    else:
        if AXIS == 0:
            tl.store(O + tl.arange(0, N), s)
        else:
            tl.store(O + tl.arange(0, M), s)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("how", ["store", "arith", "broadcast"])
@pytest.mark.parametrize(
    "shape,num_warps",
    [((16, 32), 1), ((32, 32), 2)],
    ids=["16x32_w1_e16", "32x32_w2_e16"],
)
def test_rank2_reduce_consumer_shapes_match_torch(axis, how, shape, num_warps):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal backend requires an MPS-enabled PyTorch")
    m, n = shape
    torch.manual_seed(0x513E)
    x = (torch.randn(m * n, dtype=torch.float32) * 0.1).contiguous()
    xm = x.reshape(m, n)
    s = xm.sum(axis)
    if how == "broadcast":
        expected = xm - (s[None, :] if axis == 0 else s[:, None])
    elif how == "arith":
        expected = s * 2.0 + 1.0
    else:
        expected = s
    out = torch.zeros(expected.numel(), dtype=torch.float32).contiguous()
    _rank2_reduce_consumer_kernel[(1, 1, 1)](
        x, out, M=m, N=n, AXIS=axis, HOW=how, num_warps=num_warps)
    torch.testing.assert_close(out.reshape(expected.shape), expected,
                               atol=2e-3, rtol=2e-3)
