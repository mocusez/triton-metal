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
