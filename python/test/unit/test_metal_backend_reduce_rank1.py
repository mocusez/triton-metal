"""Rank-1 tt.reduce on the Metal backend (f32/i32, sum/max).

The core f32 rank-1 reduce cases (sum+max,
BLOCK ∈ {32, 64, 128, 256, 512, 1024}) pass end-to-end via the spt-fold in
`lowerRank1Reduce`. A 4096-element constexpr-masked case and the Leet GroupNorm
entry point cover the larger cyclic-tile and loop-offset paths.

Kernel shape: load a row of length N from a 1D `!tt.ptr`, reduce via
`tl.sum(row, axis=0)` or `tl.max(row, axis=0)`, store the resulting scalar.

Coverage:
  BLOCK_SIZE ∈ {32, 64, 128, 256, 512, 1024}
  dtype      ∈ {f32, i32}
  op         ∈ {sum, max}
  integration: f32 sum with BLOCK=4096 and Leet GroupNorm

Notes:
  - f32 sum+max: all 12 parametrized cases pass.
  - i32 sum: all BLOCK pass. BLOCK <= threads_per_block (256) via the butterfly;
    BLOCK > 256 via the scf.for tile-loop, whose iter_arg accumulator is now
    emitted with its real integer type. The init/identity constant is built as
    a signless `arith.constant` (the verifier rejects signed/unsigned-typed
    integer constants) and then bridged to ui32 storage.
  - i32 max: all BLOCK pass. tl.max on i32 emits an `arith.maxsi` combine, now
    accepted by the convert-tritongpu-to-metal pre-pass and lowered via a
    signed (si32) butterfly/accumulator. The MSL `max` is emitted as
    `max(int32_t(a), int32_t(b))` so the comparison is signed even though i32
    is stored as ui32.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

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
def reduce_sum_rank1_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + 0, s)


@triton.jit
def reduce_sum_rank1_constexpr_mask_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    tl.store(out_ptr, tl.sum(x, axis=0))


@triton.jit
def reduce_max_rank1_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.max(x, axis=0)
    tl.store(out_ptr + 0, s)


@triton.jit
def reduce_min_rank1_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.min(x, axis=0)
    tl.store(out_ptr + 0, s)


@triton.jit
def argmax_rank1_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=float("-inf"))
    idx = tl.argmax(x, axis=0)
    tl.store(out_ptr + 0, idx)


@triton.jit
def argmax_per_row_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + offs, mask=offs < N,
                other=float("-inf"))
    idx = tl.argmax(x, axis=0)
    tl.store(out_ptr + row, idx)


@triton.jit
def moe_top_k_gating_kernel(
    logits_ptr,
    topk_w_ptr,
    topk_idx_ptr,
    E,
    K,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    logits = tl.load(logits_ptr + row * E + offs_e, mask=offs_e < E,
                     other=float("-inf"))
    offs_k = tl.arange(0, BLOCK_K)
    topk_vals = tl.full((BLOCK_K,), float("-inf"), tl.float32)
    topk_idxs = tl.full((BLOCK_K,), 0, tl.int32)

    for i in range(K):
        curr_max_val = tl.max(logits, axis=0)
        curr_max_idx = tl.argmax(logits, axis=0)
        topk_vals = tl.where(offs_k == i, curr_max_val, topk_vals)
        topk_idxs = tl.where(offs_k == i, curr_max_idx, topk_idxs)
        logits = tl.where(offs_e == curr_max_idx, float("-inf"), logits)

    max_val = tl.max(topk_vals, axis=0)
    topk_vals = tl.exp(topk_vals - max_val)
    topk_vals /= tl.sum(topk_vals, axis=0)
    tl.store(topk_w_ptr + row * K + offs_k, topk_vals, mask=offs_k < K)
    tl.store(topk_idx_ptr + row * K + offs_k, topk_idxs, mask=offs_k < K)


_NUM_WARPS = 8   # threads_per_block = num_warps * 32 = 256


def _block_params():
    return [pytest.param(BLOCK) for BLOCK in [32, 64, 128, 256, 512, 1024]]


@pytest.mark.parametrize("BLOCK", _block_params())
def test_reduce_sum_rank1_f32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((BLOCK,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_sum_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.sum(x.cpu())
    torch.testing.assert_close(out[0].item(), expected.item(),
                               atol=1e-3, rtol=1e-3)


def test_reduce_sum_rank1_f32_block4096_constexpr_mask():
    """A constexpr mask bound lowers from dense<splat>, not tt.splat."""
    N = 1000
    BLOCK = 4096
    torch.manual_seed(0x4096)
    x = torch.randn(N, dtype=torch.float32, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")
    reduce_sum_rank1_constexpr_mask_kernel[(1,)](
        x, out, N=N, BLOCK=BLOCK, num_warps=_NUM_WARPS
    )
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), x.cpu().sum().reshape(1),
                               atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("BLOCK", _block_params())
def test_reduce_max_rank1_f32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((BLOCK,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_max_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.max(x.cpu())
    torch.testing.assert_close(out[0].item(), expected.item(),
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("BLOCK", _block_params())
def test_reduce_sum_rank1_i32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randint(-100, 100, (BLOCK,), dtype=torch.int32).contiguous()
    out = torch.zeros((1,), dtype=torch.int32).contiguous()
    reduce_sum_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.sum(x.cpu()).to(torch.int32)
    assert out[0].item() == expected.item()


@pytest.mark.parametrize("BLOCK", _block_params())
def test_reduce_max_rank1_i32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randint(-100, 100, (BLOCK,), dtype=torch.int32).contiguous()
    out = torch.zeros((1,), dtype=torch.int32).contiguous()
    reduce_max_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.max(x.cpu()).to(torch.int32)
    assert out[0].item() == expected.item()


@pytest.mark.parametrize("BLOCK", _block_params())
def test_reduce_min_rank1_i32(BLOCK):
    # tl.min on i32 emits an arith.minsi combine, accepted by the pre-pass and
    # lowered via a signed (si32) accumulator with identity INT32_MAX. MSL emits
    # `min(int32_t(a), int32_t(b))` so the comparison is signed.
    torch.manual_seed(0xC0FFEE)
    x = torch.randint(-100, 100, (BLOCK,), dtype=torch.int32).contiguous()
    out = torch.zeros((1,), dtype=torch.int32).contiguous()
    reduce_min_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.min(x.cpu()).to(torch.int32)
    assert out[0].item() == expected.item()


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 7.0, 3.0, 7.0, -2.0], 1),
        ([float("-inf")] * 13, 0),
    ],
)
def test_argmax_rank1_f32_tie_break_left_and_padding(values, expected):
    x = torch.tensor(values, dtype=torch.float32, device="mps")
    out = torch.full((1,), -1, dtype=torch.int32, device="mps")
    argmax_rank1_kernel[(1,)](x, out, len(values), BLOCK=32, num_warps=4)
    torch.mps.synchronize()
    assert out.item() == expected


def test_argmax_rank1_f32_multiprogram():
    x = torch.tensor(
        [
            [1.0, 9.0, 3.0, 9.0, -2.0],
            [-4.0, -3.0, -2.0, -1.0, -5.0],
            [8.0, 7.0, 6.0, 5.0, 4.0],
        ],
        dtype=torch.float32,
        device="mps",
    )
    out = torch.full((x.shape[0],), -1, dtype=torch.int32, device="mps")
    argmax_per_row_kernel[(x.shape[0],)](
        x, out, x.shape[1], BLOCK=32, num_warps=4
    )
    torch.mps.synchronize()
    torch.testing.assert_close(
        out.cpu(), torch.tensor([1, 3, 0], dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "M, E, K", [(1, 13, 1), (4, 13, 4), (3, 32, 8), (2, 127, 16)]
)
def test_moe_top_k_gating(M, E, K):
    torch.manual_seed(M * 10000 + E * 100 + K)
    logits = torch.randn((M, E), dtype=torch.float32, device="mps")
    weights = torch.empty((M, K), dtype=torch.float32, device="mps")
    indices = torch.empty((M, K), dtype=torch.int32, device="mps")

    moe_top_k_gating_kernel[(M,)](
        logits,
        weights,
        indices,
        E,
        K,
        BLOCK_E=triton.next_power_of_2(E),
        BLOCK_K=triton.next_power_of_2(K),
        num_warps=4,
    )
    torch.mps.synchronize()

    ref_vals, ref_indices = torch.topk(logits.cpu(), K, dim=1)
    ref_weights = torch.softmax(ref_vals, dim=1)
    torch.testing.assert_close(indices.cpu(), ref_indices.to(torch.int32),
                               rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), ref_weights, rtol=1e-5, atol=1e-6)


# --- W-B: rich rank-1 reduce over a COMPUTED cone (select/cmp/andi/make_range +
# masked loads + per-program scalar base offset). These are the reduce shapes in
# medium-speculative_decoding_verification.py; the general per-element cone
# evaluator `evalRank1ValueAt` re-derives each logical element. BLOCK=1024 > tpb
# so the spt-fold / scf.for + butterfly path drives the evaluator per element. ---


@triton.jit
def _sum_relu_diff_kernel(p_ptr, q_ptr, out_ptr, V, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < V
    p = tl.load(p_ptr + b * V + idx, mask=mask, other=0.0)
    q = tl.load(q_ptr + b * V + idx, mask=mask, other=0.0)
    adj = tl.where(q > p, q - p, 0.0)
    adj = tl.where(mask, adj, 0.0)
    tl.store(out_ptr + b, tl.sum(adj, axis=0))


@pytest.mark.parametrize("B, V", [(1, 100), (4, 777), (3, 1024)])
def test_reduce_sum_computed_cone(B, V):
    torch.manual_seed(V)
    p = torch.rand((B, V), dtype=torch.float32, device="mps")
    q = torch.rand((B, V), dtype=torch.float32, device="mps")
    out = torch.zeros((B,), dtype=torch.float32, device="mps")
    _sum_relu_diff_kernel[(B,)](p, q, out, V, BLOCK=1024, num_warps=4)
    expected = torch.clamp(q - p, min=0.0).sum(dim=1)
    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)


@triton.jit
def _per_row_mean_kernel(X, Out, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    a = tl.load(X + row * N + cols, mask=cols < N, other=0.0)
    tl.store(Out + row, tl.sum(a, axis=0) / N)


@pytest.mark.parametrize("M, N", [(1, 128), (4, 128), (8, 300), (2, 1024)])
def test_reduce_per_row_multiprogram(M, N):
    # Multi-program per-row reduce: each program (grid=(M,)) reduces its OWN row
    # via the `X + row*N` scalar base offset. Regression for the masked-reduce
    # path dropping that offset (every program read row 0).
    torch.manual_seed(M * 1000 + N)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    _per_row_mean_kernel[(M,)](x, out, N, BLOCK=triton.next_power_of_2(N))
    torch.testing.assert_close(out.cpu(), x.cpu().mean(dim=1), atol=1e-4, rtol=1e-4)


@triton.jit
def _min_idx_ge_kernel(val_ptr, out_ptr, V, TARGET, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < V
    v = tl.load(val_ptr + b * V + idx, mask=mask, other=0.0)
    cond = (v >= TARGET) & mask          # arith.andi on i1 in the cone
    sel = tl.where(cond, idx, V)         # min-reduce over selected indices
    tl.store(out_ptr + b, tl.min(sel, axis=0))


@pytest.mark.parametrize("B, V", [(1, 100), (4, 777), (2, 1024)])
def test_reduce_min_idx_computed_cone(B, V):
    # Inverse-CDF-style: first index where a running value crosses TARGET, else V.
    torch.manual_seed(V + 1)
    val = torch.rand((B, V), dtype=torch.float32, device="mps").cumsum(dim=1)
    TARGET = 0.5 * val[:, -1].mean().item()
    out = torch.zeros((B,), dtype=torch.int32, device="mps")
    _min_idx_ge_kernel[(B,)](val, out, V, TARGET, BLOCK=1024, num_warps=4)
    idx = torch.arange(V)
    cond = val.cpu() >= TARGET
    sel = torch.where(cond, idx[None, :], torch.full_like(idx[None, :], V))
    expected = sel.min(dim=1).values.to(torch.int32)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


def _load_leet_group_norm_module():
    path = (Path(__file__).resolve().parents[3] / "leet-triton" /
            "medium-group-normalization.py")
    assert path.is_file(), f"required leet-triton fixture not present: {path}"
    spec = importlib.util.spec_from_file_location(
        "leet_medium_group_normalization", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "shape,groups",
    [
        ((1, 4, 2, 2), 2),
        ((1, 1, 64, 64), 1),
        ((1, 2, 64, 64), 1),
        ((8, 512, 64, 64), 32),
    ],
)
def test_leet_group_normalization(shape, groups):
    """The public Leet solve() compiles and matches PyTorch GroupNorm."""
    module = _load_leet_group_norm_module()
    n, c, h, w = shape
    eps = 1e-5
    torch.manual_seed(n * 1000 + c * 100 + h * 10 + w)
    x_cpu = torch.randn(shape, dtype=torch.float32)
    gamma_cpu = torch.randn(c, dtype=torch.float32)
    beta_cpu = torch.randn(c, dtype=torch.float32)
    expected = torch.nn.functional.group_norm(
        x_cpu, groups, gamma_cpu, beta_cpu, eps
    )

    x = x_cpu.to("mps")
    gamma = gamma_cpu.to("mps")
    beta = beta_cpu.to("mps")
    out = torch.empty_like(x)
    module.solve(x, gamma, beta, out, n, c, h, w, groups, eps)
    torch.mps.synchronize()

    torch.testing.assert_close(out.cpu(), expected, atol=5e-4, rtol=5e-4)
