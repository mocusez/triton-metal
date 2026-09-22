"""Rank-1 tt.reduce on the Metal backend (f32/i32 sum/extrema/product).

The core f32 rank-1 reduce cases (sum+max,
BLOCK ∈ {32, 64, 128, 256, 512, 1024}) pass end-to-end via the spt-fold in
`lowerRank1Reduce`. A 4096-element constexpr-masked case and the Leet GroupNorm
entry point cover the larger cyclic-tile and loop-offset paths.

Kernel shape: load a row of length N from a 1D `!tt.ptr`, reduce via
`tl.sum(row, axis=0)` or `tl.max(row, axis=0)`, store the resulting scalar.

Coverage:
  BLOCK_SIZE ∈ {32, 64, 128, 256, 512, 1024}
  dtype      ∈ {f32, i32}
  op         ∈ {sum, max, product}
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
def _product_combine(a, b):
    return a * b


@triton.jit
def reduce_product_rank1_kernel(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    product = tl.reduce(x, axis=0, combine_fn=_product_combine)
    tl.store(out_ptr, product)


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
def test_reduce_min_rank1_f32(BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((BLOCK,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_min_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    expected = torch.min(x.cpu())
    torch.testing.assert_close(out[0].item(), expected.item(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("BLOCK", [32, 512])
def test_reduce_max_rank1_f32_all_negative_infinity(BLOCK):
    x = torch.full((BLOCK,), float("-inf"), dtype=torch.float32)
    out = torch.zeros((1,), dtype=torch.float32)
    reduce_max_rank1_kernel[(1,)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    assert torch.isneginf(out[0])


@pytest.mark.parametrize("BLOCK", [32, 512])
def test_reduce_min_rank1_f32_all_positive_infinity(BLOCK):
    x = torch.full((BLOCK,), float("inf"), dtype=torch.float32)
    out = torch.zeros((1,), dtype=torch.float32)
    reduce_min_rank1_kernel[(1,)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    assert torch.isposinf(out[0])


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


@pytest.mark.parametrize("BLOCK", [32, 512])
def test_reduce_product_rank1_i32(BLOCK):
    x = torch.ones((BLOCK,), dtype=torch.int32)
    x[:4] = torch.tensor([-2, 3, -1, 2], dtype=torch.int32)
    out = torch.zeros((1,), dtype=torch.int32)
    reduce_product_rank1_kernel[(1,)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    assert out.item() == 12


def test_reduce_product_rank1_i32_wraps_modulo_32_bits():
    x = torch.ones((32,), dtype=torch.int32)
    x[:2] = 65536
    out = torch.full((1,), -1, dtype=torch.int32)
    reduce_product_rank1_kernel[(1,)](x, out, BLOCK=32, num_warps=_NUM_WARPS)
    assert out.item() == 0


@pytest.mark.parametrize("BLOCK", [32, 512])
def test_reduce_product_rank1_f32_direct_load(BLOCK):
    x = torch.ones((BLOCK,), dtype=torch.float32)
    x[:4] = torch.tensor([-2.0, 0.5, -1.0, 2.0], dtype=torch.float32)
    out = torch.zeros((1,), dtype=torch.float32)
    reduce_product_rank1_kernel[(1,)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    assert out.item() == 2.0


@pytest.mark.parametrize("BLOCK", [32, 512])
@pytest.mark.parametrize(
    "values, expected_class",
    [
        ([-1.0, 0.0], "negative-zero"),
        ([-1.0, -0.0], "positive-zero"),
        ([-1.0, float("inf")], "negative-infinity"),
        ([float("nan")], "nan"),
    ],
)
def test_reduce_product_rank1_f32_special_values(BLOCK, values, expected_class):
    x = torch.ones((BLOCK,), dtype=torch.float32)
    x[: len(values)] = torch.tensor(values, dtype=torch.float32)
    out = torch.zeros((1,), dtype=torch.float32)
    reduce_product_rank1_kernel[(1,)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)
    actual = out.cpu()
    if expected_class == "nan":
        assert torch.isnan(actual[0])
    elif expected_class == "negative-infinity":
        assert torch.isneginf(actual[0])
    else:
        expected_bits = 0x80000000 if expected_class == "negative-zero" else 0
        assert actual.view(torch.int32)[0].item() & 0xFFFFFFFF == expected_bits


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
    path = (Path(__file__).resolve().parent / "fixtures" / "metal_leet" /
            "medium-group-normalization.py")
    assert path.is_file(), f"required Metal Leet fixture not present: {path}"
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


# --- Bitwise integer combines (arith.andi / ori / xori) -------------------
#
# `tl.xor_sum` lowers to an arith.xori combine; and/or arrive through
# hand-written `tl.reduce` combines. All three are associative+commutative
# monoids, so they ride the same ui32 butterfly as arith.addi — the only new
# pieces are the identity (~0 for and, 0 for or/xor) and the bitwise
# metal.binary_exp operator (the pre-existing andOp/orOp are LOGICAL and emit
# `&&`/`||` into an i1). Verified bit-exact rather than by tolerance: a wrong
# identity or a logical-vs-bitwise mixup changes the answer exactly, never
# approximately.


@triton.jit
def _bitwise_and_combine(a, b):
    return a & b


@triton.jit
def _bitwise_or_combine(a, b):
    return a | b


@triton.jit
def reduce_xor_rank1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.xor_sum(tl.load(x_ptr + offs), 0))


@triton.jit
def reduce_and_rank1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.reduce(tl.load(x_ptr + offs), 0, _bitwise_and_combine))


@triton.jit
def reduce_or_rank1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.reduce(tl.load(x_ptr + offs), 0, _bitwise_or_combine))


def _wrap_i32(value: int) -> int:
    return (value + 2**31) % 2**32 - 2**31


@pytest.mark.parametrize("block", [32, 64, 256, 1024])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("op", ["xor", "and", "or"])
def test_reduce_rank1_bitwise_i32(block, num_warps, op):
    torch.manual_seed(block * 31 + num_warps)
    x_cpu = torch.randint(-(2**30), 2**30, (block,), dtype=torch.int32)
    x = x_cpu.to("mps")
    out = torch.zeros(1, dtype=torch.int32, device="mps")

    kernel, identity, apply = {
        "xor": (reduce_xor_rank1_kernel, 0, lambda a, b: a ^ b),
        "and": (reduce_and_rank1_kernel, -1, lambda a, b: a & b),
        "or": (reduce_or_rank1_kernel, 0, lambda a, b: a | b),
    }[op]
    kernel[(1,)](x, out, BLOCK=block, num_warps=num_warps)
    torch.mps.synchronize()

    expected = identity
    for value in x_cpu.tolist():
        expected = apply(expected, value)
    assert out.cpu().item() == _wrap_i32(expected)


# --- f16 / bf16 reduces --------------------------------------------------
#
# Half reduces in ITS OWN type rather than being promoted to f32. Triton's
# contract is that the combine runs in the tensor's element type; accumulating
# in f32 and truncating would make this backend disagree with every other one
# in a way no tolerance-based test would ever flag.
#
# Asserted BIT-EXACT by choosing half-integer inputs whose running total stays
# exactly representable, so the answer does not depend on association order. A
# tolerance here would pass just as happily on an f32-promoted accumulator,
# which is the bug this is guarding against.


@triton.jit
def reduce_sum_rank1_half_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.load(x_ptr + offs), axis=0))


@triton.jit
def reduce_max_rank1_half_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.max(tl.load(x_ptr + offs), axis=0))


@triton.jit
def reduce_sum_rank1_half_cone_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.load(x_ptr + offs) * 2.0, axis=0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block", [32, 64, 256, 1024])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_reduce_rank1_half_sum_is_exact(dtype, block, num_warps):
    torch.manual_seed(block + num_warps)
    x = torch.randint(-4, 5, (block,)).to(dtype) * 0.5
    exact = x.double().sum().item()
    assert abs(exact) < 256, "test inputs must keep the sum exactly representable"
    out = torch.zeros(1, dtype=dtype, device="mps")
    reduce_sum_rank1_half_kernel[(1,)](x.to("mps"), out, BLOCK=block,
                                       num_warps=num_warps)
    torch.mps.synchronize()
    assert out.cpu().double().item() == exact


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block", [32, 256, 1024])
def test_reduce_rank1_half_max_is_exact(dtype, block):
    torch.manual_seed(block)
    x = (torch.rand(block, dtype=dtype) - 0.5)
    out = torch.zeros(1, dtype=dtype, device="mps")
    reduce_max_rank1_half_kernel[(1,)](x.to("mps"), out, BLOCK=block)
    torch.mps.synchronize()
    # Max is an element selection, so it is exact whatever the identity is —
    # unless the tail fill uses a non-half -inf, which is the failure this
    # catches.
    assert out.cpu().item() == x.max().item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduce_rank1_half_computed_cone(dtype):
    """Rank-1 (unlike rank-2 axis=1) re-derives a computed cone per element;
    that re-emitter must keep the half type instead of widening."""
    torch.manual_seed(7)
    x = torch.randint(-4, 5, (64,)).to(dtype) * 0.5
    out = torch.zeros(1, dtype=dtype, device="mps")
    reduce_sum_rank1_half_cone_kernel[(1,)](x.to("mps"), out, BLOCK=64)
    torch.mps.synchronize()
    assert out.cpu().double().item() == (x.double() * 2).sum().item()


# ---------------------------------------------------------------------------
# Combines that are MORE THAN ONE OPERATION.
#
# Every combine above is a single binary op, and that was the blind spot: the
# backend identified the combine by taking the FIRST op of the region and
# reducing with that operator, discarding the rest. `tl.where(a*a > b*b, a, b)`
# has `arith.mulf` first, so a max-abs reduction silently returned the PRODUCT
# of its inputs — bit for bit, with no diagnostic. A single-op region is the one
# shape that walk gets right, so nothing in this file could catch it.
#
# The region is now replayed per element by `ReduceRankNLowering`, so what these
# assert is that a multi-op combine computes ITS OWN function.
# ---------------------------------------------------------------------------


@triton.jit
def _maxabs_combine(a, b):
    return tl.where(a * a > b * b, a, b)


@triton.jit
def _mul_plus_combine(a, b):
    # f(a, b) = (1 + a)(1 + b) - 1: associative and commutative, and its first
    # region op is arith.mulf — the shape that used to reduce as a product.
    return a * b + a + b


@triton.jit
def _add_headed_max_combine(a, b):
    # First region op is arith.addf, so this used to reduce as a SUM. Also
    # covers a literal captured from outside the combine region.
    return tl.maximum(a + 0.0, b)


@triton.jit
def reduce_maxabs_rank1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.reduce(tl.load(x_ptr + offs), 0, _maxabs_combine))


@triton.jit
def reduce_mulplus_rank1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.reduce(tl.load(x_ptr + offs), 0, _mul_plus_combine))


@triton.jit
def reduce_addheaded_rank1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.reduce(tl.load(x_ptr + offs), 0, _add_headed_max_combine))


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("block", [8, 32, 64])
def test_reduce_rank1_multi_op_combine_is_not_its_first_op(block, num_warps):
    if block > 32 * num_warps:
        pytest.skip("staged combine needs at most one element per thread")
    torch.manual_seed(block * 31 + num_warps)
    x = torch.rand(block) * 2 - 1
    out = torch.zeros(1, device="mps")
    reduce_maxabs_rank1_kernel[(1,)](x.to("mps"), out, BLOCK=block,
                                     num_warps=num_warps)
    torch.mps.synchronize()
    want = x[x.abs().argmax()].item()
    got = out.cpu().item()
    # The product is what the first-op walk produced; assert we are not that.
    assert got == pytest.approx(want, abs=1e-6), (
        f"max-abs combine gave {got}, want {want} "
        f"(product would be {x.double().prod().item()})")


@pytest.mark.parametrize("num_warps", [1, 2])
@pytest.mark.parametrize("block", [8, 32])
def test_reduce_rank1_mul_plus_combine(block, num_warps):
    if block > 32 * num_warps:
        pytest.skip("staged combine needs at most one element per thread")
    torch.manual_seed(block + num_warps)
    x = torch.rand(block) * 0.4 + 0.4
    out = torch.zeros(1, device="mps")
    reduce_mulplus_rank1_kernel[(1,)](x.to("mps"), out, BLOCK=block,
                                      num_warps=num_warps)
    torch.mps.synchronize()
    want = ((1.0 + x.double()).prod() - 1.0).item()
    assert out.cpu().item() == pytest.approx(want, rel=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_reduce_rank1_combine_with_captured_literal(dtype):
    """`tl.maximum(a + 0.0, b)` reduces as a MAX, not as the sum its first op
    suggests — and its f16 literal must be emitted as `half(...)`, or MSL's
    `max(float, half)` is ambiguous and the shader will not compile."""
    torch.manual_seed(11)
    x = torch.rand(32, dtype=dtype)
    out = torch.zeros(1, dtype=dtype, device="mps")
    reduce_addheaded_rank1_kernel[(1,)](x.to("mps"), out, BLOCK=32)
    torch.mps.synchronize()
    assert out.cpu().item() == x.max().item()


# ---------------------------------------------------------------------------
# Elementwise math in the reduced cone.
#
# The cone replayers (`evalRank1ValueAt` and its dry-run predicate) listed the
# unary math they could re-derive per element. exp2/log2/abs/floor/ceil and a
# plain negation were missing from that list even though each already had an
# exact Metal spelling for the elementwise path, so an ordinary
# `tl.sum(tl.sigmoid(x))` matched no reduce tier at all and failed to legalize
# AFTER FuncOpLowering had rewritten the function signature -- a process kill
# (measured 17 SIGSEGV / 30 runs), not a diagnostic.
# ---------------------------------------------------------------------------


@triton.jit
def _cone_exp2_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.exp2(tl.load(x_ptr + o)), axis=0))


@triton.jit
def _cone_log2_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.log2(tl.load(x_ptr + o)), axis=0))


@triton.jit
def _cone_absf_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.abs(tl.load(x_ptr + o)), axis=0))


@triton.jit
def _cone_floor_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.floor(tl.load(x_ptr + o)), axis=0))


@triton.jit
def _cone_ceil_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.ceil(tl.load(x_ptr + o)), axis=0))


@triton.jit
def _cone_neg_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(-tl.load(x_ptr + o), axis=0))


@triton.jit
def _cone_sigmoid_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.sigmoid(tl.load(x_ptr + o)), axis=0))


@triton.jit
def _cone_absi_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.sum(tl.abs(tl.load(x_ptr + o)), axis=0))


@triton.jit
def _cone_max_absf_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.max(tl.abs(tl.load(x_ptr + o)), axis=0))


_CONE_F32_CASES = {
    "exp2": (_cone_exp2_kernel, torch.exp2),
    "log2": (_cone_log2_kernel, torch.log2),
    "abs": (_cone_absf_kernel, torch.abs),
    "floor": (_cone_floor_kernel, torch.floor),
    "ceil": (_cone_ceil_kernel, torch.ceil),
    "neg": (_cone_neg_kernel, torch.neg),
    "sigmoid": (_cone_sigmoid_kernel, torch.sigmoid),
}


@pytest.mark.parametrize("case", sorted(_CONE_F32_CASES))
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("block", [32, 128])
def test_reduce_rank1_unary_math_cone(case, block, num_warps):
    """Every unary math op with a Metal spelling reduces through the cone."""
    kernel, ref = _CONE_F32_CASES[case]
    torch.manual_seed(block * 7 + num_warps)
    # Positive and bounded: log2 needs x > 0, and floor/ceil want a spread of
    # fractional parts rather than values that all round the same way.
    x = torch.rand(block) * 3.0 + 0.25
    out = torch.zeros(1, device="mps")
    kernel[(1,)](x.to("mps"), out, BLOCK=block, num_warps=num_warps)
    torch.mps.synchronize()
    want = ref(x.double()).sum().item()
    assert out.cpu().item() == pytest.approx(want, rel=2e-5, abs=1e-5)


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("block", [32, 128])
def test_reduce_rank1_integer_abs_cone(block, num_warps):
    """`tl.abs` on i32 is math.absi, which the translator reads back as SIGNED;
    a signless-to-storage slip here turns abs into the identity."""
    torch.manual_seed(block + num_warps)
    x = torch.randint(-500, 500, (block,), dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.int32, device="mps")
    _cone_absi_kernel[(1,)](x.to("mps"), out, BLOCK=block, num_warps=num_warps)
    torch.mps.synchronize()
    assert int(out.cpu().item()) == int(x.abs().sum().item())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reduce_rank1_absf_keeps_the_bit_preserving_half_path(dtype):
    """f16/bf16 abs is a sign-bit mask, not a float conversion; the cone must
    use the same emitter as the elementwise pattern or -0.0 and NaN payloads
    diverge between the two paths."""
    torch.manual_seed(5)
    x = torch.tensor([-0.0, 0.0, -1.5, 2.25, -3.75, 4.0, -5.5, 6.125] * 4,
                     dtype=dtype)
    out = torch.zeros(1, dtype=dtype, device="mps")
    _cone_max_absf_kernel[(1,)](x.to("mps"), out, BLOCK=32)
    torch.mps.synchronize()
    assert out.cpu().float().item() == x.abs().float().max().item()


# ---------------------------------------------------------------------------
# A masked load's `other` inside the reduced chain.
#
# The chain path substituted the COMBINER IDENTITY for the whole chained value
# of every masked-off lane, so an explicit `other` was silently dropped:
# `tl.sum(tl.exp(tl.load(p, mask, other=1.0)))` returned the sum over the
# unmasked lanes alone, where the interpreter oracle (and every other backend)
# adds exp(1.0) once per masked-off lane. The `other`-less form was correct by
# accident, and it was the only form the suite covered.
# ---------------------------------------------------------------------------


@triton.jit
def _masked_other_sum_exp_kernel(x_ptr, out_ptr, n_valid, BLOCK: tl.constexpr,
                                 OTHER: tl.constexpr):
    o = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + o, mask=o < n_valid, other=OTHER)
    tl.store(out_ptr, tl.sum(tl.exp(v), axis=0))


@triton.jit
def _masked_other_max_exp_kernel(x_ptr, out_ptr, n_valid, BLOCK: tl.constexpr,
                                 OTHER: tl.constexpr):
    o = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + o, mask=o < n_valid, other=OTHER)
    tl.store(out_ptr, tl.max(tl.exp(v), axis=0))


@triton.jit
def _masked_no_other_sum_exp_kernel(x_ptr, out_ptr, n_valid,
                                    BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + o, mask=o < n_valid)
    tl.store(out_ptr, tl.sum(tl.exp(v), axis=0))


@pytest.mark.parametrize("other", [0.0, 1.0, 5.0, -3.0])
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("block,n_valid", [(128, 77), (64, 64), (256, 1)])
def test_reduce_rank1_masked_other_survives_the_chain(block, n_valid, other,
                                                      num_warps):
    torch.manual_seed(block * 17 + n_valid)
    x = torch.randn(block) * 0.5
    full = x.clone()
    full[n_valid:] = other
    out = torch.zeros(1, device="mps")
    _masked_other_sum_exp_kernel[(1,)](x.to("mps"), out, n_valid, BLOCK=block,
                                       OTHER=other, num_warps=num_warps)
    torch.mps.synchronize()
    want = torch.exp(full.double()).sum().item()
    assert out.cpu().item() == pytest.approx(want, rel=2e-5)


@pytest.mark.parametrize("other", [-3.0, 5.0])
def test_reduce_rank1_masked_other_survives_the_chain_for_max(other):
    """A max reduce reads the same lane; -inf as the dropped-in identity hid
    the defect whenever `other` happened to be the smallest element."""
    torch.manual_seed(int(other) + 91)
    block, n_valid = 128, 77
    x = torch.randn(block) * 0.5
    full = x.clone()
    full[n_valid:] = other
    out = torch.zeros(1, device="mps")
    _masked_other_max_exp_kernel[(1,)](x.to("mps"), out, n_valid, BLOCK=block,
                                       OTHER=other, num_warps=1)
    torch.mps.synchronize()
    want = torch.exp(full.double()).max().item()
    assert out.cpu().item() == pytest.approx(want, rel=2e-5)


def test_reduce_rank1_masked_without_other_contributes_the_identity():
    """With no `other` Triton guarantees nothing about the masked-off value.
    The backend keeps contributing the combiner identity, which is what makes a
    masked reduce deterministic here; pin that so the `other` fix above cannot
    quietly turn it into a fold over this backend's zero fill."""
    torch.manual_seed(3)
    block, n_valid = 128, 77
    x = torch.randn(block) * 0.5
    out = torch.zeros(1, device="mps")
    _masked_no_other_sum_exp_kernel[(1,)](x.to("mps"), out, n_valid,
                                          BLOCK=block, num_warps=1)
    torch.mps.synchronize()
    want = torch.exp(x[:n_valid].double()).sum().item()
    assert out.cpu().item() == pytest.approx(want, rel=2e-5)


# ---------------------------------------------------------------------------
# A masked lane the program said nothing about contributes the combine
# identity, at rank one, with or without a chain in between.
#
# The chain path already did this; the direct path kept the zero the load's
# else branch fills with. So a maximum over an all-negative masked row returned
# zero while the identical kernel with one elementwise op before the reduce
# returned the maximum over the valid lanes. Making the two agree is also what
# lets a masked PRODUCT exist: its identity is one, and a single zeroed lane
# annihilates the product, which is why masked loads were refused there.
# ---------------------------------------------------------------------------


@triton.jit
def _masked_product_kernel(x_ptr, out_ptr, n_valid, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + o, mask=o < n_valid)
    tl.store(out_ptr, tl.reduce(v, 0, _mul_combine))


@triton.jit
def _mul_combine(a, b):
    return a * b


@triton.jit
def _masked_product_other_kernel(x_ptr, out_ptr, n_valid, BLOCK: tl.constexpr,
                                 OTHER: tl.constexpr):
    o = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + o, mask=o < n_valid, other=OTHER)
    tl.store(out_ptr, tl.reduce(v, 0, _mul_combine))


@triton.jit
def _masked_max_kernel(x_ptr, out_ptr, n_valid, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.max(tl.load(x_ptr + o, mask=o < n_valid), axis=0))


@triton.jit
def _masked_min_kernel(x_ptr, out_ptr, n_valid, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(out_ptr, tl.min(tl.load(x_ptr + o, mask=o < n_valid), axis=0))


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("block,n_valid", [(32, 21), (128, 77), (256, 256)])
def test_reduce_rank1_masked_product_without_other(block, n_valid, num_warps):
    """No `other`: the masked lanes take the multiplicative identity."""
    torch.manual_seed(block * 3 + n_valid)
    x = torch.rand(block) * 0.5 + 0.8
    out = torch.zeros(1, device="mps")
    _masked_product_kernel[(1,)](x.to("mps"), out, n_valid, BLOCK=block,
                                 num_warps=num_warps)
    torch.mps.synchronize()
    want = x[:n_valid].double().prod().item()
    assert out.cpu().item() == pytest.approx(want, rel=2e-4)


@pytest.mark.parametrize("other", [1.0, 2.0, 0.5])
def test_reduce_rank1_masked_product_with_other(other):
    """An explicit `other` multiplies in once per masked-off lane."""
    torch.manual_seed(int(other * 10) + 7)
    block, n_valid = 128, 77
    x = torch.rand(block) * 0.5 + 0.8
    full = x.clone()
    full[n_valid:] = other
    out = torch.zeros(1, device="mps")
    _masked_product_other_kernel[(1,)](x.to("mps"), out, n_valid, BLOCK=block,
                                       OTHER=other, num_warps=1)
    torch.mps.synchronize()
    want = full.double().prod().item()
    assert out.cpu().item() == pytest.approx(want, rel=2e-4)


@pytest.mark.parametrize("num_warps", [1, 4])
def test_reduce_rank1_masked_extrema_without_other(num_warps):
    """A masked maximum over all-negative data used to return the zero the
    load's else branch fills with; the chained form of the same kernel already
    returned the maximum over the valid lanes."""
    torch.manual_seed(29 + num_warps)
    block, n_valid = 128, 77
    neg = -(torch.rand(block) * 5.0 + 1.0)
    pos = torch.rand(block) * 5.0 + 1.0
    out = torch.zeros(1, device="mps")
    _masked_max_kernel[(1,)](neg.to("mps"), out, n_valid, BLOCK=block,
                             num_warps=num_warps)
    torch.mps.synchronize()
    assert out.cpu().item() == pytest.approx(neg[:n_valid].max().item())

    out = torch.zeros(1, device="mps")
    _masked_min_kernel[(1,)](pos.to("mps"), out, n_valid, BLOCK=block,
                             num_warps=num_warps)
    torch.mps.synchronize()
    assert out.cpu().item() == pytest.approx(pos[:n_valid].min().item())


# ---------------------------------------------------------------------------
# A general multi-value `tl.reduce((a, b, ...), 0, combine)`.
#
# Only the two shapes the frontend emits for `tl.argmax` / `tl.argmin` had a
# lowering; every other tuple combine was refused, and the refusal had to be
# made before conversion because declining inside it takes the process down.
# The lockstep butterfly generalises to N carried values by replaying the
# reduce's own region at each merge, so no combine has to be recognised.
#
# Two properties the tree needs that the argmax butterfly does not:
#   * merges run in LOGICAL order (stride 1, 2, 4, ...), because a tuple
#     combine is only required to be associative — see the first-non-zero test,
#     which the halving butterfly answers with a later index;
#   * a partner past the end of the block contributes NOTHING, since a general
#     combine has no identity to pad the tail with.
# ---------------------------------------------------------------------------


@triton.jit
def _tuple_minmax_combine(a_min, a_max, b_min, b_max):
    return tl.minimum(a_min, b_min), tl.maximum(a_max, b_max)


@triton.jit
def _tuple_minmax_kernel(x_ptr, min_ptr, max_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    lo, hi = tl.reduce((row, row), 0, _tuple_minmax_combine)
    tl.store(min_ptr, lo)
    tl.store(max_ptr, hi)


@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("block", [16, 32, 64, 128])
def test_reduce_rank1_tuple_min_and_max(block, num_warps):
    torch.manual_seed(block * 13 + num_warps)
    x = torch.randn(block)
    lo = torch.zeros(1, device="mps")
    hi = torch.zeros(1, device="mps")
    _tuple_minmax_kernel[(1,)](x.to("mps"), lo, hi, BLOCK=block,
                               num_warps=num_warps)
    torch.mps.synchronize()
    assert lo.cpu().item() == x.min().item()
    assert hi.cpu().item() == x.max().item()


@triton.jit
def _tuple_first_nonzero_combine(a_val, a_idx, b_val, b_idx):
    keep = a_val != 0.0
    return tl.where(keep, a_val, b_val), tl.where(keep, a_idx, b_idx)


@triton.jit
def _tuple_first_nonzero_kernel(x_ptr, val_ptr, idx_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    val, idx = tl.reduce((row, offs.to(tl.int32)), 0,
                         _tuple_first_nonzero_combine)
    tl.store(val_ptr, val)
    tl.store(idx_ptr, idx)


@pytest.mark.parametrize("num_warps,block,hits", [
    (4, 128, (96, 5, 70)),
    (2, 64, (48, 3)),
    (1, 32, (16, 1)),
    (4, 64, (40, 9)),
])
def test_reduce_rank1_tuple_merges_in_logical_order(block, num_warps, hits):
    """`first non-zero` is associative and order-SENSITIVE, unlike every
    combine the canonical pairs carry. A butterfly whose stride halves merges
    lane 0 with lane tpb/2 first and answers with a later element: with hits at
    {5, 70, 96} it reports 96."""
    x = torch.zeros(block)
    for h in hits:
        x[h] = float(h) + 0.5
    val = torch.zeros(1, device="mps")
    idx = torch.zeros(1, dtype=torch.int32, device="mps")
    _tuple_first_nonzero_kernel[(1,)](x.to("mps"), val, idx, BLOCK=block,
                                      num_warps=num_warps)
    torch.mps.synchronize()
    assert idx.cpu().item() == min(hits)
    assert val.cpu().item() == float(min(hits)) + 0.5


@triton.jit
def _tuple_argmax_combine(a_val, a_idx, b_val, b_idx):
    take_left = a_val > b_val
    return tl.where(take_left, a_val, b_val), tl.where(take_left, a_idx, b_idx)


@triton.jit
def _tuple_argmax_kernel(x_ptr, val_ptr, idx_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    val, idx = tl.reduce((row, offs.to(tl.int32)), 0, _tuple_argmax_combine)
    tl.store(val_ptr, val)
    tl.store(idx_ptr, idx)


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("block", [32, 128])
def test_reduce_rank1_tuple_hand_written_argmax(block, num_warps):
    """The same reduction `tl.argmax` performs, written by hand. It does not
    match the canonical structural matcher, so it exercises the general path."""
    torch.manual_seed(block + num_warps * 7)
    x = torch.randn(block)
    val = torch.zeros(1, device="mps")
    idx = torch.zeros(1, dtype=torch.int32, device="mps")
    _tuple_argmax_kernel[(1,)](x.to("mps"), val, idx, BLOCK=block,
                               num_warps=num_warps)
    torch.mps.synchronize()
    assert val.cpu().item() == x.max().item()
    assert idx.cpu().item() == x.argmax().item()


@triton.jit
def _tuple_argmax_high_tie_combine(a_val, a_idx, b_val, b_idx):
    take_left = (a_val > b_val) | ((a_val == b_val) & (a_idx > b_idx))
    return tl.where(take_left, a_val, b_val), tl.where(take_left, a_idx, b_idx)


@triton.jit
def _tuple_argmax_high_tie_kernel(x_ptr, idx_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    _, idx = tl.reduce((row, offs.to(tl.int32)), 0,
                       _tuple_argmax_high_tie_combine)
    tl.store(idx_ptr, idx)


def test_reduce_rank1_tuple_tie_breaks_to_the_higher_index():
    """An argmax that resolves ties the OTHER way. It must not be claimed by
    the canonical lowering, whose ties go to the lower index; the general path
    replays this region, so it answers 40 rather than 7."""
    x = torch.zeros(64)
    x[7] = 3.0
    x[40] = 3.0
    idx = torch.zeros(1, dtype=torch.int32, device="mps")
    _tuple_argmax_high_tie_kernel[(1,)](x.to("mps"), idx, BLOCK=64,
                                        num_warps=4)
    torch.mps.synchronize()
    assert idx.cpu().item() == 40


@triton.jit
def _tuple_three_combine(a_sum, a_min, a_max, b_sum, b_min, b_max):
    return a_sum + b_sum, tl.minimum(a_min, b_min), tl.maximum(a_max, b_max)


@triton.jit
def _tuple_three_kernel(x_ptr, sum_ptr, min_ptr, max_ptr,
                        BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    total, lo, hi = tl.reduce((row, row, row), 0, _tuple_three_combine)
    tl.store(sum_ptr, total)
    tl.store(min_ptr, lo)
    tl.store(max_ptr, hi)


@pytest.mark.parametrize("block,num_warps", [(64, 2), (256, 8)])
def test_reduce_rank1_tuple_of_three_values(block, num_warps):
    torch.manual_seed(block)
    x = torch.randn(block)
    total = torch.zeros(1, device="mps")
    lo = torch.zeros(1, device="mps")
    hi = torch.zeros(1, device="mps")
    _tuple_three_kernel[(1,)](x.to("mps"), total, lo, hi, BLOCK=block,
                              num_warps=num_warps)
    torch.mps.synchronize()
    assert total.cpu().item() == pytest.approx(x.double().sum().item(),
                                               rel=2e-5)
    assert lo.cpu().item() == x.min().item()
    assert hi.cpu().item() == x.max().item()


@triton.jit
def _tuple_sum_count_combine(a_sum, a_count, b_sum, b_count):
    return a_sum + b_sum, a_count + b_count


@triton.jit
def _tuple_sum_count_kernel(x_ptr, sum_ptr, count_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    positive = (row > 0.0).to(tl.int32)
    total, count = tl.reduce((row, positive), 0, _tuple_sum_count_combine)
    tl.store(sum_ptr, total)
    tl.store(count_ptr, count)


@pytest.mark.parametrize("block,num_warps", [(32, 1), (128, 4)])
def test_reduce_rank1_tuple_mixes_f32_and_i32(block, num_warps):
    """The i32 slot rides a ui32 threadgroup buffer while the f32 slot does
    not, so the two bridges have to agree per value, not per reduce."""
    torch.manual_seed(block * 3)
    x = torch.randn(block)
    total = torch.zeros(1, device="mps")
    count = torch.zeros(1, dtype=torch.int32, device="mps")
    _tuple_sum_count_kernel[(1,)](x.to("mps"), total, count, BLOCK=block,
                                  num_warps=num_warps)
    torch.mps.synchronize()
    assert total.cpu().item() == pytest.approx(x.double().sum().item(),
                                               rel=2e-5)
    assert count.cpu().item() == int((x > 0).sum().item())


@triton.jit
def _tuple_i32_minmax_kernel(x_ptr, min_ptr, max_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    lo, hi = tl.reduce((row, row), 0, _tuple_minmax_combine)
    tl.store(min_ptr, lo)
    tl.store(max_ptr, hi)


def test_reduce_rank1_tuple_signed_i32_pair():
    """Negative values decide it: the slots are stored as ui32, so a signed
    comparison has to survive the round trip."""
    torch.manual_seed(21)
    x = torch.randint(-1000, 1000, (128,), dtype=torch.int32)
    lo = torch.zeros(1, dtype=torch.int32, device="mps")
    hi = torch.zeros(1, dtype=torch.int32, device="mps")
    _tuple_i32_minmax_kernel[(1,)](x.to("mps"), lo, hi, BLOCK=128,
                                   num_warps=4)
    torch.mps.synchronize()
    assert lo.cpu().item() == x.min().item()
    assert hi.cpu().item() == x.max().item()


@triton.jit
def _tuple_minmax_row_kernel(x_ptr, min_ptr, max_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + pid * BLOCK + offs)
    lo, hi = tl.reduce((row, row), 0, _tuple_minmax_combine)
    tl.store(min_ptr + pid, lo)
    tl.store(max_ptr + pid, hi)


def test_reduce_rank1_tuple_multiprogram():
    """Each threadgroup must address its own slice of the scratch buffers by a
    LOCAL thread id, or the rows overwrite each other."""
    torch.manual_seed(33)
    rows, block = 5, 64
    x = torch.randn(rows, block)
    lo = torch.zeros(rows, device="mps")
    hi = torch.zeros(rows, device="mps")
    _tuple_minmax_row_kernel[(rows,)](x.to("mps").contiguous(), lo, hi,
                                      BLOCK=block, num_warps=2)
    torch.mps.synchronize()
    assert torch.equal(lo.cpu(), x.min(dim=1).values)
    assert torch.equal(hi.cpu(), x.max(dim=1).values)


def test_reduce_rank1_tuple_beyond_the_band_envelope_is_named(capfd):
    """Outside the envelope the refusal has to come from the pre-conversion
    validator: a pattern that declines inside `applyFullConversion` does not
    raise, it takes the process down, so this asserts a catchable error, the
    named reason on the diagnostic stream, AND that the interpreter still runs
    a kernel afterwards.

    More than one element per thread is no longer outside it -- a thread folds
    its own band first -- so the boundary this pins is the band LENGTH."""
    x = torch.randn(4096).to("mps")
    lo = torch.zeros(1, device="mps")
    hi = torch.zeros(1, device="mps")
    with pytest.raises(Exception) as error:
        _tuple_minmax_kernel[(1,)](x, lo, hi, BLOCK=4096, num_warps=1)
    assert "convert-tritongpu-to-metal failed" in str(error.value)
    diagnostics = capfd.readouterr()
    reason = diagnostics.err + diagnostics.out
    assert "multi-value tt.reduce" in reason
    assert "sixty-four elements per thread" in reason

    survivor = torch.arange(32, dtype=torch.float32).to("mps")
    out = torch.zeros(1, device="mps")
    reduce_sum_rank1_kernel[(1,)](survivor, out, BLOCK=32)
    torch.mps.synchronize()
    assert out.cpu().item() == 496.0


# ---------------------------------------------------------------------------
# More than one element per thread.
#
# This is where `tl.argmax` over an ordinary row lived: the canonical pair
# lowering owns one buffer slot per thread, so a block bigger than the
# threadgroup had nowhere to put a thread's second element and every such
# kernel was refused -- `tl.argmax` over 1024 elements included.
#
# A thread now folds its OWN elements first and the cross-thread tree runs
# unchanged over the partials. That is only order-correct because this path
# picks the mapping: thread t takes the contiguous band [t*E, (t+1)*E), so
# merging threads in order merges positions in order. Hand the threads the
# interleaved mapping instead and an order-sensitive combine goes wrong while
# every commutative one -- and the canonical argmax, whose ties resolve by
# index -- stays right.
# ---------------------------------------------------------------------------


@triton.jit
def _argmax_block_kernel(x_ptr, val_ptr, idx_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    row = tl.load(x_ptr + offs)
    tl.store(val_ptr, tl.max(row, axis=0))
    tl.store(idx_ptr, tl.argmax(row, axis=0))


@pytest.mark.parametrize("num_warps", [1, 4, 8])
@pytest.mark.parametrize("block", [64, 128, 512, 1024, 4096])
def test_argmax_rank1_above_one_element_per_thread(block, num_warps):
    tpb = 32 * num_warps
    if block > tpb and (block % tpb or block // tpb > 64):
        pytest.skip("outside the band envelope")
    torch.manual_seed(block * 3 + num_warps)
    x = torch.randn(block)
    val = torch.zeros(1, device="mps")
    idx = torch.zeros(1, dtype=torch.int32, device="mps")
    _argmax_block_kernel[(1,)](x.to("mps"), val, idx, BLOCK=block,
                               num_warps=num_warps)
    torch.mps.synchronize()
    assert val.cpu().item() == x.max().item()
    assert idx.cpu().item() == x.argmax().item()


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("block", [128, 1024])
def test_argmax_rank1_above_one_element_per_thread_ties_go_left(block,
                                                                num_warps):
    """Ties resolve to the lower index. Across a band fold that is the pair the
    thread keeps; across the tree it is the partner on the left."""
    tpb = 32 * num_warps
    if block > tpb and (block % tpb or block // tpb > 64):
        pytest.skip("outside the band envelope")
    x = torch.zeros(block)
    first, second = block // 3, 2 * block // 3
    x[first] = 5.0
    x[second] = 5.0
    val = torch.zeros(1, device="mps")
    idx = torch.zeros(1, dtype=torch.int32, device="mps")
    _argmax_block_kernel[(1,)](x.to("mps"), val, idx, BLOCK=block,
                               num_warps=num_warps)
    torch.mps.synchronize()
    assert val.cpu().item() == 5.0
    assert idx.cpu().item() == first


@pytest.mark.parametrize("num_warps", [1, 4, 8])
@pytest.mark.parametrize("block", [64, 256, 2048])
def test_reduce_rank1_tuple_above_one_element_per_thread(block, num_warps):
    tpb = 32 * num_warps
    if block > tpb and (block % tpb or block // tpb > 64):
        pytest.skip("outside the band envelope")
    torch.manual_seed(block + num_warps * 5)
    x = torch.randn(block)
    lo = torch.zeros(1, device="mps")
    hi = torch.zeros(1, device="mps")
    _tuple_minmax_kernel[(1,)](x.to("mps"), lo, hi, BLOCK=block,
                               num_warps=num_warps)
    torch.mps.synchronize()
    assert lo.cpu().item() == x.min().item()
    assert hi.cpu().item() == x.max().item()


@pytest.mark.parametrize("num_warps,block,hits", [
    (1, 128, (96, 5, 70)),
    (1, 512, (400, 33, 260)),
    (4, 1024, (900, 17, 500)),
    (8, 4096, (3000, 129, 2048)),
])
def test_reduce_rank1_tuple_band_fold_keeps_the_logical_order(block, num_warps,
                                                              hits):
    """The order-sensitive probe, now above one element per thread. With the
    interleaved mapping instead of the contiguous band this reports a later
    index; with the canonical argmax it would report the same either way, which
    is why the combine here is `first non-zero` and not an argmax."""
    x = torch.zeros(block)
    for h in hits:
        x[h] = float(h) + 0.5
    val = torch.zeros(1, device="mps")
    idx = torch.zeros(1, dtype=torch.int32, device="mps")
    _tuple_first_nonzero_kernel[(1,)](x.to("mps"), val, idx, BLOCK=block,
                                      num_warps=num_warps)
    torch.mps.synchronize()
    assert idx.cpu().item() == min(hits)
    assert val.cpu().item() == float(min(hits)) + 0.5


def test_reduce_rank1_tuple_above_one_element_per_thread_multiprogram():
    """Each threadgroup folds its own row and addresses the scratch by a LOCAL
    thread id; the band base is a local offset too."""
    torch.manual_seed(77)
    rows, block = 4, 256
    x = torch.randn(rows, block)
    lo = torch.zeros(rows, device="mps")
    hi = torch.zeros(rows, device="mps")
    _tuple_minmax_row_kernel[(rows,)](x.to("mps").contiguous(), lo, hi,
                                      BLOCK=block, num_warps=2)
    torch.mps.synchronize()
    assert torch.equal(lo.cpu(), x.min(dim=1).values)
    assert torch.equal(hi.cpu(), x.max(dim=1).values)


def test_reduce_rank1_tuple_of_three_above_one_element_per_thread():
    torch.manual_seed(78)
    block = 1024
    x = torch.randn(block)
    total = torch.zeros(1, device="mps")
    lo = torch.zeros(1, device="mps")
    hi = torch.zeros(1, device="mps")
    _tuple_three_kernel[(1,)](x.to("mps"), total, lo, hi, BLOCK=block,
                              num_warps=4)
    torch.mps.synchronize()
    assert total.cpu().item() == pytest.approx(x.double().sum().item(),
                                               rel=2e-5)
    assert lo.cpu().item() == x.min().item()
    assert hi.cpu().item() == x.max().item()


# ---------------------------------------------------------------------------
# A rank-1 aggregate whose SOURCE BUFFER the same kernel writes.
#
# Above one element per thread the body runs inside a band loop while a
# whole-CTA aggregate is materialized in the prologue, so the aggregate's read
# and the kernel's store end up on opposite sides of a boundary the source
# program never had. `sweeps/s_scan_source_effects.py` measured it over op x
# write target x order x BLOCK x warps, 168 cells: 27 silently wrong, and every
# one of them writes the aggregate's OWN buffer at more than one element per
# thread. Writing any other buffer is exact, rank 2 is exact, and one element
# per thread is exact -- so the refusals below are about the pairing, not about
# stores or aggregates.
#
# The `tl.sum` case is asymmetric on purpose, and the sweep is why: with the
# store AFTER the load the prologue read is the correct one, and that shape is
# an in-place softmax (load a row, reduce it, write the row back). Only a store
# that PRECEDES the load is refused there.
# ---------------------------------------------------------------------------


@triton.jit
def _scan_then_overwrite_source(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    c = tl.cumsum(v, 0)
    tl.store(X + i, v * 0.0)
    tl.store(O + i, c)


@triton.jit
def _scan_into_another_buffer(X, Y, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    c = tl.cumsum(v, 0)
    tl.store(Y + i, v * 0.0)
    tl.store(O + i, c)


@triton.jit
def _overwrite_source_then_reduce(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(X + i, tl.zeros((BLOCK,), tl.float32))
    v = tl.load(X + i)
    tl.store(O + i, v * 0.0 + tl.sum(v, 0))


@triton.jit
def _reduce_then_overwrite_source(X, O, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(X + i)
    s = tl.sum(v, 0)
    tl.store(X + i, v * 0.0)
    tl.store(O + i, v * 0.0 + s)


@pytest.mark.parametrize("block,num_warps", [(64, 1), (128, 1), (256, 4)],
                         ids=["n64_w1_e2", "n128_w1_e4", "n256_w4_e2"])
def test_rank1_scan_over_a_buffer_the_kernel_writes_is_named(block, num_warps,
                                                             capfd):
    x = torch.arange(block, dtype=torch.float32).to("mps")
    out = torch.zeros(block, device="mps")
    with pytest.raises(Exception) as error:
        _scan_then_overwrite_source[(1,)](x, out, BLOCK=block,
                                          num_warps=num_warps)
    assert "convert-tritongpu-to-metal failed" in str(error.value)
    reason = "".join(capfd.readouterr())
    assert "rank-1 scan reads a buffer the kernel also writes" in reason

    survivor = torch.arange(32, dtype=torch.float32).to("mps")
    kept = torch.zeros(1, device="mps")
    reduce_sum_rank1_kernel[(1,)](survivor, kept, BLOCK=32)
    torch.mps.synchronize()
    assert kept.cpu().item() == 496.0


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_rank1_scan_over_its_own_buffer_runs_at_one_element_per_thread(
        num_warps):
    """The guard on the refusal: at one element per thread there is no band
    loop for the read and the store to disagree about."""
    block = 32 * num_warps
    ref = torch.arange(block, dtype=torch.float32)
    x = ref.clone().to("mps")
    out = torch.zeros(block, device="mps")
    _scan_then_overwrite_source[(1,)](x, out, BLOCK=block, num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cumsum(0))
    assert torch.equal(x.cpu(), torch.zeros(block))


@pytest.mark.parametrize("block,num_warps", [(128, 1), (256, 4)],
                         ids=["n128_w1_e4", "n256_w4_e2"])
def test_rank1_scan_writing_another_buffer_still_runs(block, num_warps):
    """The other half of the guard: the refusal is about the aggregate's own
    buffer, and a kernel that scans one buffer and writes a second is the
    common shape."""
    ref = torch.arange(block, dtype=torch.float32)
    x = ref.clone().to("mps")
    y = torch.ones(block, device="mps")
    out = torch.zeros(block, device="mps")
    _scan_into_another_buffer[(1,)](x, y, out, BLOCK=block,
                                    num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cumsum(0))
    assert torch.equal(x.cpu(), ref)


@pytest.mark.parametrize("block,num_warps", [(64, 1), (128, 1), (256, 4)],
                         ids=["n64_w1_e2", "n128_w1_e4", "n256_w4_e2"])
def test_rank1_reduce_after_writing_its_source_is_named(block, num_warps,
                                                        capfd):
    x = torch.arange(block, dtype=torch.float32).to("mps")
    out = torch.zeros(block, device="mps")
    with pytest.raises(Exception) as error:
        _overwrite_source_then_reduce[(1,)](x, out, BLOCK=block,
                                            num_warps=num_warps)
    assert "convert-tritongpu-to-metal failed" in str(error.value)
    reason = "".join(capfd.readouterr())
    assert "rank-1 reduce reads a buffer the kernel wrote earlier" in reason


@pytest.mark.parametrize("block,num_warps", [(128, 1), (256, 4), (1024, 1)],
                         ids=["n128_w1_e4", "n256_w4_e2", "n1024_w1_e32"])
def test_rank1_reduce_then_overwriting_its_source_still_runs(block, num_warps):
    """The in-place softmax shape -- load a row, reduce it, write the row back
    -- and the reason the reduce refusal asks about ORDER instead of matching
    the scan's."""
    ref = torch.arange(block, dtype=torch.float32)
    x = ref.clone().to("mps")
    out = torch.zeros(block, device="mps")
    _reduce_then_overwrite_source[(1,)](x, out, BLOCK=block,
                                        num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(),
                               torch.full((block,), float(ref.sum())))
    assert torch.equal(x.cpu(), torch.zeros(block))
