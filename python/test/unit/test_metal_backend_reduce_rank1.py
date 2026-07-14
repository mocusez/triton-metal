"""Rank-1 tt.reduce on the Metal backend (f32/i32, sum/max).

All f32 rank-1 reduce cases (sum+max, BLOCK ∈ {32, 64, 128, 256, 512, 1024})
pass end-to-end via the spt-fold in `lowerRank1Reduce`.

Kernel shape: load a row of length N from a 1D `!tt.ptr`, reduce via
`tl.sum(row, axis=0)` or `tl.max(row, axis=0)`, store the resulting scalar.

Coverage:
  BLOCK_SIZE ∈ {32, 64, 128, 256, 512, 1024}
  dtype      ∈ {f32, i32}
  op         ∈ {sum, max}

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
