"""Rank-1 tt.reduce on the Metal backend with BLOCK > tpb (Wall 14).

Tutorial02 autotune sweep produces BLOCK_SIZE = next_power_of_2(n_cols) up
to 16384. With num_warps=8 and threads_per_warp=32, tpb=256, so the new
envelope covers E = BLOCK/tpb up to 64. This file exercises the bounded-
unroll path at three representative shapes:

  (N=2048,  BLOCK=2048)    — E=8     simplest BLOCK > tpb
  (N=4097,  BLOCK=8192)    — E=32    masked (n_cols < BLOCK)
  (N=12672, BLOCK=16384)   — E=64    autotune endpoint

Both `tl.sum` and `tl.max` combiners are exercised ⇒ 6 sub-tests.
Single-value blocked and flattened linear-layout replay additionally cover
E=128/256, with E=512 kept as a named rejection. Tuple and scan envelopes
are independent.

See the implementation notes AC8.
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
def reduce_sum_rank1_masked_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + 0, s)


@triton.jit
def reduce_max_rank1_masked_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    NEG_INF = float("-inf")
    x = tl.load(x_ptr + offs, mask=mask, other=NEG_INF)
    s = tl.max(x, axis=0)
    tl.store(out_ptr + 0, s)


_NUM_WARPS = 8  # threads_per_block = num_warps * warp_size = 8 * 32 = 256


_SHAPES = [
    # (N, BLOCK) — N is the actual length; BLOCK is next_power_of_2(N).
    (2048, 2048),    # E = 2048 / 256 = 8
    (4097, 8192),    # E = 8192 / 256 = 32, masked (n_cols < BLOCK)
    (12672, 16384),  # E = 16384 / 256 = 64, autotune endpoint
]


@pytest.mark.parametrize("N,BLOCK", _SHAPES)
def test_reduce_sum_rank1_f32_block_gt_tpb(N, BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((N,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_sum_rank1_masked_kernel[(1, 1, 1)](
        x, out, N, BLOCK=BLOCK, num_warps=_NUM_WARPS
    )
    expected = torch.sum(x.cpu())
    torch.testing.assert_close(
        out[0].item(), expected.item(), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("N,BLOCK", _SHAPES)
def test_reduce_max_rank1_f32_block_gt_tpb(N, BLOCK):
    torch.manual_seed(0xC0FFEE)
    x = torch.randn((N,), dtype=torch.float32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    reduce_max_rank1_masked_kernel[(1, 1, 1)](
        x, out, N, BLOCK=BLOCK, num_warps=_NUM_WARPS
    )
    expected = torch.max(x.cpu())
    torch.testing.assert_close(
        out[0].item(), expected.item(), atol=1e-5, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Replayed source addresses at more than one element per thread.
#
# Above one element per thread the reduce rebuilds each element's address. The
# single-load walker can only spell a contiguous range plus scalar pointer
# offsets; every other spelling below is handed to the general per-element cone
# evaluator, which walks the same pointer chain. At one element per thread none
# of this runs — the band's converted scalar is read directly — so each form is
# also checked there as the control that the two geometries agree.
#
# The adjacent rejection (an index neither path can rebuild) is
# test_metal_backend_msl.py::test_reduce_rank1_address_shape_unsupported_fails_without_crashing.


@triton.jit
def _reduce_rank1_address_form_kernel(
    X,
    IDX,
    O,
    STRIDE,
    N,
    BLOCK: tl.constexpr,
    FORM: tl.constexpr,
    OP: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    if FORM == "contiguous":
        values = tl.load(X + offs)
    elif FORM == "runtime_stride":
        values = tl.load(X + offs * STRIDE)
    elif FORM == "constant_stride":
        values = tl.load(X + offs * 3)
    elif FORM == "modulo":
        values = tl.load(X + offs % 8)
    elif FORM == "index_shift":
        values = tl.load(X + (offs + 8))
    elif FORM == "reversed":
        values = tl.load(X + (BLOCK - 1 - offs))
    elif FORM == "gather":
        values = tl.load(X + tl.load(IDX + offs))
    elif FORM == "nested_levels":
        base = X + offs * STRIDE
        values = tl.load(base + offs)
    elif FORM == "masked_shift":
        values = tl.load(X + (offs + 8), mask=offs < N, other=0.0)
    elif FORM == "masked_shift_other":
        # `other` deliberately away from both monoid identities: 0.0 is add's
        # and -inf is max's, so either one would hide a dropped `other`.
        values = tl.load(X + (offs + 8), mask=offs < N, other=7.5)
    else:  # splat_base: every lane reads the same element
        values = tl.load(X + (offs - offs))
    result = tl.max(values, axis=0) if OP == "max" else tl.sum(values, axis=0)
    tl.store(O, result)


_ADDRESS_FORMS = [
    "contiguous",
    "runtime_stride",
    "constant_stride",
    "modulo",
    "index_shift",
    "reversed",
    "gather",
    "nested_levels",
    "masked_shift",
    "masked_shift_other",
    "splat_base",
]


def _address_form_reference(form, x, idx, stride, n, block):
    if form == "contiguous":
        return x[:block]
    if form == "runtime_stride":
        return x[: block * stride : stride]
    if form == "constant_stride":
        return x[: block * 3 : 3]
    if form == "modulo":
        return x[:8].repeat(block // 8)
    if form == "index_shift":
        return x[8 : 8 + block]
    if form == "reversed":
        return x[:block].flip(0)
    if form == "gather":
        return x[idx.long()]
    if form == "nested_levels":
        return x[(torch.arange(block) * stride + torch.arange(block)).long()]
    if form == "masked_shift":
        shifted = x[8 : 8 + block].clone()
        shifted[n:] = 0.0
        return shifted
    if form == "masked_shift_other":
        shifted = x[8 : 8 + block].clone()
        shifted[n:] = 7.5
        return shifted
    return x[:1].repeat(block)


@pytest.mark.parametrize("form", _ADDRESS_FORMS)
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize(
    "block,num_warps",
    [(128, 1), (256, 2), (64, 2)],
    ids=["e4", "e4_w2", "e1"],
)
def test_reduce_rank1_replayed_address_forms(form, op, block, num_warps):
    torch.manual_seed(0xA11CE)
    stride = 3
    n = block - block // 4
    # Long enough for every stride/shift above to stay inside the tensor.
    x = torch.randn((block * (stride + 2),), dtype=torch.float32).contiguous()
    idx = torch.randperm(block).to(torch.int32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    _reduce_rank1_address_form_kernel[(1, 1, 1)](
        x, idx, out, stride, n, BLOCK=block, FORM=form, OP=op,
        num_warps=num_warps,
    )
    ref = _address_form_reference(form, x.cpu(), idx.cpu(), stride, n, block)
    expected = ref.max() if op == "max" else ref.sum()
    torch.testing.assert_close(
        out[0].item(), expected.item(), atol=2e-3, rtol=2e-3
    )


# ---------------------------------------------------------------------------
# Replayed masks and `other` values at more than one element per thread.
#
# The legacy single-load fold rebuilds only `cmpi slt(range, splat(N))` with a
# splat-constant `other`. Everything below is handed to the general cone
# evaluator, which replays the mask and `other` cones per element like any other
# value. `other` is 7.5 throughout, away from both the add identity (0) and the
# max identity (-inf), so a dropped `other` cannot hide inside the combine.
#
# The adjacent rejections are a mask the evaluator cannot rebuild either
# (reduce_unsupported.mlir::@reduce_rank1_masked_loop_carried_deferred) and a
# per-lane `other`, which `tt.load` itself refuses.


@triton.jit
def _reduce_rank1_mask_form_kernel(
    X,
    MVEC,
    O,
    N,
    K,
    FILL,
    BLOCK: tl.constexpr,
    FORM: tl.constexpr,
    OP: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    if FORM == "canonical":
        values = tl.load(X + offs, mask=offs < N, other=7.5)
    elif FORM == "sle":
        values = tl.load(X + offs, mask=offs <= N, other=7.5)
    elif FORM == "sgt":
        values = tl.load(X + offs, mask=offs > N, other=7.5)
    elif FORM == "ne":
        values = tl.load(X + offs, mask=offs != N, other=7.5)
    elif FORM == "band":
        values = tl.load(X + offs, mask=(offs < N) & (offs >= K), other=7.5)
    elif FORM == "either":
        values = tl.load(X + offs, mask=(offs < K) | (offs >= N), other=7.5)
    elif FORM == "shifted_index":
        values = tl.load(X + offs, mask=(offs + 8) < N, other=7.5)
    elif FORM == "strided_index":
        values = tl.load(X + offs, mask=(offs * 2) < N, other=7.5)
    elif FORM == "mask_from_memory":
        values = tl.load(X + offs, mask=tl.load(MVEC + offs) != 0, other=7.5)
    elif FORM == "runtime_other":
        values = tl.load(X + offs, mask=offs < N, other=FILL)
    else:  # computed_other
        values = tl.load(X + offs, mask=offs < N, other=FILL * 2.0)
    result = tl.max(values, axis=0) if OP == "max" else tl.sum(values, axis=0)
    tl.store(O, result)


_MASK_FORMS = [
    "canonical",
    "sle",
    "sgt",
    "ne",
    "band",
    "either",
    "shifted_index",
    "strided_index",
    "mask_from_memory",
    "runtime_other",
    "computed_other",
]


def _mask_form_condition(form, idx, mvec, n, k):
    if form == "canonical":
        return idx < n
    if form == "sle":
        return idx <= n
    if form == "sgt":
        return idx > n
    if form == "ne":
        return idx != n
    if form == "band":
        return (idx < n) & (idx >= k)
    if form == "either":
        return (idx < k) | (idx >= n)
    if form == "shifted_index":
        return idx + 8 < n
    if form == "strided_index":
        return idx * 2 < n
    if form == "mask_from_memory":
        return mvec != 0
    return idx < n


@pytest.mark.parametrize("form", _MASK_FORMS)
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize(
    "block,num_warps",
    [(128, 1), (256, 2), (64, 2)],
    ids=["e4", "e4_w2", "e1"],
)
def test_reduce_rank1_replayed_mask_forms(form, op, block, num_warps):
    torch.manual_seed(0xB0A7)
    n = block // 2 + 1
    k = block // 4
    fill = 7.5
    x = torch.randn((block,), dtype=torch.float32).contiguous()
    mvec = (torch.arange(block) % 3 != 0).to(torch.int32).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    _reduce_rank1_mask_form_kernel[(1, 1, 1)](
        x, mvec, out, n, k, fill, BLOCK=block, FORM=form, OP=op,
        num_warps=num_warps,
    )
    idx = torch.arange(block)
    other = fill * 2.0 if form == "computed_other" else fill
    cond = _mask_form_condition(form, idx, mvec.cpu(), n, k)
    ref = torch.where(cond, x.cpu(), torch.full((block,), float(other)))
    expected = ref.max() if op == "max" else ref.sum()
    torch.testing.assert_close(
        out[0].item(), expected.item(), atol=2e-3, rtol=2e-3
    )


# ---------------------------------------------------------------------------
# Float-to-int casts inside a replayed cone.
#
# `evalRank1ValueAt` has always rebuilt int->float (`sitofp`, `uitofp`) and every
# width cast, but not the float->int direction, and neither did its dry-run
# predicate `rank1ConeSupported` nor the index pair
# (`indexConeSupported` / `scalarizeConeAtIndex`). Above one element per thread
# that refused `tl.load(Y + o).to(tl.int32)` wherever it appeared: as the reduced
# value, as a mask bound, and as a gather index. At one element per thread the
# band's converted scalar is read directly, so all three have always worked
# there — which is the control each case is checked against.


@triton.jit
def _reduce_rank1_fptosi_kernel(
    X,
    Y,
    O,
    BLOCK: tl.constexpr,
    FORM: tl.constexpr,
    OP: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    if FORM == "value":
        v = tl.load(X + offs).to(tl.int32).to(tl.float32)
    elif FORM == "value_unsigned":
        v = tl.load(Y + offs).to(tl.uint32).to(tl.float32)
    elif FORM == "value_arith":
        v = ((tl.load(X + offs) * 4.0).to(tl.int32) + 1).to(tl.float32)
    elif FORM == "mask_bound":
        v = tl.load(X + offs, mask=offs < tl.load(Y + offs).to(tl.int32),
                    other=7.5)
    else:  # address: the gather index lives in a float buffer
        v = tl.load(X + tl.load(Y + offs).to(tl.int32))
    result = tl.max(v, axis=0) if OP == "max" else tl.sum(v, axis=0)
    tl.store(O, result)


_FPTOSI_FORMS = ["value", "value_unsigned", "value_arith", "mask_bound",
                 "address"]


@pytest.mark.parametrize("form", _FPTOSI_FORMS)
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize(
    "block,num_warps",
    [(128, 1), (256, 2), (64, 2)],
    ids=["e4", "e4_w2", "e1"],
)
def test_reduce_rank1_replayed_float_to_int_cone(form, op, block, num_warps):
    torch.manual_seed(0x5EED)
    x = (torch.randn((block,), dtype=torch.float32) * 3.0).contiguous()
    # Non-negative and inside the tile so the same tensor serves as a bound, an
    # unsigned value and an index.
    y = ((torch.arange(block, dtype=torch.float32) % 17) + 0.25).contiguous()
    out = torch.zeros((1,), dtype=torch.float32).contiguous()
    _reduce_rank1_fptosi_kernel[(1, 1, 1)](
        x, y, out, BLOCK=block, FORM=form, OP=op, num_warps=num_warps,
    )
    idx = torch.arange(block)
    yi = y.cpu().to(torch.int32)
    if form == "value":
        ref = x.cpu().to(torch.int32).float()
    elif form == "value_unsigned":
        ref = yi.float()
    elif form == "value_arith":
        ref = ((x.cpu() * 4.0).to(torch.int32) + 1).float()
    elif form == "mask_bound":
        ref = torch.where(idx < yi, x.cpu(), torch.full((block,), 7.5))
    else:
        ref = x.cpu()[yi.to(torch.int64)]
    expected = ref.max() if op == "max" else ref.sum()
    torch.testing.assert_close(
        out[0].item(), expected.item(), atol=2e-3, rtol=2e-3
    )


# ---------------------------------------------------------------------------
# Flattened tiles carry a linear layout and replay at logical positions.


@triton.jit
def _reduce_flattened_linear_rank1_kernel(
    X,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    OP: tl.constexpr,
):
    offsets = tl.arange(0, M * N)
    tile = tl.reshape(tl.load(X + offsets), (M, N))
    flat = tl.reshape(tile, (M * N,))
    result = tl.max(flat, axis=0) if OP == "max" else tl.sum(flat, axis=0)
    tl.store(O, result)


@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("M,N", [(64, 64), (64, 128)], ids=["e128", "e256"])
def test_reduce_flattened_linear_rank1_wide(M, N, op):
    torch.manual_seed(M * 41 + N)
    x = torch.randn(M * N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    _reduce_flattened_linear_rank1_kernel[(1,)](
        x, out, M=M, N=N, OP=op, num_warps=1,
    )
    torch.mps.synchronize()
    expected = x.max() if op == "max" else x.sum()
    torch.testing.assert_close(out.cpu(), expected.reshape(1), atol=2e-3,
                               rtol=2e-3)


def test_reduce_flattened_linear_rank1_e512_is_named(capfd):
    M, N = 64, 256
    x = torch.randn(M * N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(Exception, match="convert-tritongpu-to-metal failed"):
        _reduce_flattened_linear_rank1_kernel[(1,)](
            x, out, M=M, N=N, OP="sum", num_warps=1,
        )
    diagnostic = capfd.readouterr()
    output = diagnostic.err + diagnostic.out
    assert ("at most 256 elements per thread" in output or
            "outside supported envelope [1, 256]" in output)

    survivor = torch.zeros(1, dtype=torch.float32)
    _reduce_flattened_linear_rank1_kernel[(1,)](
        x[:64 * 64], survivor, M=64, N=64, OP="sum", num_warps=1,
    )
    torch.mps.synchronize()
    torch.testing.assert_close(survivor.cpu(), x[:64 * 64].sum().reshape(1),
                               atol=2e-3, rtol=2e-3)
