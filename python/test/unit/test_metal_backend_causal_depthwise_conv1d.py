"""Causal depthwise conv1d on the Metal backend.

Runs the verbatim `python/test/unit/fixtures/metal_leet/medium-causal_depthwise_conv1d.py` kernel, but the
compiler gap it exposed has nothing to do with convolution: it is the general
rank-2 index math for a tile whose inner dimension is a RUNTIME value that is
not a multiple of 16.

`D % 16 == 0` makes the Triton frontend attach `tt.divisibility = 16` to the
argument, the tile vectorizes to one `#blocked` with `sizePerThread = [1, 4]`,
and BOTH `tl.arange`s are `tt.make_range` under `#ttg.slice<..., parent = that
blocked>` — the shape `MakeRangeLowering`'s 2D path handles, emitting
`row = flat / BLOCK_N`, `col = flat % BLOCK_N`.

Without the hint there is no vectorization and the frontend instead emits THREE
layouts — a rank-1 `#blocked` for the range, the rank-2 store layout (order
[1, 0]), and a rank-2 order-[0, 1] intermediate — bridging the column range into
the tile as

    cvt #blockedRank1 -> slice<dim=0, parent=#blockedMid>
    expand_dims                  -> 1xN, #blockedMid
    cvt #blockedMid -> #blockedStore

Both cvts used to be waved through as scalar identities, so the column index
stayed the raw lane id emitted by `MakeRangeLowering`'s rank-1 path: no
`% BLOCK_N`, no tile-loop `+ iv * T`. The store mask `col < D` then switched off
every thread with `lane >= D`, and only the first row of each tile-loop
iteration was ever written — 75% of the output untouched at the kernel's own
`num_warps = 8`, with no crash and no legalization failure.

Three changes make it work, all in `normalizeBlockedDivergentCvt` /
`isScalarIdentityConvert`:

* the cvt normalizer accepts a SLICE destination (rank-1 `#blocked` cone
  bridged into a 2D tile — the mirror of the slice-source case it already had),
  and gets the same clone-shared-leaves allowance, since `offs_d` legitimately
  feeds both the rank-1 `bias_ptr + offs_d` load and the tile's column index;
* the normalizer walks cvts in REVERSE order, so the outer `#blockedMid ->
  #blockedStore` cvt collapses first while its cone is still two values, and the
  inner bridge then re-encodes the range against the layout the store uses. In
  forward order the outer pass has to re-clone shared index arithmetic
  (`weight_ptr + offs_d * K + k`, CSE'd across the unrolled `tl.static_range`),
  which the blocked->blocked path deliberately refuses — it bailed, left
  `#blockedMid` (order [0, 1]) in place, and MakeRangeLowering's order-driven
  div/rem then emitted `flat / BLOCK_N` for the column. That is why `K == 1`
  used to pass while `K >= 2` did not;
* `isScalarIdentityConvert` refuses the blocked->slice identity when the cone
  bottoms out in a `tt.load` / `tt.make_range`, mirroring the guard already on
  the slice->blocked direction. If normalization ever bails the kernel gets a
  clean rejection instead of silently writing a quarter of its output.

`test_rank2_runtime_inner_dim_*` are the general pins and do not mention
convolution at all — a 12-line rank-2 copy reproduces the whole thing.
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


# Verbatim copy of python/test/unit/fixtures/metal_leet/medium-causal_depthwise_conv1d.py.
@triton.jit
def _causal_depthwise_conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    L, D,
    K: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_l = offs_l < L
    mask_d = offs_d < D

    bias = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0)
    acc = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32) + bias[None, :]

    batch_base = pid_b * L * D

    for k in tl.static_range(K):
        pos = offs_l - k
        pos_c = tl.maximum(pos, 0)

        load_mask = mask_l & (pos >= 0)

        w = tl.load(weight_ptr + offs_d * K + k, mask=mask_d, other=0.0)
        xv = tl.load(
            x_ptr + batch_base + pos_c[:, None] * D + offs_d[None, :],
            mask=load_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc += xv * w[None, :]

    tl.store(
        out_ptr + batch_base + offs_l[:, None] * D + offs_d[None, :],
        acc,
        mask=mask_l[:, None] & mask_d[None, :],
    )


def _solve(x, weight, bias, output, B, L, D, K, num_warps=8):
    BLOCK_L, BLOCK_D = 64, 64
    grid = (triton.cdiv(L, BLOCK_L), triton.cdiv(D, BLOCK_D), B)
    _causal_depthwise_conv1d_kernel[grid](
        x, weight, bias, output, L, D,
        K=K, BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D, num_warps=num_warps,
    )
    return output


def _reference(x, weight, bias, B, L, D, K):
    out = bias.float().view(1, 1, D).expand(B, L, D).clone()
    for k in range(K):
        if k == 0:
            out += x * weight[:, 0].view(1, 1, D)
        else:
            out[:, k:, :] += x[:, :-k, :] * weight[:, k].view(1, 1, D)
    return out


def _run_conv(B, L, D, K, num_warps=8, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, L, D, device="mps")
    weight = torch.randn(D, K, device="mps")
    bias = torch.randn(D, device="mps")
    # NaN fill, not zeros: the bug left elements UNWRITTEN, and a zero fill
    # cannot tell "never stored" from "stored a wrong value".
    out = torch.full((B, L, D), float("nan"), device="mps")
    _solve(x, weight, bias, out, B, L, D, K, num_warps=num_warps)
    torch.mps.synchronize()
    unwritten = int(torch.isnan(out).sum())
    assert unwritten == 0, (
        f"{unwritten}/{out.numel()} output elements were never stored "
        f"(B={B} L={L} D={D} K={K} num_warps={num_warps})"
    )
    torch.testing.assert_close(out, _reference(x, weight, bias, B, L, D, K),
                               atol=1e-4, rtol=1e-4)


# D sweeps both sides of the `% 16` divisibility split that decides whether the
# frontend vectorizes. 63/65/17/100/129 are the shapes that used to lose rows;
# 16/32/64/128 are the ones that always worked and must keep working.
@pytest.mark.parametrize("D", [1, 8, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65,
                               80, 96, 100, 128, 129])
def test_causal_depthwise_conv1d_channels(D):
    _run_conv(B=1, L=64, D=D, K=4)


# L only ever feeds the ROW index, which was correct throughout; pinned so a
# future change to the row projection cannot regress unnoticed.
@pytest.mark.parametrize("L", [1, 15, 16, 17, 32, 63, 64, 65, 70, 100, 128])
def test_causal_depthwise_conv1d_length(L):
    _run_conv(B=1, L=L, D=63, K=4)


# K == 1 folds `offs_d * K + k` away; K >= 2 leaves real index arithmetic shared
# across the unrolled static_range, which is what made the outer cvt bail under
# the original forward normalization order.
@pytest.mark.parametrize("K", [1, 2, 3, 4, 8, 16])
def test_causal_depthwise_conv1d_taps(K):
    _run_conv(B=1, L=64, D=63, K=K)


# num_warps decides threads-per-block T. At T == BLOCK_D the broken column index
# (a raw lane id) coincidentally equalled `flat % BLOCK_D`, so num_warps=2 passed
# even while 1/4/8 lost 1/2 to 3/4 of the rows. All four must pass.
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
def test_causal_depthwise_conv1d_num_warps(num_warps):
    _run_conv(B=1, L=64, D=63, K=4, num_warps=num_warps)


@pytest.mark.parametrize("B,L,D,K", [
    (1, 64, 64, 4),
    (2, 128, 128, 3),
    (1, 70, 65, 5),
    (3, 33, 17, 1),
    (2, 256, 64, 8),
    (4, 65, 63, 2),
])
def test_causal_depthwise_conv1d_shapes(B, L, D, K):
    _run_conv(B=B, L=L, D=D, K=K)


# --------------------------------------------------------------------------
# General pins: the defect is rank-2 index math, not convolution.
# --------------------------------------------------------------------------


@triton.jit
def _copy2d_kernel(x_ptr, out_ptr, M, N,
                   BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    m = (rm < M)[:, None] & (rn < N)[None, :]
    p = rm[:, None] * N + rn[None, :]
    tl.store(out_ptr + p, tl.load(x_ptr + p, mask=m, other=0.0), mask=m)


@triton.jit
def _copy2d_bias_kernel(x_ptr, b_ptr, out_ptr, M, N,
                        BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mn = rn < N
    m = (rm < M)[:, None] & mn[None, :]
    b = tl.load(b_ptr + rn, mask=mn, other=0.0)
    p = rm[:, None] * N + rn[None, :]
    tl.store(out_ptr + p, tl.load(x_ptr + p, mask=m, other=0.0) + b[None, :],
             mask=m)


@pytest.mark.parametrize("N", [1, 15, 16, 31, 32, 63, 64, 65, 100, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_rank2_runtime_inner_dim_copy(N, num_warps):
    """Rank-2 tile, runtime inner dim, no rank-1 vector anywhere."""
    torch.manual_seed(0)
    M = 64
    x = torch.randn(M, N, device="mps")
    out = torch.full((M, N), float("nan"), device="mps")
    _copy2d_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        x, out, M, N, BM=64, BN=64, num_warps=num_warps)
    torch.mps.synchronize()
    assert int(torch.isnan(out).sum()) == 0, "elements were never stored"
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("N", [1, 15, 16, 31, 32, 63, 64, 65, 100, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_rank2_runtime_inner_dim_with_rank1_load(N, num_warps):
    """Same, plus a rank-1 load indexed by the column range.

    This is the shape that forces the extra rank-1 `#blocked` layout and the
    two-step `#blockedRank1 -> slice -> expand_dims -> cvt` bridge: `rn` has to
    serve BOTH the rank-1 `b_ptr + rn` gather (thread t holds element t) and the
    tile's column coordinate (`flat % BN`), so the cvt between them is a real
    relabel and its cone must be duplicated, not waved through.
    """
    torch.manual_seed(0)
    M = 64
    x = torch.randn(M, N, device="mps")
    b = torch.randn(N, device="mps")
    out = torch.full((M, N), float("nan"), device="mps")
    _copy2d_bias_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
        x, b, out, M, N, BM=64, BN=64, num_warps=num_warps)
    torch.mps.synchronize()
    assert int(torch.isnan(out).sum()) == 0, "elements were never stored"
    torch.testing.assert_close(out, x + b[None, :])
