"""Rank-2 axis=1 f32 extrema/argmin and direct-load i32 product on Metal.

Companion to `test_metal_backend_reduce_sum.py`. The rank-2 `ReduceLowering`
row-scan body originally only implemented the sum combine (arith.addf /
arith.addi). This exercises the MAX combine (`tl.max`, which Triton emits as
arith.maxnumf): the kernel loads a 2D `(M, N)` block, calls
`tl.max(x, axis=1)`, and stores the resulting `(M,)` vector.

The combine is emitted as `metal.binary_exp ... maxOp` (MSL `max(a, b)`) and
the per-row scf.for iter_arg is identity-initialised to exact -infinity. Max is
an exact element selection, so the result is compared bit-tight against
`torch.amax(input, dim=1)`.
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


@triton.jit
def reduce_min_axis1_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(x_ptr + addr)
    m = tl.min(x, axis=1)
    tl.store(out_ptr + offs_m, m)


@triton.jit
def _product_combine(a, b):
    return a * b


@triton.jit
def reduce_product_axis1_kernel(
    x_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(x_ptr + addr)
    product = tl.reduce(x, axis=1, combine_fn=_product_combine)
    tl.store(out_ptr + offs_m, product)


@triton.jit
def reduce_masked_min_argmin_axis1_kernel(
    x_ptr,
    min_ptr,
    index_ptr,
    VALID_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    addr = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    x = tl.load(x_ptr + addr)
    masked = tl.where(offs_n[None, :] < VALID_N, x, float("inf"))
    tl.store(min_ptr + offs_m, tl.min(masked, axis=1))
    tl.store(index_ptr + offs_m, tl.argmin(masked, axis=1))


@triton.jit
def nearest_neighbor_3d_reduce_kernel(
    points_ptr,
    index_ptr,
    n_points,
    BLOCK_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_points
    main_x = tl.load(points_ptr + rows * 3, mask=row_mask, other=0.0)
    main_y = tl.load(points_ptr + rows * 3 + 1, mask=row_mask, other=0.0)
    main_z = tl.load(points_ptr + rows * 3 + 2, mask=row_mask, other=0.0)

    best_distance = tl.full([BLOCK_M], float("inf"), tl.float32)
    best_index = tl.zeros([BLOCK_M], tl.int32)
    tile_offsets = tl.arange(0, TILE_N)
    for tile_start in range(0, n_points, TILE_N):
        cols = tile_start + tile_offsets
        col_mask = cols < n_points
        tile_x = tl.load(points_ptr + cols * 3, mask=col_mask, other=0.0)
        tile_y = tl.load(points_ptr + cols * 3 + 1, mask=col_mask, other=0.0)
        tile_z = tl.load(points_ptr + cols * 3 + 2, mask=col_mask, other=0.0)
        dx = main_x[:, None] - tile_x[None, :]
        dy = main_y[:, None] - tile_y[None, :]
        dz = main_z[:, None] - tile_z[None, :]
        squared_distance = dx * dx + dy * dy + dz * dz
        valid = col_mask[None, :] & (rows[:, None] != cols[None, :])
        masked = tl.where(valid, squared_distance, float("inf"))
        tile_distance = tl.min(masked, axis=1)
        tile_index = tl.argmin(masked, axis=1)
        replace = tile_distance < best_distance
        best_distance = tl.where(replace, tile_distance, best_distance)
        best_index = tl.where(replace, tile_start + tile_index, best_index)
    tl.store(index_ptr + rows, best_index, mask=row_mask)


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
    """All-negative rows keep their real finite maximum."""
    M, N = 8, 16
    x = -torch.rand((M, N), dtype=torch.float32).contiguous() - 1.0  # in [-2, -1)
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_max_axis1_kernel[(1, 1, 1)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.amax(x.cpu(), dim=1)
    assert (out.cpu() < 0).all(), f"max identity leaked into result: {out}"
    torch.testing.assert_close(out.cpu(), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("M,N", [(8, 16), (16, 32)])
def test_reduce_max_axis1_all_negative_infinity(M, N):
    x = torch.full((M, N), float("-inf"), dtype=torch.float32)
    out = torch.zeros((M,), dtype=torch.float32)
    reduce_max_axis1_kernel[(1,)](x, out, BLOCK_M=M, BLOCK_N=N)
    assert torch.isneginf(out).all()


@pytest.mark.parametrize("M, N", [(8, 16), (4, 32), (16, 16), (128, 64)])
def test_reduce_min_axis1_f32(M, N):
    torch.manual_seed(0xBADC0DE)
    x = torch.randn((M, N), dtype=torch.float32).contiguous()
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_min_axis1_kernel[(1, 1, 1)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.amin(x.cpu(), dim=1)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-6, rtol=0)


def test_reduce_min_axis1_positive_rows():
    """All-positive rows preserve the real finite minimum."""
    M, N = 8, 16
    x = torch.rand((M, N), dtype=torch.float32).contiguous() + 1.0
    out = torch.zeros((M,), dtype=torch.float32).contiguous()
    reduce_min_axis1_kernel[(1, 1, 1)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.amin(x.cpu(), dim=1)
    assert (out.cpu() > 0).all(), f"non-positive minimum returned: {out}"
    torch.testing.assert_close(out.cpu(), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("M,N", [(8, 16), (16, 32)])
def test_reduce_product_axis1_i32_direct_load(M, N):
    x = torch.ones((M, N), dtype=torch.int32)
    x[:, :2] = torch.tensor([2, 3], dtype=torch.int32)
    x[0, :2] = 65536
    x[1, 0] = -2
    out = torch.zeros((M,), dtype=torch.int32)
    reduce_product_axis1_kernel[(1,)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.prod(x.to(torch.int64), dim=1).to(torch.int32)
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("M,N", [(8, 16), (16, 32)])
def test_reduce_product_axis1_f32_direct_load(M, N):
    x = torch.ones((M, N), dtype=torch.float32)
    x[:, 0] = 1.25
    x[:, 1] = -0.5
    x[0, 2] = 0.0
    x[1, 2] = float("inf")
    x[2, 2] = float("nan")
    x[3, 2] = -0.0
    out = torch.zeros((M,), dtype=torch.float32)
    reduce_product_axis1_kernel[(1,)](x, out, BLOCK_M=M, BLOCK_N=N)
    expected = torch.prod(x, dim=1)
    actual = out.cpu()
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=1e-6, rtol=1e-6)
    assert torch.isneginf(actual[1])
    assert torch.isnan(actual[2])
    assert torch.equal(actual.view(torch.int32)[[0, 3]],
                       expected.view(torch.int32)[[0, 3]])


@pytest.mark.parametrize(
    "M, N, valid_n",
    [
        (8, 16, 13),
        (16, 1024, 257),  # nearest-neighbor reduction tile
        (8, 16, 0),       # fully masked row: min=+inf, argmin tie -> index 0
    ],
)
def test_reduce_masked_min_argmin_axis1_f32(M, N, valid_n):
    torch.manual_seed(0xA691 + M + N + valid_n)
    x = torch.randn((M, N), dtype=torch.float32, device="mps")
    if valid_n >= 8:
        # Two equal minima pin tie_break_left=True (lower source index wins).
        x[:, 3] = -10.0
        x[:, 7] = -10.0
    min_out = torch.empty((M,), dtype=torch.float32, device="mps")
    index_out = torch.empty((M,), dtype=torch.int32, device="mps")

    reduce_masked_min_argmin_axis1_kernel[(1,)](
        x,
        min_out,
        index_out,
        VALID_N=valid_n,
        BLOCK_M=M,
        BLOCK_N=N,
    )
    torch.mps.synchronize()

    masked = x.cpu()
    if valid_n < N:
        masked[:, valid_n:] = float("inf")
    expected_min, expected_index = masked.min(dim=1)
    torch.testing.assert_close(min_out.cpu(), expected_min, atol=0, rtol=0)
    torch.testing.assert_close(
        index_out.cpu(), expected_index.to(torch.int32), atol=0, rtol=0
    )


@pytest.mark.parametrize("n_points", [15, 17, 257, 1025])
def test_nearest_neighbor_3d_computed_min_argmin(n_points):
    """Pins computed-cone replay and the 1024-column loop boundary."""
    torch.manual_seed(0x3D00 + n_points)
    points = torch.randn((n_points, 3), dtype=torch.float32, device="mps")
    actual = torch.empty((n_points,), dtype=torch.int32, device="mps")
    nearest_neighbor_3d_reduce_kernel[(triton.cdiv(n_points, 16),)](
        points,
        actual,
        n_points,
        BLOCK_M=16,
        TILE_N=1024,
    )
    torch.mps.synchronize()

    cpu_points = points.cpu()
    distances = ((cpu_points[:, None, :] - cpu_points[None, :, :]) ** 2).sum(2)
    distances.fill_diagonal_(float("inf"))
    expected = distances.argmin(1).to(torch.int32)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


# --- Bitwise integer combines along axis=1 --------------------------------
#
# Same three monoids as the rank-1 case (see
# `test_metal_backend_reduce_rank1.py::test_reduce_rank1_bitwise_i32`), here
# through the rank-2 unrolled per-row column scan. Like i32 product/extrema,
# these require a direct unmasked tt.load — the computed-cone row scanner is
# integer-sum only — which the convert pre-pass enforces.


@triton.jit
def _rank2_bitwise_and(a, b):
    return a & b


@triton.jit
def _rank2_bitwise_or(a, b):
    return a | b


@triton.jit
def reduce_xor_rank2_axis1_kernel(x_ptr, out_ptr, M: tl.constexpr,
                                  N: tl.constexpr):
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(out_ptr + tl.arange(0, M), tl.xor_sum(tl.load(x_ptr + offs), 1))


@triton.jit
def reduce_and_rank2_axis1_kernel(x_ptr, out_ptr, M: tl.constexpr,
                                  N: tl.constexpr):
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(out_ptr + tl.arange(0, M),
             tl.reduce(tl.load(x_ptr + offs), 1, _rank2_bitwise_and))


@triton.jit
def reduce_or_rank2_axis1_kernel(x_ptr, out_ptr, M: tl.constexpr,
                                 N: tl.constexpr):
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(out_ptr + tl.arange(0, M),
             tl.reduce(tl.load(x_ptr + offs), 1, _rank2_bitwise_or))


@pytest.mark.parametrize("shape", [(4, 8), (8, 16), (16, 32), (32, 8)])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("op", ["xor", "and", "or"])
def test_reduce_rank2_axis1_bitwise_i32(shape, num_warps, op):
    m, n = shape
    torch.manual_seed(m * 977 + n * 31 + num_warps)
    x_cpu = torch.randint(-(2**30), 2**30, (m, n), dtype=torch.int32)
    x = x_cpu.to("mps")
    out = torch.zeros(m, dtype=torch.int32, device="mps")

    kernel, identity, apply = {
        "xor": (reduce_xor_rank2_axis1_kernel, 0, lambda a, b: a ^ b),
        "and": (reduce_and_rank2_axis1_kernel, -1, lambda a, b: a & b),
        "or": (reduce_or_rank2_axis1_kernel, 0, lambda a, b: a | b),
    }[op]
    kernel[(1,)](x, out, M=m, N=n, num_warps=num_warps)
    torch.mps.synchronize()

    rows = x_cpu.tolist()
    expected = []
    for row in rows:
        acc = identity
        for value in row:
            acc = apply(acc, value)
        expected.append((acc + 2**31) % 2**32 - 2**31)
    assert out.cpu().tolist() == expected


# --- f16 / bf16 axis=1 reduces -------------------------------------------
#
# Same own-type rule as the rank-1 path (see
# `test_metal_backend_reduce_rank1.py::test_reduce_rank1_half_sum_is_exact`),
# here through the per-row column scan. Restricted to a DIRECT unmasked load:
# the computed-cone re-emitter is f32-only, which the convert pre-pass now
# reports by name instead of letting dialect conversion fail bare.


@triton.jit
def reduce_sum_rank2_half_kernel(x_ptr, out_ptr, M: tl.constexpr,
                                 N: tl.constexpr):
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(out_ptr + tl.arange(0, M), tl.sum(tl.load(x_ptr + offs), axis=1))


@triton.jit
def reduce_max_rank2_half_kernel(x_ptr, out_ptr, M: tl.constexpr,
                                 N: tl.constexpr):
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(out_ptr + tl.arange(0, M), tl.max(tl.load(x_ptr + offs), axis=1))


@triton.jit
def softmax_rank2_half_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    v = tl.load(x_ptr + offs)
    e = tl.exp(v - tl.max(v, axis=1)[:, None])
    tl.store(out_ptr + offs, e / tl.sum(e, axis=1)[:, None])


@pytest.mark.parametrize("shape", [(4, 8), (8, 16), (16, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduce_rank2_axis1_half_sum_is_exact(shape, dtype):
    m, n = shape
    torch.manual_seed(m * 31 + n)
    x = torch.randint(-4, 5, (m, n)).to(dtype) * 0.5
    out = torch.zeros(m, dtype=dtype, device="mps")
    reduce_sum_rank2_half_kernel[(1,)](x.to("mps"), out, M=m, N=n)
    torch.mps.synchronize()
    assert out.cpu().double().tolist() == x.double().sum(1).tolist()


@pytest.mark.parametrize("shape", [(4, 8), (16, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduce_rank2_axis1_half_max_is_exact(shape, dtype):
    m, n = shape
    torch.manual_seed(m * 17 + n)
    x = torch.rand(m, n, dtype=dtype) - 0.5
    out = torch.zeros(m, dtype=dtype, device="mps")
    reduce_max_rank2_half_kernel[(1,)](x.to("mps"), out, M=m, N=n)
    torch.mps.synchronize()
    assert out.cpu().tolist() == x.amax(1).tolist()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_rank2_half_end_to_end(dtype):
    """The shape half reduces exist for: max and sum over a row, each
    broadcast back into the tile."""
    m, n = 8, 16
    torch.manual_seed(3)
    x = torch.rand(m, n, dtype=dtype) - 0.5
    out = torch.zeros(m, n, dtype=dtype, device="mps")
    softmax_rank2_half_kernel[(1,)](x.to("mps"), out, M=m, N=n)
    torch.mps.synchronize()
    expected = torch.softmax(x.float(), dim=1)
    tolerance = 2e-2 if dtype is torch.bfloat16 else 3e-3
    torch.testing.assert_close(out.cpu().float(), expected,
                               atol=tolerance, rtol=tolerance)


# --- flat-position -> row must follow the tile's `order` -------------------
#
# The reduce result broadcast back into a rank-2 tile is read as rowBuf[row],
# and `row` used to be computed as `flat / N` unconditionally — the row-major
# formula. Triton picks the COLUMN-major order (`order = [0, 1]`) for shapes
# like an Mx4 tile whose runtime column count is 1, and there the flat position
# is `c*M + r`, so the row is `flat % M`.
#
# Under that layout the load/store address and mask decomposed with `% M` while
# the rowBuf read used `/ N`: every thread read a DIFFERENT row's reduce result
# than the one it was computing. A masked 2-D softmax came out wrong with no
# diagnostic anywhere, which is why these assert VALUES, not compilability.


@triton.jit
def _masked_softmax_rank2_kernel(x_ptr, out_ptr, M, N, BM: tl.constexpr,
                                 BN: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=mask,
                other=-float("inf"))
    e = tl.where(mask, tl.exp(v - tl.max(v, axis=1)[:, None]), 0.0)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :],
             e / tl.sum(e, axis=1)[:, None], mask=mask)


@triton.jit
def _masked_center_rank2_kernel(x_ptr, out_ptr, M, N, BM: tl.constexpr,
                                BN: tl.constexpr):
    """One reduce broadcast straight back — the minimal form of the bug, with
    no second reduce to confuse the diagnosis."""
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=mask,
                other=-float("inf"))
    tl.store(out_ptr + rm[:, None] * N + rn[None, :],
             v - tl.max(v, axis=1)[:, None], mask=mask)


@pytest.mark.parametrize("bm,bn", [(8, 4), (16, 4), (32, 4), (64, 4), (64, 2),
                                   (64, 8)])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_rank2_reduce_broadcast_single_runtime_column(bm, bn, num_warps):
    """N == 1 with BN > 1 is what selects the column-major tile order."""
    m, n = bm, 1
    torch.manual_seed(bm * 31 + bn)
    x = torch.rand(m, n, dtype=torch.float32)

    out = torch.zeros(m, n, dtype=torch.float32, device="mps")
    _masked_center_rank2_kernel[(1,)](x.to("mps"), out, m, n, BM=bm, BN=bn,
                                      num_warps=num_warps)
    torch.mps.synchronize()
    # Only column 0 is unmasked, so v - rowmax is exactly 0 everywhere.
    torch.testing.assert_close(out.cpu(), torch.zeros_like(x), atol=0, rtol=0)

    out = torch.zeros(m, n, dtype=torch.float32, device="mps")
    _masked_softmax_rank2_kernel[(1,)](x.to("mps"), out, m, n, BM=bm, BN=bn,
                                       num_warps=num_warps)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), torch.softmax(x, dim=1),
                               atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("bm,bn", [(8, 4), (32, 4), (64, 8)])
def test_rank2_reduce_broadcast_full_columns_still_correct(bm, bn):
    """The row-major decomposition it used to assume must keep working."""
    torch.manual_seed(bm + bn)
    x = torch.rand(bm, bn, dtype=torch.float32)
    out = torch.zeros(bm, bn, dtype=torch.float32, device="mps")
    _masked_softmax_rank2_kernel[(1,)](x.to("mps"), out, bm, bn, BM=bm, BN=bn)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), torch.softmax(x, dim=1),
                               atol=1e-6, rtol=1e-6)


@triton.jit
def _axis0_reduce_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rn, tl.sum(v, axis=0))


def test_axis0_reduce_wide_tile_is_rejected_not_crashed(capfd):
    """An axis=0 reduce with more columns than threads is deferred. That
    decline used to happen INSIDE applyFullConversion, where a
    notifyMatchFailure does not produce a catchable error — it segfaults the
    process during the failed conversion's rollback, taking the caller with it.
    Pre-pass rejections unwind cleanly, so this asserts both that the error is
    catchable AND that the process can still compile afterwards."""
    m, n = 2, 64  # tpb = 32 at num_warps=1, so N > tpb
    x = torch.rand(m, n, dtype=torch.float32, device="mps")
    out = torch.zeros(n, dtype=torch.float32, device="mps")
    with pytest.raises(Exception):
        _axis0_reduce_kernel[(1,)](x, out, M=m, N=n, num_warps=1)
    assert "one output column per thread" in capfd.readouterr().err

    # Still usable: a compile after the caught failure must succeed.
    y = torch.rand(64, dtype=torch.float32, device="mps")
    got = torch.zeros(64, dtype=torch.float32, device="mps")
    _masked_center_rank2_kernel[(1,)](y.reshape(64, 1).to("mps"), got.reshape(64, 1),
                                      64, 1, BM=64, BN=4)
    torch.mps.synchronize()
    torch.testing.assert_close(got.cpu(), torch.zeros(64), atol=0, rtol=0)
