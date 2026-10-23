"""Rank-2 axis=1 f32 extrema/argmin and direct-load i32 product on Metal.

Companion to `test_metal_backend_reduce_sum.py`. The rank-2 `ReduceLowering`
row-scan body originally only implemented the sum combine (arith.addf /
arith.addi). This exercises the MAX combine (`tl.max`, which Triton emits as
arith.maxnumf): the kernel loads a 2D `(M, N)` block, calls
`tl.max(x, axis=1)`, and stores the resulting `(M,)` vector.

The combine is emitted as `metal.binary_exp ... maxOp` (MSL `max(a, b)`) and
the per-row scf.for iter_arg starts at the first source element. This keeps
all-NaN rows NaN under number-preferring max, whose -infinity identity would
otherwise replace every NaN. Finite maxima remain exact element selections.
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


@triton.jit
def _reduce_extrema_seed_axis1_kernel(
    x_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GUARD: tl.constexpr, BROADCAST: tl.constexpr, SOURCE: tl.constexpr,
    OP: tl.constexpr,
):
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rows = tl.program_id(0) * BLOCK_M + rm
    offsets = rows[:, None] * BLOCK_N + rn[None, :]
    if SOURCE == "masked":
        valid = (rn[None, :] < BLOCK_N - 3) & (rm[:, None] != BLOCK_M - 1)
        x = tl.load(x_ptr + GUARD + offsets, mask=valid, other=float("nan"))
    else:
        x = tl.load(x_ptr + GUARD + offsets)
        if SOURCE == "computed":
            x = x * 0.5
    # Match public tl.max/min promotion for bf16, whose scalar maximum/minimum
    # also promotes to f32 and therefore cannot form a bare bf16 combine region.
    if x.dtype == tl.bfloat16:
        x = x.to(tl.float32)
    if OP == "maxnum":
        result = tl.reduce(x, axis=1, combine_fn=_rank2_seed_maxnum)
    elif OP == "minnum":
        result = tl.reduce(x, axis=1, combine_fn=_rank2_seed_minnum)
    elif OP == "max":
        result = tl.reduce(x, axis=1, combine_fn=_rank2_seed_maximum)
    elif OP == "min":
        result = tl.reduce(x, axis=1, combine_fn=_rank2_seed_minimum)
    elif OP == "sum":
        result = tl.sum(x, axis=1)
    else:
        result = tl.reduce(x, axis=1, combine_fn=_product_combine)
    if BROADCAST:
        tl.store(out_ptr + GUARD + offsets, result[:, None])
    else:
        tl.store(out_ptr + GUARD + rows, result)


@triton.jit
def _rank2_seed_maxnum(a, b):
    return tl.maximum(a, b).to(a.dtype)


@triton.jit
def _rank2_seed_minnum(a, b):
    return tl.minimum(a, b).to(a.dtype)


@triton.jit
def _rank2_seed_maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _rank2_seed_minimum(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
def test_reduce_rank2_axis1_maxnum_all_nan_seed(broadcast):
    """A synthetic -inf must not displace the first real NaN in either mapping."""
    m, n = 16, 256
    x = torch.full((m, n), float("nan"), dtype=torch.float32, device="mps")
    out = torch.zeros((m, n) if broadcast else (m,), device="mps")
    _reduce_extrema_seed_axis1_kernel[(1,)](
        x, out, BLOCK_M=m, BLOCK_N=n, GUARD=0, BROADCAST=broadcast,
        SOURCE="direct", OP="maxnum", num_warps=4,
    )
    torch.mps.synchronize()
    assert torch.isnan(out.cpu()).all()


def _check_rank2_extrema_seed(op, source, broadcast, num_warps, dtype=torch.float32,
                             scalar_offset=False):
    """Each program has distinct finite rows, NaN placements, and zero signs."""
    m, n, programs, guard = 16, 256, 3, 7
    values = torch.empty((programs, m, n), dtype=dtype)
    for program in range(programs):
        tile = values[program]
        tile[:] = (torch.arange(n) % 7 - 3).to(dtype)
        tile[0] = float("nan")
        tile[1, 0] = float("nan")
        tile[2, n // 2] = float("nan")
        tile[3, -1] = float("nan")
        tile[4] = -float("inf")
        tile[5] = float("inf")
        tile[6] = 0.0
        tile[7] = -0.0
        tile[8, ::2] = -float("inf")
        tile[8, 1::2] = float("inf")
        tile[9] = program + 0.5
        tile[10:14] = float("nan")
        tile[10, n // 3] = program + 2.0
        tile[11, n // 3] = -float("inf")
        tile[12, n // 3] = float("inf")
        tile[13, n // 3] = -0.0
        tile[14] = -program - 0.5
        tile[15] = program + 4.0
    expected_input = values.clone()
    if source == "masked":
        expected_input[:, :, n - 3:] = float("nan")
        expected_input[:, -1] = float("nan")
    elif source == "computed":
        expected_input *= 0.5
    expected_input = expected_input.reshape(programs * m, n)
    if op in ("sum", "product"):
        expected = torch.full((programs * m,), 0.0 if op == "sum" else 1.0, dtype=dtype)
    else:
        expected = expected_input[:, 0].clone()
    combine = {
        "maxnum": torch.fmax,
        "minnum": torch.fmin,
        "max": torch.maximum,
        "min": torch.minimum,
        "sum": torch.add,
        "product": torch.mul,
    }[op]
    for column in range(0 if op in ("sum", "product") else 1, n):
        expected = combine(expected, expected_input[:, column])
    if broadcast:
        expected = expected[:, None].expand(-1, n).contiguous()

    x_cpu = torch.full((values.numel() + 2 * guard,), 257.0, dtype=dtype)
    x_cpu[guard:-guard] = values.flatten()
    x = x_cpu.to("mps")
    out = torch.full((expected.numel() + 2 * guard,), 257.0, dtype=dtype, device="mps")
    # Both modes retain physical allocation guards. Raw pointers also exercise
    # replay of the scalar addptr term when the row scan precedes the tile loop.
    x_arg = x if scalar_offset else x[guard:-guard]
    out_arg = out if scalar_offset else out[guard:-guard]
    _reduce_extrema_seed_axis1_kernel[(programs,)](
        x_arg, out_arg, BLOCK_M=m, BLOCK_N=n, GUARD=guard if scalar_offset else 0, BROADCAST=broadcast,
        SOURCE=source, OP=op, num_warps=num_warps,
    )
    torch.mps.synchronize()
    actual = out.cpu()
    torch.testing.assert_close(actual[guard:-guard].reshape(expected.shape), expected,
                               atol=0, rtol=0, equal_nan=True)
    # Check the sign of homogeneous zero rows; mixed-sign Num ties are unspecified.
    actual_values = actual[guard:-guard].reshape(expected.shape)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual_values[zeros]), torch.signbit(expected[zeros]))
    assert (actual[:guard] == 257).all() and (actual[-guard:] == 257).all()
    torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("source", ["direct", "masked", "computed"])
@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_reduce_rank2_axis1_maxnum_seed_ieee(source, broadcast, num_warps):
    """Number-preferring max ignores partial NaNs and preserves all-NaN rows."""
    _check_rank2_extrema_seed("maxnum", source, broadcast, num_warps)


@pytest.mark.parametrize("op", ["minnum", "max", "min"])
@pytest.mark.parametrize("source", ["direct", "masked", "computed"])
@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
def test_reduce_rank2_axis1_extrema_seed_neighbors(op, source, broadcast):
    """First-value seeding also preserves minimum and NaN-propagating behavior."""
    _check_rank2_extrema_seed(op, source, broadcast, 4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("op", ["maxnum", "minnum"])
@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
def test_reduce_rank2_axis1_half_extrema_seed_ieee(dtype, op, broadcast):
    """Direct f16 and promoted bf16 inputs retain NaNs and zero signs."""
    _check_rank2_extrema_seed(op, "direct", broadcast, 4, dtype)


@pytest.mark.parametrize("op", ["sum", "product"])
@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
def test_reduce_rank2_axis1_sum_product_seed_identity(op, broadcast):
    """The extrema change leaves sum's +0 and product's +1 seeds intact."""
    _check_rank2_extrema_seed(op, "direct", broadcast, 4)


@pytest.mark.parametrize("op,source", [
    (op, source)
    for op in ("maxnum", "minnum", "max", "min", "sum", "product")
    for source in ("direct", "masked", "computed")
    if op != "product" or source == "direct"
])
@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
def test_reduce_rank2_axis1_scalar_offset_constant(op, source, broadcast):
    """A scalar pointer displacement must dominate the replayed row source."""
    _check_rank2_extrema_seed(op, source, broadcast, 4, scalar_offset=True)


@triton.jit
def _reduce_scalar_offset_tile(
    x_ptr, out_ptr, base_offset, input_program_stride, output_program_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BROADCAST: tl.constexpr,
    OP: tl.constexpr,
):
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    input_offsets = tl.program_id(0) * input_program_stride + rm[:, None] * BLOCK_N + rn[None, :]
    x = tl.load(x_ptr + base_offset + input_offsets)
    if OP == "maxnum":
        result = tl.reduce(x, axis=1, combine_fn=_rank2_seed_maxnum)
    elif OP == "sum":
        result = tl.sum(x, axis=1)
    else:
        result = tl.reduce(x, axis=1, combine_fn=_product_combine)
    if BROADCAST:
        output_offsets = tl.program_id(0) * output_program_stride + rm[:, None] * BLOCK_N + rn[None, :]
        tl.store(out_ptr + base_offset + output_offsets, result[:, None])
    else:
        output_offsets = tl.program_id(0) * output_program_stride + rm
        tl.store(out_ptr + base_offset + output_offsets, result)


@triton.jit(do_not_specialize=["base_offset", "input_program_stride", "output_program_stride",
                              "input_tile_stride", "output_tile_stride", "n_tiles"])
def _reduce_scalar_offset_runtime_kernel(
    x_ptr, out_ptr, base_offset, input_program_stride, output_program_stride,
    input_tile_stride, output_tile_stride, n_tiles,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BROADCAST: tl.constexpr,
    OP: tl.constexpr, LOOP: tl.constexpr,
):
    base_offset = tl.where(base_offset > 0, base_offset, -base_offset)
    if LOOP:
        for tile in range(n_tiles):
            # These scalar terms vary with the enclosing runtime loop, while
            # remaining uniform across every element of the logical block.
            input_base = base_offset + tile * input_tile_stride
            output_base = base_offset + tile * output_tile_stride
            _reduce_scalar_offset_tile(
                x_ptr + input_base, out_ptr + output_base, 0,
                input_program_stride, output_program_stride,
                BLOCK_M, BLOCK_N, BROADCAST, OP,
            )
    else:
        _reduce_scalar_offset_tile(
            x_ptr, out_ptr, base_offset, input_program_stride, output_program_stride,
            BLOCK_M, BLOCK_N, BROADCAST, OP,
        )


def _check_rank2_scalar_runtime_offset(op, broadcast, loop):
    m, n, programs, guard = 16, 256, 3, 7
    n_tiles = 2 if loop else 1
    input_program_stride = m * n + 11
    output_elements = m * n if broadcast else m
    output_program_stride = output_elements + 13
    input_tile_stride = programs * input_program_stride + 17
    output_tile_stride = programs * output_program_stride + 19
    # Both launches share a compiled runtime signature: neither displacement is
    # constexpr or an alignment/divisibility specialization.
    for base_offset in (guard, guard + 6):
        x_cpu = torch.full((base_offset + n_tiles * input_tile_stride + guard,), 257.0)
        expected = torch.full((base_offset + n_tiles * output_tile_stride + guard,), 257.0)
        for tile in range(n_tiles):
            for program in range(programs):
                values = torch.ones((m, n))
                values[:, 0] = program * 32 + tile * 16 + torch.arange(m) + 1
                values[:, 1] = 2.0
                values[:, 2] = 0.5
                input_start = base_offset + tile * input_tile_stride + program * input_program_stride
                x_cpu[input_start:input_start + m * n] = values.flatten()
                if op == "maxnum":
                    result = values.max(dim=1).values
                elif op == "sum":
                    result = values.sum(dim=1)
                else:
                    result = values.prod(dim=1)
                if broadcast:
                    result = result[:, None].expand(-1, n).contiguous()
                output_start = base_offset + tile * output_tile_stride + program * output_program_stride
                expected[output_start:output_start + output_elements] = result.flatten()
        x = x_cpu.to("mps")
        out = torch.full_like(expected, 257.0, device="mps")
        _reduce_scalar_offset_runtime_kernel[(programs,)](
            x, out, base_offset, input_program_stride, output_program_stride,
            input_tile_stride, output_tile_stride, n_tiles,
            BLOCK_M=m, BLOCK_N=n, BROADCAST=broadcast, OP=op, LOOP=loop,
            num_warps=4,
        )
        torch.mps.synchronize()
        # Comparing the entire allocation also verifies all program/tile gaps,
        # the runtime displacement prefix, and trailing allocation guards.
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0)


@pytest.mark.parametrize("op", ["maxnum", "sum", "product"])
@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
def test_reduce_rank2_axis1_scalar_offset_runtime_argument(op, broadcast):
    """Runtime block arguments remain valid when row-source replay is hoisted."""
    _check_rank2_scalar_runtime_offset(op, broadcast, loop=False)


@pytest.mark.parametrize("op", ["maxnum", "sum"])
@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
def test_reduce_rank2_axis1_scalar_offset_loop_iteration(op, broadcast):
    """Enclosing-loop scalar offsets select a fresh tile on each iteration."""
    _check_rank2_scalar_runtime_offset(op, broadcast, loop=True)


@triton.jit(do_not_specialize=["left", "right", "input_program_stride", "output_program_stride",
                              "input_tile_stride", "output_tile_stride", "n_tiles"])
def _reduce_scalar_producer_offset_kernel(
    x_ptr, out_ptr, left, right, input_program_stride, output_program_stride,
    input_tile_stride, output_tile_stride, n_tiles,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GUARD: tl.constexpr,
    BROADCAST: tl.constexpr, OP: tl.constexpr, LOOP: tl.constexpr,
    PRODUCER: tl.constexpr, AXIS: tl.constexpr,
):
    if PRODUCER == "num_programs":
        selected = tl.num_programs(AXIS)
    elif PRODUCER == "minsi":
        selected = tl.minimum(left.to(tl.int32), right.to(tl.int32))
    elif PRODUCER == "maxsi":
        selected = tl.maximum(left.to(tl.int32), right.to(tl.int32))
    elif PRODUCER == "minui":
        selected = tl.minimum(left.to(tl.uint32), right.to(tl.uint32))
    else:
        selected = tl.maximum(left.to(tl.uint32), right.to(tl.uint32))
    # Preserve the comparison's signedness while bounding actual addresses.
    base_offset = GUARD + (selected & 31).to(tl.int32)
    program = tl.program_id(AXIS)
    input_base = base_offset + program * input_program_stride
    output_base = base_offset + program * output_program_stride
    if LOOP:
        for tile in range(n_tiles):
            _reduce_scalar_offset_tile(
                x_ptr + input_base + tile * input_tile_stride,
                out_ptr + output_base + tile * output_tile_stride,
                0, 0, 0, BLOCK_M, BLOCK_N, BROADCAST, OP,
            )
    else:
        _reduce_scalar_offset_tile(
            x_ptr + input_base, out_ptr + output_base,
            0, 0, 0, BLOCK_M, BLOCK_N, BROADCAST, OP,
        )


@pytest.mark.parametrize("producer,axis", [
    ("num_programs", 0), ("num_programs", 1), ("num_programs", 2),
    ("minsi", 0), ("maxsi", 0), ("minui", 0), ("maxui", 0),
])
@pytest.mark.parametrize("op", ["maxnum", "sum"])
@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
@pytest.mark.parametrize("loop", [False, True], ids=["no_loop", "runtime_loop"])
def test_reduce_rank2_axis1_scalar_offset_producer(producer, axis, op, broadcast, loop):
    """Replay grid sizes and signed/unsigned extrema in scalar pointer offsets."""
    m, n, programs, guard = 16, 256, 3, 7
    n_tiles = 2 if loop else 1
    output_elements = m * n if broadcast else m
    input_program_stride = m * n + 11
    output_program_stride = output_elements + 13
    input_tile_stride = programs * input_program_stride + 17
    output_tile_stride = programs * output_program_stride + 19
    grid = tuple(programs if dim == axis else 1 for dim in range(3))
    if producer == "num_programs":
        operands = [(5, 9)]
    elif producer.endswith("ui"):
        # Both arguments retain an i32 runtime signature; casting changes which
        # operand wins when its high bit is set. Operand order changes as well.
        operands = [(-2147483645, 5), (11, -2147483631)]
    else:
        operands = [(-9, 5), (12, -3)]
    for left, right in operands:
        if producer == "num_programs":
            selected = grid[axis]
        else:
            values = (left & 0xffffffff, right & 0xffffffff) if producer.endswith("ui") else (left, right)
            selected = min(values) if producer.startswith("min") else max(values)
        base_offset = guard + (selected & 31)
        x_cpu = torch.full((guard + 31 + n_tiles * input_tile_stride + guard,), 257.0)
        expected = torch.full((guard + 31 + n_tiles * output_tile_stride + guard,), 257.0)
        for tile in range(n_tiles):
            for program in range(programs):
                values = torch.ones((m, n))
                values[:, 0] = program * 32 + tile * 16 + torch.arange(m) + 1
                input_start = base_offset + tile * input_tile_stride + program * input_program_stride
                x_cpu[input_start:input_start + m * n] = values.flatten()
                result = values.max(dim=1).values if op == "maxnum" else values.sum(dim=1)
                if broadcast:
                    result = result[:, None].expand(-1, n).contiguous()
                output_start = base_offset + tile * output_tile_stride + program * output_program_stride
                expected[output_start:output_start + output_elements] = result.flatten()
        x = x_cpu.to("mps")
        out = torch.full_like(expected, 257.0, device="mps")
        _reduce_scalar_producer_offset_kernel[grid](
            x, out, left, right, input_program_stride, output_program_stride,
            input_tile_stride, output_tile_stride, n_tiles,
            BLOCK_M=m, BLOCK_N=n, GUARD=guard, BROADCAST=broadcast,
            OP=op, LOOP=loop, PRODUCER=producer, AXIS=axis, num_warps=4,
        )
        torch.mps.synchronize()
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0)


@triton.jit
def _reduce_maxnum_seed_loop_kernel(
    x_ptr, out_ptr, n_tiles, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GUARD: tl.constexpr, BROADCAST: tl.constexpr,
):
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    for tile in range(n_tiles):
        offsets = tile * BLOCK_M * BLOCK_N + rm[:, None] * BLOCK_N + rn[None, :]
        x = tl.load(x_ptr + GUARD + offsets)
        result = tl.reduce(x, axis=1, combine_fn=_rank2_seed_maxnum)
        if BROADCAST:
            tl.store(out_ptr + GUARD + offsets, result[:, None])
        else:
            tl.store(out_ptr + GUARD + tile * BLOCK_M + rm, result)


@pytest.mark.parametrize("broadcast", [False, True], ids=["vector", "broadcast"])
def test_reduce_rank2_axis1_maxnum_seed_loop_iteration(broadcast):
    """The first source element changes from NaN to finite and back across iterations."""
    m, n, guard = 16, 256, 7
    values = torch.full((2, m, n), float("nan"))
    values[0, :m // 2] = 2.0
    values[1, m // 2:] = -3.0
    expected = values[:, :, 0]
    if broadcast:
        expected = expected[:, :, None].expand(-1, -1, n).contiguous()
    x_cpu = torch.full((values.numel() + 2 * guard,), 257.0)
    x_cpu[guard:-guard] = values.flatten()
    x = x_cpu.to("mps")
    out = torch.full((expected.numel() + 2 * guard,), 257.0, device="mps")
    _reduce_maxnum_seed_loop_kernel[(1,)](
        x, out, 2, BLOCK_M=m, BLOCK_N=n, GUARD=guard, BROADCAST=broadcast,
        num_warps=4,
    )
    torch.mps.synchronize()
    actual = out.cpu()
    torch.testing.assert_close(actual[guard:-guard].reshape(expected.shape), expected,
                               atol=0, rtol=0, equal_nan=True)
    assert (actual[:guard] == 257).all() and (actual[-guard:] == 257).all()
    torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0, equal_nan=True)


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


@triton.jit
def _axis1_two_pass_arg_extrema(In, IndexOut, ValueOut, M, N, Iterations, OutStride,
                               BM: tl.constexpr, BN: tl.constexpr,
                               IS_MIN: tl.constexpr, COMPUTED: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    for iteration in range(Iterations):
        tile = tl.load(In + iteration * M * N + rows[:, None] * N + cols[None, :],
                       mask=mask, other=float("nan"))
        if COMPUTED:
            tile = tile * 2.0
        if IS_MIN:
            index = tl.argmin(tile, axis=1)
            value = tl.min(tile, axis=1)
        else:
            index = tl.argmax(tile, axis=1)
            value = tl.max(tile, axis=1)
        tl.store(IndexOut + iteration * OutStride + rows, index, mask=rows < M)
        tl.store(ValueOut + iteration * OutStride + rows, value, mask=rows < M)


@pytest.mark.parametrize("is_min", [True, False], ids=["argmin", "argmax"])
@pytest.mark.parametrize("computed", [False, True], ids=["direct", "computed"])
@pytest.mark.parametrize("BN, num_warps, iterations", [
    (64, 1, 1), (1024, 4, 1), (1024, 1, 3), (64, 4, 2),
])
def test_reduce_axis1_two_pass_arg_extrema(BN, num_warps, iterations, computed, is_min):
    # The i32 index pass must read the f32 extremum for its own row, including
    # inside a runtime loop whose input and winning column change each trip.
    BM, guard = 16, 11
    M, N = BM + 7, BN - 7
    patterns = torch.tensor([
        [3., -4., 3., -4., 1.],             # tied finite extrema
        [float("nan"), 2., -3., float("nan"), 4.],
        [float("nan")] * 5,
        [float("inf")] * 5,
        [-float("inf")] * 5,
        [-0., 0., -0., 0., -0.],
        [0.] * 5,
        [0.] * 5,
    ], dtype=torch.float32)[:, torch.arange(N) % 5]
    # Winners near the end also exercise the complete wide-column fold.
    patterns[6, N - 3], patterns[6, N - 2] = -9., 7.
    patterns[7, N - 4], patterns[7, N - 1] = -8., 9.
    tile = patterns[torch.arange(M) % patterns.shape[0]]
    source = torch.stack([tile.roll(i * (N // 3), dims=1) for i in range(iterations)])
    stride = M + 2 * guard
    indices = torch.full((iterations * stride,), -12345, dtype=torch.int32, device="mps")
    values = torch.full((iterations * stride,), 12345., dtype=torch.float32, device="mps")
    _axis1_two_pass_arg_extrema[(triton.cdiv(M, BM),)](
        source.to("mps"), indices[guard:], values[guard:], M, N, iterations, stride,
        BM=BM, BN=BN, IS_MIN=is_min, COMPUTED=computed, num_warps=num_warps,
    )
    torch.mps.synchronize()

    reference = source * 2.0 if computed else source
    is_nan = reference.isnan()
    identity = float("inf") if is_min else -float("inf")
    finite = reference.masked_fill(is_nan, identity)
    extrema = finite.amin(dim=2) if is_min else finite.amax(dim=2)
    all_nan = is_nan.all(dim=2)
    extrema = extrema.masked_fill(all_nan, float("nan"))
    matches = torch.where(all_nan[:, :, None], is_nan, reference == extrema[:, :, None])
    expected_indices = torch.where(matches, torch.arange(N)[None, None, :], 0x7fffffff).amin(dim=2)
    actual_indices = indices.cpu().reshape(iterations, stride)
    actual_values = values.cpu().reshape(iterations, stride)
    torch.testing.assert_close(actual_indices[:, guard:guard + M], expected_indices.to(torch.int32),
                               atol=0, rtol=0)
    torch.testing.assert_close(actual_values[:, guard:guard + M], extrema, atol=0, rtol=0, equal_nan=True)
    for output, canary in ((actual_indices, -12345), (actual_values, 12345.)):
        assert torch.all(output[:, :guard] == canary)
        assert torch.all(output[:, guard + M:] == canary)


@triton.jit
def _axis1_argmin_source_snapshot(In, IndexOut, ValueOut,
                                 BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    tile = tl.load(In + rows[:, None] * BN + cols[None, :])
    value = tl.min(tile, axis=1)
    # Both reductions use the original SSA tile, even after its backing memory
    # has changed. Replaying the load after this write would pick another index.
    tl.store(In + rows * BN, 1000.0)
    index = tl.argmin(tile, axis=1)
    tl.store(IndexOut + rows, index)
    tl.store(ValueOut + rows, value)


@triton.jit
def _axis1_argmin_source_snapshot_loop(In, IndexOut, ValueOut, Iterations,
                                      BM: tl.constexpr, BN: tl.constexpr):
    for iteration in range(Iterations):
        _axis1_argmin_source_snapshot(In, IndexOut, ValueOut, BM=BM, BN=BN)


@pytest.mark.parametrize("in_loop", [False, True], ids=["top_level", "loop_rejected"])
def test_reduce_axis1_argmin_preserves_source_snapshot(capfd, in_loop):
    BM, BN, guard = 16, 1024, 11
    original = torch.arange(BM * BN, dtype=torch.float32).reshape(BM, BN) + 1.0
    original[:, 0] = -10.0
    source = original.to("mps")
    indices = torch.full((BM + 2 * guard,), -12345, dtype=torch.int32, device="mps")
    values = torch.full((BM + 2 * guard,), 12345., dtype=torch.float32, device="mps")
    if in_loop:
        with pytest.raises(Exception, match="convert-tritongpu-to-metal failed"):
            _axis1_argmin_source_snapshot_loop[(1,)](
                source, indices[guard:], values[guard:], 2, BM=BM, BN=BN, num_warps=4,
            )
        diagnostic = capfd.readouterr()
        assert "rank-2 axis=1 reduce over a computed cone" in diagnostic.err + diagnostic.out
        torch.testing.assert_close(source.cpu(), original, atol=0, rtol=0)
        assert torch.all(indices.cpu() == -12345)
        assert torch.all(values.cpu() == 12345.)
        return

    _axis1_argmin_source_snapshot[(1,)](
        source, indices[guard:], values[guard:], BM=BM, BN=BN, num_warps=4,
    )
    torch.mps.synchronize()
    actual_indices, actual_values = indices.cpu(), values.cpu()
    assert torch.all(actual_indices[guard:guard + BM] == 0)
    assert torch.all(actual_values[guard:guard + BM] == -10.)
    for output, canary in ((actual_indices, -12345), (actual_values, 12345.)):
        assert torch.all(output[:guard] == canary)
        assert torch.all(output[guard + BM:] == canary)
    original[:, 0] = 1000.
    torch.testing.assert_close(source.cpu(), original, atol=0, rtol=0)


@triton.jit
def _axis1_arg_extrema_loop_reexpanded(In, IndexOut, ValueOut, TileOut, Iterations,
                                      BM: tl.constexpr, BN: tl.constexpr,
                                      LEAK_INDEX: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    value = tl.zeros((BM,), tl.float32)
    index = tl.zeros((BM,), tl.int32)
    for iteration in range(Iterations):
        tile = tl.load(In + iteration * BM * BN + rows[:, None] * BN + cols[None, :])
        value = tl.min(tile, axis=1)
        index = tl.argmin(tile, axis=1)
    tl.store(IndexOut + rows, index)
    tl.store(ValueOut + rows, value)
    if LEAK_INDEX:
        expanded = index[:, None].to(tl.float32) + tl.zeros((BM, BN), tl.float32)
    else:
        expanded = value[:, None] + tl.zeros((BM, BN), tl.float32)
    tl.store(TileOut + rows[:, None] * BN + cols[None, :], expanded)


@pytest.mark.parametrize("leak_index", [False, True], ids=["value", "index"])
def test_reduce_axis1_loop_carried_reexpanded_result_is_named(capfd, leak_index):
    # Follow the row value through scf.yield and the parent loop result before
    # deciding whether the index pass has only row-vector consumers.
    BM, BN = 16, 1024
    source = torch.randn(2, BM, BN, device="mps")
    indices = torch.full((BM,), -12345, dtype=torch.int32, device="mps")
    values = torch.full((BM,), 12345., dtype=torch.float32, device="mps")
    expanded = torch.full((BM, BN), 12345., dtype=torch.float32, device="mps")
    with pytest.raises(Exception, match="convert-tritongpu-to-metal failed"):
        _axis1_arg_extrema_loop_reexpanded[(1,)](
            source, indices, values, expanded, 2,
            BM=BM, BN=BN, LEAK_INDEX=leak_index, num_warps=4,
        )
    diagnostic = capfd.readouterr()
    assert "rank-2 axis=1 reduce over a computed cone" in diagnostic.err + diagnostic.out
    assert torch.all(indices.cpu() == -12345)
    assert torch.all(values.cpu() == 12345.)
    assert torch.all(expanded.cpu() == 12345.)


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


def test_axis0_reduce_wide_tile_is_lowered():
    """An axis=0 reduce with more columns than threads used to be deferred, on
    the assumption that the output layout was one the tile-loop model would
    mis-address. The lowering now reads that layout off the rank-1 value the
    result becomes, so this shape -- two rows, sixty-four columns, one warp --
    is an ordinary column sum."""
    m, n = 2, 64  # tpb = 32 at num_warps=1, so N > tpb
    x = torch.rand(m, n, dtype=torch.float32)
    out = torch.zeros(n, dtype=torch.float32, device="mps")
    _axis0_reduce_kernel[(1,)](x.to("mps"), out, M=m, N=n, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), x.sum(0), atol=1e-5, rtol=1e-5)


@triton.jit
def _axis0_reduce_reexpand_kernel(x_ptr, out_ptr, M: tl.constexpr,
                                  N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    tl.store(out_ptr + rm[:, None] * N + rn[None, :],
             v - tl.sum(v, axis=0)[None, :])


@triton.jit
def _axis0_reduce_reexpand_with_vector_kernel(x_ptr, sum_ptr, out_ptr,
                                              M: tl.constexpr,
                                              N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    s = tl.sum(v, axis=0)
    tl.store(sum_ptr + rn, s)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], v - s[None, :])


def test_axis0_reduce_reexpanded_result_is_lowered():
    """A result pushed back into a 2-D tile at more columns than threads used to
    be refused. The reduce now leaves all of its columns in a per-column buffer
    and the re-expansion reads that buffer at the column its lane needs, so
    centring a tile against its own column sums is an ordinary kernel."""
    m, n = 2, 64  # tpb = 32 at num_warps=1, so N > tpb
    x = torch.rand(m, n, dtype=torch.float32)
    sums = torch.zeros(n, dtype=torch.float32, device="mps")
    out = torch.zeros(m, n, dtype=torch.float32, device="mps")
    _axis0_reduce_reexpand_with_vector_kernel[(1,)](x.to("mps"), sums, out,
                                                    M=m, N=n, num_warps=1)
    torch.mps.synchronize()
    torch.testing.assert_close(sums.cpu(), x.sum(0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.cpu(), x - x.sum(0), atol=1e-5, rtol=1e-5)


def test_axis0_reduce_reexpanded_without_a_rank1_consumer_is_rejected(capfd):
    """A result that never becomes a rank-1 value has no per-thread output
    geometry to take, so it stays refused -- before conversion, because a
    decline inside it does not raise, it takes the process down."""
    m, n = 2, 64
    x = torch.rand(m, n, dtype=torch.float32, device="mps")
    out = torch.zeros(m, n, dtype=torch.float32, device="mps")
    with pytest.raises(Exception):
        _axis0_reduce_reexpand_kernel[(1,)](x, out, M=m, N=n, num_warps=1)
    assert "rank-1 consumer carrying exactly those columns" in (
        capfd.readouterr().err)


@triton.jit
def _axis0_reduce_reexpand_i32_kernel(x_ptr, sum_ptr, out_ptr, M: tl.constexpr,
                                      N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    v = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    s = tl.sum(v, axis=0)
    tl.store(sum_ptr + rn, s)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], v - s[None, :])


def test_axis0_reduce_reexpanded_i32_is_rejected_not_crashed(capfd):
    """The per-column buffer is read back through the rank-1 cone evaluator,
    whose arithmetic is signless, while an i32 buffer hands back its ui32
    storage type; the two would meet at a type mismatch, so i32 keeps a named
    refusal. It has to be named before conversion, because a decline inside it
    does not raise -- it takes the process down."""
    m, n = 2, 64
    x = torch.randint(-50, 50, (m, n), dtype=torch.int32, device="mps")
    sums = torch.zeros(n, dtype=torch.int32, device="mps")
    out = torch.zeros(m, n, dtype=torch.int32, device="mps")
    with pytest.raises(Exception):
        _axis0_reduce_reexpand_i32_kernel[(1,)](x, sums, out, M=m, N=n,
                                                num_warps=1)
    assert "f32 result whose elementwise cone" in capfd.readouterr().err

    # Still usable: a compile after the caught failure must succeed.
    y = torch.rand(64, dtype=torch.float32)
    got = torch.zeros(64, dtype=torch.float32, device="mps")
    _masked_center_rank2_kernel[(1,)](y.reshape(64, 1).to("mps"), got.reshape(64, 1),
                                      64, 1, BM=64, BN=4)
    torch.mps.synchronize()
    torch.testing.assert_close(got.cpu(), torch.zeros(64), atol=0, rtol=0)


@triton.jit
def _keepdims_reduce_store_kernel(x_ptr, y_ptr, M: tl.constexpr,
                                  N: tl.constexpr, OP: tl.constexpr,
                                  AXIS: tl.constexpr):
    r = tl.arange(0, M)
    c = tl.arange(0, N)
    x = tl.load(x_ptr + r[:, None] * N + c[None, :])
    if AXIS == 1:
        m = tl.sum(x, 1, keep_dims=True) if OP == 0 else tl.max(x, 1, keep_dims=True)
        tl.store(y_ptr + r[:, None] + tl.arange(0, 1)[None, :], m)
    else:
        m = tl.sum(x, 0, keep_dims=True) if OP == 0 else tl.max(x, 0, keep_dims=True)
        tl.store(y_ptr + tl.arange(0, 1)[:, None] + c[None, :], m)


# A reduce result stored with its reduced axis KEPT -- a per-row statistic next
# to its tile, and what `tl.topk(x, 1)` lowers to. Triton moves the result
# between two slices that drop the SAME axis of two rank-2 parents, and a rank-1
# value holds element `localTid % n` under either, so that conversion is a
# relabel. When slice-to-slice conversions began comparing the two parents'
# plans, this one became an exchange that published `localTid % n` into the
# slot the source parent names for the lane: every row got row 0's value, and
# other shapes were refused. 📏 `s_keepdims_reduce_store.py` 26 wrong / 62
# refused -> 0 / 6.
# An axis-0 reduce with more output columns than threads is a named refusal of
# its own, before and after, so those cells are left out.
_KEEPDIMS_CASES = [
    (M, N, w, op, axis)
    for (M, N, w) in ((2, 16, 1), (4, 32, 1), (4, 32, 4), (8, 32, 2),
                      (16, 16, 4), (2, 64, 1), (32, 8, 4), (8, 128, 1))
    for op in ("sum", "max")
    for axis in (1, 0)
    if not (axis == 0 and N > 32 * w)
]


@pytest.mark.parametrize("M,N,num_warps,op,axis", _KEEPDIMS_CASES)
def test_keep_dims_reduce_stored_as_a_tile(M, N, num_warps, op, axis):
    torch.manual_seed(M * 31 + N + num_warps)
    x = torch.randint(-64, 64, (M, N), device="mps").float()
    n = M if axis == 1 else N
    y = torch.zeros(n, device="mps")
    _keepdims_reduce_store_kernel[(1,)](x, y, M, N, 0 if op == "sum" else 1,
                                        axis, num_warps=num_warps)
    torch.mps.synchronize()
    xc = x.cpu()
    want = xc.sum(axis) if op == "sum" else xc.amax(axis)
    assert torch.equal(y.cpu(), want)


# An i32 keep-dims reduce stored as a column never finished COMPILING: the MSL
# emitter re-inlines every multi-use value that is not on its let-binding
# allowlist, and the scalar `unrealized_conversion_cast` (the ui32 <-> i32
# signedness reinterpret integer kernels are full of) was not on it -- 62 of
# them in this kernel, re-expanded at every use. Run in a subprocess with a
# timeout so a regression fails here instead of hanging the suite.
_I32_KEEPDIMS_SCRIPT = r"""
import torch, triton, triton.language as tl
@triton.jit
def k(X, Y, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R); c = tl.arange(0, C)
    x = tl.load(X + r[:, None] * C + c[None, :])
    tl.store(Y + r[:, None] + tl.arange(0, 1)[None, :], tl.max(x, 1, keep_dims=True))
x = (torch.randperm(128) - 64).to(torch.int32).reshape(4, 32)
y = torch.zeros(4, dtype=torch.int32, device="mps")
c = k[(1,)](x.to("mps"), y, 4, 32, num_warps=1)
torch.mps.synchronize()
assert torch.equal(y.cpu(), x.amax(1)), (y.cpu(), x.amax(1))
msl = c.asm["metal"]
msl = msl if isinstance(msl, str) else msl.decode()
print("MSL_BYTES", len(msl))
"""


def test_i32_keep_dims_reduce_compiles_in_bounded_time(tmp_path):
    import subprocess
    import sys
    script = tmp_path / "i32_keepdims.py"  # @triton.jit needs a source file
    script.write_text(_I32_KEEPDIMS_SCRIPT)
    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    size = int(proc.stdout.split("MSL_BYTES")[1].split()[0])
    assert size < 20000, f"emitted MSL is {size} bytes"


# `tl.sum(x)` with no axis is a reduce over `tl.reshape(x, (n,))`; the
# whole-tile aggregate replays that cone at a flat index, and a rank-2
# broadcast in the address must read its source at `i / N` (or `i % N`), not
# at `i`. Replaying `i` read every address with row == col.
# The two address spellings fail differently: an offset tile broadcasts
# integers, the inline spelling broadcasts a POINTER tile, whose walk used to
# stop at the broadcast and drop the row offset under it.
@triton.jit
def _whole_tile_reduce_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                              OP: tl.constexpr, INLINE: tl.constexpr):
    if INLINE:
        x = tl.load(x_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    else:
        off = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + off)
    if OP == 0:
        r = tl.sum(x)
    elif OP == 1:
        r = tl.max(x)
    elif OP == 2:
        r = tl.min(x)
    else:
        r = tl.sum(x * x)
    tl.store(out_ptr, r)


@pytest.mark.parametrize("shape", [(4, 32), (16, 16), (8, 128), (64, 16)])
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("op", [0, 1, 2, 3])
@pytest.mark.parametrize("inline", [False, True])
def test_no_axis_reduce_over_rank2_tile(shape, num_warps, op, inline):
    M, N = shape
    x = (torch.randperm(M * N, generator=torch.Generator().manual_seed(M * N + op))
         % 97 - 48).float().reshape(M, N)
    out = torch.zeros(1, device="mps")
    _whole_tile_reduce_kernel[(1,)](x.to("mps"), out, M, N, op, inline,
                                    num_warps=num_warps)
    torch.mps.synchronize()
    want = [x.sum(), x.max(), x.min(), (x * x).sum()][op]
    assert out.cpu()[0] == want


# A tile whose load reads a broadcast POINTER gets a column-major layout from
# Triton. Flattening it with `tl.reshape` and storing it through a rank-1
# address is an exchange whose two halves must use the labels the emitter
# addresses the tiles by, not the layouts Triton inferred for the reshape.
@triton.jit
def _flatten_broadcast_pointer_kernel(v_ptr, out_ptr, M: tl.constexpr,
                                      N: tl.constexpr):
    off = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    row = tl.arange(0, M)[:, None] + tl.zeros((M, N), tl.int32)
    x = (off % 13 - 6).to(tl.float32) + tl.load(v_ptr + row)
    tl.store(out_ptr + tl.arange(0, M * N), tl.reshape(x, (M * N,)))


@pytest.mark.parametrize("shape", [(16, 16), (8, 128), (32, 8), (4, 32)])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_flatten_broadcast_pointer_tile(shape, num_warps):
    M, N = shape
    v = (torch.arange(M) % 5 - 2).float()
    out = torch.zeros(M * N, device="mps")
    _flatten_broadcast_pointer_kernel[(1,)](v.to("mps"), out, M, N,
                                            num_warps=num_warps)
    torch.mps.synchronize()
    flat = torch.arange(M * N)
    assert torch.equal(out.cpu(), (flat % 13 - 6).float() + v[flat // N])


# The same flattened tile at rank 3, where Triton's order for it is neither
# row- nor column-major ([0, 2, 1]), and under order-sensitive consumers other
# than a store: an argmax, whose index is a range over the reshaped value, and
# a sum weighted by flat position. Triton marks a reshape that only feeds
# reduces `allow_reorder`; the argmax still needs row-major element t in lane t.
@triton.jit
def _flatten_broadcast_pointer_consumer_kernel(v_ptr, out_ptr, A: tl.constexpr,
                                               B: tl.constexpr, C: tl.constexpr,
                                               RANK: tl.constexpr,
                                               OP: tl.constexpr):
    if RANK == 2:
        off = tl.arange(0, A)[:, None] * B + tl.arange(0, B)[None, :]
        row = tl.arange(0, A)[:, None] + tl.zeros((A, B), tl.int32)
    else:
        off = (tl.arange(0, A)[:, None, None] * (B * C) +
               tl.arange(0, B)[None, :, None] * C + tl.arange(0, C)[None, None, :])
        row = tl.arange(0, A)[:, None, None] + tl.zeros((A, B, C), tl.int32)
    x = (off % 13 - 6).to(tl.float32) + tl.load(v_ptr + row)
    flat = tl.reshape(x, (A * B * C,))
    if OP == 0:
        tl.store(out_ptr + tl.arange(0, A * B * C), flat)
    elif OP == 1:
        tl.store(out_ptr, tl.argmax(flat, 0).to(tl.float32))
    else:
        tl.store(out_ptr, tl.sum(flat * (tl.arange(0, A * B * C) % 7).to(tl.float32), 0))


@pytest.mark.parametrize("shape", [(2, 16), (4, 32), (16, 16), (2, 4, 8), (4, 8, 16),
                                   (2, 16, 32)])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("op", ["store", "argmax", "weighted"])
def test_flatten_broadcast_pointer_tile_consumers(shape, num_warps, op):
    A, B, C = (list(shape) + [1])[:3]
    n = A * B * C
    v = (torch.arange(A) % 5 - 2).float()
    out = torch.zeros(n, device="mps")
    _flatten_broadcast_pointer_consumer_kernel[(1,)](
        v.to("mps"), out, A, B, C, len(shape), ["store", "argmax", "weighted"].index(op),
        num_warps=num_warps)
    torch.mps.synchronize()
    flat = torch.arange(n)
    want = (flat % 13 - 6).float() + v[flat // (B * C)]
    got = out.cpu()
    if op == "store":
        assert torch.equal(got, want)
    elif op == "argmax":
        assert int(got[0]) == int(torch.argmax(want))
    else:
        assert got[0] == (want * (flat % 7).float()).sum()


# An argmax over a flattened tile beside a `tl.flip` of the same tile. The flip
# splits the function into exchange phases, and the argmax's flattened source is
# rebuilt in its own phase; that copy has to be lowered like the original.
@triton.jit
def _flatten_argmax_beside_flip_kernel(x_ptr, y_ptr, i_ptr, M: tl.constexpr,
                                       N: tl.constexpr):
    off = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + off)
    tl.store(y_ptr + off, tl.flip(x, 0))
    tl.store(i_ptr, tl.argmax(tl.reshape(x, (M * N,)), 0))


@pytest.mark.parametrize("shape", [(4, 32), (16, 16), (8, 128)])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_flatten_argmax_beside_flip(shape, num_warps):
    M, N = shape
    g = torch.Generator().manual_seed(M * N + num_warps)
    x = (torch.randperm(M * N, generator=g) % 23 - 11).float()
    y = torch.zeros(M, N, device="mps")
    i = torch.zeros(1, dtype=torch.int32, device="mps")
    _flatten_argmax_beside_flip_kernel[(1,)](x.to("mps"), y, i, M, N,
                                             num_warps=num_warps)
    torch.mps.synchronize()
    assert torch.equal(y.cpu(), x.reshape(M, N).flip(0))
    assert x[int(i.cpu()[0])] == x.max()


# A whole-tile reduce whose cone casts an integer to f16. The rank-1 replay
# emitted every int -> float cast as f32, so `(i % 13).to(tl.float16) + x`
# became an f32 plus an f16 and the kernel failed the verifier with an
# internal type mismatch instead of reducing.
@triton.jit
def _int_to_half_cone_reduce_kernel(v_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                                    OP: tl.constexpr):
    off = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    row = tl.arange(0, M)[:, None] + tl.zeros((M, N), tl.int32)
    x = (off % 13 - 6).to(tl.float16) + tl.load(v_ptr + row)
    if OP == 0:
        tl.store(out_ptr, tl.sum(x))
    else:
        tl.store(out_ptr, tl.max(x))


@pytest.mark.parametrize("shape", [(16, 16), (32, 32), (8, 128)])
@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_int_to_half_cone_whole_tile_reduce(shape, num_warps, op):
    M, N = shape
    v = (torch.arange(M) % 5 - 2).half()
    out = torch.zeros(1, dtype=torch.float16, device="mps")
    _int_to_half_cone_reduce_kernel[(1,)](v.to("mps"), out, M, N, ["sum", "max"].index(op),
                                          num_warps=num_warps)
    torch.mps.synchronize()
    flat = torch.arange(M * N)
    t = (flat % 13 - 6).double() + v.double()[flat // N]
    want = t.sum() if op == "sum" else t.max()
    assert out.cpu()[0].item() == want.item()
