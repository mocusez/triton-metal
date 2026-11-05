"""Host-side `TensorDescriptor` arguments (Triton's TMA surface).

The compiler side has worked for a while: a `tensordesc<...>` kernel argument is
flattened by the frontend into 4 * rank + 3 parameters —

    base pointer, *shape (i64), *strides (i64), padding == "nan" (i1),
    round_f32_to_tf32 (i1), *shape (i32), *strides (i64)

(shape and strides appear twice: once as the pair the descriptor carries, once
as the pair that survives descriptor lowering) — and the MSL signature has a
slot for each. The LAUNCHER did not: it passed the `TensorDescriptor` object
through as a single argument, which `torch.mps.compile_shader` rejected with a
bare "RuntimeError: Unsupported argument type" that named neither the argument
nor the reason.

The launcher now expands the descriptor with Triton's own
`expand_signature` / `decompose_descriptor`, the same pair the NVIDIA launcher
uses, so the two sides cannot drift on the order. Metal explicitly selects
`TensorDescABI.DECOMPOSED`: it has no hardware descriptor object, so the
decomposed form is the calling convention.

Metal has no TMA hardware and none of this is asynchronous — descriptor loads,
stores, and reducing stores are rewritten to ordinary pointer operations before
Metal conversion. Reducing stores become masked atomics, preserving descriptor
view bounds. These tests cover that rewrite and the calling convention, not a
DMA engine.
"""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: E402

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


@triton.jit
def desc_load_1d(d, o_ptr, B: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, B), d.load([0]))


@triton.jit
def desc_load_1d_offset(d, o_ptr, off, B: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, B), d.load([off]))


@triton.jit
def desc_load_2d(d, o_ptr, M: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, M)
    j = tl.arange(0, N)
    tl.store(o_ptr + i[:, None] * N + j[None, :], d.load([0, 0]))


@triton.jit
def desc_store_1d(d, x_ptr, B: tl.constexpr):
    d.store([0], tl.load(x_ptr + tl.arange(0, B)))


@triton.jit
def desc_atomic_i32(d, x_ptr, offset, KIND: tl.constexpr, B: tl.constexpr):
    values = tl.load(x_ptr + tl.arange(0, B))
    if KIND == "add":
        d.atomic_add([offset], values)
    elif KIND == "min":
        d.atomic_min([offset], values)
    elif KIND == "max":
        d.atomic_max([offset], values)
    elif KIND == "and":
        d.atomic_and([offset], values)
    elif KIND == "or":
        d.atomic_or([offset], values)
    else:
        tl.static_assert(KIND == "xor")
        d.atomic_xor([offset], values)


@triton.jit
def desc_atomic_add_1d(d, x_ptr, offset, B: tl.constexpr):
    d.atomic_add([offset], tl.load(x_ptr + tl.arange(0, B)))


@triton.jit
def desc_atomic_minmax_1d(d, x_ptr, offset, KIND: tl.constexpr, B: tl.constexpr):
    values = tl.load(x_ptr + tl.arange(0, B))
    if KIND == "min":
        d.atomic_min([offset], values)
    else:
        tl.static_assert(KIND == "max")
        d.atomic_max([offset], values)


@triton.jit
def desc_atomic_add_2d(d, x_ptr, offset0, offset1, M: tl.constexpr, N: tl.constexpr):
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    d.atomic_add([offset0, offset1], tl.load(x_ptr + offsets))


@triton.jit
def desc_atomic_add_3d(d, offset0, offset1, offset2):
    d.atomic_add(
        [offset0, offset1, offset2], tl.full((2, 4, 8), 1, tl.int32)
    )


@triton.jit
def desc_atomic_add_4d(d, offset0, offset1, offset2, offset3):
    d.atomic_add(
        [offset0, offset1, offset2, offset3],
        tl.full((2, 4, 4, 8), 1, tl.int32),
    )


@triton.jit
def desc_atomic_add_5d(d, offset0, offset1, offset2, offset3, offset4):
    d.atomic_add(
        [offset0, offset1, offset2, offset3, offset4],
        tl.full((2, 2, 4, 4, 8), 1, tl.int32),
    )


@triton.jit
def device_desc_atomic_add_3d(
    out,
    shape0,
    shape1,
    shape2,
    stride0,
    stride1,
    offset0,
    offset1,
    offset2,
):
    d = tl.make_tensor_descriptor(
        out,
        shape=[shape0, shape1, shape2],
        strides=[stride0, stride1, 1],
        block_shape=[2, 4, 8],
    )
    d.atomic_add(
        [offset0, offset1, offset2], tl.full((2, 4, 8), 1, tl.int32)
    )


@triton.jit
def device_desc_atomic_add_4d(
    out, shape0, shape1, shape2, shape3, stride0, stride1, stride2,
    offset0, offset1, offset2, offset3,
):
    d = tl.make_tensor_descriptor(
        out,
        shape=[shape0, shape1, shape2, shape3],
        strides=[stride0, stride1, stride2, 1],
        block_shape=[2, 4, 4, 8],
    )
    d.atomic_add(
        [offset0, offset1, offset2, offset3],
        tl.full((2, 4, 4, 8), 1, tl.int32),
    )


@triton.jit
def device_desc_atomic_add_5d(
    out, shape0, shape1, shape2, shape3, shape4,
    stride0, stride1, stride2, stride3,
    offset0, offset1, offset2, offset3, offset4,
):
    d = tl.make_tensor_descriptor(
        out,
        shape=[shape0, shape1, shape2, shape3, shape4],
        strides=[stride0, stride1, stride2, stride3, 1],
        block_shape=[2, 2, 4, 4, 8],
    )
    d.atomic_add(
        [offset0, offset1, offset2, offset3, offset4],
        tl.full((2, 2, 4, 4, 8), 1, tl.int32),
    )


@pytest.mark.parametrize("B", [16, 32, 64])
def test_tensor_descriptor_load_1d(B):
    x = torch.arange(256, dtype=torch.float32, device="mps")
    out = torch.zeros(B, dtype=torch.float32, device="mps")
    desc_load_1d[(1,)](TensorDescriptor.from_tensor(x, [B]), out, B)
    assert torch.equal(out.cpu(), x.cpu()[:B])


@pytest.mark.parametrize("off", [0, 16, 32])
def test_tensor_descriptor_load_1d_offset(off):
    """The i32 copy of `shape` and the i64 copy of `strides` are what the
    offset arithmetic reads, so a non-zero offset is what catches them being
    bound to the wrong slots."""
    x = torch.arange(256, dtype=torch.float32, device="mps")
    out = torch.zeros(16, dtype=torch.float32, device="mps")
    desc_load_1d_offset[(1,)](TensorDescriptor.from_tensor(x, [16]), out, off, 16)
    assert torch.equal(out.cpu(), x.cpu()[off:off + 16])


@pytest.mark.parametrize(
    "M,N",
    [pytest.param(8, 16, id="8x16"), pytest.param(16, 16, id="16x16"),
     pytest.param(4, 32, id="4x32")],
)
def test_tensor_descriptor_load_2d(M, N):
    x = torch.arange(64 * 64, dtype=torch.float32, device="mps").reshape(64, 64)
    out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    desc_load_2d[(1,)](TensorDescriptor.from_tensor(x, [M, N]), out, M, N)
    assert torch.equal(out.cpu(), x.cpu()[:M, :N])


@pytest.mark.parametrize("B", [16, 32])
def test_tensor_descriptor_store_1d(B):
    x = torch.arange(256, dtype=torch.float32, device="mps")
    dst = torch.zeros(256, dtype=torch.float32, device="mps")
    desc_store_1d[(1,)](TensorDescriptor.from_tensor(dst, [B]), x, B)
    assert torch.equal(dst.cpu()[:B], x.cpu()[:B])


@pytest.mark.parametrize("kind", ["add", "min", "max", "and", "or", "xor"])
def test_tensor_descriptor_atomic_i32(kind):
    block = 16
    initial = torch.full((block,), 0x0F0F0F0F, dtype=torch.int32)
    values = torch.arange(1, block + 1, dtype=torch.int32)
    dst = initial.to("mps")
    values_mps = values.to("mps")

    desc_atomic_i32[(1,)](
        TensorDescriptor.from_tensor(dst, [block]),
        values_mps,
        0,
        KIND=kind,
        B=block,
    )

    references = {
        "add": initial + values,
        "min": torch.minimum(initial, values),
        "max": torch.maximum(initial, values),
        "and": torch.bitwise_and(initial, values),
        "or": torch.bitwise_or(initial, values),
        "xor": torch.bitwise_xor(initial, values),
    }
    assert torch.equal(dst.cpu(), references[kind])


def test_tensor_descriptor_atomic_add_masks_partial_view():
    block = 16
    logical_size = 18
    dst = torch.zeros(logical_size, dtype=torch.int32, device="mps")
    values = torch.arange(1, block + 1, dtype=torch.int32, device="mps")

    desc_atomic_i32[(1,)](
        TensorDescriptor.from_tensor(dst, [block]),
        values,
        16,
        KIND="add",
        B=block,
    )

    expected = torch.zeros(logical_size, dtype=torch.int32)
    expected[16:] = torch.tensor([1, 2], dtype=torch.int32)
    assert torch.equal(dst.cpu(), expected)


def test_tensor_descriptor_atomic_add_contended():
    block = 16
    programs = 4
    dst = torch.zeros(block, dtype=torch.int32, device="mps")
    values = torch.ones(block, dtype=torch.int32, device="mps")

    desc_atomic_i32[(programs,)](
        TensorDescriptor.from_tensor(dst, [block]),
        values,
        0,
        KIND="add",
        B=block,
    )

    assert torch.equal(dst.cpu(), torch.full((block,), programs, dtype=torch.int32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.uint32])
def test_tensor_descriptor_atomic_add_dtypes(dtype):
    logical_size = 17
    block = 32
    programs = 4
    initial = torch.full((logical_size,), 2, dtype=torch.int32).to(dtype)
    values = torch.arange(1, block + 1, dtype=torch.int32).to(dtype)
    dst = initial.to("mps")
    values_mps = values.to("mps")

    desc_atomic_add_1d[(programs,)](
        TensorDescriptor.from_tensor(dst, [block]), values_mps, 0, B=block
    )

    expected = [2 + programs * value for value in range(1, logical_size + 1)]
    assert dst.cpu().tolist() == expected


def test_tensor_descriptor_atomic_add_f16_rank2():
    m, n = 4, 8
    programs = 4
    initial = torch.full((m, n), 3, dtype=torch.float16)
    values = torch.ones(m * n, dtype=torch.float16)
    dst = initial.to("mps")

    desc_atomic_add_2d[(programs,)](
        TensorDescriptor.from_tensor(dst, [m, n]), values.to("mps"), 0, 0, M=m, N=n
    )

    assert torch.equal(dst.cpu(), initial + programs)


@pytest.mark.parametrize("kind", ["min", "max"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.uint32, torch.uint64])
def test_tensor_descriptor_atomic_minmax_dtypes(kind, dtype):
    if dtype == torch.int32:
        initial_values = [17, -8, 100, 3]
        value_items = [5, -12, 100, 99]
    elif dtype == torch.uint32:
        initial_values = [0, 2**31 - 1, 2**31, 2**32 - 1]
        value_items = [2**32 - 1, 2**31, 2**31 - 1, 0]
    else:
        initial_values = [0, 2**63 - 1, 2**63, 2**64 - 1]
        value_items = [2**64 - 1, 2**63, 2**63 - 1, 0]
    initial = torch.tensor(initial_values, dtype=dtype)
    values = torch.tensor(value_items, dtype=dtype)
    dst = initial.to("mps")

    desc_atomic_minmax_1d[(1,)](
        TensorDescriptor.from_tensor(dst, [4]),
        values.to("mps"),
        0,
        KIND=kind,
        B=4,
    )

    operation = min if kind == "min" else max
    expected = [operation(a, b) for a, b in zip(initial_values, value_items)]
    assert dst.cpu().tolist() == expected


def test_tensor_descriptor_atomic_bitwise_u32():
    block = 16
    initial = torch.full((block,), 0xF0F0F0F0, dtype=torch.int64).to(torch.uint32)
    values = torch.arange(1, block + 1, dtype=torch.int64).to(torch.uint32)

    for kind in ["and", "or", "xor"]:
        current = initial.to("mps")
        desc_atomic_i32[(1,)](
            TensorDescriptor.from_tensor(current, [block]),
            values.to("mps"),
            0,
            KIND=kind,
            B=block,
        )
        if kind == "and":
            expected = initial & values
        elif kind == "or":
            expected = initial | values
        else:
            expected = initial ^ values
        assert torch.equal(current.cpu(), expected)


def _check_desc_atomic_add_strided(
    kernel, shape, strides, block, origin, use_device=False
):
    storage = sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1
    programs = 3
    initial = torch.full((storage,), 117, dtype=torch.int32)
    dst = initial.to("mps")
    if use_device:
        device_kernels = {
            3: device_desc_atomic_add_3d,
            4: device_desc_atomic_add_4d,
            5: device_desc_atomic_add_5d,
        }
        device_kernels[len(shape)][(programs,)](
            dst, *shape, *strides[:-1], *origin
        )
    else:
        desc = TensorDescriptor(dst, shape, strides, block)
        kernel[(programs,)](desc, *origin)

    expected = initial.clone()
    for block_index in itertools.product(*(range(size) for size in block)):
        logical = [offset + index for offset, index in zip(origin, block_index)]
        if all(0 <= index < size for index, size in zip(logical, shape)):
            flat = sum(index * stride for index, stride in zip(logical, strides))
            expected[flat] += programs
    actual = dst.cpu()
    mismatches = [
        (index, actual[index].item(), expected[index].item())
        for index in range(storage)
        if actual[index].item() != expected[index].item()
    ]
    assert not mismatches, (mismatches[:5] + mismatches[-5:])


@pytest.mark.parametrize(
    "kernel,shape,strides,block,origin",
    [
        (
            desc_atomic_add_3d,
            (3, 5, 9),
            (64, 16, 1),
            (2, 4, 8),
            (-1, 2, 3),
        ),
        (
            desc_atomic_add_4d,
            (2, 3, 4, 9),
            (152, 48, 12, 1),
            (2, 4, 4, 8),
            (0, -1, 2, 5),
        ),
        (
            desc_atomic_add_5d,
            (2, 2, 3, 4, 9),
            (304, 148, 48, 12, 1),
            (2, 2, 4, 4, 8),
            (1, 0, -1, 1, 4),
        ),
    ],
    ids=["rank3", "rank4", "rank5"],
)
def test_tensor_descriptor_atomic_add_strided_rank(
    kernel, shape, strides, block, origin
):
    _check_desc_atomic_add_strided(kernel, shape, strides, block, origin)


@pytest.mark.parametrize(
    "shape,strides,block,origin",
    [
        ((3, 5, 9), (64, 16, 1), (2, 4, 8), (-1, 2, 3)),
        ((2, 3, 4, 9), (152, 48, 12, 1), (2, 4, 4, 8), (0, -1, 2, 5)),
        (
            (2, 2, 3, 4, 9),
            (304, 148, 48, 12, 1),
            (2, 2, 4, 4, 8),
            (1, 0, -1, 1, 4),
        ),
    ],
    ids=["rank3", "rank4", "rank5"],
)
def test_device_tensor_descriptor_atomic_add_strided_rank(
    shape, strides, block, origin
):
    _check_desc_atomic_add_strided(
        None, shape, strides, block, origin, use_device=True
    )
