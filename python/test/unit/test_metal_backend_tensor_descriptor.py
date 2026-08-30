"""Host-side `TensorDescriptor` arguments (Triton's TMA surface).

The compiler side has worked for a while: a `tensordesc<...>` kernel argument is
flattened by the frontend into seven parameters —

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
uses, so the two sides cannot drift on the order. `tensordesc_meta` is None for
this backend on purpose: Metal has no hardware descriptor object, so the
decomposed form IS the calling convention.

Metal has no TMA hardware and none of this is asynchronous — a descriptor load
lowers to the same per-element addressing an ordinary `tl.load` does. These
tests are about the calling convention, not about a DMA engine.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

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
