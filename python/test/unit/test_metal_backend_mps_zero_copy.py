"""Zero-cost PyTorch MPS <-> Triton integration.

These tests assert the Metal backend launches Triton kernels directly on
PyTorch MPS tensors with no host staging: the kernel writes in place into the
MPS tensor's own MTLBuffer (data_ptr stable), honors storage_offset for
views/slices, binds Python scalars, supports multi-dim grids, and still accepts
CPU tensors via a copy-back path. See
the implementation notes.

The launch path routes through `torch.mps.compile_shader` (driver.py
`MetalUtils.load_binary` + `MetalLauncher`). Set `TRITON_METAL_USE_MPS=0` to
fall back to the native metallib runtime.
"""

import pytest
import torch

import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires a PyTorch build with an available MPS device",
)


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)


@triton.jit
def _scale_kernel(x_ptr, out_ptr, scale, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * scale, mask=mask)


@pytest.mark.parametrize("N", [256, 5000, 1 << 16])
def test_add_mps_zero_copy(N):
    BLOCK = 1024
    x = torch.randn(N, device="mps")
    y = torch.randn(N, device="mps")
    out = torch.empty(N, device="mps")
    out_ptr = out.data_ptr()
    _add_kernel[(triton.cdiv(N, BLOCK),)](x, y, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(out, x + y)
    # In-place into the tensor's own MTLBuffer: no realloc, no host round-trip.
    assert out.data_ptr() == out_ptr


def test_add_mps_sliced_storage_offset():
    """A view with a 16-byte-aligned storage_offset must bind at the right offset.

    Aligned offset: out keeps tt.divisibility = 16, so all three pointers share
    one blocked layout (no convert_layout). The unaligned counterpart is
    test_add_mps_unaligned_storage_offset below.
    """
    N, BLOCK = 4096, 1024
    OFF = 16  # 16 elems * 4 bytes = 64 B, 16-byte aligned
    base = torch.zeros(N + 64, device="mps")
    x = torch.randn(N, device="mps")
    y = torch.randn(N, device="mps")
    out = base[OFF:OFF + N]  # storage_offset == OFF (non-zero)
    _add_kernel[(triton.cdiv(N, BLOCK),)](x, y, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(out, x + y)
    # Bytes outside the view stay untouched (offset honored, no overrun).
    assert torch.count_nonzero(base[:OFF]) == 0
    assert torch.count_nonzero(base[OFF + N:]) == 0


@pytest.mark.parametrize("OFF", [1, 7, 13])
def test_add_mps_unaligned_storage_offset(OFF):
    """An unaligned output view mixed with aligned inputs must still be correct.

    out = base[OFF:] with OFF not a multiple of 4 is NOT 16-byte aligned, so
    Triton drops tt.divisibility = 16 for out and coalesces it to
    sizePerThread = 1 while the aligned x/y vectorize to sizePerThread = 4. The
    frontend bridges the two with a rank-1 ttg.convert_layout, which the Metal
    backend used to reject ("broader staged-transpose deferred to L1d3"). The
    normalizeRank1DivergentCvts pre-pass now collapses it. Regression for that
    fix; IR-level coverage is convert_layout_rank1_divergent_spt.mlir.
    """
    N, BLOCK = 4096, 1024
    base = torch.zeros(N + 64, device="mps")
    x = torch.randn(N, device="mps")
    y = torch.randn(N, device="mps")
    out = base[OFF:OFF + N]
    assert out.data_ptr() % 16 != 0, "offset must be unaligned to exercise the fix"
    _add_kernel[(triton.cdiv(N, BLOCK),)](x, y, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(out, x + y)
    # Offset honored, no overrun outside the view.
    assert torch.count_nonzero(base[:OFF]) == 0
    assert torch.count_nonzero(base[OFF + N:]) == 0


def test_scale_mps_scalar_arg():
    """Python scalar binds into the kernel's scalar (1-element buffer) slot."""
    N, BLOCK = 2048, 1024
    x = torch.randn(N, device="mps")
    out = torch.empty(N, device="mps")
    _scale_kernel[(triton.cdiv(N, BLOCK),)](x, out, 3.5, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(out, x * 3.5)


def test_add_cpu_tensor_writeback():
    """CPU inputs are bridged to MPS and the result is copied back in place."""
    N, BLOCK = 1024, 1024
    x = torch.randn(N)
    y = torch.randn(N)
    out = torch.empty(N)
    _add_kernel[(triton.cdiv(N, BLOCK),)](x, y, out, N, BLOCK=BLOCK)
    torch.testing.assert_close(out, x + y)


# A @triton.jit function's name becomes the MSL entry-point name verbatim, and
# `kernel` / `device` are reserved MSL keywords -> `kernel void kernel(...)` is a
# torch.mps.compile_shader SyntaxError. This is exactly what
# python/test/unit/fixtures/metal_leet/medium-monte_carlo_integration.py hit (its jit fn is literally
# `kernel`). sanitizeKernelName (ModuleTranslation.cpp) mangles a reserved entry
# name to `triton_<name>`; compiler.py make_msl re-greps that name into
# metadata["name"] and driver.py load_binary does getattr(lib, name), so the
# emitter, the metadata, and the dispatch lookup must all agree. This exercises
# that whole chain end-to-end; IR-level coverage is
# test/Dialect/Metal/metal-translate/kernel-name-reserved-word.mlir.
@triton.jit
def kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)


@triton.jit
def device(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)


@pytest.mark.parametrize("fn, mangled", [(kernel, "triton_kernel"), (device, "triton_device")])
def test_reserved_word_kernel_name_runs(fn, mangled):
    N, BLOCK = 4096, 1024
    x = torch.randn(N, device="mps")
    y = torch.randn(N, device="mps")
    out = torch.empty(N, device="mps")
    compiled = fn[(triton.cdiv(N, BLOCK),)](x, y, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(out, x + y)
    # The emitted MSL entry point -- and thus metadata["name"] and the
    # getattr(lib, name) dispatch lookup -- is the mangled, non-reserved name.
    assert compiled.name == mangled
