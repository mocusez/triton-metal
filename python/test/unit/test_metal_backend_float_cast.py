"""Floating-point casts on the Metal backend.

fp16-in / fp32-compute / fp16-out is the ubiquitous shape for layer-norm,
softmax, and attention kernels: `tl.load(fp16).to(tl.float32)` lowers to a
tensor `arith.extf`, and storing an fp32 result back to an fp16 buffer emits a
tensor `arith.truncf`. `ArithExtFLowering` / `ArithTruncFLowering` scalarize
these (mirroring `ArithSIToFPLowering`); the MSL emitter already spells them as
`float(x)` / `half(x)` constructor casts. bf16 rides the same path.

Runtime bit-exact (the casts are exact for the value ranges here) — not just a
compile smoke.
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
def _cast_roundtrip_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)     # low-precision load
    xf = x.to(tl.float32)                     # arith.extf  (widen)
    yf = xf * 2.0 + 1.0                       # fp32 compute
    y = yf.to(x.dtype)                         # arith.truncf (narrow back)
    tl.store(y_ptr + offs, y, mask=mask)


# bf16 rides the same extf/truncf path; its float constants (e.g. the masked-load
# `other=0.0`) are wrapped `bfloat(...)` since MSL forbids implicit float->bfloat.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N", [64, 200, 1000])
def test_extf_truncf_roundtrip(dtype, N):
    torch.manual_seed(N)
    x = torch.randn(N, dtype=dtype, device="mps")
    y = torch.empty_like(x)
    _cast_roundtrip_kernel[(1,)](x, y, N, BLOCK=triton.next_power_of_2(N))
    torch.mps.synchronize()
    ref = (x.cpu().float() * 2.0 + 1.0).to(dtype)
    # extf(compute)truncf reproduces the torch cast chain bit-for-bit.
    torch.testing.assert_close(y.cpu().float(), ref.float(), atol=0, rtol=0)


@triton.jit
def _sum_fp16_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # extf inside a reduce cone: sum a low-precision row in fp32.
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(x, axis=0))


@pytest.mark.parametrize("N", [128, 777, 1024])
def test_extf_in_reduce_cone(N):
    torch.manual_seed(N)
    x = torch.randn(N, dtype=torch.float16, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")
    _sum_fp16_kernel[(1,)](x, out, N, BLOCK=triton.next_power_of_2(N))
    torch.mps.synchronize()
    ref = x.cpu().float().sum()
    torch.testing.assert_close(out.cpu()[0], ref, atol=1e-2, rtol=1e-2)


@triton.jit
def _bool_to_float_after_reduce_kernel(x_ptr, out_ptr, N,
                                       BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    col = tl.arange(0, 2)
    mask = row < N
    x = tl.load(
        x_ptr + row[:, None] * 2 + col[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    distance_sq = tl.sum(x * x, axis=1)
    selected = ((distance_sq < 25.0) & mask).to(tl.float32)
    tl.store(out_ptr + row, selected, mask=mask)


@triton.jit
def _fptosi_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x.to(tl.int32)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _fptoui_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x.to(tl.uint32)
    tl.store(out_ptr + offs, y, mask=mask)


@pytest.mark.parametrize("N", [17, 1000])
def test_bool_to_float_after_reduce(N):
    """A slice-encoded i1 tile must scalarize through arith.uitofp."""
    torch.manual_seed(N)
    x = torch.randn((N, 2), dtype=torch.float32, device="mps") * 4.0
    out = torch.empty(N, dtype=torch.float32, device="mps")
    _bool_to_float_after_reduce_kernel[(1,)](
        x, out, N, BLOCK=1024, num_warps=4
    )
    torch.mps.synchronize()
    expected = ((x.cpu() * x.cpu()).sum(dim=1) < 25.0).float()
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("N", [17, 1000])
def test_fptosi_runtime_truncates_toward_zero(N):
    torch.manual_seed(N)
    x = (torch.randn(N, dtype=torch.float32, device="mps") * 100.0).contiguous()
    out = torch.empty(N, dtype=torch.int32, device="mps")
    _fptosi_kernel[(1,)](x, out, N, BLOCK=triton.next_power_of_2(N))
    torch.mps.synchronize()
    expected = torch.trunc(x.cpu()).to(torch.int32)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("N", [17, 1000])
def test_fptoui_runtime_truncates_toward_zero(N):
    # Include the upper half of u32 so a mistaken signed MSL cast is visible.
    # Near 2^32, fp32 values are spaced by 256; all values below are exact.
    values = torch.tensor(
        [0.0, 1.75, float(2**31), float(2**31 + 256), float(2**32 - 256)],
        dtype=torch.float32,
        device="mps",
    )
    x = values.repeat((N + values.numel() - 1) // values.numel())[:N].contiguous()
    out = torch.empty(N, dtype=torch.uint32, device="mps")
    _fptoui_kernel[(1,)](x, out, N, BLOCK=triton.next_power_of_2(N))
    torch.mps.synchronize()
    expected = torch.trunc(x.cpu()).to(torch.uint32)
    assert out.cpu().tolist() == expected.tolist()
