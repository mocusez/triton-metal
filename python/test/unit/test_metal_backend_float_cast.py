"""Float widen/narrow (arith.extf / arith.truncf) on the Metal backend.

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
