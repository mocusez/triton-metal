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


@triton.jit
def _numeric_cast_kernel(X, O, N: tl.constexpr, DT: tl.constexpr, BLOCK: tl.constexpr):
    if N == 1:
        offset = 0
    else:
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offset, mask=offset < N, other=0)
    tl.store(O + offset, value.to(DT), mask=offset < N)


@pytest.mark.parametrize("src,dst", [
    ("float16", "float32"), ("bfloat16", "float32"),
    ("float32", "float16"), ("float32", "bfloat16"),
    ("int32", "float32"), ("int64", "float32"),
    ("uint32", "float32"), ("uint64", "float32"),
    ("float32", "int8"), ("float32", "int16"),
    ("float32", "int32"), ("float32", "int64"),
    ("float32", "uint8"), ("float32", "uint16"),
    ("float32", "uint32"), ("float32", "uint64"),
])
@pytest.mark.parametrize("n", [1, 259], ids=["scalar", "tensor_tail"])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_numeric_cast_supported_widths_preserve_values(src, dst, n, num_warps):
    if src.startswith("uint"):
        width = int(src[4:])
        values = [2**(width - 1), 0, 1, 2**(width - 1) - 1, 2**width - 1]
    elif src.startswith("int"):
        width = int(src[3:])
        values = [-2**(width - 1), -1, 0, 1, 2**(width - 1) - 1]
    elif dst.startswith("uint"):
        width = int(dst[4:])
        step = max(1, 2**(width - 24))
        values = [1.75, 0.0, 127.5, float(2**(width - 1)), float(2**width - step)]
    elif dst.startswith("int"):
        values = [-1.75, 0.0, -0.0, 1.75, -127.5, 127.5]
    else:
        values = [-1.75, 0.0, -0.0, 1.75, -127.5, 256.0]
    x = torch.tensor(values, dtype=getattr(torch, src))
    x = x.repeat(triton.cdiv(n, x.numel()))[:n]
    expected = x.to(getattr(torch, dst))
    # MPS can bind uint64 buffers, but PyTorch's MPS fill kernel cannot
    # initialize them. Construct the sentinel on CPU before transferring it.
    output = torch.full((n + 3,), 37, dtype=getattr(torch, dst)).to("mps")
    _numeric_cast_kernel[(triton.cdiv(n, 256),)](
        x.to("mps"), output, N=n, DT=getattr(tl, dst), BLOCK=256, num_warps=num_warps,
    )
    torch.mps.synchronize()
    actual = output.cpu()
    assert actual[:n].tolist() == expected.tolist()
    assert actual[n:].tolist() == [37, 37, 37]
    if expected.is_floating_point():
        zeros = expected == 0
        assert torch.equal(torch.signbit(actual[:n][zeros]), torch.signbit(expected[zeros]))


# --- fp8: e4m3fn and e5m2 ----------------------------------------------------
#
# MSL has no 8-bit float type, so these casts run in software and the payload
# travels as i8 bits. Before this, fp8 existed only as a fully consumed
# `tt.dot_scaled` payload, which ruled out quantizing or dequantizing a tensor
# in a kernel at all.
#
# The assertions are bit-exact against `torch.Tensor.to`, which is the whole
# point: rounding is round-to-nearest-even applied ONCE off the f32 pattern.
# Rounding through f16 first would double-round — 119 of 60032 fuzz values came
# out a ulp away when it did — and the two formats differ at the top of their
# range: e5m2 saturates to its infinity, e4m3fn has none and overflows to NaN.


@triton.jit
def _fp8_round_trip_kernel(X, Out, N: tl.constexpr, E5: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    q = v.to(tl.float8e5) if E5 else v.to(tl.float8e4nv)
    tl.store(Out + i, q.to(tl.float32))


@triton.jit
def _fp8_store_kernel(X, Out, N: tl.constexpr, E5: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(X + i)
    tl.store(Out + i, v.to(tl.float8e5) if E5 else v.to(tl.float8e4nv))


@triton.jit
def _fp8_load_kernel(X, Out, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Out + i, tl.load(X + i).to(tl.float32))


def _fp8_dtype(e5):
    return torch.float8_e5m2 if e5 else torch.float8_e4m3fn


@pytest.mark.parametrize("e5", [False, True])
def test_fp8_round_trip_matches_torch_bitwise(e5):
    N = 1024
    torch.manual_seed(0)
    vals = torch.cat([torch.randn(N // 2) * 20.0,
                      torch.randn(N // 2) * 0.01]).to(torch.float32)
    out = torch.zeros(N, dtype=torch.float32, device="mps")
    _fp8_round_trip_kernel[(1,)](vals.to("mps"), out, N, e5, num_warps=4)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), vals.to(_fp8_dtype(e5)).to(torch.float32))


@pytest.mark.parametrize("e5", [False, True])
def test_fp8_store_writes_the_same_bytes_as_torch(e5):
    N = 512
    torch.manual_seed(1)
    vals = (torch.rand(N) * 1000.0 - 500.0).to(torch.float32)
    out = torch.zeros(N, dtype=_fp8_dtype(e5), device="mps")
    _fp8_store_kernel[(1,)](vals.to("mps"), out, N, e5, num_warps=4)
    torch.mps.synchronize()
    assert torch.equal(out.cpu().view(torch.uint8),
                       vals.to(_fp8_dtype(e5)).view(torch.uint8))


@pytest.mark.parametrize("e5", [False, True])
def test_fp8_load_decodes_every_bit_pattern(e5):
    dt = _fp8_dtype(e5)
    pats = torch.arange(256, dtype=torch.uint8).view(dt)
    out = torch.zeros(256, dtype=torch.float32, device="mps")
    _fp8_load_kernel[(1,)](pats.to("mps"), out, 256, num_warps=4)
    torch.mps.synchronize()
    got, want = out.cpu(), pats.to(torch.float32)
    for i in range(256):
        if want[i] != want[i]:            # NaN patterns compare by kind
            assert got[i] != got[i], (i, got[i].item())
        else:
            assert got[i] == want[i], (i, got[i].item(), want[i].item())
