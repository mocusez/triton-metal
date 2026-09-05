"""L3a-tileloop-compiler-A: scalar tt.load on bare !tt.ptr<f32>.

Drives a small @triton.jit kernel that performs a scalar `tl.load(scalar_ptr
+ scalar_offset)` inside a `tl.static_range` loop, and asserts bit-exact
agreement with a CPU reference across 5 deterministic runs. The kernel
mirrors the conv1d Variant B shape that motivated this session.

See the implementation notes.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not torch.backends.mps.is_available(),
    reason="Metal backend requires Darwin + MPS",
)


@triton.jit
def _scalar_load_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    N,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """For each i in [0, BLOCK): output[i] = sum(weight[k] * input[i+k] for k in [0, K))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(K):
        offs_in = offs + k
        mask_in = offs_in < N
        inp = tl.load(input_ptr + offs_in, mask_in)
        kv = tl.load(weight_ptr + k)  # SCALAR load — the gap this session unblocks.
        acc += kv * inp
    mask_out = offs < (N - K + 1)
    tl.store(output_ptr + offs, acc, mask_out)


def _run_one(N: int = 4096, K: int = 7, BLOCK: int = 1024) -> None:
    torch.manual_seed(0)
    inp = torch.randn(N, dtype=torch.float32, device="mps")
    weight = torch.randn(K, dtype=torch.float32, device="mps")
    out_size = N - K + 1
    out = torch.zeros(out_size, dtype=torch.float32, device="mps")

    n_blocks = triton.cdiv(out_size, BLOCK)
    _scalar_load_kernel[(n_blocks,)](weight, inp, out, N, K, BLOCK)

    expected = torch.nn.functional.conv1d(
        inp.view(1, 1, -1), weight.view(1, 1, -1)
    ).flatten()
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3), (
        f"max abs err = {(out - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("run_idx", range(5))
def test_scalar_load_conv1d_shape_bit_exact(run_idx: int) -> None:
    _run_one(N=4096, K=7, BLOCK=1024)


# Driver smoke that `tl.load(..., other=-float('inf'))` compiles past
# `xcrun metal` and that the masked-out tail loads as -inf.


@triton.jit
def _masked_load_neg_inf_other_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs, mask=offs < N, other=-float("inf"))
    tl.store(out_ptr + offs, x)


@pytest.mark.parametrize("N,BLOCK", [(100, 128), (200, 256), (700, 1024)])
def test_masked_load_with_neg_inf_other(N: int, BLOCK: int) -> None:
    torch.manual_seed(N)
    inp = torch.randn(N, dtype=torch.float32, device="mps")
    out = torch.zeros(BLOCK, dtype=torch.float32, device="mps")
    _masked_load_neg_inf_other_kernel[(1,)](inp, out, N, BLOCK)

    # In-range entries match the input bit-exactly.
    assert torch.equal(out[:N], inp), (
        f"N={N},BLOCK={BLOCK}: in-range mismatch; max abs err="
        f"{(out[:N] - inp).abs().max().item()}"
    )
    # Out-of-range tail is the masked `other` value: -inf.
    tail = out[N:]
    assert torch.all(torch.isinf(tail) & (tail < 0)), (
        f"N={N},BLOCK={BLOCK}: tail should be -inf, got {tail[:8].tolist()}"
    )


# The dtype envelope. `ScalarLoadLowering` admitted f32/i32 only -- a leftover
# from this session's original spec -- while the TENSOR load path next to it has
# never had a dtype gate at all, taking whatever the memref stores. So
# `v = tl.load(scalar_ptr)` on an i8/i16/f16 scalar declined inside
# `applyFullConversion`, and that decline does not raise on this backend: the
# rollback leaves the module unverifiable and the process dies (SIGTRAP, SIGBUS
# or SIGSEGV, intermittently -- 8 kills in 30 runs for the i64 shape, so a retry
# could look clean). The identical load written as a 1-element TENSOR worked
# throughout, which is why nothing in the suite noticed.
#
# f32/i32 are kept as the controls: they are the two that always worked.
_SCALAR_DTYPES = [
    ("int8", 1),
    ("int16", 2),
    ("int32", 3),
    ("int64", 4),
    ("float16", 5),
    ("bfloat16", 6),
    ("float32", 7),
]


@triton.jit
def _scalar_load_dtype_kernel(s_ptr, out_ptr, BLOCK: tl.constexpr):
    v = tl.load(s_ptr)
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, offs.to(tl.float32) + v.to(tl.float32))


@pytest.mark.parametrize("dtype_name,value", _SCALAR_DTYPES)
def test_scalar_load_dtype_envelope(dtype_name: str, value: int) -> None:
    BLOCK = 64
    dtype = getattr(torch, dtype_name)
    s = torch.full((1,), value, dtype=dtype, device="mps")
    out = torch.zeros(BLOCK, dtype=torch.float32, device="mps")
    _scalar_load_dtype_kernel[(1,)](s, out, BLOCK)
    torch.mps.synchronize()

    expected = torch.arange(BLOCK, dtype=torch.float32) + float(value)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _scalar_load_negative_kernel(s_ptr, out_ptr):
    tl.store(out_ptr, tl.load(s_ptr).to(tl.float32))


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_scalar_load_negative_int(dtype_name: str) -> None:
    """A negative scalar: storage is unsigned, so the sitofp must say signed.

    Widening the dtype envelope moved i8/i16/i64 onto the same ui-storage
    bridge i32 already used, which is the path that was silently wrong before
    the op-carries-signedness fix. Pin it here for the widths that just arrived.
    """
    dtype = getattr(torch, dtype_name)
    s = torch.full((1,), -7, dtype=dtype, device="mps")
    out = torch.zeros(1, dtype=torch.float32, device="mps")
    _scalar_load_negative_kernel[(1,)](s, out)
    torch.mps.synchronize()
    assert out.cpu().item() == -7.0


@triton.jit
def _masked_scalar_load_kernel(x_ptr, fallback_ptr, out_ptr, N, fallback, MODE: tl.constexpr):
    pid = tl.program_id(0)
    if MODE == "constant":
        value = tl.load(x_ptr + pid, pid < N, other=-7)
    elif MODE == "runtime":
        value = tl.load(x_ptr + pid, pid < N, other=fallback + pid)
    elif MODE == "loaded":
        other = tl.load(fallback_ptr + pid)
        value = tl.load(x_ptr + pid, pid < N, other=other)
    else:
        value = tl.load(x_ptr + pid, pid < N)
    tl.store(out_ptr + pid, value)


@pytest.mark.parametrize("dtype_name,value", _SCALAR_DTYPES)
@pytest.mark.parametrize("mode", ["constant", "runtime", "loaded", "none"])
@pytest.mark.parametrize("n", [0, 5, 17])
def test_masked_scalar_load_dtype_and_other(dtype_name, value, mode, n):
    dtype = getattr(torch, dtype_name)
    programs = 17
    # The masked-off program IDs are outside the input's logical extent.
    source = (torch.arange(max(n, 1)) + value).to(dtype)
    fallback = (-torch.arange(programs) - 11).to(dtype)
    out = torch.empty(programs, dtype=dtype, device="mps")
    _masked_scalar_load_kernel[(programs,)](
        source.to("mps"), fallback.to("mps"), out, n, -29, MODE=mode,
    )
    torch.mps.synchronize()
    actual = out.cpu()
    torch.testing.assert_close(actual[:n], source[:n], atol=0, rtol=0)
    if mode == "constant":
        expected = torch.full((programs,), -7, dtype=dtype)
    elif mode == "runtime":
        expected = (-29 + torch.arange(programs)).to(dtype)
    elif mode == "loaded":
        expected = fallback
    else:
        # Triton leaves the masked-off result undefined without `other`.
        return
    torch.testing.assert_close(actual[n:], expected[n:], atol=0, rtol=0)


@triton.jit
def _masked_scalar_load_loop_kernel(x_ptr, out_ptr, N, TRIPS, STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    for trip in range(TRIPS):
        ptr = x_ptr + trip * STRIDE + pid
        value = tl.load(ptr, pid < N - trip, other=trip - 9)
        tl.store(out_ptr + trip * STRIDE + pid, value)


@pytest.mark.parametrize("dtype_name", ["int16", "float32"])
@pytest.mark.parametrize("trips", [0, 1, 3])
def test_masked_scalar_load_loop_preserves_iteration_offsets(dtype_name, trips):
    dtype = getattr(torch, dtype_name)
    stride, n = 8, 5
    source = torch.arange(3 * stride).to(dtype)
    out = torch.full((3 * stride,), -99, dtype=dtype, device="mps")
    _masked_scalar_load_loop_kernel[(stride,)](source.to("mps"), out, n, trips, STRIDE=stride)
    torch.mps.synchronize()
    expected = torch.full_like(source, -99)
    for trip in range(trips):
        start = trip * stride
        expected[start:start + stride] = trip - 9
        expected[start:start + n - trip] = source[start:start + n - trip]
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_masked_scalar_load_preserves_float_bits(dtype_name, num_warps):
    dtype = getattr(torch, dtype_name)
    int_dtype = torch.int32 if dtype_name == "float32" else torch.int16
    if dtype_name == "float32":
        patterns = [0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7f800001, 0xffc12345]
    elif dtype_name == "float16":
        patterns = [0x8000, 1, 0x8001, 0x7c00, 0xfc00, 0x7c01, 0xfe15]
    else:
        patterns = [0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7f81, 0xffc5]
    bits = torch.tensor(patterns, dtype=torch.int64).to(int_dtype)
    n = len(patterns)
    # Read the special values through both branches. Integer uploads/downloads
    # keep any signaling NaN intact so an unintended float conversion is visible.
    fallback_bits = bits.repeat(2)
    source = bits.to("mps").view(dtype)
    fallback = fallback_bits.to("mps").view(dtype)
    out = torch.empty(2 * n, dtype=int_dtype, device="mps")
    _masked_scalar_load_kernel[(2 * n,)](
        source, fallback, out.view(dtype), n, 0, MODE="loaded", num_warps=num_warps,
    )
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), fallback_bits)
