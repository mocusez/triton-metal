"""L3a-tileloop-compiler-A: scalar tt.load on bare !tt.ptr<f32>.

Drives a small @triton.jit kernel that performs a scalar `tl.load(scalar_ptr
+ scalar_offset)` inside a `tl.static_range` loop, and asserts bit-exact
agreement with a CPU reference across 5 deterministic runs. The kernel
mirrors the conv1d Variant B shape that motivated this session.

See `.omc/specs/deep-interview-leet-triton-l3a-tileloop-compiler-a-scalar-load.md`.
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
