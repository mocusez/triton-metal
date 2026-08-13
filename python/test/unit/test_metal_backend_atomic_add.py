"""Per-element masked `tl.atomic_add` and histogram output on Metal.

`tt.atomic_rmw fadd` in tensor form is the lock-free accumulation the
layer-norm backward uses in place of the tutorial's global spin lock (Apple
GPUs give no cross-threadgroup forward-progress guarantee, so a `while
atomic_cas(Lock,0,1): pass` spin lock can deadlock). Many programs atomically
add their partial row into a shared buffer; the atomic serializes the colliding
read-modify-writes.

`AtomicRmwLowering`'s tensor branch models it on the masked device store: the
typeconverter scalarizes value/index/mask to the per-thread cone, the E>1 tile
loop replicates the op in place, and the device atomic is guarded by the mask
cone so masked-off lanes never touch a potentially-OOB (and, zero-copy, live)
address. Covers E==1 / E>1, a real `cols<N` mask, and the exact grouped
`_dw[lock_id*N + cols]` accumulation shape of `_layer_norm_bwd_dx_fused`.

The integer cases cover the `tl.histogram` workload: one output lane scans the
logical input tile for its bin, then a masked i32 tensor atomic accumulates the
per-program counts into the final histogram.
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
def _scatter_add_rows(In, Out, N, BLOCK: tl.constexpr):
    # grid=(G,): every program atomically adds its row into the shared Out[cols].
    g = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    v = tl.load(In + g * N + cols, mask=mask, other=0.0)
    tl.atomic_add(Out + cols, v, mask=mask)


# E==1 (BLOCK==tpb at 32), E>1 (BLOCK>tpb at 256/1024), non-pow2 N (masked).
@pytest.mark.parametrize("G, N", [(8, 32), (5, 200), (16, 256), (3, 1000), (7, 1024)])
def test_atomic_add_scatter(G, N):
    torch.manual_seed(G * 100 + N)
    inp = torch.randn(G, N, device="mps")
    out = torch.zeros(N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _scatter_add_rows[(G,)](inp, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    ref = inp.cpu().sum(0)
    # Float-add reorder across programs is non-bit-deterministic; close in fp32.
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)


@triton.jit
def _grouped_accumulate(In, DW, N, GROUP_SIZE_M, BLOCK: tl.constexpr):
    # Exact shape of the layer-norm backward dw accumulation:
    # DW is [GROUP_SIZE_M, N]; row `r` accumulates into bucket `r % GROUP`.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    lock_id = row % GROUP_SIZE_M
    v = tl.load(In + row * N + cols, mask=mask, other=0.0)
    tl.atomic_add(DW + lock_id * N + cols, v, mask=mask)


@pytest.mark.parametrize("M, N, GROUP", [(8, 128, 4), (16, 256, 4), (10, 200, 3), (12, 1024, 8)])
def test_atomic_add_grouped(M, N, GROUP):
    torch.manual_seed(M * 1000 + N + GROUP)
    inp = torch.randn(M, N, device="mps")
    dw = torch.zeros(GROUP, N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _grouped_accumulate[(M,)](inp, dw, N, GROUP, BLOCK=BLOCK)
    torch.mps.synchronize()
    # reference: rows folded by (row % GROUP)
    ref = torch.zeros(GROUP, N)
    ic = inp.cpu()
    for r in range(M):
        ref[r % GROUP] += ic[r]
    torch.testing.assert_close(dw.cpu(), ref, atol=1e-4, rtol=1e-4)


@triton.jit
def _atomic_add_ones(Out, N, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    tl.atomic_add(Out + cols, tl.full([BLOCK], 1.0, tl.float32), mask=mask)


@pytest.mark.parametrize("G, N", [(64, 128), (100, 300)])
def test_atomic_add_ones_exact(G, N):
    # Adding 1.0 G times is exact regardless of order -> bit-exact contention check.
    out = torch.zeros(N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _atomic_add_ones[(G,)](out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    ref = torch.full((N,), float(G))
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)


@triton.jit
def _histogram_atomic(In, Out, N, NUM_BINS, PADDED_BINS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    values = tl.load(In + offsets, mask=valid)
    hist = tl.histogram(values, PADDED_BINS)
    # A masked load without an explicit `other` contributes zero for its
    # inactive lanes. Match the leet kernel by removing those padding zeros.
    hist -= (BLOCK - tl.sum(valid.to(tl.int32))) * (tl.arange(0, PADDED_BINS) == 0)
    bins = tl.arange(0, PADDED_BINS)
    tl.atomic_add(Out + bins, hist, mask=(bins < NUM_BINS) & (hist != 0))


@pytest.mark.parametrize("N, NUM_BINS", [(1024, 16), (1000, 16), (2500, 37)])
def test_histogram_atomic_matches_bincount(N, NUM_BINS):
    torch.manual_seed(N + NUM_BINS)
    inp = torch.randint(0, NUM_BINS, (N,), dtype=torch.int32, device="mps")
    out = torch.zeros(NUM_BINS, dtype=torch.int32, device="mps")
    _histogram_atomic[(triton.cdiv(N, 1024),)](
        inp,
        out,
        N,
        NUM_BINS,
        PADDED_BINS=triton.next_power_of_2(NUM_BINS),
        BLOCK=1024,
    )
    torch.mps.synchronize()
    ref = torch.bincount(inp.cpu().to(torch.int64), minlength=NUM_BINS)
    torch.testing.assert_close(out.cpu().to(torch.int64), ref, atol=0, rtol=0)


@triton.jit
def _masked_histogram_store(In, Out, N, BINS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < N
    values = tl.load(In + offsets, mask=valid)
    hist = tl.histogram(values, BINS, mask=valid)
    tl.store(Out + tl.arange(0, BINS), hist)


@pytest.mark.parametrize("N", [100, 1000])
def test_masked_histogram_matches_bincount(N):
    BINS = 16
    torch.manual_seed(N)
    inp = torch.randint(0, BINS, (N,), dtype=torch.int32, device="mps")
    out = torch.zeros(BINS, dtype=torch.int32, device="mps")
    _masked_histogram_store[(1,)](inp, out, N, BINS=BINS, BLOCK=1024)
    torch.mps.synchronize()
    ref = torch.bincount(inp.cpu().to(torch.int64), minlength=BINS)
    torch.testing.assert_close(out.cpu().to(torch.int64), ref, atol=0, rtol=0)


@triton.jit
def _atomic_max_scalar_f32(In, Out):
    value = tl.load(In + tl.program_id(0))
    tl.atomic_max(Out, value)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 7.0, 3.0, 29.0, 11.0],
        [-91.0, -7.0, -31.0, -2.0, -18.0],
    ],
)
def test_atomic_max_scalar_f32_contended(values):
    inp = torch.tensor(values, dtype=torch.float32, device="mps")
    out = torch.full((1,), -2147483648.0, dtype=torch.float32, device="mps")
    _atomic_max_scalar_f32[(len(values),)](inp, out)
    torch.mps.synchronize()
    assert out.cpu().item() == max(values)


@triton.jit
def _max_subarray_sum_kernel(
    input, output, N, windows_size, length, BLOCK_SIZE: tl.constexpr
):
    # Verbatim kernel body from leet-triton/medium-max_subarray_sum.py.
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < length
    result = tl.zeros([BLOCK_SIZE], tl.float32)
    for i in range(windows_size):
        offs_i = offs + i
        mask_i = offs_i < N
        val = tl.load(input + offs_i, mask_i)
        result += val
    ret = tl.where(mask, result, float("-inf"))
    ret = tl.max(ret)
    tl.atomic_max(output, ret)


@pytest.mark.parametrize(
    "N, window_size",
    [(1, 1), (31, 5), (32, 7), (33, 7), (1000, 37), (1025, 64)],
)
def test_max_subarray_sum_matches_torch(N, window_size):
    torch.manual_seed(N * 100 + window_size)
    inp = torch.randn(N, dtype=torch.float32, device="mps")
    # Pin the all-negative case as well as mixed random inputs.
    if N == 33:
        inp = -inp.abs() - 1.0
    out = torch.empty(1, dtype=torch.float32, device="mps")
    out.fill_(-2147483648)
    length = N - window_size + 1
    BLOCK_SIZE = 32
    _max_subarray_sum_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
        inp, out, N, window_size, length, BLOCK_SIZE
    )
    torch.mps.synchronize()
    expected = inp.cpu().unfold(0, window_size, 1).sum(1).max()
    torch.testing.assert_close(out.cpu()[0], expected, atol=1e-5, rtol=1e-5)
