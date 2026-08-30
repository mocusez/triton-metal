"""Per-launch latency / throughput bench for rank-1 f32 sum reduce (MPS path).

Measures per-launch latency (µs) and effective GB/s for BLOCK ∈ {512, 1024}
using triton.testing.do_bench on PyTorch MPS tensors.
Saves results to /tmp/optionbeta_rank1_reduce_bench.csv.

## What is measured (and why it is latency-bound, not BW-bound)

The Metal backend currently supports rank-1 reduce only for BLOCK ≤ 1024
(BLOCK=2048+ fails at `tt.reduce` lowering). A BLOCK=1024 reduce operates on
1024 × 4B = 4KB of data — entirely cache-resident on Apple Silicon UMA.

On the MPS launch path (the default; see the implementation notes)
do_bench times kernels with `torch.mps.Event` (driver `get_device_interface`),
which captures the *GPU-timeline span* of the dispatched launches. For a tiny
kernel that span is dominated by per-launch dispatch latency (Python JIT runtime
+ MPS dispatch), not GPU compute, which is a fraction of a microsecond. So:

    GB/s = 4KB / (hundreds of µs) ≈ 0.003–0.01

is a *correct, expected* dispatch-bound number — NOT a memory-bandwidth figure
and NOT a timer malfunction. The GPU's peak DRAM bandwidth (~100–400 GB/s on
M1/M2/M3) is irrelevant when the dataset fits in a handful of cache lines.

The primary signal of this bench is therefore the per-launch latency (µs) and
that the timer fires with a positive, finite value; GB/s is informational.

## Timer note (MPS vs legacy)

This bench targets the MPS path, where do_bench times via `_MPSFlooredEvent`
(driver) — a wrapper over `torch.mps.Event` that floors elapsed_time and
swallows torch's flaky sub-µs ordering error. The legacy native GPU-time
accumulator was removed with the native runtime. Batching N launches per
do_bench callable yields a stable per-launch average and amortizes do_bench's
fixed estimate overhead.

## Tensors

Tensors live on `device="mps"`, so the launcher binds them zero-copy into their
own MTLBuffer (no host staging, no per-launch alloc/copy). This is the path the
Metal backend actually uses at runtime.

Run:
    pixi run --frozen pytest python/test/unit/test_metal_perf_rank1_reduce.py -s --tb=short
"""

from __future__ import annotations

import csv
import os
import platform

import pytest
import torch

import triton
import triton.language as tl
import triton.testing

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)

# This bench exercises the MPS launch path (zero-copy dispatch via
# torch.mps.compile_shader). It is meaningless on the legacy metallib runtime
# (removed; native GPU-time counter dormant), so require an MPS-enabled torch
# and skip when the user has opted out with TRITON_METAL_USE_MPS=0.
pytestmark = pytest.mark.skipif(
    not (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and torch.backends.mps.is_available()
        and os.environ.get("TRITON_METAL_USE_MPS", "1") != "0"
    ),
    reason="Metal MPS perf bench runs only on Apple Silicon with an MPS-enabled torch "
    "(and TRITON_METAL_USE_MPS != 0).",
)

_NUM_WARPS = 8  # threadgroup = 256 threads

# Launches per do_bench callable. The per-launch result is N-invariant
# (total / N), so N only trades runtime for averaging stability. Kept modest
# since torch.mps.Event has no elapsed-time floor to escape on this path.
_N_CALLS_PER_ITER = 500

# Generous upper bound on per-launch latency (~7-10x the observed dispatch
# latency). Exceeding it points at a timer/compilation issue, not normal
# Python+MPS dispatch overhead, even under full-suite load.
_MAX_PER_KERNEL_US = 5000.0


@triton.jit
def _reduce_sum_rank1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + 0, s)


_BENCH_RESULTS: list[dict] = []


@pytest.mark.parametrize("BLOCK", [512, 1024])
def test_rank1_reduce_sum_bench(BLOCK):
    """Measure per-launch latency (µs) and effective GB/s on the MPS path.

    For BLOCK ≤ 1024 these kernels are latency-bound (not memory-BW-bound):
    the dataset is 2-4KB, well within cache. Report µs as the primary metric;
    GB/s is informational and reflects dispatch overhead, not DRAM bandwidth.
    """
    x = torch.randn((BLOCK,), dtype=torch.float32, device="mps")
    out = torch.zeros((1,), dtype=torch.float32, device="mps")

    # Warm the JIT cache before benchmarking.
    _reduce_sum_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)

    def _batch_fn():
        for _ in range(_N_CALLS_PER_ITER):
            _reduce_sum_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)

    # total_ms = GPU-timeline span for N_CALLS_PER_ITER sequential launches.
    # Divide by N to recover per-launch latency.
    total_ms = triton.testing.do_bench(_batch_fn, warmup=10, rep=50)
    per_kernel_us = (total_ms / _N_CALLS_PER_ITER) * 1e3
    bytes_read = BLOCK * 4  # f32 elements read
    gbps = (bytes_read * 1e-9) / (per_kernel_us * 1e-6)

    print(
        f"\nBLOCK={BLOCK}: {per_kernel_us:.2f} µs/launch  "
        f"({gbps:.3f} GB/s — dispatch-bound, not DRAM-BW-bound)"
    )

    # The torch.mps.Event timer must have fired (non-zero span) and the
    # per-launch latency must be positive and within a plausible dispatch range.
    # GB/s is intentionally NOT gated: for a 2-4KB latency-bound kernel its value
    # (~0.003-0.01) is a correct dispatch-bound number, not a timer fault.
    assert total_ms > 0.0, "torch.mps.Event timer returned a non-positive span"
    assert per_kernel_us > 0.0, "Expected positive per-launch latency"
    assert per_kernel_us < _MAX_PER_KERNEL_US, (
        f"per_kernel_us={per_kernel_us:.1f} exceeds {_MAX_PER_KERNEL_US:.0f}µs — "
        "likely a timer or compilation issue, not normal dispatch latency. "
        f"total_ms={total_ms:.4f}"
    )

    _BENCH_RESULTS.append({"BLOCK": BLOCK, "ms": per_kernel_us * 1e-3, "gbps": gbps})


def test_save_bench_csv():
    """Write accumulated bench results to /tmp/optionbeta_rank1_reduce_bench.csv."""
    csv_path = "/tmp/optionbeta_rank1_reduce_bench.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["BLOCK", "ms", "gbps"])
        writer.writeheader()
        writer.writerows(_BENCH_RESULTS)
    print(f"\nBench CSV saved to {csv_path} ({len(_BENCH_RESULTS)} rows)")
    assert True  # always passes; bench numbers saved for inspection
