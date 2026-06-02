"""Kernel-only latency/throughput bench for rank-1 f32 sum reduce.

Measures per-kernel GPU latency (µs) and effective GB/s for BLOCK ∈ {512, 1024}
using triton.testing.do_bench with the Metal _PerfCounterEvent timer.
Saves results to /tmp/optionbeta_rank1_reduce_bench.csv.

## Why GB/s is low for these BLOCK sizes

The Metal backend currently supports rank-1 reduce only for BLOCK ≤ 1024
(BLOCK=2048+ fails at `tt.reduce` lowering). A BLOCK=1024 reduce operates on
1024 × 4B = 4KB of data. Apple Silicon M-series takes ~6–10µs of *kernel-only*
GPU time for such a tiny dispatch. At those durations:

    GB/s = 4KB / 8µs ≈ 0.5 GB/s

This is not a memory-bandwidth number — it is a kernel-launch-latency number.
The GPU's peak DRAM bandwidth (~100–400 GB/s on M1/M2/M3) is irrelevant when
the entire dataset fits in a handful of cache lines and the kernel finishes in
a fraction of a microsecond.

## Timer implementation

do_bench on Metal uses _PerfCounterEvent (driver.py), which accumulates
MTLCommandBuffer.GPUStartTime/GPUEndTime — kernel-only GPU time, excluding H2D/D2H.
_ELAPSED_TIME_FLOOR_MS = 0.1ms clamps each do_bench iteration's elapsed_time.
To escape the floor we batch N_CALLS_PER_ITER=2000 launches per iteration so the
accumulated GPU time (~12ms) >> 0.1ms, then divide by N to recover per-kernel time.

## Tensors

Tensors use device="cpu" (default on macOS). MetalLauncher copies CPU tensors
to MTLBuffer (alloc + copy_h2d) per launch and copies results back (copy_d2h)
after. These transfers are excluded from the GPU timer.

Run:
    pixi run --frozen pytest python/test/unit/test_metal_perf_rank1_reduce.py -s --tb=short
"""

from __future__ import annotations

import csv
import platform

import pytest
import torch

import triton
import triton.language as tl
import triton.testing

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not hasattr(libmetal, "launch_kernel_with_pipeline"):
    pytest.skip(
        "Metal runtime not compiled (non-Darwin build or Xcode CLT absent)",
        allow_module_level=True,
    )

pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64"),
    reason="Metal backend perf tests run only on Apple Silicon.",
)

_NUM_WARPS = 8  # threadgroup = 256 threads

# Batch count for latency measurement. 2000 launches × ~6-10µs GPU time each
# = ~12-20ms total, well above the 0.1ms _ELAPSED_TIME_FLOOR_MS in driver.py.
# Per-kernel time = total_ms / N_CALLS_PER_ITER.
_N_CALLS_PER_ITER = 2000


@triton.jit
def _reduce_sum_rank1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + 0, s)


_BENCH_RESULTS: list[dict] = []


@pytest.mark.parametrize("BLOCK", [512, 1024])
def test_rank1_reduce_sum_bench(BLOCK):
    """Measure kernel-only GPU latency (µs) and effective GB/s.

    For BLOCK ≤ 1024 these kernels are latency-bound (not memory-BW-bound):
    the dataset is 2-4KB, well within L1 cache. Report µs as the primary
    metric; GB/s is included for completeness but reflects dispatch overhead,
    not DRAM bandwidth.

    Expected on Apple Silicon M-series:
      - BLOCK=512:  ~4–10 µs kernel time
      - BLOCK=1024: ~6–15 µs kernel time
      - GB/s: 0.1–1.0 (latency-dominated; NOT comparable to DRAM bandwidth)
    """
    x = torch.randn((BLOCK,), dtype=torch.float32)
    out = torch.zeros((1,), dtype=torch.float32)

    # Warm the JIT cache before benchmarking.
    _reduce_sum_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)

    def _batch_fn():
        for _ in range(_N_CALLS_PER_ITER):
            _reduce_sum_rank1_kernel[(1, 1, 1)](x, out, BLOCK=BLOCK, num_warps=_NUM_WARPS)

    # total_ms = GPU time for N_CALLS_PER_ITER sequential kernel launches.
    # Divide by N to recover per-kernel GPU time.
    total_ms = triton.testing.do_bench(_batch_fn, warmup=10, rep=50)
    per_kernel_us = (total_ms / _N_CALLS_PER_ITER) * 1e3
    bytes_read = BLOCK * 4  # f32 elements read
    gbps = (bytes_read * 1e-9) / (per_kernel_us * 1e-6)

    print(
        f"\nBLOCK={BLOCK}: {per_kernel_us:.2f} µs/kernel  "
        f"({gbps:.3f} GB/s — latency-bound, not DRAM-BW-bound)"
    )

    # Sanity checks: timer must have fired (total_ms > floor*rep) and latency
    # must be in a plausible range for a tiny Metal kernel dispatch.
    assert per_kernel_us > 0.0, "Expected positive kernel latency"
    assert per_kernel_us < 1000.0, (
        f"per_kernel_us={per_kernel_us:.1f} exceeds 1ms — "
        "likely timer or compilation issue"
    )
    # GB/s floor is intentionally low: these are 2-4KB kernels, latency-bound.
    # If you see < 0.01 GB/s the timer accumulator likely returned 0.
    assert gbps > 0.01, (
        f"GB/s={gbps:.6f} is suspiciously near zero — "
        f"check _PerfCounterEvent accumulator (read_gpu_time_ns_total). "
        f"total_ms={total_ms:.4f}, per_kernel_us={per_kernel_us:.4f}"
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
