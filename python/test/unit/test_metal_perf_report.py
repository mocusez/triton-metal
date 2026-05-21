"""Smoke tests for the Metal-backend triton.testing perf_report path.

Covers the runtime acceptance criteria for the Metal `_PerfCounterEvent`,
no-op cache primitives, `get_benchmarker()` wrapper, and a tiny
`@triton.testing.perf_report` sweep. See `.omc/plans/metal-perf-report.md`.
"""

import os
import pathlib
import platform
import time

import pytest
import torch

import triton
import triton.language as tl
import triton.testing

# Import _PerfCounterEvent directly so the ns->ms math and enable_timing /
# record guards can be unit-tested without round-tripping through the driver
# (the driver path is exercised separately below).
from triton.backends.metal.driver import _PerfCounterEvent


pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64"),
    reason="Metal backend tests run only on Apple Silicon.",
)


# --- _PerfCounterEvent unit tests -------------------------------------------------


def test_perf_counter_event_elapsed_ms_positive():
    # Pure ns-math test of _PerfCounterEvent.elapsed_time arithmetic. Kernel-
    # driven coverage lives in test_gpu_timer_* below; keeping this test
    # source-independent means the ns->ms math is verified without depending
    # on a real Metal runtime.
    a = _PerfCounterEvent(enable_timing=True)
    b = _PerfCounterEvent(enable_timing=True)
    a._t_ns = 0
    b._t_ns = 200_000  # 200 microseconds
    dt = a.elapsed_time(b)
    assert dt > 0.0
    # 200us -> 0.2 ms; generous outer bounds.
    assert 0.1 < dt < 100.0


def test_perf_counter_event_requires_enable_timing():
    a = _PerfCounterEvent(enable_timing=False)
    b = _PerfCounterEvent(enable_timing=False)
    a.record()
    b.record()
    with pytest.raises(RuntimeError):
        a.elapsed_time(b)


def test_perf_counter_event_requires_record():
    a = _PerfCounterEvent(enable_timing=True)
    b = _PerfCounterEvent(enable_timing=True)
    with pytest.raises(RuntimeError):
        a.elapsed_time(b)


def test_perf_counter_event_clamps_zero_to_floor():
    from triton.backends.metal.driver import _ELAPSED_TIME_FLOOR_MS

    a = _PerfCounterEvent(enable_timing=True)
    b = _PerfCounterEvent(enable_timing=True)
    a.record()
    # Force identical timestamps to exercise the floor without relying on
    # counter resolution. Pin to the exact constant so a future change to
    # the floor value or the clamp is caught here.
    b._t_ns = a._t_ns
    assert a.elapsed_time(b) == _ELAPSED_TIME_FLOOR_MS


# --- Driver-level cache primitives -----------------------------------------------


def test_cache_primitives_are_noop():
    drv = triton.runtime.driver.active
    cache = drv.get_empty_cache_for_benchmark()
    assert cache is None
    for _ in range(1000):
        drv.clear_cache(cache)  # must not raise


# --- triton.testing.do_bench round-trip ------------------------------------------


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


def test_do_bench_returns_positive_quantiles():
    x = torch.rand(4096, device="cpu")
    y = torch.rand(4096, device="cpu")
    # Warm the kernel cache once so do_bench's estimate phase doesn't pay
    # one-shot compilation cost (which would skew the very first sample).
    _add(x, y)
    median, lo, hi = triton.testing.do_bench(
        lambda: _add(x, y),
        warmup=25,
        rep=100,
        quantiles=[0.5, 0.2, 0.8],
    )
    assert lo > 0.0 and median > 0.0 and hi > 0.0
    assert lo <= median <= hi


def test_get_benchmarker_callable():
    bench = triton.runtime.driver.active.get_benchmarker()
    x = torch.rand(1024, device="cpu")
    y = torch.rand(1024, device="cpu")
    _add(x, y)
    result = bench(lambda: _add(x, y), quantiles=[0.5, 0.2, 0.8])
    assert len(result) == 3
    assert all(v > 0.0 for v in result)


# --- @perf_report end-to-end smoke -----------------------------------------------


def test_perf_report_smoke(tmp_path):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["size"],
            x_vals=[1024, 4096],
            line_arg="provider",
            line_vals=["triton"],
            line_names=["Triton"],
            styles=[("blue", "-")],
            ylabel="GB/s",
            plot_name="vector-add-smoke",
            args={},
        ),
    )
    def bench(size, provider):
        x = torch.rand(size, device="cpu")
        y = torch.rand(size, device="cpu")
        _add(x, y)  # warm
        ms, lo_ms, hi_ms = triton.testing.do_bench(
            lambda: _add(x, y), quantiles=[0.5, 0.2, 0.8])
        gbps = lambda t: 3 * x.numel() * x.element_size() * 1e-9 / (t * 1e-3)
        return gbps(ms), gbps(hi_ms), gbps(lo_ms)

    save = str(tmp_path)
    os.makedirs(save, exist_ok=True)
    bench.run(print_data=False, show_plots=False, save_path=save)

    csv_path = pathlib.Path(save) / "vector-add-smoke.csv"
    png_path = pathlib.Path(save) / "vector-add-smoke.png"
    assert csv_path.exists(), f"CSV not produced at {csv_path}"
    assert csv_path.stat().st_size > 0
    assert png_path.exists(), f"PNG not produced at {png_path}"


# --- GPU-side timer behavior (AC#2, AC#3, AC#5 from the v3 plan) ----------------


def test_gpu_timer_smaller_than_cpu_walltime():
    """AC#3 — direction-only: GPU-only elapsed_time must be smaller than
    wallclock-around-the-launcher (which on Metal includes alloc + h2d +
    launch + wait + d2h + free). Direction-only assertion is robust to CI
    noise; the absolute magnitude is captured by AC#2 (cross-size scaling)
    and AC#4 (tutorial CSV delta vs baseline)."""
    x = torch.rand(2**20, device="cpu")
    y = torch.rand(2**20, device="cpu")
    _add(x, y)  # warm JIT + caches
    # CPU walltime around one full launcher call.
    t0 = time.perf_counter_ns()
    _add(x, y)
    t1 = time.perf_counter_ns()
    cpu_ms = (t1 - t0) / 1e6
    # GPU-only via the Event.
    a = _PerfCounterEvent(enable_timing=True)
    b = _PerfCounterEvent(enable_timing=True)
    a.record()
    _add(x, y)
    b.record()
    gpu_ms = a.elapsed_time(b)
    assert gpu_ms > 0.0
    assert gpu_ms < cpu_ms, f"gpu_ms={gpu_ms} not less than cpu_ms={cpu_ms}"


def test_gpu_timer_cross_size_monotonic():
    """GPU-timer median ms must grow with problem size AND stay under a
    sanity ceiling. Sanity-ceiling math: a 2^22 fp32 add moves ~48 MiB
    (2 reads + 1 write x 16 MiB); at ~68 GB/s M1 unified-memory bandwidth
    that is ~0.7 ms, so the 10 ms ceiling gives ~14x headroom — enough to
    absorb thermal throttling and CI contention without flaking."""
    sizes = [2**12, 2**22]
    medians = []
    for n in sizes:
        x = torch.rand(n, device="cpu")
        y = torch.rand(n, device="cpu")
        _add(x, y)  # warm
        median, _lo, _hi = triton.testing.do_bench(
            lambda: _add(x, y),
            warmup=25, rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )
        medians.append(median)
    assert all(m > 0.0 for m in medians), f"non-positive medians: {medians}"
    assert medians[1] > medians[0], f"non-monotonic medians: {medians}"
    assert medians[1] < 10.0, (
        f"sanity ceiling: {medians[1]} ms exceeds 10 ms for size=2**22")


def test_autotune_with_gpu_timer_smoke():
    """Autotune's per-config bench loop must produce non-zero ms with the
    new timer.

    Canary for a regression class: if the accumulator ever returns to a
    destructive read-and-reset contract, an inner do_bench call (autotune
    sweeping configs) would drain the outer accumulator. The monotonic
    counter makes this safe; the assertion is correctness + that autotune
    sweeps run to completion without raising."""

    # num_warps=4 is the Metal driver default (driver.py: `nw = ..., 4`).
    # num_warps=1 yields a 32-thread threadgroup which, paired with
    # BLOCK_SIZE in {128,256}, would hit a loop-unroll path not exercised
    # elsewhere — risk of masking the autotune signal with an unrelated
    # runtime failure.
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _tuned_add(x_ptr, y_ptr, out_ptr, n_elements,
                   BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(out_ptr + offsets,
                 tl.load(x_ptr + offsets, mask=mask)
                 + tl.load(y_ptr + offsets, mask=mask),
                 mask=mask)

    x = torch.rand(4096, device="cpu")
    y = torch.rand(4096, device="cpu")
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    _tuned_add[grid](x, y, out, x.numel())  # forces autotune sweep
    torch.testing.assert_close(out, x + y, atol=1e-5, rtol=1e-5)
