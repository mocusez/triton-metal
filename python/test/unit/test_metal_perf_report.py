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
    a = _PerfCounterEvent(enable_timing=True)
    b = _PerfCounterEvent(enable_timing=True)
    a.record()
    # Busy-spin ~200 microseconds so b - a is comfortably above the 1e-6 ms floor.
    target = a._t_ns + 200_000
    while time.perf_counter_ns() < target:
        pass
    b.record()
    dt = a.elapsed_time(b)
    assert dt > 0.0
    # Generous bounds: at least 0.1 ms, well under 100 ms in a sane environment.
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
