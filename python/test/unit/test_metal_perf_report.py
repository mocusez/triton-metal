"""Smoke tests for the Metal-backend triton.testing perf_report path.

Covers the runtime acceptance criteria for the Metal `_MPSFlooredEvent` timer,
no-op cache primitives, `get_benchmarker()` wrapper, and a tiny
`@triton.testing.perf_report` sweep. See the implementation notes.
"""

import os
import pathlib
import platform

import pytest
import torch

import triton
import triton.language as tl
import triton.testing

# Import the MPS timer + floor directly so the floor / error-swallow behavior
# can be unit-tested without round-tripping through the driver (the driver
# path is exercised separately below).
from triton.backends.metal.driver import _MPSFlooredEvent, _MPS_ELAPSED_TIME_FLOOR_MS


pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64"),
    reason="Metal backend tests run only on Apple Silicon.",
)


# --- _MPSFlooredEvent unit tests -------------------------------------------------


def test_mps_floored_event_clamps_tiny_span_to_floor():
    # Back-to-back record() with no GPU work between yields a sub-floor (or
    # unorderable) span; elapsed_time must report at least the floor so
    # do_bench's n_repeat = rep/estimate_ms doesn't explode.
    #
    # NOTE: this exact sequence used to DEADLOCK (torch 2.10.0) — a per-event
    # synchronize() leaves the event unqueryable and elapsed_time() then blocks
    # forever holding the GIL. _MPSFlooredEvent now forces a device-wide sync.
    # This test only passes/hangs, it cannot fail cleanly, so the fast-failing
    # guard below is the real regression fence.
    a = _MPSFlooredEvent(enable_timing=True)
    b = _MPSFlooredEvent(enable_timing=True)
    a.record()
    b.record()
    a.synchronize()
    b.synchronize()
    assert a.elapsed_time(b) >= _MPS_ELAPSED_TIME_FLOOR_MS


def test_mps_floored_event_forces_device_wide_sync(monkeypatch):
    """Fast-failing fence for the deadlock the test above can only hang on.

    torch 2.10.0's `at::mps::MPSEventPool::elapsedTime` blocks on a condition
    variable forever unless a DEVICE-WIDE `torch.mps.synchronize()` has run —
    the per-event `torch.mps.Event.synchronize()` is not enough, with or without
    real GPU work between the two `record()`s. It blocks while holding the GIL,
    so pytest-timeout / faulthandler / signal.alarm cannot break in: a
    regression wedges the whole run and only an external SIGKILL clears it.

    So assert the device-wide sync directly, with a spy that still delegates to
    the real one. Both calls below are made safe first by an unspied real sync,
    which is what lets a regression FAIL here instead of hanging.
    """
    import torch

    real_sync = torch.mps.synchronize
    calls = []

    def spy():
        calls.append(1)
        real_sync()

    a = _MPSFlooredEvent(enable_timing=True)
    b = _MPSFlooredEvent(enable_timing=True)
    a.record()
    b.record()
    real_sync()  # makes the elapsed_time() below safe even if the fix regressed

    monkeypatch.setattr(torch.mps, "synchronize", spy)

    a.synchronize()
    assert calls, ("_MPSFlooredEvent.synchronize() must call the device-wide "
                   "torch.mps.synchronize(), not torch.mps.Event.synchronize()")

    calls.clear()
    a.elapsed_time(b)
    assert calls, ("_MPSFlooredEvent.elapsed_time() must force a device-wide "
                   "torch.mps.synchronize() before reading timestamps")


def test_mps_floored_event_swallows_runtime_error():
    # torch MPS raises "End event N was not recorded after start event M" for
    # sub-µs spans; the wrapper must swallow it and report the floor rather
    # than crash the whole benchmark.
    class _Raising:
        def elapsed_time(self, _other):
            raise RuntimeError("End event 1 was not recorded after start event 0")

    a = _MPSFlooredEvent(enable_timing=True)
    b = _MPSFlooredEvent(enable_timing=True)
    a._ev = _Raising()  # force the wrapped torch event to raise
    assert a.elapsed_time(b) == _MPS_ELAPSED_TIME_FLOOR_MS


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


# --- GPU-side timer behavior (AC#2, AC#5 from the v3 plan) ----------------
#
# The legacy AC#3 "GPU-only elapsed_time < CPU walltime" test was dropped with
# the native runtime: it was specific to the legacy staging path (the GPU timer
# excluded H2D/D2H memcpy, so it was much smaller than wallclock). On the
# zero-copy MPS path there is no staging and the timer is floored, so that
# direction-only comparison is no longer meaningful.


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
