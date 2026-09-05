"""Smoke tests and the report-only P4 Metal performance baseline.

Covers the runtime acceptance criteria for the Metal `_MPSFlooredEvent` timer,
no-op cache primitives, `get_benchmarker()` wrapper, and a tiny
`@triton.testing.perf_report` sweep. See the implementation notes.
"""

import argparse
import csv
from dataclasses import dataclass
import gc
import importlib.util
import json
import os
import pathlib
import platform
import statistics
import subprocess
import time
from collections.abc import Callable

import pytest
import torch

import triton
import triton.language as tl
import triton.testing

# Import the MPS timer + floor directly so the floor / error-swallow behavior
# can be unit-tested without round-tripping through the driver (the driver
# path is exercised separately below).
from triton.backends.metal.driver import _MPSFlooredEvent, _MPS_ELAPSED_TIME_FLOOR_MS


P4_WARMUP = 5
P4_SAMPLES = 7
P4_LAUNCHES_PER_SAMPLE = 20
P4_WORKLOAD_NAMES = (
    "vector_add",
    "fused_softmax",
    "matmul",
    "group_norm",
)


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
    assert calls, (
        "_MPSFlooredEvent.synchronize() must call the device-wide "
        "torch.mps.synchronize(), not torch.mps.Event.synchronize()"
    )

    calls.clear()
    a.elapsed_time(b)
    assert calls, (
        "_MPSFlooredEvent.elapsed_time() must force a device-wide torch.mps.synchronize() before reading timestamps"
    )


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


@triton.jit
def _p4_softmax_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    values = tl.load(x_ptr + row * N + cols)
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * N + cols, numerator / denominator)


@triton.jit
def _p4_matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc)


@dataclass(frozen=True)
class _P4Workload:
    name: str
    shape: tuple[int, ...]
    dtype: str
    num_warps: tuple[int, ...]
    operation: Callable[[], torch.Tensor]
    validate: Callable[[], None]


def _make_p4_vector_add() -> _P4Workload:
    n = 2**22
    torch.manual_seed(0)
    x = torch.randn(n, dtype=torch.float32, device="mps")
    y = torch.randn(n, dtype=torch.float32, device="mps")
    out = torch.empty_like(x)

    def operation():
        _add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK_SIZE=1024, num_warps=4)
        return out

    def validate():
        torch.testing.assert_close(out, x + y, atol=1e-5, rtol=1e-5)

    return _P4Workload("vector_add", (n,), "float32", (4,), operation, validate)


def _make_p4_fused_softmax() -> _P4Workload:
    shape = (4096, 1024)
    torch.manual_seed(1)
    x = torch.randn(shape, dtype=torch.float32, device="mps")
    out = torch.empty_like(x)

    def operation():
        _p4_softmax_kernel[(shape[0],)](x, out, N=shape[1], BLOCK_N=shape[1], num_warps=4)
        return out

    def validate():
        torch.testing.assert_close(out, torch.softmax(x, dim=1), atol=2e-5, rtol=2e-5)

    return _P4Workload("fused_softmax", shape, "float32", (4,), operation, validate)


def _make_p4_matmul(
    *,
    shape: tuple[int, int, int] = (512, 512, 64),
    block: tuple[int, int, int] = (8, 8, 8),
    num_warps: int = 4,
    expected_threads_per_group: int | None = None,
) -> _P4Workload:
    # Eight K tiles is the largest canonical loop currently covered by the
    # Metal simdgroup-matrix lowering; keep this baseline inside that envelope.
    block_m, block_n, block_k = block
    if any(value <= 0 for value in (*shape, *block)):
        raise ValueError("P4 matmul shapes and blocks must be positive")
    if shape[0] % block_m or shape[1] % block_n or shape[2] % block_k:
        raise ValueError("P4 matmul shape must be divisible by its block shape")
    if shape[2] // block_k > 8:
        raise ValueError("P4 matmul supports at most eight K tiles")
    torch.manual_seed(2)
    a = torch.randn((shape[0], shape[2]), dtype=torch.float16, device="mps")
    b = torch.randn((shape[2], shape[1]), dtype=torch.float16, device="mps")
    out = torch.empty((shape[0], shape[1]), dtype=torch.float32, device="mps")
    compiled_kernel = None

    def operation():
        nonlocal compiled_kernel
        compiled_kernel = _p4_matmul_kernel[(shape[0] // block_m, shape[1] // block_n)](
            a,
            b,
            out,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            K_TILES=shape[2] // block_k,
            num_warps=num_warps,
        )
        return out

    def validate():
        torch.testing.assert_close(out, a.float() @ b.float(), atol=0.25, rtol=2e-2)
        if expected_threads_per_group is not None:
            assert compiled_kernel is not None
            metadata = compiled_kernel.metadata
            actual_threads_per_group = getattr(
                metadata,
                "threads_per_group",
                metadata.num_warps * 32,
            )
            assert actual_threads_per_group == expected_threads_per_group

    return _P4Workload("matmul", shape, "float16", (num_warps,), operation, validate)


def _load_p4_group_norm_module():
    source = pathlib.Path(__file__).resolve().parent / "fixtures" / "metal_leet" / "medium-group-normalization.py"
    spec = importlib.util.spec_from_file_location("p4_group_norm_fixture", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import GroupNorm fixture: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_p4_group_norm() -> _P4Workload:
    shape = (8, 512, 64, 64)
    groups = 32
    eps = 1e-5
    module = _load_p4_group_norm_module()
    torch.manual_seed(3)
    x = torch.randn(shape, dtype=torch.float32, device="mps")
    gamma = torch.randn(shape[1], dtype=torch.float32, device="mps")
    beta = torch.randn(shape[1], dtype=torch.float32, device="mps")
    out = torch.empty_like(x)

    def operation():
        module.solve(
            x,
            gamma,
            beta,
            out,
            *shape,
            groups,
            eps,
        )
        return out

    def validate():
        reference = torch.nn.functional.group_norm(x, groups, gamma, beta, eps)
        torch.testing.assert_close(out, reference, atol=5e-4, rtol=5e-4)

    return _P4Workload("group_norm", shape, "float32", (4, 32), operation, validate)


def _measure_p4_workload(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    samples: int,
    launches_per_sample: int,
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.mps.synchronize()

    timings = []
    for _ in range(samples):
        torch.mps.synchronize()
        started = time.perf_counter_ns()
        for _ in range(launches_per_sample):
            result = operation()
        torch.mps.synchronize()
        timings.append((time.perf_counter_ns() - started) / 1e6 / launches_per_sample)
        del result
    return timings


def _p4_gpu_family() -> str:
    try:
        completed = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
        )
        displays = json.loads(completed.stdout).get("SPDisplaysDataType", [])
        if displays:
            return displays[0].get("sppci_model") or displays[0].get("_name", "unknown")
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return str(triton.runtime.driver.active.get_current_target())


def _p4_git_metadata() -> tuple[str, bool]:
    root = pathlib.Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return commit, dirty


def _validate_p4_report(report: dict, *, rounds: int) -> None:
    config = report["benchmark_config"]
    assert config == {
        "warmup": P4_WARMUP,
        "samples": P4_SAMPLES,
        "launches_per_sample": P4_LAUNCHES_PER_SAMPLE,
        "rounds": rounds,
    }
    metadata = report["metadata"]
    for key in ("commit", "gpu_family", "macos", "pytorch", "triton_target"):
        assert metadata[key]
    workloads = report["workloads"]
    assert tuple(workload["name"] for workload in workloads) == P4_WORKLOAD_NAMES
    for workload in workloads:
        assert workload["shape"]
        assert workload["dtype"]
        assert workload["num_warps"]
        assert workload["compile_and_first_launch_ms"] > 0.0
        assert len(workload["rounds"]) == rounds
        for round_result in workload["rounds"]:
            sample_ms = round_result["samples_ms"]
            assert len(sample_ms) == P4_SAMPLES
            assert all(value > 0.0 for value in sample_ms)
            assert round_result["median_ms"] == statistics.median(sample_ms)


def _write_p4_report(report: dict, output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metal-p4-baseline.json"
    csv_path = output_dir / "metal-p4-baseline.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    fieldnames = [
        "commit",
        "dirty",
        "gpu_family",
        "macos",
        "pytorch",
        "triton_target",
        "workload",
        "shape",
        "dtype",
        "num_warps",
        "compile_and_first_launch_ms",
        "round",
        "median_ms",
        "samples_ms",
        "warmup",
        "samples",
        "launches_per_sample",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        metadata = report["metadata"]
        config = report["benchmark_config"]
        for workload in report["workloads"]:
            for round_result in workload["rounds"]:
                writer.writerow(
                    {
                        **metadata,
                        "workload": workload["name"],
                        "shape": "x".join(map(str, workload["shape"])),
                        "dtype": workload["dtype"],
                        "num_warps": "/".join(map(str, workload["num_warps"])),
                        "compile_and_first_launch_ms": workload["compile_and_first_launch_ms"],
                        "round": round_result["round"],
                        "median_ms": round_result["median_ms"],
                        "samples_ms": json.dumps(round_result["samples_ms"]),
                        "warmup": config["warmup"],
                        "samples": config["samples"],
                        "launches_per_sample": config["launches_per_sample"],
                    }
                )
    return json_path, csv_path


def _compare_p4_reports(
    candidate: dict,
    reference: dict,
    *,
    primary_workload: str,
    min_speedup: float,
    max_canary_regression: float,
    allow_primary_num_warps_change: bool = False,
) -> dict:
    candidate_rounds = candidate["benchmark_config"]["rounds"]
    reference_rounds = reference["benchmark_config"]["rounds"]
    _validate_p4_report(candidate, rounds=candidate_rounds)
    _validate_p4_report(reference, rounds=reference_rounds)
    if candidate_rounds != reference_rounds:
        raise ValueError("candidate and reference must use the same round count")
    if primary_workload not in P4_WORKLOAD_NAMES:
        raise ValueError(f"unknown primary workload: {primary_workload}")
    if min_speedup <= 1.0:
        raise ValueError("minimum speedup must be greater than 1")
    if max_canary_regression < 0.0:
        raise ValueError("maximum canary regression must be nonnegative")

    candidate_by_name = {workload["name"]: workload for workload in candidate["workloads"]}
    reference_by_name = {workload["name"]: workload for workload in reference["workloads"]}
    workload_results = []
    violations = []
    for name in P4_WORKLOAD_NAMES:
        current = candidate_by_name[name]
        baseline = reference_by_name[name]
        for field in ("shape", "dtype", "num_warps"):
            if field == "num_warps" and name == primary_workload and allow_primary_num_warps_change:
                continue
            current_value = current[field]
            baseline_value = baseline[field]
            if field in ("shape", "num_warps"):
                current_value = tuple(current_value)
                baseline_value = tuple(baseline_value)
            if current_value != baseline_value:
                raise ValueError(
                    f"{name} {field} differs from the P4.0 baseline: {current[field]} != {baseline[field]}"
                )

        round_results = []
        for current_round, baseline_round in zip(current["rounds"], baseline["rounds"], strict=True):
            if current_round["round"] != baseline_round["round"]:
                raise ValueError(f"{name} round numbering differs from baseline")
            current_ms = current_round["median_ms"]
            baseline_ms = baseline_round["median_ms"]
            speedup = baseline_ms / current_ms
            regression = current_ms / baseline_ms - 1.0
            passed = speedup >= min_speedup if name == primary_workload else regression <= max_canary_regression
            round_results.append(
                {
                    "round": current_round["round"],
                    "reference_median_ms": baseline_ms,
                    "candidate_median_ms": current_ms,
                    "speedup": speedup,
                    "regression": regression,
                    "passed": passed,
                }
            )
        current_median_ms = statistics.median(round_result["median_ms"] for round_result in current["rounds"])
        baseline_median_ms = statistics.median(round_result["median_ms"] for round_result in baseline["rounds"])
        median_speedup = baseline_median_ms / current_median_ms
        median_regression = current_median_ms / baseline_median_ms - 1.0
        if name == primary_workload:
            median_passed = median_speedup >= min_speedup
            if not median_passed:
                violations.append(f"{name} median: {median_speedup:.4f}x < {min_speedup:.4f}x")
        else:
            median_passed = median_regression <= max_canary_regression
            if not median_passed:
                violations.append(f"{name} median: {median_regression:.2%} > {max_canary_regression:.2%} regression")
        workload_results.append(
            {
                "name": name,
                "primary": name == primary_workload,
                "median": {
                    "reference_ms": baseline_median_ms,
                    "candidate_ms": current_median_ms,
                    "speedup": median_speedup,
                    "regression": median_regression,
                    "passed": median_passed,
                },
                "rounds": round_results,
            }
        )

    return {
        "primary_workload": primary_workload,
        "min_speedup": min_speedup,
        "max_canary_regression": max_canary_regression,
        "allow_primary_num_warps_change": allow_primary_num_warps_change,
        "passed": not violations,
        "violations": violations,
        "workloads": workload_results,
    }


def _run_p4_baseline(
    output_dir: pathlib.Path,
    *,
    rounds: int,
    matmul_shape: tuple[int, int, int] = (512, 512, 64),
    matmul_block: tuple[int, int, int] = (8, 8, 8),
    matmul_num_warps: int = 4,
    matmul_expected_threads_per_group: int | None = None,
) -> dict:
    if not torch.backends.mps.is_available():
        raise RuntimeError("P4 Metal baseline requires MPS")
    if rounds < 1:
        raise ValueError("rounds must be positive")

    workload_factories = (
        _make_p4_vector_add,
        _make_p4_fused_softmax,
        lambda: _make_p4_matmul(
            shape=matmul_shape,
            block=matmul_block,
            num_warps=matmul_num_warps,
            expected_threads_per_group=matmul_expected_threads_per_group,
        ),
        _make_p4_group_norm,
    )
    workloads = []
    for factory in workload_factories:
        workload = factory()
        started = time.perf_counter_ns()
        workload.operation()
        torch.mps.synchronize()
        compile_and_first_launch_ms = (time.perf_counter_ns() - started) / 1e6
        workload.validate()
        torch.mps.synchronize()
        workloads.append(
            {
                "workload": workload,
                "compile_and_first_launch_ms": compile_and_first_launch_ms,
                "rounds": [],
            }
        )

    for round_index in range(rounds):
        gc.collect()
        torch.mps.empty_cache()
        ordered = workloads if round_index % 2 == 0 else reversed(workloads)
        for entry in ordered:
            sample_ms = _measure_p4_workload(
                entry["workload"].operation,
                warmup=P4_WARMUP,
                samples=P4_SAMPLES,
                launches_per_sample=P4_LAUNCHES_PER_SAMPLE,
            )
            entry["rounds"].append(
                {
                    "round": round_index + 1,
                    "median_ms": statistics.median(sample_ms),
                    "samples_ms": sample_ms,
                }
            )

    commit, dirty = _p4_git_metadata()
    target = triton.runtime.driver.active.get_current_target()
    report = {
        "schema_version": 1,
        "metadata": {
            "commit": commit,
            "dirty": dirty,
            "gpu_family": _p4_gpu_family(),
            "macos": platform.mac_ver()[0],
            "pytorch": torch.__version__,
            "triton_target": str(target),
        },
        "benchmark_config": {
            "warmup": P4_WARMUP,
            "samples": P4_SAMPLES,
            "launches_per_sample": P4_LAUNCHES_PER_SAMPLE,
            "rounds": rounds,
        },
        "workloads": [
            {
                "name": entry["workload"].name,
                "shape": entry["workload"].shape,
                "dtype": entry["workload"].dtype,
                "num_warps": entry["workload"].num_warps,
                "compile_and_first_launch_ms": entry["compile_and_first_launch_ms"],
                "rounds": entry["rounds"],
            }
            for entry in workloads
        ],
    }
    _validate_p4_report(report, rounds=rounds)
    json_path, csv_path = _write_p4_report(report, output_dir)
    print(json.dumps(report, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return report


def test_p4_report_contract_round_trip(tmp_path):
    rounds = 3
    report = {
        "schema_version": 1,
        "metadata": {
            "commit": "0" * 40,
            "dirty": False,
            "gpu_family": "test-gpu",
            "macos": "test-macos",
            "pytorch": "test-pytorch",
            "triton_target": "test-target",
        },
        "benchmark_config": {
            "warmup": P4_WARMUP,
            "samples": P4_SAMPLES,
            "launches_per_sample": P4_LAUNCHES_PER_SAMPLE,
            "rounds": rounds,
        },
        "workloads": [
            {
                "name": name,
                "shape": (index + 1, index + 2),
                "dtype": "float32",
                "num_warps": (4,),
                "compile_and_first_launch_ms": float(index + 1),
                "rounds": [
                    {
                        "round": round_index + 1,
                        "median_ms": float(index + round_index + 4),
                        "samples_ms": [
                            float(index + round_index + sample_index + 1) for sample_index in range(P4_SAMPLES)
                        ],
                    }
                    for round_index in range(rounds)
                ],
            }
            for index, name in enumerate(P4_WORKLOAD_NAMES)
        ],
    }

    _validate_p4_report(report, rounds=rounds)
    json_path, csv_path = _write_p4_report(report, tmp_path)

    persisted = json.loads(json_path.read_text())
    _validate_p4_report(persisted, rounds=rounds)
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == len(P4_WORKLOAD_NAMES) * rounds
    assert {row["workload"] for row in rows} == set(P4_WORKLOAD_NAMES)

    candidate = json.loads(json.dumps(report))
    for workload in candidate["workloads"]:
        factor = 0.8 if workload["name"] == "vector_add" else 1.04
        for round_result in workload["rounds"]:
            candidate_ms = round_result["median_ms"] * factor
            round_result["median_ms"] = candidate_ms
            round_result["samples_ms"] = [candidate_ms] * P4_SAMPLES
    comparison = _compare_p4_reports(
        candidate,
        report,
        primary_workload="vector_add",
        min_speedup=1.2,
        max_canary_regression=0.05,
    )
    assert comparison["passed"]

    failing = json.loads(json.dumps(candidate))
    for failed_round, baseline_round in zip(
        failing["workloads"][0]["rounds"], report["workloads"][0]["rounds"], strict=True
    ):
        failed_round["median_ms"] = baseline_round["median_ms"] / 1.1
        failed_round["samples_ms"] = [failed_round["median_ms"]] * P4_SAMPLES
    comparison = _compare_p4_reports(
        failing,
        report,
        primary_workload="vector_add",
        min_speedup=1.2,
        max_canary_regression=0.05,
    )
    assert not comparison["passed"]
    assert comparison["violations"] == ["vector_add median: 1.1000x < 1.2000x"]

    different_primary_geometry = json.loads(json.dumps(candidate))
    different_primary_geometry["workloads"][0]["num_warps"] = [16]
    comparison = _compare_p4_reports(
        different_primary_geometry,
        report,
        primary_workload="vector_add",
        min_speedup=1.2,
        max_canary_regression=0.05,
        allow_primary_num_warps_change=True,
    )
    assert comparison["passed"]

    noisy_canary = json.loads(json.dumps(candidate))
    noisy_round = noisy_canary["workloads"][1]["rounds"][0]
    baseline_ms = report["workloads"][1]["rounds"][0]["median_ms"]
    noisy_round["median_ms"] = baseline_ms * 1.20
    noisy_round["samples_ms"] = [noisy_round["median_ms"]] * P4_SAMPLES
    comparison = _compare_p4_reports(
        noisy_canary,
        report,
        primary_workload="vector_add",
        min_speedup=1.2,
        max_canary_regression=0.05,
    )
    assert comparison["passed"]


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
        ms, lo_ms, hi_ms = triton.testing.do_bench(lambda: _add(x, y), quantiles=[0.5, 0.2, 0.8])
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
            warmup=25,
            rep=100,
            quantiles=[0.5, 0.2, 0.8],
        )
        medians.append(median)
    assert all(m > 0.0 for m in medians), f"non-positive medians: {medians}"
    assert medians[1] > medians[0], f"non-monotonic medians: {medians}"
    assert medians[1] < 10.0, f"sanity ceiling: {medians[1]} ms exceeds 10 ms for size=2**22"


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
    def _tuned_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(
            out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask), mask=mask
        )

    x = torch.rand(4096, device="cpu")
    y = torch.rand(4096, device="cpu")
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    _tuned_add[grid](x, y, out, x.numel())  # forces autotune sweep
    torch.testing.assert_close(out, x + y, atol=1e-5, rtol=1e-5)


def _parse_p4_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the report-only Triton Metal P4 baseline.")
    parser.add_argument(
        "--p4-report-dir",
        type=pathlib.Path,
        required=True,
        help="Directory for metal-p4-baseline.json and .csv",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--p4-reference-json", type=pathlib.Path)
    parser.add_argument("--p4-primary-workload", choices=P4_WORKLOAD_NAMES)
    parser.add_argument("--p4-min-speedup", type=float)
    parser.add_argument("--p4-max-canary-regression", type=float)
    parser.add_argument(
        "--p4-matmul-shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        default=(512, 512, 64),
    )
    parser.add_argument(
        "--p4-matmul-block",
        type=int,
        nargs=3,
        metavar=("BLOCK_M", "BLOCK_N", "BLOCK_K"),
        default=(8, 8, 8),
    )
    parser.add_argument("--p4-matmul-num-warps", type=int, default=4)
    parser.add_argument("--p4-matmul-expected-threads-per-group", type=int)
    parser.add_argument("--p4-allow-primary-num-warps-change", action="store_true")
    args = parser.parse_args()
    comparison_args = (
        args.p4_reference_json,
        args.p4_primary_workload,
        args.p4_min_speedup,
        args.p4_max_canary_regression,
    )
    if any(value is not None for value in comparison_args) and not all(value is not None for value in comparison_args):
        parser.error("all P4 comparison options must be supplied together")
    return args


if __name__ == "__main__":
    args = _parse_p4_args()
    report = _run_p4_baseline(
        args.p4_report_dir,
        rounds=args.rounds,
        matmul_shape=tuple(args.p4_matmul_shape),
        matmul_block=tuple(args.p4_matmul_block),
        matmul_num_warps=args.p4_matmul_num_warps,
        matmul_expected_threads_per_group=args.p4_matmul_expected_threads_per_group,
    )
    if args.p4_reference_json is not None:
        reference = json.loads(args.p4_reference_json.read_text())
        comparison = _compare_p4_reports(
            report,
            reference,
            primary_workload=args.p4_primary_workload,
            min_speedup=args.p4_min_speedup,
            max_canary_regression=args.p4_max_canary_regression,
            allow_primary_num_warps_change=args.p4_allow_primary_num_warps_change,
        )
        comparison["reference_json"] = str(args.p4_reference_json)
        report["comparison"] = comparison
        _write_p4_report(report, args.p4_report_dir)
        print(json.dumps(comparison, indent=2))
        if not comparison["passed"]:
            raise RuntimeError("P4 performance gate failed: " + "; ".join(comparison["violations"]))
