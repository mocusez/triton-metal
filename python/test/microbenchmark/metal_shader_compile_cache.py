#!/usr/bin/env python3
"""Measure source-specific cross-process caching in ``torch.mps.compile_shader``.

Run this through the repository Pixi environment.  Each observation is made in
a fresh Python process: repeated-source samples reuse identical MSL while the
control samples change both the kernel name and source constant.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import uuid

import torch


_CHILD = """
import sys
import time

import torch

source = sys.stdin.read()
torch.mps.synchronize()
started = time.perf_counter_ns()
torch.mps.compile_shader(source)
elapsed_ms = (time.perf_counter_ns() - started) / 1e6
print(f"{elapsed_ms:.6f}")
"""


def _source(kernel_name: str, value: int) -> str:
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void {kernel_name}(
    device float* x [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {{
  x[tid] += {value}.0f;
}}
"""


def _compile_in_fresh_process(source: str) -> float:
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=source,
        capture_output=True,
        check=True,
        text=True,
    )
    return float(completed.stdout.strip())


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--assert-max-warm-ratio",
        type=float,
        help=(
            "Fail if repeated-source median / unique-source median exceeds "
            "this value. The assertion is optional because the cache belongs "
            "to the host PyTorch/Metal stack, not Triton."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    token = uuid.uuid4().hex
    repeated_source = _source(f"cache_probe_{token}", 1)

    # Prime this exact source in one process, then measure only fresh-process
    # loads. Unique-source controls keep changing the source cache key.
    cold_ms = _compile_in_fresh_process(repeated_source)
    repeated_ms = []
    unique_ms = []
    for index in range(args.repeats):
        repeated_ms.append(_compile_in_fresh_process(repeated_source))
        unique_ms.append(
            _compile_in_fresh_process(
                _source(f"cache_probe_{token}_{index}", index + 2)
            )
        )

    repeated_median = statistics.median(repeated_ms)
    unique_median = statistics.median(unique_ms)
    ratio = repeated_median / unique_median
    report = {
        "process_model": "fresh_process_per_sample",
        "cold_same_source_ms": cold_ms,
        "repeated_source_ms": repeated_ms,
        "unique_source_ms": unique_ms,
        "repeated_source_median_ms": repeated_median,
        "unique_source_median_ms": unique_median,
        "repeated_over_unique_ratio": ratio,
    }
    print(json.dumps(report, indent=2))

    if (
        args.assert_max_warm_ratio is not None
        and ratio > args.assert_max_warm_ratio
    ):
        raise SystemExit(
            f"FAIL: repeated/unique median ratio {ratio:.4f} exceeds "
            f"{args.assert_max_warm_ratio:.4f}"
        )
    if args.assert_max_warm_ratio is not None:
        print(
            "PASS: repeated/unique median ratio "
            f"{ratio:.4f} <= {args.assert_max_warm_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
