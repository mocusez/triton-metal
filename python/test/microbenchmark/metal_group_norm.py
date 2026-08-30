#!/usr/bin/env python3
"""Steady-state MPS benchmark for the Leet Triton GroupNorm kernel."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "unit"
    / "fixtures"
    / "metal_leet"
    / "medium-group-normalization.py"
)
SHAPE = (8, 512, 64, 64)
GROUPS = 32
EPS = 1e-5


def _load_group_norm_module():
    spec = importlib.util.spec_from_file_location(
        "leet_medium_group_normalization", SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import GroupNorm fixture: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure(operation, warmup: int, iterations: int, repeats: int):
    for _ in range(warmup):
        operation()
    torch.mps.synchronize()

    samples = []
    for _ in range(repeats):
        torch.mps.synchronize()
        started = time.perf_counter_ns()
        for _ in range(iterations):
            result = operation()
        torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6 / iterations)
        del result
    return samples


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--num-warps",
        type=int,
        help="Call the JIT kernel directly with this num_warps value.",
    )
    parser.add_argument(
        "--assert-max-ratio",
        type=float,
        help="Fail if any round has Triton/PyTorch median above this ratio.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if min(args.warmup, args.iterations, args.repeats, args.rounds) < 1:
        raise ValueError("warmup, iterations, repeats, and rounds must be positive")

    module = _load_group_norm_module()
    torch.manual_seed(0)
    n, c, h, w = SHAPE
    x = torch.randn(SHAPE, dtype=torch.float32, device="mps")
    gamma = torch.randn(c, dtype=torch.float32, device="mps")
    beta = torch.randn(c, dtype=torch.float32, device="mps")
    triton_out = torch.empty_like(x)

    if args.num_warps is None:

        def triton_group_norm():
            module.solve(
                x, gamma, beta, triton_out, n, c, h, w, GROUPS, EPS
            )
            return triton_out

    else:

        def triton_group_norm():
            module._group_norm_fwd_kernel[(n * GROUPS,)](
                x,
                gamma,
                beta,
                triton_out,
                GROUPS,
                EPS,
                Cg=c // GROUPS,
                HW=h * w,
                BLOCK=4096,
                BLOCK_CH=4096,
                num_warps=args.num_warps,
            )
            return triton_out

    def pytorch_group_norm():
        return F.group_norm(x, GROUPS, gamma, beta, EPS)

    triton_group_norm()
    reference = pytorch_group_norm()
    torch.mps.synchronize()
    max_abs = (triton_out - reference).abs().max().item()
    if not torch.allclose(triton_out, reference, atol=5e-4, rtol=5e-4):
        raise AssertionError(f"GroupNorm mismatch: max_abs={max_abs}")

    results = []
    for round_index in range(args.rounds):
        gc.collect()
        torch.mps.empty_cache()
        order = (
            ("triton", triton_group_norm),
            ("pytorch", pytorch_group_norm),
        )
        if round_index % 2:
            order = tuple(reversed(order))

        samples_by_name = {}
        for name, operation in order:
            samples_by_name[name] = _measure(
                operation, args.warmup, args.iterations, args.repeats
            )

        triton_samples = samples_by_name["triton"]
        pytorch_samples = samples_by_name["pytorch"]
        triton_median = statistics.median(triton_samples)
        pytorch_median = statistics.median(pytorch_samples)
        results.append(
            {
                "round": round_index + 1,
                "triton_median_ms": triton_median,
                "pytorch_median_ms": pytorch_median,
                "ratio": triton_median / pytorch_median,
                "triton_samples_ms": triton_samples,
                "pytorch_samples_ms": pytorch_samples,
            }
        )

    report = {
        "shape": SHAPE,
        "groups": GROUPS,
        "dtype": "float32",
        "num_warps": args.num_warps,
        "max_abs": max_abs,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "results": results,
    }
    print(json.dumps(report, indent=2))

    if args.assert_max_ratio is not None:
        failed = [r for r in results if r["ratio"] > args.assert_max_ratio]
        if failed:
            ratios = ", ".join(f'{r["ratio"]:.4f}' for r in failed)
            raise SystemExit(
                f"FAIL: Triton/PyTorch ratios {ratios} exceed "
                f"{args.assert_max_ratio:.4f}"
            )
        print(
            f"PASS: every Triton/PyTorch median ratio <= "
            f"{args.assert_max_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
