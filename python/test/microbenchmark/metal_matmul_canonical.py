"""What does the canonical matmul cost, now that the canonical matmul compiles?

The perf landscape for this backend has been read twice off kernels that were
not the real thing: once off blocks that were silently skipping three quarters
of their reduction, and before that off an 8x8x8 block chosen because it was
the only correct one. `c7ffa57a14` made the form people actually write -- flat
swizzled grid, pointer advance, masked loads and store, f16 result -- compile
on the matrix unit for the first time. So measure THAT, not a stand-in.

Discipline this project has paid for: warm up, wall-clock a batch with
`torch.mps.synchronize`, never `torch.mps.Event`, five samples per arm, one
cache directory per arm, report medians AND ranges, and check every arm's
numbers in the same process so no timing is quoted for a kernel that is wrong.

Usage:  pixi run --frozen python python/test/microbenchmark/metal_matmul_canonical.py
        Add --recompute to compare masked/plain recompute-from-IV kernels
        with normal and transposed B, emitting one JSON record per arm.
        Add --lora to compare plain/masked fused LoRA and the owned verbatim
        fixture, emitting one JSON record per arm.
        METAL_PERF_SETTLE=0 ... to skip the cool-down when the machine is
        already known idle.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _canonical_matmul(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr,
                      GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BM + tl.arange(0, BM)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_am[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_bn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - kk * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - kk * BK, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    tl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :],
             acc.to(c_ptr.dtype.element_ty),
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


SHAPES = [(512, 512, 64), (1024, 1024, 1024)]
# 128x128 is in here on purpose. It used to kill the Metal compiler service
# outright (XPC_ERROR_CONNECTION_INTERRUPTED); it now compiles at 8 warps and is
# refused at 4, because a tile holds (BM*BN)/64 simdgroup accumulators and the
# canonical tier caps those at 32 PER WARP. So the block-size question and the
# warp-count question are not independent, and an arm list that sweeps them
# separately cannot see it.
ARMS = [(bm, bn, bk, nw, gm)
        for (bm, bn) in ((32, 32), (64, 64), (128, 64), (64, 128), (128, 128))
        for bk in (32, 64)
        for nw in (4, 8)
        for gm in (8,)]

BATCH = 20
SAMPLES = 5


@triton.jit
def _recompute_matmul(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr,
                      TRANS: tl.constexpr, MASKED: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + tl.arange(0, BK)
        if MASKED:
            a = tl.load(a_ptr + m[:, None] * K + kk[None, :],
                        mask=(m[:, None] < M) & (kk[None, :] < K), other=0.0)
        else:
            a = tl.load(a_ptr + m[:, None] * K + kk[None, :])
        if TRANS:
            if MASKED:
                b = tl.load(b_ptr + n[:, None] * K + kk[None, :],
                            mask=(n[:, None] < N) & (kk[None, :] < K), other=0.0)
            else:
                b = tl.load(b_ptr + n[:, None] * K + kk[None, :])
            acc = tl.dot(a, tl.trans(b), acc)
        else:
            if MASKED:
                b = tl.load(b_ptr + kk[:, None] * N + n[None, :],
                            mask=(kk[:, None] < K) & (n[None, :] < N), other=0.0)
            else:
                b = tl.load(b_ptr + kk[:, None] * N + n[None, :])
            acc = tl.dot(a, b, acc)
    tl.store(c_ptr + m[:, None] * N + n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=(m[:, None] < M) & (n[None, :] < N))


def recompute_main():
    torch.manual_seed(21)
    for M, N, K in ((1024, 1024, 1024), (256, 1024, 1024)):
        for trans in (False, True):
            a = torch.randn(M, K, dtype=torch.float16, device="mps")
            b = torch.randn((N, K) if trans else (K, N),
                            dtype=torch.float16, device="mps")
            b_cpu = b.cpu().float()
            ref = a.cpu().float() @ (b_cpu.t() if trans else b_cpu)
            for bm, bn, bk, nw in ((32, 32, 32, 4), (64, 64, 32, 4)):
                for masked in (True, False):
                    out = torch.empty(M, N, dtype=torch.float16, device="mps")

                    def arm():
                        return _recompute_matmul[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
                            a, b, out, M, N, K, BM=bm, BN=bn, BK=bk,
                            TRANS=trans, MASKED=masked, num_warps=nw)

                    compiled = arm()
                    torch.mps.synchronize()
                    torch.testing.assert_close(out.cpu().float(), ref, atol=0.1, rtol=0.01)
                    metal = compiled.asm["metal"]
                    if isinstance(metal, bytes):
                        metal = metal.decode("utf-8")
                    mma = metal.count("simdgroup_multiply_accumulate")
                    assert mma > 0, "recompute benchmark must use SIMD-group matrix multiplication"
                    for _ in range(5):
                        arm()
                    torch.mps.synchronize()
                    samples = []
                    for _ in range(SAMPLES):
                        start = time.perf_counter()
                        for _ in range(BATCH):
                            arm()
                        torch.mps.synchronize()
                        samples.append((time.perf_counter() - start) / BATCH * 1e3)
                    actual = out.cpu().float()
                    torch.testing.assert_close(actual, ref, atol=0.1, rtol=0.01)
                    print(json.dumps({
                        "mode": "recompute", "dtype": "fp16",
                        "shape": [M, N, K], "tile": [bm, bn, bk, nw],
                        "trans": trans, "masked": masked,
                        "median_ms": statistics.median(samples),
                        "min_ms": min(samples), "max_ms": max(samples),
                        "samples_ms": samples, "batch": BATCH, "mma": mma,
                        "max_error": (actual - ref).abs().max().item(),
                    }), flush=True)


@triton.jit
def _fused_lora(x_ptr, w_ptr, a_ptr, b_ptr, out_ptr, M, N, K, R, scale,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                BR: tl.constexpr, MASKED: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    acc0 = tl.zeros((BM, BN), dtype=tl.float32)
    acc1 = tl.zeros((BM, BR), dtype=tl.float32)
    for k in range(0, K, BK):
        kk = k + tl.arange(0, BK)
        if MASKED:
            x = tl.load(x_ptr + m[:, None] * K + kk[None, :],
                        mask=(m[:, None] < M) & (kk[None, :] < K), other=0.0)
            w = tl.load(w_ptr + n[:, None] * K + kk[None, :],
                        mask=(n[:, None] < N) & (kk[None, :] < K), other=0.0)
            a = tl.load(a_ptr + r[:, None] * K + kk[None, :],
                        mask=(r[:, None] < R) & (kk[None, :] < K), other=0.0)
        else:
            x = tl.load(x_ptr + m[:, None] * K + kk[None, :])
            w = tl.load(w_ptr + n[:, None] * K + kk[None, :])
            a = tl.load(a_ptr + r[:, None] * K + kk[None, :])
        acc0 = tl.dot(x, tl.trans(w), acc0, allow_tf32=False)
        acc1 = tl.dot(x, tl.trans(a), acc1, allow_tf32=False)
    if MASKED:
        b = tl.load(b_ptr + n[:, None] * R + r[None, :],
                    mask=(n[:, None] < N) & (r[None, :] < R), other=0.0)
    else:
        b = tl.load(b_ptr + n[:, None] * R + r[None, :])
    acc0 += scale * tl.dot(acc1, tl.trans(b), allow_tf32=False)
    if MASKED:
        tl.store(out_ptr + m[:, None] * N + n[None, :], acc0,
                 mask=(m[:, None] < M) & (n[None, :] < N))
    else:
        tl.store(out_ptr + m[:, None] * N + n[None, :], acc0)


def lora_main():
    fixture = Path(__file__).resolve().parents[1] / "unit/fixtures/metal_leet/medium-lora_linear.py"
    spec = importlib.util.spec_from_file_location("_metal_lora_benchmark_fixture", fixture)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(23)
    for M, N, K, R in ((256, 1024, 1024, 16), (1024, 1024, 1024, 16)):
        cpu = [torch.randn(shape, dtype=torch.float32)
               for shape in ((M, K), (N, K), (R, K), (N, R))]
        x, w, a, b = [tensor.to("mps") for tensor in cpu]
        ref = cpu[0] @ cpu[1].t() + 0.5 * ((cpu[0] @ cpu[2].t()) @ cpu[3].t())
        for bm, bn, br, nw in ((32, 32, 16, 4), (64, 64, 16, 4), (64, 128, 16, 4)):
            for form in ("plain", "masked", "verbatim"):
                out = torch.empty((M, N), dtype=torch.float32, device="mps")
                grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))

                def arm():
                    if form == "verbatim":
                        return module.lora_kernel[(grid[0] * grid[1],)](
                            x, w, a, b, out, M, N, K, R, 0.5,
                            K, 1, K, 1, K, 1, R, 1, N, 1,
                            BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=32,
                            BLOCK_SIZE_R=br, GROUP_SIZE=4, num_warps=nw)
                    return _fused_lora[grid](
                        x, w, a, b, out, M, N, K, R, 0.5,
                        BM=bm, BN=bn, BK=32, BR=br,
                        MASKED=form == "masked", num_warps=nw)

                compiled = arm()
                torch.mps.synchronize()
                torch.testing.assert_close(out.cpu(), ref, atol=0.02, rtol=0.002)
                metal = compiled.asm["metal"]
                if isinstance(metal, bytes):
                    metal = metal.decode("utf-8")
                mma = metal.count("simdgroup_multiply_accumulate")
                assert mma > 0, "LoRA benchmark must use SIMD-group matrix multiplication"
                for _ in range(5):
                    arm()
                torch.mps.synchronize()
                samples = []
                for _ in range(SAMPLES):
                    start = time.perf_counter()
                    for _ in range(BATCH):
                        arm()
                    torch.mps.synchronize()
                    samples.append((time.perf_counter() - start) / BATCH * 1e3)
                actual = out.cpu()
                torch.testing.assert_close(actual, ref, atol=0.02, rtol=0.002)
                print(json.dumps({
                    "mode": "lora", "dtype": "fp32",
                    "shape": [M, N, K, R], "tile": [bm, bn, 32, br, nw],
                    "form": form, "median_ms": statistics.median(samples),
                    "min_ms": min(samples), "max_ms": max(samples),
                    "samples_ms": samples, "batch": BATCH, "mma": mma,
                    "source_bytes": len(metal.encode("utf-8")),
                    "max_error": (actual - ref).abs().max().item(),
                }), flush=True)


def timed(fn, ref, out):
    for _ in range(5):
        fn()
    torch.mps.synchronize()
    samples = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        for _ in range(BATCH):
            fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - t0) / BATCH * 1e3)
    ok = (out.float() - ref).abs().max().item()
    return samples, ok


def main(recompute=False, lora=False):
    # Never measure within a minute of a suite run; this script is usually
    # started right after one. Run-to-run spread within ONE build is ~10% on
    # this machine, which is larger than many changes worth making, and two
    # readings taken on a busy machine have already been quoted wrongly.
    time.sleep(float(os.environ.get("METAL_PERF_SETTLE", "75")))
    if recompute or lora:
        benchmark = lora_main if lora else recompute_main
        if "TRITON_CACHE_DIR" in os.environ:
            benchmark()
        else:
            # A fresh run must compile its own sources after a backend rebuild.
            prefix = "triton-lora-" if lora else "triton-recompute-"
            with tempfile.TemporaryDirectory(prefix=prefix) as cache_dir:
                os.environ["TRITON_CACHE_DIR"] = cache_dir
                try:
                    benchmark()
                finally:
                    del os.environ["TRITON_CACHE_DIR"]
        return
    for (M, N, K) in SHAPES:
        flops = 2.0 * M * N * K
        print(f"\n## {M}x{N}x{K} fp16   (2*M*N*K = {flops/1e9:.2f} GFLOP)")
        print(f"{'arm':>22} {'median ms':>10} {'GFLOP/s':>10} "
              f"{'vs torch':>9} {'range ms':>17}  maxdiff")
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.float16, device="mps")
        b = torch.randn(K, N, dtype=torch.float16, device="mps")
        ref = (a.float() @ b.float())
        t = torch.empty((M, N), dtype=torch.float16, device="mps")

        def torch_arm():
            torch.matmul(a, b, out=t)

        base, base_diff = timed(torch_arm, ref, t)
        bmed = statistics.median(base)
        print(f"{'torch.matmul':>22} {bmed:10.4f} {flops/bmed*1e-6:10.1f} "
              f"{1.0:9.2f} {min(base):7.4f}-{max(base):<8.4f}  {base_diff:.3f}")

        for (bm, bn, bk, nw, gm) in ARMS:
            if M % 8 or N % 8:
                continue
            out = torch.zeros((M, N), dtype=torch.float16, device="mps")
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

            def arm():
                _canonical_matmul[grid](a, b, out, M, N, K, BM=bm, BN=bn,
                                        BK=bk, GROUP_M=gm, num_warps=nw)

            try:
                arm()
                torch.mps.synchronize()
            except Exception as exc:  # a refused envelope is a legitimate answer
                msg = str(exc).strip().splitlines()[-1][:60]
                print(f"{f'{bm}x{bn}x{bk} w{nw}':>22} {'refused':>10}  {msg}")
                continue
            samples, diff = timed(arm, ref, out)
            med = statistics.median(samples)
            verdict = "" if diff <= 0.5 else f"  WRONG({diff:.3f})"
            print(f"{f'{bm}x{bn}x{bk} w{nw}':>22} {med:10.4f} "
                  f"{flops/med*1e-6:10.1f} {med/bmed:9.2f} "
                  f"{min(samples):7.4f}-{max(samples):<8.4f}  {diff:.3f}"
                  f"{verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--recompute", action="store_true",
                       help="compare masked/plain recompute kernels and report JSONL")
    modes.add_argument("--lora", action="store_true",
                       help="compare plain/masked/verbatim fused LoRA kernels and report JSONL")
    args = parser.parse_args()
    main(recompute=args.recompute, lora=args.lora)
