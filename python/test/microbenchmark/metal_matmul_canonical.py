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
        METAL_PERF_SETTLE=0 ... to skip the cool-down when the machine is
        already known idle.
"""
import os
import statistics
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


def main():
    # Never measure within a minute of a suite run; this script is usually
    # started right after one. Run-to-run spread within ONE build is ~10% on
    # this machine, which is larger than many changes worth making, and two
    # readings taken on a busy machine have already been quoted wrongly.
    time.sleep(float(os.environ.get("METAL_PERF_SETTLE", "75")))
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
    main()
