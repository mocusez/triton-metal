"""What does a causal flash-attention kernel cost on the Metal backend?

`metal.fused_attention` claims this shape and emits simdgroup MMA for both
dots, so nothing here is on the scalar floor. What is NOT known is how the
emitted form compares with the platform's own attention, and which part of the
per-key-block pipeline the time actually sits in. The one REMAINING item
recorded against the op -- holding several MMA accumulators per simdgroup
instead of one -- is worth doing only if the MMA loops are where the time is,
and the emitter interleaves three other things with them:

    stage K^T,V -> barrier -> dot1 (Q@K^T) -> barrier
      -> score transform + online softmax, ONE QUERY ROW PER THREAD  <-- scalar
      -> barrier -> dot2 (P@V) -> barrier -> accumulate O

So this harness sweeps `BLOCK_SIZE_RUNNING` (the key block, which divides the
scalar pass's trip count per key but not its total work) against `num_warps`
(which divides the MMA work but NOT the scalar pass, since that one is indexed
by `_fa_tid` and capped at BM rows). If the MMA loops dominate, more warps
help; if the scalar softmax dominates, they do not, and register blocking the
MMA is the wrong target.

The kernel is `fixtures/metal_leet/hard-causal_self-Attention.py` verbatim in
shape -- a real corpus kernel, not a stand-in. The perf landscape for this
backend has twice been read off a kernel that was not the one people write.

Discipline this project has paid for: warm up, wall-clock a batch with
`torch.mps.synchronize`, never `torch.mps.Event`, five samples per arm, report
medians AND ranges, check every arm's answer in the same process so no timing
is quoted for a kernel that is wrong, and do not measure within a minute of a
pytest run.

Usage:  pixi run --frozen python python/test/microbenchmark/metal_attention_causal.py
        METAL_PERF_SETTLE=0 ... to skip the cool-down when the machine is idle.
"""
import math
import os
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_attn(q, k, v, output, M, d,
                 BLOCK_SIZE_ROW: tl.constexpr,
                 BLOCK_SIZE_COL: tl.constexpr,
                 BLOCK_SIZE_RUNNING: tl.constexpr):
    pid = tl.program_id(0)

    q_offsets = pid * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    q_mask = q_offsets < M

    col_offsets = tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < d

    q_slice = tl.load(q + q_offsets[:, None] * d + col_offsets[None, :],
                      mask=q_mask[:, None] & col_mask[None, :], other=0.0)

    scale = 1 / tl.sqrt(d.to(tl.float32))

    s_max = tl.full([BLOCK_SIZE_ROW, ], -float('inf'), dtype=tl.float32)
    s_sum = tl.zeros([BLOCK_SIZE_ROW, ], dtype=tl.float32)
    accm = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.float32)

    for slice in tl.range(0, M, BLOCK_SIZE_RUNNING):
        slice_offsets = slice + tl.arange(0, BLOCK_SIZE_RUNNING)
        slice_mask = slice_offsets < M
        slice_ptr = slice_offsets[:, None] * d + col_offsets[None, :]

        k_slice = tl.load(k + slice_ptr,
                          mask=slice_mask[:, None] & col_mask[None, :], other=0.0)
        v_slice = tl.load(v + slice_ptr,
                          mask=slice_mask[:, None] & col_mask[None, :], other=0.0)

        ac = tl.dot(q_slice, tl.trans(k_slice)) * scale
        is_valid = (q_offsets[:, None] >= slice_offsets[None, :]) & slice_mask[None, :]
        ac_causal = tl.where(is_valid, ac, -float('inf'))

        new_max = tl.maximum(s_max, tl.max(ac_causal, axis=1))
        alpha = tl.exp(s_max - new_max)
        s_max = new_max

        ac_causal = tl.exp(ac_causal - new_max[:, None])
        s_sum = tl.fma(s_sum, alpha, tl.sum(ac_causal, axis=1))

        accm = tl.fma(accm, alpha[:, None], tl.dot(ac_causal, v_slice))
    accm = accm / s_sum[:, None]

    tl.store(output + q_offsets[:, None] * d + col_offsets[None, :], accm,
             mask=q_mask[:, None] & col_mask[None, :])


SHAPES = [(512, 64), (1024, 64), (2048, 64), (1024, 128)]
# `METAL_PERF_SHAPES="1024x64,2048x64"` narrows the run; the full list is
# several minutes because a causal kernel that walks every key block is
# doing the dense work.

# BM is held at 16 because that is what the corpus kernel picks and because the
# emitter's score transform is indexed by `_fa_tid` -- rows above BM are simply
# not computed by this pass, so BM and the warp count are coupled the same way
# the matmul's block size and warp count turned out to be.
ARMS = [(bm, bn, nw)
        for bm in (16, 32, 64)
        for bn in (32, 64, 128)
        for nw in (1, 2, 4, 8)]

BATCH = 10
SAMPLES = 5


def timed(fn):
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    samples = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter()
        for _ in range(BATCH):
            fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - t0) / BATCH * 1e3)
    return samples


def main():
    time.sleep(float(os.environ.get("METAL_PERF_SETTLE", "75")))
    want = os.environ.get("METAL_PERF_SHAPES")
    shapes = ([tuple(int(x) for x in s.split("x")) for s in want.split(",")]
              if want else SHAPES)
    for (M, d) in shapes:
        # Causal: roughly half the key blocks are fully masked, but this kernel
        # walks all of them, so quote the DENSE count and note it.
        flops = 4.0 * M * M * d
        print(f"\n## causal attention M={M} d={d} fp32  "
              f"(dense 4*M*M*d = {flops/1e9:.2f} GFLOP; the kernel walks every "
              f"key block, so this is the work it actually does)")
        print(f"{'arm':>18} {'median ms':>10} {'GFLOP/s':>10} {'vs torch':>9} "
              f"{'range ms':>17}  maxdiff")

        torch.manual_seed(0)
        q = torch.randn(M, d, device="mps")
        k = torch.randn(M, d, device="mps")
        v = torch.randn(M, d, device="mps")
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.cpu().double(), k.cpu().double(), v.cpu().double(),
            is_causal=True).float()

        tq, tk, tv = q[None, None], k[None, None], v[None, None]

        def torch_arm():
            torch.nn.functional.scaled_dot_product_attention(
                tq, tk, tv, is_causal=True)

        torch_arm()
        torch.mps.synchronize()
        tout = torch.nn.functional.scaled_dot_product_attention(
            tq, tk, tv, is_causal=True)[0, 0].cpu()
        base = timed(torch_arm)
        bmed = statistics.median(base)
        tdiff = (tout - ref).abs().max().item()
        print(f"{'torch SDPA':>18} {bmed:10.4f} {flops/bmed*1e-6:10.1f} "
              f"{1.0:9.2f} {min(base):7.4f}-{max(base):<8.4f}  {tdiff:.4f}")

        bd = max(16, triton.next_power_of_2(d))
        for (bm, bn, nw) in ARMS:
            out = torch.zeros(M, d, device="mps")
            grid = (triton.cdiv(M, bm),)

            def arm():
                _causal_attn[grid](q, k, v, out, M, d, bm, bd, bn, num_warps=nw)

            try:
                arm()
                torch.mps.synchronize()
            except Exception as exc:  # a refused envelope is a legitimate answer
                msg = str(exc).strip().splitlines()[-1][:58]
                print(f"{f'{bm}x{bn} w{nw}':>18} {'refused':>10}  {msg}")
                continue
            samples = timed(arm)
            med = statistics.median(samples)
            diff = (out.cpu() - ref).abs().max().item()
            verdict = "" if diff <= 2e-3 else f"  WRONG({diff:.3e})"
            print(f"{f'{bm}x{bn} w{nw}':>18} {med:10.4f} "
                  f"{flops/med*1e-6:10.1f} {med/bmed:9.2f} "
                  f"{min(samples):7.4f}-{max(samples):<8.4f}  {diff:.4f}"
                  f"{verdict}")


if __name__ == "__main__":
    main()
