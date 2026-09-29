"""Matmul epilogues between the K loop and the store.

A VALUE-ONLY epilogue -- a function of each accumulator element alone, with
constant operands -- is applied by the SIMD-group tiers to their accumulator
matrices in registers. Everything else (a bias vector, a runtime scalar, a
row-dependent select) must still be answered, by whichever tier takes it.

Cells vary the epilogue (a bias vector along either axis, masked or not, f32
or widened f16, a runtime scalar, and combinations), the K-loop form, the
store (masked or not, output dtype), the tile and the warps. Tags encode all
of them.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sweep_lib import Sweep  # noqa: E402


@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, bias_ptr, rbias_ptr, hbias_ptr, M, N, K, alpha,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           K_TILES: tl.constexpr, LOOP: tl.constexpr, EPI: tl.constexpr,
           MASKED: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if LOOP == 0:      # tutorial: pointer advance, runtime trip count, ragged K
        a_ptrs = a_ptr + (offs_m % M)[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + (offs_n % N)[None, :]
        for kk in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - kk * BK, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - kk * BK, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * N
    elif LOOP == 1:    # static trip count, pointer advance
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        for _ in range(0, K_TILES):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
            a_ptrs += BK
            b_ptrs += BK * N
    else:              # addresses recomputed from the induction variable
        for kk in range(0, K, BK):
            ok = kk + offs_k
            a = tl.load(a_ptr + offs_m[:, None] * K + ok[None, :],
                        mask=(offs_m[:, None] < M) & (ok[None, :] < K), other=0.0)
            b = tl.load(b_ptr + ok[:, None] * N + offs_n[None, :],
                        mask=(ok[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
    if EPI == 1:       # leaky relu
        acc = tl.where(acc >= 0, acc, 0.01 * acc)
    elif EPI == 2:     # scale and shift
        acc = acc * 0.5 + 1.0
    elif EPI == 3:     # relu
        acc = tl.maximum(acc, 0.0)
    elif EPI == 4:     # sigmoid
        acc = 1.0 / (1.0 + tl.exp(-acc))
    elif EPI == 5:     # clamp, abs
        acc = tl.minimum(tl.maximum(tl.abs(acc) - 1.0, -2.0), 3.0)
    elif EPI == 6:     # bias vector: NOT value-only
        acc = acc + tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    elif EPI == 7:     # runtime scalar: NOT constant
        acc = acc * alpha
    elif EPI == 8:     # row-dependent: NOT value-only
        acc = tl.where(offs_m[:, None] % 2 == 0, acc, -acc)
    elif EPI == 9:     # silu
        acc = acc / (1.0 + tl.exp(-acc))
    elif EPI == 10:    # row vector along M
        acc = acc + tl.load(rbias_ptr + offs_m, mask=offs_m < M, other=0.0)[:, None]
    elif EPI == 11:    # column bias with no mask: the kernel reads the whole tile
        acc = acc + tl.load(bias_ptr + offs_n)[None, :]
    elif EPI == 12:    # f16 bias widened
        acc = acc + tl.load(hbias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    elif EPI == 13:    # bias, runtime scale and a leaky relu together
        acc = acc * alpha + tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
        acc = tl.where(acc >= 0, acc, 0.01 * acc)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    out = acc.to(c_ptr.dtype.element_ty)
    if MASKED:
        tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    else:
        tl.store(c_ptrs, out)


def _epilogue(r, epi, bias, alpha, m, rbias=None, hbias=None):
    if epi == 1:
        return torch.where(r >= 0, r, 0.01 * r)
    if epi == 2:
        return r * 0.5 + 1.0
    if epi == 3:
        return r.clamp(min=0)
    if epi == 4:
        return torch.sigmoid(r)
    if epi == 5:
        return (r.abs() - 1.0).clamp(-2.0, 3.0)
    if epi == 6:
        return r + bias[None, :]
    if epi == 7:
        return r * alpha
    if epi == 8:
        sign = torch.where(torch.arange(m) % 2 == 0, 1.0, -1.0).double()
        return r * sign[:, None]
    if epi == 9:
        return r * torch.sigmoid(r)
    if epi == 10:
        return r + rbias[:, None]
    if epi == 11:
        return r + bias[None, :]
    if epi == 12:
        return r + hbias[None, :]
    if epi == 13:
        t = r * alpha + bias[None, :]
        return torch.where(t >= 0, t, 0.01 * t)
    return r


def _cell(loop, epi, shape, blk, w, masked, odt):
    m, n, k = shape
    bm, bn, bk = blk
    torch.manual_seed(m + 3 * n + 7 * k + epi)
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    # Padded to whole tiles: EPI 11 reads its bias unmasked, as the kernel says.
    bias = torch.randn(triton.cdiv(n, bn) * bn)
    rbias = torch.randn(m)
    hbias = torch.randn(n).half()
    c = torch.zeros(m, n, dtype=odt, device="mps")
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    kernel[grid](a.to("mps"), b.to("mps"), c, bias.to("mps"), rbias.to("mps"),
                 hbias.to("mps"), m, n, k, 0.75,
                 BM=bm, BN=bn, BK=bk, K_TILES=k // bk, LOOP=loop, EPI=epi,
                 MASKED=masked, num_warps=w)
    torch.mps.synchronize()
    ref = _epilogue(a.double() @ b.double(), epi, bias[:n].double(), 0.75, m,
                    rbias.double(), hbias.double())
    got = c.cpu().double()
    tol = {torch.float32: 2e-3, torch.float16: 2e-2, torch.bfloat16: 1e-1}[odt]
    err = (got - ref).abs() / (1 + ref.abs())
    if err.max().item() <= tol:
        return None
    return f"maxrel={err.max().item():.3g} ({int((err > tol).sum())}/{err.numel()})"


LOOPS = ["tutorial", "static", "recompute"]
EPIS = ["none", "leaky", "scaleshift", "relu", "sigmoid", "clampabs", "bias",
        "alpha", "rowsign", "silu", "rowbias", "biasnomask", "halfbias", "biasalphaleaky"]
ODTS = [(torch.float32, "f32"), (torch.float16, "f16"), (torch.bfloat16, "bf16")]


def cells(sw):
    for loop, ln in enumerate(LOOPS):
        for epi, en in enumerate(EPIS):
            for shape, masked in (((128, 128, 96), 0), ((100, 72, 88), 1),
                                  ((128, 128, 96), 1)):
                if loop == 1 and (masked or shape[2] % 32):
                    continue  # the static loop has no K guard; keep K whole
                if loop == 1 and masked == 0 and shape != (128, 128, 96):
                    continue
                for blk in ((64, 64, 32), (32, 32, 16)):
                    for w in (1, 4):
                        for odt, dn in ODTS:
                            if not masked and dn != "f32":
                                continue  # a narrowing store needs the masked path
                            tag = (f"{ln}_{en}_{'x'.join(map(str, shape))}_"
                                   f"b{blk[0]}x{blk[1]}x{blk[2]}_w{w}_"
                                   f"{'m' if masked else 'u'}_{dn}")
                            sw.run(tag, lambda loop=loop, epi=epi, shape=shape,
                                   blk=blk, w=w, masked=masked, odt=odt:
                                   _cell(loop, epi, shape, blk, w, masked, odt))


if __name__ == "__main__":
    sw = Sweep("s_dot_epilogue")
    cells(sw)
    sw.finish()
