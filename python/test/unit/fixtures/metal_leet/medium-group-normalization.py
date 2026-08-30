# Group Normalization forward (NCHW) — OpenAI Triton implementation
#
#   mu[n,g]  = mean over channels [g*C/G, (g+1)*C/G) x H x W
#   var[n,g] = variance over the same set
#   x_hat    = (x - mu) / sqrt(var + eps)
#   y        = gamma[c] * x_hat + beta[c]
#
# Design (tuned for Tesla T4 / Turing: 40 SMs, 320 GB/s, 4 MB L2):
#   * one CTA per (batch, group) pair -> grid = N*G = 256 CTAs at the
#     benchmark shape (N=8, C=512, H=W=64, G=32); 8 warps/CTA => ~5 CTAs
#     resident per SM, enough in-flight loads to saturate DRAM.
#   * pass 1: numerically-stable Welford/Chan statistics in one read pass.
#   * pass 2: re-reads the group channel-by-channel (T4's 4 MB L2 cannot hold
#     it, so this is a genuine second DRAM read — 192 MB total traffic, same
#     class as PyTorch's own two-kernel GroupNorm) and applies gamma/beta as
#     per-channel SCALARS -> no gather loads.
#   * Cg (channels per group) and HW are constexpr: at the benchmark shape
#     every tile is full, so masks are constant-folded away, loops unroll,
#     and all global accesses compile to 128-bit vectorized LDG.128/STG.128
#     (16 fp32 per thread = 4x float4).

import torch
import triton
import triton.language as tl


@triton.jit
def _group_norm_fwd_kernel(
    X, GAMMA, BETA, Y,
    G,
    eps,
    Cg: tl.constexpr,           # channels per group (C // G)
    HW: tl.constexpr,           # H * W
    BLOCK: tl.constexpr,        # tile size for the statistics pass
    BLOCK_CH: tl.constexpr,     # tile size for the normalize pass
):
    pid = tl.program_id(0)      # one program per (n, g)
    n = pid // G
    g = pid % G

    M: tl.constexpr = Cg * HW   # elements per (n, g) group
    c0 = g * Cg                 # first channel of this group
    base = (n * (Cg * G) + c0) * HW   # element offset of the group in X

    # ---------------- pass 1: statistics ----------------
    # Welford / Chan-et-al parallel combination: one read pass, robust to
    # large-mean / tiny-variance inputs (no E[x^2] - mean^2 cancellation).
    m = 0.0
    m2 = 0.0
    cnt = 0
    for off in range(0, M, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < M
        x = tl.load(X + base + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.minimum(BLOCK, M - off).to(tl.float32)   # valid elems in tile
        bs = tl.sum(x, axis=0)
        bm = bs / b                                     # tile mean
        d = x - bm
        bm2 = tl.sum(tl.where(mask, d * d, 0.0), axis=0)  # tile M2
        delta = bm - m
        tot = cnt.to(tl.float32) + b
        m2 += bm2 + delta * delta * cnt.to(tl.float32) * b / tot
        m += delta * b / tot
        cnt += tl.minimum(BLOCK, M - off)

    mean = m
    var = tl.maximum(m2 / M, 0.0)       # guard against negative round-off
    rstd = 1.0 / tl.sqrt(var + eps)

    # ---------------- pass 2: normalize + affine ----------------
    for c in range(0, Cg):
        gam = tl.load(GAMMA + c0 + c).to(tl.float32)
        bet = tl.load(BETA + c0 + c).to(tl.float32)
        scale = gam * rstd
        ch = base + c * HW
        for off in range(0, HW, BLOCK_CH):
            idx = off + tl.arange(0, BLOCK_CH)
            mask = idx < HW
            x = tl.load(X + ch + idx, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * scale + bet
            tl.store(Y + ch + idx, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _group_norm_stats_kernel(
    X, STATS,
    G,
    NG,
    eps,
    Cg: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute one mean/rstd pair per group without an output-tile loop.

    Metal currently chooses a function-level tile loop from tensor outputs.
    Keeping the reduction in a scalar-output kernel prevents that loop from
    replaying the complete statistics pass for every normalization tile.
    """
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    M: tl.constexpr = Cg * HW
    c0 = g * Cg
    base = (n * (Cg * G) + c0) * HW

    m = 0.0
    m2 = 0.0
    cnt = 0
    for off in range(0, M, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < M
        x = tl.load(X + base + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.minimum(BLOCK, M - off).to(tl.float32)
        bm = tl.sum(x, axis=0) / b
        d = x - bm
        bm2 = tl.sum(tl.where(mask, d * d, 0.0), axis=0)
        delta = bm - m
        tot = cnt.to(tl.float32) + b
        m2 += bm2 + delta * delta * cnt.to(tl.float32) * b / tot
        m += delta * b / tot
        cnt += tl.minimum(BLOCK, M - off)

    var = tl.maximum(m2 / M, 0.0)
    tl.store(STATS + pid, m)
    tl.store(STATS + NG + pid, 1.0 / tl.sqrt(var + eps))


@triton.jit
def _group_norm_apply_kernel(
    X, GAMMA, BETA, STATS, Y,
    G,
    NG,
    Cg: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_CH: tl.constexpr,
):
    """Normalize one group using precomputed statistics."""
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    c0 = g * Cg
    base = (n * (Cg * G) + c0) * HW
    mean = tl.load(STATS + pid)
    rstd = tl.load(STATS + NG + pid)

    for c in range(0, Cg):
        gam = tl.load(GAMMA + c0 + c).to(tl.float32)
        bet = tl.load(BETA + c0 + c).to(tl.float32)
        scale = gam * rstd
        ch = base + c * HW
        for off in range(0, HW, BLOCK_CH):
            idx = off + tl.arange(0, BLOCK_CH)
            mask = idx < HW
            x = tl.load(X + ch + idx, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * scale + bet
            tl.store(Y + ch + idx, y.to(Y.dtype.element_ty), mask=mask)


# X, gamma, beta, Y are tensors on the GPU
def solve(
    X: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    Y: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
    G: int,
    eps: float,
):
    if not X.is_contiguous():
        X = X.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()

    if Y.is_contiguous():
        Yout = Y
    else:
        Yout = torch.empty_like(X)

    HW = H * W
    Cg = C // G
    BLOCK = 4096                                      # stats-pass tile
    BLOCK_CH = min(4096, triton.next_power_of_2(HW))  # normalize-pass tile

    NG = N * G
    grid = (NG,)
    if X.device.type == "mps":
        # The current Metal lowering replays reductions inside the output tile
        # loop of a fused kernel. Splitting only the Metal path avoids that
        # compiler limitation while retaining the T4-tuned fused CUDA path.
        stats = torch.empty((2 * NG,), device=X.device, dtype=torch.float32)
        _group_norm_stats_kernel[grid](
            X, stats,
            G, NG, eps,
            Cg=Cg, HW=HW, BLOCK=BLOCK,
            num_warps=4,
        )
        _group_norm_apply_kernel[grid](
            X, gamma, beta, stats, Yout,
            G, NG,
            Cg=Cg, HW=HW, BLOCK_CH=BLOCK_CH,
            num_warps=32,
        )
    else:
        _group_norm_fwd_kernel[grid](
            X, gamma, beta, Yout,
            G,
            eps,
            Cg=Cg, HW=HW,
            BLOCK=BLOCK, BLOCK_CH=BLOCK_CH,
            num_warps=8,
        )

    if Yout is not Y:
        Y.copy_(Yout)
