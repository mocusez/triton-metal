
import torch
import triton
import triton.language as tl

@triton.jit
def _grpo_surrogate_loss_kernel(
    rewards_ptr,
    log_pi_ptr,
    log_pi_old_ptr,
    log_ref_ptr,
    output_ptr,
    clip_eps, beta,
    G, S,
    inv_total,
    BLOCK_G: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // G
    g = pid % G

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    rew = tl.load(rewards_ptr + b * G + offs_g, mask=mask_g, other = 0.0)
    mu = tl.sum(rew, axis = 0) / G
    diff = tl.where(mask_g, rew - mu, 0.0)
    sigma = tl.sqrt(tl.sum(diff * diff, axis = 0) / G)
    adv = (rew - mu) / (sigma + 1e-8)
    a = tl.sum(tl.where(offs_g == g, adv, 0.0), axis = 0)

    lo = 1.0 - clip_eps
    hi = 1.0 + clip_eps

    row_start = pid * S
    acc = 0.0
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        lp = tl.load(log_pi_ptr + row_start + offs, mask = mask, other = 0.0)
        lpo = tl.load(log_pi_old_ptr + row_start + offs, mask = mask, other = 0.0)
        lrf = tl.load(log_ref_ptr + row_start + offs, mask = mask, other = 0.0)

        ratio = tl.exp(lp - lpo)

        ratio_c = tl.minimum(tl.maximum(ratio, lo), hi)
        l_clip = tl.minimum(ratio * a, ratio_c * a)

        d = lrf - lp
        k = tl.exp(d) - d - 1.0

        acc += tl.sum(tl.where(mask, l_clip - beta * k, 0.0), axis = 0)
    tl.atomic_add(output_ptr, -acc * inv_total)



# rewards, log_pi, log_pi_old, log_ref, output are tensors on the GPU
def solve(
    rewards: torch.Tensor,
    log_pi: torch.Tensor,
    log_pi_old: torch.Tensor,
    log_ref: torch.Tensor,
    output: torch.Tensor,
    clip_eps: float,
    beta: float,
    B: int,
    G: int,
    S: int,
):
    output.zero_()

    _grpo_surrogate_loss_kernel[(B * G,)](
        rewards, log_pi, log_pi_old, log_ref, output,
        clip_eps, beta,
        G, S, 1.0 / (B * G * S),
        BLOCK_G = triton.next_power_of_2(G),
        BLOCK_S = 1024,
        num_warps = 4
    )
