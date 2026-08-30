import torch
import triton
import triton.language as tl


@triton.jit
def _zero_kernel(output):
    tl.store(output, 0.0)


@triton.jit
def _ppo_kernel(
    advantages,
    log_pi,
    log_pi_old,
    output,
    clip_eps,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load inputs
    adv = tl.load(
        advantages + offsets,
        mask=mask,
        other=0.0,
    )

    lp = tl.load(
        log_pi + offsets,
        mask=mask,
        other=0.0,
    )

    lp_old = tl.load(
        log_pi_old + offsets,
        mask=mask,
        other=0.0,
    )

    # r = exp(log_pi - log_pi_old)
    ratio = tl.exp(lp - lp_old)

    lower = 1.0 - clip_eps
    upper = 1.0 + clip_eps

    # Equivalent to:
    #
    # clipped_ratio = clip(ratio, lower, upper)
    # surrogate = min(
    #     ratio * adv,
    #     clipped_ratio * adv
    # )
    #
    # For A >= 0:
    #   surrogate = A * min(ratio, upper)
    #
    # For A < 0:
    #   surrogate = A * max(ratio, lower)
    pos_ratio = tl.minimum(ratio, upper)
    neg_ratio = tl.maximum(ratio, lower)

    effective_ratio = tl.where(
        adv >= 0.0,
        pos_ratio,
        neg_ratio,
    )

    surrogate = effective_ratio * adv

    # Masked positions have adv=0, so they contribute zero.
    block_sum = tl.sum(surrogate, axis=0)

    # PPO loss = -mean(surrogate)
    block_loss = -block_sum / N

    # One atomic add per program.
    tl.atomic_add(output, block_loss)


# advantages, log_pi, log_pi_old, output are tensors on the GPU
def solve(
    advantages: torch.Tensor,
    log_pi: torch.Tensor,
    log_pi_old: torch.Tensor,
    output: torch.Tensor,
    clip_eps: float,
    B: int,
    S: int,
):
    N = B * S

    # Important: clear output before atomic accumulation.
    _zero_kernel[(1,)](output)

    BLOCK_SIZE = 1024

    grid = (
        triton.cdiv(N, BLOCK_SIZE),
    )

    _ppo_kernel[grid](
        advantages,
        log_pi,
        log_pi_old,
        output,
        clip_eps,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )