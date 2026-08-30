import torch
import triton
import triton.language as tl


@triton.jit
def _compose_affine(a_left, b_left, a_right, b_right):
    # 将 GAE 递推式 A_t = delta_t + c * A_(t+1)
    # 表示为仿射变换 x -> a*x + b，并组合左右两个区间。
    return a_right * a_left, b_right + a_right * b_left


@triton.jit
def _multiply(left, right):
    return left * right


@triton.jit
def _gae_local_kernel(
    rewards_ptr,
    values_ptr,
    advantages_ptr,
    work_ptr,
    gamma,
    c,
    S,
    num_blocks,
    stride_rb,
    stride_rs,
    stride_vb,
    stride_vs,
    stride_ab,
    stride_as,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // num_blocks
    block = pid % num_blocks

    offsets = tl.arange(0, BLOCK_SIZE)
    t = block * BLOCK_SIZE + offsets
    valid = t < S

    b64 = batch.to(tl.int64)
    t64 = t.to(tl.int64)

    reward = tl.load(
        rewards_ptr + b64 * stride_rb + t64 * stride_rs,
        mask=valid,
        other=0.0,
    )
    value = tl.load(
        values_ptr + b64 * stride_vb + t64 * stride_vs,
        mask=valid,
        other=0.0,
    )
    next_value = tl.load(
        values_ptr + b64 * stride_vb + (t64 + 1) * stride_vs,
        mask=(t + 1) < S,
        other=0.0,
    )

    # 最后一个时间步的 next_value 为 0。
    delta = reward + gamma * next_value - value

    # 对不满 BLOCK_SIZE 的最后一个块，无效位置使用恒等变换。
    trans_scale = tl.where(valid, c, 1.0)
    trans_shift = tl.where(valid, delta, 0.0)

    # 块内反向 scan：
    # A_t = local_advantage_t + block_scale_t * A_block_end
    block_scale, local_advantage = tl.associative_scan(
        (trans_scale, trans_shift),
        axis=0,
        combine_fn=_compose_affine,
        reverse=True,
    )

    tl.store(
        advantages_ptr + b64 * stride_ab + t64 * stride_as,
        local_advantage,
        mask=valid,
    )

    # work 逻辑形状为 [B, 2, num_blocks]：
    # plane 0 保存块起点局部 GAE，之后原地改为进入该块的 carry；
    # plane 1 保存整个块的系数 c^(block_length)。
    work_base = b64 * (2 * num_blocks)
    first_lane = offsets == 0

    block_local = tl.sum(
        tl.where(first_lane, local_advantage, 0.0),
        axis=0,
    )
    block_coeff = tl.sum(
        tl.where(first_lane, block_scale, 0.0),
        axis=0,
    )

    tl.store(work_ptr + work_base + block, block_local)
    tl.store(
        work_ptr + work_base + num_blocks + block,
        block_coeff,
    )


@triton.jit
def _gae_carry_kernel(work_ptr, num_blocks):
    batch = tl.program_id(0)
    work_base = batch.to(tl.int64) * (2 * num_blocks)

    # carry 是当前块右侧边界的 advantage。
    carry = tl.zeros((), dtype=tl.float32)

    for i in range(num_blocks):
        block = num_blocks - 1 - i

        block_local = tl.load(work_ptr + work_base + block)
        block_coeff = tl.load(
            work_ptr + work_base + num_blocks + block
        )

        # 保存进入当前块的 carry。
        tl.store(work_ptr + work_base + block, carry)

        # 当前块起点的完整 advantage。
        carry = block_local + block_coeff * carry


@triton.jit
def _gae_apply_carry_kernel(
    advantages_ptr,
    work_ptr,
    c,
    S,
    num_blocks,
    stride_ab,
    stride_as,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // num_blocks
    block = pid % num_blocks

    offsets = tl.arange(0, BLOCK_SIZE)
    t = block * BLOCK_SIZE + offsets
    valid = t < S

    b64 = batch.to(tl.int64)
    t64 = t.to(tl.int64)

    local_advantage = tl.load(
        advantages_ptr + b64 * stride_ab + t64 * stride_as,
        mask=valid,
        other=0.0,
    )

    # 对位置 t，右侧 carry 的系数为 c^(block_end-t)。
    trans_scale = tl.where(valid, c, 1.0)
    carry_scale = tl.associative_scan(
        trans_scale,
        axis=0,
        combine_fn=_multiply,
        reverse=True,
    )

    work_base = b64 * (2 * num_blocks)
    carry = tl.load(work_ptr + work_base + block)

    result = local_advantage + carry_scale * carry

    tl.store(
        advantages_ptr + b64 * stride_ab + t64 * stride_as,
        result,
        mask=valid,
    )


# rewards, values, advantages are tensors on the GPU
def solve(
    rewards: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    gamma: float,
    lam: float,
    B: int,
    S: int,
):
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(S, BLOCK_SIZE)
    c = float(gamma) * float(lam)

    # 块级临时空间，每个 batch 只有 2*num_blocks 个 float32。
    work = torch.empty(
        (B, 2, num_blocks),
        device=advantages.device,
        dtype=torch.float32,
    )

    grid = (B * num_blocks,)

    _gae_local_kernel[grid](
        rewards,
        values,
        advantages,
        work,
        float(gamma),
        c,
        S,
        num_blocks,
        rewards.stride(0),
        rewards.stride(1),
        values.stride(0),
        values.stride(1),
        advantages.stride(0),
        advantages.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    _gae_carry_kernel[(B,)](
        work,
        num_blocks,
        num_warps=1,
    )

    _gae_apply_carry_kernel[grid](
        advantages,
        work,
        c,
        S,
        num_blocks,
        advantages.stride(0),
        advantages.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )