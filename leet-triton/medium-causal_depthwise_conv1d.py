import torch
import triton
import triton.language as tl

@triton.jit
def _causal_depthwise_conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    L,D,
    K: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_l = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_l = offs_l < L
    mask_d = offs_d < D

    bias = tl.load(bias_ptr + offs_d, mask = mask_d, other = 0.0)
    acc = tl.zeros((BLOCK_L, BLOCK_D), dtype = tl.float32) + bias[None, :]

    batch_base = pid_b * L * D

    for k in tl.static_range(K):
        pos = offs_l - k
        pos_c = tl.maximum(pos, 0)

        load_mask = mask_l & (pos >= 0)

        w = tl.load(weight_ptr + offs_d * K + k, mask = mask_d, other = 0.0)
        xv = tl.load(
            x_ptr + batch_base + pos_c[:, None] * D + offs_d[None, :],
            mask = load_mask[:, None] & mask_d[None, :],
            other = 0.0,
        )
        acc += xv * w[None, :]

    tl.store(
        out_ptr + batch_base + offs_l[:, None] * D + offs_d[None, :],
        acc,
        mask = mask_l[:, None] & mask_d[None, :],
    )


# x, weight, bias, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    B: int,
    L: int,
    D: int,
    K: int,
):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    BLOCK_L, BLOCK_D = 64, 64
    grid = (triton.cdiv(L, BLOCK_L), triton.cdiv(D, BLOCK_D), B)
    _causal_depthwise_conv1d_kernel[grid](
        x, weight, bias, output,
        L, D,
        K = K,
        BLOCK_L = BLOCK_L,
        BLOCK_D = BLOCK_D,
        num_warps = 8, 
    )
    return output
