import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C,
    eps,
    BLOCK_C: tl.constexpr
):
    row = tl.program_id(axis = 0)

    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    offsets = row * C + cols

    x = tl.load(x_ptr + offsets, mask = mask, other = 0.0).to(tl.float32)

    mean = tl.sum(x, axis = 0) / C

    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis = 0) / C

    inv_std = 1.0 / tl.sqrt(variance + eps)

    weight = tl.load(weight_ptr + cols, mask = mask, other = 0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask = mask, other = 0.0).to(tl.float32)

    y = weight * centered * inv_std + bias

    tl.store(output_ptr + offsets, y , mask = mask)

# input, weight, bias, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    BLOCK_C = triton.next_power_of_2(C)

    num_warps = min(max(BLOCK_C // 256, 1), 8)

    _layer_norm_forward_kernel[(N, )](
        input,
        weight,
        bias,
        output,
        C,
        eps,
        BLOCK_C = BLOCK_C,
        num_warps = num_warps
    )
