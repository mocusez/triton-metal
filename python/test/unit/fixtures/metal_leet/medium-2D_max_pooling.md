import torch
import triton
import triton.language as tl

@triton.jit
def max_pooling_kenrel(
    input_ptr, output_ptr,
    N, C, H, W, H_out, W_out,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid_ho = tl.program_id(0)
    pid_wo = tl.program_id(1)
    pid_nc = tl.program_id(2)

    input_ptr += pid_nc * H * W
    output_ptr += pid_nc * H_out * W_out

    offs_ho = pid_ho * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_wo = pid_wo * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_hi_st = offs_ho * stride - padding
    offs_wi_st = offs_wo * stride - padding

    max_val = tl.full((BLOCK_SIZE, BLOCK_SIZE), float('-inf'), dtype=tl.float32)

    for i in range(kernel_size):
        offs_hi = offs_hi_st + i

        for j in range(kernel_size):
            offs_wi = offs_wi_st + j
            offs_input = offs_hi[:,None] * W + offs_wi[None,:]
            mask_input = (offs_hi[:, None] < H) & (offs_wi[None, :] < W) & (offs_hi[:, None] >= 0) & (offs_wi[None, :] >= 0)
            input = tl.load(input_ptr + offs_input, mask=mask_input, other=float('-inf'))
            max_val = tl.maximum(input, max_val)

    offs_output = offs_ho[:,None] * W_out + offs_wo[None,:]
    mask_output = (offs_ho[:,None] < H_out) & (offs_wo[None,:] < W_out)
    tl.store(output_ptr + offs_output, max_val, mask = mask_output)



# input, output are tensors on the GPU
def solve(input, output, N, C, H, W, kernel_size, stride, padding):
    BLOCK_SIZE = 32

    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    grid = (
        triton.cdiv(H_out, BLOCK_SIZE),
        triton.cdiv(W_out, BLOCK_SIZE),
        N * C,
    )

    max_pooling_kenrel[grid](
        input, output,
        N, C, H, W, H_out, W_out,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
