import torch
import triton
import triton.language as tl

@triton.jit
def max_subarray_sum_kernel(
    input, output, N, windows_size, len, BLOCK_SIZE: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < len
    result = tl.zeros([BLOCK_SIZE], tl.float32)
    for i in range(windows_size):
        offs_i = offs + i
        mask_i = offs_i < N
        val = tl.load(input + offs_i, mask_i)
        result += val
    ret = tl.where(mask, result, float("-inf"))
    ret = tl.max(ret)
    tl.atomic_max(output, ret)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    len = N - window_size + 1
    BLOCK_SIZE = 32
    output.fill_(-2147483648)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    max_subarray_sum_kernel[grid](input, output, N, window_size, len, BLOCK_SIZE)
