import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offset < n_elements
    vals = tl.load(input + offset, mask)
    vals1 = 1/(1 + tl.exp(-vals))
    tl.store(output + offset, vals * vals1, mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    silu_kernel[grid](input, output, N, BLOCK_SIZE)
