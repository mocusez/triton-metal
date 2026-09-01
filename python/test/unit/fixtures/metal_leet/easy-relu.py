import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask = mask)
    y = tl.maximum(0,x)
    tl.store(output_ptr + offsets,y,mask = mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)
