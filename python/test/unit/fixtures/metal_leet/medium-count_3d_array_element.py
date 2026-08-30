import torch
import triton
import triton.language as tl

@triton.jit
def count_3d_kernel(input, output, P, n_elements, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    val = tl.load(input + offs, mask, -1)
    ret = tl.where(mask & (val == P), 1, 0)
    ret = tl.sum(ret, axis = 0)
    tl.atomic_add(output, ret)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int, P: int):
    BLOCK_SIZE = 1024
    n_elements = N * M * K
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    count_3d_kernel[grid](input, output, P, n_elements, BLOCK_SIZE)
