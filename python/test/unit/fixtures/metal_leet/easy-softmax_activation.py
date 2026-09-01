import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start= pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    y = 1.0/(1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets,y, mask=mask)


# X, Y are tensors on the GPU
def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sigmoid_kernel[grid](X, Y, N, BLOCK_SIZE)
