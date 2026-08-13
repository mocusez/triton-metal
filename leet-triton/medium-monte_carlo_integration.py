import torch
import triton
import triton.language as tl

@triton.jit
def kernel(y_samples, result, a, b, n_samples, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask= offset < n_samples
    y = tl.load(y_samples + offset, mask = mask, other = 0.0)
    tl.atomic_add(result, tl.sum(y) * (1.0 / n_samples) * (b - a))


# y_samples, result are tensors on the GPU
def solve(y_samples: torch.Tensor, result: torch.Tensor, a: float, b: float, n_samples: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_samples, BLOCK_SIZE),)
    kernel[grid](y_samples, result, a, b, n_samples, BLOCK_SIZE)

