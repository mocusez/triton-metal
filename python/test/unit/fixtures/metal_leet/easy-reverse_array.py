import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    addr = input + i
    addr_ = input + N - 1 - i
    mask = i < N // 2
    tmp = tl.load(addr, mask = mask)
    tmp_ = tl.load(addr_, mask = mask)
    tl.store(addr, tmp_, mask = mask)
    tl.store(addr_, tmp, mask = mask)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)
