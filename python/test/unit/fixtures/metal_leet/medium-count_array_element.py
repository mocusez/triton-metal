import torch
import triton
import triton.language as tl


@triton.jit
def kernel(input_ptr, output_ptr, N , K, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE % 4 == 0)
    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    data = tl.load(input_ptr + offs, mask = mask, other = 0.)
    sum = tl.sum(data == K)
    if(sum > 0):
        tl.atomic_add(output_ptr, sum)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 1024
    grid = lambda meta :(triton.cdiv(N, meta['BLOCK_SIZE']),)
    kernel[grid](input, output, N, K, BLOCK_SIZE = BLOCK_SIZE)
