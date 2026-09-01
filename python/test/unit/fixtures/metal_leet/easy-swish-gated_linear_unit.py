import torch
import triton
import triton.language as tl


@triton.jit
def swiglu(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    ofs = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * pid
    ofs2 = ofs + N//2
    mask = ofs2 < N
    x1 = tl.load(in_ptr + ofs, mask = mask)
    x1 *= tl.sigmoid(x1)
    x2 = tl.load(in_ptr + ofs2, mask = mask)
    tl.store(out_ptr + ofs, x1 * x2, mask = mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
