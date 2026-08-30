import torch
import triton
import triton.language as tl

@triton.jit
def vector_dot(a, b, result, n, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(axis=0)
    vector_offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = vector_offset < n

    a_slice = tl.load(a + vector_offset, mask=mask)
    b_slice = tl.load(b + vector_offset, mask=mask)

    tl.atomic_add(result, tl.sum(a_slice * b_slice))


def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_dot[grid](a, b, result, n, BLOCK_SIZE)
