import torch
import triton
import triton.language as tl

@triton.jit
def merge_kernel(A, B, C, M, N, LOG2, BLOCK_SIZE: tl.constexpr):
    out_indices = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_indices < M + N

    a_lb = tl.maximum(-1, out_indices - N)
    a_ub = tl.minimum(M - 1, out_indices)

    max_search_steps = LOG2
    for _ in range(max_search_steps):
        a_mb = (a_lb + a_ub + 1) // 2
        a_val = tl.load(A + a_mb, mask = out_mask & (a_mb >= 0), other = float('-inf'))
        b_mb = out_indices - a_mb
        b_val = tl.load(B + b_mb, mask= out_mask & (b_mb < N), other=float('inf'))
        cond = a_val <= b_val
        a_lb = tl.where(cond, a_mb, a_lb)
        a_ub = tl.where(cond, a_ub, a_mb - 1)

    a_val = tl.load(A + a_lb, mask = out_mask & (a_lb >= 0), other = float('-inf'))
    b_prev = (out_indices - a_lb) - 1
    b_prev_val = tl.load(B + b_prev, mask = out_mask & (b_prev >= 0), other = float('-inf'))
    cond2 = a_val <= b_prev_val
    out_val = tl.where(cond2, b_prev_val, a_val)
    tl.store(C + out_indices, out_val, mask = out_mask)


# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(M + N, BLOCK_SIZE),)
    merge_kernel[grid](A, B, C, M, N, max(M + 1, N + 1).bit_length(), BLOCK_SIZE)
