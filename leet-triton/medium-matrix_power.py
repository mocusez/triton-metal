import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A,B,C, n:tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_K:tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    offsets_i = BLOCK_SIZE_W * pid_i + tl.arange(0, BLOCK_SIZE_W)[:, None]
    offsets_j = BLOCK_SIZE_H * pid_j + tl.arange(0, BLOCK_SIZE_H)[None, :]
    C_tile = tl.zeros((BLOCK_SIZE_W, BLOCK_SIZE_H), dtype=tl.float32)
    for k in range(tl.cdiv(n, BLOCK_SIZE_K)):
        offsets_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_rows = tl.load(A + offsets_i * n + offsets_k[None, :], mask = (offsets_i < n) & (offsets_k[None,:] < n), other = 0.0)
        B_cols = tl.load(B + offsets_k[:, None] * n + offsets_j, mask = (offsets_j < n) & (offsets_k[:,None] < n), other = 0.0)
        C_tile = tl.dot(A_rows, B_cols, acc = C_tile)
    tl.store(C + offsets_i * n + offsets_j, C_tile, mask = (offsets_i < n) & (offsets_j < n))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, P: int):
    if P == 1:
        output.copy_(input)
        return
    tmp1 = torch.zeros_like(input)
    solve(input,tmp1, N, P//2)
    BLOCK_SIZE_W = 64
    BLOCK_SIZE_H = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(N, BLOCK_SIZE_W), triton.cdiv(N, BLOCK_SIZE_H))
    if P % 2 == 0:
        matmul_kernel[grid](tmp1, tmp1, output, N, BLOCK_SIZE_W, BLOCK_SIZE_H, BLOCK_SIZE_K)
    else:
        tmp2 = torch.zeros_like(input)
        matmul_kernel[grid](tmp1, tmp1, tmp2, N, BLOCK_SIZE_W, BLOCK_SIZE_H, BLOCK_SIZE_K)
        matmul_kernel[grid](tmp2, input, output, N, BLOCK_SIZE_W, BLOCK_SIZE_H, BLOCK_SIZE_K)
