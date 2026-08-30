import torch
import triton
import triton.language as tl

@triton.jit
def dequant_kernel(x_ptr, s_ptr, y_ptr, m ,n, tile_size, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offs_xy = offs_m[:, None] * n + offs_n[None, :]
    mask_xy = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    x = tl.load(x_ptr + offs_xy, mask = mask_xy, other = 0.0)

    offs_s = offs_m[:, None] // tile_size * tl.cdiv(n, tile_size) + offs_n[None, :] // tile_size
    s = tl.load(s_ptr + offs_s, mask = mask_xy, other = 0.0)

    y = x * s
    tl.store(y_ptr + offs_xy, y , mask = mask_xy)

# X, S, Y are tensors on the GPU
def solve(X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int):
    BLOCK_SIZE = 32
    grid = (
        triton.cdiv(M, BLOCK_SIZE),
        triton.cdiv(N, BLOCK_SIZE)
    )
    dequant_kernel[grid](X, S, Y, M, N, TILE_SIZE, BLOCK_SIZE)
