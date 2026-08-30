import torch
import triton
import triton.language as tl

@triton.jit
def mmx_kernel(
    a,b,c, BATCH, M,N,K,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K:tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = pid_b * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_b = offs_b < BATCH

    offs_m64 = offs_m.to(tl.int64)
    offs_n64 = offs_n.to(tl.int64)
    offs_b64 = offs_b.to(tl.int64)
    M64 = tl.full((), M, tl.int64)
    N64 = tl.full((), N, tl.int64)
    K64 = tl.full((), K, tl.int64)

    acc = tl.zeros((BLOCK_BATCH, BLOCK_M, BLOCK_N), dtype = tl.float32)

    for i in range(0, K, BLOCK_K):
        offs_k = i + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        offs_k64 = offs_k.to(tl.int64)

        a_ptrs = (
            a
            + offs_b64[:, None, None] * (M64 * K64)
            + offs_m64[None, : , None] * K64
            + offs_k64[None, None, :]
        )
        a_mask = mask_b[:, None, None] & mask_m[None, : , None] & mask_k[None, None, :]
        tile_a = tl.load(a_ptrs, mask = a_mask, other = 0.0)

        b_ptrs = (
            b
            + offs_b64[:, None, None] * (K64 * N64)
            + offs_k64[None,:,None] * N64
            + offs_n64[None, None, :]
        )
        b_mask = mask_b[:,None,None] & mask_k[None,:,None] & mask_n[None, None,:]
        tile_b = tl.load(b_ptrs, mask = b_mask, other = 0.0)

        acc = tl.dot(tile_a, tile_b, acc=acc, input_precision="ieee")

    c_ptrs = (
        c
        + offs_b64[:, None, None] * (M64 * N64)
        + offs_m64[None, :, None] * N64
        + offs_n64[None, None, :]
    )
    c_mask = mask_b[:, None, None] & mask_m[None, :,None] & mask_n[None, None,:]
    tl.store(c_ptrs, acc, mask=c_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()

    BLOCK_BATCH = 1
    BLOCK_M = BLOCK_N = BLOCK_K = 64

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(BATCH, BLOCK_BATCH),
    )
    mmx_kernel[grid](
        a, b, c,
        BATCH, M, N, K,
        BLOCK_BATCH,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps = 4, num_stages = 2,
    )
