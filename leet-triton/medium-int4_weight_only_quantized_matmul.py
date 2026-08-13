import torch
import triton
import triton.language as tl


@triton.jit
def int4_matmul_kernel(
    x,
    wq,
    scales,
    y,
    M,
    N,
    K,
    group_size: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wqn,
    stride_wqk,
    stride_sn,
    stride_sk,
    stride_ym,
    stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_x1k = k + tl.arange(0, BLOCK_SIZE_K // 2) * 2;
        mask_x1 = (offs_m[:, None] < M) & (offs_x1k[None, :] < K)
        tile_x1 = tl.load(
            x + offs_m[:, None] * stride_xm + offs_x1k[None, :] * stride_xk, mask = mask_x1, other = 0.0
        ).to(tl.float32)

        offs_x2k = k + tl.arange(0, BLOCK_SIZE_K // 2) * 2 + 1
        mask_x2 = (offs_m[:, None] < M) & (offs_x2k[None, :] < K)
        tile_x2 = tl.load(
            x + offs_m[:, None] * stride_xm + offs_x2k[None, :] * stride_xk, mask = mask_x2, other = 0.0
        ).to(tl.float32)

        offs_sk = (k // group_size) + tl.arange(0, BLOCK_SIZE_K // group_size)
        mask_s = (offs_n[:, None] < N) & (offs_sk[None, :] < (K // group_size))
        tile_s = tl.load(
            scales + offs_n[:, None] * stride_sn + offs_sk[None, :] * stride_sk,
            mask = mask_s,
            other = 0.0,
        ).to(tl.float32)

        tile_s = tl.broadcast_to(
            tile_s[:,:,None], (BLOCK_SIZE_N, BLOCK_SIZE_K // group_size, group_size // 2)
        )
        tile_s = tl.reshape(tile_s, (BLOCK_SIZE_N, BLOCK_SIZE_K // 2))

        offs_wk = (k // 2) + tl.arange(0, BLOCK_SIZE_K // 2)
        mask_w = (offs_n[:, None] < N) & (offs_wk[None, :] < (K // 2))
        tile_wq = tl.load(
            wq + offs_n[:, None] * stride_wqn + offs_wk[None, :] * stride_wqk,
            mask = mask_w,
            other = 0x88,
        )
        tile_w1 = (((tile_wq & 0xF0) >> 4).to(tl.float32) - 8.0) * tile_s
        tile_w2 = ((tile_wq & 0x0F).to(tl.float32) - 8.0) * tile_s

        acc = tl.dot(tile_x1, tl.trans(tile_w1), acc=acc, input_precision="ieee")
        acc = tl.dot(tile_x2, tl.trans(tile_w2), acc=acc, input_precision="ieee")
    
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(tl.float16),
        mask = mask_y,
    )


# x, w_q, scales, y are tensors on the GPU
def solve(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int,
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = max(32, group_size)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    int4_matmul_kernel[grid](
        x,
        w_q,
        scales,
        y,
        M,
        N,
        K,
        group_size,
        x.stride(0),
        x.stride(1),
        w_q.stride(0),
        w_q.stride(1),
        scales.stride(0),
        scales.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
