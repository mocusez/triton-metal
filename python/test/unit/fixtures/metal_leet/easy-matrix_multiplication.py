import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for n in range(0, N, BLOCK_K):
        offs_n = n + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        a_sub = tl.load(a_ptrs, mask = a_mask, other = 0.0)
        b_sub = tl.load(b_ptrs, mask = b_mask, other = 0.0)

        acc += tl.dot(a_sub, b_sub)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck)
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )
