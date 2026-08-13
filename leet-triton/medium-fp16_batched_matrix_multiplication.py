import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BATCH,M,N,K,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    a_ptr += pid_b * M * K
    b_ptr += pid_b * K * N
    c_ptr += pid_b * M * N

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype = tl.float32)

    for ks in range(0, K, BLOCK_SIZE):
        offs_k = ks + tl.arange(0, BLOCK_SIZE)

        offs_a = offs_m[:,None] * K + offs_k[None,:]
        mask_a = (offs_m[:,None] < M) & (offs_k[None,:] < K)
        a = tl.load(a_ptr + offs_a, mask = mask_a, other = 0.0).to(tl.float32)

        offs_b = offs_k[:, None] * N + offs_n[None,:]
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + offs_b, mask = mask_b, other = 0.0).to(tl.float32)

        acc += tl.dot(a,b)
    
    offs_c = offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None,:] < N)
    tl.store(c_ptr + offs_c, acc.to(tl.float16), mask = mask_c)


def solve(a: torch.Tensor, b: torch.Tensor, c:torch.Tensor, BATCH: int, M:int, N:int, K:int):
    BLOCK_SIZE = 64
    grid = (
        BATCH,
        triton.cdiv(M, BLOCK_SIZE),
        triton.cdiv(N, BLOCK_SIZE),
    )
    kernel[grid](
        a, b, c,
        BATCH, M, N, K,
        BLOCK_SIZE = BLOCK_SIZE,
    )


