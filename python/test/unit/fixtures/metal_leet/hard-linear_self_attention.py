import torch
import triton
import triton.language as tl

@triton.jit
def matmulKV_kernel(
    Kt_ptr, V_ptr, KV_ptr, Ksum_ptr,
    M, D,
    BLOCK_M: tl.constexpr,
    NUM_ITER: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    off_D = tl.arange(0, BLOCK_D)
    mask_D = off_D < D

    acc_KV = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    acc_K = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for i in range(NUM_ITER):
        off_M = pid * BLOCK_M * NUM_ITER + i * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_M = off_M < M

        K_DM = tl.load(
            Kt_ptr + off_D[:, None] * M + off_M[None, :],
            mask=mask_D[:, None] & mask_M[None, :],
            other=float('-inf')
        )
        K_DM = tl.where(K_DM > 0, K_DM + 1, tl.exp(K_DM))

        V_MD = tl.load(
            V_ptr + off_M[:, None] * D + off_D[None, :],
            mask=mask_M[:, None] & mask_D[None, :],
            other=0.0
        )
        acc_KV = tl.dot(K_DM, V_MD, acc=acc_KV, allow_tf32=False)
        acc_K = acc_K + tl.sum(K_DM, axis=1)

    tl.atomic_add(
        KV_ptr + off_D[:, None] * D + off_D[None, :],
        acc_KV,
        mask=mask_D[:, None] & mask_D[None, :]
    )
    tl.atomic_add(Ksum_ptr + off_D, acc_K, mask=mask_D)

@triton.jit
def linear_attn_kernel(
    Q_ptr, KV_ptr, Ksum_ptr, out_ptr,
    M, D,
    BLOCK_M: tl.constexpr,
    NUM_ITER: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_d: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    off_D = tl.arange(0, BLOCK_D)
    off_d = pid0 * BLOCK_d + tl.arange(0, BLOCK_d)
    mask_D = off_D < D
    mask_d = off_d < D

    KV_Dd = tl.load(
        KV_ptr + off_D[:, None] * D + off_d[None, :],
        mask=mask_D[:, None] & mask_d[None, :], other=0.0
    )
    Ksum_D = tl.load(Ksum_ptr + off_D, mask=mask_D, other=0.0)
    EPS = tl.full((BLOCK_M,), 1e-5, dtype=tl.float32)

    for i in range(NUM_ITER):
        off_M  = (pid1 * NUM_ITER + i) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_M = off_M < M

        Q_MD = tl.load(
            Q_ptr + off_M[:, None] * D + off_D[None, :],
            mask=mask_M[:, None] & mask_D[None, :], other=float('-inf')
        )
        Q_MD = tl.where(Q_MD > 0, Q_MD + 1, tl.exp(Q_MD))

        numer = tl.dot(Q_MD, KV_Dd, allow_tf32=False)
        denom = tl.sum(Q_MD * Ksum_D[None, :], axis=1) + EPS

        tl.store(
            out_ptr + off_M[:, None] * D + off_d[None, :],
            numer / denom[:, None],
            mask=mask_M[:, None] & mask_d[None, :]
        )

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, D: int):

    BLOCK_D = max(16, triton.next_power_of_2(D))

    K_t = K.T.contiguous()

    KV_DD = torch.zeros((D, D), dtype=torch.float32, device=Q.device)
    K_D = torch.zeros((D,), dtype=torch.float32, device=Q.device)

    BLOCK_M = 64
    NUM_ITER = 4
    grid1 = (triton.cdiv(M, BLOCK_M * NUM_ITER),)
    matmulKV_kernel[grid1](
        K_t, V, KV_DD, K_D, M, D,
        BLOCK_M=BLOCK_M, NUM_ITER=NUM_ITER, BLOCK_D=BLOCK_D,
        num_warps=16,
    )

    BLOCK_M = 32
    BLOCK_d = 64
    NUM_ITER = 2
    grid2 = (triton.cdiv(D, BLOCK_d), triton.cdiv(M, BLOCK_M * NUM_ITER))
    linear_attn_kernel[grid2](
        Q, KV_DD, K_D, output, M, D,
        BLOCK_M=BLOCK_M, NUM_ITER=NUM_ITER,
        BLOCK_D=BLOCK_D, BLOCK_d=BLOCK_d,
        num_warps=8
    )
