import torch
import triton
import triton.language as tl

@triton.jit
def lora_kernel(
    input,
    W,
    A,
    B,
    output,
    M,
    N,
    K,
    R,
    scale,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_ar,
    stride_ak,
    stride_bn,
    stride_br,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid // num_n_blocks, pid % num_n_blocks,
                                    num_m_blocks, num_n_blocks, GROUP_SIZE)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_r = tl.arange(0, BLOCK_SIZE_R)

    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_R), dtype = tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        mask_a = (offs_r[:, None] < R) & (offs_k[None, :] < K)
        x = tl.load(input + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik, mask = mask_x, other = 0.0)
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask = mask_w, other = 0.0)
        a = tl.load(A + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak, mask = mask_a, other = 0.0)
        acc0 = tl.dot(x, tl.trans(w), acc0, allow_tf32 = False)
        acc1 = tl.dot(x, tl.trans(a), acc1, allow_tf32 = False)
    mask_b = (offs_n[:, None] < N) & (offs_r[None, :] < R)
    b = tl.load(B + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br, mask = mask_b, other = 0.0)
    acc0 += scale * tl.dot(acc1, tl.trans(b), allow_tf32 = False)

    mask_y = (offs_m[:, None] < M) * (offs_n[None, :] < N)
    tl.store(output + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc0, mask = mask_y)

# x, W, A, B, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    d_in: int,
    d_out: int,
    rank: int,
    lora_scale: float,
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_R = max(16, triton.next_power_of_2(rank))
    GROUP_SIZE = 4
    grid = (triton.cdiv(batch, BLOCK_SIZE_M) * triton.cdiv(d_out, BLOCK_SIZE_N), )
    lora_kernel[grid](
        x,
        W,
        A,
        B,
        output,
        batch,
        d_out,
        d_in,
        rank,
        lora_scale,
        x.stride(0),
        x.stride(1),
        W.stride(0),
        W.stride(1),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_R,
        GROUP_SIZE
    )

