import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    Q,
    cos,
    sin,
    output,
    M,
    D,
    stride_qm,
    stride_qd,
    stride_cm,
    stride_cd,
    stride_sm,
    stride_sd,
    stride_om,
    stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d1 = tl.arange(0, BLOCK_SIZE_D)
    offs_d2 = tl.arange(0, BLOCK_SIZE_D) + D // 2
    ptrs_q1 = Q + offs_m[:, None] * stride_qm + offs_d1[None,:] * stride_qd
    ptrs_q2 = Q + offs_m[:, None] * stride_qm + offs_d2[None,:] * stride_qd
    ptrs_c1 = cos + offs_m[:, None] * stride_cm + offs_d1[None, :] * stride_cd
    ptrs_c2 = cos + offs_m[:, None] * stride_cm + offs_d2[None, :] * stride_cd
    ptrs_s1 = sin + offs_m[:, None] * stride_sm + offs_d1[None, :] * stride_sd
    ptrs_s2 = sin + offs_m[:, None] * stride_sm + offs_d2[None, :] * stride_sd
    mask1 = (offs_m[:, None] < M) & (offs_d1[None, :] < D // 2)
    mask2 = (offs_m[:, None] < M) & (offs_d2[None, :] < D)
    q1 = tl.load(ptrs_q1, mask = mask1, other = 0.0)
    q2 = tl.load(ptrs_q2, mask = mask2, other = 0.0)
    cos1 = tl.load(ptrs_c1, mask = mask1, other = 0.0)
    cos2 = tl.load(ptrs_c2, mask = mask2, other = 0.0)
    sin1 = tl.load(ptrs_s1, mask = mask1, other = 0.0)
    sin2 = tl.load(ptrs_s2, mask = mask2, other = 0.0)
    out1 = q1 * cos1 - q2 * sin1
    out2 = q2 * cos2 + q1 * sin2

    ptrs_out1 = output + offs_m[:, None] * stride_om + offs_d1[None, :] * stride_od
    ptrs_out2 = output + offs_m[:, None] * stride_om + offs_d2[None, :] * stride_od
    tl.store(ptrs_out1, out1, mask = mask1)
    tl.store(ptrs_out2, out2, mask = mask2)


# Q, cos, sin, output are tensors on the GPU
def solve(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor, M: int, D: int
):
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_D = triton.next_power_of_2(D // 2)
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    rope_kernel[grid](
        Q,
        cos,
        sin,
        output,
        M,
        D,
        Q.stride(0),
        Q.stride(1),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_D,
    )
