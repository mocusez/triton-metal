import torch
import triton
import triton.language as tl
import math

@triton.jit
def alibi_attention_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    M, N, d,
    alpha, scale,
    stride_qm, stride_qd,
    stride_kd, stride_kn,
    stride_vm, stride_vd,
    stride_om, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    m_mask = off_m < M
    d_mask = off_d < d

    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype = tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype = tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_SIZE_N):
        off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        n_mask = off_n < N

        qk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
        for start_dd in range(0, d, BLOCK_SIZE_D):
            off_dd = start_dd + tl.arange(0, BLOCK_SIZE_D)
            dd_mask = off_dd < d

            q_tile = tl.load(
                q_ptr + off_m[:, None] * stride_qm + off_dd[None, :] * stride_qd,
                mask = m_mask[:, None] & dd_mask[None, :], other = 0.0
            )

            k_tile = tl.load(
                k_ptr + off_dd[:, None] * stride_kd + off_n[None, :] * stride_kn,
                mask = dd_mask[:, None] & n_mask[None, :], other = 0.0
            )
            qk = tl.dot(q_tile, k_tile, acc = qk, allow_tf32= False)

        qk = qk * scale
        qk = qk + alpha * (off_m[:, None] - off_n[None, :]).to(tl.float32)
        qk = tl.where(m_mask[:, None] & n_mask[None,:], qk, float("-inf"))

        m_curr = tl.max(qk, axis = 1)
        m_next = tl.maximum(m_i, m_curr)
        rescale = tl.exp(m_i - m_next)
        p = tl.exp(qk - m_next[:,None])
        p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)

        l_i = l_i * rescale + tl.sum(p, axis = 1)

        v_tile = tl.load(
            v_ptr + off_n[:, None] * stride_vm + off_d[None, :] * stride_vd,
            mask = n_mask[:, None] & d_mask[None, :], other = 0.0
        )
        acc = acc * rescale[:,None] + tl.dot(p.to(tl.float32), v_tile, allow_tf32=False)
        m_i = m_next

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    tl.store(
        o_ptr + off_m[:, None] * stride_om + off_d[None, :] * stride_od,
        acc,
        mask = m_mask[:, None] & d_mask[None, :]
    )


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    alpha: float,
):
    K_t = K.T.contiguous()

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_D = 64
    scale = 1.0 / math.sqrt(d)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(d, BLOCK_SIZE_D))

    alibi_attention_fwd[grid](
        Q, K_t, V, output,
        M, N, d, alpha, scale,
        Q.stride(0), Q.stride(1),
        K_t.stride(0), K_t.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
