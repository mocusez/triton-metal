import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q,K,V, Out,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_om, stride_od,
    M, N, d,
    sm_scale,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.float32))
    K = K.to(tl.pointer_type(tl.float32))
    V = V.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    q_ptrs = Q + pid * stride_qm + offs_d * stride_qd
    mask_q = offs_d < d
    q = tl.load(q_ptrs, mask = mask_q, other=0.0)
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_SIZE_N):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        k_ptrs = K+ offs_n[:,None] * stride_kn + offs_d[None,:] * stride_kd
        k_mask = (offs_n[:,None] < N) & (offs_d[None,:] < d)
        k = tl.load(k_ptrs, mask=k_mask,other=0.0)
        qk = tl.sum(q[None,:]*k, axis=1)
        qk *= sm_scale
        qk = tl.where(offs_n < N,qk,-float('inf'))
        m_prev = m_i
        block_max = tl.max(qk,axis=0)
        m_i = tl.maximum(m_prev, block_max)
        alpha = tl.exp(m_prev - m_i)
        p = tl.exp(qk - m_i)
        l_i = l_i * alpha + tl.sum(p,axis=0)

        v_ptrs = V + offs_n[:,None] * stride_vn + offs_d[None,:] * stride_vd
        v_mask = (offs_n[:,None] < N) & (offs_d[None,:] < d)
        v = tl.load(v_ptrs, mask=v_mask,other=0.0)
        acc = acc * alpha + tl.sum(p[:,None] * v, axis=0)

    acc = acc/l_i
    out_otr = Out + pid * stride_om + offs_d * stride_od
    tl.store(out_otr,acc,mask = mask_q)

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    sm_scale = d ** -0.5
    grid = (M,)
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = triton.next_power_of_2(d)
    attention_kernel[grid](
        Q,K,V,output,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        M, N, d,
        sm_scale,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_D = BLOCK_SIZE_D,
        num_warps = 4,
        num_stages = 2
    )
