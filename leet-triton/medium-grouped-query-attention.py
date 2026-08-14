import torch
import triton
import triton.language as tl
import math

@triton.jit
def _gqa_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_oh, stride_os, stride_od,
    seq_len, head_dim,
    scale, groups,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    start_m = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    kv_head_idx = q_head_idx // groups

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < seq_len
    mask_d = offs_d < head_dim

    q_ptrs = q_ptr + q_head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seq_len

        k_ptrs = k_ptr + kv_head_idx * stride_kh + offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

        v_ptrs = v_ptr + kv_head_idx * stride_vh + offs_n_curr[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float('-inf'))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        l_i_new = alpha * l_i + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_i_new
        l_i = l_i_new

    acc = acc / l_i[:, None]

    o_ptrs = o_ptr + q_head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=(mask_m[:, None] & mask_d[None, :]))


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    BLOCK_D = triton.next_power_of_2(max(16, head_dim))

    # Restrict block sizes explicitly to pass the strict 64KB shared-memory limit.
    if BLOCK_D >= 256:
        BLOCK_M, BLOCK_N = 16, 16
    elif BLOCK_D >= 128:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 64, 64

    grid = (triton.cdiv(seq_len, BLOCK_M), num_q_heads)
    scale = 1.0 / math.sqrt(head_dim)
    groups = num_q_heads // num_kv_heads

    return _gqa_kernel[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        seq_len, head_dim,
        scale, groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )


if __name__ == "__main__":
    import sys

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0x60A)
    cases = [
        (4, 2, 33, 16),
        (8, 2, 129, 32),
    ]
    for num_q_heads, num_kv_heads, seq_len, head_dim in cases:
        Q = torch.randn(
            num_q_heads, seq_len, head_dim, device=device,
            dtype=torch.float32).contiguous()
        K = torch.randn(
            num_kv_heads, seq_len, head_dim, device=device,
            dtype=torch.float32).contiguous()
        V = torch.randn(
            num_kv_heads, seq_len, head_dim, device=device,
            dtype=torch.float32).contiguous()
        output = torch.empty_like(Q)

        solve(
            Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim)
        if device == "mps":
            torch.mps.synchronize()

        groups = num_q_heads // num_kv_heads
        expected = torch.nn.functional.scaled_dot_product_attention(
            Q.cpu(),
            K.cpu().repeat_interleave(groups, dim=0),
            V.cpu().repeat_interleave(groups, dim=0),
        )
        torch.testing.assert_close(
            output.cpu(), expected, atol=1e-3, rtol=1e-3)

    print(f"PASS [{device}] grouped_query_attention ({len(cases)} cases)")
