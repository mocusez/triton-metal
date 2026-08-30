import math
import torch
import triton
import triton.language as tl


@triton.jit
def _decay_causal_attn_fwd(
    Q, K, V, Out,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    seq_len, d_model,
    log2_gamma, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query 行 (n)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len
    d_mask = offs_d < d_model
    qd_mask = m_mask[:, None] & d_mask[None, :]

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=qd_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # 因果性：只需遍历到本 block 最后一行 query 对应的 key
    hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)       # key 行 (m)
        kv_mask = (offs_n < seq_len)[:, None] & d_mask[None, :]

        k = tl.load(K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                    mask=kv_mask, other=0.0)
        # scores = QK^T / sqrt(d)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

        # 因果衰减掩码：n>=m 时 γ^(n-m)，否则精确为 0
        diff = offs_m[:, None] - offs_n[None, :]
        exponent = tl.where(diff >= 0, diff * log2_gamma, float("-inf"))
        s = s * tl.exp2(exponent)                      # exp2(-inf) == 0

        v = tl.load(V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                    mask=kv_mask, other=0.0)
        acc = tl.dot(s, v, input_precision="ieee", acc=acc)

    tl.store(Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc, mask=qd_mask)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
    d_model: int,
    gamma: float,
):
    # tl.dot 要求各维 >= 16，特征维向上取 2 的幂并用掩码兜底
    BLOCK_D = max(16, triton.next_power_of_2(d_model))

    # 按 d_model 选择 tile，保证共享内存远低于 T4 的 64KB 上限
    if BLOCK_D <= 64:
        BLOCK_M, BLOCK_N, num_warps = 64, 64, 4
    elif BLOCK_D <= 128:
        BLOCK_M, BLOCK_N, num_warps = 32, 32, 4
    else:  # BLOCK_D == 256
        BLOCK_M, BLOCK_N, num_warps = 16, 16, 4

    # 约束保证 0 < gamma <= 1；防御性钳位使 log2 始终有限
    log2_gamma = math.log2(min(max(float(gamma), 1e-300), 1.0))
    scale = 1.0 / math.sqrt(d_model)

    grid = (triton.cdiv(seq_len, BLOCK_M),)
    _decay_causal_attn_fwd[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        seq_len, d_model,
        log2_gamma, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=1,
    )
    return output
