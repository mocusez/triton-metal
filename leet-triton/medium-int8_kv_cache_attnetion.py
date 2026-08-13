import math
import torch
import triton
import triton.language as tl

@triton.jit
def _decode_attention_int8_kv_kernel(
    Q_ptr, K_int8_ptr, V_int8_ptr, k_scale_ptr, v_scale_ptr, Out_ptr,
    stride_qh, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_k_scale_h, stride_k_scale_s,
    stride_v_scale_h, stride_v_scale_s,
    stride_oh, stride_od,
    seq_len, head_dim,
    sm_scale,
    BLOCK_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    # 每个 Block 负责一个 Attention Head
    head_idx = tl.program_id(0)

    # 定位当前 Head 的内存基地址
    Q_head_ptr = Q_ptr + head_idx * stride_qh
    K_head_ptr = K_int8_ptr + head_idx * stride_kh
    V_head_ptr = V_int8_ptr + head_idx * stride_vh
    k_scale_head_ptr = k_scale_ptr + head_idx * stride_k_scale_h
    v_scale_head_ptr = v_scale_ptr + head_idx * stride_v_scale_h
    Out_head_ptr = Out_ptr + head_idx * stride_oh

    # Head 维度的偏移与掩码 (处理非 2 的幂次)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_d = offs_d < head_dim

    # 加载 Q 向量: [HEAD_DIM]
    q = tl.load(Q_head_ptr + offs_d * stride_qd, mask=mask_d, other=0.0)

    # Online Softmax 状态初始化
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # 沿 Sequence 维度进行分块计算
    for start_s in range(0, seq_len, BLOCK_SEQ):
        offs_s = start_s + tl.arange(0, BLOCK_SEQ)
        mask_s = offs_s < seq_len
        
        # KV 加载的 2D Mask
        mask_kv = mask_s[:, None] & mask_d[None, :]

        # -----------------------------------------------------------
        # 1. 计算 Attention Scores (Q @ K^T)
        # -----------------------------------------------------------
        k_offs = offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_int8 = tl.load(K_head_ptr + k_offs, mask=mask_kv, other=0.0)
        
        k_scale_offs = offs_s * stride_k_scale_s
        k_scale = tl.load(k_scale_head_ptr + k_scale_offs, mask=mask_s, other=0.0)

        # 反量化 K: K_float = K_int8 * k_scale
        k_float = k_int8.to(tl.float32) * k_scale[:, None]

        # 内积计算与缩放: qk = (q @ K^T) / sqrt(d)
        qk = tl.sum(q[None, :] * k_float, axis=1) * sm_scale
        qk = tl.where(mask_s, qk, -float('inf'))

        # -----------------------------------------------------------
        # 2. Online Softmax 状态更新
        # -----------------------------------------------------------
        m_ij = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new)
        l_new = l_i * alpha + tl.sum(beta, axis=0)

        # -----------------------------------------------------------
        # 3. 计算 Context Value (Softmax @ V)
        # -----------------------------------------------------------
        v_offs = offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_int8 = tl.load(V_head_ptr + v_offs, mask=mask_kv, other=0.0)
        
        v_scale_offs = offs_s * stride_v_scale_s
        v_scale = tl.load(v_scale_head_ptr + v_scale_offs, mask=mask_s, other=0.0)

        # 反量化 V: V_float = V_int8 * v_scale
        v_float = v_int8.to(tl.float32) * v_scale[:, None]

        # 累加 Attention 结果
        acc = acc * alpha + tl.sum(beta[:, None] * v_float, axis=0)

        # 更新下一轮迭代的 Softmax 状态
        m_i = m_new
        l_i = l_new

    # Context 归一化
    acc = acc / l_i

    # 写回输出缓冲
    tl.store(Out_head_ptr + offs_d * stride_od, acc, mask=mask_d)


# Q, K_int8, V_int8, k_scale, v_scale, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K_int8: torch.Tensor,
    V_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    # 确保 Triton Kernel 能拿到 2 的幂次作为编译常量
    HEAD_DIM_POW2 = triton.next_power_of_2(head_dim)
    
    # Grid: 每个 Attention Head 分配一个独立的程序
    grid = (num_heads,)
    
    # 缩放因子
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # 调优参数: Sequence 分块大小 (对于较长的 seq_len，可调大至 64 或 128 以充分利用 SRAM)
    BLOCK_SEQ = 32

    # 启动内核
    _decode_attention_int8_kv_kernel[grid](
        Q, K_int8, V_int8, k_scale, v_scale, output,
        # 各个 Tensor 的 Strides (直接通过 torch Tensor 方法获取，安全且鲁棒)
        Q.stride(0), Q.stride(1),
        K_int8.stride(0), K_int8.stride(1), K_int8.stride(2),
        V_int8.stride(0), V_int8.stride(1), V_int8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        v_scale.stride(0), v_scale.stride(1),
        output.stride(0), output.stride(1),
        # 形状与缩放
        seq_len, head_dim,
        sm_scale,
        # 常量参数
        BLOCK_SEQ=BLOCK_SEQ,
        HEAD_DIM=HEAD_DIM_POW2
    )
