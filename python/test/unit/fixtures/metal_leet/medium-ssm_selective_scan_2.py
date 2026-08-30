import torch
import triton
import triton.language as tl

@triton.jit
def _ssm_selective_scan_kernel(
    u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, skip_ptr, y_ptr,
    batch, seq_len, d_model, d_state,
    stride_u_b, stride_u_t, stride_u_d,
    stride_delta_b, stride_delta_t, stride_delta_d,
    stride_A_d, stride_A_n,
    stride_B_b, stride_B_t, stride_B_n,
    stride_C_b, stride_C_t, stride_C_n,
    stride_skip_d,
    stride_y_b, stride_y_t, stride_y_d,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 获取当前 Program 的 Batch 索引和 d_model 维度块索引
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    # 构造 d_model 维度的偏移量与掩码
    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < d_model

    # 构造 d_state 维度的偏移量与掩码
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < d_state

    # 在时间循环外部提前加载静态的 A 矩阵与 skip 向量（高重用率）
    A_ptr_block = A_ptr + d_offsets[:, None] * stride_A_d + n_offsets[None, :] * stride_A_n
    A_mask = d_mask[:, None] & n_mask[None, :]
    A_shared = tl.load(A_ptr_block, mask=A_mask, other=0.0)

    skip_ptr_block = skip_ptr + d_offsets * stride_skip_d
    skip_shared = tl.load(skip_ptr_block, mask=d_mask, other=0.0)

    # 初始化隐藏状态 h 为全 0，形状为 (BLOCK_SIZE_D, BLOCK_SIZE_N)
    h = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_N), dtype=tl.float32)

    # 预先计算基础指针，减少循环内部的标量乘加开销
    u_base = u_ptr + pid_b * stride_u_b + d_offsets * stride_u_d
    delta_base = delta_ptr + pid_b * stride_delta_b + d_offsets * stride_delta_d
    B_base = B_ptr + pid_b * stride_B_b + n_offsets * stride_B_n
    C_base = C_ptr + pid_b * stride_C_b + n_offsets * stride_C_n
    y_base = y_ptr + pid_b * stride_y_b + d_offsets * stride_y_d

    # 顺着序列长度进行串行扫描更新
    for t in range(0, seq_len):
        # 顺次加载当前时间步 t 的输入
        delta_shared = tl.load(delta_base + t * stride_delta_t, mask=d_mask, other=0.0)
        u_shared = tl.load(u_base + t * stride_u_t, mask=d_mask, other=0.0)
        B_shared = tl.load(B_base + t * stride_B_t, mask=n_mask, other=0.0)
        C_shared = tl.load(C_base + t * stride_C_t, mask=n_mask, other=0.0)

        # 将一维向量显式扩展为二维张量用于高效的广播操作
        delta_2d = tl.expand_dims(delta_shared, 1) # (BLOCK_SIZE_D, 1)
        u_2d = tl.expand_dims(u_shared, 1)         # (BLOCK_SIZE_D, 1)
        B_2d = tl.expand_dims(B_shared, 0)         # (1, BLOCK_SIZE_N)
        C_2d = tl.expand_dims(C_shared, 0)         # (1, BLOCK_SIZE_N)

        # 离散化计算：A_bar = exp(delta * A)
        A_bar = tl.exp(delta_2d * A_shared)

        # 乘法结合律优化：将 (delta * B) * u 优化为 (delta * u) * B
        delta_u = delta_2d * u_2d                  # (BLOCK_SIZE_D, 1)

        # 隐状态状态转移更新
        h = A_bar * h + delta_u * B_2d

        # 计算输出值并加上残差连接：y = sum(C * h) + skip * u
        y_val = tl.sum(C_2d * h, axis=1) + skip_shared * u_shared

        # 写回全局内存
        tl.store(y_base + t * stride_y_t, y_val, mask=d_mask)

def solve(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state):
    # T4 优化的分块超参数配置
    BLOCK_SIZE_D = 32
    BLOCK_SIZE_N = 64 # 确保大于等于所有的 d_state 限制范围(<=64)

    # 网格大小定义
    grid = (batch, triton.cdiv(d_model, BLOCK_SIZE_D))

    # 调用 Triton 核函数
    _ssm_selective_scan_kernel[grid](
        u, delta, A, B, C, skip, y,
        batch, seq_len, d_model, d_state,
        u.stride(0), u.stride(1), u.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        skip.stride(0),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4
    )
    return y
