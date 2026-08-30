import torch
import triton
import triton.language as tl


@triton.jit
def selective_scan_fwd_kernel(
    # 张量指针
    u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, skip_ptr, y_ptr,
    # 维度
    batch, seq_len, d_model, d_state,
    # 各张量 stride（元素粒度）
    u_stride_b, u_stride_t, u_stride_d,
    delta_stride_b, delta_stride_t, delta_stride_d,
    A_stride_d, A_stride_n,
    B_stride_b, B_stride_t, B_stride_n,
    C_stride_b, C_stride_t, C_stride_n,
    skip_stride_d,
    y_stride_b, y_stride_t, y_stride_d,
    # 编译时常量
    BLOCK_DSTATE: tl.constexpr,
):
    """
    每个 block 负责一个独立的 (batch, d_model) 通道。
    block 内线程在 d_state 维度上向量化，seq_len 维度做顺序扫描。
    """
    # 当前 block 对应的 (batch, d_model)
    pid = tl.program_id(0)
    b = pid // d_model
    d = pid % d_model

    # block 内线程索引
    tid = tl.arange(0, BLOCK_DSTATE)
    mask = tid < d_state

    # 一次性加载 A[d, :] 与 skip[d]（标量）
    A_d = tl.load(A_ptr + d * A_stride_d + tid * A_stride_n, mask=mask, other=0.0)
    skip_d = tl.load(skip_ptr + d * skip_stride_d)

    # 初始隐藏状态 h = 0
    h = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)

    # 在 seq_len 上顺序扫描（递推依赖，无法并行）
    for t in range(seq_len):
        # 加载标量 u[b,t,d] 与 delta[b,t,d]
        u_btd = tl.load(u_ptr + b * u_stride_b + t * u_stride_t + d * u_stride_d)
        delta_btd = tl.load(delta_ptr + b * delta_stride_b + t * delta_stride_t + d * delta_stride_d)

        # A_bar = exp(delta * A)
        A_bar = tl.exp(delta_btd * A_d)

        # 加载 B[b,t,:] 与 C[b,t,:]
        B_bt = tl.load(B_ptr + b * B_stride_b + t * B_stride_t + tid * B_stride_n, mask=mask, other=0.0)
        C_bt = tl.load(C_ptr + b * C_stride_b + t * C_stride_t + tid * C_stride_n, mask=mask, other=0.0)

        # B_bar = delta * B
        B_bar = delta_btd * B_bt

        # 更新隐藏状态: h = A_bar * h + B_bar * u
        h = A_bar * h + B_bar * u_btd

        # y = sum(C * h) + skip * u
        # tl.sum 执行 block-level reduce；tl.where 保证超出 d_state 部分贡献为 0
        y_val = tl.sum(tl.where(mask, C_bt * h, 0.0)) + skip_d * u_btd

        # 写回 y[b,t,d] —— y_val 是标量（reduce 结果），所有线程值相同
        tl.store(y_ptr + b * y_stride_b + t * y_stride_t + d * y_stride_d, y_val)


def solve(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    skip: torch.Tensor,
    y: torch.Tensor,
    batch: int,
    seq_len: int,
    d_model: int,
    d_state: int,
):
    """
    SSM Selective Scan 前向传播 (Triton GPU Kernel)
    直接写入预分配的 y 张量，无需额外内存分配。
    """
    # 根据实际 d_state 选择最小 2 的幂作为 block 大小（上限 64），以最大化 warp 利用率
    BLOCK_DSTATE = 1
    while BLOCK_DSTATE < d_state:
        BLOCK_DSTATE *= 2
    BLOCK_DSTATE = min(BLOCK_DSTATE, 64)

    # warp 数量：BLOCK_DSTATE / 32 向上取整，至少 1 个 warp
    num_warps = max(1, (BLOCK_DSTATE + 31) // 32)

    # grid: 每个 block 一个 (batch, d_model) 对，完全并行
    grid = (batch * d_model,)

    # 启动 kernel
    selective_scan_fwd_kernel[grid](
        u, delta, A, B, C, skip, y,
        batch, seq_len, d_model, d_state,
        u.stride(0), u.stride(1), u.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        skip.stride(0),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=num_warps,
    )
