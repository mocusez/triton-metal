import torch
import triton
import triton.language as tl


@triton.jit
def _attention_sinks_kernel(
    Q,
    K,
    V,
    output,
    M,
    d,
    num_sinks,
    window_size,
    sm_scale,

    stride_qm,
    stride_qd,
    stride_km,
    stride_kd,
    stride_vm,
    stride_vd,
    stride_om,
    stride_od,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LOCAL_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)

    start_m = pid_m * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_d = offs_d < d

    # ============================================================
    # Q: keep FP32
    # ============================================================
    q = tl.load(
        Q
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    neg_inf = -float("inf")

    # online softmax state
    m_i = tl.where(mask_m, neg_inf, 0.0)
    l_i = tl.where(mask_m, 0.0, 1.0)

    acc = tl.zeros(
        (BLOCK_M, BLOCK_D),
        dtype=tl.float32,
    )

    # ============================================================
    # 1. Sink tokens
    # ============================================================
    offs_s = tl.arange(0, BLOCK_S)
    sink_col_mask = offs_s < num_sinks

    # [D, S], FP32
    k_sink = tl.load(
        K
        + offs_d[:, None] * stride_kd
        + offs_s[None, :] * stride_km,
        mask=mask_d[:, None] & sink_col_mask[None, :],
        other=0.0,
    )

    # FP32 x FP32, IEEE
    qk_sink = tl.dot(
        q,
        k_sink,
        input_precision="ieee",
    )

    # sm_scale = log2(e) / sqrt(d)
    qk_sink = qk_sink * sm_scale

    valid_sink = (
        mask_m[:, None]
        & sink_col_mask[None, :]
        & (offs_s[None, :] <= offs_m[:, None])
    )

    qk_sink = tl.where(
        valid_sink,
        qk_sink,
        neg_inf,
    )

    block_max_sink = tl.max(
        qk_sink,
        axis=1,
    )

    m_new_sink = tl.maximum(
        m_i,
        block_max_sink,
    )

    alpha_sink = tl.exp2(
        m_i - m_new_sink
    )

    # IMPORTANT: p_sink stays FP32
    p_sink = tl.exp2(
        qk_sink - m_new_sink[:, None]
    )

    l_i = (
        l_i * alpha_sink
        + tl.sum(p_sink, axis=1)
    )

    # [S, D], FP32
    v_sink = tl.load(
        V
        + offs_s[:, None] * stride_vm
        + offs_d[None, :] * stride_vd,
        mask=sink_col_mask[:, None] & mask_d[None, :],
        other=0.0,
    )

    # IMPORTANT:
    # no p_sink.to(float16)
    # no v_sink.to(float16)
    acc = (
        acc * alpha_sink[:, None]
        + tl.dot(
            p_sink,
            v_sink,
            input_precision="ieee",
        )
    )

    m_i = m_new_sink

    # ============================================================
    # 2. Sliding window
    # ============================================================
    local_start = start_m - window_size + 1

    # Sink positions already handled above.
    local_start = tl.maximum(
        local_start,
        num_sinks,
    )

    offs_bn = tl.arange(0, BLOCK_N)

    for block_idx in tl.range(
        0,
        N_LOCAL_BLOCKS,
    ):
        offs_n = (
            local_start
            + block_idx * BLOCK_N
            + offs_bn
        )

        mask_n = offs_n < M

        # [D, N], FP32
        k_local = tl.load(
            K
            + offs_d[:, None] * stride_kd
            + offs_n[None, :] * stride_km,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.0,
        )

        qk_local = tl.dot(
            q,
            k_local,
            input_precision="ieee",
        )

        qk_local = qk_local * sm_scale

        window_left = (
            offs_m[:, None]
            - window_size
            + 1
        )

        valid_local = (
            mask_m[:, None]
            & mask_n[None, :]
            & (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] >= window_left)
            & (offs_n[None, :] >= num_sinks)
        )

        qk_local = tl.where(
            valid_local,
            qk_local,
            neg_inf,
        )

        # ========================================================
        # Online softmax update
        # ========================================================
        block_max_local = tl.max(
            qk_local,
            axis=1,
        )

        m_new_local = tl.maximum(
            m_i,
            block_max_local,
        )

        alpha_local = tl.exp2(
            m_i - m_new_local
        )

        # stays FP32
        p_local = tl.exp2(
            qk_local - m_new_local[:, None]
        )

        l_i = (
            l_i * alpha_local
            + tl.sum(
                p_local,
                axis=1,
            )
        )

        # [N, D], FP32
        v_local = tl.load(
            V
            + offs_n[:, None] * stride_vm
            + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )

        # FP32 P @ V
        acc = (
            acc * alpha_local[:, None]
            + tl.dot(
                p_local,
                v_local,
                input_precision="ieee",
            )
        )

        m_i = m_new_local

    # ============================================================
    # Normalize
    # ============================================================
    out = acc / l_i[:, None]

    tl.store(
        output
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od,
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    num_sinks: int,
    window_size: int,
):
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_S = 16

    BLOCK_D = max(
        16,
        triton.next_power_of_2(d),
    )

    N_LOCAL_BLOCKS = triton.cdiv(
        window_size + BLOCK_M - 1,
        BLOCK_N,
    )

    # softmax using exp2:
    #
    # exp(x) = exp2(x * log2(e))
    #
    # score = dot(Q, K) / sqrt(d)
    sm_scale = (
        1.4426950408889634
        / (d ** 0.5)
    )

    grid = (
        triton.cdiv(M, BLOCK_M),
    )

    _attention_sinks_kernel[grid](
        Q,
        K,
        V,
        output,

        M,
        d,
        num_sinks,
        window_size,
        sm_scale,

        Q.stride(0),
        Q.stride(1),

        K.stride(0),
        K.stride(1),

        V.stride(0),
        V.stride(1),

        output.stride(0),
        output.stride(1),

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        N_LOCAL_BLOCKS=N_LOCAL_BLOCKS,

        num_warps=8,
        num_stages=1,
    )