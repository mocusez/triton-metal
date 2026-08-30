import torch
import triton
import triton.language as tl


@triton.jit
def _token_embedding_layernorm_kernel(
    token_ids_ptr,      # (B, T) int32
    position_ids_ptr,   # (T,)   int32
    token_emb_ptr,      # (V, D)
    position_emb_ptr,   # (P, D)
    gamma_ptr,          # (D,)
    beta_ptr,           # (D,)
    output_ptr,         # (B, T, D)
    N,                  # B * T, 摊平后的总行数
    T,
    D,
    eps,
    BLOCK_T: tl.constexpr,   # 每个 program 处理的 (b, t) 行数
    BLOCK_D: tl.constexpr,   # D 向上取整到 2 的幂
):
    pid = tl.program_id(axis=0)

    # ---- 行索引: (b, t) 摊平为 r = b * T + t ----
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < N
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    mask2d = row_mask[:, None] & col_mask[None, :]

    # ---- gather: 取 token / position 行号 ----
    tok_ids = tl.load(token_ids_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    pos_ids = tl.load(position_ids_ptr + rows % T, mask=row_mask, other=0).to(tl.int64)

    # ---- 从两张嵌入表取向量并相加 (fp32 计算, T4 上最稳) ----
    emb_off = cols[None, :]
    tok = tl.load(token_emb_ptr + tok_ids[:, None] * D + emb_off,
                  mask=mask2d, other=0.0).to(tl.float32)
    pos = tl.load(position_emb_ptr + pos_ids[:, None] * D + emb_off,
                  mask=mask2d, other=0.0).to(tl.float32)
    s = tok + pos

    # ---- LayerNorm: 均值 / 方差 (除以 D, 无 Bessel 校正) ----
    mean = tl.sum(s, axis=1) / D
    diff = tl.where(mask2d, s - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    # ---- 缩放平移并写回 ----
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = gamma[None, :] * diff * rstd[:, None] + beta[None, :]

    out_off = rows.to(tl.int64)[:, None] * D + emb_off
    tl.store(output_ptr + out_off, y.to(output_ptr.dtype.element_ty), mask=mask2d)


# token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output
# are tensors on the GPU
def solve(
    token_ids: torch.Tensor,
    position_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    position_embeddings: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    B: int,
    T: int,
    V: int,
    P: int,
    D: int,
    eps: float,
):
    N = B * T

    BLOCK_D = triton.next_power_of_2(D)      # D <= 1024, 单块覆盖整行, 无需跨块归约
    BLOCK_T = max(1, 4096 // BLOCK_D)        # 每个 program 约处理 4K 元素
    num_warps = 8 if BLOCK_T * BLOCK_D >= 4096 else 4

    grid = (triton.cdiv(N, BLOCK_T),)
    _token_embedding_layernorm_kernel[grid](
        token_ids, position_ids,
        token_embeddings, position_embeddings,
        gamma, beta, output,
        N, T, D, eps,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return output
