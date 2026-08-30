import torch
import triton
import triton.language as tl

@triton.jit
def attention(
    Q, K, V,
    output,
    M, N, d, window_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_d = tl.arange(0, BLOCK_D)
    mask_m = offset_m < M
    mask_d = offset_d < d
    vals_q = tl.load(Q + offset_m[:, None] * d + offset_d[None, :], mask_m[:, None] & mask_d[None, :], other = 0.0)
    scale = tl.sqrt(d.to(tl.float32))

    out_vals = tl.zeros((BLOCK_M, BLOCK_D), dtype = tl.float32)
    ma = tl.full((BLOCK_M,), float("-inf"), dtype = tl.float32)
    sum = tl.full((BLOCK_M,), 0.0, dtype = tl.float32)

    for step in range(0, tl.cdiv(N, BLOCK_N)):
        offset_n = step * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offset_n < N

        vals_k = tl.load(K + offset_n[:, None] * d + offset_d[None, :], mask_n[:, None] & mask_d[None, :], other = 0.0)

        vals_qk = tl.dot(vals_q, tl.permute(vals_k, (1, 0)), allow_tf32 = False) / scale

        mask = tl.abs(offset_m[:, None] - offset_n[None, :]) <= window_size

        vals_qk_ma = tl.where(mask, vals_qk, float(-100))
        ma_now = tl.maximum(tl.max(vals_qk_ma, axis = 1), ma)
        vals_exp = tl.where(mask_m[:, None] & mask_n[None, :], tl.exp(vals_qk - ma_now[:, None]), 0.0)
        vals_exp = tl.where(mask, vals_exp, 0.0)
        sum_now = tl.sum(vals_exp, axis = 1)

        vals_v = tl.load(V + offset_n[:, None] * d + offset_d[None, :], mask_n[:, None] & mask_d[None, :], other = 0.0)

        out_vals = out_vals * tl.exp(ma - ma_now)[:, None] + tl.dot(vals_exp, vals_v, allow_tf32 = False)

        sum = sum * tl.exp(ma - ma_now) + sum_now
        ma = ma_now

    tl.store(output + offset_m[:, None] * d + offset_d[None, :], out_vals / sum[:, None], mask = mask_m[:, None] & mask_d[None, :])




# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    window_size: int,
):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = max(16, triton.next_power_of_2(d))
    grid = (triton.cdiv(M, BLOCK_M), )
    attention[grid](Q, K, V, output, M, M, d, window_size, BLOCK_M, BLOCK_N, BLOCK_D)
