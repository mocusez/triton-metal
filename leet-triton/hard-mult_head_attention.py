import torch
import triton
import triton.language as tl


@triton.jit
def mha_kernel(Q_ptr, K_ptr, V_ptr, output_ptr,
                N, d_model, h,
                BLOCKSIZE_N: tl.constexpr,
                BLOCKSIZE_d: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offset = tl.arange(0, BLOCKSIZE_N)

    d_head = d_model // h

    offset_N = pid0 * BLOCKSIZE_N + offset
    offset_d = pid1 * d_head + tl.arange(0, BLOCKSIZE_d)

    mask_d = tl.arange(0, BLOCKSIZE_d) < d_head

    offset_Q = offset_N[:,None] * d_model + offset_d[None,:]
    mask_Q = (offset_N[:,None] < N) & mask_d[None,:]

    data_Q = tl.load(Q_ptr + offset_Q, mask = mask_Q)

    attention_logit_scale = 1.0 / tl.sqrt(d_head + 0.0)

    accumulator = tl.zeros((BLOCKSIZE_N, BLOCKSIZE_d), dtype=tl.float32)
    softmax_running_sum = tl.zeros([BLOCKSIZE_N], dtype=tl.float32)
    softmax_current_max = tl.full([BLOCKSIZE_N], float("-inf"), dtype=tl.float32)

    for current_index in range(0, N, BLOCKSIZE_N):
        current_N_offset = current_index + offset
        current_N_mask = current_N_offset < N

        offset_K = current_N_offset[:,None] * d_model + offset_d[None,:]
        mask_K = current_N_mask[:,None] & mask_d[None,:]
        data_K = tl.load(K_ptr + offset_K, mask=mask_K, other=0.0)

        offset_V = current_N_offset[:,None] * d_model + offset_d[None,:]
        mask_V = current_N_mask[:,None] & mask_d[None,:]
        data_V = tl.load(V_ptr + offset_V, mask=mask_V, other=0.0)

        attention_logits = tl.dot(data_Q, tl.trans(data_K)) * attention_logit_scale
        attention_logits_mask = (offset_N[:,None] < N) & (current_N_offset[None,:] < N)
        attention_logits = tl.where(attention_logits_mask, attention_logits, float("-inf"))

        current_block_max = tl.max(attention_logits, axis=1)
        max_value = tl.maximum(current_block_max, softmax_current_max)

        softmax_scaler = tl.exp(softmax_current_max - max_value)
        softmax_current_max = max_value

        attention_logits_shift = attention_logits - max_value[:, None]

        softmax_nom = tl.exp(attention_logits_shift)
        softmax_denom = tl.sum(softmax_nom, axis=1)

        softmax_running_sum = tl.fma(softmax_running_sum, softmax_scaler, softmax_denom)
        accumulator = tl.fma(accumulator, softmax_scaler[:, None], tl.dot(softmax_nom, data_V))
    
    accumulator = accumulator/softmax_running_sum[:,None]
    tl.store(output_ptr + offset_Q, accumulator, mask = mask_Q)



# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    BLOCKSIZE_N = 32
    BLOCKSIZE_d = max(16, d_model // h)

    grid = (triton.cdiv(N, BLOCKSIZE_N), h)
    mha_kernel[grid](Q,K,V, output,
                    N, d_model,h,
                    BLOCKSIZE_N, BLOCKSIZE_d,
                    num_warps = 4)
