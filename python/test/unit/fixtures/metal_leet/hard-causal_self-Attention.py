import torch
import triton
import triton.language as tl

@triton.jit
def cross_attn(q, k, v, output, M, d,
                BLOCK_SIZE_ROW: tl.constexpr,
                BLOCK_SIZE_COL: tl.constexpr,
                BLOCK_SIZE_RUNNING: tl.constexpr):

    pid = tl.program_id(0)

    q_offsets = pid * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    q_mask = q_offsets < M

    col_offsets = tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < d

    q_slice = tl.load(q + q_offsets[:,None] * d + col_offsets[None,:], mask = q_mask[:,None]&col_mask[None,:], other = 0.0)

    scale = 1/tl.sqrt(d.to(tl.float32))

    s_max = tl.full([BLOCK_SIZE_ROW,], -float('inf'), dtype=tl.float32)
    s_sum = tl.zeros([BLOCK_SIZE_ROW,], dtype=tl.float32)
    accm = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.float32)

    for slice in tl.range(0, M, BLOCK_SIZE_RUNNING):
        slice_offsets =  slice + tl.arange(0, BLOCK_SIZE_RUNNING)
        slice_mask = slice_offsets < M
        slice_ptr = slice_offsets[:,None]*d + col_offsets[None,:]

        k_slice = tl.load(k+slice_ptr, mask = slice_mask[:, None]&col_mask[None,:],other=0.0)
        v_slice = tl.load(v+slice_ptr, mask = slice_mask[:, None]&col_mask[None,:],other=0.0)

        ac = tl.dot(q_slice, tl.trans(k_slice)) * scale
        is_valid = (q_offsets[:,None] >= slice_offsets[None,:]) & slice_mask[None,:]
        ac_causal = tl.where(is_valid, ac , -float('inf'))

        new_max = tl.maximum(s_max, tl.max(ac_causal, axis = 1))
        alpha = tl.exp(s_max-new_max)
        s_max = new_max

        ac_causal = tl.exp(ac_causal-new_max[:,None])
        s_sum = tl.fma(s_sum, alpha, tl.sum(ac_causal, axis = 1))

        accm = tl.fma(accm, alpha[:,None], tl.dot(ac_causal, v_slice))
    accm = accm/s_sum[:,None]

    tl.store(
        output + q_offsets[:,None] * d + col_offsets[None, :],
        accm,
        mask=q_mask[:,None]&col_mask[None,:]
    )


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    BLOCK_SIZE_ROW = 16
    BLOCK_SIZE_D = max(16, triton.next_power_of_2(d))
    RUNNING_BLOCK_SIZE = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_ROW),)
    cross_attn[grid](Q, K ,V,
                output, M, d,
                BLOCK_SIZE_ROW, BLOCK_SIZE_D, RUNNING_BLOCK_SIZE)
