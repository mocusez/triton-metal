import torch
import triton
import triton.language as tl

@triton.jit
def kernel_moe(logits_ptr, topk_w_ptr, topk_idx_ptr, M, E, K,
                BLOCK_SIZE_E: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):

    pid = tl.program_id(0)
    offs_le = tl.arange(0, BLOCK_SIZE_E)
    mask_le = offs_le < E
    logits = tl.load(logits_ptr + pid * E + offs_le, mask = mask_le, other = float('-inf'))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_k = offs_k < K
    topk_vals = tl.full((BLOCK_SIZE_K, ), value = float("-inf"), dtype = tl.float32)
    topk_idxs = tl.full((BLOCK_SIZE_K, ), value = 0, dtype = tl.int32)
    for i in range(K):
        curr_max_val = tl.max(logits, axis = -1)
        curr_max_idx = tl.argmax(logits, axis = -1)

        topk_vals = tl.where(offs_k == i, curr_max_val, topk_vals)
        topk_idxs = tl.where(offs_k == i, curr_max_idx, topk_idxs)

        logits = tl.where(offs_le == curr_max_idx, float("-inf"), logits)
    mx = tl.max(topk_vals, axis = -1)

    topk_vals = tl.exp(topk_vals - mx)
    topk_vals = topk_vals / tl.sum(topk_vals, axis = -1)

    tl.store(topk_w_ptr + pid * K + offs_k, topk_vals, mask = mask_k)
    tl.store(topk_idx_ptr + pid * K + offs_k, topk_idxs, mask = mask_k)

# logits, topk_weights, topk_indices are tensors on the GPU
def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    block_e = triton.next_power_of_2(E)
    block_k = triton.next_power_of_2(k)
    grid = (M,)
    kernel_moe[grid](logits,topk_weights,topk_indices,M,E,k,block_e,block_k)
