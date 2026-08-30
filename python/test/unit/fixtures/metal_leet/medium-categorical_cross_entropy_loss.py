import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    logits_ptr, labels_ptr, loss_ptr, N, C,
    BLOCK_SIZE: tl.constexpr, C_BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    N_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    N_mask = N_offset < N
    C_mask = tl.arange(0, C_BLOCK_SIZE) < C

    offset = C * N_offset[:,None] + tl.arange(0, C_BLOCK_SIZE)[None,:]
    mask = N_mask[:,None] & C_mask[None,:]

    logits = tl.load(logits_ptr + offset, mask = mask, other = float("-inf"))

    result = logits.exp()
    result = tl.where(N_mask, result.sum(-1),1.0)
    result = result.log().sum()

    labels = tl.load(labels_ptr + N_offset, mask = N_mask, other = 0)
    offset = C * N_offset + labels
    logits = tl.load(logits_ptr + offset, mask = N_mask)

    result -= logits.sum()

    result /= N

    tl.atomic_add(loss_ptr, result, sem="relaxed")



# logits, true_labels, loss are tensors on the GPU
def solve(logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
    C_BLOCK_SIZE = triton.next_power_of_2(C)
    BLOCK_SIZE = 1024 // C_BLOCK_SIZE

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    kernel[grid](
        logits,
        true_labels,
        loss,
        N,
        C,
        BLOCK_SIZE, C_BLOCK_SIZE
    )
