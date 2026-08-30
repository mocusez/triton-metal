import torch
import triton
import triton.language as tl


@triton.jit
def fp16_kernel(A, B, result, N, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)
    offset_N = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset_N < N
    A_val = tl.load(A + offset_N, mask = mask, other = 0.)
    B_val = tl.load(B + offset_N, mask = mask, other = 0.)
    cum_sum = tl.sum(A_val.to(tl.float32) * B_val.to(tl.float32))
    tl.atomic_add(result, cum_sum)

# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    res_fp32 = torch.zeros(1, dtype = torch.float32, device='cuda')
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fp16_kernel[grid](A, B, res_fp32, N, BLOCK_SIZE)
    result.copy_(res_fp32.to(torch.bfloat16))
