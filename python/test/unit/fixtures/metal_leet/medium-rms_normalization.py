import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_1d_kernel(
    input_ptr,
    output_ptr,
    gamma,
    beta,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    sum_sq = 0.0
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask = mask, other = 0.0)
        sum_sq += tl.sum(x * x, axis = 0)

    mean_sq = sum_sq / N
    rms = tl.sqrt(mean_sq + eps)

    for i in range(0, num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask = mask, other = 0.0)

        x_hat = x / rms
        y = gamma * x_hat + beta

        tl.store(output_ptr + offsets, y, mask = mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = 1024
    rms_norm_1d_kernel[(1,)](
        input,
        output,
        gamma,
        beta,
        N,
        eps,
        BLOCK_SIZE = BLOCK_SIZE
    )
