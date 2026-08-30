import torch
import triton
import triton.language as tl

@triton.jit
def hist_compute(input, output, N, n_bins, padded_num_bins: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    input_array = tl.load(input + offset, mask=mask)
    hist = tl.histogram(input_array, padded_num_bins)
    zero_from_padding = BLOCK_SIZE - tl.sum(mask.to(tl.int32))

    output_offset = tl.arange(0, padded_num_bins)
    bad_zeros = output_offset == 0
    hist = hist - zero_from_padding * bad_zeros

    output_mask = (output_offset < n_bins) & (hist != 0)

    tl.atomic_add(output + output_offset, hist, mask=output_mask)

# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N,BLOCK_SIZE),)
    hist_compute[grid](input, histogram, N, num_bins, triton.next_power_of_2(num_bins), BLOCK_SIZE)
