import torch
import triton
import triton.language as tl

@triton.jit
def binoticsort_des_kernel(
    input_ptr, N,
    stage, stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis = 0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    slice_1_offset = (offset // stride) * 2 * stride + (offset % stride)
    slice_2_offset = slice_1_offset + stride

    slice_1_t = tl.load(input_ptr + slice_1_offset, mask = slice_1_offset < N, other = -float('inf'))
    slice_2_t = tl.load(input_ptr + slice_2_offset, mask = slice_2_offset < N, other = -float('inf'))

    descend = (slice_1_offset // stage) % 2 == 1
    greater = slice_1_t > slice_2_t

    new_slice_1_t = tl.where(descend == greater, slice_2_t, slice_1_t)
    new_slice_2_t = tl.where(descend == greater, slice_1_t, slice_2_t)

    tl.store(input_ptr + slice_1_offset, new_slice_1_t, mask = slice_1_offset < N)
    tl.store(input_ptr + slice_2_offset, new_slice_2_t, mask = slice_2_offset < N)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    paddingLen = triton.next_power_of_2(N)
    inputPadding = torch.zeros((paddingLen), device = input.device, dtype = input.dtype)
    inputPadding[:N] = input
    inputPadding[N:] = -float('inf')

    BLOCK_SIZE = 1024
    grid = lambda metadata: (triton.cdiv(paddingLen, metadata['BLOCK_SIZE'] * 2),)

    stage = 2
    while stage <= paddingLen:
        stride = (stage >> 1)
        while stride:
            binoticsort_des_kernel[grid](inputPadding, paddingLen, stage, stride, BLOCK_SIZE)
            stride >>= 1
        stage <<= 1
    output.copy_(inputPadding[:k])
