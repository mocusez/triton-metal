import torch
import triton
import triton.language as tl


@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261

    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)

    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME

    return hash_val


@triton.jit
def fnv1a_hash_kernel(input, output, n_elements, n_rounds, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(axis = 0)

    initial_offset = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_slice = initial_offset < n_elements

    accumulated_array = tl.load(input + initial_offset, mask = mask_slice).to(tl.uint32)
    for _ in range(n_rounds):
        accumulated_array = fnv1a_hash(accumulated_array)

    tl.store(output + initial_offset, accumulated_array, mask = mask_slice)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](input, output, N, R, BLOCK_SIZE)
