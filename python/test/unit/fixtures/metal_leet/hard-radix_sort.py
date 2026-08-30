import torch
import triton
import triton.language as tl


@triton.jit
def count_kernel(src_ptr, count_ptr, N, shift, num_blocks, BLOCK_SIZE:tl.constexpr):
    src_ptr = src_ptr.to(tl.pointer_type(tl.uint32))

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(src_ptr + offsets, mask = mask, other = 0)

    digits = (vals >> shift) & 0xF
    digits = tl.where(mask, digits, 16)

    for b in tl.static_range(16):
        is_b = (digits == b)
        sum_b = tl.sum(tl.cast(is_b, tl.int32))
        tl.store(count_ptr + b * num_blocks + pid, sum_b)

@triton.jit
def scatter_kernel(src_ptr, dst_ptr, offset_ptr, N, shift, num_blocks, BLOCK_SIZE:tl.constexpr):
    src_ptr = src_ptr.to(tl.pointer_type(tl.uint32))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.uint32))

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(src_ptr + offsets, mask = mask, other = 0)
    digits = (vals >> shift) & 0xF
    digits = tl.where(mask, digits, 16)

    loc_offsets = tl.zeros([BLOCK_SIZE], dtype = tl.int32)
    glob_offsets = tl.zeros([BLOCK_SIZE], dtype = tl.int32)

    for b in tl.static_range(16):
        is_b = (digits == b)
        is_b_int = tl.cast(is_b, tl.int32)

        cum_b = tl.cumsum(is_b_int)
        loc_offsets = tl.where(is_b, cum_b - is_b_int, loc_offsets)

        g_off = tl.load(offset_ptr + b * num_blocks + pid)
        glob_offsets = tl.where(is_b, g_off, glob_offsets)

    write_idx = glob_offsets + loc_offsets
    tl.store(dst_ptr + write_idx, vals, mask = mask)



# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    buf = torch.empty_like(input, memory_format = torch.contiguous_format)

    counts = torch.empty((16, num_blocks), dtype = torch.int32, device = input.device)
    global_offsets = torch.empty_like(counts)

    for pass_idx in range(8):
        shift = pass_idx * 4

        if pass_idx == 0:
            src = input
            dst = buf
        elif pass_idx % 2 == 1:
            src = buf
            dst = output
        else:
            src = output
            dst = buf

        count_kernel[(num_blocks,)](src, counts, N, shift, num_blocks, BLOCK_SIZE=BLOCK_SIZE)

        counts_flat = counts.view(-1)
        global_offsets_flat = global_offsets.view(-1)
        global_offsets_flat[0] = 0
        if counts_flat.numel() > 1:
            global_offsets_flat[1:] = torch.cumsum(counts_flat[:-1], dim=0)

        scatter_kernel[(num_blocks,)](src, dst, global_offsets, N, shift, num_blocks, BLOCK_SIZE = BLOCK_SIZE)
