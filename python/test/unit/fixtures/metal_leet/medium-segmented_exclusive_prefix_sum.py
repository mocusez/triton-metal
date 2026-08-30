import torch
import triton
import triton.language as tl
import math

@triton.jit
def seg_scan_combine(v1, f1, v2, f2):
    v_out = tl.where(f2, v2, v1 + v2)
    f_out = f1 | f2
    return v_out, f_out

@triton.jit
def pass1_reduce(
    values_ptr, flags_ptr,
    tile_sums_ptr, tile_flags_ptr,
    N,
    TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    tile_start = pid * TILE_SIZE

    acc_sum = 0.0
    acc_flag = 0

    for i in range(0, TILE_SIZE, BLOCK_SIZE):
        offsets = tile_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        vals = tl.load(values_ptr + offsets, mask = mask, other = 0.0)
        flgs = tl.load(flags_ptr + offsets, mask = mask, other = 0.0)
        flgs_bool = flgs == 1

        loc_sums, loc_flags = tl.associative_scan((vals, flgs_bool), axis = 0, combine_fn = seg_scan_combine)

        mask_last = tl.arange(0, BLOCK_SIZE) == BLOCK_SIZE - 1
        chunk_sum = tl.sum(tl.where(mask_last, loc_sums, 0.0))
        chunk_flag = tl.max(tl.where(mask_last, tl.cast(loc_flags, tl.int32), 0))

        acc_sum = chunk_sum if chunk_flag == 1 else acc_sum + chunk_sum
        acc_flag = acc_flag | chunk_flag

    tl.store(tile_sums_ptr + pid, acc_sum)
    tl.store(tile_flags_ptr + pid, acc_flag)

@triton.jit
def pass2_scan(
    tile_sums_ptr, tile_flags_ptr,
    NUM_TILES,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_TILES

    sums = tl.load(tile_sums_ptr + offsets, mask = mask, other = 0.0)
    flags = tl.load(tile_flags_ptr + offsets, mask = mask, other = 0)
    flags_bool = flags == 1

    inc_sums, inc_flags = tl.associative_scan((sums, flags_bool), axis = 0, combine_fn = seg_scan_combine)

    tl.store(tile_sums_ptr + offsets, inc_sums, mask = mask)
    tl.store(tile_flags_ptr + offsets, tl.cast(inc_flags, tl.int32), mask = mask)

@triton.jit
def pass3_downsweep(
    values_ptr, flags_ptr, output_ptr,
    tile_sums_ptr, tile_flags_ptr,
    N,
    TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    tile_start = pid * TILE_SIZE

    if pid == 0:
        acc_sum = 0.0
        acc_flag = 0
    else:
        acc_sum = tl.load(tile_sums_ptr + pid - 1)
        acc_flag = tl.load(tile_flags_ptr + pid - 1)

    for i in range(0, TILE_SIZE, BLOCK_SIZE):
        offsets = tile_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        vals = tl.load(values_ptr + offsets, mask = mask, other = 0.0)
        flgs = tl.load(flags_ptr + offsets, mask = mask, other = 0)
        flgs_bool = flgs == 1

        loc_sums, loc_flags = tl.associative_scan((vals, flgs_bool), axis = 0, combine_fn = seg_scan_combine)

        global_inc_sums = tl.where(loc_flags, loc_sums, acc_sum + loc_sums)

        exc_sums = global_inc_sums - vals

        tl.store(output_ptr + offsets, exc_sums, mask = mask)

        mask_last = tl.arange(0, BLOCK_SIZE) == BLOCK_SIZE - 1
        chunk_sum = tl.sum(tl.where(mask_last, loc_sums, 0.0))
        chunk_flag = tl.max(tl.where(mask_last, tl.cast(loc_flags, tl.int32), 0))

        acc_sum = chunk_sum if chunk_flag == 1 else acc_sum + chunk_sum
        acc_flag = acc_flag | chunk_flag


# values, flags, output are tensors on the GPU
def solve(values: torch.Tensor, flags: torch.Tensor, output: torch.Tensor, N: int):
    TILE_SIZE = 65536
    BLOCK_SIZE = 4096
    NUM_TILES = math.ceil(N / TILE_SIZE)

    tile_sums = torch.empty(NUM_TILES, device = values.device, dtype = torch.float32)
    tile_flags = torch.empty(NUM_TILES, device = flags.device, dtype = torch.int32)

    pass1_reduce[(NUM_TILES,)](
        values, flags, tile_sums, tile_flags, N,
        TILE_SIZE=TILE_SIZE, BLOCK_SIZE=BLOCK_SIZE,
        num_warps = 8
    )

    BLOCK_SIZE_2 = max(16, triton.next_power_of_2(NUM_TILES))
    pass2_scan[(1, )](
        tile_sums, tile_flags, NUM_TILES,
        BLOCK_SIZE=BLOCK_SIZE_2,
        num_warps=4
    )

    pass3_downsweep[(NUM_TILES,)](
        values, flags, output, tile_sums, tile_flags, N,
        TILE_SIZE = TILE_SIZE, BLOCK_SIZE=BLOCK_SIZE,
        num_warps = 8
    )
