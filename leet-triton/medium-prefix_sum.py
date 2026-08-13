import torch
import triton
import triton.language as tl


@triton.jit
def single_block_scan_kernel(
    data_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    res = tl.cumsum(x, axis=0)
    tl.store(output_ptr + offsets, res, mask=mask)

@triton.jit
def local_scan_kernel(
    data_ptr,
    output_ptr,
    block_sums_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    
    local_scan = tl.cumsum(x, axis=0)
    tl.store(output_ptr + offsets, local_scan, mask=mask)
    

    block_sum = tl.sum(x, axis=0)
    tl.store(block_sums_ptr + pid, block_sum)


@triton.jit
def add_base_kernel(
    output_ptr,
    scanned_block_sums_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    if pid == 0:
        return
        
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    base = tl.load(scanned_block_sums_ptr + pid - 1)
    
    local_scan = tl.load(output_ptr + offsets, mask=mask)
    global_scan = local_scan + base
    tl.store(output_ptr + offsets, global_scan, mask=mask)


def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    if n <= 0:
        return
        
    BLOCK_SIZE = 1024
    if n <= BLOCK_SIZE:
        single_block_scan_kernel[(1,)](
            data, output, n, BLOCK_SIZE=BLOCK_SIZE
        )
        return
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    block_sums = torch.empty((num_blocks,), dtype=data.dtype, device=data.device)
    scanned_block_sums = torch.empty((num_blocks,), dtype=data.dtype, device=data.device)
    
    local_scan_kernel[(num_blocks,)](
        data, output, block_sums, n, BLOCK_SIZE=BLOCK_SIZE
    )
    

    solve(block_sums, scanned_block_sums, num_blocks)
    
    add_base_kernel[(num_blocks,)](
        output, scanned_block_sums, n, BLOCK_SIZE=BLOCK_SIZE
    )