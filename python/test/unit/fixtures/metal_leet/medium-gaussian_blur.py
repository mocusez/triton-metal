import torch
import triton
import triton.language as tl

@triton.jit
def convolve_2d(input, kernel, output, R,C,kR, kC, BLOCK_SIZE_ROWS:tl.constexpr, BLOCK_SIZE_COLS:tl.constexpr):
    program_row_id = tl.program_id(0)
    program_col_id = tl.program_id(1)

    input_row_offset = program_row_id * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    input_col_offset = program_col_id * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
    mask_input_row = input_row_offset < R
    mask_input_col = input_col_offset < C

    acc = tl.zeros((BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS), dtype=tl.float32)
    for kernel_row in range(0, kR):
        offset_row_current = input_row_offset + kernel_row - tl.floor(kR/2).to(tl.int32)
        mask_rows = (offset_row_current[:,None] >= 0) & (offset_row_current[:,None] < R)
        for kernel_col in range(0, kC):
            offset_col_current = input_col_offset + kernel_col - tl.floor(kC / 2).to(tl.int32)
            mask_cols = (offset_col_current[None,:] >= 0) & (offset_col_current[None,:] < C)
            input_slice = tl.load(
                input
                + (offset_row_current * C)[:,None]
                + offset_col_current[None,:],
                mask = mask_rows & mask_cols
            )
            kernel_elem = tl.load(kernel + kernel_row * kC + kernel_col)
            acc += input_slice * kernel_elem
    tl.store(
        output
        + (input_row_offset * C)[:,None]
        + input_col_offset[None,:],
        acc,
        mask = mask_input_row[:,None] & mask_input_col[None,:]
    )


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    BLOCK_SIZE_ROW = 64
    BLOCK_SIZE_COL = 32

    grid = (triton.cdiv(input_rows, BLOCK_SIZE_ROW), triton.cdiv(input_cols, BLOCK_SIZE_COL))

    convolve_2d[grid](
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
        BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL
    )
