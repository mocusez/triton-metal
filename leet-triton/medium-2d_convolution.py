import torch
import triton
import triton.language as tl

@triton.jit
def conv_kernel(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, BLOCK_SIZE: tl.constexpr):
    bx = tl.program_id(0)
    by = tl.program_id(1)
    ar = tl.arange(0, BLOCK_SIZE)
    col = bx * BLOCK_SIZE
    row = by * BLOCK_SIZE

    col_offset = (col + ar)[None,:]
    row_offset = (row + ar)[:,None]

    Pvalue = (row_offset + col_offset) * 0.0
    for i in range(kernel_rows):
        dr = row_offset + i
        for j in range(kernel_cols):
            dl = col_offset + j
            data = tl.load(
                input + dr * input_cols + dl,
                mask = (dr < input_rows) & (dl < input_cols), other = 0.0,
                cache_modifier = '.ca'
            )
            kdata = tl.load(kernel + i * kernel_cols + j, cache_modifier='.ca')
            Pvalue += data * kdata

    out_cols = input_cols - kernel_cols + 1
    out_rows = input_rows - kernel_rows + 1
    tl.store(
        output + row_offset * out_cols + col_offset, Pvalue,
        mask = (row_offset < out_rows) & (col_offset < out_cols)
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
    block_size = 32
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    mrows = triton.cdiv(output_rows, block_size)
    mcols = triton.cdiv(output_cols, block_size)
    grid = (mcols, mrows)
    conv_kernel[grid](
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
        BLOCK_SIZE=block_size
    )


if __name__ == "__main__":
    import sys
    import torch.nn.functional as F

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0)
    cases = ((35, 41, 3, 3), (64, 64, 5, 5))
    for input_rows, input_cols, kernel_rows, kernel_cols in cases:
        output_rows = input_rows - kernel_rows + 1
        output_cols = input_cols - kernel_cols + 1
        inp = torch.randn(input_rows, input_cols, dtype=torch.float32, device=device).contiguous()
        ker = torch.randn(kernel_rows, kernel_cols, dtype=torch.float32, device=device).contiguous()
        out = torch.empty(output_rows, output_cols, dtype=torch.float32, device=device).contiguous()

        solve(inp, ker, out, input_rows, input_cols, kernel_rows, kernel_cols)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        expected = F.conv2d(
            inp.cpu().view(1, 1, input_rows, input_cols),
            ker.cpu().view(1, 1, kernel_rows, kernel_cols),
        ).view(output_rows, output_cols)
        torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)
        print(
            f"PASS [{device}] 2D_convolution "
            f"in=({input_rows},{input_cols}) k=({kernel_rows},{kernel_cols})"
        )
