import torch
import triton
import triton.language as tl


@triton.jit
def conv_3d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    input_depth, input_rows, input_cols,
    kernel_depth, kernel_rows, kernel_cols,
    output_depth, output_rows, output_cols,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    d_pid = tl.program_id(axis = 2)
    r_pid = tl.program_id(axis = 1)
    c_pid = tl.program_id(axis = 0)

    offset_d = d_pid * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offset_r = r_pid * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offset_c = c_pid * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    output_ = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_R, BLOCK_SIZE_C), dtype = tl.float32)
    for i in range(kernel_depth):
        for j in range(kernel_rows):
            for k in range(kernel_cols):
                kernel_offset = i * kernel_rows * kernel_cols + j * kernel_cols + k
                k_val = tl.load(kernel_ptr + kernel_offset)
                offset_d_ = offset_d + i
                offset_r_ = offset_r + j
                offset_c_ = offset_c + k
                mask = (offset_d_[:,None,None] < input_depth) & \
                        (offset_r_[None,:,None] < input_rows) & \
                        (offset_c_[None,None,:] < input_cols)
                input_ptrs = input_ptr + offset_d_[:,None,None] * input_rows * input_cols + \
                                offset_r_[None,:,None] * input_cols + \
                                offset_c_[None,None,:]
                input_t = tl.load(input_ptrs, mask = mask, other = 0.0)
                output_ += input_t * k_val
    mask = (offset_d[:,None,None] < output_depth) & \
            (offset_r[None,:,None] < output_rows) & \
            (offset_c[None, None,:] < output_cols)
    output_ptrs = output_ptr + offset_d[:,None,None] * output_rows * output_cols + \
                    offset_r[None,:,None] * output_cols + \
                    offset_c[None, None,:]
    tl.store(output_ptrs, output_, mask=mask)

# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    BLOCK_SIZE_D = 4
    BLOCK_SIZE_R = 4
    BLOCK_SIZE_C = 64

    grid = (triton.cdiv(output_cols, BLOCK_SIZE_C), triton.cdiv(output_rows, BLOCK_SIZE_R), triton.cdiv(output_depth, BLOCK_SIZE_D))
    conv_3d_kernel[grid](
        input, kernel, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        output_depth, output_rows, output_cols,
        BLOCK_SIZE_D, BLOCK_SIZE_R, BLOCK_SIZE_C
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
    input_depth, input_rows, input_cols = 8, 12, 70
    kernel_depth, kernel_rows, kernel_cols = 3, 3, 3
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    inp = torch.randn(input_depth, input_rows, input_cols,
                      dtype=torch.float32, device=device).contiguous()
    ker = torch.randn(kernel_depth, kernel_rows, kernel_cols,
                      dtype=torch.float32, device=device).contiguous()
    out = torch.zeros(output_depth, output_rows, output_cols,
                      dtype=torch.float32, device=device).contiguous()

    solve(inp, ker, out, input_depth, input_rows, input_cols,
          kernel_depth, kernel_rows, kernel_cols)

    # the kernel computes a valid cross-correlation == F.conv3d (no kernel flip);
    # reference on CPU to avoid relying on MPS conv3d support
    expected = F.conv3d(
        inp.cpu().view(1, 1, input_depth, input_rows, input_cols),
        ker.cpu().view(1, 1, kernel_depth, kernel_rows, kernel_cols),
    ).view(output_depth, output_rows, output_cols)
    assert torch.allclose(out.cpu(), expected, atol=1e-2, rtol=1e-2), (
        f"max abs err = {(out.cpu() - expected).abs().max().item()}"
    )
    print(f"PASS [{device}] 3D_convolution "
          f"in=({input_depth},{input_rows},{input_cols}) "
          f"k=({kernel_depth},{kernel_rows},{kernel_cols})")
