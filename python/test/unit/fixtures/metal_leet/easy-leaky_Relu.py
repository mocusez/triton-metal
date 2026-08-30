import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask = mask, other = 0.0)

    output = tl.where(x > 0, x , 0.01 * x)

    tl.store(output_ptr + offsets, output, mask = mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](input, output, N, BLOCK_SIZE)


if __name__ == "__main__":
    import sys
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0)
    N = 1 << 14
    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    solve(x, y, N)

    expected = torch.where(x > 0, x, 0.01 * x)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)
    print(f"PASS [{device}] leaky_relu N={N}")
