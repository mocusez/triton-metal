import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    input_size,
    kernel_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    p0 = tl.program_id(0)
    offs_out = p0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(0, kernel_size, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < kernel_size
        kernel = tl.load(kernel_ptr + offs_k, mask_k)

        offs_i = offs_out[:, None] + offs_k[None, :]
        mask_i = offs_i < input_size
        input = tl.load(input_ptr + offs_i, mask_i)

        sum += tl.sum(kernel[None, :] * input, axis=1)

    mask_out = offs_out < (input_size - kernel_size + 1)
    tl.store(output_ptr + offs_out, sum, mask_out)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    BLOCK_K = 65536 // BLOCK_SIZE
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](
        input,
        kernel,
        output,
        input_size,
        kernel_size,
        BLOCK_SIZE,
        BLOCK_K,
    )


if __name__ == "__main__":
    import sys
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0)
    input_size, kernel_size = 4096, 7
    out_size = input_size - kernel_size + 1
    inp = torch.randn(input_size, dtype=torch.float32, device=device)
    kern = torch.randn(kernel_size, dtype=torch.float32, device=device)
    out = torch.zeros(out_size, dtype=torch.float32, device=device)

    solve(inp, kern, out, input_size, kernel_size)

    expected = torch.nn.functional.conv1d(
        inp.view(1, 1, -1), kern.view(1, 1, -1)
    ).flatten()
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3), (
        f"max abs err = {(out - expected).abs().max().item()}"
    )
    print(f"PASS [{device}] conv1d N={input_size} K={kernel_size}")
