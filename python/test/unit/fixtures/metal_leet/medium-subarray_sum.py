import torch
import triton
import triton.language as tl


@triton.jit
def subarray_sum_kernel(
    input,
    output,
    S: tl.constexpr,
    E: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + S + tl.arange(0, BLOCK_SIZE)
    mask = offsets <= E

    local_input = tl.load(input + offsets, mask, other=0)
    local_sum = tl.sum(local_input)

    tl.atomic_add(output, local_sum)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    total = E - S + 1
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(total, BLOCK_SIZE),)
    subarray_sum_kernel[grid](input, output, S, E, BLOCK_SIZE)


if __name__ == "__main__":
    import sys

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    N, S, E = 4113, 113, 4098
    input_cpu = (torch.arange(N, dtype=torch.float32) % 29) * 0.125 - 1.5
    input = input_cpu.to(device)
    output = torch.zeros(1, dtype=torch.float32, device=device)

    solve(input, output, N, S, E)

    expected = input_cpu[S : E + 1].sum()
    torch.testing.assert_close(output.cpu()[0], expected, atol=2e-3, rtol=2e-5)
    print(f"PASS [{device}] subarray_sum N={N} range=[{S}, {E}]")
