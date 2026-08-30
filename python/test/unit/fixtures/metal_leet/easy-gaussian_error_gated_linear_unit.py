import torch
import triton
import triton.language as tl


@triton.jit
def geglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x1 = tl.load(input + offset, mask = offset < N // 2, other = 0.0)
    x2 = tl.load(input + offset + N // 2, mask = offset < N // 2, other = 0.0)
    gelu_x2 = x2 * (1 + tl.erf(x2 * tl.sqrt(2.0) * 0.5)) * 0.5
    output_ = x1 * gelu_x2
    tl.store(output + offset, output_, mask = offset < N // 2)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    import sys
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0)
    N = 4096
    inp = torch.randn(N, dtype=torch.float32, device=device)
    out = torch.zeros(N // 2, dtype=torch.float32, device=device)

    solve(inp, out, N)

    x1, x2 = inp[: N // 2], inp[N // 2 :]
    expected = x1 * torch.nn.functional.gelu(x2)
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3), (
        f"max abs err = {(out - expected).abs().max().item()}"
    )
    print(f"PASS [{device}] geglu N={N}")
