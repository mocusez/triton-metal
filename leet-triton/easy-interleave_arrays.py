import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
    mask = offsets < N * 2
    halved_offsets = offsets >> 1

    a_vals = tl.load(A_ptr + halved_offsets, mask=mask)
    b_vals = tl.load(B_ptr + halved_offsets, mask=mask)

    result = tl.where(offsets % 2, b_vals, a_vals)
    tl.store(output_ptr + offsets, result, mask=mask)


# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    interleave_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    import sys
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0)
    N = 1024
    A = torch.randn(N, dtype=torch.float32, device=device)
    B = torch.randn(N, dtype=torch.float32, device=device)
    out = torch.zeros(2 * N, dtype=torch.float32, device=device)

    solve(A, B, out, N)

    expected = torch.stack([A, B], dim=1).reshape(-1)
    assert torch.equal(out, expected), (
        f"max abs err = {(out - expected).abs().max().item()}"
    )
    print(f"PASS [{device}] interleave N={N}")
