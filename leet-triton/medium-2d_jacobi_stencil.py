import torch
import triton
import triton.language as tl

@triton.jit
def jacobi_stencil(
    input, stride_im, stride_in,
    output, stride_om, stride_on,
    rows, cols,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    m_offs = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    n_offs = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    top_ptrs = (
        input + 
        (m_offs[:, None] - 1) * stride_im + 
        n_offs[None, :] * stride_in
    )
    top_vals = tl.load(
        top_ptrs,
        mask = (
            ((m_offs[:, None] - 1) >= 0) &
            ((m_offs[:, None] - 1) < rows) &
            (n_offs[None, :] < cols)
        ),
        other = 0.0
    )
    bottom_ptrs = (
        input + 
        (m_offs[:, None] + 1) * stride_im + 
        n_offs[None, :] * stride_in
    )
    bottom_vals = tl.load(
        bottom_ptrs,
        mask = (
            ((m_offs[:, None] + 1) < rows) &
            (n_offs[None, :] < cols)
        ),
        other = 0.0
    )
    left_ptrs = (
        input +
        m_offs[:, None] * stride_im +
        (n_offs[None, :] - 1) * stride_in
    )
    left_vals = tl.load(
        left_ptrs,
        mask = (
            (m_offs[:, None] < rows) &
            ((n_offs[None, :] - 1) >= 0) &
            (n_offs[None, :] < cols)
        ),
        other = 0.0
    )
    right_ptrs = (
        input +
        m_offs[:, None] * stride_im +
        (n_offs[None, :] + 1) * stride_in
    )
    right_vals = tl.load(
        right_ptrs,
        mask = (
            (m_offs[:, None] < rows) &
            ((n_offs[None, :] + 1) < cols)
        ),
        other = 0.0
    )
    acc += 0.25 * (top_vals + bottom_vals + left_vals + right_vals)

    output_ptrs = (
        output +
        m_offs[:, None] * stride_om + 
        n_offs[None, :] * stride_on
    )
    output_mask = (
        (m_offs[:, None] >= 1) &
        (m_offs[:, None] < (rows - 1)) &
        (n_offs[None, :] >= 1) &
        (n_offs[None, :] < (cols - 1))
    )
    tl.store(output_ptrs, mask = output_mask, value = acc)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(cols, BLOCK_N), triton.cdiv(rows, BLOCK_M))
    output.copy_(input)
    jacobi_stencil[grid](
        input, input.stride(0), input.stride(1),
        output, output.stride(0), output.stride(1),
        rows, cols,
        BLOCK_M, BLOCK_N
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
    rows, cols = 70, 100  # non-multiples of BLOCK=32 exercise the masks
    inp = torch.randn(rows, cols, dtype=torch.float32, device=device).contiguous()
    out = torch.zeros(rows, cols, dtype=torch.float32, device=device).contiguous()

    solve(inp, out, rows, cols)

    # borders are copied through; interior is the 4-neighbour average
    expected = inp.clone()
    expected[1:-1, 1:-1] = 0.25 * (
        inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]
    )
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3), (
        f"max abs err = {(out - expected).abs().max().item()}"
    )
    print(f"PASS [{device}] jacobi_stencil {rows}x{cols}")
