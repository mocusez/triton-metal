import torch
import triton
import triton.language as tl


@triton.jit
def subarray_sum_kernel(input_ptr,  output_ptr,
                        N, M,
                        S_ROW,E_ROW, S_COL, E_COL,
                        BLOCK_SIZE: tl.constexpr,
                        BLOCK_SIZE_COL: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offset_row = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + S_ROW
    offset_col = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL) + S_COL

    mask_row = offset_row <= E_ROW
    mask_col = offset_col <= E_COL

    offset = offset_row[:,None] * M + offset_col[None,:]
    mask = mask_row[:,None] & mask_col[None,:]

    input_data = tl.load(input_ptr + offset, mask=mask)
    input_data_sum = input_data.sum()

    if input_data_sum > 0:
        tl.atomic_add(output_ptr, input_data.sum())


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_SIZE = 1
    BLOCK_SIZE_COL = 1024

    grid = (triton.cdiv(E_ROW - S_ROW + 1, BLOCK_SIZE), triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE_COL))
    subarray_sum_kernel[grid](input, output, N, M, S_ROW, E_ROW, S_COL, E_COL, BLOCK_SIZE, BLOCK_SIZE_COL, num_warps = 4)


if __name__ == "__main__":
    import sys
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0)
    N, M = 100, 2000
    # non-multiple-of-1024 column span exercises the col mask
    S_ROW, E_ROW, S_COL, E_COL = 10, 80, 5, 1500
    # positive inputs: the kernel only atomic-adds block sums that are > 0
    inp = torch.rand(N, M, dtype=torch.float32, device=device).contiguous()
    out = torch.zeros(1, dtype=torch.float32, device=device)

    solve(inp, out, N, M, S_ROW, E_ROW, S_COL, E_COL)

    # reference on CPU (MPS has no float64)
    expected = inp.cpu().double()[S_ROW:E_ROW + 1, S_COL:E_COL + 1].sum().item()
    got = out.cpu().double().item()
    assert abs(got - expected) <= 1e-2 * abs(expected) + 1.0, (
        f"got={got} expected={expected}"
    )
    print(f"PASS [{device}] 2D_subarray_sum N={N} M={M} "
          f"rows[{S_ROW},{E_ROW}] cols[{S_COL},{E_COL}]")

