import torch
import triton
import triton.language as tl

@triton.jit
def subarray_sum_kernel(input_ptr, output_ptr,
                        N, M, K,
                        S_DEP, E_DEP,
                        S_ROW, E_ROW,
                        S_COL, E_COL,
                        BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_M: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    offset_0 = pid0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) + S_DEP
    offset_1 = pid1 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) + S_ROW
    offset_2 = pid2 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) + S_COL

    mask_0 = offset_0 <= E_DEP
    mask_1 = offset_1 <= E_ROW
    mask_2 = offset_2 <= E_COL

    offset = offset_0[:, None, None] * M * K + offset_1[None,:,None] * K + offset_2[None, None, :]
    mask = mask_0[:,None, None] & mask_1[None,:,None] & mask_2[None,None,:]

    input_data = tl.load(input_ptr + offset, mask=mask)
    input_data_sum = input_data.sum()

    if input_data_sum != 0:
        tl.atomic_add(output_ptr, input_data_sum)


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    M: int,
    K: int,
    S_DEP: int,
    E_DEP: int,
    S_ROW: int,
    E_ROW: int,
    S_COL: int,
    E_COL: int,
):
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_K = 1024

    grid = (triton.cdiv(E_DEP - S_DEP + 1, BLOCK_SIZE_N), triton.cdiv(E_ROW - S_ROW + 1, BLOCK_SIZE_M), triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE_K))
    subarray_sum_kernel[grid](input, output,
                            N, M, K,
                            S_DEP, E_DEP,
                            S_ROW, E_ROW,
                            S_COL, E_COL,
                            BLOCK_SIZE_N,
                            BLOCK_SIZE_M,
                            BLOCK_SIZE_K,
                            num_warps = 4)


if __name__ == "__main__":
    import sys
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0)
    N, M, K = 8, 16, 2000
    S_DEP, E_DEP = 1, 6
    S_ROW, E_ROW = 2, 12
    # non-multiple-of-1024 column span exercises the col mask
    S_COL, E_COL = 3, 1500
    # positive inputs keep the magnitude well away from zero for a robust
    # relative comparison (kernel skips exactly-zero block sums via != 0)
    inp = torch.rand(N, M, K, dtype=torch.float32, device=device).contiguous()
    out = torch.zeros(1, dtype=torch.float32, device=device)

    solve(inp, out, N, M, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL)

    # reference on CPU (MPS has no float64)
    expected = inp.cpu().double()[
        S_DEP:E_DEP + 1, S_ROW:E_ROW + 1, S_COL:E_COL + 1
    ].sum().item()
    got = out.cpu().double().item()
    assert abs(got - expected) <= 1e-2 * abs(expected) + 1.0, (
        f"got={got} expected={expected}"
    )
    print(f"PASS [{device}] 3D_subarray_sum N={N} M={M} K={K} "
          f"dep[{S_DEP},{E_DEP}] rows[{S_ROW},{E_ROW}] cols[{S_COL},{E_COL}]")
