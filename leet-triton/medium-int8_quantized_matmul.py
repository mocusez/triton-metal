import torch
import triton
import triton.language as tl


@triton.jit
def int8_quant_matmul_kernel(a_ptr, b_ptr, c_ptr,
                             M, N, K,
                             scale_A, scale_B, scale_C,
                             zero_point_A, zero_point_B, zero_point_C,
                             BLOCK_SIZE_M: tl.constexpr,
                             BLOCK_SIZE_N: tl.constexpr,
                             BLOCK_SIZE_K: tl.constexpr,
                             GROUPSIZE: tl.constexpr):
    hw_pid0 = tl.program_id(0)
    hw_pid1 = tl.program_id(1)

    num_programs_pid0 = tl.num_programs(0)
    num_programs_pid1 = tl.num_programs(1)

    pid0, pid1 = tl.swizzle2d(hw_pid0, hw_pid1, num_programs_pid0, num_programs_pid1, GROUPSIZE)

    offset_M = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_N = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_K = tl.arange(0, BLOCK_SIZE_K)

    mask_M = offset_M < M
    mask_N = offset_N < N
    mask_K = offset_K < K

    offset_A = offset_M[:, None] * K + offset_K[None, :]
    mask_A = mask_M[:, None] & mask_K[None, :]

    offset_B = offset_K[:, None] * N + offset_N[None, :]
    mask_B = mask_K[:, None] & mask_N[None, :]

    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.int32)
    accumulator_sum_A = tl.zeros([BLOCK_SIZE_M], dtype=tl.int32)
    accumulator_sum_B = tl.zeros([BLOCK_SIZE_N], dtype=tl.int32)

    scale_AB = scale_A * scale_B / scale_C

    for current_k_index in range(0, K, BLOCK_SIZE_K):
        if current_k_index + BLOCK_SIZE_K >= K:
                current_mask_K = (offset_K + current_k_index) < K
                mask_A = mask_M[:, None] & current_mask_K[None, :]
                mask_B = current_mask_K[:, None] & mask_N[None, :]

        data_A = tl.load(a_ptr + offset_A, mask=mask_A).to(tl.float32)
        data_B = tl.load(b_ptr + offset_B, mask=mask_B).to(tl.float32)

        accumulator += tl.dot(data_A, data_B).to(tl.int32)
        accumulator_sum_A += data_A.sum(1).to(tl.int32)
        accumulator_sum_B += data_B.sum(0).to(tl.int32)

        offset_A += BLOCK_SIZE_K
        offset_B += BLOCK_SIZE_K * N

    result = accumulator - (accumulator_sum_A[:, None] * zero_point_B) - (accumulator_sum_B[None, :] * zero_point_A) + (K * zero_point_A * zero_point_B)
    result = result.to(tl.float32) * scale_AB
    result = tl.floor(result + 0.5) + zero_point_C
    result = tl.clamp(result, -128, 127)

    offset_C = offset_M[:, None] * N + offset_N[None, :]
    mask_C = mask_M[:, None] & mask_N[None, :]

    tl.store(c_ptr + offset_C, result.to(tl.int8), mask=mask_C)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int, scale_A: float, scale_B: float, scale_C: float, zero_point_A: int, zero_point_B: int, zero_point_C: int):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE = 8

    grid = (triton.cdiv(M,BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    int8_quant_matmul_kernel[grid](a, b, c,
                                   M, N, K, 
                                   scale_A, scale_B, scale_C,
                                   zero_point_A, zero_point_B, zero_point_C,
                                   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE)