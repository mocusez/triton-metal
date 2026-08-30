"""Sparse-dense matrix multiplication with a true COO execution path.

``solve`` accepts either a 2-D ``torch.sparse_coo_tensor`` or the historical
strided dense tensor. Repeated callers can use ``prepare_coo`` once, then call
``solve_coo`` with the returned GPU-resident buffers to avoid repacking.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dense_matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_am,
    stride_an,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, N, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        a_block = a + (
            offs_am[:, None] * stride_am + offs_n[None, :] * stride_an
        )
        b_block = b + (
            offs_n[:, None] * stride_bn + offs_bk[None, :] * stride_bk
        )
        a_val = tl.load(a_block, mask=offs_n[None, :] < N, other=0.0)
        b_val = tl.load(b_block, mask=offs_n[:, None] < N, other=0.0)
        accumulator = tl.dot(
            a_val, b_val, acc=accumulator, allow_tf32=False
        )

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_block = c + stride_cm * offs_cm[:, None] + stride_ck * offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_block, accumulator, mask=c_mask)


@triton.jit
def zero_output_kernel(output, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output + offsets, 0.0, mask=offsets < size)


@triton.jit
def coo_spmm_kernel(
    row_indices,
    col_indices,
    values,
    B,
    C,
    K,
    BLOCK_K: tl.constexpr,
):
    sparse_offset = tl.program_id(0)
    output_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    output_mask = output_offsets < K

    row = tl.load(row_indices + sparse_offset)
    inner = tl.load(col_indices + sparse_offset)
    sparse_value = tl.load(values + sparse_offset)
    dense_values = tl.load(
        B + inner * K + output_offsets, mask=output_mask, other=0.0
    )
    tl.atomic_add(
        C + row * K + output_offsets,
        sparse_value * dense_values,
        mask=output_mask,
    )


def solve_coo(
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    nnz: int,
):
    """Multiply trusted ``prepare_coo`` buffers in O(nnz * K) work.

    Index ranges are not rechecked here because doing so would synchronize the
    device. Callers constructing these buffers directly must guarantee
    ``0 <= row < M`` and ``0 <= col < N``.
    """
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError("M, N, and K must be positive")
    if nnz < 0:
        raise ValueError("nnz must be non-negative")
    if row_indices.numel() != nnz or col_indices.numel() != nnz:
        raise ValueError("COO index lengths must equal nnz")
    if values.numel() != nnz:
        raise ValueError("COO values length must equal nnz")
    if row_indices.dtype != torch.int32 or col_indices.dtype != torch.int32:
        raise TypeError("COO indices must use torch.int32 on the GPU")
    if values.dtype != torch.float32:
        raise TypeError("COO values must use torch.float32")
    if B.dtype != torch.float32 or C.dtype != torch.float32:
        raise TypeError("B and C must use torch.float32")
    if not B.is_contiguous() or not C.is_contiguous():
        raise ValueError("B and C must be contiguous row-major tensors")
    if B.numel() != N * K or C.numel() != M * K:
        raise ValueError("B/C sizes do not match M, N, and K")
    if not (
        row_indices.device == col_indices.device == values.device
        == B.device == C.device
    ):
        raise ValueError("all prepacked COO buffers must share one device")

    ZERO_BLOCK = 256
    output_size = M * K
    zero_output_kernel[(triton.cdiv(output_size, ZERO_BLOCK),)](
        C, output_size, BLOCK=ZERO_BLOCK
    )
    if nnz == 0:
        return

    BLOCK_K = 256
    coo_spmm_kernel[(nnz, triton.cdiv(K, BLOCK_K))](
        row_indices,
        col_indices,
        values,
        B,
        C,
        K,
        BLOCK_K=BLOCK_K,
    )


def prepare_coo(
    A: torch.Tensor,
    device: torch.device | str,
    M: int,
    N: int,
    nnz: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coalesce a CPU COO tensor and pack its components for the GPU.

    ``nnz`` is the input tensor's stored-entry count. Duplicate coordinates
    may reduce the number of buffers returned after coalescing.
    """
    if A.layout != torch.sparse_coo:
        raise TypeError("A must be a sparse COO tensor")
    if tuple(A.shape) != (M, N):
        raise ValueError("sparse A shape does not match M and N")
    if A._nnz() != nnz:
        raise ValueError("nnz must equal the sparse input's stored-entry count")
    if A.dtype != torch.float32:
        raise TypeError("sparse A must use torch.float32")
    A = A.coalesce()
    indices = A.indices()
    return (
        indices[0].to(device=device, dtype=torch.int32).contiguous(),
        indices[1].to(device=device, dtype=torch.int32).contiguous(),
        A.values().to(device=device, dtype=torch.float32).contiguous(),
    )


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    nnz: int,
):
    """Run true COO SpMM, with a strided dense compatibility fallback."""
    if A.layout == torch.sparse_coo:
        row_indices, col_indices, values = prepare_coo(
            A, B.device, M, N, nnz
        )
        packed_nnz = values.numel()
        solve_coo(
            row_indices,
            col_indices,
            values,
            B,
            C,
            M,
            N,
            K,
            packed_nnz,
        )
        return
    if A.layout != torch.strided:
        raise TypeError("A must be a strided or sparse COO tensor")
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError("M, N, and K must be positive")
    if tuple(A.shape) != (M, N):
        raise ValueError("dense A shape does not match M and N")
    if tuple(B.shape) != (N, K) or tuple(C.shape) != (M, K):
        raise ValueError("B/C shapes do not match M, N, and K")
    if A.dtype != torch.float32 or B.dtype != torch.float32 \
            or C.dtype != torch.float32:
        raise TypeError("dense A, B, and C must use torch.float32")
    if not A.is_contiguous() or not B.is_contiguous() \
            or not C.is_contiguous():
        raise ValueError("dense A, B, and C must be contiguous row-major tensors")
    if not (A.device == B.device == C.device):
        raise ValueError("dense A, B, and C must share one device")

    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(K, meta["BLOCK_SIZE_K"]),
    )
    dense_matrix_multiplication_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        stride_am,
        stride_an,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_ck,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
    )


if __name__ == "__main__":
    import sys

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0x5B5B)
    cases = ((8, 8, 8, 0.25), (64, 64, 64, 0.1),
             (70, 66, 96, 0.05), (4, 7, 5, 0.0))
    for M, N, K, density in cases:
        A_cpu = torch.randn((M, N), dtype=torch.float32)
        A_cpu *= torch.rand((M, N)) < density
        A_sparse = A_cpu.to_sparse_coo().coalesce()
        B_cpu = torch.randn((N, K), dtype=torch.float32)
        C = torch.empty((M, K), dtype=torch.float32, device=device)

        solve(
            A_sparse,
            B_cpu.to(device),
            C,
            M,
            N,
            K,
            A_sparse._nnz(),
        )
        if device == "mps":
            torch.mps.synchronize()
        torch.testing.assert_close(
            C.cpu(), A_cpu @ B_cpu, atol=1e-3, rtol=1e-3
        )

    print(f"PASS [{device}] COO sparse_matrix_dense_matrix cases={list(cases)}")
