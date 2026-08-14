"""Sparse matrix-vector multiplication with a true COO execution path.

``solve`` accepts either a 2-D ``torch.sparse_coo_tensor`` or the historical
strided dense tensor. Repeated callers can use ``prepare_coo`` once, then call
``solve_coo`` with the returned GPU-resident buffers to avoid repacking.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dense_spmv_kernel(A, x, y, M, N, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offset_m < M
    total = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for pid_n in range(0, tl.cdiv(N, BLOCK_N)):
        offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offset_n < N
        vals_a = tl.load(
            A + offset_m[:, None] * N + offset_n[None, :],
            mask=(mask_m[:, None] & mask_n[None, :]),
            other=0.0,
        )
        vals_x = tl.load(x + offset_n, mask=mask_n, other=0.0)
        total += tl.sum(vals_a * vals_x[None, :], axis=1)
    tl.store(y + offset_m, total, mask=mask_m)


@triton.jit
def zero_output_kernel(output, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output + offsets, 0.0, mask=offsets < size)


@triton.jit
def coo_spmv_kernel(
    row_indices,
    col_indices,
    values,
    x,
    y,
):
    sparse_offset = tl.program_id(0)
    row = tl.load(row_indices + sparse_offset)
    col = tl.load(col_indices + sparse_offset)
    sparse_value = tl.load(values + sparse_offset)
    x_value = tl.load(x + col)
    tl.atomic_add(y + row, sparse_value * x_value)


def solve_coo(
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    nnz: int,
):
    """Multiply trusted ``prepare_coo`` buffers in O(nnz) core work.

    Index ranges are not rechecked here because doing so would synchronize the
    device. Callers constructing these buffers directly must guarantee
    ``0 <= row < M`` and ``0 <= col < N``.
    """
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive")
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
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("x and y must use torch.float32")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    if x.numel() != N or y.numel() != M:
        raise ValueError("x/y sizes do not match M and N")
    if not (
        row_indices.device == col_indices.device == values.device
        == x.device == y.device
    ):
        raise ValueError("all prepacked COO buffers must share one device")

    ZERO_BLOCK = 256
    zero_output_kernel[(triton.cdiv(M, ZERO_BLOCK),)](
        y, M, BLOCK=ZERO_BLOCK
    )
    if nnz == 0:
        return

    coo_spmv_kernel[(nnz,)](
        row_indices,
        col_indices,
        values,
        x,
        y,
        num_warps=1,
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
    x: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    nnz: int,
):
    """Run true COO SpMV, with a strided dense compatibility fallback."""
    if A.layout == torch.sparse_coo:
        row_indices, col_indices, values = prepare_coo(
            A, x.device, M, N, nnz
        )
        packed_nnz = values.numel()
        solve_coo(
            row_indices,
            col_indices,
            values,
            x,
            y,
            M,
            N,
            packed_nnz,
        )
        return
    if A.layout != torch.strided:
        raise TypeError("A must be a strided or sparse COO tensor")
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive")
    if tuple(A.shape) != (M, N):
        raise ValueError("dense A shape does not match M and N")
    if tuple(x.shape) != (N,) or tuple(y.shape) != (M,):
        raise ValueError("x/y shapes do not match M and N")
    if A.dtype != torch.float32 or x.dtype != torch.float32 \
            or y.dtype != torch.float32:
        raise TypeError("dense A, x, and y must use torch.float32")
    if not A.is_contiguous() or not x.is_contiguous() \
            or not y.is_contiguous():
        raise ValueError("dense A, x, and y must be contiguous")
    if not (A.device == x.device == y.device):
        raise ValueError("dense A, x, and y must share one device")

    BLOCK_M = 1
    BLOCK_N = 1024
    grid = (triton.cdiv(M, BLOCK_M),)
    dense_spmv_kernel[grid](A, x, y, M, N, BLOCK_M, BLOCK_N)


if __name__ == "__main__":
    import sys

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0x5A5A)
    cases = ((1, 1, 0.0), (3, 5, 0.3), (5, 1031, 0.05))
    for M, N, density in cases:
        A_cpu = torch.randn((M, N), dtype=torch.float32)
        A_cpu *= torch.rand((M, N)) < density
        A_sparse = A_cpu.to_sparse_coo().coalesce()
        x_cpu = torch.randn((N,), dtype=torch.float32)
        y = torch.empty((M,), dtype=torch.float32, device=device)

        solve(
            A_sparse,
            x_cpu.to(device),
            y,
            M,
            N,
            A_sparse._nnz(),
        )
        if device == "mps":
            torch.mps.synchronize()
        torch.testing.assert_close(
            y.cpu(), A_cpu @ x_cpu, atol=1e-4, rtol=1e-4
        )

    print(f"PASS [{device}] COO sparse_matrix_vector cases={list(cases)}")
