import torch
import triton
import triton.language as tl

@triton.jit
def floyd_warshall(
    dist, stride_dm, stride_dn,
    output, stride_om, stride_on,
    N, k, BLOCK_N: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    i_offs = tl.arange(0, BLOCK_N) + pid_i * BLOCK_N
    j_offs = tl.arange(0, BLOCK_N) + pid_j * BLOCK_N
    k_offs = tl.full((BLOCK_N,), k, dtype=tl.int32)

    dist_ptrs = (
        dist +
        i_offs[:, None] * stride_dm +
        j_offs[None, :] * stride_dn
    )
    dist_mask = (i_offs[:, None] < N) & (j_offs[None, :] < N)
    dist_ij = tl.load(
        dist_ptrs,
        mask = dist_mask,
        other=float('inf')
    )

    dist_ik = tl.load(
        dist +
        i_offs[:, None] * stride_dm +
        k_offs[None, :] * stride_dn,
        mask = (i_offs[:, None] < N),
        other=float('inf')
    )

    dist_kj = tl.load(
        dist +
        k_offs[:, None] * stride_dm +
        j_offs[None, :] * stride_dn,
        mask = (j_offs[None, :] < N),
        other=float('inf')
    )

    out = tl.minimum(dist_ij, dist_ik + dist_kj)
    tl.store(
        dist_ptrs,
        mask=dist_mask,
        value=out
    )

# dist, output are tensors on the GPU
def solve(dist: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_N = 64
    dist = dist.view(N, N)
    output = output.view(N, N)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(N, BLOCK_N))
    for k in range(N):
        floyd_warshall[grid](
            dist, dist.stride(0), dist.stride(1),
            output, output.stride(0), output.stride(1),
            N,
            k,
            BLOCK_N,
        )
    output.copy_(dist)

