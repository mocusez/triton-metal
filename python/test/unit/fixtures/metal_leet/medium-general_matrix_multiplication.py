
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a,b,c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, TILE_SIZE: tl.constexpr):
  bx = tl.program_id(0)
  by = tl.program_id(1)

  ar = tl.arange(0, TILE_SIZE)

  row = by * TILE_SIZE
  col = bx * TILE_SIZE

  iters = tl.cdiv(K, TILE_SIZE)
  output = (ar[:, None] * ar[None, :]) * 0.0
  output = tl.cast(output, tl.float32)
  ay_off = ar[:, None] + row
  bx_off = ar[None, :] + col

  ay_off_mask = ay_off < M
  bx_off_mask = bx_off < N

  for i in range(iters):
    ax_off = ar[None, :] + (i * TILE_SIZE)
    by_off = ar[:, None] + (i * TILE_SIZE)
    adata = tl.load(a + ay_off * K + ax_off, mask=(ax_off < K) & ay_off_mask, other=0.0)
    bdata = tl.load(b + by_off * N + bx_off, mask=bx_off_mask & (by_off < K), other=0.0)
    output = tl.dot(tl.cast(adata, tl.float32), tl.cast(bdata, tl.float32), acc=output)

  c_offset = c + ay_off * N + bx_off
  c_mask = ay_off_mask & bx_off_mask
  cdata = tl.load(c_offset, mask=c_mask, other=0.0)
  output = output * alpha + tl.cast(cdata, tl.float32) * beta
  output = tl.cast(output, tl.float16)
  tl.store(c_offset, output, mask=c_mask)


# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    TILE_SIZE = 64
    rows = triton.cdiv(M, TILE_SIZE)
    cols = triton.cdiv(N, TILE_SIZE)

    grid =(cols, rows)
    kernel[grid](
        a,b,c, M=M, N=N, K=K, alpha=alpha, beta=beta, TILE_SIZE=TILE_SIZE
    )
