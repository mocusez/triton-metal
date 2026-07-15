"""Layer-norm forward on the Metal backend (official tutorial 05).

The fused forward kernel computes per-row mean and variance by accumulating a
tensor across a loop (`_mean += x`, `_var += (x-mean)^2`) and reducing it, then
normalizes. This exercises several Metal-backend features together:

  - reduce over a LOOP-CARRIED tensor accumulator at BLOCK>tpb, reassociated to a
    scalar-accumulating loop `sum(acc += delta) -> s += sum(delta)` (addf is
    associative) so the reduce is over a device-rooted cone;
  - per-row masked loads/stores in a multi-program launch (grid=(M,)), whose mask
    reads the actual per-thread column cone (not a global-flat index);
  - fp16 in / fp32 compute / fp16 out via arith.extf / arith.truncf.

Verified vs a torch reference in both fp32 (tight) and fp16 (loose) across E==1
(BLOCK==tpb) and E>1 (BLOCK>tpb) row widths.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


@triton.jit
def _layer_norm_fwd(X, Y, W, B, Mean, Rstd, stride, N, eps,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y, mask=mask)


# Power-of-2 N: E==1 (128), E>1 (256/512/1024). Non-power-of-2 N hits a separate
# ttg.convert_layout (spt>1 staged-transpose) limitation, unrelated to layer-norm.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M, N", [(4, 128), (8, 256), (2, 512), (16, 1024)])
def test_layer_norm_forward(dtype, M, N):
    torch.manual_seed(M * 1000 + N)
    dev = "mps"
    x = torch.randn(M, N, dtype=dtype, device=dev)
    w = torch.randn(N, dtype=dtype, device=dev)
    b = torch.randn(N, dtype=dtype, device=dev)
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=dev)
    rstd = torch.empty(M, dtype=torch.float32, device=dev)
    eps = 1e-5
    BLOCK = triton.next_power_of_2(N)
    _layer_norm_fwd[(M,)](x, y, w, b, mean, rstd, x.stride(0), N, eps,
                          BLOCK_SIZE=BLOCK)
    torch.mps.synchronize()

    xc = x.cpu().float()
    mu = xc.mean(1, keepdim=True)
    var = xc.var(1, unbiased=False, keepdim=True)
    ref = (xc - mu) / torch.sqrt(var + eps) * w.cpu().float() + b.cpu().float()
    tol = 5e-3 if dtype == torch.float16 else 1e-3
    torch.testing.assert_close(y.cpu().float(), ref, atol=tol, rtol=tol)
