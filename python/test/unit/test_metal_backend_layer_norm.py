"""Layer-norm forward AND backward on the Metal backend (official tutorial 05).

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

BACKWARD is a Metal-appropriate port of the tutorial's two-stage design. The dx
computation is verbatim (two rank-1 reduces feeding an elementwise expression +
masked store). The dw/db accumulation, which the tutorial guards with a global
spin lock (`while atomic_cas(Lock,0,1): pass ... atomic_xchg(Lock,0)`), is
instead done LOCK-FREE with `tl.atomic_add`: a global spin lock cannot be ported
faithfully to Apple GPUs (no cross-threadgroup forward-progress guarantee, so it
can deadlock). Stage 1 atomically accumulates partial dw/db into GROUP_SIZE_M
f32 buckets (Metal atomics are f32); Stage 2 is the VERBATIM tutorial
`_layer_norm_bwd_dwdb` — a rank-2 axis=0 (per-column) reduce over two
loop-carried 2D accumulators, now supported on Metal (reassociateLoopCarried-
Axis0Reduce + lowerRank2Axis0Reduce). Stage 1 launches with num_warps = BLOCK/32
(tpb == BLOCK, elem-per-thread == 1) because a tensor atomic is always spt=1 and
the backend's single-E tile loop can't bridge the mixed-spt divergence coalescing
creates at E>1; Stage 2 runs at BLOCK_SIZE_N=128 == tpb (output E==1). Verified
vs torch.autograd across dtypes, non-pow2 N, and M > GROUP_SIZE_M (real atomic
contention).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

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


FUSED_RMS_NORM_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "metal_leet"
    / "medium-fused_residual_add_and_rms_norm.py"
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


# Power-of-2 N (E==1 at 128, E>1 at 256/1024) AND non-power-of-2 N (700/333) —
# the latter coalesces to a divergent spt=4↔spt=1 blocked convert_layout that is
# normalized away (cloning the CSE-shared mask-bound splat).
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(4, 128), (8, 256), (16, 1024), (3, 700),
                                  (5, 333)])
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
    tol = {torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 5e-2}[dtype]
    torch.testing.assert_close(y.cpu().float(), ref, atol=tol, rtol=tol)


def _load_fused_rms_norm_module():
    spec = importlib.util.spec_from_file_location(
        "fused_residual_add_rms_norm", FUSED_RMS_NORM_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stable_rms_norm_reference(x, residual, weight, eps):
    z = x.cpu().to(torch.float64) + residual.cpu().to(torch.float64)
    eps_sqrt = math.sqrt(eps)
    scale = torch.maximum(
        z.abs().amax(dim=1, keepdim=True),
        torch.tensor(eps_sqrt, dtype=torch.float64),
    )
    q = z / scale
    eps_scaled = (eps_sqrt / scale).square()
    return (
        q
        * torch.rsqrt((q * q).mean(dim=1, keepdim=True) + eps_scaled)
        * weight.cpu().to(torch.float64)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("C", [1, 333, 8191, 8192, 8193, 12289])
def test_fused_residual_add_rms_norm_original_solve_matches_torch(dtype, C):
    torch.manual_seed(23000 + C)
    module = _load_fused_rms_norm_module()
    N = 3
    dev = "mps"
    x = torch.randn(N, C, dtype=dtype, device=dev)
    residual = torch.randn(N, C, dtype=dtype, device=dev)
    weight = torch.randn(C, dtype=dtype, device=dev)
    out = torch.empty_like(x)
    eps = 1e-5

    module.solve(x, residual, weight, out, N, C, eps)
    torch.mps.synchronize()

    z = x.cpu().float() + residual.cpu().float()
    ref = z * torch.rsqrt((z * z).mean(dim=1, keepdim=True) + eps) * weight.cpu().float()
    tol = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-2}[dtype]
    assert out.shape == (N, C)
    assert out.dtype == dtype
    assert torch.isfinite(out.cpu().float()).all()
    torch.testing.assert_close(out.cpu().float(), ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_residual_add_rms_norm_dynamic_fallback_large_width(dtype):
    torch.manual_seed(65000)
    module = _load_fused_rms_norm_module()
    N = 2
    C = 65537
    dev = "mps"
    x = torch.randn(N, C, dtype=dtype, device=dev)
    residual = torch.randn(N, C, dtype=dtype, device=dev)
    weight = torch.randn(C, dtype=dtype, device=dev)
    out = torch.empty_like(x)
    eps = 1e-5

    module.solve(x, residual, weight, out, N, C, eps)
    torch.mps.synchronize()

    ref = _stable_rms_norm_reference(x, residual, weight, eps)
    tol = {torch.float32: 1e-4, torch.bfloat16: 5e-2}[dtype]
    assert out.shape == (N, C)
    assert out.dtype == dtype
    assert torch.isfinite(out.cpu().float()).all()
    torch.testing.assert_close(out.cpu().float(), ref.float(), atol=tol, rtol=tol)


def test_fused_residual_add_rms_norm_accepts_noncontiguous_readonly_inputs():
    torch.manual_seed(24001)
    module = _load_fused_rms_norm_module()
    N = 3
    C = 333
    dev = "mps"
    dtype = torch.float32
    x = torch.randn(C, N, dtype=dtype, device=dev).t()
    residual = torch.randn(C, N, dtype=dtype, device=dev).t()
    weight = torch.randn(C * 2, dtype=dtype, device=dev)[::2]
    out = torch.empty((N, C), dtype=dtype, device=dev)
    eps = 1e-5
    assert not x.is_contiguous()
    assert not residual.is_contiguous()
    assert not weight.is_contiguous()

    module.solve(x, residual, weight, out, N, C, eps)
    torch.mps.synchronize()

    ref = _stable_rms_norm_reference(x, residual, weight, eps)
    torch.testing.assert_close(out.cpu().float(), ref.float(), atol=1e-4, rtol=1e-4)


def test_fused_residual_add_rms_norm_accepts_stride_zero_weight():
    torch.manual_seed(24002)
    module = _load_fused_rms_norm_module()
    N = 2
    C = 8193
    dev = "mps"
    dtype = torch.float32
    x = torch.randn(N, C, dtype=dtype, device=dev)
    residual = torch.randn(N, C, dtype=dtype, device=dev)
    weight = torch.tensor([1.25], dtype=dtype, device=dev).expand(C)
    out = torch.empty_like(x)
    eps = 1e-5
    assert weight.stride(0) == 0

    module.solve(x, residual, weight, out, N, C, eps)
    torch.mps.synchronize()

    ref = _stable_rms_norm_reference(x, residual, weight, eps)
    torch.testing.assert_close(out.cpu().float(), ref.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("C", [333, 8193])
def test_fused_residual_add_rms_norm_stable_for_extreme_scales(dtype, C):
    module = _load_fused_rms_norm_module()
    N = 3
    dev = "mps"
    x_cpu = torch.zeros((N, C), dtype=torch.float32)
    residual_cpu = torch.zeros((N, C), dtype=torch.float32)
    x_cpu[0].fill_(1e20)
    x_cpu[2] = torch.linspace(-1e-20, 1e-20, C)
    weight_cpu = torch.ones(C, dtype=torch.float32)
    x = x_cpu.to(device=dev, dtype=dtype)
    residual = residual_cpu.to(device=dev, dtype=dtype)
    weight = weight_cpu.to(device=dev, dtype=dtype)
    out = torch.empty_like(x)
    eps = 1e-5

    module.solve(x, residual, weight, out, N, C, eps)
    torch.mps.synchronize()

    ref = _stable_rms_norm_reference(x, residual, weight, eps)
    tol = {torch.float32: 1e-4, torch.bfloat16: 5e-2}[dtype]
    got = out.cpu().float()
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, ref.float(), atol=tol, rtol=tol)
    torch.testing.assert_close(got[0], torch.ones(C), atol=tol, rtol=tol)
    torch.testing.assert_close(got[1], torch.zeros(C), atol=0, rtol=0)


# ===========================================================================
# Backward (Metal-appropriate two-stage atomic port of tutorial 05).
# ===========================================================================


@triton.jit
def _layer_norm_bwd_dx_atomic(DX, DY, DW, DB, X, W, Mean, Rstd, stride, N,
                              GROUP_SIZE_M: tl.constexpr,
                              BLOCK_SIZE_N: tl.constexpr):
    # Verbatim dx; lock-free atomic accumulation into f32 dw/db buckets.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    # Partial sums kept in f32 (Metal atomic_add is f32); the tutorial casts to
    # w.dtype only to shrink the bucket buffer.
    partial_dw = dy * xhat
    partial_db = dy
    tl.atomic_add(DW, partial_dw, mask=mask)
    tl.atomic_add(DB, partial_db, mask=mask)


@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Stage 2, VERBATIM tutorial-05: reduce the GROUP_SIZE_M partial buckets to
    # the final [N] gradients via a rank-2 axis=0 (per-column) reduce over two
    # loop-carried 2D accumulators. Runs on Metal via reassociateLoopCarried-
    # Axis0Reduce + lowerRank2Axis0Reduce (output E==1: BLOCK_SIZE_N == tpb).
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


def _layer_norm_backward_metal(x, w, dy, mean, rstd, N):
    M = x.shape[0]
    GROUP_SIZE_M = 64
    if N <= 4096:
        GROUP_SIZE_M = 128
    if N <= 1024:
        GROUP_SIZE_M = 256
    BLOCK = triton.next_power_of_2(N)
    # Launch with tpb == BLOCK (num_warps = BLOCK/32) so elem-per-thread == 1 and
    # the whole kernel stays a single spt=1 layout. A tensor atomic is always
    # spt=1 (atomics can't vectorize); at E>1 the coalescer would give the
    # surrounding loads/store spt>1, and the backend's single-E tile loop can't
    # bridge that mixed-spt divergence. E==1 makes coalescing a no-op. Caps N at
    # 1024 (32 warps = Metal's 1024-thread threadgroup limit) — fine for the
    # <=64KB/row layer norms this kernel targets.
    num_warps = min(max(BLOCK // 32, 1), 32)
    _dw = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device="mps")
    _db = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device="mps")
    dx = torch.empty_like(x)
    _layer_norm_bwd_dx_atomic[(M,)](
        dx, dy, _dw, _db, x, w, mean, rstd, x.stride(0), N,
        GROUP_SIZE_M=GROUP_SIZE_M, BLOCK_SIZE_N=BLOCK, num_warps=num_warps)
    # Stage 2: verbatim tutorial-05 dw/db reduction (rank-2 axis=0 reduce over
    # the GROUP_SIZE_M partial buckets). BLOCK_SIZE_N=128 == tpb (num_warps=4)
    # so the output is E==1; grid tiles the N columns.
    G = min(GROUP_SIZE_M, M)
    dw = torch.empty(N, dtype=torch.float32, device="mps")
    db = torch.empty(N, dtype=torch.float32, device="mps")
    grid = (triton.cdiv(N, 128),)
    _layer_norm_bwd_dwdb[grid](_dw, _db, dw, db, G, N, BLOCK_SIZE_M=32,
                               BLOCK_SIZE_N=128, num_warps=4)
    torch.mps.synchronize()
    return dx, dw, db


# M > GROUP_SIZE_M at (300,1024) [GROUP=256] and (400,512) exercises real
# multi-row atomic contention on a shared bucket; non-pow2 N at 700/333.
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(4, 128), (8, 256), (16, 1024), (3, 700),
                                  (5, 333), (300, 1024), (400, 512)])
def test_layer_norm_backward(dtype, M, N):
    torch.manual_seed(M * 4099 + N)
    dev = "mps"
    x = torch.randn(M, N, dtype=dtype, device=dev)
    w = torch.randn(N, dtype=dtype, device=dev)
    b = torch.randn(N, dtype=dtype, device=dev)
    dy = torch.randn(M, N, dtype=dtype, device=dev)
    eps = 1e-5
    mean = x.float().mean(1)
    rstd = 1.0 / torch.sqrt(x.float().var(1, unbiased=False) + eps)

    dx, dw, db = _layer_norm_backward_metal(x, w, dy, mean, rstd, N)

    # Reference via torch autograd, in fp32 on CPU for a stable oracle.
    xc = x.cpu().float().detach().requires_grad_(True)
    wc = w.cpu().float().detach().requires_grad_(True)
    bc = b.cpu().float().detach().requires_grad_(True)
    yc = torch.nn.functional.layer_norm(xc, (N,), wc, bc, eps)
    yc.backward(dy.cpu().float())

    # dw/db sum over M rows, so tolerances scale a bit with M and precision.
    tol = {torch.float32: (1e-3, 1e-3),
           torch.float16: (1e-2, 1e-2),
           torch.bfloat16: (5e-2, 5e-2)}[dtype]
    atol, rtol = tol
    torch.testing.assert_close(dx.cpu().float(), xc.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dw.cpu(), wc.grad, atol=atol * M, rtol=rtol)
    torch.testing.assert_close(db.cpu(), bc.grad, atol=atol * M, rtol=rtol)
