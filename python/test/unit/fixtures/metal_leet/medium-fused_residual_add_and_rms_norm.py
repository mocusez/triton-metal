import math

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_fused_kernel(
    x_ptr, res_ptr, w_ptr, scale_ptr, out_ptr,
    C, eps,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < C

    x = tl.load(x_ptr + row * C + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + row * C + cols, mask=mask, other=0.0).to(tl.float32)
    z = x + r

    eps_sqrt = tl.sqrt(eps)
    scale = tl.load(scale_ptr + row).to(tl.float32)
    q = z / scale
    eps_scaled = eps_sqrt / scale
    eps_scaled = eps_scaled * eps_scaled
    ms = tl.sum(q * q, axis=0) / C
    rstd = 1.0 / tl.sqrt(ms + eps_scaled)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * C + cols, q * rstd * w, mask=mask)


@triton.jit
def _add_rmsnorm_fused_loop_kernel(
    x_ptr, res_ptr, w_ptr, scale_ptr, out_ptr,
    C, eps,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    eps_sqrt = tl.sqrt(eps)
    scale = tl.load(scale_ptr + row).to(tl.float32)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, C, BLOCK):
        offs = start + cols
        mask = offs < C
        x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        z = x + r
        q = z / scale
        acc += q * q

    ms = tl.sum(acc, axis=0) / C
    eps_scaled = eps_sqrt / scale
    eps_scaled = eps_scaled * eps_scaled
    rstd = 1.0 / tl.sqrt(ms + eps_scaled)
    for start in range(0, C, BLOCK):
        offs = start + cols
        mask = offs < C
        x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * C + offs, ((x + r) / scale) * rstd * w, mask=mask)

_MAX_SINGLE_BLOCK = 8192


def _validate_solve_inputs(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    for name, value in (("N", N), ("C", C)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if N <= 0 or C <= 0:
        raise ValueError("N and C must be positive")
    if x.shape != (N, C):
        raise ValueError("x shape must match (N, C)")
    if residual.shape != (N, C):
        raise ValueError("residual shape must match (N, C)")
    if weight.shape != (C,):
        raise ValueError("weight shape must match (C,)")
    if out.shape != (N, C):
        raise ValueError("out shape must match (N, C)")
    if (
        x.device != residual.device
        or x.device != weight.device
        or x.device != out.device
    ):
        raise ValueError("x, residual, weight, and out must be on the same device")
    if x.dtype != residual.dtype or x.dtype != weight.dtype or x.dtype != out.dtype:
        raise ValueError("x, residual, weight, and out must have the same dtype")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("dtype must be float32, float16, or bfloat16")
    try:
        eps_value = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("eps must be finite and positive") from exc
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps must be finite and positive")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")


# x, residual, weight, out are tensors on the GPU
def solve(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    _validate_solve_inputs(x, residual, weight, out, N, C, eps)
    if not x.is_contiguous():
        x = x.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    scale = (x.float() + residual.float()).abs().amax(dim=1).clamp_min(math.sqrt(eps))
    block = triton.next_power_of_2(C)
    if block <= _MAX_SINGLE_BLOCK:
        num_warps = max(1, min(32, block // 512))
        _add_rmsnorm_fused_kernel[(N,)](
            x, residual, weight, scale, out,
            C, eps,
            BLOCK=block,
            num_warps=num_warps,
        )
    else:
        _add_rmsnorm_fused_loop_kernel[(N,)](
            x, residual, weight, scale, out,
            C, eps,
            BLOCK=4096,
            num_warps=8
        )
