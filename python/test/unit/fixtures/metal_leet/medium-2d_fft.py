import math

import torch
import triton
import triton.language as tl


TILE = 8


@triton.jit
def _real_matmul_8x8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for _ in range(0, K_TILES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK * stride_ak
        b_ptrs += BLOCK * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def _combine_real_imag_kernel(
    rr_ptr,
    ii_ptr,
    ri_ptr,
    ir_ptr,
    out_r_ptr,
    out_i_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    rr = tl.load(rr_ptr + offs, mask=mask, other=0.0)
    ii = tl.load(ii_ptr + offs, mask=mask, other=0.0)
    ri = tl.load(ri_ptr + offs, mask=mask, other=0.0)
    ir = tl.load(ir_ptr + offs, mask=mask, other=0.0)

    tl.store(out_r_ptr + offs, rr - ii, mask=mask)
    tl.store(out_i_ptr + offs, ri + ir, mask=mask)


@triton.jit
def _combine_crop_interleaved_kernel(
    rr_ptr,
    ii_ptr,
    ri_ptr,
    ir_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    PAD_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (row < M) & (col < N)
    padded_offset = row * PAD_N + col

    rr = tl.load(rr_ptr + padded_offset, mask=mask, other=0.0)
    ii = tl.load(ii_ptr + padded_offset, mask=mask, other=0.0)
    ri = tl.load(ri_ptr + padded_offset, mask=mask, other=0.0)
    ir = tl.load(ir_ptr + padded_offset, mask=mask, other=0.0)

    out_offset = row * N * 2 + col * 2
    tl.store(out_ptr + out_offset, rr - ii, mask=mask)
    tl.store(out_ptr + out_offset + 1, ri + ir, mask=mask)


def _ceil_to_tile(x: int) -> int:
    return triton.cdiv(x, TILE) * TILE


def _empty_padded(device, rows: int, cols: int) -> torch.Tensor:
    return torch.empty((rows, cols), device=device, dtype=torch.float32).contiguous()


def _zero_padded(device, rows: int, cols: int) -> torch.Tensor:
    return torch.zeros((rows, cols), device=device, dtype=torch.float32).contiguous()


def _copy_signal_planes(signal: torch.Tensor, M: int, N: int, pad_m: int, pad_n: int):
    real = _zero_padded(signal.device, pad_m, pad_n)
    imag = _zero_padded(signal.device, pad_m, pad_n)
    real[:M, :N].copy_(signal[:M, :N, 0])
    imag[:M, :N].copy_(signal[:M, :N, 1])
    return real, imag


def _twiddle(rows: int, cols: int, inner: int, device):
    lhs = torch.arange(rows, device=device, dtype=torch.float32)
    rhs = torch.arange(cols, device=device, dtype=torch.float32)
    angle = -2.0 * math.pi * lhs[:, None] * rhs[None, :] / inner
    return torch.cos(angle).contiguous(), torch.sin(angle).contiguous()


def _copy_twiddle_padded(real: torch.Tensor, imag: torch.Tensor, rows: int, cols: int):
    padded_real = _zero_padded(real.device, rows, cols)
    padded_imag = _zero_padded(imag.device, rows, cols)
    padded_real[: real.shape[0], : real.shape[1]].copy_(real)
    padded_imag[: imag.shape[0], : imag.shape[1]].copy_(imag)
    return padded_real, padded_imag


def _real_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    M = out.shape[0]
    N = out.shape[1]
    K = a.shape[1]
    grid = (triton.cdiv(M, TILE), triton.cdiv(N, TILE))
    _real_matmul_8x8_kernel[grid](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK=TILE,
        K_TILES=K // TILE,
        num_warps=1,
    )


def _complex_matmul(a_r: torch.Tensor, a_i: torch.Tensor, b_r: torch.Tensor, b_i: torch.Tensor):
    out_shape = (a_r.shape[0], b_r.shape[1])
    rr = _empty_padded(a_r.device, *out_shape)
    ii = _empty_padded(a_r.device, *out_shape)
    ri = _empty_padded(a_r.device, *out_shape)
    ir = _empty_padded(a_r.device, *out_shape)

    _real_matmul(a_r, b_r, rr)
    _real_matmul(a_i, b_i, ii)
    _real_matmul(a_r, b_i, ri)
    _real_matmul(a_i, b_r, ir)
    return rr, ii, ri, ir


def _combine_real_imag(rr: torch.Tensor, ii: torch.Tensor, ri: torch.Tensor, ir: torch.Tensor):
    out_r = _empty_padded(rr.device, rr.shape[0], rr.shape[1])
    out_i = _empty_padded(rr.device, rr.shape[0], rr.shape[1])
    total = rr.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _combine_real_imag_kernel[grid](rr, ii, ri, ir, out_r, out_i, TOTAL=total, BLOCK=block)
    return out_r, out_i


def _combine_crop_interleaved(
    rr: torch.Tensor,
    ii: torch.Tensor,
    ri: torch.Tensor,
    ir: torch.Tensor,
    out: torch.Tensor,
    M: int,
    N: int,
):
    block_m = 16
    block_n = 16
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _combine_crop_interleaved_kernel[grid](
        rr,
        ii,
        ri,
        ir,
        out,
        M=M,
        N=N,
        PAD_N=rr.shape[1],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )


def _validate(signal: torch.Tensor, spectrum: torch.Tensor, M: int, N: int):
    expected_shape = (M, N, 2)
    if signal.shape != expected_shape:
        raise ValueError(f"signal must have shape {expected_shape}, got {tuple(signal.shape)}")
    if spectrum.shape != expected_shape:
        raise ValueError(f"spectrum must have shape {expected_shape}, got {tuple(spectrum.shape)}")
    if signal.dtype != torch.float32:
        raise TypeError(f"signal must be float32, got {signal.dtype}")
    if spectrum.dtype != torch.float32:
        raise TypeError(f"spectrum must be float32, got {spectrum.dtype}")
    if signal.device != spectrum.device:
        raise ValueError("signal and spectrum must be on the same device")
    if not signal.is_contiguous():
        raise ValueError("signal must be contiguous")
    if not spectrum.is_contiguous():
        raise ValueError("spectrum must be contiguous")


def solve(signal: torch.Tensor, spectrum: torch.Tensor, M: int, N: int):
    _validate(signal, spectrum, M, N)

    device = signal.device
    pad_m = _ceil_to_tile(M)
    pad_n = _ceil_to_tile(N)

    signal_r, signal_i = _copy_signal_planes(signal, M, N, pad_m, pad_n)

    row_r, row_i = _twiddle(N, N, N, device)
    row_r, row_i = _copy_twiddle_padded(row_r, row_i, pad_n, pad_n)
    rr, ii, ri, ir = _complex_matmul(signal_r, signal_i, row_r, row_i)
    temp_r, temp_i = _combine_real_imag(rr, ii, ri, ir)

    col_r, col_i = _twiddle(M, M, M, device)
    col_r, col_i = _copy_twiddle_padded(col_r, col_i, pad_m, pad_m)
    rr, ii, ri, ir = _complex_matmul(col_r, col_i, temp_r, temp_i)
    _combine_crop_interleaved(rr, ii, ri, ir, spectrum, M, N)


def _reference(signal: torch.Tensor) -> torch.Tensor:
    complex_signal = torch.view_as_complex(signal.contiguous())
    return torch.view_as_real(torch.fft.fft2(complex_signal)).contiguous()


def _run_case(M: int, N: int, seed: int):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("No GPU device (MPS or CUDA) available")
    signal_cpu = torch.randn((M, N, 2), dtype=torch.float32)
    signal = signal_cpu.to(device).contiguous()
    spectrum = torch.empty_like(signal)

    solve(signal, spectrum, M, N)
    if device == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()

    ref = _reference(signal_cpu)
    got = spectrum.cpu()
    max_abs = (got - ref).abs().max().item()
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-3)
    print(f"{M}x{N}: max_abs={max_abs:.6f}")


if __name__ == "__main__":
    for seed, shape in enumerate([(32, 32), (64, 64), (35, 41)]):
        _run_case(*shape, seed=0x2D00 + seed)
