import torch
import triton
import triton.language as tl
import math

@triton.jit
def row_dft_kernel(
    signal_ptr, temp_ptr,
    w_r_ptr, w_i_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    1D DFT along rows using Precomputed Twiddle Factors.
    """
    m_start = tl.program_id(0) * BLOCK_M
    k_start = tl.program_id(1) * BLOCK_K

    m = m_start + tl.arange(0, BLOCK_M)[:, None]
    k = k_start + tl.arange(0, BLOCK_K)[None, :]

    acc_real = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_w = n_start + tl.arange(0, BLOCK_N)[:, None]
        n_v = n_start + tl.arange(0, BLOCK_N)[None, :]

        # Load Input Signal [BLOCK_M, BLOCK_N]
        x_offset = m * N + n_v
        x_mask = (m < M) & (n_v < N)
        sig_r = tl.load(signal_ptr + 2 * x_offset, mask=x_mask, other=0.0)
        sig_i = tl.load(signal_ptr + 2 * x_offset + 1, mask=x_mask, other=0.0)

        # Load Precomputed Weights [BLOCK_N, BLOCK_K]
        w_offset = n_w * N + k
        w_mask = (n_w < N) & (k < N)
        w_r = tl.load(w_r_ptr + w_offset, mask=w_mask, other=0.0)
        w_i = tl.load(w_i_ptr + w_offset, mask=w_mask, other=0.0)

        # Complex Dot Product: (xr + i xi) * (wr + i wi)
        acc_real += tl.dot(sig_r, w_r) - tl.dot(sig_i, w_i)
        acc_imag += tl.dot(sig_r, w_i) + tl.dot(sig_i, w_r)

    # Store temp results
    out_offset = m * N + k
    out_mask = (m < M) & (k < N)
    tl.store(temp_ptr + 2 * out_offset, acc_real, mask=out_mask)
    tl.store(temp_ptr + 2 * out_offset + 1, acc_imag, mask=out_mask)


@triton.jit
def col_dft_kernel(
    temp_ptr, spectrum_ptr,
    w_r_ptr, w_i_ptr,
    M, N,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    """
    1D DFT along columns using Precomputed Twiddle Factors.
    """
    k_start = tl.program_id(0) * BLOCK_K
    n_start = tl.program_id(1) * BLOCK_N

    k = k_start + tl.arange(0, BLOCK_K)[:, None]
    n_out = n_start + tl.arange(0, BLOCK_N)[None, :]

    acc_real = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_v = m_start + tl.arange(0, BLOCK_M)[None, :]
        m_x = m_start + tl.arange(0, BLOCK_M)[:, None]

        # Load Precomputed Weights [BLOCK_K, BLOCK_M]
        w_offset = k * M + m_v
        w_mask = (k < M) & (m_v < M)
        w_r = tl.load(w_r_ptr + w_offset, mask=w_mask, other=0.0)
        w_i = tl.load(w_i_ptr + w_offset, mask=w_mask, other=0.0)

        # Load Temp Signal [BLOCK_M, BLOCK_N]
        x_offset = m_x * N + n_out
        x_mask = (m_x < M) & (n_out < N)
        x_r = tl.load(temp_ptr + 2 * x_offset, mask=x_mask, other=0.0)
        x_i = tl.load(temp_ptr + 2 * x_offset + 1, mask=x_mask, other=0.0)

        # Complex Dot Product: (wr + i wi) * (xr + i xi)
        acc_real += tl.dot(w_r, x_r) - tl.dot(w_i, x_i)
        acc_imag += tl.dot(w_r, x_i) + tl.dot(w_i, x_r)

    # Store final spectrum
    out_offset = k * N + n_out
    out_mask = (k < M) & (n_out < N)
    tl.store(spectrum_ptr + 2 * out_offset, acc_real, mask=out_mask)
    tl.store(spectrum_ptr + 2 * out_offset + 1, acc_imag, mask=out_mask)


def solve(signal: torch.Tensor, spectrum: torch.Tensor, M: int, N: int):
    device = signal.device

    # -----------------------------------------------------------
    # 1. Precompute Row DFT Matrix (N x N) entirely in PyTorch
    # -----------------------------------------------------------
    n_idx = torch.arange(N, device=device, dtype=torch.float32)
    k_idx = torch.arange(N, device=device, dtype=torch.float32)
    angle_row = -2.0 * math.pi * n_idx[:, None] * k_idx[None, :] / N
    w_row_r = torch.cos(angle_row)
    w_row_i = torch.sin(angle_row)

    # -----------------------------------------------------------
    # 2. Precompute Col DFT Matrix (M x M) entirely in PyTorch
    # -----------------------------------------------------------
    m_idx = torch.arange(M, device=device, dtype=torch.float32)
    k_col = torch.arange(M, device=device, dtype=torch.float32)
    angle_col = -2.0 * math.pi * k_col[:, None] * m_idx[None, :] / M
    w_col_r = torch.cos(angle_col)
    w_col_i = torch.sin(angle_col)

    temp = torch.empty_like(signal)

    # Block sizes optimized to fit strictly within T4's 64KB Shared Memory limits
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32

    # 1st Pass: DFT on all Rows -> shape [M, N]
    grid_row = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_K))
    row_dft_kernel[grid_row](
        signal, temp,
        w_row_r, w_row_i,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=4
    )

    # 2nd Pass: DFT on all Columns -> shape [M, N]
    grid_col = (triton.cdiv(M, BLOCK_K), triton.cdiv(N, BLOCK_N))
    col_dft_kernel[grid_col](
        temp, spectrum,
        w_col_r, w_col_i,
        M, N,
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
        num_warps=4
    )


if __name__ == "__main__":
    import sys

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0x2DFF)
    M, N = 32, 32
    signal_cpu = torch.randn((M, N, 2), dtype=torch.float32)
    signal = signal_cpu.to(device).contiguous()
    spectrum = torch.empty_like(signal)

    solve(signal, spectrum, M, N)
    if device == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()

    expected = torch.view_as_real(
        torch.fft.fft2(torch.view_as_complex(signal_cpu.contiguous()))
    ).contiguous()
    torch.testing.assert_close(spectrum.cpu(), expected, atol=1e-2, rtol=1e-3)
    print(f"PASS [{device}] 2d_fft M={M} N={N}")
