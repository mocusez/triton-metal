import torch
import triton
import triton.language as tl

# All sizes are RUNTIME arguments -> every kernel compiles exactly once per
# process, no matter how many different N values are tested.
_BLOCK = 1024          # tile width (elements per program)


@triton.jit
def _bitrev_copy(x_ptr, y_ptr, N, LOG_N, CONJ: tl.constexpr, BLOCK: tl.constexpr):
    """y[bitrev(i)] = x[i]  (optionally conjugated, for the inverse FFT)."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    r = tl.zeros((BLOCK,), dtype=tl.int32)
    for b in range(LOG_N):                       # runtime trip count
        r = (r << 1) | ((i >> b) & 1)
    re = tl.load(x_ptr + 2 * i, mask=mask, other=0.0)
    im = tl.load(x_ptr + 2 * i + 1, mask=mask, other=0.0)
    if CONJ:
        im = -im
    tl.store(y_ptr + 2 * r, re, mask=mask)
    tl.store(y_ptr + 2 * r + 1, im, mask=mask)


@triton.jit
def _fft_local(x_ptr, N, STAGES, BLOCK: tl.constexpr):
    """
    In-place radix-2 DIT stages with butterfly distance d = 1,2,...,2^(STAGES-1)
    inside each contiguous BLOCK-sized chunk (BLOCK = 2^STAGES).
    One program per chunk; partner fetch via gather (register/shuffle).
    """
    pid = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    base = pid * BLOCK + i
    re = tl.load(x_ptr + 2 * base)
    im = tl.load(x_ptr + 2 * base + 1)
    for s in range(STAGES):                      # runtime trip count
        d = 1 << s
        pr = tl.gather(re, i ^ d, 0)             # partner = x[i ^ d]
        pi = tl.gather(im, i ^ d, 0)
        k = i & (d - 1)
        ang = (k.to(tl.float32) / d.to(tl.float32)) * -3.141592653589793
        wr = tl.cos(ang)
        wi = tl.sin(ang)
        lower = (i & d) == 0
        lo_re = re + wr * pr - wi * pi
        lo_im = im + wr * pi + wi * pr
        up_re = pr - wr * re + wi * im
        up_im = pi - wi * re - wr * im
        re = tl.where(lower, lo_re, up_re)
        im = tl.where(lower, lo_im, up_im)
    tl.store(x_ptr + 2 * base, re)
    tl.store(x_ptr + 2 * base + 1, im)


@triton.jit
def _fft_global(x_ptr, N, d, LOG_D, BLOCK: tl.constexpr):
    """One in-place radix-2 DIT stage with butterfly distance d (>= BLOCK)."""
    pid = tl.program_id(0)
    b = pid * BLOCK + tl.arange(0, BLOCK)        # butterfly index in [0, N/2)
    k = b & (d - 1)
    g = b >> LOG_D
    i = (g << (LOG_D + 1)) | k                   # lower element of the pair
    xr = tl.load(x_ptr + 2 * i)
    xi = tl.load(x_ptr + 2 * i + 1)
    yr = tl.load(x_ptr + 2 * (i + d))
    yi = tl.load(x_ptr + 2 * (i + d) + 1)
    ang = (k.to(tl.float32) / d.to(tl.float32)) * -3.141592653589793
    wr = tl.cos(ang)
    wi = tl.sin(ang)
    tr = wr * yr - wi * yi
    ti = wr * yi + wi * yr
    tl.store(x_ptr + 2 * i, xr + tr)
    tl.store(x_ptr + 2 * i + 1, xi + ti)
    tl.store(x_ptr + 2 * (i + d), xr - tr)
    tl.store(x_ptr + 2 * (i + d) + 1, xi - ti)


@triton.jit
def _dft_naive(x_ptr, y_ptr, N, KB: tl.constexpr, NB: tl.constexpr):
    """O(N^2) DFT, used only for tiny N (< 1024)."""
    pid = tl.program_id(0)
    k = pid * KB + tl.arange(0, KB)
    kmask = k < N
    acc_re = tl.zeros((KB,), dtype=tl.float32)
    acc_im = tl.zeros((KB,), dtype=tl.float32)
    for n0 in range(0, N, NB):           # N < 1024 here: k*n < 2^20 fits int32
        n = n0 + tl.arange(0, NB)
        nmask = n < N
        xr = tl.load(x_ptr + 2 * n, mask=nmask, other=0.0)
        xi = tl.load(x_ptr + 2 * n + 1, mask=nmask, other=0.0)
        p = (k[:, None] * n[None, :]) % N
        ang = (p.to(tl.float32) / N) * -6.283185307179586
        c = tl.cos(ang)
        s = tl.sin(ang)
        acc_re += tl.sum(xr[None, :] * c - xi[None, :] * s, axis=1)
        acc_im += tl.sum(xr[None, :] * s + xi[None, :] * c, axis=1)
    tl.store(y_ptr + 2 * k, acc_re, mask=kmask)
    tl.store(y_ptr + 2 * k + 1, acc_im, mask=kmask)

# ------------------------- Bluestein (non-power-of-two) -------------------------

@triton.jit
def _bp_prep(x_ptr, a_ptr, b_ptr, N, M, BLOCK: tl.constexpr):
    """
    a[j] = x[j] * exp(-i*pi*j^2/N)  (j < N, else 0)
    b[j] = exp(+i*pi*j^2/N) for j < N,  b[j] = exp(+i*pi*(M-j)^2/N) for M-j < N
    """
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)        # 0 <= j < M (M % BLOCK == 0)
    jn = j < N
    p = (j.to(tl.int64) * j.to(tl.int64)) % (2 * N)
    ang = (p.to(tl.float32) / N) * -3.141592653589793
    xr = tl.load(x_ptr + 2 * j, mask=jn, other=0.0)
    xi = tl.load(x_ptr + 2 * j + 1, mask=jn, other=0.0)
    ca = tl.cos(ang)
    sa = tl.sin(ang)
    tl.store(a_ptr + 2 * j, xr * ca - xi * sa)
    tl.store(a_ptr + 2 * j + 1, xr * sa + xi * ca)
    t = tl.where(jn, j, M - j)
    valid = jn | (t < N)
    p2 = (t.to(tl.int64) * t.to(tl.int64)) % (2 * N)
    ang2 = (p2.to(tl.float32) / N) * 3.141592653589793   # conjugate chirp
    tl.store(b_ptr + 2 * j, tl.where(valid, tl.cos(ang2), 0.0))
    tl.store(b_ptr + 2 * j + 1, tl.where(valid, tl.sin(ang2), 0.0))


@triton.jit
def _cmul(a_ptr, b_ptr, L, BLOCK: tl.constexpr):
    """a[j] *= b[j]  (complex, interleaved)."""
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    ar = tl.load(a_ptr + 2 * j)
    ai = tl.load(a_ptr + 2 * j + 1)
    br = tl.load(b_ptr + 2 * j)
    bi = tl.load(b_ptr + 2 * j + 1)
    tl.store(a_ptr + 2 * j, ar * br - ai * bi)
    tl.store(a_ptr + 2 * j + 1, ar * bi + ai * br)


@triton.jit
def _bp_finalize(y_ptr, out_ptr, N, M, BLOCK: tl.constexpr):
    """out[k] = exp(-i*pi*k^2/N) * conj(y[k]) / M   for k < N."""
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < N
    p = (k.to(tl.int64) * k.to(tl.int64)) % (2 * N)
    ang = (p.to(tl.float32) / N) * -3.141592653589793
    cr = tl.cos(ang)
    ci = tl.sin(ang)
    yr = tl.load(y_ptr + 2 * k, mask=mask, other=0.0) / M
    yi = -tl.load(y_ptr + 2 * k + 1, mask=mask, other=0.0) / M   # conjugate
    tl.store(out_ptr + 2 * k, cr * yr - ci * yi, mask=mask)
    tl.store(out_ptr + 2 * k + 1, cr * yi + ci * yr, mask=mask)

# ------------------------------ host wrapper ------------------------------

def _fft_big(x, y, N, bits):
    """Power-of-two FFT (N >= 1024): x -> y, all kernels runtime-parametric."""
    _bitrev_copy[(triton.cdiv(N, _BLOCK),)](x, y, N, bits, False, _BLOCK,
                                            num_warps=4)
    _fft_local[(N // _BLOCK,)](y, N, 10, _BLOCK, num_warps=4)
    d = _BLOCK
    while d < N:
        _fft_global[(N // (2 * _BLOCK),)](y, N, d, d.bit_length() - 1, _BLOCK,
                                          num_warps=4)
        d += d


# signal and spectrum are tensors on the GPU
def solve(signal: torch.Tensor, spectrum: torch.Tensor, N: int):
    if not signal.is_contiguous():
        signal = signal.contiguous()
    if not spectrum.is_contiguous():
        tmp = torch.empty(2 * N, device=spectrum.device, dtype=torch.float32)
        solve(signal, tmp, N)
        spectrum.copy_(tmp)
        return

    if N < _BLOCK:                               # tiny: one kernel, any N
        _dft_naive[(triton.cdiv(N, 64),)](signal, spectrum, N, 64, 64,
                                          num_warps=4)
    elif N & (N - 1) == 0:                       # power of two
        _fft_big(signal, spectrum, N, N.bit_length() - 1)
    else:                                        # Bluestein: any N, O(N log N)
        M = 1 << (2 * N - 2).bit_length()        # pow2 >= 2N-1
        bits = M.bit_length() - 1
        a = torch.empty(2 * M, device=signal.device, dtype=torch.float32)
        b = torch.empty(2 * M, device=signal.device, dtype=torch.float32)
        c = torch.empty(2 * M, device=signal.device, dtype=torch.float32)
        _bp_prep[(M // _BLOCK,)](signal, a, b, N, M, _BLOCK, num_warps=4)
        _fft_big(a, c, M, bits)                  # A = FFT(a) in c
        _fft_big(b, a, M, bits)                  # B = FFT(b) in a
        _cmul[(M // _BLOCK,)](c, a, M, _BLOCK, num_warps=4)   # C = A*B in c
        _bitrev_copy[(M // _BLOCK,)](c, a, M, bits, True, _BLOCK,
                                     num_warps=4)             # inverse FFT:
        _fft_local[(M // _BLOCK,)](a, M, 10, _BLOCK, num_warps=4)
        d = _BLOCK
        while d < M:
            _fft_global[(M // (2 * _BLOCK),)](a, M, d, d.bit_length() - 1,
                                              _BLOCK, num_warps=4)
            d += d
        _bp_finalize[(triton.cdiv(N, _BLOCK),)](a, spectrum, N, M, _BLOCK,
                                                num_warps=4)