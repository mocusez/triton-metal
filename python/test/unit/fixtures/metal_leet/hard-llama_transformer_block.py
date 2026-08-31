"""End-to-end Llama transformer block workload for the Metal backend."""

import math
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

D_MODEL = 512
N_Q_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 64
FFN_HIDDEN = 1408
QKV_DIM = 768                      # W_Q|W_K|W_V contiguous from offset 512
GU_DIM = 2 * FFN_HIDDEN            # 2816, W_gate|W_up contiguous from 656384
EPS = 1e-5
WEIGHTS_NUMEL = 2_819_072

OFF_W1 = 0
OFF_WQKV = 512
OFF_WO = 393728
OFF_W2 = 655872
OFF_WGU = 656384
OFF_WD = 2098176

# sync after every stage only for small (correctness) sizes -> precise fault
# attribution if anything ever goes wrong; benchmark size stays fully async.
_SYNC_MAX_T = 256


def _synchronize(device):
    device_module = getattr(torch, device.type, None)
    synchronize = getattr(device_module, "synchronize", None)
    if synchronize is None:
        raise RuntimeError(
            f"[llama_block] device {device} does not expose a synchronize() API")
    synchronize()


def _stage(name, sync_device, launch):
    try:
        launch()
        if sync_device is not None:
            _synchronize(sync_device)
    except RuntimeError as e:
        raise RuntimeError(f"[llama_block] fault at stage <{name}>: {e}") from e


# ---------------- RMSNorm (full weights buffer + offset) ----------------
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, w_off, y_ptr,
                    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + EPS)
    w = tl.load(w_ptr + w_off + offs, mask=mask, other=0.0)
    tl.store(y_ptr + row * N + offs, x * rstd * w, mask=mask)


# ---------------- GEMM: C(M,N) = A(M,K) @ B(N,K)^T, B = weights[b_off:] ----------------
@triton.jit
def _gemm_nt_kernel(a_ptr, b_ptr, b_off, c_ptr,
                    M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + b_off + rn[:, None] * K + rk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K - k0), other=0.0)
        b = tl.load(b_ptrs, mask=(rn[:, None] < N) & (rk[None, :] < K - k0), other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=c_mask)


# ---------------- elementwise residual add: out = a + b ----------------
@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    b = tl.load(b_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=m)


# ---------------- RoPE (in place) ----------------
@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, T, row_stride, base_off,
                 HALF: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs = tl.arange(0, HALF)
    m = rows < T
    c = tl.load(cos_ptr + rows[:, None] * HALF + offs[None, :], mask=m[:, None], other=0.0)
    s = tl.load(sin_ptr + rows[:, None] * HALF + offs[None, :], mask=m[:, None], other=0.0)
    base = x_ptr + base_off + pid_h * (2 * HALF) + rows[:, None] * row_stride
    x1 = tl.load(base + offs[None, :], mask=m[:, None], other=0.0)
    x2 = tl.load(base + HALF + offs[None, :], mask=m[:, None], other=0.0)
    tl.store(base + offs[None, :], x1 * c - x2 * s, mask=m[:, None])
    tl.store(base + HALF + offs[None, :], x1 * s + x2 * c, mask=m[:, None])


# ---------------- causal GQA attention (flash-style) ----------------
@triton.jit
def _attn_kernel(qkv_ptr, out_ptr, T, qkv_stride, scale,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // 4                       # GQA: q head h uses kv head h//4
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_ok = offs_m < T

    q_base = qkv_ptr + pid_h * D            # Q cols   0:512
    k_base = qkv_ptr + 512 + kv_h * D       # K cols 512:640
    v_base = qkv_ptr + 640 + kv_h * D       # V cols 640:768

    q = tl.load(q_base + offs_m[:, None] * qkv_stride + offs_d[None, :],
                mask=m_ok[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    hi = pid_m * BLOCK_M + BLOCK_M          # causal: keys only up to this block's last row
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_ok = offs_n < T
        k = tl.load(k_base + offs_n[:, None] * qkv_stride + offs_d[None, :],
                    mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        s = tl.where((offs_m[:, None] >= offs_n[None, :]) & n_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + offs_n[:, None] * qkv_stride + offs_d[None, :],
                    mask=n_ok[:, None], other=0.0)
        acc = tl.dot(p, v, acc, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(out_ptr + pid_h * D + offs_m[:, None] * 512 + offs_d[None, :],
             acc, mask=m_ok[:, None])


# ---------------- SiLU(gate) * up ----------------
@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, N: tl.constexpr, gu_stride,
                     BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    g = tl.load(gu_ptr + pid_t * gu_stride + offs, mask=mask, other=0.0)
    u = tl.load(gu_ptr + pid_t * gu_stride + N + offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid_t * N + offs, g * tl.sigmoid(g) * u, mask=mask)


# ---------------- host wrapper ----------------
def solve(x, output, weights, cos, sin, seq_len):
    T = int(seq_len)
    f32 = torch.float32
    dev = x.device
    if dev.type == "cpu":
        dev = triton.runtime.driver.active.get_active_torch_device()
    if dev.type == "cpu":
        raise RuntimeError(
            "[llama_block] Triton execution requires an accelerator tensor or "
            "an active accelerator backend")

    device_module = getattr(torch, dev.type, None)
    device_context = getattr(device_module, "device", None)
    dev_ctx = device_context(dev) if device_context is not None else nullcontext()

    with dev_ctx:
        sync_device = dev if T <= _SYNC_MAX_T else None

        # normalize: right device, fp32, contiguous (all no-ops on the harness)
        x = x.to(device=dev, dtype=f32).contiguous()
        w = weights.to(device=dev, dtype=f32).contiguous().view(-1)
        cos = cos.to(device=dev, dtype=f32).contiguous()
        sin = sin.to(device=dev, dtype=f32).contiguous()

        # sanity checks (never fire on the harness; keep diagnostics precise)
        if w.numel() < WEIGHTS_NUMEL:
            raise RuntimeError(
                f"[llama_block] weights buffer too short: numel={w.numel()}, "
                f"expected {WEIGHTS_NUMEL} — check the solve() argument order; "
                f"the harness calls solve(x, output, weights, cos, sin, seq_len)")
        if x.numel() < T * D_MODEL or cos.numel() < T * (HEAD_DIM // 2):
            raise RuntimeError(
                f"[llama_block] bad input shapes: x={tuple(x.shape)} "
                f"cos={tuple(cos.shape)} seq_len={T}")

        # write straight into the caller's output when it is already fp32/contiguous
        out_view = output.to(device=dev, dtype=f32)
        if out_view.is_contiguous() and out_view.shape == (T, D_MODEL):
            out_buf = out_view
            need_copy = out_view.data_ptr() != output.data_ptr()
        else:
            out_buf = torch.empty((T, D_MODEL), device=dev, dtype=f32)
            need_copy = True

        norm1 = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        qkv = torch.empty((T, QKV_DIM), device=dev, dtype=f32)
        attn = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        proj = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        x1 = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        norm2 = torch.empty((T, D_MODEL), device=dev, dtype=f32)
        gu = torch.empty((T, GU_DIM), device=dev, dtype=f32)
        ffn = torch.empty((T, FFN_HIDDEN), device=dev, dtype=f32)
        ffn_out = torch.empty((T, D_MODEL), device=dev, dtype=f32)

        BM, BN, BK, NW = 64, 64, 32, 4
        add_grid = (triton.cdiv(T * D_MODEL, 1024),)

        # 1) RMSNorm 1
        _stage("1-rmsnorm1", sync_device, lambda: _rmsnorm_kernel[(T,)](
            x, w, OFF_W1, norm1, N=D_MODEL, EPS=EPS, BLOCK=512, num_warps=4))

        # 2) QKV projection  (T,512) @ (768,512)^T -> (T,768)
        _stage("2-qkv-gemm", sync_device, lambda: _gemm_nt_kernel[(triton.cdiv(T, BM), QKV_DIM // BN)](
            norm1, w, OFF_WQKV, qkv, T, QKV_DIM, D_MODEL,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=NW))

        # 3) RoPE on Q (8 heads, col offset 0) and K (2 heads, col offset 512), in place
        def rope():
            _rope_kernel[(triton.cdiv(T, 32), N_Q_HEADS)](
                qkv, cos, sin, T, QKV_DIM, 0, HALF=HEAD_DIM // 2, BLOCK_T=32, num_warps=1)
            _rope_kernel[(triton.cdiv(T, 32), N_KV_HEADS)](
                qkv, cos, sin, T, QKV_DIM, D_MODEL, HALF=HEAD_DIM // 2, BLOCK_T=32, num_warps=1)
        _stage("3-rope", sync_device, rope)

        # 4) causal GQA attention -> (T,512)
        _stage("4-attention", sync_device, lambda: _attn_kernel[(triton.cdiv(T, 64), N_Q_HEADS)](
            qkv, attn, T, QKV_DIM, 1.0 / math.sqrt(HEAD_DIM),
            BLOCK_M=64, BLOCK_N=32, D=HEAD_DIM, num_warps=4, num_stages=1))

        # 5) output projection, then residual: x1 = x + attn @ W_O^T
        _stage("5-outproj-gemm", sync_device, lambda: _gemm_nt_kernel[(triton.cdiv(T, BM), D_MODEL // BN)](
            attn, w, OFF_WO, proj, T, D_MODEL, D_MODEL,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=NW))
        _stage("5b-residual-add", sync_device, lambda: _add_kernel[add_grid](
            x, proj, x1, T * D_MODEL, BLOCK=1024, num_warps=4))

        # 6) RMSNorm 2
        _stage("6-rmsnorm2", sync_device, lambda: _rmsnorm_kernel[(T,)](
            x1, w, OFF_W2, norm2, N=D_MODEL, EPS=EPS, BLOCK=512, num_warps=4))

        # 7) fused gate|up projection -> (T,2816)
        _stage("7-gateup-gemm", sync_device, lambda: _gemm_nt_kernel[(triton.cdiv(T, BM), GU_DIM // BN)](
            norm2, w, OFF_WGU, gu, T, GU_DIM, D_MODEL,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=NW))

        # 8) SwiGLU: silu(gate) * up -> (T,1408)
        _stage("8-silu-mul", sync_device, lambda: _silu_mul_kernel[(T, triton.cdiv(FFN_HIDDEN, 512))](
            gu, ffn, N=FFN_HIDDEN, gu_stride=GU_DIM, BLOCK_N=512, num_warps=4))

        # 9) down projection, then residual: out = x1 + ffn @ W_down^T
        _stage("9-down-gemm", sync_device, lambda: _gemm_nt_kernel[(triton.cdiv(T, BM), D_MODEL // BN)](
            ffn, w, OFF_WD, ffn_out, T, D_MODEL, FFN_HIDDEN,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=NW))
        _stage("9b-residual-add", sync_device, lambda: _add_kernel[add_grid](
            x1, ffn_out, out_buf, T * D_MODEL, BLOCK=1024, num_warps=4))

        if need_copy:
            output.copy_(out_buf)


def forward(x, weights, cos, sin):
    out = torch.empty_like(x)
    solve(x, out, weights, cos, sin, x.shape[0])
    return out


# ---------------- self test (mirrors the official challenge harness) ----------------
if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev = triton.runtime.driver.active.get_active_torch_device()
    assert dev.type != "cpu", "needs an active Triton accelerator backend"

    def make_case(T, zero_x=False):
        scale = 0.02
        parts = [
            torch.empty(D_MODEL, device=dev).uniform_(0.8, 1.2),
            torch.empty(D_MODEL, D_MODEL, device=dev).normal_(0, scale).flatten(),
            torch.empty(128, D_MODEL, device=dev).normal_(0, scale).flatten(),
            torch.empty(128, D_MODEL, device=dev).normal_(0, scale).flatten(),
            torch.empty(D_MODEL, D_MODEL, device=dev).normal_(0, scale).flatten(),
            torch.empty(D_MODEL, device=dev).uniform_(0.8, 1.2),
            torch.empty(FFN_HIDDEN, D_MODEL, device=dev).normal_(0, scale).flatten(),
            torch.empty(FFN_HIDDEN, D_MODEL, device=dev).normal_(0, scale).flatten(),
            torch.empty(D_MODEL, FFN_HIDDEN, device=dev).normal_(0, scale).flatten(),
        ]
        weights = torch.cat(parts)
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, device=dev).float() / HEAD_DIM))
        ang = torch.outer(pos, freqs)
        if zero_x:
            x = torch.zeros(T, D_MODEL, device=dev)
        else:
            x = torch.empty(T, D_MODEL, device=dev).uniform_(-1.0, 1.0)
        return x, weights, ang.cos(), ang.sin()

    def reference(x, weights, cos, sin):
        T = x.shape[0]
        w = weights

        def rms(z, g):
            return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + EPS) * g

        def rope(t):
            t1, t2 = t[..., :32], t[..., 32:]
            c, s = cos[:, None, :], sin[:, None, :]
            return torch.cat([t1 * c - t2 * s, t1 * s + t2 * c], -1)

        n1 = rms(x, w[:512])
        Q = (n1 @ w[512:262656].view(512, 512).T).view(T, 8, 64)
        K = (n1 @ w[262656:328192].view(128, 512).T).view(T, 2, 64)
        V = (n1 @ w[328192:393728].view(128, 512).T).view(T, 2, 64)
        Q, K = rope(Q), rope(K)
        K, V = K.repeat_interleave(4, 1), V.repeat_interleave(4, 1)
        s = torch.einsum("thd,shd->hts", Q, K) / math.sqrt(64.0)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        s = s.masked_fill(mask[None], float("-inf"))
        ctx = torch.einsum("hts,shd->thd", torch.softmax(s, -1), V).reshape(T, 512)
        x1 = x + ctx @ w[393728:655872].view(512, 512).T
        n2 = rms(x1, w[655872:656384])
        h = F.silu(n2 @ w[656384:1377280].view(1408, 512).T) \
            * (n2 @ w[1377280:2098176].view(1408, 512).T)
        return x1 + h @ w[2098176:].view(512, 1408).T

    # official functional suite: 1, 4(zero), 2, 4, 16, 64, 30, 100, 128, 256
    cases = [(1, False), (4, True), (2, False), (4, False), (16, False),
             (64, False), (30, False), (100, False), (128, False), (256, False)]
    for T, zx in cases:
        x, weights, cos, sin = make_case(T, zero_x=zx)
        out = forward(x, weights, cos, sin)
        ref = reference(x, weights, cos, sin)
        ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        err = (out - ref).abs().max().item()
        print(f"seq_len={T:5d} zero_x={int(zx)}  max abs err = {err:.3e}  "
              f"{'OK' if ok else 'FAIL'}")
        assert ok

    T = 2048
    x, weights, cos, sin = make_case(T)
    for _ in range(3):
        forward(x, weights, cos, sin)
    _synchronize(dev)
    import time
    t0 = time.perf_counter()
    for _ in range(20):
        forward(x, weights, cos, sin)
    _synchronize(dev)
    print(f"seq_len=2048: {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms / iter")
