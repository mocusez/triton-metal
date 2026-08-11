"""Decaying causal attention on the Metal backend.

The verbatim `leet-triton/medium-decaying_causal_attention.py` kernel: causal
attention with an exponential decay weight and NO softmax at all,

    out[n] = sum_{m <= n} gamma^(n-m) * (q[n] . k[m] / sqrt(d)) * v[m]

STATUS: supported via `metal.fused_attention`, the generalized
`Q@K^T -> score transform -> P@V` op whose score transform is carried as a
REGION rather than baked into a per-variant emitter.

This kernel is why that op exists. It matched neither predecessor:
`metal.flash_attention`'s matcher walks online-softmax roles (there is no
softmax here), and `metal.sink_attention` targets a different mask entirely. It
is also NOT rescuable by the scalar-GEMM fallback — `metal.scalar_dot` reads its
operands from device buffers by re-deriving addresses, and the second dot's A
operand is a computed register tile that is not in memory at all. So before the
fused op the result was a hard `convert-tritongpu-to-metal failed` at every
shape, not a wrong answer.

What the matcher does NOT do is recognize a decay mask. It absorbs every op
between the two dots into the score region and declines if any op has no scalar
translation; the causal decay survives because `subi/cmpi/sitofp/select/exp2/
mulf` all do. Coverage is established by construction rather than by pattern
enumeration -- the mechanism that previously let a band-masked kernel be claimed
as plain attention and silently miscompiled.

The emitted body walks keys one at a time with one query row per lane, so
summation order differs from the source's per-block `tl.dot`; the bar is
therefore a float64 reference at 1e-4 relative, not bit-exactness.
"""

from __future__ import annotations

import math

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


# Verbatim copy of leet-triton/medium-decaying_causal_attention.py.
@triton.jit
def _decay_causal_attn_fwd(
    Q, K, V, Out,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    seq_len, d_model,
    log2_gamma, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len
    d_mask = offs_d < d_model
    qd_mask = m_mask[:, None] & d_mask[None, :]

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=qd_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    hi = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = (offs_n < seq_len)[:, None] & d_mask[None, :]

        k = tl.load(K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                    mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

        diff = offs_m[:, None] - offs_n[None, :]
        exponent = tl.where(diff >= 0, diff * log2_gamma, float("-inf"))
        s = s * tl.exp2(exponent)

        v = tl.load(V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                    mask=kv_mask, other=0.0)
        acc = tl.dot(s, v, input_precision="ieee", acc=acc)

    tl.store(Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc, mask=qd_mask)


def solve(Q, K, V, output, seq_len, d_model, gamma):
    BLOCK_D = max(16, triton.next_power_of_2(d_model))
    if BLOCK_D <= 64:
        BLOCK_M, BLOCK_N, num_warps = 64, 64, 4
    elif BLOCK_D <= 128:
        BLOCK_M, BLOCK_N, num_warps = 32, 32, 4
    else:
        BLOCK_M, BLOCK_N, num_warps = 16, 16, 4

    log2_gamma = math.log2(min(max(float(gamma), 1e-300), 1.0))
    scale = 1.0 / math.sqrt(d_model)

    grid = (triton.cdiv(seq_len, BLOCK_M),)
    _decay_causal_attn_fwd[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        seq_len, d_model,
        log2_gamma, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=1,
    )
    return output


def _reference(Q, K, V, seq_len, d_model, gamma):
    q, k, v = Q.cpu().double(), K.cpu().double(), V.cpu().double()
    s = (q @ k.T) / math.sqrt(d_model)
    n = torch.arange(seq_len).view(-1, 1)
    m = torch.arange(seq_len).view(1, -1)
    diff = (n - m).double()
    g = torch.tensor(float(gamma), dtype=torch.float64)
    decay = torch.where(diff >= 0, torch.pow(g, diff), torch.zeros_like(diff))
    return (s * decay) @ v


def _run(seq_len, d_model, gamma, seed=0):
    torch.manual_seed(seed)
    mk = lambda: torch.randn(seq_len, d_model, dtype=torch.float32,
                             device="mps").contiguous()
    Q, K, V = mk(), mk(), mk()
    out = torch.zeros(seq_len, d_model, dtype=torch.float32,
                      device="mps").contiguous()
    solve(Q, K, V, out, seq_len, d_model, gamma)
    torch.mps.synchronize()
    ref = _reference(Q, K, V, seq_len, d_model, gamma)
    return out.cpu().double(), ref


@pytest.mark.parametrize("seq_len", [7, 33, 64, 100, 128, 256])
@pytest.mark.parametrize("d_model", [8, 32, 64])
def test_decaying_causal_attention(seq_len, d_model):
    got, ref = _run(seq_len, d_model, 0.9)
    err = (got - ref).abs().max().item()
    assert err <= 1e-4 * max(1.0, ref.abs().max().item()), f"max abs err {err:.3e}"


# BLOCK_D crosses the host's three tile tiers at 64 / 128 / 256, and each tier
# picks a different (BLOCK_M, BLOCK_N). d_model=48 also exercises a non-power-of-2
# feature width, where the padded columns must contribute exactly zero.
@pytest.mark.parametrize("d_model", [16, 48, 128, 256])
def test_decaying_causal_attention_tile_tiers(d_model):
    got, ref = _run(64, d_model, 0.9)
    err = (got - ref).abs().max().item()
    assert err <= 1e-4 * max(1.0, ref.abs().max().item()), f"max abs err {err:.3e}"


@pytest.mark.parametrize("gamma", [0.5, 0.8, 0.99, 1.0])
def test_decaying_causal_attention_gammas(gamma):
    got, ref = _run(96, 32, gamma)
    err = (got - ref).abs().max().item()
    assert err <= 1e-4 * max(1.0, ref.abs().max().item()), f"max abs err {err:.3e}"


def test_decaying_causal_attention_uses_the_fused_op():
    """The point of the kernel: it must go through metal.fused_attention.

    Without this the numeric tests would still pass on any future path that
    happens to compile it, hiding the fact that the generalized op stopped
    claiming it.
    """
    # Launch for real rather than hand-building an ASTSource: the address model
    # is `base + major*stride + feature` with UNIT column stride, which holds
    # only because Triton drops an argument equal to 1 from the signature. A
    # hand-written signature that declares the column strides as runtime i32
    # makes the matcher decline for a reason that has nothing to do with the
    # kernel.
    seq_len, d_model = 64, 64
    torch.manual_seed(0)
    mk = lambda: torch.randn(seq_len, d_model, dtype=torch.float32,
                             device="mps").contiguous()
    Q, K, V = mk(), mk(), mk()
    out = torch.zeros(seq_len, d_model, dtype=torch.float32,
                      device="mps").contiguous()
    compiled = _decay_causal_attn_fwd[(1,)](
        Q, K, V, out,
        Q.stride(0), Q.stride(1), K.stride(0), K.stride(1),
        V.stride(0), V.stride(1), out.stride(0), out.stride(1),
        seq_len, d_model,
        math.log2(0.9), 1.0 / math.sqrt(d_model),
        BLOCK_M=64, BLOCK_N=64, BLOCK_D=64,
        num_warps=4, num_stages=1,
    )
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert "metal.fused_attention" in msl, msl[:2000]
    # The score transform must come from the region, not from a hard-coded body.
    assert "exp2" in msl
    # norm = none: no running softmax state, no epilogue divide.
    assert "_fa_rmax" not in msl and "_fa_den" not in msl
    # The causal loop bound is REPRODUCED, not left to the region to zero out.
    assert "_fa_kend" in msl


def test_decaying_causal_attention_writes_every_element():
    """A masked-store or tile-origin bug shows up as untouched output, not as a
    numeric error, so check occupancy separately against a sentinel fill."""
    seq_len, d_model = 100, 48
    torch.manual_seed(0)
    mk = lambda: torch.randn(seq_len, d_model, dtype=torch.float32,
                             device="mps").contiguous()
    sentinel = -12345.0
    out = torch.full((seq_len, d_model), sentinel, dtype=torch.float32,
                     device="mps").contiguous()
    solve(mk(), mk(), mk(), out, seq_len, d_model, 0.9)
    torch.mps.synchronize()
    untouched = int((out.cpu() == sentinel).sum())
    assert untouched == 0, f"{untouched} output elements were never written"
