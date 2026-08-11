"""Attention with sink tokens on the Metal backend.

The verbatim `leet-triton/medium-attention_with_sinks.py` kernel: causal
attention where the first `num_sinks` tokens stay visible to every query on top
of a one-sided sliding window of width `window_size`,

    keep(i, j) = j <= i and (j < num_sinks or j >= i - window_size + 1)

evaluated in two phases — a straight-line sink block, then `N_LOCAL_BLOCKS`
blocks of the window — with `exp2` and a host-computed `sm_scale` that already
folds in `log2(e)/sqrt(d)`.

STATUS: supported since metal-attention-with-sinks-plan.md, via a dedicated
`metal.sink_attention` op and its own matcher (`trySinkAttention`). It is
deliberately NOT part of `metal.flash_attention`: this kernel loads K already
transposed (no `tt.trans` for the FA matcher to classify its dots by), puts a
whole attention block OUTSIDE the loop feeding its iter_args, and deletes the
`scf.for` altogether when `N_LOCAL_BLOCKS == 1`. Three separate reasons the
loop-anchored FA matcher cannot see it at all — which is why the pre-fix
behaviour was a hard `convert-tritongpu-to-metal failed`, not a wrong answer.

The emitter walks keys one at a time with one query row per lane and updates
the online-softmax state per key, so it reproduces the source kernel's two
phases exactly (including their dependence on the host-side `N_LOCAL_BLOCKS`
constexpr) rather than reimplementing the mask above. Per-key merging is not
bit-identical to the source's per-block form — different summation order — so
the bar here is 1e-6 against a float64 reference, not exactness.
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


# Verbatim copy of leet-triton/medium-attention_with_sinks.py.
@triton.jit
def _attention_sinks_kernel(
    Q, K, V, output,
    M, d, num_sinks, window_size, sm_scale,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LOCAL_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < d

    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    neg_inf = -float("inf")
    m_i = tl.where(mask_m, neg_inf, 0.0)
    l_i = tl.where(mask_m, 0.0, 1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # 1. Sink tokens
    offs_s = tl.arange(0, BLOCK_S)
    sink_col_mask = offs_s < num_sinks
    k_sink = tl.load(
        K + offs_d[:, None] * stride_kd + offs_s[None, :] * stride_km,
        mask=mask_d[:, None] & sink_col_mask[None, :],
        other=0.0,
    )
    qk_sink = tl.dot(q, k_sink, input_precision="ieee")
    qk_sink = qk_sink * sm_scale
    valid_sink = (
        mask_m[:, None] & sink_col_mask[None, :] & (offs_s[None, :] <= offs_m[:, None])
    )
    qk_sink = tl.where(valid_sink, qk_sink, neg_inf)
    block_max_sink = tl.max(qk_sink, axis=1)
    m_new_sink = tl.maximum(m_i, block_max_sink)
    alpha_sink = tl.exp2(m_i - m_new_sink)
    p_sink = tl.exp2(qk_sink - m_new_sink[:, None])
    l_i = l_i * alpha_sink + tl.sum(p_sink, axis=1)
    v_sink = tl.load(
        V + offs_s[:, None] * stride_vm + offs_d[None, :] * stride_vd,
        mask=sink_col_mask[:, None] & mask_d[None, :],
        other=0.0,
    )
    acc = acc * alpha_sink[:, None] + tl.dot(p_sink, v_sink, input_precision="ieee")
    m_i = m_new_sink

    # 2. Sliding window
    local_start = start_m - window_size + 1
    local_start = tl.maximum(local_start, num_sinks)
    offs_bn = tl.arange(0, BLOCK_N)

    for block_idx in tl.range(0, N_LOCAL_BLOCKS):
        offs_n = local_start + block_idx * BLOCK_N + offs_bn
        mask_n = offs_n < M
        k_local = tl.load(
            K + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_km,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.0,
        )
        qk_local = tl.dot(q, k_local, input_precision="ieee")
        qk_local = qk_local * sm_scale
        window_left = offs_m[:, None] - window_size + 1
        valid_local = (
            mask_m[:, None]
            & mask_n[None, :]
            & (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] >= window_left)
            & (offs_n[None, :] >= num_sinks)
        )
        qk_local = tl.where(valid_local, qk_local, neg_inf)
        block_max_local = tl.max(qk_local, axis=1)
        m_new_local = tl.maximum(m_i, block_max_local)
        alpha_local = tl.exp2(m_i - m_new_local)
        p_local = tl.exp2(qk_local - m_new_local[:, None])
        l_i = l_i * alpha_local + tl.sum(p_local, axis=1)
        v_local = tl.load(
            V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc = acc * alpha_local[:, None] + tl.dot(
            p_local, v_local, input_precision="ieee"
        )
        m_i = m_new_local

    out = acc / l_i[:, None]
    tl.store(
        output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


# Same kernel with the `offs_n >= num_sinks` term dropped from the window mask.
# That double-counts every sink token that also falls inside the window, so it
# is a DIFFERENT kernel that happens to share the whole skeleton.
@triton.jit
def _attention_sinks_no_escape_kernel(
    Q, K, V, output,
    M, d, num_sinks, window_size, sm_scale,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LOCAL_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < d

    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    neg_inf = -float("inf")
    m_i = tl.where(mask_m, neg_inf, 0.0)
    l_i = tl.where(mask_m, 0.0, 1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    offs_s = tl.arange(0, BLOCK_S)
    sink_col_mask = offs_s < num_sinks
    k_sink = tl.load(
        K + offs_d[:, None] * stride_kd + offs_s[None, :] * stride_km,
        mask=mask_d[:, None] & sink_col_mask[None, :],
        other=0.0,
    )
    qk_sink = tl.dot(q, k_sink, input_precision="ieee") * sm_scale
    valid_sink = (
        mask_m[:, None] & sink_col_mask[None, :] & (offs_s[None, :] <= offs_m[:, None])
    )
    qk_sink = tl.where(valid_sink, qk_sink, neg_inf)
    m_new_sink = tl.maximum(m_i, tl.max(qk_sink, axis=1))
    alpha_sink = tl.exp2(m_i - m_new_sink)
    p_sink = tl.exp2(qk_sink - m_new_sink[:, None])
    l_i = l_i * alpha_sink + tl.sum(p_sink, axis=1)
    v_sink = tl.load(
        V + offs_s[:, None] * stride_vm + offs_d[None, :] * stride_vd,
        mask=sink_col_mask[:, None] & mask_d[None, :],
        other=0.0,
    )
    acc = acc * alpha_sink[:, None] + tl.dot(p_sink, v_sink, input_precision="ieee")
    m_i = m_new_sink

    local_start = tl.maximum(start_m - window_size + 1, num_sinks)
    offs_bn = tl.arange(0, BLOCK_N)
    for block_idx in tl.range(0, N_LOCAL_BLOCKS):
        offs_n = local_start + block_idx * BLOCK_N + offs_bn
        mask_n = offs_n < M
        k_local = tl.load(
            K + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_km,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.0,
        )
        qk_local = tl.dot(q, k_local, input_precision="ieee") * sm_scale
        window_left = offs_m[:, None] - window_size + 1
        valid_local = (
            mask_m[:, None]
            & mask_n[None, :]
            & (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] >= window_left)
        )  # <-- the `>= num_sinks` escape is gone
        qk_local = tl.where(valid_local, qk_local, neg_inf)
        m_new_local = tl.maximum(m_i, tl.max(qk_local, axis=1))
        alpha_local = tl.exp2(m_i - m_new_local)
        p_local = tl.exp2(qk_local - m_new_local[:, None])
        l_i = l_i * alpha_local + tl.sum(p_local, axis=1)
        v_local = tl.load(
            V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc = acc * alpha_local[:, None] + tl.dot(
            p_local, v_local, input_precision="ieee"
        )
        m_i = m_new_local

    tl.store(
        output + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc / l_i[:, None],
        mask=mask_m[:, None] & mask_d[None, :],
    )


def _solve(Q, K, V, output, M, d, num_sinks, window_size, kernel=None,
           block_d=None):
    BLOCK_M, BLOCK_N, BLOCK_S = 32, 64, 16
    BLOCK_D = block_d if block_d else max(16, triton.next_power_of_2(d))
    N_LOCAL_BLOCKS = triton.cdiv(window_size + BLOCK_M - 1, BLOCK_N)
    sm_scale = 1.4426950408889634 / (d ** 0.5)
    grid = (triton.cdiv(M, BLOCK_M),)
    (kernel or _attention_sinks_kernel)[grid](
        Q, K, V, output,
        M, d, num_sinks, window_size, sm_scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S,
        N_LOCAL_BLOCKS=N_LOCAL_BLOCKS,
        num_warps=8, num_stages=1,
    )


def _reference(Q, K, V, M, d, num_sinks, window_size, sink_escape=True):
    q, k, v = Q.cpu().double(), K.cpu().double(), V.cpu().double()
    scores = (q @ k.T) / math.sqrt(d)
    idx = torch.arange(M)
    i, j = idx[:, None], idx[None, :]
    window = j >= i - window_size + 1
    if sink_escape:
        keep = (j <= i) & ((j < num_sinks) | window)
    else:
        keep = (j <= i) & ((j < num_sinks) | (window & (j >= num_sinks)))
    return torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1) @ v


def _inputs(M, d, seed):
    torch.manual_seed(seed)
    mk = lambda: torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    return mk(), mk(), mk(), torch.zeros(M, d, dtype=torch.float32, device="mps")


@pytest.mark.parametrize("M", [32, 33, 64, 100, 256])
@pytest.mark.parametrize("d", [16, 32, 64])
@pytest.mark.parametrize("num_sinks,window_size", [(4, 32), (16, 128), (2, 16)])
def test_attention_with_sinks(M, d, num_sinks, window_size):
    """The verbatim kernel, across the shapes `solve()` can produce.

    `window_size=16` and `32` give `N_LOCAL_BLOCKS == 1`, which Triton unrolls
    into straight-line code with no `scf.for` at all; `128` keeps a 3-iteration
    loop. Both forms go through the same chain walk in the matcher.
    """
    Q, K, V, out = _inputs(M, d, seed=0xBEEF + M * 31 + d)
    _solve(Q, K, V, out, M, d, num_sinks, window_size)
    torch.mps.synchronize()
    ref = _reference(Q, K, V, M, d, num_sinks, window_size)
    err = (out.cpu().double() - ref).abs().max().item()
    assert err <= 1e-6, f"max abs err {err:.3e}"


@pytest.mark.parametrize("num_sinks", [0, 1, 8])
def test_attention_with_sinks_sink_counts(num_sinks):
    """`num_sinks = 0` (no sinks visible) and `1` (Triton folds the argument out
    of the signature entirely and it arrives as a constant, hence
    `sinks_const`)."""
    M, d, window_size = 128, 32, 64
    Q, K, V, out = _inputs(M, d, seed=0x51 + num_sinks)
    _solve(Q, K, V, out, M, d, num_sinks, window_size)
    torch.mps.synchronize()
    ref = _reference(Q, K, V, M, d, num_sinks, window_size)
    err = (out.cpu().double() - ref).abs().max().item()
    assert err <= 1e-6, f"max abs err {err:.3e}"


def test_attention_with_sinks_window_one():
    """`window_size = 1` is the other folded-constant path (`window_const`):
    each query sees only itself plus the sinks."""
    M, d, num_sinks, window_size = 64, 16, 4, 1
    Q, K, V, out = _inputs(M, d, seed=0x1)
    _solve(Q, K, V, out, M, d, num_sinks, window_size)
    torch.mps.synchronize()
    ref = _reference(Q, K, V, M, d, num_sinks, window_size)
    err = (out.cpu().double() - ref).abs().max().item()
    assert err <= 1e-6, f"max abs err {err:.3e}"


def test_attention_with_sinks_uses_the_fused_op():
    """The kernel must go through `metal.fused_attention`, not some other path.

    Without this the numeric tests above would still pass if the op were
    bypassed — and there is no other working path today, so a silent change of
    lowering would show up here first.

    What is checked beyond the op name is that BOTH key phases survive. The
    sink block and the sliding window are separate sweeps, and collapsing them
    into one masked sweep over `[0, M)` would still pass every numeric test
    here while being wrong in general: it is valid only when
    `n_local_blocks*bn >= window + bm - 1`, and `window` is a runtime argument.
    """
    M, d, num_sinks, window_size = 64, 16, 4, 32
    Q, K, V, out = _inputs(M, d, seed=0x2)
    BLOCK_M, BLOCK_N, BLOCK_S = 32, 64, 16
    BLOCK_D = max(16, triton.next_power_of_2(d))
    compiled = _attention_sinks_kernel[(triton.cdiv(M, BLOCK_M),)](
        Q, K, V, out,
        M, d, num_sinks, window_size, 1.4426950408889634 / (d ** 0.5),
        Q.stride(0), Q.stride(1), K.stride(0), K.stride(1),
        V.stride(0), V.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S,
        N_LOCAL_BLOCKS=triton.cdiv(window_size + BLOCK_M - 1, BLOCK_N),
        num_warps=8, num_stages=1,
    )
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert "metal.fused_attention" in msl
    # The two phases the emitter reproduces, and the exp2 the kernel asked for.
    assert "// --- key phase 0 ---" in msl, msl[:2000]
    assert "// --- key phase 1 ---" in msl, msl[:2000]
    assert "// --- key phase 2 ---" not in msl
    assert "exp2" in msl
    # norm = online_softmax: the running state is present, and the epilogue
    # divides by it.
    assert "_fa_rmax" in msl and "_fa_rsum" in msl


def test_attention_with_sinks_missing_escape_is_not_claimed():
    """A near-miss kernel must be REFUSED, never claimed and miscompiled.

    Dropping `offs_n >= num_sinks` from the window mask changes the answer
    (sink tokens inside the window get counted twice). The matcher's mask-tag
    set is exact, so the step no longer matches and the kernel falls through to
    the general path and its hard error. The failure mode this pins is the one
    `metal-sliding-window-attention-plan.md` §1a documents: a matcher that
    accepts on skeleton alone and silently drops a mask term.
    """
    M, d, num_sinks, window_size = 64, 16, 4, 32
    Q, K, V, out = _inputs(M, d, seed=0x3)
    sentinel = 1234.5
    out = torch.full((M, d), sentinel, dtype=torch.float32, device="mps").contiguous()
    try:
        _solve(Q, K, V, out, M, d, num_sinks, window_size,
               kernel=_attention_sinks_no_escape_kernel)
    except RuntimeError:
        return  # refused, which is the expected outcome today
    # If some future path DOES compile it, it must compute the variant's own
    # semantics — not the sink-escape ones.
    torch.mps.synchronize()
    ref = _reference(Q, K, V, M, d, num_sinks, window_size, sink_escape=False)
    err = (out.cpu().double() - ref).abs().max().item()
    assert err <= 1e-6, (
        "a near-miss kernel was claimed and compiled to the WRONG semantics "
        f"(max abs err {err:.3e} against its own reference)"
    )


def test_attention_with_sinks_wide_head_is_correct_not_rejected():
    """`BLOCK_D = 128` used to be a hard reject: `2*32*128 + 64 = 8256`
    threadgroup floats against Apple's 8192, and the predecessor op's body had
    nowhere else to go.

    `metal.fused_attention` chunks the query-row block to fit the budget
    instead of declining, so the shape now compiles and must be RIGHT. The
    assertion is numeric on purpose — what can go wrong with a chunked body is
    a wrong answer, not a crash, and a bare "it compiles now" check would pass
    on a body that dropped half the rows."""
    M, d, num_sinks, window_size = 64, 128, 4, 32
    Q, K, V, out = _inputs(M, d, seed=0x4)
    _solve(Q, K, V, out, M, d, num_sinks, window_size)
    torch.mps.synchronize()
    ref = _reference(Q, K, V, M, d, num_sinks, window_size)
    err = (out.cpu().double() - ref).abs().max().item()
    assert err <= 1e-5, f"max abs err {err:.3e}"


def test_attention_with_sinks_writes_every_element():
    """Regression lock for the silent-no-write mode.

    The sibling sliding-window kernel once compiled "successfully" while
    writing nothing at all (its scalar operands had been bound to buffer 0).
    The same backstop applies here: `metal.sink_attention`'s `bufName` refuses
    to fall back to buffer 0, so either the launch raises or every output
    element is written.
    """
    M, d, num_sinks, window_size = 100, 32, 4, 48
    torch.manual_seed(0x5)
    mk = lambda: torch.randn(M, d, dtype=torch.float32, device="mps").contiguous()
    Q, K, V = mk(), mk(), mk()
    sentinel = 4321.0
    out = torch.full((M, d), sentinel, dtype=torch.float32, device="mps").contiguous()
    _solve(Q, K, V, out, M, d, num_sinks, window_size)
    torch.mps.synchronize()
    untouched = int((out.cpu() == sentinel).sum())
    assert untouched == 0, (
        f"kernel reported success but left {untouched}/{M * d} elements at the "
        "sentinel value"
    )
