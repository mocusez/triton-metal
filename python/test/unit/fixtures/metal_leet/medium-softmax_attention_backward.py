import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources


@triton.jit(do_not_specialize=["N_true"])
def _attn_bwd_pre(
    Q, KT, VT, dO,
    S, ST, DP, DPT,
    Mp, Lp, Deltap,
    stride_qm, stride_kt, stride_vt, stride_dom,
    stride_sm, stride_stn, stride_dpm, stride_dptn,
    M, N_true, D, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask_md = mask_m[:, None] & mask_d[None, :]

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_md, other=0.0)
    do = tl.load(dO + offs_m[:, None] * stride_dom + offs_d[None, :], mask=mask_md, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    delta = tl.zeros([BLOCK_M], tl.float32)

    for start_n in range(0, stride_sm, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < stride_sm
        mask_dn = mask_d[:, None] & mask_n[None, :]
        kt = tl.load(KT + offs_d[:, None] * stride_kt + offs_n[None, :], mask=mask_dn, other=0.0)
        vt = tl.load(VT + offs_d[:, None] * stride_vt + offs_n[None, :], mask=mask_dn, other=0.0)

        s = tl.dot(q, kt, input_precision="ieee") * sm_scale
        s = tl.where(offs_n[None, :] < N_true, s, float("-inf"))
        dp = tl.dot(do, vt, input_precision="ieee")

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        delta = delta * alpha + tl.sum(p * dp, 1)
        m_i = m_new

        mask_mn = mask_m[:, None] & mask_n[None, :]
        tl.store(S + offs_m[:, None] * stride_sm + offs_n[None, :], s, mask=mask_mn)
        tl.store(DP + offs_m[:, None] * stride_dpm + offs_n[None, :], dp, mask=mask_mn)
        st = tl.trans(s)
        dpt = tl.trans(dp)
        mask_nm = mask_n[:, None] & mask_m[None, :]
        tl.store(ST + offs_n[:, None] * stride_stn + offs_m[None, :], st, mask=mask_nm)
        tl.store(DPT + offs_n[:, None] * stride_dptn + offs_m[None, :], dpt, mask=mask_nm)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(Mp + offs_m, m_i, mask=mask_m)
    tl.store(Lp + offs_m, l_i, mask=mask_m)
    tl.store(Deltap + offs_m, delta / l_safe, mask=mask_m)


@triton.jit
def _attn_bwd_dq(
    K, S, DP, Mp, Lp, Deltap, DQ,
    stride_kn, stride_sm, stride_dpm, stride_dqm,
    M, D, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D

    m_i = tl.load(Mp + offs_m, mask=mask_m, other=0.0)
    l_i = tl.load(Lp + offs_m, mask=mask_m, other=1.0)
    delta = tl.load(Deltap + offs_m, mask=mask_m, other=0.0)
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)

    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for start_n in range(0, stride_sm, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < stride_sm
        mask_mn = mask_m[:, None] & mask_n[None, :]
        s = tl.load(S + offs_m[:, None] * stride_sm + offs_n[None, :], mask=mask_mn,
                    other=float("-inf"))
        dp = tl.load(DP + offs_m[:, None] * stride_dpm + offs_n[None, :], mask=mask_mn,
                     other=0.0)
        k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        p = tl.exp(s - m_i[:, None]) / l_safe[:, None]
        ds = p * (dp - delta[:, None]) * sm_scale
        acc = tl.dot(ds, k, acc, input_precision="ieee")

    tl.store(DQ + offs_m[:, None] * stride_dqm + offs_d[None, :], acc,
             mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _attn_bwd_dkdv(
    Q, dO, ST, DPT, Mp, Lp, Deltap, DK, DV,
    stride_qm, stride_dom, stride_stn, stride_dptn,
    stride_dkn, stride_dvn,
    M, N, D, sm_scale, accumulate,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < N
    mask_d = offs_d < D

    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        mask_nm = mask_n[:, None] & mask_m[None, :]
        st = tl.load(ST + offs_n[:, None] * stride_stn + offs_m[None, :], mask=mask_nm,
                     other=float("-inf"))
        dpt = tl.load(DPT + offs_n[:, None] * stride_dptn + offs_m[None, :], mask=mask_nm,
                      other=0.0)
        m_i = tl.load(Mp + offs_m, mask=mask_m, other=0.0)
        l_i = tl.load(Lp + offs_m, mask=mask_m, other=1.0)
        delta = tl.load(Deltap + offs_m, mask=mask_m, other=0.0)
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do = tl.load(dO + offs_m[:, None] * stride_dom + offs_d[None, :],
                     mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        p = tl.exp(st - m_i[None, :]) / l_safe[None, :]
        ds = p * (dpt - delta[None, :]) * sm_scale
        acc_dv = tl.dot(p, do, acc_dv, input_precision="ieee")
        acc_dk = tl.dot(ds, q, acc_dk, input_precision="ieee")

    mask_nd = mask_n[:, None] & mask_d[None, :]
    if accumulate != 0:
        prev_k = tl.load(DK + offs_n[:, None] * stride_dkn + offs_d[None, :], mask=mask_nd,
                         other=0.0)
        prev_v = tl.load(DV + offs_n[:, None] * stride_dvn + offs_d[None, :], mask=mask_nd,
                         other=0.0)
        acc_dk += prev_k
        acc_dv += prev_v
    tl.store(DK + offs_n[:, None] * stride_dkn + offs_d[None, :], acc_dk, mask=mask_nd)
    tl.store(DV + offs_n[:, None] * stride_dvn + offs_d[None, :], acc_dv, mask=mask_nd)


_SCRATCH_ELEMS = 1 << 27
_PRE_CONFIGS = [(16, 32), (16, 16)]
_DQ_CONFIGS = [(32, 32), (16, 32)]
_DKDV_CONFIGS = [(32, 32), (32, 16)]


def _pad16(n):
    return (n + 15) // 16 * 16


def _launch_pre(grid, args, M, N_true, D, sm_scale, BLOCK_D):
    last_error = None
    for bm, bn in _PRE_CONFIGS:
        try:
            return _attn_bwd_pre[grid(bm)](
                *args, M, N_true, D, sm_scale, BLOCK_M=bm,
                BLOCK_N=bn, BLOCK_D=BLOCK_D, num_warps=8, num_stages=1)
        except OutOfResources as exc:
            last_error = exc
    raise last_error or RuntimeError("no attention prepass launch configuration succeeded")


def _launch_dq(grid, args, M, D, sm_scale, BLOCK_D):
    last_error = None
    for bm, bn in _DQ_CONFIGS:
        try:
            return _attn_bwd_dq[grid(bm)](
                *args, M, D, sm_scale, BLOCK_M=bm, BLOCK_N=bn,
                BLOCK_D=BLOCK_D, num_warps=8, num_stages=1)
        except OutOfResources as exc:
            last_error = exc
    raise last_error or RuntimeError("no attention dQ launch configuration succeeded")


def _launch_dkdv(grid, args, M, N, D, sm_scale, accumulate, BLOCK_D):
    last_error = None
    for bn, bm in _DKDV_CONFIGS:
        try:
            return _attn_bwd_dkdv[grid(bn)](
                *args, M, N, D, sm_scale, accumulate, BLOCK_N=bn,
                BLOCK_M=bm, BLOCK_D=BLOCK_D, num_warps=8, num_stages=1)
        except OutOfResources as exc:
            last_error = exc
    raise last_error or RuntimeError("no attention dK/dV launch configuration succeeded")


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, dO: torch.Tensor,
          dQ: torch.Tensor, dK: torch.Tensor, dV: torch.Tensor,
          M: int, N: int, d: int):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    dO = dO.contiguous()

    sm_scale = 1.0 / (d ** 0.5)
    BLOCK_D = 128 if d <= 128 else triton.next_power_of_2(d)
    dev = Q.device
    f32 = torch.float32

    M2, N2, d2 = _pad16(M), _pad16(N), _pad16(d)
    padded = (M2, N2, d2) != (M, N, d)

    if padded:
        Qp = Q.new_zeros((M2, d2)); Qp[:M, :d].copy_(Q)
        Kp = K.new_zeros((N2, d2)); Kp[:N, :d].copy_(K)
        Vp = V.new_zeros((N2, d2)); Vp[:N, :d].copy_(V)
        dOp = dO.new_zeros((M2, d2)); dOp[:M, :d].copy_(dO)
        dQp = torch.empty((M2, d2), device=dev, dtype=f32)
        dKp = torch.empty((N2, d2), device=dev, dtype=f32)
        dVp = torch.empty((N2, d2), device=dev, dtype=f32)
    else:
        Qp, Kp, Vp, dOp = Q, K, V, dO
        dQp = dQ if dQ.is_contiguous() else torch.empty((M2, d2), device=dev, dtype=f32)
        dKp = dK if dK.is_contiguous() else torch.empty((N2, d2), device=dev, dtype=f32)
        dVp = dV if dV.is_contiguous() else torch.empty((N2, d2), device=dev, dtype=f32)

    KT = Kp.t().contiguous()
    VT = Vp.t().contiguous()

    m_chunk = M2 if M2 * N2 <= _SCRATCH_ELEMS else max(16, (_SCRATCH_ELEMS // N2) // 16 * 16)
    multi = m_chunk < M2

    S = torch.empty((m_chunk, N2), device=dev, dtype=f32)
    ST = torch.empty((N2, m_chunk), device=dev, dtype=f32)
    DP = torch.empty((m_chunk, N2), device=dev, dtype=f32)
    DPT = torch.empty((N2, m_chunk), device=dev, dtype=f32)
    Mp = torch.empty(m_chunk, device=dev, dtype=f32)
    Lp = torch.empty(m_chunk, device=dev, dtype=f32)
    Deltap = torch.empty(m_chunk, device=dev, dtype=f32)

    if multi:
        dKp.zero_()
        dVp.zero_()

    for m0 in range(0, M2, m_chunk):
        mc = min(m_chunk, M2 - m0)
        q_c = Qp[m0:m0 + mc]
        do_c = dOp[m0:m0 + mc]
        dq_c = dQp[m0:m0 + mc]

        pre_args = (q_c, KT, VT, do_c, S, ST, DP, DPT, Mp, Lp, Deltap,
                    q_c.stride(0), KT.stride(0), VT.stride(0), do_c.stride(0),
                    S.stride(0), ST.stride(0), DP.stride(0), DPT.stride(0))
        pre_compiled = _launch_pre(
            lambda bm: (triton.cdiv(mc, bm),), pre_args,
            mc, N, d2, sm_scale, BLOCK_D)

        dq_args = (Kp, S, DP, Mp, Lp, Deltap, dq_c,
                   Kp.stride(0), S.stride(0), DP.stride(0), dq_c.stride(0))
        dq_compiled = _launch_dq(
            lambda bm: (triton.cdiv(mc, bm),), dq_args,
            mc, d2, sm_scale, BLOCK_D)

        dkdv_args = (q_c, do_c, ST, DPT, Mp, Lp, Deltap, dKp, dVp,
                     q_c.stride(0), do_c.stride(0), ST.stride(0), DPT.stride(0),
                     dKp.stride(0), dVp.stride(0))
        dkdv_compiled = _launch_dkdv(
            lambda bn: (triton.cdiv(N2, bn),), dkdv_args,
            mc, N2, d2, sm_scale, 1 if (multi and m0 > 0) else 0, BLOCK_D)

    if padded or dQp is not dQ:
        dQ[:M, :d].copy_(dQp[:M, :d])
    if padded or dKp is not dK:
        dK[:N, :d].copy_(dKp[:N, :d])
    if padded or dVp is not dV:
        dV[:N, :d].copy_(dVp[:N, :d])

    # Tests inspect the generated MSL for each whole-kernel lowering.  Real
    # callers may ignore this return value, just as they ignored the launch
    # helper return values before this fixture exposed them.
    return pre_compiled, dq_compiled, dkdv_compiled
