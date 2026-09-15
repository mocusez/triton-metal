"""Attention query rows against the threadgroup width: right answers?

`metal.fused_attention`'s MMA bodies launch `min(source num_warps, bm/8) * 32`
threads -- the cap exists so no SIMD-group starts without an 8-row QK/PV tile
to run. Nothing raises it. But the score transform, the online-softmax update
and the epilogue store are each ONE QUERY ROW PER COOPERATIVE THREAD, so a
source asking for one warp at bm = 64 hands 32 threads 64 rows. They read
`q = _fa_tid` under a `q < bm` bound, which is not a loop, and the rows at and
above the threadgroup width were never transformed, never normalised and never
stored.

That shape answered silently wrong, not loudly: no diagnostic, no signal, just
zeros where the tail rows should be. So a decline-verdict probe cannot see this
family -- only an oracle can, which is why this is a `Sweep` and not a
`probe_lib` round.

Axes, and why each one is here:

  BM {8, 16, 32, 64}      the query-row block. The whole family lives in
                          `BM` vs `min(nw, BM/8) * 32`; only BM = 64 at one
                          warp crosses it today, but the crossing is what is
                          being swept, not the constant.
  num_warps {1, 2, 4, 8}  decides the threadgroup width, and separately how
                          many row tiles have an owner.
  d_head {16..256}        picks the body: small working sets take the
                          full-tile MMA body, larger ones the CHUNKED one, and
                          BOTH carry the three per-row passes. From
                          `bm * bd >= 8192` the O accumulator alone fills the
                          threadgroup budget and the chunked body walks the
                          program's rows in several ROW-BLOCKED passes -- a
                          third shape of the same code, and the one whose back
                          edge carries a write-after-read hazard on the O
                          buffer and the running state.
  N {aligned, ragged}     `row < M` is the OTHER guard on the same passes;
                          a ragged tail is where the two guards interact.
  kernel {causal, plain}  causal reaches the key-bounds region, plain does not.

`BM` is swept only over powers of two: `tl.arange` refuses the rest in the
Triton FRONTEND, so 24/40/48/56 are not reachable from a kernel at all.

Run:  ./drive.sh sweeps/s_attn_rows.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from sweep_lib import Sweep  # noqa: E402


@triton.jit
def causal_attn(q, k, v, out, M, d, BM: tl.constexpr, BD: tl.constexpr,
                BN: tl.constexpr):
    """`fixtures/metal_leet/hard-causal_self-Attention.py`, verbatim in shape."""
    pid = tl.program_id(0)
    qo = pid * BM + tl.arange(0, BM)
    qm = qo < M
    co = tl.arange(0, BD)
    cm = co < d
    qs = tl.load(q + qo[:, None] * d + co[None, :],
                 mask=qm[:, None] & cm[None, :], other=0.0)
    scale = 1 / tl.sqrt(d.to(tl.float32))
    smax = tl.full([BM], -float('inf'), dtype=tl.float32)
    ssum = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, BD], dtype=tl.float32)
    for s in tl.range(0, M, BN):
        so = s + tl.arange(0, BN)
        sm = so < M
        sp = so[:, None] * d + co[None, :]
        ks = tl.load(k + sp, mask=sm[:, None] & cm[None, :], other=0.0)
        vs = tl.load(v + sp, mask=sm[:, None] & cm[None, :], other=0.0)
        a = tl.dot(qs, tl.trans(ks)) * scale
        ok = (qo[:, None] >= so[None, :]) & sm[None, :]
        a = tl.where(ok, a, -float('inf'))
        nm = tl.maximum(smax, tl.max(a, axis=1))
        alpha = tl.exp(smax - nm)
        smax = nm
        a = tl.exp(a - nm[:, None])
        ssum = tl.fma(ssum, alpha, tl.sum(a, axis=1))
        acc = tl.fma(acc, alpha[:, None], tl.dot(a, vs))
    acc = acc / ssum[:, None]
    tl.store(out + qo[:, None] * d + co[None, :], acc,
             mask=qm[:, None] & cm[None, :])


@triton.jit
def plain_attn(q, k, v, out, M, d, BM: tl.constexpr, BD: tl.constexpr,
               BN: tl.constexpr):
    """The same body with no causal term: the key-bounds region is trivial."""
    pid = tl.program_id(0)
    qo = pid * BM + tl.arange(0, BM)
    qm = qo < M
    co = tl.arange(0, BD)
    cm = co < d
    qs = tl.load(q + qo[:, None] * d + co[None, :],
                 mask=qm[:, None] & cm[None, :], other=0.0)
    scale = 1 / tl.sqrt(d.to(tl.float32))
    smax = tl.full([BM], -float('inf'), dtype=tl.float32)
    ssum = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, BD], dtype=tl.float32)
    for s in tl.range(0, M, BN):
        so = s + tl.arange(0, BN)
        sm = so < M
        sp = so[:, None] * d + co[None, :]
        ks = tl.load(k + sp, mask=sm[:, None] & cm[None, :], other=0.0)
        vs = tl.load(v + sp, mask=sm[:, None] & cm[None, :], other=0.0)
        a = tl.dot(qs, tl.trans(ks)) * scale
        ok = qm[:, None] & sm[None, :]
        a = tl.where(ok, a, -float('inf'))
        nm = tl.maximum(smax, tl.max(a, axis=1))
        alpha = tl.exp(smax - nm)
        smax = nm
        a = tl.exp(a - nm[:, None])
        ssum = tl.fma(ssum, alpha, tl.sum(a, axis=1))
        acc = tl.fma(acc, alpha[:, None], tl.dot(a, vs))
    acc = acc / ssum[:, None]
    tl.store(out + qo[:, None] * d + co[None, :], acc,
             mask=qm[:, None] & cm[None, :])


KERNELS = {"causal": (causal_attn, True), "plain": (plain_attn, False)}

sw = Sweep("s_attn_rows")

for kname, (kern, causal) in KERNELS.items():
    for d in (16, 32, 64, 128, 256):
        for bm in (8, 16, 32, 64):
            for bn in (16, 64):
                for nw in (1, 2, 4, 8):
                    for nlab, M in (("aligned", 128), ("ragged", 100)):
                        tag = (f"{kname}_d{d}_bm{bm}_bn{bn}_nw{nw}_{nlab}"
                               f"_tpg{min(nw, max(1, bm // 8)) * 32}")

                        def run(kern=kern, causal=causal, d=d, bm=bm, bn=bn,
                                nw=nw, M=M):
                            torch.manual_seed(0)
                            q = torch.randn(M, d)
                            k = torch.randn(M, d)
                            v = torch.randn(M, d)
                            out = torch.zeros(M, d, device="mps")
                            kern[(triton.cdiv(M, bm),)](
                                q.to("mps"), k.to("mps"), v.to("mps"), out, M, d,
                                bm, max(16, triton.next_power_of_2(d)), bn,
                                num_warps=nw)
                            torch.mps.synchronize()
                            ref = torch.nn.functional.scaled_dot_product_attention(
                                q.double(), k.double(), v.double(),
                                is_causal=causal).float()
                            err = (out.cpu() - ref).abs().max().item()
                            tol = 2e-4 * max(1.0, ref.abs().max().item())
                            if err <= tol:
                                return None
                            # Name the tail explicitly: the failure mode of this
                            # family is whole ROWS left at their initial value,
                            # not a drifting numeric error.
                            bad = (out.cpu() - ref).abs().max(dim=1).values > tol
                            return (f"maxerr={err:.3e} tol={tol:.1e} "
                                    f"badrows={int(bad.sum())}/{M} "
                                    f"first={int(bad.nonzero()[0]) if bad.any() else -1}")

                        sw.run(tag, run)

sw.finish()
sys.exit(1 if sw.wrong else 0)
