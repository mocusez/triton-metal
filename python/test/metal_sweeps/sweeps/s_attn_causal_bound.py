"""Pruning the causal half of the key sweep: is every answer still exact?

A causal kernel writes its mask INSIDE the score region and its key loop over
the WHOLE range, so the emitter -- which reproduces the source's own loop bound
-- used to sweep every key block. For the blocks above the diagonal every score
is the region's false arm, which under an online softmax is a -inf logit and so
exactly zero weight, leaving the running (max, sum, accumulator) untouched.
`fusedAttentionKeysAboveRowAreInert` recognises that structurally and the
emitter clamps `_fa_kend` to `_fa_rowoff + bm`.

That is a semantic claim used to DELETE work, which is the most dangerous kind
of change in this backend, so the axes here are the ones that decide where the
deleted region STARTS and ENDS:

  M                       lands the diagonal inside a key block, on a block
                          boundary, and past the tail -- including M < BM and
                          M < BN, where the clamp and the ragged-tail guard
                          interact
  BM x BN                 the clamp is `rowoff + BM` and the loop steps by BN,
                          so every ratio of the two is a different rounding
  num_warps               decides the threadgroup width and (since 2026-09-16)
                          the lanes-per-row split of the pass that consumes the
                          scores
  causal vs plain         the predicate must NOT fire without a `row >= key`
                          conjunct; the plain arm is the control, and a WRONG
                          there would mean it fired when it must not

Both arms are compared against `scaled_dot_product_attention`, which is the
oracle that does not share the backend's idea of the mask.

Run:  ./drive.sh sweeps/s_attn_causal_bound.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from sweep_lib import Sweep  # noqa: E402


@triton.jit
def _attn(q, k, v, out, M, d, BM: tl.constexpr, BD: tl.constexpr,
          BN: tl.constexpr, CAUSAL: tl.constexpr):
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
        if CAUSAL:
            ok = (qo[:, None] >= so[None, :]) & sm[None, :]
        else:
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


# Around every block boundary the clamp can land on, plus a ragged tail and the
# degenerate M == 1.
SIZES = (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 129)

sw = Sweep("s_attn_causal_bound")

for causal in (True, False):
    for M in SIZES:
        for bm in (8, 16, 32, 64):
            for bn in (8, 16, 32, 64):
                for nw in (1, 4):
                    d = 16
                    tag = (f"{'causal' if causal else 'plain'}_M{M}_bm{bm}"
                           f"_bn{bn}_nw{nw}_d{d}")

                    def run(causal=causal, M=M, bm=bm, bn=bn, nw=nw, d=d):
                        torch.manual_seed(0)
                        q = torch.randn(M, d)
                        k = torch.randn(M, d)
                        v = torch.randn(M, d)
                        out = torch.zeros(M, d, device="mps")
                        _attn[(triton.cdiv(M, bm),)](
                            q.to("mps"), k.to("mps"), v.to("mps"), out, M, d,
                            bm, max(16, triton.next_power_of_2(d)), bn,
                            causal, num_warps=nw)
                        torch.mps.synchronize()
                        ref = torch.nn.functional.scaled_dot_product_attention(
                            q.double(), k.double(), v.double(),
                            is_causal=causal).float()
                        err = (out.cpu() - ref).abs().max().item()
                        tol = 2e-4 * max(1.0, ref.abs().max().item())
                        if err <= tol:
                            return None
                        bad = (out.cpu() - ref).abs().max(dim=1).values > tol
                        return (f"maxerr={err:.3e} tol={tol:.1e} "
                                f"badrows={int(bad.sum())}/{M} "
                                f"first={int(bad.nonzero()[0]) if bad.any() else -1}")

                    sw.run(tag, run)

sw.finish()
sys.exit(1 if sw.wrong else 0)
