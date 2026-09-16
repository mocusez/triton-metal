"""The other five attention kernels, against block shape and warp count.

`s_attn_rows.py` and `s_attn_causal_bound.py` sweep two kernels: a causal one
and a plain one. But every change the fused-attention emitter has taken lands
in code that ALL of its matches run -- the per-row passes, the warp/tile-pair
distribution, the lane split, the key-bounds region and the row-blocked
chunked body are one body of code with a schedule in front of it. The five
kernels in `fixtures/metal_leet/` that reach the same emitter by a different
route were covered at ONE shape each, by `leet-all`, with the block sizes their
own `solve()` hardcodes -- which is the shape most likely to have been the one
that was tried while the emitter was being written.

So sweep them the way the causal one is swept. Each kernel is imported from its
fixture and launched DIRECTLY, so the body is the corpus file verbatim and only
the constexprs move.

  sliding  `hard-sliding_window_self_attention` -- a SYMMETRIC |i-j| <= w band,
           so keys above the row are live and the causal key-bound predicate
           must not fire. The control for the pruning change.
  sinks    `medium-attention_with_sinks` -- TWO score regions in one kernel (a
           sink block, then a windowed scan) updating one online-softmax state,
           and the only one whose key range is computed from the program id.
  alibi    `medium-attention_with_linear_biases` -- a score transform that is
           not a mask: `+ alpha*(i-j)` is finite everywhere, so no key is
           inert, and the kernel carries a SECOND reduction loop over d.
  gqa      `medium-grouped-query-attention` -- a 2-D grid whose second axis
           picks a different K/V head; the row block is indexed off axis 0
           while the buffers move under axis 1.
  decay    `medium-decaying_causal_attention` -- norm = NONE. No softmax, no
           division: the epilogue that the other four exercise is absent, and
           the causal term is a MULTIPLICATIVE decay, not a select, so the
           key-bound predicate must decline it and the answer must still be
           right.

Axes: d_head {16, 64, 128, 256} x BM {16, 32, 64} x BN {16, 64} x
num_warps {1, 2, 4, 8} x M {aligned 128, ragged 100}.

  BM against num_warps      the MMA bodies launch min(nw, BM/8)*32 threads and
                            nothing raises it, so the per-row passes are handed
                            more rows than threads at BM = 64, nw = 1.
  d_head                    picks the body: full-tile, chunked, or -- from
                            `BM*BD >= 8192`, where the O accumulator alone
                            fills Apple's 32 KiB -- the ROW-BLOCKED chunked
                            body, whose back edge is a write-after-read hazard
                            on the O buffer and the running state.
  M ragged                  `row < M` is the other guard on the same passes.

The reference is torch in double precision, written from each kernel's own
arithmetic -- including, for `sinks`, the key range its block loop actually
visits rather than the range it means to. An oracle nobody checked is a guess,
so `tools/check_attn_variant_refs.py` runs every kernel under
`TRITON_INTERPRET=1` and diffs it against these same functions.

Run:  ./drive.sh sweeps/s_attn_variants.py
"""

import importlib.util
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402
import triton  # noqa: E402

from sweep_lib import Sweep  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    *([os.pardir] * 4)))
LEET = os.path.join(REPO, "python", "test", "unit", "fixtures", "metal_leet")


def _fixture(stem):
    """Import a corpus file by path: the names carry dashes."""
    path = os.path.join(LEET, stem + ".py")
    spec = importlib.util.spec_from_file_location("attnvar_" + stem.replace("-", "_"),
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_SLIDING = _fixture("hard-sliding_window_self_attention")
_SINKS = _fixture("medium-attention_with_sinks")
_ALIBI = _fixture("medium-attention_with_linear_biases")
_GQA = _fixture("medium-grouped-query-attention")
_DECAY = _fixture("medium-decaying_causal_attention")

WINDOW = 20        # not a multiple of either BN, on purpose
SINK_WINDOW = 24
NUM_SINKS = 4
BLOCK_S = 16       # what the sinks fixture hardcodes
ALIBI_ALPHA = -0.1
GAMMA = 0.9
Q_HEADS, KV_HEADS = 4, 2


# --------------------------------------------------------------------------
# sliding window: a symmetric band, softmax over it
# --------------------------------------------------------------------------
def sliding_inputs(M, d):
    return [torch.randn(M, d), torch.randn(M, d), torch.randn(M, d)]


def sliding_launch(t, out, M, d, BM, BN, BD, nw):
    return _SLIDING.attention[(triton.cdiv(M, BM),)](
        t[0], t[1], t[2], out, M, M, d, WINDOW, BM, BN, BD, num_warps=nw)


def sliding_ref(t, M, d, BM, BN):
    q, k, v = (x.double() for x in t)
    s = (q @ k.T) / math.sqrt(d)
    i = torch.arange(M)[:, None]
    j = torch.arange(M)[None, :]
    valid = (i - j).abs() <= WINDOW
    # The kernel takes its running max over `where(valid, qk, -100)`, so the
    # floor participates. It only ever wins if no key in a row beats -100.
    floored = torch.where(valid, s, torch.full_like(s, -100.0))
    mx = floored.max(dim=1, keepdim=True).values
    p = torch.where(valid, torch.exp(s - mx), torch.zeros_like(s))
    return (p @ v) / p.sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------
# sinks: a sink block plus a windowed scan, one softmax state
# --------------------------------------------------------------------------
def sinks_inputs(M, d):
    return [torch.randn(M, d), torch.randn(M, d), torch.randn(M, d)]


def _sinks_local_blocks(BM, BN):
    return triton.cdiv(SINK_WINDOW + BM - 1, BN)


def sinks_launch(t, out, M, d, BM, BN, BD, nw):
    q, k, v = t
    sm_scale = 1.4426950408889634 / (d ** 0.5)
    return _SINKS._attention_sinks_kernel[(triton.cdiv(M, BM),)](
        q, k, v, out, M, d, NUM_SINKS, SINK_WINDOW, sm_scale,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD, BLOCK_S=BLOCK_S,
        N_LOCAL_BLOCKS=_sinks_local_blocks(BM, BN), num_warps=nw, num_stages=1)


def sinks_ref(t, M, d, BM, BN):
    """Block-exact: the keys the loop VISITS, not the keys it means to."""
    q, k, v = (x.double() for x in t)
    s = (q @ k.T) / math.sqrt(d)
    n_local = _sinks_local_blocks(BM, BN)
    j = torch.arange(M)[None, :]
    out = torch.zeros(M, d, dtype=torch.float64)
    for start_m in range(0, M, BM):
        stop = min(start_m + BM, M)
        i = torch.arange(start_m, stop)[:, None]
        # sink columns: arange(BLOCK_S) kept by `offs_s < num_sinks`, and the
        # loads carry no `j < M` bound at all.
        sink = (j < min(NUM_SINKS, BLOCK_S)) & (j <= i)
        local_start = max(start_m - SINK_WINDOW + 1, NUM_SINKS)
        scanned = (j >= local_start) & (j < local_start + n_local * BN)
        local = scanned & (j < M) & (j <= i) & (j >= i - SINK_WINDOW + 1) \
            & (j >= NUM_SINKS)
        valid = sink | local
        blk = s[start_m:stop]
        mx = torch.where(valid, blk, torch.full_like(blk, -float("inf")))
        mx = mx.max(dim=1, keepdim=True).values
        p = torch.where(valid, torch.exp(blk - mx), torch.zeros_like(blk))
        out[start_m:stop] = (p @ v) / p.sum(dim=1, keepdim=True)
    return out


# --------------------------------------------------------------------------
# alibi: a finite additive bias, and a second reduction loop over d
# --------------------------------------------------------------------------
def alibi_inputs(M, d):
    return [torch.randn(M, d), torch.randn(M, d), torch.randn(M, d)]


def alibi_launch(t, out, M, d, BM, BN, BD, nw):
    q, k, v = t
    kt = k.T.contiguous()
    return _ALIBI.alibi_attention_fwd[(triton.cdiv(M, BM), triton.cdiv(d, BD))](
        q, kt, v, out, M, M, d, ALIBI_ALPHA, 1.0 / math.sqrt(d),
        q.stride(0), q.stride(1), kt.stride(0), kt.stride(1),
        v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_D=BD, num_warps=nw)


def alibi_ref(t, M, d, BM, BN):
    q, k, v = (x.double() for x in t)
    s = (q @ k.T) / math.sqrt(d)
    i = torch.arange(M)[:, None].double()
    j = torch.arange(M)[None, :].double()
    s = s + ALIBI_ALPHA * (i - j)
    return torch.softmax(s, dim=1) @ v


# --------------------------------------------------------------------------
# gqa: a second grid axis choosing the K/V head
# --------------------------------------------------------------------------
def gqa_inputs(S, D):
    return [torch.randn(Q_HEADS, S, D), torch.randn(KV_HEADS, S, D),
            torch.randn(KV_HEADS, S, D)]


def gqa_out(S, D):
    return torch.zeros(Q_HEADS, S, D, device="mps")


def gqa_launch(t, out, S, D, BM, BN, BD, nw):
    q, k, v = t
    return _GQA._gqa_kernel[(triton.cdiv(S, BM), Q_HEADS)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        S, D, 1.0 / math.sqrt(D), Q_HEADS // KV_HEADS,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD, num_warps=nw)


def gqa_ref(t, S, D, BM, BN):
    q, k, v = (x.double() for x in t)
    groups = Q_HEADS // KV_HEADS
    out = torch.zeros(Q_HEADS, S, D, dtype=torch.float64)
    for h in range(Q_HEADS):
        kv = h // groups
        s = (q[h] @ k[kv].T) / math.sqrt(D)
        out[h] = torch.softmax(s, dim=1) @ v[kv]
    return out


# --------------------------------------------------------------------------
# decay: norm = none, and a multiplicative causal term
# --------------------------------------------------------------------------
def decay_inputs(S, d):
    return [torch.randn(S, d), torch.randn(S, d), torch.randn(S, d)]


def decay_launch(t, out, S, d, BM, BN, BD, nw):
    q, k, v = t
    return _DECAY._decay_causal_attn_fwd[(triton.cdiv(S, BM),)](
        q, k, v, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        S, d, math.log2(GAMMA), 1.0 / math.sqrt(d),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD, num_warps=nw, num_stages=1)


def decay_ref(t, S, d, BM, BN):
    q, k, v = (x.double() for x in t)
    s = (q @ k.T) / math.sqrt(d)
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    diff = (i - j).double()
    w = torch.where(diff >= 0, torch.pow(torch.tensor(GAMMA, dtype=torch.float64),
                                         diff), torch.zeros_like(diff))
    return (s * w) @ v


VARIANTS = {
    "sliding": (sliding_inputs, sliding_launch, sliding_ref, None),
    "sinks": (sinks_inputs, sinks_launch, sinks_ref, None),
    "alibi": (alibi_inputs, alibi_launch, alibi_ref, None),
    "gqa": (gqa_inputs, gqa_launch, gqa_ref, gqa_out),
    "decay": (decay_inputs, decay_launch, decay_ref, None),
}

D_HEADS = (16, 64, 128, 256)
BLOCK_MS = (16, 32, 64)
BLOCK_NS = (16, 64)
WARPS = (1, 2, 4, 8)
LENGTHS = (("aligned", 128), ("ragged", 100))


def cell(name, d, bm, bn, nw, M):
    """Run one kernel once and diff it against its double-precision oracle."""
    mk_in, launch, ref_fn, mk_out = VARIANTS[name]
    bd = max(16, triton.next_power_of_2(d))
    torch.manual_seed(0)
    host = mk_in(M, d)
    dev = [x.to("mps") for x in host]
    out = mk_out(M, d) if mk_out else torch.zeros(M, d, device="mps")
    launch(dev, out, M, d, bm, bn, bd, nw)
    torch.mps.synchronize()
    ref = ref_fn(host, M, d, bm, bn)
    got = out.cpu().double()
    err = (got - ref).abs().max().item()
    tol = 2e-4 * max(1.0, ref.abs().max().item())
    if err <= tol:
        return None
    # This family fails by leaving whole ROWS at their initial value, so say
    # how many rows moved rather than only how far the worst element did.
    per_row = (got - ref).abs().flatten(0, -2).max(dim=-1).values
    bad = per_row > tol
    first = int(bad.nonzero()[0][0]) if bad.any() else -1
    return (f"maxerr={err:.3e} tol={tol:.1e} "
            f"badrows={int(bad.sum())}/{bad.numel()} first={first}")


def main():
    sw = Sweep("s_attn_variants")
    for name in VARIANTS:
        for d in D_HEADS:
            for bm in BLOCK_MS:
                for bn in BLOCK_NS:
                    for nw in WARPS:
                        for nlab, M in LENGTHS:
                            tpg = min(nw, max(1, bm // 8)) * 32
                            bd = max(16, triton.next_power_of_2(d))
                            # The row-block crossing is a RELATION, so put it
                            # in the tag: it cannot be grepped out of the axes.
                            rows = "blocked" if bm * bd >= 8192 else "one"
                            tag = (f"{name}_d{d}_bm{bm}_bn{bn}_nw{nw}_{nlab}"
                                   f"_tpg{tpg}_rows{rows}")
                            sw.run(tag, lambda name=name, d=d, bm=bm, bn=bn,
                                   nw=nw, M=M: cell(name, d, bm, bn, nw, M))
    sw.finish()
    return 1 if sw.wrong else 0


if __name__ == "__main__":
    sys.exit(main())
