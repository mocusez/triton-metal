"""Speculative-decoding verification on the Metal backend.

The verbatim leet-triton `medium-speculative_decoding_verification.py` kernel: a
per-batch sequential accept/reject loop with an early-return guard, an i1
loop-carried `accepted_all` flag, and an inverse-CDF sampler built on
`tl.cumsum` (prefix-sum) feeding a `tl.min` reduce. Exercises the full W-A..W-C
stack: rank-1 i32 min-reduce (arith.minsi), cf.cond_br early-return structuring,
scalar i32 load/store over multi-level addptr chains, the rich rank-1 reduce
cone (select/cmp/andi/masked-load via evalRank1ValueAt), and the `tt.scan`
(cumsum) lowering — a distributed threadgroup prefix-sum whose result is read
back per element as a staged leaf by the consuming min-reduce.

Numerically verified against a numpy reference (bit-exact on the integer output
tokens). The `_cumsum_min_idx` test isolates the scan → min-reduce dataflow.
"""

from __future__ import annotations

import numpy as np
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


@triton.jit
def _speculative_decoding_verification_kernel(
    draft_tokens_ptr, draft_probs_ptr, target_probs_ptr, uniform_samples_ptr,
    output_tokens_ptr, B, T, V, BLOCK_SIZE_V: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return
    b = pid
    for idx in range(T + 1):
        tl.store(output_tokens_ptr + b * (T + 1) + idx, 0)
    accepted_all = True
    for i in range(T):
        if accepted_all:
            t_i = tl.load(draft_tokens_ptr + b * T + i)
            p_i_ti = tl.load(draft_probs_ptr + b * T * V + i * V + t_i)
            q_i_ti = tl.load(target_probs_ptr + b * T * V + i * V + t_i)
            alpha_i = q_i_ti / p_i_ti
            if alpha_i > 1.0:
                alpha_i = 1.0
            u_i = tl.load(uniform_samples_ptr + b * (T + 1) + i)
            if u_i < alpha_i:
                tl.store(output_tokens_ptr + b * (T + 1) + i, t_i)
            else:
                accepted_all = False
                sum_adj = 0.0
                for v_offset in range(0, V, BLOCK_SIZE_V):
                    v_idx = v_offset + tl.arange(0, BLOCK_SIZE_V)
                    v_mask = v_idx < V
                    p_ptr = draft_probs_ptr + b * T * V + i * V + v_idx
                    q_ptr = target_probs_ptr + b * T * V + i * V + v_idx
                    p_val = tl.load(p_ptr, mask=v_mask, other=0.0)
                    q_val = tl.load(q_ptr, mask=v_mask, other=0.0)
                    adj_val = tl.where(q_val > p_val, q_val - p_val, 0.0)
                    adj_val = tl.where(v_mask, adj_val, 0.0)
                    sum_adj += tl.sum(adj_val, axis=0)
                r = tl.load(uniform_samples_ptr + b * (T + 1) + T)
                is_uniform = sum_adj <= 0.0
                target_r = tl.where(is_uniform, r * V, r * sum_adj)
                running_sum = 0.0
                chosen_k = V
                for v_offset in range(0, V, BLOCK_SIZE_V):
                    v_idx = v_offset + tl.arange(0, BLOCK_SIZE_V)
                    v_mask = v_idx < V
                    if is_uniform:
                        adj_val = tl.where(v_mask, 1.0, 0.0)
                    else:
                        p_ptr = draft_probs_ptr + b * T * V + i * V + v_idx
                        q_ptr = target_probs_ptr + b * T * V + i * V + v_idx
                        p_val = tl.load(p_ptr, mask=v_mask, other=0.0)
                        q_val = tl.load(q_ptr, mask=v_mask, other=0.0)
                        adj_val = tl.where(q_val > p_val, q_val - p_val, 0.0)
                        adj_val = tl.where(v_mask, adj_val, 0.0)
                    chunk_cumsum = tl.cumsum(adj_val, axis=0)
                    total_cumsum = running_sum + chunk_cumsum
                    cond = (total_cumsum >= target_r) & v_mask
                    v_idx_selected = tl.where(cond, v_idx, V)
                    min_idx = tl.min(v_idx_selected, axis=0)
                    chosen_k = tl.where((chosen_k == V) & (min_idx < V), min_idx, chosen_k)
                    running_sum += tl.sum(adj_val, axis=0)
                chosen_k = tl.where(chosen_k == V, V - 1, chosen_k)
                chosen_k = tl.maximum(0, tl.minimum(chosen_k, V - 1))
                tl.store(output_tokens_ptr + b * (T + 1) + i, chosen_k)
    if accepted_all:
        r = tl.load(uniform_samples_ptr + b * (T + 1) + T)
        running_sum = 0.0
        bonus_k = V
        for v_offset in range(0, V, BLOCK_SIZE_V):
            v_idx = v_offset + tl.arange(0, BLOCK_SIZE_V)
            v_mask = v_idx < V
            q_ptr = target_probs_ptr + b * T * V + (T - 1) * V + v_idx
            q_val = tl.load(q_ptr, mask=v_mask, other=0.0)
            q_val = tl.where(v_mask, q_val, 0.0)
            chunk_cumsum = tl.cumsum(q_val, axis=0)
            total_cumsum = running_sum + chunk_cumsum
            cond = (total_cumsum >= r) & v_mask
            v_idx_selected = tl.where(cond, v_idx, V)
            min_idx = tl.min(v_idx_selected, axis=0)
            bonus_k = tl.where((bonus_k == V) & (min_idx < V), min_idx, bonus_k)
            running_sum += tl.sum(q_val, axis=0)
        bonus_k = tl.where(bonus_k == V, V - 1, bonus_k)
        bonus_k = tl.maximum(0, tl.minimum(bonus_k, V - 1))
        tl.store(output_tokens_ptr + b * (T + 1) + T, bonus_k)


def _reference(dt, dp, tp, us, B, T, V):
    dt = dt.cpu().numpy()
    dp = dp.cpu().numpy().astype(np.float64)
    tp = tp.cpu().numpy().astype(np.float64)
    us = us.cpu().numpy().astype(np.float64)
    out = np.zeros((B, T + 1), dtype=np.int32)
    for b in range(B):
        accepted_all = True
        for i in range(T):
            if not accepted_all:
                break
            t_i = int(dt[b, i])
            alpha = min(tp[b, i, t_i] / dp[b, i, t_i], 1.0)
            if us[b, i] < alpha:
                out[b, i] = t_i
            else:
                accepted_all = False
                adj = np.maximum(tp[b, i] - dp[b, i], 0.0)
                sum_adj = adj.sum()
                r = us[b, T]
                if sum_adj <= 0.0:
                    target_r, dist = r * V, np.ones(V)
                else:
                    target_r, dist = r * sum_adj, adj
                cs = np.cumsum(dist)
                k = int(np.argmax(cs >= target_r)) if np.any(cs >= target_r) else V
                out[b, i] = max(0, min(V - 1 if k == V else k, V - 1))
        if accepted_all:
            cs = np.cumsum(tp[b, T - 1])
            r = us[b, T]
            k = int(np.argmax(cs >= r)) if np.any(cs >= r) else V
            out[b, T] = max(0, min(V - 1 if k == V else k, V - 1))
    return out


@pytest.mark.parametrize(
    "B, T, V, seed",
    [(2, 3, 8, 0), (4, 5, 100, 1), (1, 4, 777, 2), (3, 2, 1024, 3),
     (8, 6, 50, 4), (2, 8, 200, 5)],
)
def test_speculative_decoding_matches_reference(B, T, V, seed):
    torch.manual_seed(seed)
    dev = "mps"
    dt = torch.randint(0, V, (B, T), dtype=torch.int32, device=dev)
    dp = torch.rand((B, T, V), dtype=torch.float32, device=dev)
    dp /= dp.sum(-1, keepdim=True)
    tp = torch.rand((B, T, V), dtype=torch.float32, device=dev)
    tp /= tp.sum(-1, keepdim=True)
    us = torch.rand((B, T + 1), dtype=torch.float32, device=dev)
    out = torch.zeros((B, T + 1), dtype=torch.int32, device=dev)
    _speculative_decoding_verification_kernel[(B,)](
        dt, dp, tp, us, out, B, T, V, BLOCK_SIZE_V=1024)
    torch.mps.synchronize()
    ref = _reference(dt, dp, tp, us, B, T, V)
    np.testing.assert_array_equal(out.cpu().numpy(), ref)


@triton.jit
def _cumsum_min_idx_kernel(in_ptr, out_ptr, V, target, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < V
    v = tl.load(in_ptr + b * V + idx, mask=mask, other=0.0)
    cs = tl.cumsum(v, axis=0)               # tt.scan → threadgroup prefix-sum
    cond = (cs >= target) & mask
    sel = tl.where(cond, idx, V)
    tl.store(out_ptr + b, tl.min(sel, axis=0))   # min-reduce reads scanbuf[idx]


@pytest.mark.parametrize("B, V", [(1, 8), (4, 100), (3, 777), (2, 1024)])
def test_cumsum_min_idx(B, V):
    # Isolated inverse-CDF: min logical index where cumsum crosses a target.
    torch.manual_seed(V)
    inp = torch.rand(B, V, dtype=torch.float32, device="mps")
    target = 0.4 * inp.sum(dim=1).mean().item()
    out = torch.zeros(B, dtype=torch.int32, device="mps")
    _cumsum_min_idx_kernel[(B,)](inp, out, V, target, BLOCK=1024, num_warps=4)
    torch.mps.synchronize()
    cs = torch.cumsum(inp.cpu(), dim=1)
    ai = torch.arange(V)
    ref = torch.where(cs >= target, ai[None, :], torch.full_like(ai[None, :], V))
    ref = ref.min(dim=1).values.to(torch.int32)
    np.testing.assert_array_equal(out.cpu().numpy(), ref.numpy())


@triton.jit
def _cumsum_store_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.cumsum(x, axis=0), mask=mask)


@pytest.mark.parametrize("BLOCK", [8, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("n_frac", [1.0, 0.5])
def test_cumsum_stored_directly(BLOCK, n_frac):
    # The scan result consumed by a plain `tl.store` rather than by a reduce.
    #
    # Regression: ScanLowering's per-thread placeholder read `scanbuf[tid]`. That
    # is right only when each thread owns ONE element (BLOCK == tpb == 128). For
    # BLOCK > tpb the FuncOpLowering tile loop gives each thread E = BLOCK/tpb
    # elements, and all E of them read the same slot — the store came out as
    # `out[i] == scan[i // E]` while compiling and running clean. The reduce
    # consumer (test_cumsum_min_idx) was always correct because it re-reads
    # scanbuf[idx] through g_scanBuffers, which is why this went unnoticed.
    n = max(1, int(BLOCK * n_frac))
    torch.manual_seed(BLOCK + n)
    inp = torch.randn(n, dtype=torch.float32, device="mps")
    out = torch.zeros(n, dtype=torch.float32, device="mps")
    _cumsum_store_kernel[(1,)](inp, out, n, BLOCK=BLOCK, num_warps=4)
    torch.mps.synchronize()
    ref = torch.cumsum(inp.cpu().double(), dim=0)
    np.testing.assert_allclose(
        out.cpu().double().numpy(), ref.numpy(), rtol=0, atol=2e-5)

@pytest.mark.parametrize("BLOCK, num_warps", [
    (8, 1), (16, 1), (32, 1), (64, 2),      # BLOCK == tpb, or sub-warp
    (8, 4), (16, 4), (32, 4), (64, 4),      # BLOCK < tpb (128): padded window
])
def test_cumsum_sub_tpb_window(BLOCK, num_warps):
    # BLOCK < tpb, where tpb = 32 * num_warps.
    #
    # tt.scan used to require BLOCK % tpb == 0, so a small cumsum failed to
    # legalize outright at the default num_warps=4 (tpb=128) — num_warps, a pure
    # perf knob, decided whether the kernel compiled. A sub-warp BLOCK (1..16)
    # was unreachable at ANY num_warps, since tpb is at least one warp.
    #
    # Sub-tpb tiles are now padded up to a full tpb window with 0.0, the identity
    # for the addf combine, so prefixes below BLOCK are unaffected and the
    # prefix-sum template is reused verbatim (no divergent barrier).
    torch.manual_seed(BLOCK * 31 + num_warps)
    inp = torch.randn(BLOCK, dtype=torch.float32, device="mps")
    out = torch.zeros(BLOCK, dtype=torch.float32, device="mps")
    _cumsum_store_kernel[(1,)](inp, out, BLOCK, BLOCK=BLOCK, num_warps=num_warps)
    torch.mps.synchronize()
    ref = torch.cumsum(inp.cpu().double(), dim=0)
    np.testing.assert_allclose(
        out.cpu().double().numpy(), ref.numpy(), rtol=0, atol=2e-5)
