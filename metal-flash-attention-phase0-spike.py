"""Phase 0 spike: hand-written simdgroup flash-attention on MPS.

Validates the Option-A design before building the compiler matcher:
  - two matmuls (Q@K^T, P@V) on simdgroup hardware, tiles read from threadgroup
  - S / P / O accumulator + online-softmax state live in threadgroup scalar domain
  - P (softmax output) is written to threadgroup then simdgroup_load'd for dot B
Compares against torch.scaled_dot_product_attention (per-head).

Tile constants hardcoded for the MHA solve() shapes: BM=32 (query block),
BN=32 (key block), d_head=16. Generalization is a later phase.
"""
import math
import torch

MSL = r"""
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

kernel void fa_kernel(
  device float *Q [[buffer(0)]],
  device float *K [[buffer(1)]],
  device float *V [[buffer(2)]],
  device float *O [[buffer(3)]],
  device uint32_t *pN [[buffer(4)]],
  device uint32_t *pDmodel [[buffer(5)]],
  device uint32_t *pH [[buffer(6)]],
  uint3 ltid [[thread_position_in_threadgroup]],
  uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint BM = 32u, BN = 32u, BD = 16u;   // query block, key block, d_head
  // LOCAL thread index within the threadgroup (NOT thread_position_in_grid,
  // which is global — the 2nd query block's warp has global x in [32,64)).
  uint lane = ltid.x & 31u;
  uint pid0 = tgid.x;         // query-row block
  uint pid1 = tgid.y;         // head
  uint N = pN[0];
  uint dmodel = pDmodel[0];
  uint h = pH[0];
  uint dhead = dmodel / h;    // == BD == 16 for this spike
  uint coloff = pid1 * dhead;
  uint rowoff = pid0 * BM;
  float scale = 1.0f / sqrt((float)dhead);

  threadgroup float qbuf[512];    // [32][16]  Q tile
  threadgroup float ktbuf[512];   // [16][32]  K^T tile (ktbuf[d][key]=K[key][d])
  threadgroup float vbuf[512];    // [32][16]  V tile
  threadgroup float sbuf[1024];   // [32][32]  S = Q@K^T (raw, unscaled)
  threadgroup float pbuf[1024];   // [32][32]  P = exp(shift)
  threadgroup float obuf[512];    // [32][16]  O accumulator
  threadgroup float otbuf[512];   // [32][16]  P@V temp
  threadgroup float run_max[32];
  threadgroup float run_sum[32];

  // The FA body runs on ONE warp (simdgroup 0). Under num_warps>1 the extra
  // warps sit idle but MUST still reach every threadgroup_barrier, so barriers
  // stay OUTSIDE the `active` guard while all compute is inside it.
  bool active = (ltid.x < 32u);

  // --- load Q tile + zero-init accumulator/state ---
  if (active) {
    for (uint c = lane; c < 512u; c += 32u) {
      uint q = c / BD; uint d = c % BD;
      uint row = rowoff + q;
      qbuf[c] = (row < N) ? Q[row*dmodel + coloff + d] : 0.0f;
      obuf[c] = 0.0f;
    }
    if (lane < BM) { run_max[lane] = -INFINITY; run_sum[lane] = 0.0f; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kb = 0; kb < N; kb += BN) {
    // --- stage K^T and V into threadgroup (masked: OOB keys -> 0) ---
    if (active) {
      for (uint c = lane; c < 512u; c += 32u) {
        uint d = c / BN; uint key = c % BN;   // ktbuf[d][key]
        uint kk = kb + key;
        ktbuf[c] = (kk < N) ? K[kk*dmodel + coloff + d] : 0.0f;
      }
      for (uint c = lane; c < 512u; c += 32u) {
        uint key = c / BD; uint d = c % BD;    // vbuf[key][d]
        uint kk = kb + key;
        vbuf[c] = (kk < N) ? V[kk*dmodel + coloff + d] : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- S = Q @ K^T   (simdgroup, 4x4 output tiles, 2 k-tiles) ---
    if (active) {
      for (uint mi = 0; mi < 4u; ++mi)
        for (uint ni = 0; ni < 4u; ++ni) {
          simdgroup_float8x8 acc(0.0f);
          for (uint ki = 0; ki < 2u; ++ki) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, &qbuf[(mi*8u)*BD + ki*8u], BD);
            simdgroup_load(b, &ktbuf[(ki*8u)*BN + ni*8u], BN);
            simdgroup_multiply_accumulate(acc, a, b, acc);
          }
          simdgroup_store(acc, &sbuf[(mi*8u)*BN + ni*8u], BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- online softmax, one query row per lane ---
    if (active) {
      uint q = lane;               // 0..31
      uint row = rowoff + q;
      if (row < N) {
        float m_cur = -INFINITY;
        for (uint kk = 0; kk < BN; ++kk) {
          if (kb + kk < N) m_cur = max(m_cur, sbuf[q*BN + kk] * scale);
        }
        float m_old = run_max[q];
        float m_new = max(m_old, m_cur);
        float scaler = exp(m_old - m_new);   // first iter: exp(-inf)=0
        float denom = 0.0f;
        for (uint kk = 0; kk < BN; ++kk) {
          float p = (kb + kk < N) ? exp(sbuf[q*BN + kk]*scale - m_new) : 0.0f;
          pbuf[q*BN + kk] = p;
          denom += p;
        }
        run_sum[q] = run_sum[q]*scaler + denom;
        run_max[q] = m_new;
        for (uint d = 0; d < BD; ++d) obuf[q*BD + d] *= scaler;
      } else {
        for (uint kk = 0; kk < BN; ++kk) pbuf[q*BN + kk] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- O_tile = P @ V   (simdgroup, 4x2 output tiles, 4 k-tiles) ---
    if (active) {
      for (uint mi = 0; mi < 4u; ++mi)
        for (uint di = 0; di < 2u; ++di) {
          simdgroup_float8x8 acc(0.0f);
          for (uint ki = 0; ki < 4u; ++ki) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, &pbuf[(mi*8u)*BN + ki*8u], BN);
            simdgroup_load(b, &vbuf[(ki*8u)*BD + di*8u], BD);
            simdgroup_multiply_accumulate(acc, a, b, acc);
          }
          simdgroup_store(acc, &otbuf[(mi*8u)*BD + di*8u], BD);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) { for (uint c = lane; c < 512u; c += 32u) obuf[c] += otbuf[c]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // --- epilogue: O = obuf / run_sum, masked store ---
  if (active) {
    uint q = lane;
    uint row = rowoff + q;
    if (row < N) {
      float denom = run_sum[q];
      float inv = (denom != 0.0f) ? (1.0f / denom) : 0.0f;
      for (uint d = 0; d < BD; ++d)
        O[row*dmodel + coloff + d] = obuf[q*BD + d] * inv;
    }
  }
}
"""


def cdiv(a, b):
    return -(-a // b)


_LIB = None


def run_fa(Q, K, V, N, d_model, h, num_warps=1):
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(MSL)
    O = torch.zeros(N, d_model, device="mps", dtype=torch.float32).contiguous()
    grid = (cdiv(N, 32), h, 1)
    tg = num_warps * 32
    threads = (grid[0]*tg, grid[1]*1, grid[2]*1)
    group_size = (tg, 1, 1)
    _LIB.fa_kernel(Q, K, V, O, N, d_model, h, threads=threads, group_size=group_size)
    torch.mps.synchronize()
    return O


def reference(Q, K, V, N, d_model, h):
    d_head = d_model // h
    ref = torch.zeros(N, d_model, dtype=torch.float32)
    Qc, Kc, Vc = Q.cpu(), K.cpu(), V.cpu()
    for hi in range(h):
        sl = slice(hi*d_head, (hi+1)*d_head)
        q, k, v = Qc[:, sl], Kc[:, sl], Vc[:, sl]
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
        ref[:, sl] = out
    return ref


def check(N, d_model, h, num_warps=1):
    torch.manual_seed(0xBEEF + N)
    Q = torch.randn(N, d_model, device="mps", dtype=torch.float32).contiguous()
    K = torch.randn(N, d_model, device="mps", dtype=torch.float32).contiguous()
    V = torch.randn(N, d_model, device="mps", dtype=torch.float32).contiguous()
    O = run_fa(Q, K, V, N, d_model, h, num_warps=num_warps)
    ref = reference(Q, K, V, N, d_model, h)
    err = (O.cpu() - ref).abs().max().item()
    ok = err < 1e-3
    print(f"[{'PASS' if ok else 'FAIL'}] N={N} d_model={d_model} h={h} "
          f"(d_head={d_model//h}) num_warps={num_warps} max_abs_err={err:.3e}")
    return ok


if __name__ == "__main__":
    assert torch.backends.mps.is_available(), "need MPS"
    results = []
    results.append(check(64, 64, 4))    # clean: N%32==0, d_head=16
    results.append(check(48, 64, 4))    # masked tail: N=48 -> blocks 32+16
    results.append(check(96, 64, 4))    # 3 blocks
    results.append(check(33, 64, 4))    # ragged
    # num_warps=4 (128 threads): extra warps idle, barriers still reached.
    # This is the config the real solve() uses (num_warps=4).
    results.append(check(64, 64, 4, num_warps=4))
    results.append(check(48, 64, 4, num_warps=4))
    print("\nALL PASS" if all(results) else "\nSOME FAILED")
