"""Phase 0 spike: hand-written scalar sink-attention body on MPS.

Validates the `metal.sink_attention` emitter body (metal-attention-with-sinks-plan.md
§3.2) before building the op + matcher:
  - one query row per lane, keys walked one at a time
  - online softmax updated PER KEY (accumulator rescaled per key, not per block),
    so no S/P staging buffers and no key blocking at all
  - the two phases the source kernel evaluates (sinks, then the local window
    blocks starting at `local_start`) are reproduced exactly, including their
    dependence on the host-side `N_LOCAL_BLOCKS` constexpr
  - `exp2` with the runtime f32 scale the kernel is handed, NOT `exp` with an
    emitter-computed `1/sqrt(d)`

Compares against a float64 CPU reference of the mask in §1 of the plan.
"""
import math

import torch

MSL_TEMPLATE = r"""
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

kernel void sink_attention_kernel(
  device float *q [[buffer(0)]],
  device float *k [[buffer(1)]],
  device float *v [[buffer(2)]],
  device float *out [[buffer(3)]],
  device uint32_t *pM [[buffer(4)]],
  device uint32_t *pDhead [[buffer(5)]],
  device float *pScale [[buffer(6)]],
  device uint32_t *pStrideQ [[buffer(7)]],
  device uint32_t *pStrideK [[buffer(8)]],
  device uint32_t *pStrideV [[buffer(9)]],
  device uint32_t *pStrideO [[buffer(10)]],
  device uint32_t *pSinks [[buffer(11)]],
  device uint32_t *pWindow [[buffer(12)]],
  uint3 ltid [[thread_position_in_threadgroup]],
  uint3 tgid [[threadgroup_position_in_grid]])
{{
  // ---- metal.sink_attention (per-key online softmax, one query row per lane) ----
  threadgroup float _sa_qbuf[{SZ_Q}];
  threadgroup float _sa_obuf[{SZ_Q}];
  threadgroup float _sa_rmax[{BM}];
  threadgroup float _sa_rsum[{BM}];
  {{
  uint _sa_lane = ltid.x & 31u;
  bool _sa_active = ltid.x < 32u;
  uint _sa_M = pM[0];
  uint _sa_dh = pDhead[0];
  float _sa_scale = pScale[0];
  uint _sa_sq = pStrideQ[0];
  uint _sa_sk = pStrideK[0];
  uint _sa_sv = pStrideV[0];
  uint _sa_so = pStrideO[0];
  int _sa_S = (int)pSinks[0];
  int _sa_W = (int)pWindow[0];
  uint _sa_rowoff = tgid.x * {BM}u;
  if (_sa_active) {{
    for (uint c = _sa_lane; c < {SZ_Q}u; c += 32u) {{
      uint qq = c / {BD}u; uint d = c % {BD}u; uint row = _sa_rowoff + qq;
      _sa_qbuf[c] = (row < _sa_M && d < _sa_dh) ? q[row * _sa_sq + d] : 0.0f;
      _sa_obuf[c] = 0.0f;
    }}
    if (_sa_lane < {BM}u) {{ _sa_rmax[_sa_lane] = -INFINITY; _sa_rsum[_sa_lane] = 0.0f; }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (_sa_active) {{
    uint _sa_q = _sa_lane; uint _sa_row = _sa_rowoff + _sa_q;
    if (_sa_q < {BM}u && _sa_row < _sa_M) {{
      int _sa_irow = (int)_sa_row;
      // phase 0: sink tokens, keys [0, bs)
      for (int _sa_key = 0; _sa_key < {BS}; ++_sa_key) {{
        if (!(_sa_key < _sa_S && _sa_key <= _sa_irow)) continue;
        STEP_BODY
      }}
      // phase 1: local window, keys [local_start, local_start + local_len)
      int _sa_lstart = max((int)_sa_rowoff - _sa_W + 1, _sa_S);
      for (int _sa_t = 0; _sa_t < {LOCAL_LEN}; ++_sa_t) {{
        int _sa_key = _sa_lstart + _sa_t;
        if (!(_sa_key < (int)_sa_M && _sa_key <= _sa_irow &&
              _sa_key >= _sa_irow - _sa_W + 1 && _sa_key >= _sa_S)) continue;
        STEP_BODY
      }}
      float _sa_denom = _sa_rsum[_sa_q];
      for (uint d = 0; d < _sa_dh; ++d)
        out[_sa_row * _sa_so + d] = _sa_obuf[_sa_q * {BD}u + d] / _sa_denom;
    }}
  }}
  }}
}}
"""

# One key's contribution: scalar inner product, then the online-softmax merge.
STEP_BODY = r"""{
          float _sa_a = 0.0f;
          for (uint d = 0; d < _sa_dh; ++d)
            _sa_a += _sa_qbuf[_sa_q * BDu + d] * k[(uint)_sa_key * _sa_sk + d];
          float _sa_s = _sa_a * _sa_scale;
          float _sa_mold = _sa_rmax[_sa_q];
          float _sa_mnew = max(_sa_mold, _sa_s);
          float _sa_sc = (_sa_mold == _sa_mnew) ? 1.0f : exp2(_sa_mold - _sa_mnew);
          float _sa_p = exp2(_sa_s - _sa_mnew);
          _sa_rsum[_sa_q] = _sa_rsum[_sa_q] * _sa_sc + _sa_p;
          _sa_rmax[_sa_q] = _sa_mnew;
          for (uint d = 0; d < _sa_dh; ++d)
            _sa_obuf[_sa_q * BDu + d] =
                _sa_obuf[_sa_q * BDu + d] * _sa_sc + _sa_p * v[(uint)_sa_key * _sa_sv + d];
        }"""


def cdiv(a, b):
    return -(-a // b)


_CACHE = {}


def build(BM, BD, BS, LOCAL_LEN):
    key = (BM, BD, BS, LOCAL_LEN)
    if key not in _CACHE:
        src = MSL_TEMPLATE.format(BM=BM, BD=BD, BS=BS, LOCAL_LEN=LOCAL_LEN,
                                  SZ_Q=BM * BD)
        src = src.replace("STEP_BODY", STEP_BODY.replace("BDu", f"{BD}u"))
        _CACHE[key] = torch.mps.compile_shader(src)
    return _CACHE[key]


def run(Q, K, V, M, d, num_sinks, window_size, num_warps=8):
    BM, BS, BN = 32, 16, 64
    BD = max(16, 1 << (d - 1).bit_length())
    LOCAL_LEN = cdiv(window_size + BM - 1, BN) * BN
    sm_scale = 1.4426950408889634 / (d ** 0.5)

    out = torch.zeros(M, d, device="mps", dtype=torch.float32).contiguous()
    lib = build(BM, BD, BS, LOCAL_LEN)
    grid_x = cdiv(M, BM)
    tg = num_warps * 32
    lib.sink_attention_kernel(
        Q, K, V, out,
        M, d, sm_scale,
        Q.stride(0), K.stride(0), V.stride(0), out.stride(0),
        num_sinks, window_size,
        threads=(grid_x * tg, 1, 1), group_size=(tg, 1, 1))
    torch.mps.synchronize()
    return out


def reference(Q, K, V, M, d, num_sinks, window_size):
    q = Q.cpu().double()
    k = K.cpu().double()
    v = V.cpu().double()
    scores = (q @ k.T) / math.sqrt(d)
    idx = torch.arange(M)
    i, j = idx[:, None], idx[None, :]
    keep = (j <= i) & ((j < num_sinks) | (j >= i - window_size + 1))
    scores = scores.masked_fill(~keep, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def check(M, d, num_sinks, window_size, num_warps=8):
    torch.manual_seed(0xBEEF + M * 31 + d)
    Q = torch.randn(M, d, device="mps", dtype=torch.float32).contiguous()
    K = torch.randn(M, d, device="mps", dtype=torch.float32).contiguous()
    V = torch.randn(M, d, device="mps", dtype=torch.float32).contiguous()
    out = run(Q, K, V, M, d, num_sinks, window_size, num_warps=num_warps)
    ref = reference(Q, K, V, M, d, num_sinks, window_size)
    err = (out.cpu().double() - ref).abs().max().item()
    ok = err <= 1e-6
    print(f"[{'PASS' if ok else 'FAIL'}] M={M:4d} d={d:3d} S={num_sinks:3d} "
          f"W={window_size:4d} warps={num_warps}  max_abs_err={err:.3e}")
    return ok


if __name__ == "__main__":
    assert torch.backends.mps.is_available(), "need MPS"
    res = []
    for M in (33, 64, 100, 256):
        for d in (16, 32, 64):
            for S in (0, 1, 4, 16):
                for W in (16, 32, 128):
                    res.append(check(M, d, S, W))
    res.append(check(64, 16, 4, 32, num_warps=1))
    res.append(check(64, 16, 4, 32, num_warps=4))
    print(f"\n{sum(res)}/{len(res)} " + ("ALL PASS" if all(res) else "SOME FAILED"))
