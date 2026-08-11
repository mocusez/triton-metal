// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// metal.sink_attention: causal attention with sink tokens and a one-sided
// sliding window (leet-triton/medium-attention_with_sinks.py). Two kernels
// here, covering both spellings of the optional counts:
//   sink_attn_kernel       — num_sinks and window as runtime buffers
//   sink_attn_const_kernel — both folded to constants (Triton drops a kernel
//                            argument equal to 1 from the signature)
//
// Buffers: v0..v3 = q/k/v/out, v4 = m (rows == keys), v5 = d_head,
// v6 = scale (f32), v7..v10 = the four row strides, v11 = num_sinks,
// v12 = window. bm=32 query rows, bd=16 feature tile, bs=16 sink keys,
// local_len=64 window keys.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel sink_attn_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sc: !metal.memref<? x f32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>, %ns: !metal.memref<? x ui32>, %w: !metal.memref<? x ui32>):
      metal.sink_attention %q, %k, %v, %o, %m, %dh, %sc strides %sq, %sk, %sv, %so sinks %ns window %w {bm = 32 : i64, bd = 16 : i64, bs = 16 : i64, local_len = 64 : i64} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>
      metal.return
    }
    metal.kernel sink_attn_const_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q2: !metal.memref<? x f32>, %k2: !metal.memref<? x f32>, %v2: !metal.memref<? x f32>, %o2: !metal.memref<? x f32>, %m2: !metal.memref<? x ui32>, %dh2: !metal.memref<? x ui32>, %sc2: !metal.memref<? x f32>, %sq2: !metal.memref<? x ui32>, %sk2: !metal.memref<? x ui32>, %sv2: !metal.memref<? x ui32>, %so2: !metal.memref<? x ui32>):
      metal.sink_attention %q2, %k2, %v2, %o2, %m2, %dh2, %sc2 strides %sq2, %sk2, %sv2, %so2 {bm = 32 : i64, bd = 16 : i64, bs = 16 : i64, local_len = 64 : i64, sinks_const = 1 : i64, window_const = 1 : i64} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void sink_attn_kernel(

// Working set is qbuf + obuf + the two per-row scalars: 2*32*16 + 2*32 floats.
// CHECK: threadgroup float _sa_qbuf[512];
// CHECK: threadgroup float _sa_obuf[512];
// CHECK: threadgroup float _sa_rmax[32];
// CHECK: threadgroup float _sa_rsum[32];

// One warp does the work; the idle warps still reach the barrier below, which
// is why the barrier sits OUTSIDE this guard.
// CHECK: bool _sa_active = ltid.x < 32u;

// Every scalar comes from its OWN buffer: this kernel passes four separate row
// strides and a feature width distinct from all of them, which is why the op
// cannot reuse metal.flash_attention's single `d_model`.
// CHECK: uint _sa_M = v4[0];
// CHECK: uint _sa_dh = v5[0];
// CHECK: float _sa_scale = v6[0];
// CHECK: uint _sa_sq = v7[0];
// CHECK: uint _sa_sk = v8[0];
// CHECK: uint _sa_sv = v9[0];
// CHECK: uint _sa_so = v10[0];

// The counts are read SIGNED: `row - W + 1` and `rowoff - W + 1` go negative,
// and an unsigned compare would silently turn that into a huge positive bound.
// CHECK: int _sa_S = (int)v11[0];
// CHECK: int _sa_W = (int)v12[0];

// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup);

// Guard on `q < bm` BEFORE `row < M`: every per-row buffer is sized by bm, and
// `row < M` does not imply `q < bm` when bm < 32.
// CHECK: if (_sa_q < 32u && _sa_row < _sa_M) {

// Phase 0, the sink tokens: keys [0, bs), predicate `s < S && s <= row`.
// CHECK: for (int _sa_key = 0; _sa_key < 16; ++_sa_key) {
// CHECK: if (!(_sa_key < _sa_S && _sa_key <= _sa_irow)) continue;

// Phase 1, the window: keys [local_start, local_start + local_len). The origin
// is the query BLOCK's, not the row's — that is what the source kernel computes.
// CHECK: int _sa_lstart = max((int)_sa_rowoff - _sa_W + 1, _sa_S);
// CHECK: for (int _sa_t = 0; _sa_t < 64; ++_sa_t) {
// CHECK: int _sa_key = _sa_lstart + _sa_t;
// CHECK: if (!(_sa_key < (int)_sa_M && _sa_key <= _sa_irow &&
// CHECK: _sa_key >= _sa_irow - _sa_W + 1 && _sa_key >= _sa_S)) continue;

// exp2, not exp: the scale the host passes already folds in log2(e).
// CHECK: float _sa_sc = (_sa_mold == _sa_mnew) ? 1.0f : exp2(_sa_mold - _sa_mnew);
// CHECK: float _sa_p = exp2(_sa_s - _sa_mnew);

// Plain divide, no `denom != 0` test: a row with no visible key produces NaN,
// which is exactly what the source kernel's `acc / l_i` produces.
// CHECK: float _sa_denom = _sa_rsum[_sa_q];
// CHECK: v3[_sa_row * _sa_so + d] = _sa_obuf[_sa_q * 16u + d] / _sa_denom;

// CHECK: kernel void sink_attn_const_kernel(
// Folded counts become literals and claim no buffer — there is no v11/v12 here.
// CHECK: int _sa_S = 1;
// CHECK: int _sa_W = 1;
