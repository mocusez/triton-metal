// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Phase 1: metal.flash_attention emits the online-softmax flash-attention body
// validated in metal-flash-attention-phase0-spike.py (bit-close to torch SDPA).
// Both matmuls run on simdgroup hardware (Q/K^T/V/P tiles staged in threadgroup,
// read via simdgroup_load); the online softmax + O accumulator + running max/sum
// live in the threadgroup scalar domain. The body runs on ONE warp (guard
// `_fa_active = ltid.x < 32`); barriers stay OUTSIDE the guard so idle warps
// (num_warps>1) still reach them. Block sizes bm=bn=32, bd=16 (d_head).
//
// This is the HEAD-SPLIT, NO-WINDOW configuration: `heads` present (so
// d_head = d_model/h and the column offset is tgid.y * d_head), `window`
// absent (so the band predicate degenerates to `true` and is folded away).
// The sliding-window counterpart is sliding_window_attention_emit.mlir.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel fa_kernel address_space_device [true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dm: !metal.memref<? x ui32>, %h: !metal.memref<? x ui32>):
      metal.flash_attention %q, %k, %v, %o, %m, %n, %dm heads %h {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>
      metal.return
    }
    metal.module_end
  }
}

// The FA op forces both the program-id (tgid) and LOCAL thread-index (ltid)
// builtins into the kernel signature.
// CHECK: kernel void fa_kernel(
// CHECK: device float *v0 {{\[\[}}buffer(0)]]
// CHECK: device uint32_t *v7 {{\[\[}}buffer(7)]]
// CHECK: uint3 tgid {{\[\[}}threadgroup_position_in_grid]]
// CHECK: uint3 ltid {{\[\[}}thread_position_in_threadgroup]]

// Threadgroup-resident working set (BM*BD=512, BM*BN=1024, per-row state=32).
// CHECK: threadgroup float _fa_qbuf[512];
// CHECK: threadgroup float _fa_sbuf[1024];
// CHECK: threadgroup float _fa_pbuf[1024];
// CHECK: threadgroup float _fa_obuf[512];
// CHECK: threadgroup float _fa_rmax[32];
// CHECK: threadgroup float _fa_rsum[32];

// Single-warp guard keyed off the LOCAL thread index; runtime scale/head slice.
// CHECK: bool _fa_active = ltid.x < 32u;
// CHECK: uint _fa_M = v4[0];
// CHECK: uint _fa_N = v5[0];
// CHECK: uint _fa_dhead = v6[0] / v7[0];
// CHECK: uint _fa_coloff = tgid.y * _fa_dhead;
// CHECK: float _fa_scale = 1.0f / sqrt((float)_fa_dhead);

// Masked K-tile stage (out-of-range keys AND padded columns d>=d_head read 0).
// CHECK: _fa_ktbuf[c] = (kk < _fa_N && d < _fa_dhead) ? v1[kk * _fa_dm + _fa_coloff + d] : 0.0f;

// Dot A: S = Q @ K^T on simdgroup hardware, tiles read from threadgroup.
// CHECK: simdgroup_load(a, &_fa_qbuf[
// CHECK: simdgroup_load(b, &_fa_ktbuf[
// CHECK: simdgroup_multiply_accumulate(acc, a, b, acc);
// CHECK: simdgroup_store(acc, &_fa_sbuf[

// Online softmax in the scalar domain: running max, rescale, exp, running sum.
// CHECK: float m_new = max(m_old, m_cur);
// The (m_old == m_new) arm is what keeps a fully-masked block from turning
// exp(-inf - -inf) into a NaN; unreachable without a window, emitted anyway.
// CHECK: float scaler = (m_old == m_new) ? 1.0f : exp(m_old - m_new);
// CHECK: float p = (kb + kk < _fa_N && (true)) ? exp(_fa_sbuf[q*32u + kk]*_fa_scale - m_new) : 0.0f;
// CHECK: _fa_rsum[q] = _fa_rsum[q]*scaler + denom;
// CHECK: _fa_obuf[q*16u + d] *= scaler;

// Dot B: O_tile = P @ V (computed P read back from threadgroup) + accumulate.
// CHECK: simdgroup_load(a, &_fa_pbuf[
// CHECK: simdgroup_load(b, &_fa_vbuf[
// CHECK: _fa_obuf[c] += _fa_otbuf[c];

// Epilogue: normalize by running sum, masked store to O (v3).
// CHECK: float inv = (denom != 0.0f) ? (1.0f / denom) : 0.0f;
// CHECK: v3[row * _fa_dm + _fa_coloff + d] = _fa_obuf[q*16u + d] * inv;
//
// No window operand -> no band predicate anywhere.
// CHECK-NOT: _fa_win
