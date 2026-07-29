// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Phase C: metal.flash_attention with the two OPTIONAL operands exercised in
// their "sliding window, no head split" configuration — `window` present,
// `heads` absent. The head-split / no-window configuration is
// flash_attention_emit.mlir; between them the two files cover all four
// combinations of the optional operands that the matcher can produce.
//
// Buffers: v0..v3 = q/k/v/out, v4 = m (query rows), v5 = n (keys),
// v6 = d_model (row stride), v7 = window. Block sizes bm=bn=bd=16.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel swa_kernel address_space_device [true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dm: !metal.memref<? x ui32>, %w: !metal.memref<? x ui32>):
      metal.flash_attention %q, %k, %v, %o, %m, %n, %dm window %w {bm = 16 : i64, bn = 16 : i64, bd = 16 : i64} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>
      metal.return
    }
    // Same op with the band width as a FOLDED CONSTANT instead of a buffer:
    // Triton removes a kernel argument equal to 1 from the signature, so a
    // tridiagonal band (window_size = 1) has nothing to point at and travels
    // as the `window_const` attribute. Note there is no v7 in this kernel.
    metal.kernel swa_const_kernel address_space_device [true, true, true, true, true, true, true] {
    ^bb0(%q2: !metal.memref<? x f32>, %k2: !metal.memref<? x f32>, %v2: !metal.memref<? x f32>, %o2: !metal.memref<? x f32>, %m2: !metal.memref<? x ui32>, %n2: !metal.memref<? x ui32>, %dm2: !metal.memref<? x ui32>):
      metal.flash_attention %q2, %k2, %v2, %o2, %m2, %n2, %dm2 {bm = 16 : i64, bn = 16 : i64, bd = 16 : i64, window_const = 1 : i64} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>
      metal.return
    }
    metal.module_end
  }
}

// CHECK: kernel void swa_kernel(

// No head split: d_head is the whole row and the column offset is a literal 0.
// (Passing "h = 1" instead would need a kernel buffer holding the constant 1,
// which does not exist — that is why `h` is optional. See MetalOps.td.)
// CHECK: uint _fa_M = v4[0];
// CHECK: uint _fa_N = v5[0];
// CHECK: uint _fa_dm = v6[0];
// CHECK: uint _fa_dhead = _fa_dm;
// CHECK: uint _fa_coloff = 0u;
// CHECK-NOT: tgid.y * _fa_dhead

// The band half-width is read SIGNED. It arrives in a `device uint32_t*`
// buffer, and comparing the unsigned difference of two uints would turn any
// negative row-key offset into a huge positive one, i.e. silently widen the
// band to everything.
// CHECK: int _fa_win = (int)v7[0];

// Row bound is M (the query count), key bound is N — separate operands.
// CHECK: _fa_qbuf[c] = (row < _fa_M && d < _fa_dhead) ? v0[row * _fa_dm + _fa_coloff + d] : 0.0f;

// Lane guard is on `q < bm` FIRST: bm is 16 here, so lanes 16..31 would
// otherwise write past the end of _fa_rmax/_fa_rsum/_fa_pbuf/_fa_obuf.
// CHECK: if (q < 16u && row < _fa_M) {

// Band predicate on BOTH softmax passes: the running max...
// CHECK: if (kb + kk < _fa_N && (abs((int)row - (int)(kb + kk)) <= _fa_win)) m_cur = max(m_cur,

// With a window, a whole key block can fall outside the band, leaving
// m_cur == -inf; if m_old is -inf too (every earlier block was outside as
// well, e.g. row 50 with window 3 at kb = 0) then exp(-inf - -inf) is
// exp(NaN) = NaN and it poisons the row's running sum and accumulator for
// good. Nothing needs rescaling in that case, so the factor is 1.
// CHECK: float scaler = (m_old == m_new) ? 1.0f : exp(m_old - m_new);

// ...and the numerator. Zeroing the numerator matters beyond the softmax
// itself — _fa_pbuf feeds the second simdgroup matmul, so an out-of-band entry
// left nonzero would leak into the output through P@V.
// CHECK: float p = (kb + kk < _fa_N && (abs((int)row - (int)(kb + kk)) <= _fa_win)) ? exp(

// The pbuf-zeroing arm for lanes with no row is lane-guarded too.
// CHECK: } else if (q < 16u) {

// Epilogue: normalize by the running sum, store the M rows.
// CHECK: if (q < 16u && row < _fa_M) {
// CHECK: v3[row * _fa_dm + _fa_coloff + d] = _fa_obuf[q*16u + d] * inv;

// The folded-constant band: a literal, and the band predicate is still emitted
// on both softmax passes.
// CHECK: kernel void swa_const_kernel(
// CHECK: int _fa_win = 1;
// CHECK: if (kb + kk < _fa_N && (abs((int)row - (int)(kb + kk)) <= _fa_win)) m_cur = max(m_cur,
// CHECK: float p = (kb + kk < _fa_N && (abs((int)row - (int)(kb + kk)) <= _fa_win)) ? exp(
