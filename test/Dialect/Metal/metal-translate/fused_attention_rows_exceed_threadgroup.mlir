// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// Query rows outnumbering the threadgroup.
//
// `getFusedAttentionSchedule` sets `threads_per_group` to
// `min(source num_warps, bm/8) * 32` -- it CAPS the launch at the row-tile
// count so no SIMD-group is started without a QK/PV tile to run. Nothing
// raises it, so a source that asks for ONE warp at bm = 64 gets 32 threads to
// cover 64 query rows.
//
// The score transform, the online-softmax update and the epilogue store are
// each one query row per cooperative thread. They used to read `q = _fa_tid`
// under a `q < bm` BOUND, which is not a loop: rows at and above the
// threadgroup width were never transformed, never normalised and never stored,
// and the kernel answered silently wrong (measured 9.2e-01 against
// `scaled_dot_product_attention` on `hard-mult_head_attention.py` at
// bm = 64, num_warps = 1). Every OTHER cooperative pass in these emitters --
// Q staging, K^T/V staging, the O accumulate -- already strides by
// `threads_per_group`; only the per-row ones did not.
//
// Both MMA bodies are pinned here because both carry the three passes.
// `rows_full_kernel` fits the 8192-float budget; `rows_chunked_kernel` does
// not and takes the chunked body.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel rows_full_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 64 : i64, bn = 16 : i64, bd = 16 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        metal.score_yield %scaled : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
    metal.kernel rows_chunked_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 64 : i64, bn = 64 : i64, bd = 64 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        metal.score_yield %scaled : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
  }
}

// One warp at bm = 64: 32 threads, 64 query rows.
// CHECK-LABEL: kernel void rows_full_kernel
// CHECK: bool _fa_active = _fa_tid < 32u
// The running-state init, ...
// CHECK: for (uint q = _fa_tid; q < 64u; q += 32u) { _fa_rmax[q] = -INFINITY; _fa_rsum[q] = 0.0f; }
// ... the score transform + online softmax, ...
// CHECK: for (uint q = _fa_tid; q < 64u; q += 32u) {
// CHECK-NEXT: uint row = _fa_rowoff + q;
// CHECK: _fa_rsum[q] = _fa_rsum[q]*scaler + denom;
// ... and the epilogue store.
// CHECK: for (uint q = _fa_tid; q < 64u; q += 32u) {
// CHECK-NEXT: uint row = _fa_rowoff + q;

// CHECK-LABEL: kernel void rows_chunked_kernel
// CHECK: bool _fa_active = _fa_tid < 32u
// CHECK: for (uint q = _fa_tid; q < 64u; q += 32u) { _fa_rmax[q] = -INFINITY; _fa_rsum[q] = 0.0f; }
// CHECK: for (uint q = _fa_tid; q < 64u; q += 32u) {
// CHECK-NEXT: uint row = _fa_rowoff + q;
// CHECK: _fa_rsum[q] = _fa_rsum[q]*scaler + denom;
// CHECK: for (uint q = _fa_tid; q < 64u; q += 32u) {
// CHECK-NEXT: uint row = _fa_rowoff + q;

// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s --check-prefix=NOBOUND
// The bound-only form is what was wrong. It must not come back anywhere, in
// either body -- including for the shapes where bm happens to fit.
// NOBOUND-NOT: uint q = _fa_tid; uint row =
