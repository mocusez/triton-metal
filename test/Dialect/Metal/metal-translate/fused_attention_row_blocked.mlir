// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s

// The O accumulator alone does not fit, so the program walks its rows.
//
// The chunked MMA body holds a `bm x bd` output accumulator across the WHOLE
// key sweep. At `bm * bd >= 8192` that is Apple's entire 32 KiB threadgroup
// budget, so no blocking of (key, d_head, out_d) can fit alongside it and the
// op used to fall to the scalar correctness floor -- silently, and measured
// 108x slower at bm=64/d_head=128 (the shape a Llama block is written in) and
// 243x at bm=64/d_head=256.
//
// The schedule now searches the ROW BLOCK as the outer axis: a program carries
// `blockRows` of its `bm` rows at a time, each pass a complete key sweep.
// Chunking rows rather than output features recomputes nothing -- a row block
// needs its own scores anyway -- at the cost of one more read of K and V per
// program, which is the traffic the smaller `bm` would have had.
//
// ⚠️ The pass's epilogue READS the O buffer and the running state and the next
// pass ZEROES them, so the back edge carries a write-after-read hazard across
// the whole threadgroup. Without the barrier below, 72 of the row-blocked cells
// of `sweeps/s_attn_rows.py` answer wrong. It is pinned here because it is
// invisible in the shape of the code and only a race shows it.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    // bm * bd = 64 * 128 = 8192 == the whole budget.
    metal.kernel row_blocked_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 64 : i64, bn = 32 : i64, bd = 128 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
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
    // bm * bd = 64 * 64 = 4096: one pass, and byte-identical to what this body
    // emitted before row blocking existed.
    metal.kernel unblocked_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 64 : i64, bn = 32 : i64, bd = 64 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
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

// CHECK-LABEL: kernel void row_blocked_kernel
// Half the rows at a time, so the accumulator is half the budget, not all of it.
// CHECK: threadgroup float _fa_obuf[4096]
// CHECK: threadgroup float _fa_rmax[32]
// CHECK: for (uint _fa_rb = 0u; _fa_rb < 64u; _fa_rb += 32u) {
// CHECK-NEXT: uint _fa_rowoff = tgid.x * 64u + _fa_rb;
// The running state is re-initialised per pass, ...
// CHECK: for (uint q = _fa_tid; q < 32u; q += 256u) { _fa_rmax[q] = -INFINITY; _fa_rsum[q] = 0.0f; }
// ... both matmuls still run on the matrix unit -- this is the whole point, the
// shape used to take the scalar floor ...
// CHECK: simdgroup_multiply_accumulate
// ... and the epilogue store is INSIDE the pass.
// CHECK: _fa_obuf[q*128u + d] / denom;
// ⚠️ The back edge: the next pass zeroes what this one just read.
// CHECK: threadgroup_barrier(mem_flags::mem_threadgroup);
// CHECK-NEXT: }

// CHECK-LABEL: kernel void unblocked_kernel
// CHECK: threadgroup float _fa_obuf[4096]
// CHECK: uint _fa_rowoff = tgid.x * 64u;
// CHECK: simdgroup_multiply_accumulate

// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s --check-prefix=ONEPASS
// ONEPASS: kernel void unblocked_kernel
// ONEPASS-NOT: _fa_rb
// ONEPASS: return;
