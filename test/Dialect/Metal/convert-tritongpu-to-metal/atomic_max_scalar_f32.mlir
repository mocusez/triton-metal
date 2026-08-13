// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Triton expands scalar f32 `tl.atomic_max` into integer atomics over bitcast
// views of the value and pointer. Positive values use signed MAX; negative
// values use unsigned MIN so IEEE-754 bit ordering yields the float maximum.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_max_scalar_f32(%out_ptr: !tt.ptr<f32>, %v: f32) {
    %bits = tt.bitcast %v : f32 -> i32
    %int_ptr = tt.bitcast %out_ptr : !tt.ptr<f32> -> !tt.ptr<i32>
    %c31 = arith.constant 31 : i32
    %c0 = arith.constant 0 : i32
    %sign = arith.shrui %bits, %c31 : i32
    %neg = arith.cmpi ne, %sign, %c0 : i32
    %pos = arith.cmpi eq, %sign, %c0 : i32
    %old_pos = tt.atomic_rmw max, acq_rel, gpu, %int_ptr, %bits, %pos : (!tt.ptr<i32>, i32, i1) -> i32
    %old_neg = tt.atomic_rmw umin, acq_rel, gpu, %int_ptr, %bits, %neg : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}

// METAL-LABEL: metal.kernel atomic_max_scalar_f32
// METAL: metal.bitcast
// METAL: metal.atomic_rmw
// METAL: metal.atomic_rmw
// METAL-NOT: tt.atomic_rmw

// MSL: atomic_fetch_max_explicit((device atomic_int*)&
// MSL: atomic_fetch_min_explicit((device atomic_uint*)&
