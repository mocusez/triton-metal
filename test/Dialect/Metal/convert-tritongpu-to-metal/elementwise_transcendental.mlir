// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_fma_scalar(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %x = tt.load %a : !tt.ptr<f32>
    %y = tt.load %b : !tt.ptr<f32>
    %z = tt.load %c : !tt.ptr<f32>
    %r = math.fma %x, %y, %z : f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_fma_scalar
// CHECK: metal.fma
// CHECK: metal.store

// -----
//
// Session L4 positive fixture: fp32 `math.sqrt` and `math.erf` lower through
// `convert-tritongpu-to-metal` to `metal.unary_exp` with the correct
// `UnaryExpOperator` enum case (`sqrtOp` / `erfOp`). After conversion the
// `math.*` tensor form must be gone and a `metal.kernel` must be present.
// See the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_sqrt_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.sqrt %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_sqrt_kernel
// CHECK-NOT:   math.sqrt {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} sqrtOp

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_erf_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.erf %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_erf_kernel
// CHECK-NOT:   math.erf {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} erfOp

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_exp_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.exp %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_exp_kernel
// CHECK-NOT:   math.exp {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} expOp

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_log_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.log %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_log_kernel
// CHECK-NOT:   math.log {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} logOp

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_rsqrt_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.rsqrt %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_rsqrt_kernel
// CHECK-NOT:   math.rsqrt {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} rsqrtOp

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_exp2_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.exp2 %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_exp2_kernel
// CHECK-NOT:   math.exp2 {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} exp2Op

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_log2_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.log2 %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_log2_kernel
// CHECK-NOT:   math.log2 {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} log2Op

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_absf_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.absf %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_absf_kernel
// CHECK-NOT:   math.absf {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} absOp

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @math_fma_kernel(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %as = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %aa = tt.addptr %as, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %av = tt.load %aa : tensor<256x!tt.ptr<f32>, #blocked>
    %bs = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ba = tt.addptr %bs, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %bv = tt.load %ba : tensor<256x!tt.ptr<f32>, #blocked>
    %cs = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ca = tt.addptr %cs, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %cv = tt.load %ca : tensor<256x!tt.ptr<f32>, #blocked>
    %r = math.fma %av, %bv, %cv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel math_fma_kernel
// CHECK-NOT:   math.fma {{.*}} tensor
// CHECK:       metal.fma

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @precise_sqrt_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %r = tt.precise_sqrt %iv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel precise_sqrt_kernel
// CHECK-NOT:   tt.precise_sqrt {{.*}} tensor
// CHECK:       metal.unary_exp {{.*}} sqrtOp

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @precise_divf_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %xs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xs, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %xv = tt.load %xa : tensor<256x!tt.ptr<f32>, #blocked>
    %ys = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ya = tt.addptr %ys, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %yv = tt.load %ya : tensor<256x!tt.ptr<f32>, #blocked>
    %r = tt.precise_divf %xv, %yv : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel precise_divf_kernel
// CHECK-NOT:   tt.precise_divf {{.*}} tensor
// CHECK:       metal.binary_exp {{.*}}, {{.*}}, divOp

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clampf_none_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %lo = arith.constant dense<-1.000000e+00> : tensor<256xf32, #blocked>
    %hi = arith.constant dense<1.000000e+00> : tensor<256xf32, #blocked>
    %r = tt.clampf %iv, %lo, %hi, propagateNan = none : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel clampf_none_kernel
// CHECK-NOT:   tt.clampf
// CHECK:       metal.clampf {{.*}}, propagate_nan = false

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clampf_all_kernel(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %lo = arith.constant dense<-1.000000e+00> : tensor<256xf32, #blocked>
    %hi = arith.constant dense<1.000000e+00> : tensor<256xf32, #blocked>
    %r = tt.clampf %iv, %lo, %hi, propagateNan = all : tensor<256xf32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %r : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel clampf_all_kernel
// CHECK-NOT:   tt.clampf
// CHECK:       metal.clampf {{.*}}, propagate_nan = true

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @abs_f16_scalar(%input: !tt.ptr<f16>, %output: !tt.ptr<f16>) {
    %value = tt.load %input : !tt.ptr<f16>
    %result = math.absf %value : f16
    tt.store %output, %result : !tt.ptr<f16>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel abs_f16_scalar
// CHECK: %[[BITS:.*]] = metal.bitcast {{.*}} : (f16) -> ui16
// CHECK: %[[MASK:.*]] = metal.constant 32767 : ui16
// CHECK: %[[MAG:.*]] = metal.binary_exp %[[BITS]], %[[MASK]], bitAndOp : (ui16, ui16) -> ui16
// CHECK: metal.bitcast %[[MAG]] : (ui16) -> f16
// CHECK: metal.store

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @abs_f16_tensor(%input: !tt.ptr<f16>, %output: !tt.ptr<f16>) {
    %offset = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %input : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offset : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    %value = tt.load %ptr : tensor<256x!tt.ptr<f16>, #blocked>
    %result = math.absf %value : tensor<256xf16, #blocked>
    %out_base = tt.splat %output : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %out_ptr = tt.addptr %out_base, %offset : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %out_ptr, %result : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel abs_f16_tensor
// CHECK: %[[BITS:.*]] = metal.bitcast {{.*}} : (f16) -> ui16
// CHECK: %[[MASK:.*]] = metal.constant 32767 : ui16
// CHECK: %[[MAG:.*]] = metal.binary_exp %[[BITS]], %[[MASK]], bitAndOp : (ui16, ui16) -> ui16
// CHECK: metal.bitcast %[[MAG]] : (ui16) -> f16
// CHECK: metal.store

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @abs_bf16_scalar(%input: !tt.ptr<bf16>, %output: !tt.ptr<bf16>) {
    %value = tt.load %input : !tt.ptr<bf16>
    %result = math.absf %value : bf16
    tt.store %output, %result : !tt.ptr<bf16>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel abs_bf16_scalar
// CHECK: %[[BITS:.*]] = metal.bitcast {{.*}} : (bf16) -> ui16
// CHECK: %[[MASK:.*]] = metal.constant 32767 : ui16
// CHECK: %[[MAG:.*]] = metal.binary_exp %[[BITS]], %[[MASK]], bitAndOp : (ui16, ui16) -> ui16
// CHECK: metal.bitcast %[[MAG]] : (ui16) -> bf16
// CHECK: metal.store

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @abs_bf16_tensor(%input: !tt.ptr<bf16>, %output: !tt.ptr<bf16>) {
    %offset = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %input : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %ptr = tt.addptr %base, %offset : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    %value = tt.load %ptr : tensor<256x!tt.ptr<bf16>, #blocked>
    %result = math.absf %value : tensor<256xbf16, #blocked>
    %out_base = tt.splat %output : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %out_ptr = tt.addptr %out_base, %offset : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %out_ptr, %result : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel abs_bf16_tensor
// CHECK: %[[BITS:.*]] = metal.bitcast {{.*}} : (bf16) -> ui16
// CHECK: %[[MASK:.*]] = metal.constant 32767 : ui16
// CHECK: %[[MAG:.*]] = metal.binary_exp %[[BITS]], %[[MASK]], bitAndOp : (ui16, ui16) -> ui16
// CHECK: metal.bitcast %[[MAG]] : (ui16) -> bf16
// CHECK: metal.store
