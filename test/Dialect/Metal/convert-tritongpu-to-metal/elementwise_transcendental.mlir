// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Session L4 positive fixture: fp32 `math.sqrt` and `math.erf` lower through
// `convert-tritongpu-to-metal` to `metal.unary_exp` with the correct
// `UnaryExpOperator` enum case (`sqrtOp` / `erfOp`). After conversion the
// `math.*` tensor form must be gone and a `metal.kernel` must be present.
// See `.omc/specs/deep-interview-leet-triton-l4-transcendentals.md`.

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
