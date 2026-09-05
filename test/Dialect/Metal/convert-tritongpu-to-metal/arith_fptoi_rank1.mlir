// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// P0b regression coverage: rank-1 blocked tensor f32 -> i32/u32 casts must
// lower to scalar Metal casts per lane instead of leaving tensor arith casts
// behind or crashing during failed conversion cleanup.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fptosi_rank1_f32_i32(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<i32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %cast = arith.fptosi %iv : tensor<256xf32, #blocked> to tensor<256xi32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<i32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %cast : tensor<256x!tt.ptr<i32>, #blocked>
    tt.return
  }

  tt.func public @fptoui_rank1_f32_u32(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<i32>) {
    %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %is = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ia = tt.addptr %is, %offs : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %iv = tt.load %ia : tensor<256x!tt.ptr<f32>, #blocked>
    %cast = arith.fptoui %iv : tensor<256xf32, #blocked> to tensor<256xi32, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked>
    %oa = tt.addptr %os, %offs : tensor<256x!tt.ptr<i32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %oa, %cast : tensor<256x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel fptosi_rank1_f32_i32
// CHECK-NOT:   arith.fptosi {{.*}} tensor
// CHECK:       arith.fptosi {{.*}} : f32 to i32
// CHECK:       metal.store

// CHECK-LABEL: metal.kernel fptoui_rank1_f32_u32
// CHECK-NOT:   arith.fptoui {{.*}} tensor
// CHECK:       metal.cast {{.*}} : (f32) -> ui32
// CHECK:       metal.store

// MSL-LABEL: kernel void fptosi_rank1_f32_i32
// MSL:       (int32_t)(

// FPToUI must stay unsigned through MSL emission; `(int32_t)` would corrupt
// the upper half of the u32 range before the bit-preserving store.
// MSL-LABEL: kernel void fptoui_rank1_f32_u32
// MSL:       uint32_t(
