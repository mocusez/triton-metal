// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// `tt.reshape` changes the logical tensor shape but not the per-thread
// scalar value under our tensor->scalar TypeConverter. Same identity
// passthrough as expand_dims / broadcast. See
// `.omc/specs/deep-interview-metal-matmul-session1-reshape-trans-simd-scaffold.md`.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reshape_kernel(%x_ptr: !tt.ptr<f32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %r2d = tt.reshape %r allow_reorder : tensor<128xi32, #blocked> -> tensor<8x16xi32, #blocked2>
    // Force a use chain so DCE doesn't erase the reshape.
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked2>
    %x_addr = tt.addptr %x_splat, %r2d : tensor<8x16x!tt.ptr<f32>, #blocked2>, tensor<8x16xi32, #blocked2>
    %x_val = tt.load %x_addr : tensor<8x16x!tt.ptr<f32>, #blocked2>
    tt.store %x_addr, %x_val : tensor<8x16x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel reshape_kernel
// CHECK: metal.return
