// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// `tt.expand_dims` adds a size-1 axis to a tensor. Under our
// tensor->scalar TypeConverter, each thread holds 1 element pre and
// post the shape change, so the lowering is identity passthrough
// (matches SplatLowering's shape). See
// the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @expand_dims_kernel(%x_ptr: !tt.ptr<f32>) {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r2d = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    // Force a use so DCE doesn't erase the expand_dims chain entirely.
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %r2d : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<8x1x!tt.ptr<f32>, #blocked>
    tt.store %x_addr, %x_val : tensor<8x1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel expand_dims_kernel
// CHECK: metal.return
