// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// `tt.broadcast` replicates a size-1 axis to a larger size. Under our
// tensor->scalar TypeConverter, each thread holds 1 element regardless,
// so the lowering is identity passthrough. See
// the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @broadcast_kernel(%x_ptr: !tt.ptr<f32>) {
    // Build the 8x1 input via the canonical Triton path (make_range +
    // expand_dims), then broadcast it to 8x16.
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r2d = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %r_bcast = tt.broadcast %r2d : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    // Force a use chain so the broadcast survives DCE.
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %r_bcast : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<8x16x!tt.ptr<f32>, #blocked>
    tt.store %x_addr, %x_val : tensor<8x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel broadcast_kernel
// CHECK: metal.return
