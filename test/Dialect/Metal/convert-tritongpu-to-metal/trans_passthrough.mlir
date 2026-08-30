// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// `tt.trans` permutes tensor axes. Under our tensor->scalar TypeConverter
// each thread holds one scalar, so the value is preserved; the downstream
// IR uses the permuted shape's indices to re-derive addresses. Identity
// passthrough. See
// the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @trans_kernel(%x_ptr: !tt.ptr<f32>) {
    // Build a 16x32 tensor via make_range + expand_dims + broadcast.
    %rm = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rm2d = tt.expand_dims %rm {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %rm_b = tt.broadcast %rm2d : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    // Transpose 16x32 -> 32x16.
    %rt = tt.trans %rm_b {order = array<i32: 1, 0>} : tensor<16x32xi32, #blocked> -> tensor<32x16xi32, #blocked_t>
    // Force a use chain so DCE doesn't erase the trans.
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>, #blocked_t>
    %x_addr = tt.addptr %x_splat, %rt : tensor<32x16x!tt.ptr<f32>, #blocked_t>, tensor<32x16xi32, #blocked_t>
    %x_val = tt.load %x_addr : tensor<32x16x!tt.ptr<f32>, #blocked_t>
    tt.store %x_addr, %x_val : tensor<32x16x!tt.ptr<f32>, #blocked_t>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel trans_kernel
// CHECK: metal.return
