// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// `arith.constant dense<C>` on a tensor type with a splat attribute
// lowers to a scalar `arith.constant C`. Triton's `tl.full([N], C,
// dtype)` compiles to this shape in TTGIR. See
// the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @const_dense_splat(%x_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    // Splat-of-constant: every element is 7. Lowers to scalar arith.constant 7.
    %seven = arith.constant dense<7> : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %seven : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %seven : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %o_addr, %x_val : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel const_dense_splat
// CHECK: metal.return
