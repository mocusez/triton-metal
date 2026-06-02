// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 12 lit fixture: rank-1 arith.divf lowering via ArithDivFLowering.
// Mirror of arith_subf_rank1.mlir with arith.divf in place of arith.subf.
// Before Wall 12 ships, this fixture FAILS with
// "failed to legalize operation 'arith.divf'". After Wall 12 ships, the
// op lowers to `metal.binary_exp ..., ..., divOp`.
// See .omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC6-AC7.

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @div_kernel_spt8_unmasked(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_val = tt.load %y_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %quot = arith.divf %x_val, %y_val : tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %quot : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel div_kernel_spt8_unmasked
// One get_element per operand, then a single divOp per lane.
// CHECK: metal.get_element %arg0
// CHECK: metal.get_element %arg1
// CHECK: metal.binary_exp {{.*}}, {{.*}}, divOp
// CHECK: metal.store
// CHECK: metal.return
