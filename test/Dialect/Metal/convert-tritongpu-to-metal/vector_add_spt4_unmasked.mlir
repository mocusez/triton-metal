// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Regression lock for the AC4-v6 misplaced-guard bug. BLOCK_SIZE=1024 with
// sizePerThread=[4] (4 contiguous elems per thread) without a mask:
// covers the unmasked half of the sizePerThread x mask matrix the previous
// session identified as a coverage gap.
// See the implementation notes (AC3).

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_spt4_unmasked(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_val = tt.load %y_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %sum = arith.addf %x_val, %y_val : tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %sum : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// Unmasked path: tile-loop wraps direct get_element load/store (no scf.if
// guard). addf → metal.binary_exp addOp. E = 1024 / 128 = 8.
// METAL: metal.module
// METAL: metal.kernel add_kernel_spt4_unmasked
// METAL: scf.for
// METAL: metal.get_element
// METAL: metal.binary_exp
// METAL: metal.return

// MSL: kernel void add_kernel_spt4_unmasked(
// MSL: for (int v{{[0-9]+}} = 0; v{{[0-9]+}} < 8; v{{[0-9]+}} += 1)
// MSL: return;
