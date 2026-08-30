// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Regression lock for the AC4-v6 misplaced-guard bug. BLOCK_SIZE=1024 with
// sizePerThread=[2] (2 contiguous elems per thread) + mask. Adds a mid-
// point sample between the existing spt=[1] masked fixture and the new
// spt=[4] masked fixture, covering the `sizePerThread x mask` matrix.
// See the implementation notes (AC3).

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_spt2_masked(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>, %n_elements: i32) {
    %c256_i32 = arith.constant 256 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c256_i32 : i32
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<256xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<256xi32, #blocked>
    %n_splat = tt.splat %n_elements : i32 -> tensor<256xi32, #blocked>
    %mask = arith.cmpi slt, %abs_off, %n_splat : tensor<256xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %abs_off : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %x_val = tt.load %x_addr, %mask : tensor<256x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %abs_off : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %y_val = tt.load %y_addr, %mask : tensor<256x!tt.ptr<f32>, #blocked>
    %sum = arith.addf %x_val, %y_val : tensor<256xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %abs_off : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %o_addr, %sum, %mask : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// 256 elements / 128 threads = 2 per thread. addf → metal.binary_exp addOp.
// METAL: metal.module
// METAL: metal.kernel add_kernel_spt2_masked
// METAL: scf.for
// METAL: arith.cmpi slt
// METAL: scf.if
// METAL: metal.get_element
// METAL: metal.binary_exp
// METAL: metal.return

// MSL: kernel void add_kernel_spt2_masked(
// MSL: for (int v{{[0-9]+}} = 0; v{{[0-9]+}} < 2; v{{[0-9]+}} += 1)
// MSL: if (
// MSL: return;
