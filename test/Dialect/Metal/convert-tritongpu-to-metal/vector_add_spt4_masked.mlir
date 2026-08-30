// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Regression lock for the AC4-v6 misplaced-guard bug. BLOCK_SIZE=1024 with
// sizePerThread=[4] (4 contiguous elems per thread) combined with a masked
// load: this is the exact TTGIR shape the Triton frontend selects for the
// failing vector_add tutorial that triggered the universal `tt.load`
// legalization failure on metal-develop HEAD `8c5ef140b0` before the fix.
// See the implementation notes
// and the implementation notes (AC3).

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_spt4_masked(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>, %n_elements: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<1024xi32, #blocked>
    %n_splat = tt.splat %n_elements : i32 -> tensor<1024xi32, #blocked>
    %mask = arith.cmpi slt, %abs_off, %n_splat : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_val = tt.load %y_addr, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    %sum = arith.addf %x_val, %y_val : tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %sum, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// Minimum-viable METAL coverage: kernel emitted, tile-loop wraps the
// masked load/compute/store path, scf.if guards each per-iter load and
// store. Don't over-specify the index arithmetic — that's a moving target
// across MakeRange / Lmultiload phases. The regression we lock is
// "compile completes at all", asserted by `metal.return`. Note: addf
// gets rewritten to `metal.binary_exp ..., addOp` during conversion.
// METAL: metal.module
// METAL: metal.kernel add_kernel_spt4_masked
// METAL: scf.for
// METAL: arith.cmpi slt
// METAL: scf.if
// METAL: metal.get_element
// METAL: metal.binary_exp
// METAL: metal.return

// MSL kernel emission completes; the for-loop iterates the per-thread
// tile (E = 1024 / 128 threads = 8); the masked load and store both fire
// inside the per-iter guard.
// MSL: kernel void add_kernel_spt4_masked(
// MSL: for (int v{{[0-9]+}} = 0; v{{[0-9]+}} < 8; v{{[0-9]+}} += 1)
// MSL: if (
// MSL: return;
