// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// AC3 half-slice end-to-end: TTGIR vector_add (unmasked, BLOCK_SIZE=128,
// 1-elt-per-thread) → metal dialect → MSL. See
// `.omc/specs/deep-interview-ac3-half-slice-finish.md`.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %c128_i32 = arith.constant 128 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c128_i32 : i32
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<128xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %y_val = tt.load %y_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %sum = arith.addf %x_val, %y_val : tensor<128xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %abs_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %o_addr, %sum : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// METAL: metal.module
// METAL: metal.kernel add_kernel
// METAL: metal.threadgroup_id "x"
// METAL: metal.thread_id "x"
// METAL: metal.threadgroup_id "x"
// METAL: arith.subi
// METAL: arith.addi
// METAL: metal.get_element
// METAL: metal.get_element
// METAL: metal.binary_exp
// METAL-SAME: addOp
// METAL: metal.store
// METAL: metal.return
// METAL: metal.module_end

// Post-Lmultiload-Phase-C: 1D canonical short-circuit deleted.
// MakeRange emits `localTid = id.x - tgid.x*tpb`; AddPtrLowering
// accumulates the full chain → MSL is arithmetic-explicit. See
// `.omc/specs/deep-interview-lmultiload-phase-c-makerange.md`.
// MSL: kernel void add_kernel(
// MSL: device float
// MSL: device float
// MSL: device float
// MSL: thread_position_in_grid
// MSL: threadgroup_position_in_grid
// MSL: v{{[0-9]+}}[((tgid.x * 128) + (id.x - (tgid.x * 128)))] = (v{{[0-9]+}}[((tgid.x * 128) + (id.x - (tgid.x * 128)))]) + (v{{[0-9]+}}[((tgid.x * 128) + (id.x - (tgid.x * 128)))]);
// MSL: return;
