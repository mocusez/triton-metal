// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// BLOCK_SIZE=1024 vector_add with contiguous per-thread layout
// (sizePerThread=[8]). 128 threads each process 8 contiguous elements
// with idx = tid * 8 + iv. Tile loop wraps load/compute/store; per-iter
// mask check guards each iteration.
// See `.omc/specs/deep-interview-metal-block-size-loop.md`.

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_contiguous(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>, %n_elements: i32) {
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

// METAL: metal.module
// METAL: metal.kernel add_kernel_contiguous
// METAL: arith.constant 0 : i32
// METAL: arith.constant 8 : i32
// METAL: arith.constant 1 : i32
// METAL: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// METAL: metal.thread_id "x"
// METAL: arith.constant 8 : i32
// METAL: arith.muli
// METAL: arith.addi
// METAL: arith.cmpi slt
// METAL: scf.if {{.*}} -> (f32)
// METAL: metal.get_element
// METAL: metal.return

// Post-Lmultiload-Phase-C: 1D canonical short-circuit deleted. MakeRange
// emits `localTid*E + iv` (contiguous tile-loop form); AddPtr accumulates
// `pid*BLOCK + (localTid*E + iv)`. The mask compare still rebuilds against
// `id.x` directly (mask reconstruction is independent of the load index).
// See `.omc/specs/deep-interview-lmultiload-phase-c-makerange.md`.
// MSL: kernel void add_kernel_contiguous(
// MSL: device float *v{{[0-9]+}}
// MSL: device float *v{{[0-9]+}}
// MSL: device float *v{{[0-9]+}}
// MSL: device uint32_t *v{{[0-9]+}}
// MSL: thread_position_in_grid
// MSL: threadgroup_position_in_grid
// MSL: for (int v{{[0-9]+}} = 0; v{{[0-9]+}} < 8; v{{[0-9]+}} += 1)
// MSL: if ((((id.x * 8) + v{{[0-9]+}}) < v{{[0-9]+}}[0]))
// MSL: v{{[0-9]+}} = v{{[0-9]+}}[((tgid.x * 1024) + (((id.x - (tgid.x * 128)) * 8) + v{{[0-9]+}}))];
// MSL: return;
