// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// BLOCK_SIZE=1024 vector_add with strided per-thread layout
// (sizePerThread=[1]). 128 threads each process 8 elements, interleaved
// across the 1024-element tensor with idx = tid + iv * 128. Tile loop
// wraps load/compute/store; per-iter mask check guards each iteration.
// See `.omc/specs/deep-interview-metal-block-size-loop.md`.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_strided(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>, %n_elements: i32) {
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
// METAL: metal.kernel add_kernel_strided
// METAL: arith.constant 0 : i32
// METAL: arith.constant 8 : i32
// METAL: arith.constant 1 : i32
// METAL: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// METAL: metal.thread_id "x"
// METAL: arith.constant 128 : i32
// METAL: arith.muli
// METAL: arith.addi
// METAL: arith.cmpi slt
// METAL: scf.if {{.*}} -> (f32)
// METAL: metal.get_element
// METAL: metal.return

// Wall 13 fix (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC8):
// MakeRange now emits `localTid + iv*tpb` for BOTH the load index and the
// mask predicate (was global `id.x + iv*128` pre-fix). For vector_add, the
// load index combines this with `pid*BLOCK` via AddPtr chained accumulation.
// MSL: kernel void add_kernel_strided(
// MSL: device float *v{{[0-9]+}}
// MSL: device float *v{{[0-9]+}}
// MSL: device float *v{{[0-9]+}}
// MSL: device uint32_t *v{{[0-9]+}}
// MSL: thread_position_in_grid
// MSL: threadgroup_position_in_grid
// MSL: for (int v{{[0-9]+}} = 0; v{{[0-9]+}} < 8; v{{[0-9]+}} += 1)
// Mask reads the FULL index cone `pid*1024 + localtid + iv*128 < N` (address v6),
// not the old local-only form (which was wrong for grid>1).
// MSL: int v{{[0-9]+}} = ((tgid.x * 1024) + ((id.x - (tgid.x * 128)) + (v{{[0-9]+}} * 128)));
// MSL: bool v{{[0-9]+}} = ((int32_t)(v{{[0-9]+}}) < (int32_t)(v{{[0-9]+}}[0]));
// MSL: if (v{{[0-9]+}})
// MSL: v{{[0-9]+}} = v{{[0-9]+}}[v{{[0-9]+}}];
// MSL: return;
