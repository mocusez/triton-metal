// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// BLOCK_SIZE=1024 vector_add with contiguous per-thread layout
// (sizePerThread=[8]). 128 threads each process 8 contiguous elements.
// The three buffers are proven 16-byte aligned, so the exact canonical
// masked-add envelope may use two float4 transactions per thread. The emitted
// op retains a scalar tail for the last partial program.
// See the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_contiguous(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
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
// METAL-LABEL: metal.kernel add_kernel_contiguous
// METAL: metal.get_element
// METAL: metal.contiguous_vector_add
// METAL-SAME: elements_per_thread = 8
// METAL-SAME: vector_width = 4
// METAL-NOT: scf.for
// METAL: metal.return

// The full-vector arm performs two aligned float4 loads from each input and
// two aligned float4 stores. The fallback loop preserves the original mask for
// a partial final program and therefore never touches out-of-range elements.
// MSL: kernel void add_kernel_contiguous(
// MSL: device float *v{{[0-9]+}}
// MSL: device float *v{{[0-9]+}}
// MSL: device float *v{{[0-9]+}}
// MSL: device uint32_t *v{{[0-9]+}}
// MSL: thread_position_in_grid
// MSL: uint base = id.x * 8u;
// MSL: int n =
// MSL: float4
// MSL: (device float4*)
// MSL: (device float4*)
// MSL: for (uint lane = 0u; lane < 8u; ++lane)
// MSL: if (idx < uint(n))
// MSL: return;
