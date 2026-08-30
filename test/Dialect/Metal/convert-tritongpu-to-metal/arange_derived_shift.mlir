// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=METAL
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Lmultiload Phase C — AC.T4.
//
// Regression for the interleave-shaped pattern `tl.load(ptr + (arange >> 1))`.
// Pre-Phase-C the 1D `MakeRangeLowering` emitted `arith.constant 0`,
// causing `arith.shrsi(0, 1) == 0` to collapse the per-thread offset to
// the same memory location for every thread (the leet
// `easy-interleave_arrays.py` runtime FAIL). Post-Phase-C MakeRange
// emits the per-thread localTid term so the `>> 1` survives into MSL.
//
// See the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @arange_shifted(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %c1 = arith.constant dense<1> : tensor<128xi32, #blocked>
    %halved = arith.shrsi %offsets, %c1 : tensor<128xi32, #blocked>
    %in_splat = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %in_addr = tt.addptr %in_splat, %halved : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_val = tt.load %in_addr : tensor<128x!tt.ptr<f32>, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %o_addr, %x_val : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// METAL: metal.kernel arange_shifted
// METAL: arith.subi
// METAL: arith.shrsi
// METAL: metal.get_element
// METAL: metal.store

// MSL: kernel void arange_shifted(
// MSL: thread_position_in_grid
// MSL: threadgroup_position_in_grid
// MSL: int v{{[0-9]+}} = (id.x - (tgid.x * 128));
// MSL: v{{[0-9]+}}[v{{[0-9]+}}] = v{{[0-9]+}}[((int32_t)(v{{[0-9]+}}) >> (int32_t)(1))];
