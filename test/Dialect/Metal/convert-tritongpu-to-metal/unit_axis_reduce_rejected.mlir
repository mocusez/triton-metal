// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Singleton-axis reductions are views, including tuple reductions. No combine
// operation or neutral seed may change the original value. These shapes used
// to be rejected before conversion because their slice lowering was missing.

// -----
// axis=1 over a BLOCK x 1 tile.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unit_axis1(%out: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32x1xf32, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 1 : i32} : (tensor<32x1xf32, #blocked>) -> tensor<32xf32, #slice>
    %i = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #slice>
    %ptr = tt.addptr %base, %i : tensor<32x!tt.ptr<f32>, #slice>, tensor<32xi32, #slice>
    tt.store %ptr, %r : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----
// axis=0 over a 1 x BLOCK tile — the same degeneracy on the other axis.
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice2 = #ttg.slice<{dim = 0, parent = #blocked2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unit_axis0(%out: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<1x32xf32, #blocked2>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1x32xf32, #blocked2>) -> tensor<32xf32, #slice2>
    %i = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice2>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #slice2>
    %ptr = tt.addptr %base, %i : tensor<32x!tt.ptr<f32>, #slice2>, tensor<32xi32, #slice2>
    tt.store %ptr, %r : tensor<32x!tt.ptr<f32>, #slice2>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel unit_axis1
// CHECK-NOT: metal.barrier
// CHECK-NOT: metal.simd_sum
// CHECK: metal.store
// CHECK: metal.return
// CHECK-LABEL: metal.kernel unit_axis0
// CHECK-NOT: metal.barrier
// CHECK-NOT: metal.simd_sum
// CHECK: metal.store
// CHECK: metal.return

// -----
// Four warps replicate the 32 logical rows. The input coordinate must wrap
// just like the squeezed output coordinate, so extra warps cannot read padding
// and race the first warp's valid stores.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#vector = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unit_axis1_subtpb_load(%input: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %i = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice>
    %row = tt.expand_dims %i {axis = 1 : i32} : tensor<32xi32, #slice> -> tensor<32x1xi32, #blocked>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %input_ptr = tt.addptr %input_base, %row : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %x = tt.load %input_ptr : tensor<32x1x!tt.ptr<f32>, #blocked>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.minnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 1 : i32} : (tensor<32x1xf32, #blocked>) -> tensor<32xf32, #slice>
    %result = ttg.convert_layout %r : tensor<32xf32, #slice> -> tensor<32xf32, #vector>
    %j = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #vector>
    %output_base = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #vector>
    %output_ptr = tt.addptr %output_base, %j : tensor<32x!tt.ptr<f32>, #vector>, tensor<32xi32, #vector>
    tt.store %output_ptr, %result : tensor<32x!tt.ptr<f32>, #vector>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel unit_axis1_subtpb_load
// CHECK: [[EXTENT:%.*]] = arith.constant 32 : i32
// CHECK: [[ROW:%.*]] = arith.remsi {{%.*}}, [[EXTENT]] : i32
// CHECK: [[INPUT_INDEX:%.*]] = builtin.unrealized_conversion_cast [[ROW]] : i32 to ui32
// CHECK: [[VALUE:%.*]] = metal.get_element %arg0{{\[}}[[INPUT_INDEX]]{{\]}}
// CHECK: metal.store [[VALUE]], %arg1
// CHECK: metal.return
