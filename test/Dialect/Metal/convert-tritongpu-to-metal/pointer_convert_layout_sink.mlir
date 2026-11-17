// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A `ttg.convert_layout` of a rank >= 2 POINTER tile used to leave its load with
// no kernel-argument root: the convert exchanged the addresses as integers, the
// load pattern declined inside the conversion, and the rollback could read a
// block `FuncOpLowering` had already freed. The pass now converts the OFFSETS
// beneath the pointer instead (`cvt(addptr(splat p, off))` becomes
// `addptr(splat p, cvt(off))`), so the load reads `%x` directly.
// CHECK-LABEL: metal.kernel load_through_converted_ptr
// CHECK: metal.get_element %arg0[
// CHECK: metal.store
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @load_through_converted_ptr(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %xsp = tt.splat %x : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %xrow = tt.addptr %xsp, %row2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %xcvt = ttg.convert_layout %xrow : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x1x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %v = tt.load %xcvt : tensor<64x1x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %osp = tt.splat %o : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %oaddr = tt.addptr %osp, %row2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %ocvt = ttg.convert_layout %v : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #blocked>
    tt.store %oaddr, %ocvt : tensor<64x1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
