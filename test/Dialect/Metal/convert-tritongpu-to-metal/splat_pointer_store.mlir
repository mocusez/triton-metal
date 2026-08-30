// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A store whose address folded down to a bare `tt.splat` of the kernel
// argument. At BLOCK=1 the offset is zero, so Triton leaves no `tt.addptr` at
// all — the shape the verbatim leet-triton RoPE kernel produces at head dim 2,
// where BLOCK_SIZE_D = next_pow2(D/2) = 1 gives a 1x1 tile.
//
// It used to match no store pattern, and an unmatched op in this backend is a
// process kill during rollback rather than an error.
//
// CHECK-LABEL: metal.kernel splat_store_1x1
// The load already accepted this shape.
// CHECK: metal.get_element %arg0[
// The store's index is a plain constant — the whole tile is ONE address, so
// there is no div/rem of the thread id in it...
// CHECK: %[[OFF:.*]] = arith.constant 0 : i32
// CHECK: %[[OFFU:.*]] = builtin.unrealized_conversion_cast %[[OFF]] : i32 to ui32
// ...and the sub-tpb guard leaves exactly one lane writing it.
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: arith.cmpi slt, %{{.*}}, %[[ONE]]
// CHECK: scf.if
// CHECK: metal.store %{{.*}}, %arg1[%[[OFFU]]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @splat_store_1x1(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x1xf32, #blocked>
    %px = tt.splat %x : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>, #blocked>
    %v = tt.load %px : tensor<1x1x!tt.ptr<f32>, #blocked>
    %po = tt.splat %o : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>, #blocked>
    %r = arith.mulf %v, %cst : tensor<1x1xf32, #blocked>
    tt.store %po, %r : tensor<1x1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
