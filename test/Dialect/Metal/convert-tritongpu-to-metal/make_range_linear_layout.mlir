// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A rank-1 `tt.make_range` under a `#ttg.linear` layout, evaluated out of the
// layout's own basis vectors (`planLinearRange`).
//
// This is the TTGIR of a rank-3 `tl.permute`:
//
//     i = tl.arange(0, 64)
//     v = tl.reshape(tl.load(x + i), (2, 4, 8))
//     tl.store(o + i, tl.reshape(tl.permute(v, (2, 0, 1)), (64,)))
//
// Triton folds the ENTIRE permutation into the load index's layout and leaves
// the reshape and the transpose as flat identities, so the permutation exists
// nowhere else. Imposing this backend's usual `element == localTid` mapping on
// the range erases the op and the kernel comes back a plain copy — which is why
// this used to be a named refusal rather than a wrong answer.
//
// The layout maps the bits of (lane, warp) to an element offset by XOR-ing in
// one basis per set bit:
//     lane = [[8], [16], [32], [1], [2]]   warp = [[4], [0]]
// so element = 8*l0 ^ 16*l1 ^ 32*l2 ^ 1*l3 ^ 2*l4 ^ 4*w0, and the second warp
// basis is zero — a BROADCAST, two threads holding the same element. It
// contributes no term but still costs a thread, so the threadgroup stride below
// must be 128 and not 64.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 1, 4], order = [1, 0, 2]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 2, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [], lane = [[8], [16], [32], [1], [2]], warp = [[4], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @permute_rank3(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %i = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #linear>
    %i0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #linear>
    %xa = tt.addptr %xs, %i : tensor<64x!tt.ptr<f32>, #linear>, tensor<64xi32, #linear>
    %v = tt.load %xa : tensor<64x!tt.ptr<f32>, #linear>
    %v3 = tt.reshape %v : tensor<64xf32, #linear> -> tensor<2x4x8xf32, #blocked1>
    %t = tt.trans %v3 {order = array<i32: 2, 0, 1>} : tensor<2x4x8xf32, #blocked1> -> tensor<8x2x4xf32, #blocked2>
    %os = tt.splat %o : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %os, %i0 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %r = tt.reshape %t : tensor<8x2x4xf32, #blocked2> -> tensor<64xf32, #blocked>
    tt.store %oa, %r : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// The threadgroup-local thread id is split at the layout's own warp size (32),
// and the element index is an XOR of the shifted-and-masked bits — NOT the bare
// localTid the blocked path would emit.
//
// CHECK-LABEL: metal.kernel permute_rank3
// CHECK: arith.constant 128 : i32
// CHECK-DAG: arith.remsi
// CHECK-DAG: arith.divsi
// CHECK: arith.xori
// CHECK: metal.get_element %arg0
// CHECK: metal.store %{{.*}}, %arg1
