// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// `tt.join` / `tt.split` / `tt.cat` — what `tl.join`, `tl.split`, `tl.cat` and
// `tl.interleave` lower to. All three were rejected outright.
//
// Triton picks layouts that keep them per-thread: the join result below is
// `sizePerThread = [1, 2]`, i.e. one thread owns both columns of its row. This
// backend does not honour a layout's thread assignment — it imposes
// `flat = localTid*E + iv`, decomposed by `order` — so under ITS mapping the
// result element (i, k) lives in lane `i*2 + k` while source element i lives in
// lane i. That is a cross-lane move, and it goes through a threadgroup buffer:
// each lane publishes what it holds, barrier, then reads the slot it needs.
//
// EXCEPT under the sub-tpb companion mapping. A rank-2 tile smaller than the
// threadgroup makes every rank-1 value of its row extent held per ROW
// (`findSubTpbRank2Companion`, and MakeRangeLowering agrees), so each lane
// already holds a[row] and b[row] and the join is a local select. Reading the
// wrong one of these two conventions is a silent wrong answer, not a crash: it
// returned a0,b0,a0,b0,a1,b1,a1,b1 where a0,b0,a1,b1,a2,b2,a3,b3 was meant.

// 16x2 == threads per block: no companion, so the shuffle path.
// CHECK-LABEL: metal.kernel join_full_tile
// CHECK: metal.threadgroup_alloca
// CHECK: metal.barrier
// CHECK: metal.store
// CHECK: metal.barrier
// CHECK: metal.get_element
// MSL-LABEL: kernel void join_full_tile
// MSL: threadgroup float v[[BUF:[0-9]+]][64];
// Both operands are published, the second in the band at slot tpb.
// MSL: int v[[TID:[0-9]+]] = (id.x - (tgid.x * 32));
// MSL: v[[BUF]][v[[TID]]] =
// MSL: v[[BUF]][(v[[TID]] + 32)] =
// MSL: threadgroup_barrier(mem_flags::mem_threadgroup);
// The read is clamped: lanes past the tile still execute it, and an
// out-of-range threadgroup access is undefined rather than merely wasted.
// MSL: = v[[BUF]][min((int)(max(

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @join_full_tile(%x_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %j = arith.constant dense<1.000000e+02> : tensor<16xf32, #blocked>
    %cst = arith.constant dense<2> : tensor<16x1xi32, #blocked1>
    %a = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %a_0 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %a_1 = tt.addptr %a_0, %a : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %a_2 = tt.load %a_1 : tensor<16x!tt.ptr<f32>, #blocked>
    %j_3 = arith.addf %a_2, %j : tensor<16xf32, #blocked>
    %j_4 = tt.join %a_2, %j_3 : tensor<16xf32, #blocked> -> tensor<16x2xf32, #blocked2>
    %0 = ttg.convert_layout %j_4 : tensor<16x2xf32, #blocked2> -> tensor<16x2xf32, #blocked1>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1>
    %3 = arith.muli %2, %cst : tensor<16x1xi32, #blocked1>
    %4 = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked1>
    %5 = tt.addptr %4, %3 : tensor<16x1x!tt.ptr<f32>, #blocked1>, tensor<16x1xi32, #blocked1>
    %6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x2xi32, #blocked1>
    %8 = tt.broadcast %5 : tensor<16x1x!tt.ptr<f32>, #blocked1> -> tensor<16x2x!tt.ptr<f32>, #blocked1>
    %9 = tt.broadcast %7 : tensor<1x2xi32, #blocked1> -> tensor<16x2xi32, #blocked1>
    %10 = tt.addptr %8, %9 : tensor<16x2x!tt.ptr<f32>, #blocked1>, tensor<16x2xi32, #blocked1>
    tt.store %10, %0 : tensor<16x2x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
