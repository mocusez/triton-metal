// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s
//
// `tl.cat` under the sub-tpb companion mapping.
//
// Upstream deleted `tt.cat`, so `tl.cat(a, b, can_reorder=True)` now lowers to
// `tt.join` -> `ttg.convert_layout` -> `tt.trans` -> `tt.reshape`. The two
// sides of that convert_layout carry OPPOSITE `order` ([1, 0] and [0, 1]), and
// `normalizeBlockedDivergentCvts` re-encodes the join's result into the far
// side of it. The join then had its ROW from the companion (frozen from the
// pristine module, order [1, 0]) and its COLUMN from its own re-encoded result
// type (order [0, 1]) — two index mappings inside one op. Lane pairs got the
// same row and different columns, so `cat` of [0..7] with [100..107] returned
// 0,0,1,1,2,2,3,3,104,104,105,105,106,106,107,107: half the elements twice,
// half of them not at all.
//
// Both coordinates now come from the companion, which is also what
// `MakeRangeLowering` put the rank-1 operands under.

// CHECK-LABEL: kernel void cat_sub_tpb
// The operand is read at the companion's ROW, `(localTid / 2) % 8` ...
// CHECK: v0[((int32_t)(((int32_t)((id.x - (tgid.x * 32))) / (int32_t)(2))) % (int32_t)(8))]
// ... and the lhs/rhs select tests the companion's COLUMN, `localTid % 2`.
// The wrong answer tested `localTid / 8` — the re-encoded result's column.
// CHECK: ((int32_t)((id.x - (tgid.x * 32))) % (int32_t)(2))) != (int32_t)(0)) ?
// CHECK-NOT: / (int32_t)(8)

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cat_sub_tpb(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %o_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+02> : tensor<8xf32, #blocked>
    %a = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %a_0 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked>
    %a_1 = tt.addptr %a_0, %a : tensor<8x!tt.ptr<f32>, #blocked>, tensor<8xi32, #blocked>
    %a_2 = tt.load %a_1 : tensor<8x!tt.ptr<f32>, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %1 = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %3 = arith.addf %a_2, %cst : tensor<8xf32, #blocked>
    %4 = tt.join %a_2, %3 : tensor<8xf32, #blocked> -> tensor<8x2xf32, #blocked1>
    %5 = ttg.convert_layout %4 : tensor<8x2xf32, #blocked1> -> tensor<8x2xf32, #blocked2>
    %6 = tt.trans %5 {order = array<i32: 1, 0>} : tensor<8x2xf32, #blocked2> -> tensor<2x8xf32, #blocked3>
    %7 = tt.reshape %6 : tensor<2x8xf32, #blocked3> -> tensor<16xf32, #blocked>
    tt.store %2, %7 : tensor<16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
