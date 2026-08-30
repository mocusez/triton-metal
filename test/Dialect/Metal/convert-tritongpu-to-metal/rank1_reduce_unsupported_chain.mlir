// RUN: not --crash triton-metal-opt --convert-tritongpu-to-metal -debug-only=dialect-conversion %s 2>&1 | FileCheck %s
//
// NOTE on `not --crash`: triton-metal-opt has a known cleanup-on-conversion-
// failure bug where the post-failure region teardown SIGSEGVs (see exit code
// 139 in the lit output). The bug is unrelated to Wall 11 — it occurs on any
// conversion failure, not specifically chain rejection. `not --crash` accepts
// the crash exit so we can FileCheck the diagnostic that fires BEFORE the
// crash.
//
// Negative fixture: a rank-1 reduce whose producer chain contains an op outside
// BOTH the Wall-11 single-load walker whitelist AND the W-B rich cone evaluator
// (`evalRank1ValueAt` / `rank1ConeSupported`, which cover splat/const/make_range/
// masked-load/sitofp/unary-math/f32-arith/int-arith/select/cmp).
//
// `arith.maxsi` is chosen because it HAS a Metal tensor lowering
// (`ArithMaxSILowering`) — so the conversion driver does NOT hard-crash on a
// totally-unconvertible op — yet it is NOT a rank-1 cone producer, so the reduce
// is cleanly rejected: the walker emits
// `unsupported producer in elementwise chain:` and the rich fallback's
// `rank1ConeSupported` returns false. (Earlier versions used `math.log`, then
// `arith.remsi`; both are now supported by the rich cone evaluator.)
//
// See the implementation notes AC4.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unsupported_chain(%x_ptr: !tt.ptr<i32>, %m: i32) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    %x = tt.load %x_addr : tensor<1024x!tt.ptr<i32>, #blocked>
    // arith.maxsi: has a Metal tensor lowering but is NOT a cone producer.
    %msplat = tt.splat %m : i32 -> tensor<1024xi32, #blocked>
    %max = arith.maxsi %x, %msplat : tensor<1024xi32, #blocked>
    %r = "tt.reduce"(%max) ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.addi %a, %b : i32
      tt.reduce.return %s : i32
    }) {axis = 0 : i32} : (tensor<1024xi32, #blocked>) -> i32
    tt.return
  }
}

// CHECK: unsupported producer in elementwise chain:
