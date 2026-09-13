// RUN: not triton-metal-opt --convert-tritongpu-to-metal %s 2>&1 | FileCheck %s --implicit-check-not="failed to legalize"
//
// arith.maxsi has a Metal tensor lowering but cannot be replayed by either
// rank-1 source evaluator. Reject before dialect conversion: this fixture used
// to require `not --crash` because failed-conversion teardown raised SIGSEGV.
// The shared source-cone preflight must now fail with a normal diagnostic.

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

// CHECK: rank-1 reduce cannot replay its source:
// CHECK-SAME: unsupported producer in elementwise chain: arith.maxsi
