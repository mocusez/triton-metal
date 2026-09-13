// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Adjacent negatives for the general multi-value scan (the positive is
// rank1_scan_tuple.mlir). Every one of these is refused BEFORE conversion: a
// pattern that declines inside `applyFullConversion` does not raise on this
// backend, it takes the process down during the failed conversion's rollback.
// The predicate is the one the pattern itself dispatches on, so the two cannot
// disagree about which shapes are lowered.

// -----
// A per-thread band longer than the prefix is implemented for. Above one
// element per thread a thread scans its own band first (see
// rank1_scan_tuple.mlir), and that band is capped: a four-thousand-element
// block over one warp gives each thread a hundred and twenty-eight elements.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_band_too_long(%x: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<4096xf32, #blocked>
    %w = arith.constant dense<2.0> : tensor<4096xf32, #blocked>
    // expected-error @+1 {{a general tuple scan is implemented up to sixty-four elements per thread}}
    %r:2 = "tt.scan"(%v, %w) ({
    ^bb0(%a1: f32, %a2: f32, %b1: f32, %b2: f32):
      %s = arith.addf %a1, %b1 : f32
      %m = arith.maxnumf %a2, %b2 : f32
      tt.scan.return %s, %m : f32, f32
    }) {axis = 0 : i32, reverse = false} : (tensor<4096xf32, #blocked>, tensor<4096xf32, #blocked>) -> (tensor<4096xf32, #blocked>, tensor<4096xf32, #blocked>)
    tt.return
  }
}

// -----
// The same band limit applies in reverse.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_reverse_band_too_long(%x: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<4096xf32, #blocked>
    %w = arith.constant dense<2.0> : tensor<4096xf32, #blocked>
    // expected-error @+1 {{a general tuple scan is implemented up to sixty-four elements per thread}}
    %r:2 = "tt.scan"(%v, %w) ({
    ^bb0(%a1: f32, %a2: f32, %b1: f32, %b2: f32):
      %s = arith.addf %a1, %b1 : f32
      %m = arith.maxnumf %a2, %b2 : f32
      tt.scan.return %s, %m : f32, f32
    }) {axis = 0 : i32, reverse = true} : (tensor<4096xf32, #blocked>, tensor<4096xf32, #blocked>) -> (tensor<4096xf32, #blocked>, tensor<4096xf32, #blocked>)
    tt.return
  }
}

// -----
// f16 slots. Each buffer element is one 32-bit slot, and only f32 and i32 have
// a spelling on both sides of its ui32 storage bridge.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_f16_deferred(%x: !tt.ptr<f16>) {
    %v = arith.constant dense<1.0> : tensor<32xf16, #blocked>
    %w = arith.constant dense<2.0> : tensor<32xf16, #blocked>
    // expected-error @+1 {{a general tuple scan carries f32 and i32 values only}}
    %r:2 = "tt.scan"(%v, %w) ({
    ^bb0(%a1: f16, %a2: f16, %b1: f16, %b2: f16):
      %s = arith.addf %a1, %b1 : f16
      %m = arith.maxnumf %a2, %b2 : f16
      tt.scan.return %s, %m : f16, f16
    }) {axis = 0 : i32, reverse = false} : (tensor<32xf16, #blocked>, tensor<32xf16, #blocked>) -> (tensor<32xf16, #blocked>, tensor<32xf16, #blocked>)
    tt.return
  }
}

// -----
// A combine holding an op the replayer has no scalar spelling for. The prefix
// clones the region at every merge, so an op it cannot clone is not a combine
// it can perform.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_unreplayable_combine(%x: !tt.ptr<i32>) {
    %v = arith.constant dense<1> : tensor<32xi32, #blocked>
    %w = arith.constant dense<2> : tensor<32xi32, #blocked>
    // expected-error @+1 {{the combine region must be one block of replayable scalar ops over its own arguments}}
    %r:2 = "tt.scan"(%v, %w) ({
    ^bb0(%a1: i32, %a2: i32, %b1: i32, %b2: i32):
      %pop = math.ctpop %a1 : i32
      %s = arith.addi %pop, %b1 : i32
      %m = arith.addi %a2, %b2 : i32
      tt.scan.return %s, %m : i32, i32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi32, #blocked>, tensor<32xi32, #blocked>) -> (tensor<32xi32, #blocked>, tensor<32xi32, #blocked>)
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "metal:0", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_rank2_over_budget(%x: i32) {
    %v = tt.splat %x : i32 -> tensor<128x64xi32, #blocked>
    // expected-error @+1 {{a multidimensional tuple scan exceeds the threadgroup budget}}
    %scan:2 = "tt.scan"(%v, %v) <{axis = 1 : i32, reverse = true}> ({
    ^bb0(%lhs0: i32, %lhs1: i32, %rhs0: i32, %rhs1: i32):
      %sum = arith.addi %lhs0, %rhs0 : i32
      tt.scan.return %sum, %rhs1 : i32, i32
    }) : (tensor<128x64xi32, #blocked>, tensor<128x64xi32, #blocked>) -> (tensor<128x64xi32, #blocked>, tensor<128x64xi32, #blocked>)
    tt.return
  }
}
