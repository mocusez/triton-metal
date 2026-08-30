// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// A rank-2 reduce over an axis of extent ONE is a no-op reduce, but the tile it
// leaves behind is degenerate — `tl.arange(0, 1)` as the inner axis gives a
// BLOCK x 1 tile whose slice-encoded rank-1 values have no lowering. Declining
// it DURING conversion is a process kill in this backend (the rollback
// segfaults) and poisons the context for the next compile, so it is rejected
// in the pre-pass, where the error is catchable.
//
// Independent of the sub-tpb companion path: this fires at num_warps=1 too,
// where the tile is exactly threadgroup-sized and no companion is claimed.

// -----
// axis=1 over a BLOCK x 1 tile.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unit_axis1(%out: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32x1xf32, #blocked>
    // expected-error @+1 {{reduce over an axis of extent 1 is not supported}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 1 : i32} : (tensor<32x1xf32, #blocked>) -> tensor<32xf32, #slice>
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
    // expected-error @+1 {{reduce over an axis of extent 1 is not supported}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1x32xf32, #blocked2>) -> tensor<32xf32, #slice2>
    tt.return
  }
}
