// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics %s
//
// The same uniform (`tt.splat`) address as splat_pointer_store.mlir, but under
// a tile of MORE than one element: eight lanes holding values that need not
// agree, all aimed at one address. That is a race, not a store.
//
// Triton reaches this shape — `tl.store(p + tl.zeros((8,), tl.int32), v)` folds
// the zero offset away — so it has to be answered rather than assumed absent.
// Declining it inside conversion aborted the process, so the pre-pass names it.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @splat_store_wide(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %cst = arith.constant dense<2.000000e+00> : tensor<8xf32, #blocked>
    %px = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked>
    %v = tt.load %px : tensor<8x!tt.ptr<f32>, #blocked>
    %po = tt.splat %o : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked>
    %r = arith.mulf %v, %cst : tensor<8xf32, #blocked>
    // expected-error @+1 {{sends every element of the tile to one address}}
    tt.store %po, %r : tensor<8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
