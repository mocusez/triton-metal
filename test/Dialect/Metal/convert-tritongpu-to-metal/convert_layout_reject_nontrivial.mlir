// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Negative fixture: ttg.convert_layout pre-pass classification (Session L1d2,
// post the implementation notes).
// The pre-pass partitions non-identity cvts into:
//
//   - normalizable: a self-contained blocked↔blocked gather cone is rewritten
//     to the dst encoding so the cvt collapses to a direct gather/scatter
//     (`normalizeBlockedDivergentCvts`; covered by the rank-1 divergent-spt and
//     rank-2 transpose-normalized fixtures). This runs FIRST.
//   - in-envelope (rank-2 blocked↔blocked, same shape/elem-type, sizePerThread
//     = [1,1]) that did NOT normalize → `ConvertLayoutLowering`'s
//     staged-transpose body (covered by `convert_layout_staged_transpose.mlir`).
//   - everything else → rejected with the "deferred to L1d3" diagnostic.
//
// Each section below stores the loaded tile untransposed too, so the cvt cone
// is NOT self-contained and the normalizer bails — isolating the reject path
// (a self-contained cone would otherwise be normalized away, not rejected).

// -----
// Section 1: rank-3 blocked↔blocked cvt (out-of-envelope → L1d3 hand-off).
// rank-3 fails the in-envelope predicate's rank-2 check.

#blocked3a = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#blocked3b = #ttg.blocked<{sizePerThread = [1, 2, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 2, 2], order = [0, 1, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_reject_out_of_envelope(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>) {
    %offs = arith.constant dense<0> : tensor<4x8x8xi32, #blocked3a>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x8x8x!tt.ptr<f32>, #blocked3a>
    %x_addr = tt.addptr %x_splat, %offs : tensor<4x8x8x!tt.ptr<f32>, #blocked3a>, tensor<4x8x8xi32, #blocked3a>
    %x_val = tt.load %x_addr : tensor<4x8x8x!tt.ptr<f32>, #blocked3a>
    // External (non-cone) use → keeps the cone non-self-contained.
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<4x8x8x!tt.ptr<f32>, #blocked3a>
    %y_addr = tt.addptr %y_splat, %offs : tensor<4x8x8x!tt.ptr<f32>, #blocked3a>, tensor<4x8x8xi32, #blocked3a>
    tt.store %y_addr, %x_val : tensor<4x8x8x!tt.ptr<f32>, #blocked3a>
    // expected-error @+1 {{deferred to L1d3}}
    %x_cvt = ttg.convert_layout %x_val : tensor<4x8x8xf32, #blocked3a> -> tensor<4x8x8xf32, #blocked3b>
    %offs_b = arith.constant dense<0> : tensor<4x8x8xi32, #blocked3b>
    %x_splat_b = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x8x8x!tt.ptr<f32>, #blocked3b>
    %x_addr_b = tt.addptr %x_splat_b, %offs_b : tensor<4x8x8x!tt.ptr<f32>, #blocked3b>, tensor<4x8x8xi32, #blocked3b>
    tt.store %x_addr_b, %x_cvt : tensor<4x8x8x!tt.ptr<f32>, #blocked3b>
    tt.return
  }
}

// Section 2 USED to live here — a rank-2 blocked↔blocked transpose-shaped cvt
// with sizePerThread > 1 on both sides, the canonical leet
// `easy-matrix_transpose` post-coalesce shape, rejected because the staged body
// can only exchange one element per thread. That is what L1d3 implements, by
// re-encoding the consumer cone instead of moving data, so the shape now
// compiles and its coverage moved to
// `convert_layout_consumer_reencode.mlir`.
