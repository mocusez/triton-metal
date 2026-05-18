// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Negative fixture: ttg.convert_layout pre-pass classification (Session L1d2,
// post `.omc/specs/deep-interview-leet-triton-l1d2-staged-transpose-body.md`).
// After L1d2 the pre-pass partitions non-identity cvts into:
//
//   - in-envelope (rank-2 blocked↔blocked, same shape, same elem-type,
//     sizePerThread = [1,1] on both sides) → allow-through to
//     `ConvertLayoutLowering`'s staged-transpose body (covered by
//     `convert_layout_staged_transpose.mlir`).
//
//   - out-of-envelope → rejected with the "deferred to L1d3" diagnostic.
//     Out-of-envelope covers rank ≠ 2, shape/elem-type change, non-blocked
//     encodings, AND rank-2 blocked↔blocked with sizePerThread > 1.

// -----
// Section 1: rank-3 blocked↔blocked cvt (out-of-envelope → L1d3 hand-off).
// rank-3 fails the in-envelope predicate's rank-2 check.

#blocked3a = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#blocked3b = #ttg.blocked<{sizePerThread = [1, 2, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 2, 2], order = [0, 1, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_reject_out_of_envelope(%x_ptr: !tt.ptr<f32>) {
    %offs = arith.constant dense<0> : tensor<4x8x8xi32, #blocked3a>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x8x8x!tt.ptr<f32>, #blocked3a>
    %x_addr = tt.addptr %x_splat, %offs : tensor<4x8x8x!tt.ptr<f32>, #blocked3a>, tensor<4x8x8xi32, #blocked3a>
    %x_val = tt.load %x_addr : tensor<4x8x8x!tt.ptr<f32>, #blocked3a>
    // expected-error @+1 {{deferred to L1d3}}
    %x_cvt = ttg.convert_layout %x_val : tensor<4x8x8xf32, #blocked3a> -> tensor<4x8x8xf32, #blocked3b>
    %offs_b = arith.constant dense<0> : tensor<4x8x8xi32, #blocked3b>
    %x_splat_b = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x8x8x!tt.ptr<f32>, #blocked3b>
    %x_addr_b = tt.addptr %x_splat_b, %offs_b : tensor<4x8x8x!tt.ptr<f32>, #blocked3b>, tensor<4x8x8xi32, #blocked3b>
    tt.store %x_addr_b, %x_cvt : tensor<4x8x8x!tt.ptr<f32>, #blocked3b>
    tt.return
  }
}

// -----
// Section 2: rank-2 blocked↔blocked transpose-shaped cvt but with
// sizePerThread > 1 on both sides (the canonical leet `easy-matrix_transpose`
// post-coalesce shape). Now classified out-of-envelope because L1d2 only
// admits sizePerThread = [1, 1]. Vectorized cvt staging is L1d3 territory.

#blocked2a = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2b = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_reject_size_per_thread_gt_one(%x_ptr: !tt.ptr<f32>) {
    %offs = arith.constant dense<0> : tensor<16x16xi32, #blocked2a>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %x_addr = tt.addptr %x_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked2a>, tensor<16x16xi32, #blocked2a>
    %x_val = tt.load %x_addr : tensor<16x16x!tt.ptr<f32>, #blocked2a>
    // expected-error @+1 {{deferred to L1d3}}
    %x_cvt = ttg.convert_layout %x_val : tensor<16x16xf32, #blocked2a> -> tensor<16x16xf32, #blocked2b>
    %offs_b = arith.constant dense<0> : tensor<16x16xi32, #blocked2b>
    %x_splat_b = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2b>
    %x_addr_b = tt.addptr %x_splat_b, %offs_b : tensor<16x16x!tt.ptr<f32>, #blocked2b>, tensor<16x16xi32, #blocked2b>
    tt.store %x_addr_b, %x_cvt : tensor<16x16x!tt.ptr<f32>, #blocked2b>
    tt.return
  }
}
