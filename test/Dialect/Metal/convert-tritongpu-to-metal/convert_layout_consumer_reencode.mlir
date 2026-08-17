// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// L1d3: a rank-2 blocked↔blocked relabel with sizePerThread > 1 on both sides —
// the canonical `easy-matrix_transpose` post-coalesce shape, and what
// `tl.trans(x) * tl.load(y)` produces at the default num_warps.
//
// `ConvertLayoutLowering`'s staged transpose cannot take it: with more than one
// element per thread the publish and the read both sit inside the scalarized
// tile loop, and the slot a lane needs at iteration `iv` is written by another
// lane at some other iteration. This module used to be a REJECT fixture
// (`convert_layout_reject_nontrivial.mlir` §2).
//
// `normalizeConsumerSideBlockedDivergentCvt` does not move the data at all. It
// re-encodes the forward cone — the `arith.addf`, and the store's address cone —
// into the cvt's SOURCE layout and erases the cvt. Every lane then handles a
// different element, and the set of (address, value) pairs is unchanged.
//
// The loaded tile is also stored untransposed via %y_ptr, so the BACKWARD
// normalizer bails; the elementwise op between the cvt and its store keeps the
// store-only normalizer out. This is the consumer-side rewrite or nothing.

#blocked2a = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2b = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_consumer_reencode_spt_gt_one(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>) {
    %offs = arith.constant dense<0> : tensor<16x16xi32, #blocked2a>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %x_addr = tt.addptr %x_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked2a>, tensor<16x16xi32, #blocked2a>
    %x_val = tt.load %x_addr : tensor<16x16x!tt.ptr<f32>, #blocked2a>
    // External (non-cone) use → the backward normalizer bails.
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %y_addr = tt.addptr %y_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked2a>, tensor<16x16xi32, #blocked2a>
    tt.store %y_addr, %x_val : tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %x_cvt = ttg.convert_layout %x_val : tensor<16x16xf32, #blocked2a> -> tensor<16x16xf32, #blocked2b>
    %one_b = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #blocked2b>
    %x_add = arith.addf %x_cvt, %one_b : tensor<16x16xf32, #blocked2b>
    %offs_b = arith.constant dense<0> : tensor<16x16xi32, #blocked2b>
    %x_splat_b = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2b>
    %x_addr_b = tt.addptr %x_splat_b, %offs_b : tensor<16x16x!tt.ptr<f32>, #blocked2b>, tensor<16x16xi32, #blocked2b>
    tt.store %x_addr_b, %x_add : tensor<16x16x!tt.ptr<f32>, #blocked2b>
    tt.return
  }
}

// It compiles, and it does so with NO cross-lane exchange: the point of the
// rewrite is that no threadgroup memory and no barrier are needed at any
// sizePerThread. A `metal.threadgroup_alloca` here would mean the cvt fell
// through to the staged body instead.
//
// CHECK-LABEL: metal.kernel cvt_consumer_reencode_spt_gt_one
// CHECK-NOT: metal.threadgroup_alloca
// CHECK-NOT: metal.barrier
// CHECK: metal.return
