// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// 2D blocked-layout foundation: the conversion pipeline now accepts
// rank=2 tensors when computing tile info. This fixture exercises
// `tileFromTensor` on a 2D blocked tensor that flows through the
// already-lowered `tt.splat` op, then returns. No 2D load/store/arith
// is performed (those require `tt.expand_dims` / `tt.broadcast` /
// 2D mask lowering which are deferred to the next slice).
// See the implementation notes.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @blocked_2d_passthrough(%x_ptr: !tt.ptr<f32>) {
    // tt.splat creates a 2D blocked tensor. `tileFromTensor` (called from
    // FuncOpLowering) walks the body, sees this result type, and now
    // (after the rank>=1 lift) returns a valid TileInfo for rank=2
    // instead of nullopt. SplatLowering passes the source through, so
    // the splat itself disappears in the converted IR.
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK: metal.module
// CHECK: metal.kernel blocked_2d_passthrough
// CHECK: metal.return
