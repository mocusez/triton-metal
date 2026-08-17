// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Session L1d2 staged-transpose body fixture (FALLBACK path).
// (`.omc/specs/deep-interview-leet-triton-l1d2-staged-transpose-body.md`):
// a rank-2 16×16 blocked↔blocked cvt with sizePerThread=[1,1] on both sides
// lowers to the staged-transpose sequence:
//   metal.threadgroup_alloca → metal.tg_store_indexed → metal.barrier →
//   metal.tg_load_indexed → metal.barrier → replaceOp.
// Both barriers are `mem_threadgroup` (BarrierOp has no flag operand —
// the MSL emitter pins `mem_threadgroup` per `MetalOps.td` §BarrierOp).
//
// Reaching the staged body takes work, because THREE normalizers get a turn
// first and each one erases the cvt where it can:
//   * the loaded tile is ALSO stored untransposed via %y_ptr, so the cvt's
//     producer cone is not self-contained and `normalizeBlockedDivergentCvt`
//     bails;
//   * the converted result passes through arith.addf before its store, so
//     `normalizeStoreSideBlockedDivergentCvt` (store-only consumers) bails;
//   * the result ALSO feeds a `tt.reduce`, which carries a region and its own
//     layout contract, so the L1d3 consumer-side re-encode
//     (`normalizeConsumerSideBlockedDivergentCvt`) bails too. Without this the
//     cvt would be rewritten away into the reduce-free cone and the staged body
//     would never run — that rewrite is the right answer whenever it applies,
//     and this fixture exists for what is left.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_staged_transpose_16x16(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>) {
    %offs = arith.constant dense<0> : tensor<16x16xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<16x16x!tt.ptr<f32>, #blocked>
    // External (non-cone) use of the loaded tile → divergent-cvt normalizer bails.
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %y_addr, %x_val : tensor<16x16x!tt.ptr<f32>, #blocked>
    %x_cvt = ttg.convert_layout %x_val : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked_t>
    %zero_t = arith.constant dense<0.0> : tensor<16x16xf32, #blocked_t>
    %x_cvt_used = arith.addf %x_cvt, %zero_t : tensor<16x16xf32, #blocked_t>
    %offs_t = arith.constant dense<0> : tensor<16x16xi32, #blocked_t>
    %x_splat_t = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked_t>
    %x_addr_t = tt.addptr %x_splat_t, %offs_t : tensor<16x16x!tt.ptr<f32>, #blocked_t>, tensor<16x16xi32, #blocked_t>
    tt.store %x_addr_t, %x_cvt_used : tensor<16x16x!tt.ptr<f32>, #blocked_t>
    // Region-bearing consumer: keeps the L1d3 consumer-side re-encode out.
    %red = "tt.reduce"(%x_cvt) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16x16xf32, #blocked_t>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked_t}>>
    %ridx = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked_t}>>
    %rsp = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked_t}>>
    %rp = tt.addptr %rsp, %ridx : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked_t}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked_t}>>
    tt.store %rp, %red : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked_t}>>
    tt.return
  }
}

// The 256-element buffer is the cvt's; the reduce allocates a 16-element row
// buffer of its own, which is why the size is pinned here.
// CHECK-LABEL: metal.kernel cvt_staged_transpose_16x16
// CHECK: %[[BUF:.+]] = metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.tg_store_indexed %[[BUF]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[BUF]]
// CHECK: metal.barrier
