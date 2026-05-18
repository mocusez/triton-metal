// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Session L1d2 positive fixture
// (`.omc/specs/deep-interview-leet-triton-l1d2-staged-transpose-body.md`):
// rank-2 16×16 blocked↔blocked cvt with sizePerThread=[1,1] on both sides
// must lower to the staged-transpose sequence:
//   metal.threadgroup_alloca → metal.tg_store_indexed → metal.barrier →
//   metal.tg_load_indexed → metal.barrier → replaceOp.
// Both barriers are `mem_threadgroup` (BarrierOp has no flag operand —
// the MSL emitter pins `mem_threadgroup` per `MetalOps.td` §BarrierOp).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_staged_transpose_16x16(%x_ptr: !tt.ptr<f32>) {
    %offs = arith.constant dense<0> : tensor<16x16xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<16x16x!tt.ptr<f32>, #blocked>
    %x_cvt = ttg.convert_layout %x_val : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked_t>
    %offs_t = arith.constant dense<0> : tensor<16x16xi32, #blocked_t>
    %x_splat_t = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked_t>
    %x_addr_t = tt.addptr %x_splat_t, %offs_t : tensor<16x16x!tt.ptr<f32>, #blocked_t>, tensor<16x16xi32, #blocked_t>
    tt.store %x_addr_t, %x_cvt : tensor<16x16x!tt.ptr<f32>, #blocked_t>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel cvt_staged_transpose_16x16
// CHECK: %[[BUF:.+]] = metal.threadgroup_alloca : !metal.memref<256 x f32>
// CHECK: metal.tg_store_indexed %[[BUF]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[BUF]]
// CHECK: metal.barrier
