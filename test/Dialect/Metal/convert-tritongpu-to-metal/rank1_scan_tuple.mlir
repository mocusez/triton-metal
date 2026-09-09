// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A general multi-value `tl.associative_scan((a, b), 0, combine)`: an inclusive
// Hillis-Steele prefix over one lockstep threadgroup buffer per carried value,
// replaying the scan's own combine region at each merge. The dedicated
// `metal.threadgroup_prefix_sum` the single-value path uses carries ONE value
// and one of two hard-wired monoids, so it cannot serve this.
//
// This fixture exists to pin the MECHANISM rather than an answer. Each step
// reads the buffers, then barriers, then writes them, so every lane has read
// the previous step's values before any lane overwrites them; and each read is
// forced into a named binding, because the emitter inlines a single-use value
// at its USE, which would move the read past that barrier. Neither property is
// observable from a numeric test: with the read-before-write barrier deleted,
// a hundred and forty differential runs over one, two, four and eight warps all
// still passed. A race that does not fire is still a race.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_sum_and_max(%x: !tt.ptr<f32>, %oa: !tt.ptr<f32>, %ob: !tt.ptr<f32>) {
    %o = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %o : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %v = tt.load %ap : tensor<32x!tt.ptr<f32>, #blocked>
    %r:2 = "tt.scan"(%v, %v) ({
    ^bb0(%a1: f32, %a2: f32, %b1: f32, %b2: f32):
      %s = arith.addf %a1, %b1 : f32
      %m = arith.maxnumf %a2, %b2 : f32
      tt.scan.return %s, %m : f32, f32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>) -> (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>)
    %asp = tt.splat %oa : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %aap = tt.addptr %asp, %o : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %aap, %r#0 : tensor<32x!tt.ptr<f32>, #blocked>
    %bsp = tt.splat %ob : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %bap = tt.addptr %bsp, %o : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %bap, %r#1 : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// One buffer per carried value; both are published under one pair of barriers.
// CHECK-LABEL: metal.kernel scan_tuple_sum_and_max
// CHECK: %[[BUF0:.*]] = metal.threadgroup_alloca : !metal.memref<32 x f32>
// CHECK: %[[BUF1:.*]] = metal.threadgroup_alloca : !metal.memref<32 x f32>
// CHECK: metal.barrier
// CHECK: metal.store %{{.*}}, %[[BUF0]]
// CHECK: metal.store %{{.*}}, %[[BUF1]]
// CHECK: metal.barrier
// The partner is `tid - stride`, clamped so a lane below the stride still
// addresses inside the buffer; its merge is discarded by the select.
// CHECK: %[[STRIDE:.*]] = arith.constant 1 : i32
// CHECK: %[[RAW:.*]] = arith.subi %{{.*}}, %[[STRIDE]] : i32
// CHECK: %[[HAS:.*]] = arith.cmpi sge, %{{.*}}, %[[STRIDE]] : i32
// CHECK: %[[PARTNER:.*]] = arith.select %[[HAS]], %[[RAW]], %{{.*}} : i32
// Every read is a named binding pinned to this position.
// CHECK: %[[L0:.*]] = metal.get_element %[[BUF0]][%{{.*}}] {metal.materialize}
// CHECK: %[[R0:.*]] = metal.get_element %[[BUF0]][%{{.*}}] {metal.materialize}
// CHECK: %[[L1:.*]] = metal.get_element %[[BUF1]][%{{.*}}] {metal.materialize}
// CHECK: %[[R1:.*]] = metal.get_element %[[BUF1]][%{{.*}}] {metal.materialize}
// The region, replayed with the LOWER position as the left operand, so an
// associative-but-not-commutative combine concatenates its ranges in order.
// CHECK: %[[SUM:.*]] = arith.addf %[[L0]], %[[R0]] : f32
// CHECK: %[[MAX:.*]] = arith.maxnumf %[[L1]], %[[R1]] : f32
// A lane below the stride keeps what it holds; there is no identity to pad with.
// CHECK: %[[N0:.*]] = arith.select %[[HAS]], %[[SUM]], %[[R0]] : f32
// CHECK: %[[N1:.*]] = arith.select %[[HAS]], %[[MAX]], %[[R1]] : f32
// The barrier that separates this step's reads from its writes.
// CHECK: metal.barrier
// CHECK: metal.store %[[N0]], %[[BUF0]]
// CHECK: metal.store %[[N1]], %[[BUF1]]
// CHECK: metal.barrier
// Five steps for a thirty-two element block, then each lane reads its own slot.
// CHECK: arith.constant 16 : i32
// CHECK: metal.barrier
// CHECK: metal.store
// CHECK: metal.store
// CHECK: metal.barrier
// CHECK: metal.get_element %[[BUF0]][%{{.*}}] {metal.materialize}
// CHECK: metal.get_element %[[BUF1]][%{{.*}}] {metal.materialize}
