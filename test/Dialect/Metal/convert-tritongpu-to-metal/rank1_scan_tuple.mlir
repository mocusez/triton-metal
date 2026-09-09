// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
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

// -----

// The reverse form is the MIRROR of the same traversal, not a flag on the
// result: the partner is `tid + stride`, live while it is inside the block, and
// the partner's range is STILL the left operand. Triton defines a reverse scan
// as a forward scan of the reversed sequence, so out[i] folds N-1 down to i and
// the higher positions are applied first.
//
// ⚠️ Getting that backwards is invisible to every commutative combine -- a
// reversed running sum and count are identical under both orders. Measured with
// the operands swapped: fifty-four of a hundred and eight differential cells
// against `TRITON_INTERPRET=1` disagreed, and every one of them was a
// reverse=True cell.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_reverse(%x: !tt.ptr<f32>, %oa: !tt.ptr<f32>, %ob: !tt.ptr<f32>) {
    %o = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %o : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %v = tt.load %ap : tensor<32x!tt.ptr<f32>, #blocked>
    %r:2 = "tt.scan"(%v, %v) ({
    ^bb0(%a1: f32, %a2: f32, %b1: f32, %b2: f32):
      %s = arith.addf %a1, %b1 : f32
      %m = arith.maxnumf %a2, %b2 : f32
      tt.scan.return %s, %m : f32, f32
    }) {axis = 0 : i32, reverse = true} : (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>) -> (tensor<32xf32, #blocked>, tensor<32xf32, #blocked>)
    %asp = tt.splat %oa : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %aap = tt.addptr %asp, %o : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %aap, %r#0 : tensor<32x!tt.ptr<f32>, #blocked>
    %bsp = tt.splat %ob : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %bap = tt.addptr %bsp, %o : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %bap, %r#1 : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_tuple_reverse
// CHECK: %[[BUF0:.*]] = metal.threadgroup_alloca : !metal.memref<32 x f32>
// CHECK: %[[BUF1:.*]] = metal.threadgroup_alloca : !metal.memref<32 x f32>
// CHECK: metal.barrier
// CHECK: metal.store %{{.*}}, %[[BUF0]]
// CHECK: metal.store %{{.*}}, %[[BUF1]]
// CHECK: metal.barrier
// The partner is ABOVE this lane, and it is live while it is inside the block.
// CHECK: %[[STRIDE:.*]] = arith.constant 1 : i32
// CHECK: %[[RAW:.*]] = arith.addi %{{.*}}, %[[STRIDE]] : i32
// CHECK: %[[HAS:.*]] = arith.cmpi slt, %[[RAW]], %{{.*}} : i32
// CHECK: %[[PARTNER:.*]] = arith.select %[[HAS]], %[[RAW]], %{{.*}} : i32
// CHECK: %[[L0:.*]] = metal.get_element %[[BUF0]][%{{.*}}] {metal.materialize}
// CHECK: %[[R0:.*]] = metal.get_element %[[BUF0]][%{{.*}}] {metal.materialize}
// CHECK: %[[L1:.*]] = metal.get_element %[[BUF1]][%{{.*}}] {metal.materialize}
// CHECK: %[[R1:.*]] = metal.get_element %[[BUF1]][%{{.*}}] {metal.materialize}
// The partner's range is the left operand here too.
// CHECK: %[[SUM:.*]] = arith.addf %[[L0]], %[[R0]] : f32
// CHECK: %[[MAX:.*]] = arith.maxnumf %[[L1]], %[[R1]] : f32
// CHECK: %[[N0:.*]] = arith.select %[[HAS]], %[[SUM]], %[[R0]] : f32
// CHECK: %[[N1:.*]] = arith.select %[[HAS]], %[[MAX]], %[[R1]] : f32
// CHECK: metal.barrier
// CHECK: metal.store %[[N0]], %[[BUF0]]
// CHECK: metal.store %[[N1]], %[[BUF1]]
// CHECK: metal.barrier

// -----

// More than one element per thread: three phases. Each thread scans its OWN
// contiguous band into a buffer with one slot per POSITION, the band totals are
// scanned across threads by the same tree the one-element case uses, and each
// thread then folds the exclusive prefix of the bands before it into every
// element of its own. Thread zero has no such prefix and there is no identity
// to stand in for one, so it keeps what its band holds.
//
// The band has to be contiguous for a scan of the totals to stand for a scan of
// the positions, and the result buffer needs a slot per position because the
// consumer reads a DIFFERENT position on each tile-loop iteration -- one that
// need not lie in this thread's band, which is what the last barrier is for.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_band(%x: !tt.ptr<f32>, %oa: !tt.ptr<f32>, %ob: !tt.ptr<i32>) {
    %o = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %o : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %v = tt.load %ap : tensor<64x!tt.ptr<f32>, #blocked>
    %r:2 = "tt.scan"(%v, %o) ({
    ^bb0(%a1: f32, %a2: i32, %b1: f32, %b2: i32):
      %s = arith.addf %a1, %b1 : f32
      %m = arith.addi %a2, %b2 : i32
      tt.scan.return %s, %m : f32, i32
    }) {axis = 0 : i32, reverse = false} : (tensor<64xf32, #blocked>, tensor<64xi32, #blocked>) -> (tensor<64xf32, #blocked>, tensor<64xi32, #blocked>)
    %asp = tt.splat %oa : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %aap = tt.addptr %asp, %o : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    tt.store %aap, %r#0 : tensor<64x!tt.ptr<f32>, #blocked>
    %bsp = tt.splat %ob : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
    %bap = tt.addptr %bsp, %o : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
    tt.store %bap, %r#1 : tensor<64x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_tuple_band
// One totals slot per thread and one result slot per position, per value.
// CHECK: %[[TOTV:.*]] = metal.threadgroup_alloca : !metal.memref<32 x f32>
// CHECK: %[[RESV:.*]] = metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK: %[[TOTI:.*]] = metal.threadgroup_alloca : !metal.memref<32 x ui32>
// CHECK: %[[RESI:.*]] = metal.threadgroup_alloca : !metal.memref<64 x ui32>
// CHECK: metal.barrier
// Phase one: the thread's own band, `[t*E, (t+1)*E)`.
// CHECK: %[[E:.*]] = arith.constant 2 : i32
// CHECK: %[[BASE:.*]] = arith.muli %{{.*}}, %[[E]] : i32
// CHECK: %[[SEED:.*]] = metal.get_element
// CHECK: metal.store %[[SEED]], %[[RESV]]
// CHECK: metal.store %{{.*}}, %[[RESI]]
// CHECK: %[[BAND:.*]]:2 = scf.for %[[K:.*]] = %{{.*}} to %[[E]] step %{{.*}} iter_args(%[[PV:.*]] = %[[SEED]], %[[PI:.*]] = %[[BASE]])
// CHECK:   %[[POS:.*]] = arith.addi %[[BASE]], %[[K]] : i32
// The running prefix is the LOWER range, so it is the left operand.
// CHECK:   %[[SUM:.*]] = arith.addf %[[PV]], %{{.*}} : f32
// CHECK:   arith.addi %[[PI]], %[[POS]] : i32
// CHECK:   metal.store %[[SUM]], %[[RESV]]
// CHECK:   metal.store %{{.*}}, %[[RESI]]
// Phase two: the band totals, scanned across threads by the ordinary tree.
// CHECK: metal.store %[[BAND]]#0, %[[TOTV]]
// CHECK: metal.store %{{.*}}, %[[TOTI]]
// CHECK: metal.barrier
// CHECK: arith.cmpi sge
// CHECK: metal.barrier
// Phase three: fold the exclusive prefix of the earlier bands into every
// element of this one. Thread zero keeps what it has -- there is no identity.
// CHECK: %[[HAS:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
// CHECK: %[[PREV:.*]] = arith.subi %{{.*}}, %{{.*}} : i32
// CHECK: arith.select %[[HAS]], %[[PREV]], %{{.*}} : i32
// CHECK: %[[CARRYV:.*]] = metal.get_element %[[TOTV]][%{{.*}}] {metal.materialize}
// CHECK: scf.for
// CHECK:   %[[HERE:.*]] = metal.get_element %[[RESV]][%{{.*}}] {metal.materialize}
// CHECK:   %[[ADDED:.*]] = arith.addf %[[CARRYV]], %[[HERE]] : f32
// CHECK:   %[[OUT:.*]] = arith.select %[[HAS]], %[[ADDED]], %[[HERE]] : f32
// CHECK:   metal.store %[[OUT]], %[[RESV]]
// Every band has to be finished before any of them is read.
// CHECK: metal.barrier
// CHECK: metal.get_element %[[RESV]][%{{.*}}] {metal.materialize}
// CHECK: metal.get_element %[[RESI]][%{{.*}}] {metal.materialize}
