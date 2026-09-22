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

// -----

// Reverse each band and carry the suffix from the next thread.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_reverse_band(%x: !tt.ptr<f32>, %oa: !tt.ptr<f32>, %ob: !tt.ptr<i32>) {
    %o = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %o : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %ap : tensor<128x!tt.ptr<f32>, #blocked>
    %r:2 = "tt.scan"(%v, %o) ({
    ^bb0(%a1: f32, %a2: i32, %b1: f32, %b2: i32):
      %s = arith.addf %a1, %b1 : f32
      %m = arith.addi %a2, %b2 : i32
      tt.scan.return %s, %m : f32, i32
    }) {axis = 0 : i32, reverse = true} : (tensor<128xf32, #blocked>, tensor<128xi32, #blocked>) -> (tensor<128xf32, #blocked>, tensor<128xi32, #blocked>)
    %asp = tt.splat %oa : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %aap = tt.addptr %asp, %o : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %aap, %r#0 : tensor<128x!tt.ptr<f32>, #blocked>
    %bsp = tt.splat %ob : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %bap = tt.addptr %bsp, %o : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %bap, %r#1 : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_tuple_reverse_band
// CHECK: %[[TOTV:.*]] = metal.threadgroup_alloca : !metal.memref<32 x f32>
// CHECK: %[[RESV:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: %[[TOTI:.*]] = metal.threadgroup_alloca : !metal.memref<32 x ui32>
// CHECK: %[[RESI:.*]] = metal.threadgroup_alloca : !metal.memref<128 x ui32>
// CHECK-NEXT: metal.barrier
// Start at the last position of the band, then subtract the loop counter.
// CHECK: %[[E:.*]] = arith.constant 4 : i32
// CHECK: %[[BASE:.*]] = arith.muli %[[TID:.*]], %[[E]] : i32
// CHECK: %[[LAST:.*]] = arith.constant 3 : i32
// CHECK: %[[START:.*]] = arith.addi %[[BASE]], %[[LAST]] : i32
// CHECK: %[[STARTUI:.*]] = builtin.unrealized_conversion_cast %[[START]] : i32 to ui32
// CHECK: %[[SEED:.*]] = metal.get_element
// CHECK: metal.store %[[SEED]], %[[RESV]][%[[STARTUI]]]
// CHECK: metal.store %{{.*}}, %[[RESI]][%[[STARTUI]]]
// CHECK: %[[BAND:.*]]:2 = scf.for %[[K:.*]] = %{{.*}} to %[[E]] step %{{.*}} iter_args(%[[PV:.*]] = %[[SEED]], %[[PI:.*]] = %[[START]])
// CHECK:   %[[POS:.*]] = arith.subi %[[START]], %[[K]] : i32
// CHECK:   %[[POSUI:.*]] = builtin.unrealized_conversion_cast %[[POS]] : i32 to ui32
// The already scanned, higher positions stay on the left of the combine.
// CHECK:   %[[SUM:.*]] = arith.addf %[[PV]], %{{.*}} : f32
// CHECK:   %[[ISUM:.*]] = arith.addi %[[PI]], %[[POS]] : i32
// CHECK:   metal.store %[[SUM]], %[[RESV]][%[[POSUI]]]
// CHECK:   metal.store %{{.*}}, %[[RESI]][%[[POSUI]]]
// CHECK:   scf.yield %[[SUM]], %[[ISUM]] : f32, i32
// CHECK: metal.store %[[BAND]]#0, %[[TOTV]]
// CHECK: metal.store %{{.*}}, %[[TOTI]]
// CHECK-NEXT: metal.barrier
// Phase two uses the higher band's total as its left operand.
// CHECK: %[[RAW:.*]] = arith.addi %[[TID]], %{{.*}} : i32
// CHECK: arith.cmpi slt, %[[RAW]], %{{.*}} : i32
// CHECK: %[[LEFT:.*]] = metal.get_element %[[TOTV]][%{{.*}}] {metal.materialize}
// CHECK: %[[OWN:.*]] = metal.get_element %[[TOTV]][%{{.*}}] {metal.materialize}
// CHECK: %[[MERGE:.*]] = arith.addf %[[LEFT]], %[[OWN]] : f32
// CHECK: %[[NEXT:.*]] = arith.select %{{.*}}, %[[MERGE]], %[[OWN]] : f32
// CHECK: metal.barrier
// CHECK-NEXT: metal.store %[[NEXT]], %[[TOTV]]
// CHECK: metal.store %{{.*}}, %[[TOTI]]
// CHECK-NEXT: metal.barrier
// The last band has no carry. Clamp its unused read to an allocated slot.
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[LASTTID:.*]] = arith.constant 31 : i32
// CHECK: %[[HAS:.*]] = arith.cmpi slt, %[[TID]], %[[LASTTID]] : i32
// CHECK: %[[NEXTTID:.*]] = arith.addi %[[TID]], %[[ONE]] : i32
// CHECK: %[[SAFE:.*]] = arith.select %[[HAS]], %[[NEXTTID]], %{{.*}} : i32
// CHECK: %[[SAFEUI:.*]] = builtin.unrealized_conversion_cast %[[SAFE]] : i32 to ui32
// CHECK: %[[CARRY:.*]] = metal.get_element %[[TOTV]][%[[SAFEUI]]] {metal.materialize}
// CHECK: scf.for %[[J:.*]] =
// CHECK:   %[[OUTPOS:.*]] = arith.addi %[[BASE]], %[[J]] : i32
// CHECK:   %[[OUTUI:.*]] = builtin.unrealized_conversion_cast %[[OUTPOS]] : i32 to ui32
// CHECK:   %[[HERE:.*]] = metal.get_element %[[RESV]][%[[OUTUI]]] {metal.materialize}
// CHECK:   %[[ADDED:.*]] = arith.addf %[[CARRY]], %[[HERE]] : f32
// CHECK:   %[[OUT:.*]] = arith.select %[[HAS]], %[[ADDED]], %[[HERE]] : f32
// CHECK:   metal.store %[[OUT]], %[[RESV]][%[[OUTUI]]]
// CHECK:   metal.store %{{.*}}, %[[RESI]][%[[OUTUI]]]
// CHECK-NEXT: }
// CHECK-NEXT: metal.barrier
// Consumer reads use the same logical position for both values.
// CHECK: metal.get_element %[[RESV]][%[[READPOS:.*]]] {metal.materialize}
// CHECK: metal.get_element %[[RESI]][%[[READPOS]]] {metal.materialize}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "metal:0", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_rank2_reverse(%out: !tt.ptr<i32>, %x: i32) {
    %v = tt.splat %x : i32 -> tensor<4x8xi32, #blocked>
    %scan:2 = "tt.scan"(%v, %v) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%lhs0: i32, %lhs1: i32, %rhs0: i32, %rhs1: i32):
      %sum = arith.addi %lhs0, %rhs0 : i32
      tt.scan.return %sum, %rhs1 : i32, i32
    }) : (tensor<4x8xi32, #blocked>, tensor<4x8xi32, #blocked>) -> (tensor<4x8xi32, #blocked>, tensor<4x8xi32, #blocked>)
    %row = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #row>
    %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #col>
    %eight = arith.constant dense<8> : tensor<4xi32, #row>
    %rowOff = arith.muli %row, %eight : tensor<4xi32, #row>
    %r = tt.expand_dims %rowOff {axis = 1 : i32} : tensor<4xi32, #row> -> tensor<4x1xi32, #blocked>
    %c = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #col> -> tensor<1x8xi32, #blocked>
    %rr = tt.broadcast %r : tensor<4x1xi32, #blocked> -> tensor<4x8xi32, #blocked>
    %cc = tt.broadcast %c : tensor<1x8xi32, #blocked> -> tensor<4x8xi32, #blocked>
    %off = arith.addi %rr, %cc : tensor<4x8xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>, #blocked>
    %ptr = tt.addptr %base, %off : tensor<4x8x!tt.ptr<i32>, #blocked>, tensor<4x8xi32, #blocked>
    %combined = arith.addi %scan#0, %scan#1 : tensor<4x8xi32, #blocked>
    tt.store %ptr, %combined : tensor<4x8x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_tuple_rank2_reverse
// CHECK: %[[T0:.*]] = metal.threadgroup_alloca : !metal.memref<32 x ui32>
// CHECK: %[[T1:.*]] = metal.threadgroup_alloca : !metal.memref<32 x ui32>
// CHECK: metal.barrier
// CHECK: metal.store {{.*}}, %[[T0]][%[[TID:.*]]]
// CHECK-NOT: metal.barrier
// CHECK: metal.store {{.*}}, %[[T1]][%[[TID]]]
// CHECK-NEXT: metal.barrier
// Reverse axis 0 starts in row 3 (offset 24) and next consumes row 2.
// CHECK: %[[LAST:.*]] = arith.constant 24 : i32
// CHECK: arith.addi {{.*}}, %[[LAST]] : i32
// CHECK: %[[SEEDU:.*]] = metal.tg_load_indexed %[[T0]][
// CHECK: %[[SEED:.*]] = builtin.unrealized_conversion_cast %[[SEEDU]] : ui32 to i32
// CHECK: metal.tg_load_indexed %[[T1]][
// CHECK: %[[ACTIVE:.*]] = arith.cmpi sge
// CHECK: scf.if %[[ACTIVE]] -> (i32, i32) {
// CHECK-NOT: metal.barrier
// CHECK: arith.constant 16 : i32
// CHECK: %[[NEXTU:.*]] = metal.tg_load_indexed %[[T0]][
// CHECK: %[[NEXT:.*]] = builtin.unrealized_conversion_cast %[[NEXTU]] : ui32 to i32
// CHECK: metal.tg_load_indexed %[[T1]][
// CHECK: %[[SUM:.*]] = arith.addi %[[SEED]], %[[NEXT]] : i32
// CHECK: scf.yield %[[SUM]], {{.*}} : i32, i32
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[SEED]], {{.*}} : i32, i32
// CHECK-NOT: tt.scan

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "metal:0", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_rank2_multiband(%out: !tt.ptr<i32>, %x: i32) {
    %v = tt.splat %x : i32 -> tensor<8x8xi32, #blocked>
    %scan:2 = "tt.scan"(%v, %v) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%lhs0: i32, %lhs1: i32, %rhs0: i32, %rhs1: i32):
      %sum = arith.addi %lhs0, %rhs0 : i32
      tt.scan.return %sum, %rhs1 : i32, i32
    }) : (tensor<8x8xi32, #blocked>, tensor<8x8xi32, #blocked>) -> (tensor<8x8xi32, #blocked>, tensor<8x8xi32, #blocked>)
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #row>
    %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #col>
    %eight = arith.constant dense<8> : tensor<8xi32, #row>
    %rowOff = arith.muli %row, %eight : tensor<8xi32, #row>
    %r = tt.expand_dims %rowOff {axis = 1 : i32} : tensor<8xi32, #row> -> tensor<8x1xi32, #blocked>
    %c = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #col> -> tensor<1x8xi32, #blocked>
    %rr = tt.broadcast %r : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %cc = tt.broadcast %c : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %off = arith.addi %rr, %cc : tensor<8x8xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x8x!tt.ptr<i32>, #blocked>
    %ptr = tt.addptr %base, %off : tensor<8x8x!tt.ptr<i32>, #blocked>, tensor<8x8xi32, #blocked>
    %combined = arith.addi %scan#0, %scan#1 : tensor<8x8xi32, #blocked>
    tt.store %ptr, %combined : tensor<8x8x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_tuple_rank2_multiband
// CHECK: %[[B0:.*]] = metal.threadgroup_alloca : !metal.memref<64 x ui32>
// CHECK: %[[B1:.*]] = metal.threadgroup_alloca : !metal.memref<64 x ui32>
// CHECK: scf.for
// CHECK: metal.tg_store_indexed %[[B0]][
// CHECK-NOT: metal.barrier
// CHECK-NOT: scf.for
// CHECK: metal.tg_store_indexed %[[B1]][
// CHECK: metal.barrier
// CHECK-NEXT: {{.*}}arith.constant
// CHECK: scf.for
// CHECK: metal.tg_load_indexed %[[B0]][
// CHECK: metal.tg_load_indexed %[[B1]][
// CHECK: scf.if {{.*}} -> (i32, i32)
// CHECK-NOT: metal.barrier
// CHECK: arith.addi
// CHECK-NOT: tt.scan

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "metal:0", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_tuple_rank2_runtime_loop(%out: !tt.ptr<i32>, %x: i32, %iterations: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iteration = %c0 to %iterations step %c1 : i32 {
      %v = tt.splat %x : i32 -> tensor<8x8xi32, #blocked>
      %scan:2 = "tt.scan"(%v, %v) <{axis = 0 : i32, reverse = true}> ({
      ^bb0(%lhs0: i32, %lhs1: i32, %rhs0: i32, %rhs1: i32):
        %sum = arith.addi %lhs0, %rhs0 : i32
        tt.scan.return %sum, %rhs1 : i32, i32
      }) : (tensor<8x8xi32, #blocked>, tensor<8x8xi32, #blocked>) -> (tensor<8x8xi32, #blocked>, tensor<8x8xi32, #blocked>)
      %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #row>
      %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #col>
      %eight = arith.constant dense<8> : tensor<8xi32, #row>
      %rowOff = arith.muli %row, %eight : tensor<8xi32, #row>
      %r = tt.expand_dims %rowOff {axis = 1 : i32} : tensor<8xi32, #row> -> tensor<8x1xi32, #blocked>
      %c = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #col> -> tensor<1x8xi32, #blocked>
      %rr = tt.broadcast %r : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
      %cc = tt.broadcast %c : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
      %off = arith.addi %rr, %cc : tensor<8x8xi32, #blocked>
      %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x8x!tt.ptr<i32>, #blocked>
      %ptr = tt.addptr %base, %off : tensor<8x8x!tt.ptr<i32>, #blocked>, tensor<8x8xi32, #blocked>
      %combined = arith.addi %scan#0, %scan#1 : tensor<8x8xi32, #blocked>
      tt.store %ptr, %combined : tensor<8x8x!tt.ptr<i32>, #blocked>
      scf.yield
    }
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_tuple_rank2_runtime_loop
// CHECK: %[[L0:.*]] = metal.threadgroup_alloca : !metal.memref<64 x ui32>
// CHECK: %[[L1:.*]] = metal.threadgroup_alloca : !metal.memref<64 x ui32>
// The runtime loop encloses both band phases; its leading barrier protects reuse.
// CHECK: scf.for %[[USER:.*]] =
// CHECK: metal.barrier
// CHECK: scf.for %[[BAND:.*]] =
// CHECK: metal.tg_store_indexed %[[L0]][
// CHECK-NOT: metal.barrier
// CHECK: metal.tg_store_indexed %[[L1]][
// CHECK: metal.scan_band_loop
// CHECK-NEXT: metal.barrier
// CHECK: scf.for
// CHECK: metal.tg_load_indexed %[[L0]][
// CHECK: metal.tg_load_indexed %[[L1]][
// CHECK: metal.scan_band_loop
// CHECK-NOT: tt.scan

// -----

// A canonical AFFINE scan whose source the dedicated primitive cannot replay.
// That primitive rebuilds both sources at a logical position, and `math.fma`
// is not in the replay table; before this, it declined from inside the
// conversion and took the process down. At one element per thread the general
// tuple scan needs no replay at all -- it reads the converted scalars -- so
// the op falls through to it rather than being refused. The census is the
// ABSENCE of the primitive: a numeric test cannot tell the two apart, both
// being correct inclusive affine scans.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "metal:0", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_affine_unreplayable_source_falls_through(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %ptr : tensor<128x!tt.ptr<f32>, #blocked>
    // `arith.maxsi` stands in for "a producer no cone replays". It is not on
    // the widening path, unlike the `math.fma` that used to sit here, so this
    // fixture does not need revisiting each time the tables learn another op.
    %a_i = arith.fptosi %v : tensor<128xf32, #blocked> to tensor<128xi32, #blocked>
    %a_c = arith.constant dense<3> : tensor<128xi32, #blocked>
    %a_m = arith.maxsi %a_i, %a_c : tensor<128xi32, #blocked>
    %a = arith.sitofp %a_m : tensor<128xi32, #blocked> to tensor<128xf32, #blocked>
    %r:2 = "tt.scan"(%a, %v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%la: f32, %lx: f32, %ra: f32, %rx: f32):
      %ca = arith.mulf %la, %ra : f32
      %cm = arith.mulf %ra, %lx : f32
      %cx = arith.addf %cm, %rx : f32
      tt.scan.return %ca, %cx : f32, f32
    }) : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>)
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %op_0 = tt.addptr %op, %off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %op_0, %r#1 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_affine_unreplayable_source_falls_through
// CHECK-NOT: metal.threadgroup_affine_prefix_scan
// CHECK: %[[Z:.*]] = arith.sitofp
// CHECK: %[[BUFA:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: %[[BUFX:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: metal.barrier
// CHECK: metal.store %[[Z]], %[[BUFA]]
// CHECK: metal.barrier
// CHECK-NOT: metal.threadgroup_affine_prefix_scan
// CHECK-NOT: tt.scan
