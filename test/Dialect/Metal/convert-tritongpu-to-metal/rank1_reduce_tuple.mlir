// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// A general multi-value `tl.reduce((a, b), 0, combine)`: N lockstep threadgroup
// buffers, one per carried value, with the reduce's own combine region replayed
// at each merge. Both shapes below used to be REFUSED as "not one of the two
// canonical paired reductions" -- and the first of them reached that refusal
// only because a pre-pass mirrored the pattern's bail-out line by hand; without
// it the decline happened inside `applyFullConversion` and took the process
// down.
//
// The tree merges in LOGICAL order (stride 1, 2, 4, ...), unlike the argmax
// butterfly's halving stride: a tuple combine is only required to be
// associative, so `((v0 . v1) . (v2 . v3))` and `((v0 . v2) . (v1 . v3))` are
// not interchangeable.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_tuple_maxnum_pair(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %y = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %r:2 = "tt.reduce"(%x, %y) ({
    ^bb0(%a1: f32, %b1: f32, %a2: f32, %b2: f32):
      %m1 = arith.maxnumf %a1, %a2 : f32
      %m2 = arith.maxnumf %b1, %b2 : f32
      tt.reduce.return %m1, %m2 : f32, f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> (f32, f32)
    tt.return
  }
}

// One f32 buffer per carried value, published together under one pair of
// barriers.
// CHECK-LABEL: metal.kernel reduce_tuple_maxnum_pair
// CHECK: %[[BUF0:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: %[[BUF1:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: metal.barrier
// CHECK: metal.store %{{.*}}, %[[BUF0]]
// CHECK: metal.store %{{.*}}, %[[BUF1]]
// CHECK: metal.barrier
// The active lanes of a level are the multiples of 2*stride, and the strides
// ASCEND -- 1 then 2 then 4 -- so slot 0 accumulates in logical order.
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: arith.andi %{{.*}}, %[[C1]]
// CHECK: scf.if
// CHECK:   arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:   metal.get_element %[[BUF0]]
// CHECK:   metal.get_element %[[BUF0]]
// CHECK:   metal.get_element %[[BUF1]]
// CHECK:   metal.get_element %[[BUF1]]
// Both halves of the region are replayed, in the region's own order.
// CHECK:   arith.maxnumf
// CHECK:   arith.maxnumf
// CHECK:   metal.store %{{.*}}, %[[BUF0]]
// CHECK:   metal.store %{{.*}}, %[[BUF1]]
// CHECK: metal.barrier
// CHECK: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: %[[M3:.*]] = arith.constant 3 : i32
// CHECK: arith.andi %{{.*}}, %[[M3]]
// BLOCK == tpb here, so every partner slot holds a live element and no
// liveness guard is emitted at all.
// CHECK-NOT: arith.cmpi slt

// -----

// A hand-written argmax that breaks ties towards the HIGHER index. It is not
// the canonical argmax (whose ties go to the lower index) and must not be
// claimed by it; the general path replays this region verbatim. BLOCK=32 over
// tpb=128 also exercises the partner-liveness guard, which is what replaces
// the identity padding a general combine cannot have.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_tuple_argmax_high_tie(%x_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked>
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %r:2 = "tt.reduce"(%x, %idx) ({
    ^bb0(%lhs_value: f32, %lhs_index: i32, %rhs_value: f32, %rhs_index: i32):
      %equal = arith.cmpf oeq, %lhs_value, %rhs_value : f32
      %higher_index = arith.cmpi sgt, %lhs_index, %rhs_index : i32
      %equal_and_higher = arith.andi %equal, %higher_index : i1
      %greater = arith.cmpf ogt, %lhs_value, %rhs_value : f32
      %take_left = arith.ori %greater, %equal_and_higher : i1
      %value = arith.select %take_left, %lhs_value, %rhs_value : f32
      %index = arith.select %take_left, %lhs_index, %rhs_index : i32
      tt.reduce.return %value, %index : f32, i32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked>, tensor<32xi32, #blocked>) -> (f32, i32)
    tt.store %o_ptr, %r#1 : !tt.ptr<i32>
    tt.return
  }
}

// The i32 slot rides a ui32 buffer -- i32 has no signless memref spelling --
// and is bridged back before the region sees it, so the region's signed
// comparison stays signed.
// CHECK-LABEL: metal.kernel reduce_tuple_argmax_high_tie
// CHECK: %[[VBUF:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: %[[IBUF:.*]] = metal.threadgroup_alloca : !metal.memref<128 x ui32>
// CHECK: metal.barrier
// CHECK: metal.store %{{.*}}, %[[VBUF]]
// CHECK: metal.store %{{.*}}, %[[IBUF]]
// CHECK: metal.barrier
// CHECK: scf.if
// CHECK:   %[[PARTNER:.*]] = arith.addi
// CHECK:   metal.get_element %[[VBUF]]
// CHECK:   metal.get_element %[[IBUF]]
// CHECK:   builtin.unrealized_conversion_cast %{{.*}} : ui32 to i32
// The region, replayed in its own order: ties to the higher index.
// CHECK:   arith.cmpf oeq
// CHECK:   arith.cmpi sgt
// CHECK:   arith.andi
// CHECK:   arith.cmpf ogt
// CHECK:   arith.ori
// CHECK:   arith.select
// CHECK:   arith.select
// A partner past the end of the block contributes nothing: this lane keeps
// what it already held. There is no identity to pad the tail with.
// CHECK:   %[[LIVE:.*]] = arith.cmpi slt, %[[PARTNER]], %{{.*}} : i32
// CHECK:   arith.select %[[LIVE]]
// CHECK:   metal.store %{{.*}}, %[[VBUF]]
// CHECK:   arith.select %[[LIVE]]
// CHECK:   metal.store %{{.*}}, %[[IBUF]]

// -----

// More than one element per thread: a thread folds its OWN contiguous band
// first, and the cross-thread tree then runs unchanged over the partials.
// Thread t takes [t*E, (t+1)*E), so merging threads in order merges positions
// in order -- which is what an associative-but-not-commutative combine needs,
// and what the interleaved mapping would not give.
//
// The band fold is an `scf.for` rather than an unrolled sequence, so the
// combine region is cloned ONCE however many elements a thread owns.
//
// This is the geometry `tl.argmax` over an ordinary row has: the canonical pair
// lowering owns one slot per thread, so every block bigger than the threadgroup
// used to be refused.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_tuple_band_fold(%x: !tt.ptr<f32>, %ov: !tt.ptr<f32>, %oi: !tt.ptr<i32>) {
    %o = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %sp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %o : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %ap : tensor<128x!tt.ptr<f32>, #blocked>
    %r:2 = "tt.reduce"(%v, %o) ({
    ^bb0(%a1: f32, %a2: i32, %b1: f32, %b2: i32):
      %gt = arith.cmpf ogt, %a1, %b1 : f32
      %mv = arith.select %gt, %a1, %b1 : f32
      %mi = arith.select %gt, %a2, %b2 : i32
      tt.reduce.return %mv, %mi : f32, i32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>, tensor<128xi32, #blocked>) -> (f32, i32)
    tt.store %ov, %r#0 : !tt.ptr<f32>
    tt.store %oi, %r#1 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel reduce_tuple_band_fold
// One slot per THREAD, not per position -- the band fold is what keeps the
// scratch at that size.
// CHECK: %[[VBUF:.*]] = metal.threadgroup_alloca : !metal.memref<32 x f32>
// CHECK: %[[IBUF:.*]] = metal.threadgroup_alloca : !metal.memref<32 x ui32>
// CHECK: %[[E:.*]] = arith.constant 4 : i32
// CHECK: %[[BASE:.*]] = arith.muli %{{.*}}, %[[E]] : i32
// The band's first element seeds the accumulator; there is no identity for a
// general combine to start from.
// CHECK: %[[SEED:.*]] = metal.get_element
// CHECK: %[[FOLD:.*]]:2 = scf.for %[[K:.*]] = %{{.*}} to %[[E]] step %{{.*}} iter_args(%[[ACCV:.*]] = %[[SEED]], %[[ACCI:.*]] = %[[BASE]])
// CHECK:   %[[POS:.*]] = arith.addi %[[BASE]], %[[K]] : i32
// CHECK:   %[[ELT:.*]] = metal.get_element
// The accumulated prefix is the LOWER range, so it is the left operand.
// CHECK:   %[[GT:.*]] = arith.cmpf ogt, %[[ACCV]], %[[ELT]] : f32
// CHECK:   arith.select %[[GT]], %[[ACCV]], %[[ELT]] : f32
// CHECK:   arith.select %[[GT]], %[[ACCI]], %[[POS]] : i32
// CHECK: metal.barrier
// CHECK: metal.store %[[FOLD]]#0, %[[VBUF]]
// CHECK: metal.store %{{.*}}, %[[IBUF]]
// CHECK: metal.barrier
// and then the ordinary cross-thread tree over the partials.
// CHECK: arith.andi
// CHECK: scf.if
// CHECK:   arith.cmpf ogt
