// RUN: split-file %s %t
// RUN: triton-metal-opt --convert-tritongpu-to-metal %t/supported.mlir | FileCheck %s
// RUN: not triton-metal-opt --convert-tritongpu-to-metal %t/unsupported_scalar.mlir 2>&1 | FileCheck %s --check-prefix=BAD
//
// `tl.gather` along axis 0. A gather is an arbitrary cross-thread read — thread
// i's result lives wherever idx[i] points, i.e. in some other thread's element —
// so the tile has to pass through the threadgroup buffer, which is the only
// place all of it is visible at once:
//
//   tg_store_indexed(buf, pos)   for each owned position
//   barrier
//   tg_load_indexed(buf, idx)
//
// The read is `tg_load_indexed`, not `get_element`: it force-materializes as a
// named let-binding at its IR position, so the load cannot be re-inlined past a
// later barrier (the L1d2b inline-barrier contract).

//--- supported.mlir
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked_loop = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gather_f32(%x: !tt.ptr<f32>, %ix: !tt.ptr<i32>, %o: !tt.ptr<f32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %xp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %xa = tt.addptr %xp, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %xa : tensor<128x!tt.ptr<f32>, #blocked>
    %ixp = tt.splat %ix : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %ixa = tt.addptr %ixp, %r : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %i = tt.load %ixa : tensor<128x!tt.ptr<i32>, #blocked>
    %g = tt.gather %v[%i] {axis = 0 : i32} : (tensor<128xf32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128xf32, #blocked>
    %op = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %oa = tt.addptr %op, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %oa, %g : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @gather_loop_i32(%x: !tt.ptr<i32>, %ix: !tt.ptr<i32>, %o: !tt.ptr<i32>, %steps: i32) {
    %c1024 = arith.constant 1024 : i32
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %three = arith.constant dense<3> : tensor<1024xi32, #blocked_loop>
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked_loop>
    %xp = tt.splat %x : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked_loop>
    %xa = tt.addptr %xp, %r : tensor<1024x!tt.ptr<i32>, #blocked_loop>, tensor<1024xi32, #blocked_loop>
    %initial = tt.load %xa : tensor<1024x!tt.ptr<i32>, #blocked_loop>
    %result = scf.for %step = %c0 to %steps step %c1 iter_args(%state = %initial) -> (tensor<1024xi32, #blocked_loop>)  : i32 {
      %step_offset = arith.muli %step, %c1024 : i32
      %step_base = tt.addptr %ix, %step_offset : !tt.ptr<i32>, i32
      %ip = tt.splat %step_base : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked_loop>
      %ia = tt.addptr %ip, %r : tensor<1024x!tt.ptr<i32>, #blocked_loop>, tensor<1024xi32, #blocked_loop>
      %indices = tt.load %ia : tensor<1024x!tt.ptr<i32>, #blocked_loop>
      %gathered = tt.gather %state[%indices] {axis = 0 : i32} : (tensor<1024xi32, #blocked_loop>, tensor<1024xi32, #blocked_loop>) -> tensor<1024xi32, #blocked_loop>
      %next = arith.addi %gathered, %three : tensor<1024xi32, #blocked_loop>
      scf.yield %next : tensor<1024xi32, #blocked_loop>
    }
    %op = tt.splat %o : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked_loop>
    %oa = tt.addptr %op, %r : tensor<1024x!tt.ptr<i32>, #blocked_loop>, tensor<1024xi32, #blocked_loop>
    tt.store %oa, %result : tensor<1024x!tt.ptr<i32>, #blocked_loop>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel gather_f32
// BLOCK == tpb == 128, so one buffer slot per thread and one staged write.
// CHECK: metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: metal.tg_store_indexed
// The barrier between the fill and the read is what makes the gather correct;
// without it a thread can read a slot its owner has not written yet.
// CHECK: metal.barrier
// An out-of-range index would be an actual out-of-bounds threadgroup access,
// so the index is clamped into [0, BLOCK) rather than left undefined.
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: metal.tg_load_indexed
// CHECK: metal.return

// CHECK-LABEL: metal.kernel gather_loop_i32
// Current and next state are distinct so every recurrence trip reads a stable
// full tile while publishing its successor.
// CHECK: metal.threadgroup_alloca : !metal.memref<1024 x ui32>
// CHECK: metal.threadgroup_alloca : !metal.memref<1024 x ui32>
// CHECK: metal.tg_store_indexed
// CHECK: metal.barrier
// The runtime recurrence is emitted before the fixed eight-band function loop.
// CHECK: scf.for
// CHECK: metal.get_element
// CHECK: metal.tg_store_indexed
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed
// CHECK: metal.tg_store_indexed
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: metal.tg_load_indexed
// CHECK: metal.store
// CHECK: metal.return

// The loop-replacement prepass must reject scalar cones it cannot replay.
// Even though `%abs_step` precedes the original loop, the generated state loop
// is hoisted to the function prologue and must not reuse that non-dominating
// SSA value.
//
//--- unsupported_scalar.mlir
#blocked_loop = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gather_loop_unsupported_scalar(%x: !tt.ptr<i32>, %ix: !tt.ptr<i32>, %o: !tt.ptr<i32>, %steps: i32) {
    %c1024 = arith.constant 1024 : i32
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %three = arith.constant dense<3> : tensor<1024xi32, #blocked_loop>
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked_loop>
    %xp = tt.splat %x : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked_loop>
    %xa = tt.addptr %xp, %r : tensor<1024x!tt.ptr<i32>, #blocked_loop>, tensor<1024xi32, #blocked_loop>
    %initial = tt.load %xa : tensor<1024x!tt.ptr<i32>, #blocked_loop>
    %delta = arith.subi %steps, %c1 : i32
    %abs_step = math.absi %delta : i32
    %result = scf.for %step = %c0 to %steps step %c1 iter_args(%state = %initial) -> (tensor<1024xi32, #blocked_loop>)  : i32 {
      %step_offset = arith.muli %abs_step, %c1024 : i32
      %step_base = tt.addptr %ix, %step_offset : !tt.ptr<i32>, i32
      %ip = tt.splat %step_base : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked_loop>
      %ia = tt.addptr %ip, %r : tensor<1024x!tt.ptr<i32>, #blocked_loop>, tensor<1024xi32, #blocked_loop>
      %indices = tt.load %ia : tensor<1024x!tt.ptr<i32>, #blocked_loop>
      %gathered = tt.gather %state[%indices] {axis = 0 : i32} : (tensor<1024xi32, #blocked_loop>, tensor<1024xi32, #blocked_loop>) -> tensor<1024xi32, #blocked_loop>
      %next = arith.addi %gathered, %three : tensor<1024xi32, #blocked_loop>
      scf.yield %next : tensor<1024xi32, #blocked_loop>
    }
    %op = tt.splat %o : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked_loop>
    %oa = tt.addptr %op, %r : tensor<1024x!tt.ptr<i32>, #blocked_loop>, tensor<1024xi32, #blocked_loop>
    tt.store %oa, %result : tensor<1024x!tt.ptr<i32>, #blocked_loop>
    tt.return
  }
}

// BAD: loop-carried tl.gather recurrence contains a scalar or tensor cone that cannot be replayed safely
