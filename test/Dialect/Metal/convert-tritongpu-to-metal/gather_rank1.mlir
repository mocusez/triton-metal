// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
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

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
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
