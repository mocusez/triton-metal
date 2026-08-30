// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// A tensor atomic whose address folded down to a bare `tt.splat` of the kernel
// argument. `tl.atomic_add(out + offs * 0, v)` — and every BLOCK=1 tile — leaves
// no `tt.addptr` for the pattern to read an index from, and the tensor branch
// used to require one. An unmatched op in this backend is a PROCESS KILL during
// rollback rather than an error: this shape took the caller down on 7 of 8 runs,
// SIGSEGV or abort depending on the run.
//
// Unlike a STORE through a uniform pointer, this is not a race and so is
// lowered rather than rejected: every lane read-modify-writes the same cell
// atomically, which is precisely what the shape asks for (fold a whole tile
// into one accumulator). The index degenerates to the scalar offset below the
// splat.
//
// CHECK-LABEL: metal.kernel splat_atomic_add_f32
// The address is a plain constant — one cell for the whole tile, so no div/rem
// of the thread id appears in the atomic's index...
// CHECK: %[[OFF:.*]] = arith.constant 0 : i32
// CHECK: %[[OFFU:.*]] = builtin.unrealized_conversion_cast %[[OFF]] : i32 to ui32
// ...while the VALUE stays per-lane, so all 128 elements are accumulated.
// CHECK: metal.atomic_rmw Add %{{.*}}, %arg1[%[[OFFU]]]
// CHECK-NOT: scf.if

// A tile smaller than the threadgroup must keep its lane guard, or the lanes
// past the tile would each contribute a duplicate of element 0.
// CHECK-LABEL: metal.kernel splat_atomic_add_sub_tpb
// CHECK: %[[N:.*]] = arith.constant 16 : i32
// CHECK: %[[OK:.*]] = arith.cmpi slt, %{{.*}}, %[[N]]
// CHECK: scf.if %[[OK]]
// CHECK: metal.atomic_rmw Add

// A mask is the other way this shape reaches the backend — `offs * 0` folds the
// address while the mask leaves one contributor.
// CHECK-LABEL: metal.kernel splat_atomic_add_masked
// CHECK: scf.if
// CHECK: metal.atomic_rmw Add

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @splat_atomic_add_f32(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %px = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %pxo = tt.addptr %px, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %pxo : tensor<128x!tt.ptr<f32>, #blocked>
    %po = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %old = tt.atomic_rmw fadd, acq_rel, gpu, %po, %v : (tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.return
  }

  tt.func public @splat_atomic_add_sub_tpb(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %px = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %pxo = tt.addptr %px, %r : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %v = tt.load %pxo : tensor<16x!tt.ptr<f32>, #blocked>
    %po = tt.splat %o : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %old = tt.atomic_rmw fadd, acq_rel, gpu, %po, %v : (tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xf32, #blocked>) -> tensor<16xf32, #blocked>
    tt.return
  }

  tt.func public @splat_atomic_add_masked(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
    %c0 = arith.constant dense<0> : tensor<128xi32, #blocked>
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %px = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %pxo = tt.addptr %px, %r : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %v = tt.load %pxo : tensor<128x!tt.ptr<f32>, #blocked>
    %mask = arith.cmpi eq, %r, %c0 : tensor<128xi32, #blocked>
    %po = tt.splat %o : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %old = tt.atomic_rmw fadd, acq_rel, gpu, %po, %v, %mask : (tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xf32, #blocked>, tensor<128xi1, #blocked>) -> tensor<128xf32, #blocked>
    tt.return
  }
}
