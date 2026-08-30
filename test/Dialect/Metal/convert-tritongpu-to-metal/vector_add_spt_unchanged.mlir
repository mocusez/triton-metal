// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Principle-2 tripwire: locks the lowered IR shape for a non-reduce kernel
// with sizePerThread = [8] (spt8, unmasked, BLOCK=1024, 128 threads).
//
// The Synthesis-path Phase C makes ZERO changes to LoadLowering/TileInfo/
// tileFromTensor — so the lowered output for this kernel must be equivalent
// to pre-Phase-C. Any regression here means a future change touched the load
// surface and broke the invariant.
//
// Structural invariants locked by this fixture:
//   (I1) NO metal.threadgroup_alloca  — non-reduce path never stages via TG.
//   (I2) NO metal.barrier             — no synchronisation barrier emitted.
//   (I3) NO metal.tg_store_indexed / metal.tg_load_indexed — TG store/load
//        ops appear only in the reduce path.
//   (I4) Exactly one metal.get_element per operand per loop iteration —
//        the reduce path emits multiple get_element per (tid, iv) for
//        butterfly fan-out; this kernel must not.
//   (I5) scf.for trip count == 8 (BLOCK 1024 / threadsPerCTA 128).
//   (I6) arith.addf result feeds metal.binary_exp addOp (not a reduce op).
//   (I7) metal.store is unconditional — no scf.if wrapping the store.
//
// Canary: corresponds to vector_add_matrix_spt8 in the pytest inventory.
// See audit at the implementation notes:88.

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_spt8_unmasked(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024_i32 : i32
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %bs_splat = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %abs_off = arith.addi %bs_splat, %offsets : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_val = tt.load %y_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %sum = arith.addf %x_val, %y_val : tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %abs_off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %sum : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel add_kernel_spt8_unmasked
//
// I1 + I2 + I3: non-reduce path emits none of these.
// CHECK-NOT: metal.threadgroup_alloca
// CHECK-NOT: metal.barrier
// CHECK-NOT: metal.tg_store_indexed
// CHECK-NOT: metal.tg_load_indexed
//
// I5: tile loop with trip count 8 (1024 / 128 threads).
// CHECK: arith.constant 0 : i32
// CHECK: arith.constant 8 : i32
// CHECK: arith.constant 1 : i32
// CHECK: scf.for
//
// I4 + I6: exactly one get_element per operand, then addOp — NOT a reduce op.
// CHECK: metal.get_element %arg0
// CHECK: metal.get_element %arg1
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
//
// I7: unconditional store (no scf.if wrapping it).
// CHECK-NOT: scf.if
// CHECK: metal.store
// CHECK: metal.return
