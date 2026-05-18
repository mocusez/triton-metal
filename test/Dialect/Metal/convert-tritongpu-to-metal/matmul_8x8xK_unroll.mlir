// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=PASS
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Matmul Track Session 4+4b: K-loop unroll with memory-loaded C-init.
// The pre-pass detects a tt.dot nested inside scf.for (with single
// accumulator iter_arg, static trip count) and UNROLLS it into N
// consecutive simdgroup_multiply_accumulate ops. K=16 with BLOCK_K=8 →
// trip count N=2. Session 4b accepts `tt.load %c_addr` as the
// iter_arg init (alongside dense<0.0>) so long as the loaded ptr
// matches the store ptr.
// See `.omc/specs/deep-interview-metal-matmul-session4-kloop-unroll.md`
// and `.omc/specs/deep-interview-metal-matmul-session4b-memory-c-init.md`.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_8x8xK_unroll(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
    %c0_idx = arith.constant 0 : index
    %c2_idx = arith.constant 2 : index   // N = (2 - 0) / 1 = 2 iters
    %c1_idx = arith.constant 1 : index

    %offsA = arith.constant dense<0> : tensor<8x8xi32, #dotA>
    %offsB = arith.constant dense<0> : tensor<8x8xi32, #dotB>
    %offsC = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %a_splat = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotA>
    %a_addr = tt.addptr %a_splat, %offsA : tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8xi32, #dotA>
    %b_splat = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotB>
    %b_addr = tt.addptr %b_splat, %offsB : tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xi32, #dotB>
    %c_splat = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_splat, %offsC : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>

    // Session 4b: iter_arg init is `tt.load %c_addr` (same buffer as the final store).
    %c_init = tt.load %c_addr : tensor<8x8x!tt.ptr<f32>, #blocked>

    %acc = scf.for %k = %c0_idx to %c2_idx step %c1_idx iter_args(%acc_iv = %c_init) -> (tensor<8x8xf32, #blocked>) {
      %a = tt.load %a_addr : tensor<8x8x!tt.ptr<f32>, #dotA>
      %b = tt.load %b_addr : tensor<8x8x!tt.ptr<f32>, #dotB>
      %new_acc = tt.dot %a, %b, %acc_iv : tensor<8x8xf32, #dotA> * tensor<8x8xf32, #dotB> -> tensor<8x8xf32, #blocked>
      scf.yield %new_acc : tensor<8x8xf32, #blocked>
    }
    tt.store %c_addr, %acc : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// IR-level: scf.for is erased; the unrolled chain is
// (C-init load) (A,B load + MA) × 2 → store.
// PASS: metal.module
// PASS: metal.kernel matmul_8x8xK_unroll
// PASS-NOT: scf.for
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_multiply_accumulate
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_multiply_accumulate
// PASS: metal.simdgroup_store
// PASS: metal.return

// MSL end-to-end: same shape in the emitted text.
// MSL: kernel void matmul_8x8xK_unroll
// MSL: simdgroup_load_matrix
// MSL: simdgroup_load_matrix
// MSL: simdgroup_load_matrix
// MSL: simdgroup_matrix_multiply_accumulate
// MSL: simdgroup_load_matrix
// MSL: simdgroup_load_matrix
// MSL: simdgroup_matrix_multiply_accumulate
// MSL: simdgroup_store_matrix
// MSL: return
