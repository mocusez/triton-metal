// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=PASS
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Matmul Track Session 4c-3: canonical 3-iter_arg Triton matmul shape.
// scf.for has iter_args(a_ptrs, b_ptrs, acc); body is strictly
// [load A, load B, tt.dot, addptr A, addptr B, yield]. The pre-pass
// unrolls into N consecutive MA ops where iter i uses K-axis origin
// = i * BK on A's col and B's row. Base row origin for A and base col
// origin for B come from the iter_arg INIT pointer expressions.
// See `.omc/specs/deep-interview-metal-matmul-session4c3-per-iter-evolution.md`.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_canonical(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
    %c0_idx = arith.constant 0 : index
    %c2_idx = arith.constant 2 : index   // N = 2 iters (K=16, BK=8)
    %c1_idx = arith.constant 1 : index
    %zeros = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>

    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %cBM = arith.constant 8 : i32
    %cBN = arith.constant 8 : i32
    %base_m = arith.muli %pid_m, %cBM : i32
    %base_n = arith.muli %pid_n, %cBN : i32

    // A init: addptr(splat(a_ptr), offset_with_pid_m_origin)
    %rm_a = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotA}>>
    %base_m_splat_a = tt.splat %base_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotA}>>
    %offs_m_a = arith.addi %base_m_splat_a, %rm_a : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotA}>>
    %offs_m_a_2d = tt.expand_dims %offs_m_a {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotA}>> -> tensor<8x1xi32, #dotA>
    %offs_m_a_b = tt.broadcast %offs_m_a_2d : tensor<8x1xi32, #dotA> -> tensor<8x8xi32, #dotA>
    %a_splat = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotA>
    %a_ptrs_init = tt.addptr %a_splat, %offs_m_a_b : tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8xi32, #dotA>

    // B init: addptr(splat(b_ptr), offset_with_pid_n_origin)
    %rn_b = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotB}>>
    %base_n_splat_b = tt.splat %base_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotB}>>
    %offs_n_b = arith.addi %base_n_splat_b, %rn_b : tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotB}>>
    %offs_n_b_2d = tt.expand_dims %offs_n_b {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotB}>> -> tensor<1x8xi32, #dotB>
    %offs_n_b_b = tt.broadcast %offs_n_b_2d : tensor<1x8xi32, #dotB> -> tensor<8x8xi32, #dotB>
    %b_splat = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotB>
    %b_ptrs_init = tt.addptr %b_splat, %offs_n_b_b : tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xi32, #dotB>

    // BK bump tensors (splat of constant 8) — per-iter advance along K.
    %bk = arith.constant 8 : i32
    %a_bump = tt.splat %bk : i32 -> tensor<8x8xi32, #dotA>
    %b_bump = tt.splat %bk : i32 -> tensor<8x8xi32, #dotB>

    // C address (single program-tile destination).
    %rm_c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %base_m_splat_c = tt.splat %base_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_c = arith.addi %base_m_splat_c, %rm_c : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_c_2d = tt.expand_dims %offs_m_c {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %offs_m_c_b = tt.broadcast %offs_m_c_2d : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %rn_c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %base_n_splat_c = tt.splat %base_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_c = arith.addi %base_n_splat_c, %rn_c : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_c_2d = tt.expand_dims %offs_n_c {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %offs_n_c_b = tt.broadcast %offs_n_c_2d : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %c_offs = arith.addi %offs_m_c_b, %offs_n_c_b : tensor<8x8xi32, #blocked>
    %c_splat = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_splat, %c_offs : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>

    // Canonical scf.for: 3 iter_args (a_ptrs, b_ptrs, acc).
    %for_results:3 = scf.for %k = %c0_idx to %c2_idx step %c1_idx
        iter_args(%a_iv = %a_ptrs_init, %b_iv = %b_ptrs_init, %acc_iv = %zeros)
        -> (tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xf32, #blocked>) {
      %a = tt.load %a_iv : tensor<8x8x!tt.ptr<f32>, #dotA>
      %b = tt.load %b_iv : tensor<8x8x!tt.ptr<f32>, #dotB>
      %new_acc = tt.dot %a, %b, %acc_iv : tensor<8x8xf32, #dotA> * tensor<8x8xf32, #dotB> -> tensor<8x8xf32, #blocked>
      %a_next = tt.addptr %a_iv, %a_bump : tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8xi32, #dotA>
      %b_next = tt.addptr %b_iv, %b_bump : tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xi32, #dotB>
      scf.yield %a_next, %b_next, %new_acc : tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xf32, #blocked>
    }
    tt.store %c_addr, %for_results#2 : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// IR-level: scf.for erased; 1 C-init + 2×(A,B) loads = 5 simdgroup_load;
// 2 MA ops; 1 store.
// PASS: metal.module
// PASS: metal.kernel matmul_canonical
// PASS-NOT: scf.for
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_multiply_accumulate
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_load
// PASS: metal.simdgroup_multiply_accumulate
// PASS: metal.simdgroup_store

// MSL end-to-end: iter 0 uses K-axis origin 0, iter 1 uses K-axis origin 8.
// MSL: kernel void matmul_canonical
// MSL: simdgroup_load_matrix
// MSL: simdgroup_load_matrix
// MSL-SAME: 0
// MSL: simdgroup_load_matrix
// MSL: simdgroup_matrix_multiply_accumulate
// MSL: simdgroup_load_matrix
// MSL-SAME: 8
// MSL: simdgroup_load_matrix
// MSL: simdgroup_matrix_multiply_accumulate
// MSL: simdgroup_store_matrix
