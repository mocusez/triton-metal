// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=PASS
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Matmul Track Session 4c-1: stride extraction. The pre-pass walks the
// addptr chain (addptr → broadcast → muli → tt.splat) to find the
// kernel-arg i32 used as stride. The emitted MSL uses that kernel-arg
// scalar (materialized as `vN[0]` via FuncOpLowering's wrap-as-memref
// prologue) instead of literal `8`.
// See `.omc/specs/deep-interview-metal-matmul-session4c-stride-extraction.md`.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_real_stride(
      %a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>,
      %stride_am: i32, %stride_bn: i32) {
    // A's offset chain:
    //   row_offs_2d : tensor<8x1xi32, #dotA>      // arbitrary dense<0> for v1
    //   stride_a_splat = tt.splat %stride_am      // <-- the kernel-arg stride
    //   rows = arith.muli row_offs_2d, stride_a_splat
    //   rows_b = tt.broadcast rows                : tensor<8x8>
    //   a_addr = tt.addptr (tt.splat a_ptr), rows_b
    %row_offs_2d_A = arith.constant dense<0> : tensor<8x1xi32, #dotA>
    %stride_a_splat = tt.splat %stride_am : i32 -> tensor<8x1xi32, #dotA>
    %rows_A = arith.muli %row_offs_2d_A, %stride_a_splat : tensor<8x1xi32, #dotA>
    %rows_A_b = tt.broadcast %rows_A : tensor<8x1xi32, #dotA> -> tensor<8x8xi32, #dotA>
    %a_splat = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotA>
    %a_addr = tt.addptr %a_splat, %rows_A_b : tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8xi32, #dotA>

    // B's offset chain mirrors A's with stride_bn.
    %row_offs_2d_B = arith.constant dense<0> : tensor<8x1xi32, #dotB>
    %stride_b_splat = tt.splat %stride_bn : i32 -> tensor<8x1xi32, #dotB>
    %rows_B = arith.muli %row_offs_2d_B, %stride_b_splat : tensor<8x1xi32, #dotB>
    %rows_B_b = tt.broadcast %rows_B : tensor<8x1xi32, #dotB> -> tensor<8x8xi32, #dotB>
    %b_splat = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotB>
    %b_addr = tt.addptr %b_splat, %rows_B_b : tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xi32, #dotB>

    // C address (hardcoded stride per session 4c-1 non-goal).
    %offsC = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %c_splat = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_splat, %offsC : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>

    %a = tt.load %a_addr : tensor<8x8x!tt.ptr<f32>, #dotA>
    %b = tt.load %b_addr : tensor<8x8x!tt.ptr<f32>, #dotB>
    %c0 = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c = tt.dot %a, %b, %c0 : tensor<8x8xf32, #dotA> * tensor<8x8xf32, #dotB> -> tensor<8x8xf32, #blocked>
    tt.store %c_addr, %c : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// PASS: metal.module
// PASS: metal.kernel matmul_real_stride
// PASS-COUNT-3: metal.simdgroup_load
// PASS: metal.simdgroup_multiply_accumulate
// PASS: metal.simdgroup_store

// MSL end-to-end: the kernel-arg stride scalars `%stride_am`, `%stride_bn`
// materialize as `vN[0]` dereferences. At least one simdgroup_load_matrix
// call should use a dereference (i.e., contain `[0]`) as its stride arg
// rather than literal `8`.
// MSL: kernel void matmul_real_stride
// MSL: simdgroup_load_matrix
// MSL-SAME: [0]
// MSL: simdgroup_load_matrix
// MSL-SAME: [0]
// MSL: simdgroup_load_matrix
// MSL: simdgroup_matrix_multiply_accumulate
// MSL: simdgroup_store_matrix
