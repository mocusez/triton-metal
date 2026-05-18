// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Matmul Track Sessions 3a+3b: tt.dot preprocessing rewrites the (load A,
// load B, dot, store) chain into (metal.simdgroup_load × 3,
// metal.simdgroup_multiply_accumulate, metal.simdgroup_store), and the
// MSL emitter renders the chain end-to-end with the canonical Metal
// function-call substrings. Single-K-iter 8×8×8.
// See `.omc/specs/deep-interview-metal-matmul-session3-tt-dot-wiring.md`
// and `.omc/specs/deep-interview-metal-matmul-session3b-msl-end-to-end.md`.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_8x8x8(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
    // Build 8x8 ptr tensors via splat + addptr. The dot preprocessing
    // walks back through addptr/splat to find the kernel-arg ptrs.
    %offsA = arith.constant dense<0> : tensor<8x8xi32, #dotA>
    %offsB = arith.constant dense<0> : tensor<8x8xi32, #dotB>
    %offsC = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %a_splat = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotA>
    %a_addr = tt.addptr %a_splat, %offsA : tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8xi32, #dotA>
    %b_splat = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotB>
    %b_addr = tt.addptr %b_splat, %offsB : tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xi32, #dotB>
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

// The new pre-pass replaces the (load, load, dot, store) chain with three
// simdgroup_load ops (one each for A, B, and the C-init accumulator),
// one multiply_accumulate, and one simdgroup_store.
// CHECK: metal.module
// CHECK: metal.kernel matmul_8x8x8
// CHECK-COUNT-3: metal.simdgroup_load
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
// CHECK: metal.return

// MSL end-to-end emission: the canonical Apple SIMD-group function-call
// substrings appear in the rendered kernel.
// MSL: kernel void matmul_8x8x8
// MSL-COUNT-3: simdgroup_load_matrix
// MSL: simdgroup_matrix_multiply_accumulate
// MSL: simdgroup_store_matrix
// MSL: return
