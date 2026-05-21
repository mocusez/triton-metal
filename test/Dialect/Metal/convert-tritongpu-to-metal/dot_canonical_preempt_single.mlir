// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// L1d3 preempt — single-dot at function top-level (no scf.for).
// Path 1 of `preprocessDotCvtChains` / `rewriteSingleDot` dispatcher.
//
// The cvts (#blocked -> #dot_op) feeding tt.dot's A and B operands are
// stripped by `preprocessDotCvtChains`, then `rewriteSingleDot` matches
// the canonical body shape and rewrites to `metal.simdgroup_*`.
//
// See `.omc/specs/deep-interview-l1d3-matmul-convert-layout-preempt.md`
// and `.omc/plans/l1d3-matmul-convert-layout-preempt-consensus.md`.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_preempt_single(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
    %offsA = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %offsB = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %offsC = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %a_splat = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %a_addr = tt.addptr %a_splat, %offsA : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    %b_splat = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %b_addr = tt.addptr %b_splat, %offsB : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    %c_splat = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_splat, %offsC : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>

    // Loads in #blocked encoding (real Triton-emitted shape).
    %a_blk = tt.load %a_addr : tensor<8x8x!tt.ptr<f32>, #blocked>
    %b_blk = tt.load %b_addr : tensor<8x8x!tt.ptr<f32>, #blocked>
    // Cvts to #dot_op — preempt eliminates them.
    %a = ttg.convert_layout %a_blk : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #dotA>
    %b = ttg.convert_layout %b_blk : tensor<8x8xf32, #blocked> -> tensor<8x8xf32, #dotB>
    %c0 = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c = tt.dot %a, %b, %c0 : tensor<8x8xf32, #dotA> * tensor<8x8xf32, #dotB> -> tensor<8x8xf32, #blocked>
    tt.store %c_addr, %c : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// Post-pass: simdgroup chain emitted, all dot-feeding cvts eliminated.
// dense<0.0> C-init becomes simdgroup_matrix_zero; A/B become device-staged.
// CHECK-LABEL: metal.kernel dot_preempt_single
// CHECK-COUNT-2: metal.simdgroup_load_device_staged
// CHECK: metal.simdgroup_matrix_zero
// CHECK: metal.simdgroup_multiply_accumulate
// CHECK: metal.simdgroup_store
// CHECK: metal.return
// CHECK-NOT: ttg.convert_layout
