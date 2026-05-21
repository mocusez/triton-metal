// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --check-prefix=PASS
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL
//
// Matmul Track Session 4c-2: origin extraction. The pre-pass walks each
// load/store's tt.addptr.offset chain to find pid*BLOCK contributions for
// each axis, identified via tt.expand_dims' `axis` attribute. The emitted
// MSL uses tgid.x/tgid.y materializations (i.e. references that involve
// `tgid` dereferences) as origin operands instead of literal `0` for
// pid-driven origins.
// A: origin_row from pid_m * BM (axis 0). origin_col stays 0 (K-axis).
// B: origin_col from pid_n * BN (axis 1). origin_row stays 0 (K-axis).
// C: both origins extracted.
// See `.omc/specs/deep-interview-metal-matmul-session4c2-origin-extraction.md`.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_real_origin(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %cBM = arith.constant 8 : i32
    %cBN = arith.constant 8 : i32
    %base_m = arith.muli %pid_m, %cBM : i32
    %base_n = arith.muli %pid_n, %cBN : i32

    // A: row contribution has pid_m*BM origin (axis=0 varying, expand_dims axis=1)
    %rm_a = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotA}>>
    %base_m_splat_a = tt.splat %base_m : i32 -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotA}>>
    %offs_m_a = arith.addi %base_m_splat_a, %rm_a : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotA}>>
    %offs_m_a_2d = tt.expand_dims %offs_m_a {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotA}>> -> tensor<8x1xi32, #dotA>
    %offs_m_a_b = tt.broadcast %offs_m_a_2d : tensor<8x1xi32, #dotA> -> tensor<8x8xi32, #dotA>
    %rk_a = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotA}>>
    %rk_a_2d = tt.expand_dims %rk_a {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotA}>> -> tensor<1x8xi32, #dotA>
    %rk_a_b = tt.broadcast %rk_a_2d : tensor<1x8xi32, #dotA> -> tensor<8x8xi32, #dotA>
    %a_offs = arith.addi %offs_m_a_b, %rk_a_b : tensor<8x8xi32, #dotA>
    %a_splat = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotA>
    %a_addr = tt.addptr %a_splat, %a_offs : tensor<8x8x!tt.ptr<f32>, #dotA>, tensor<8x8xi32, #dotA>

    // B: col contribution has pid_n*BN origin (axis=1 varying, expand_dims axis=0)
    %rk_b_1d = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotB}>>
    %rk_b_2d = tt.expand_dims %rk_b_1d {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #dotB}>> -> tensor<8x1xi32, #dotB>
    %rk_b_b = tt.broadcast %rk_b_2d : tensor<8x1xi32, #dotB> -> tensor<8x8xi32, #dotB>
    %rn_b = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotB}>>
    %base_n_splat_b = tt.splat %base_n : i32 -> tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotB}>>
    %offs_n_b = arith.addi %base_n_splat_b, %rn_b : tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotB}>>
    %offs_n_b_2d = tt.expand_dims %offs_n_b {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #dotB}>> -> tensor<1x8xi32, #dotB>
    %offs_n_b_b = tt.broadcast %offs_n_b_2d : tensor<1x8xi32, #dotB> -> tensor<8x8xi32, #dotB>
    %b_offs = arith.addi %rk_b_b, %offs_n_b_b : tensor<8x8xi32, #dotB>
    %b_splat = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #dotB>
    %b_addr = tt.addptr %b_splat, %b_offs : tensor<8x8x!tt.ptr<f32>, #dotB>, tensor<8x8xi32, #dotB>

    // C: both pid_m*BM (row) and pid_n*BN (col) origins
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

    %a = tt.load %a_addr : tensor<8x8x!tt.ptr<f32>, #dotA>
    %b = tt.load %b_addr : tensor<8x8x!tt.ptr<f32>, #dotB>
    %c0 = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #blocked>
    %c = tt.dot %a, %b, %c0 : tensor<8x8xf32, #dotA> * tensor<8x8xf32, #dotB> -> tensor<8x8xf32, #blocked>
    tt.store %c_addr, %c : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// PASS: metal.module
// PASS: metal.kernel matmul_real_origin
// PASS-COUNT-2: metal.simdgroup_load_device_staged
// PASS: metal.simdgroup_matrix_zero
// PASS: metal.simdgroup_multiply_accumulate
// PASS: metal.simdgroup_store

// MSL end-to-end (iter-6): the MSL emission of tt.get_program_id
// materializes as `tgid.x` / `tgid.y`. With device-staged loads the
// `tgid` references appear inside each staging cooperative-copy loop
// (before the matching simdgroup_load) and inside the final simdgroup_store
// address expression. Device-staged A/B emit 2 `simdgroup_load(` from the
// shared threadgroup buffer; the dense<0.0> C-init is the zero-constructor
// and emits no load.
// MSL: kernel void matmul_real_origin
// MSL: tgid
// MSL: simdgroup_load(
// MSL: simdgroup_load(
// MSL: simdgroup_multiply_accumulate(
// MSL: simdgroup_store(
// MSL-SAME: tgid
