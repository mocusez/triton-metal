// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s --check-prefix=MSL

// Exact software tt.dot_scaled slices for Metal.  The matrix payloads cover
// native bf16/fp16 plus byte-backed E4M3FN/E5M2, while each K group carries an
// E8M0 byte scale.  Both A and B scales are present in the factor-32 cases so
// the regression pins the asymmetric scale indexing:
// A uses [row, k/32], B uses [column, k/32].  fastMath=false also requires raw
// scale 0xff to select NaN instead of being treated as the E8M0 infinity bit
// pattern.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#slice_row = #ttg.slice<{dim = 1, parent = #blocked}>
#slice_col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_scaled_e8m0_bf16(
      %a_ptr: !tt.ptr<bf16>, %b_ptr: !tt.ptr<bf16>,
      %a_scale_ptr: !tt.ptr<i8>, %b_scale_ptr: !tt.ptr<i8>,
      %c_ptr: !tt.ptr<f32>) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %n_as_row = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %k_as_col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_col>
    %k_as_row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_row>

    %m_2d = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %n_2d = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #slice_col> -> tensor<1x16xi32, #blocked>
    %k_a_2d = tt.expand_dims %k_as_col {axis = 0 : i32} : tensor<32xi32, #slice_col> -> tensor<1x32xi32, #blocked>
    %k_b_2d = tt.expand_dims %k_as_row {axis = 1 : i32} : tensor<32xi32, #slice_row> -> tensor<32x1xi32, #blocked>

    %c32 = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %a_row = arith.muli %m_2d, %c32 : tensor<16x1xi32, #blocked>
    %a_row_bc = tt.broadcast %a_row : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %k_a_bc = tt.broadcast %k_a_2d : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %a_off = arith.addi %a_row_bc, %k_a_bc : tensor<16x32xi32, #blocked>
    %a_base = tt.splat %a_ptr : !tt.ptr<bf16> -> tensor<16x32x!tt.ptr<bf16>, #blocked>
    %a_addr = tt.addptr %a_base, %a_off : tensor<16x32x!tt.ptr<bf16>, #blocked>, tensor<16x32xi32, #blocked>
    %a = tt.load %a_addr : tensor<16x32x!tt.ptr<bf16>, #blocked>

    %c16_k = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16_k : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<bf16> -> tensor<32x16x!tt.ptr<bf16>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<bf16>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<bf16>, #blocked>

    %c1_m = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %a_scale_off = arith.muli %m_2d, %c1_m : tensor<16x1xi32, #blocked>
    %a_scale_base = tt.splat %a_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %a_scale_addr = tt.addptr %a_scale_base, %a_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %a_scale = tt.load %a_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %n_scale_2d = tt.expand_dims %n_as_row {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %c1_n = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %b_scale_off = arith.muli %n_scale_2d, %c1_n : tensor<16x1xi32, #blocked>
    %b_scale_base = tt.splat %b_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %b_scale_addr = tt.addptr %b_scale_base, %b_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %b_scale = tt.load %b_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %result = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = bf16 rhs = bf16 {fastMath = false} : tensor<16x32xbf16, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xbf16, #blocked>, tensor<16x1xi8, #blocked> -> tensor<16x16xf32, #blocked>

    %c16_m = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %c_row = arith.muli %m_2d, %c16_m : tensor<16x1xi32, #blocked>
    %c_row_bc = tt.broadcast %c_row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_off = arith.addi %c_row_bc, %c_n_bc : tensor<16x16xi32, #blocked>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_base, %c_off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %c_addr, %result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // Adjacent exact case: fp16 compute, a single A-side scale, factor 16, and
  // fastMath=true (so no 0xff -> NaN select is emitted).
  tt.func public @dot_scaled_e8m0_fp16_factor16(
      %a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>,
      %a_scale_ptr: !tt.ptr<i8>, %c_ptr: !tt.ptr<f32>) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %k_col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_col>
    %k_row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_row>
    %scale_k = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32, #slice_col>
    %m_2d = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %n_2d = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #slice_col> -> tensor<1x16xi32, #blocked>
    %k_a_2d = tt.expand_dims %k_col {axis = 0 : i32} : tensor<32xi32, #slice_col> -> tensor<1x32xi32, #blocked>
    %k_b_2d = tt.expand_dims %k_row {axis = 1 : i32} : tensor<32xi32, #slice_row> -> tensor<32x1xi32, #blocked>

    %c32 = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %a_row = arith.muli %m_2d, %c32 : tensor<16x1xi32, #blocked>
    %a_row_bc = tt.broadcast %a_row : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %k_a_bc = tt.broadcast %k_a_2d : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %a_off = arith.addi %a_row_bc, %k_a_bc : tensor<16x32xi32, #blocked>
    %a_base = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<16x32x!tt.ptr<f16>, #blocked>
    %a_addr = tt.addptr %a_base, %a_off : tensor<16x32x!tt.ptr<f16>, #blocked>, tensor<16x32xi32, #blocked>
    %a = tt.load %a_addr : tensor<16x32x!tt.ptr<f16>, #blocked>

    %c16_k = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16_k : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<f16>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<f16>, #blocked>

    %scale_k_2d = tt.expand_dims %scale_k {axis = 0 : i32} : tensor<2xi32, #slice_col> -> tensor<1x2xi32, #blocked>
    %c2 = arith.constant dense<2> : tensor<16x1xi32, #blocked>
    %scale_row = arith.muli %m_2d, %c2 : tensor<16x1xi32, #blocked>
    %scale_row_bc = tt.broadcast %scale_row : tensor<16x1xi32, #blocked> -> tensor<16x2xi32, #blocked>
    %scale_k_bc = tt.broadcast %scale_k_2d : tensor<1x2xi32, #blocked> -> tensor<16x2xi32, #blocked>
    %scale_off = arith.addi %scale_row_bc, %scale_k_bc : tensor<16x2xi32, #blocked>
    %scale_base = tt.splat %a_scale_ptr : !tt.ptr<i8> -> tensor<16x2x!tt.ptr<i8>, #blocked>
    %scale_addr = tt.addptr %scale_base, %scale_off : tensor<16x2x!tt.ptr<i8>, #blocked>, tensor<16x2xi32, #blocked>
    %a_scale = tt.load %scale_addr : tensor<16x2x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %result = tt.dot_scaled %a scale %a_scale, %b, %acc lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x32xf16, #blocked>, tensor<16x2xi8, #blocked> * tensor<32x16xf16, #blocked> -> tensor<16x16xf32, #blocked>

    %c16_m = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %c_row = arith.muli %m_2d, %c16_m : tensor<16x1xi32, #blocked>
    %c_row_bc = tt.broadcast %c_row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_off = arith.addi %c_row_bc, %c_n_bc : tensor<16x16xi32, #blocked>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_base, %c_off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %c_addr, %result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // E5M2 payloads are byte-backed on Metal.  The exact software path widens
  // each raw byte through its bit-identical f16 representation to bf16 before
  // applying the E8M0 scale and accumulating in f32.
  tt.func public @dot_scaled_e8m0_e5m2(
      %a_ptr: !tt.ptr<f8E5M2>, %b_ptr: !tt.ptr<f8E5M2>,
      %a_scale_ptr: !tt.ptr<i8>, %b_scale_ptr: !tt.ptr<i8>,
      %c_ptr: !tt.ptr<f32>) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %n_as_row = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %k_as_col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_col>
    %k_as_row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_row>
    %m_2d = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %n_2d = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #slice_col> -> tensor<1x16xi32, #blocked>
    %k_a_2d = tt.expand_dims %k_as_col {axis = 0 : i32} : tensor<32xi32, #slice_col> -> tensor<1x32xi32, #blocked>
    %k_b_2d = tt.expand_dims %k_as_row {axis = 1 : i32} : tensor<32xi32, #slice_row> -> tensor<32x1xi32, #blocked>

    %c32 = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %a_row = arith.muli %m_2d, %c32 : tensor<16x1xi32, #blocked>
    %a_row_bc = tt.broadcast %a_row : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %k_a_bc = tt.broadcast %k_a_2d : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %a_off = arith.addi %a_row_bc, %k_a_bc : tensor<16x32xi32, #blocked>
    %a_base = tt.splat %a_ptr : !tt.ptr<f8E5M2> -> tensor<16x32x!tt.ptr<f8E5M2>, #blocked>
    %a_addr = tt.addptr %a_base, %a_off : tensor<16x32x!tt.ptr<f8E5M2>, #blocked>, tensor<16x32xi32, #blocked>
    %a = tt.load %a_addr : tensor<16x32x!tt.ptr<f8E5M2>, #blocked>

    %c16_k = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16_k : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<f8E5M2> -> tensor<32x16x!tt.ptr<f8E5M2>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<f8E5M2>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<f8E5M2>, #blocked>

    %c1_m = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %a_scale_off = arith.muli %m_2d, %c1_m : tensor<16x1xi32, #blocked>
    %a_scale_base = tt.splat %a_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %a_scale_addr = tt.addptr %a_scale_base, %a_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %a_scale = tt.load %a_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %n_scale_2d = tt.expand_dims %n_as_row {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %c1_n = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %b_scale_off = arith.muli %n_scale_2d, %c1_n : tensor<16x1xi32, #blocked>
    %b_scale_base = tt.splat %b_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %b_scale_addr = tt.addptr %b_scale_base, %b_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %b_scale = tt.load %b_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %result = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<16x32xf8E5M2, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xf8E5M2, #blocked>, tensor<16x1xi8, #blocked> -> tensor<16x16xf32, #blocked>

    %c16_m = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %c_row = arith.muli %m_2d, %c16_m : tensor<16x1xi32, #blocked>
    %c_row_bc = tt.broadcast %c_row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_off = arith.addi %c_row_bc, %c_n_bc : tensor<16x16xi32, #blocked>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_base, %c_off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %c_addr, %result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // Mixed E5M2/E4M3 payloads are still byte-backed per operand; the scalar
  // exact path must decode A and B independently before applying E8M0 scales.
  tt.func public @dot_scaled_e8m0_mixed_e5m2_e4m3(
      %a_ptr: !tt.ptr<f8E5M2>, %b_ptr: !tt.ptr<f8E4M3FN>,
      %a_scale_ptr: !tt.ptr<i8>, %b_scale_ptr: !tt.ptr<i8>,
      %c_ptr: !tt.ptr<f32>) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %n_as_row = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %k_as_col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_col>
    %k_as_row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_row>
    %m_2d = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %n_2d = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #slice_col> -> tensor<1x16xi32, #blocked>
    %k_a_2d = tt.expand_dims %k_as_col {axis = 0 : i32} : tensor<32xi32, #slice_col> -> tensor<1x32xi32, #blocked>
    %k_b_2d = tt.expand_dims %k_as_row {axis = 1 : i32} : tensor<32xi32, #slice_row> -> tensor<32x1xi32, #blocked>

    %c32 = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %a_row = arith.muli %m_2d, %c32 : tensor<16x1xi32, #blocked>
    %a_row_bc = tt.broadcast %a_row : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %k_a_bc = tt.broadcast %k_a_2d : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %a_off = arith.addi %a_row_bc, %k_a_bc : tensor<16x32xi32, #blocked>
    %a_base = tt.splat %a_ptr : !tt.ptr<f8E5M2> -> tensor<16x32x!tt.ptr<f8E5M2>, #blocked>
    %a_addr = tt.addptr %a_base, %a_off : tensor<16x32x!tt.ptr<f8E5M2>, #blocked>, tensor<16x32xi32, #blocked>
    %a = tt.load %a_addr : tensor<16x32x!tt.ptr<f8E5M2>, #blocked>

    %c16_k = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16_k : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<f8E4M3FN> -> tensor<32x16x!tt.ptr<f8E4M3FN>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<f8E4M3FN>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<f8E4M3FN>, #blocked>

    %c1_m = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %a_scale_off = arith.muli %m_2d, %c1_m : tensor<16x1xi32, #blocked>
    %a_scale_base = tt.splat %a_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %a_scale_addr = tt.addptr %a_scale_base, %a_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %a_scale = tt.load %a_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %n_scale_2d = tt.expand_dims %n_as_row {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %c1_n = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %b_scale_off = arith.muli %n_scale_2d, %c1_n : tensor<16x1xi32, #blocked>
    %b_scale_base = tt.splat %b_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %b_scale_addr = tt.addptr %b_scale_base, %b_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %b_scale = tt.load %b_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %result = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = e5m2 rhs = e4m3 {fastMath = false} : tensor<16x32xf8E5M2, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xf8E4M3FN, #blocked>, tensor<16x1xi8, #blocked> -> tensor<16x16xf32, #blocked>

    %c16_m = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %c_row = arith.muli %m_2d, %c16_m : tensor<16x1xi32, #blocked>
    %c_row_bc = tt.broadcast %c_row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_off = arith.addi %c_row_bc, %c_n_bc : tensor<16x16xi32, #blocked>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_base, %c_off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %c_addr, %result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // E4M3FN has no infinities: all finite payloads, including the extended
  // exponent encodings through +/-448, are decoded exactly to bf16.  The two
  // 0x7f-magnitude payloads preserve NaN class without promising payload bits.
  tt.func public @dot_scaled_e8m0_e4m3(
      %a_ptr: !tt.ptr<f8E4M3FN>, %b_ptr: !tt.ptr<f8E4M3FN>,
      %a_scale_ptr: !tt.ptr<i8>, %b_scale_ptr: !tt.ptr<i8>,
      %c_ptr: !tt.ptr<f32>) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %n_as_row = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %k_as_col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_col>
    %k_as_row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_row>
    %m_2d = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %n_2d = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #slice_col> -> tensor<1x16xi32, #blocked>
    %k_a_2d = tt.expand_dims %k_as_col {axis = 0 : i32} : tensor<32xi32, #slice_col> -> tensor<1x32xi32, #blocked>
    %k_b_2d = tt.expand_dims %k_as_row {axis = 1 : i32} : tensor<32xi32, #slice_row> -> tensor<32x1xi32, #blocked>

    %c32 = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %a_row = arith.muli %m_2d, %c32 : tensor<16x1xi32, #blocked>
    %a_row_bc = tt.broadcast %a_row : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %k_a_bc = tt.broadcast %k_a_2d : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %a_off = arith.addi %a_row_bc, %k_a_bc : tensor<16x32xi32, #blocked>
    %a_base = tt.splat %a_ptr : !tt.ptr<f8E4M3FN> -> tensor<16x32x!tt.ptr<f8E4M3FN>, #blocked>
    %a_addr = tt.addptr %a_base, %a_off : tensor<16x32x!tt.ptr<f8E4M3FN>, #blocked>, tensor<16x32xi32, #blocked>
    %a = tt.load %a_addr : tensor<16x32x!tt.ptr<f8E4M3FN>, #blocked>

    %c16_k = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16_k : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<f8E4M3FN> -> tensor<32x16x!tt.ptr<f8E4M3FN>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<f8E4M3FN>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<f8E4M3FN>, #blocked>

    %c1_m = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %a_scale_off = arith.muli %m_2d, %c1_m : tensor<16x1xi32, #blocked>
    %a_scale_base = tt.splat %a_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %a_scale_addr = tt.addptr %a_scale_base, %a_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %a_scale = tt.load %a_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %n_scale_2d = tt.expand_dims %n_as_row {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %c1_n = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %b_scale_off = arith.muli %n_scale_2d, %c1_n : tensor<16x1xi32, #blocked>
    %b_scale_base = tt.splat %b_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %b_scale_addr = tt.addptr %b_scale_base, %b_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %b_scale = tt.load %b_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %result = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<16x32xf8E4M3FN, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xf8E4M3FN, #blocked>, tensor<16x1xi8, #blocked> -> tensor<16x16xf32, #blocked>

    %c16_m = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %c_row = arith.muli %m_2d, %c16_m : tensor<16x1xi32, #blocked>
    %c_row_bc = tt.broadcast %c_row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_off = arith.addi %c_row_bc, %c_n_bc : tensor<16x16xi32, #blocked>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_base, %c_off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %c_addr, %result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // E2M1 is physically packed two logical K elements per byte, low nibble
  // first.  This first exact FP4 slice requires both operands packed along K,
  // so the 16-byte reduction dimension below represents logical K=32.
  tt.func public @dot_scaled_e8m0_e2m1_kpack(
      %a_ptr: !tt.ptr<i8>, %b_ptr: !tt.ptr<i8>,
      %a_scale_ptr: !tt.ptr<i8>, %b_scale_ptr: !tt.ptr<i8>,
      %c_ptr: !tt.ptr<f32>) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %n_as_row = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %k_as_col = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %k_as_row = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %m_2d = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %n_2d = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #slice_col> -> tensor<1x16xi32, #blocked>
    %k_a_2d = tt.expand_dims %k_as_col {axis = 0 : i32} : tensor<16xi32, #slice_col> -> tensor<1x16xi32, #blocked>
    %k_b_2d = tt.expand_dims %k_as_row {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>

    %c16_mk = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %a_row = arith.muli %m_2d, %c16_mk : tensor<16x1xi32, #blocked>
    %a_row_bc = tt.broadcast %a_row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %k_a_bc = tt.broadcast %k_a_2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %a_off = arith.addi %a_row_bc, %k_a_bc : tensor<16x16xi32, #blocked>
    %a_base = tt.splat %a_ptr : !tt.ptr<i8> -> tensor<16x16x!tt.ptr<i8>, #blocked>
    %a_addr = tt.addptr %a_base, %a_off : tensor<16x16x!tt.ptr<i8>, #blocked>, tensor<16x16xi32, #blocked>
    %a = tt.load %a_addr : tensor<16x16x!tt.ptr<i8>, #blocked>

    %c16_kn = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16_kn : tensor<16x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<16x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<i8> -> tensor<16x16x!tt.ptr<i8>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<16x16x!tt.ptr<i8>, #blocked>, tensor<16x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<16x16x!tt.ptr<i8>, #blocked>

    %c1_m = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %a_scale_off = arith.muli %m_2d, %c1_m : tensor<16x1xi32, #blocked>
    %a_scale_base = tt.splat %a_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %a_scale_addr = tt.addptr %a_scale_base, %a_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %a_scale = tt.load %a_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %n_scale_2d = tt.expand_dims %n_as_row {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %c1_n = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %b_scale_off = arith.muli %n_scale_2d, %c1_n : tensor<16x1xi32, #blocked>
    %b_scale_base = tt.splat %b_scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %b_scale_addr = tt.addptr %b_scale_base, %b_scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %b_scale = tt.load %b_scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %result = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<16x16xi8, #blocked>, tensor<16x1xi8, #blocked> * tensor<16x16xi8, #blocked>, tensor<16x1xi8, #blocked> -> tensor<16x16xf32, #blocked>

    %c16_out = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %c_row = arith.muli %m_2d, %c16_out : tensor<16x1xi32, #blocked>
    %c_row_bc = tt.broadcast %c_row : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %c_off = arith.addi %c_row_bc, %c_n_bc : tensor<16x16xi32, #blocked>
    %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %c_addr = tt.addptr %c_base, %c_off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    tt.store %c_addr, %result : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel dot_scaled_e8m0_bf16
// CHECK-NOT: tt.dot_scaled
// CHECK: scf.for
// CHECK: arith.extui
// CHECK: arith.shli
// CHECK: metal.bitcast
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: metal.return

// CHECK-LABEL: metal.kernel dot_scaled_e8m0_fp16_factor16
// CHECK-NOT: tt.dot_scaled
// CHECK: scf.for
// CHECK: arith.extui
// CHECK: arith.shli
// CHECK: metal.bitcast
// CHECK-NOT: arith.select
// CHECK: metal.return

// CHECK-LABEL: metal.kernel dot_scaled_e8m0_e5m2
// CHECK-NOT: tt.dot_scaled
// CHECK: %[[A_RAW:.*]] = metal.get_element {{.*}} -> i8
// CHECK: %[[A_WIDE:.*]] = arith.extui %[[A_RAW]] : i8 to i16
// CHECK: %[[A_MASKED:.*]] = arith.andi %[[A_WIDE]], {{.*}} : i16
// CHECK: %[[A_SHIFTED:.*]] = arith.shli %[[A_MASKED]], {{.*}} : i16
// CHECK: %[[A_F16:.*]] = metal.bitcast {{.*}} -> f16
// CHECK: %[[A_F32:.*]] = arith.extf %[[A_F16]] : f16 to f32
// CHECK: %[[A_BF16:.*]] = arith.truncf %[[A_F32]] : f32 to bf16
// CHECK: %[[A_SCALED:.*]] = arith.mulf %[[A_BF16]], {{.*}} : bf16
// CHECK: %[[A_SELECTED:.*]] = arith.select {{.*}}, {{.*}}, %[[A_SCALED]] : bf16
// CHECK: %[[A_ACCUM:.*]] = arith.extf %[[A_SELECTED]] : bf16 to f32
// CHECK: arith.mulf %[[A_ACCUM]], {{.*}} : f32
// CHECK: metal.return

// CHECK-LABEL: metal.kernel dot_scaled_e8m0_mixed_e5m2_e4m3
// CHECK-NOT: tt.dot_scaled
// CHECK: %[[A_RAW:.*]] = metal.get_element {{.*}} -> i8
// CHECK: %[[B_RAW:.*]] = metal.get_element {{.*}} -> i8
// CHECK: %[[A_WIDE:.*]] = arith.extui %[[A_RAW]] : i8 to i16
// CHECK: %[[A_MASKED:.*]] = arith.andi %[[A_WIDE]], {{.*}} : i16
// CHECK: %[[A_SHIFTED:.*]] = arith.shli %[[A_MASKED]], {{.*}} : i16
// CHECK: %[[A_F16:.*]] = metal.bitcast {{.*}} -> f16
// CHECK: %[[A_F32:.*]] = arith.extf %[[A_F16]] : f16 to f32
// CHECK: %[[A_BF16:.*]] = arith.truncf %[[A_F32]] : f32 to bf16
// CHECK: %[[B_WIDE:.*]] = arith.extui %[[B_RAW]] : i8 to i16
// CHECK: %[[B_MASKED:.*]] = arith.andi %[[B_WIDE]], {{.*}} : i16
// CHECK: %[[B_MAG:.*]] = arith.andi %[[B_MASKED]], {{.*}} : i16
// CHECK: arith.cmpi ult, %[[B_MAG]]
// CHECK: metal.bitcast {{.*}} -> bf16
// CHECK: arith.mulf %[[A_BF16]], {{.*}} : bf16
// CHECK: arith.mulf {{.*}} : bf16
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: metal.return

// CHECK-LABEL: metal.kernel dot_scaled_e8m0_e4m3
// CHECK-NOT: tt.dot_scaled
// CHECK: %[[E4_RAW:.*]] = metal.get_element {{.*}} -> i8
// CHECK: %[[E4_WIDE:.*]] = arith.extui %[[E4_RAW]] : i8 to i16
// CHECK: arith.andi %[[E4_WIDE]], {{.*}} : i16
// CHECK: arith.cmpi ult
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: metal.bitcast {{.*}} -> bf16
// CHECK: arith.mulf {{.*}} : bf16
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: metal.return

// CHECK-LABEL: metal.kernel dot_scaled_e8m0_e2m1_kpack
// CHECK-NOT: tt.dot_scaled
// CHECK: scf.for
// CHECK: arith.divui
// CHECK: metal.get_element {{.*}} -> i8
// CHECK: arith.shrui
// CHECK: arith.andi
// CHECK: metal.bitcast {{.*}} -> bf16
// CHECK: arith.mulf {{.*}} : bf16
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: metal.return

// MSL-LABEL: kernel void dot_scaled_e8m0_bf16
// MSL: for (int
// MSL: & 255
// MSL: == (int16_t)(255)
// MSL: bfloat(NAN)

// MSL-LABEL: kernel void dot_scaled_e8m0_fp16_factor16
// The factor-16 scale index is an arith.divui, so it carries an UNSIGNED cast
// and is distinct from the signed M/N coordinate divides. It is emitted after
// the exponent reconstruction, hence the order of these two checks.
// MSL: as_type<float>
// MSL: / (uint32_t)(16)
// MSL: & 255
// MSL-NOT: NAN

// MSL-LABEL: kernel void dot_scaled_e8m0_e5m2
// MSL: / (uint32_t)(32)
// MSL: == (int16_t)(255)
// MSL: bfloat(NAN)
// MSL: << 8
// MSL: as_type<half>(uint16_t(
// MSL: as_type<bfloat>

// MSL-LABEL: kernel void dot_scaled_e8m0_mixed_e5m2_e4m3
// MSL: as_type<half>
// MSL-SAME: v0
// MSL-SAME: << 8
// MSL-SAME: as_type<bfloat>
// MSL-SAME: v1
// MSL-SAME: & 128) << 8
// MSL-SAME: == (int16_t)(127)

// MSL-LABEL: kernel void dot_scaled_e8m0_e4m3
// MSL: / (uint32_t)(32)
// MSL: == (int16_t)(127)
// MSL: & 127
// MSL: as_type<bfloat>(uint16_t(

// MSL-LABEL: kernel void dot_scaled_e8m0_e2m1_kpack
// MSL: / (uint32_t)(2)
// MSL: & 15
// MSL: & 7
// MSL: as_type<bfloat>(uint16_t(
