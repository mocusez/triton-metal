// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s

// Unsupported adjacent formats must stop in the mutation-free preflight.  In
// particular, a residual tt.dot_scaled used to reach dialect-conversion
// teardown and assert after the ordinary legalization error was printed.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_scaled_mixed_fp8_unsupported(
      %a: tensor<16x32xf8E4M3FN, #blocked>,
      %b: tensor<32x16xf8E5M2, #blocked>,
      %a_scale: tensor<16x1xi8, #blocked>,
      %b_scale: tensor<16x1xi8, #blocked>,
      %acc: tensor<16x16xf32, #blocked>) {
    // expected-error@+1 {{'tt.dot_scaled' op Metal backend: requires matching fp16/fp16, bf16/bf16, e2m1/e2m1, e4m3/e4m3, or e5m2/e5m2 payload formats}}
    %result = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = e4m3 rhs = e5m2 {fastMath = false} : tensor<16x32xf8E4M3FN, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xf8E5M2, #blocked>, tensor<16x1xi8, #blocked> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// Packing outside K is an E2M1-only feature.  Raw MLIR that requests it for a
// different payload format must fail before scalar-dot reconstruction.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_scaled_fp16_non_k_pack_unsupported(
      %a: tensor<16x32xf16, #blocked>,
      %b: tensor<32x16xf16, #blocked>,
      %a_scale: tensor<16x1xi8, #blocked>,
      %b_scale: tensor<16x1xi8, #blocked>,
      %acc: tensor<16x16xf32, #blocked>) {
    // expected-error@+1 {{'tt.dot_scaled' op Metal backend: packing outside K is supported only for E2M1 payloads}}
    %result = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %acc lhs = fp16 rhs = fp16 {fastMath = false, lhs_k_pack = false, rhs_k_pack = true} : tensor<16x32xf16, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xf16, #blocked>, tensor<16x1xi8, #blocked> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// E5M2 is byte-backed only for the exact dot_scaled rewrite.  A residual
// ordinary load must fail before dialect conversion rather than being treated
// as a generic i8 load through the ABI storage mapping.

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @residual_e5m2_load_unsupported(%ptr: !tt.ptr<f8E5M2>) {
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked1>
    %base = tt.splat %ptr : !tt.ptr<f8E5M2> -> tensor<32x!tt.ptr<f8E5M2>, #blocked1>
    %addr = tt.addptr %base, %range : tensor<32x!tt.ptr<f8E5M2>, #blocked1>, tensor<32xi32, #blocked1>
    // expected-error@+1 {{'tt.load' op Metal backend: FP8 loads are supported only as fully consumed tt.dot_scaled payloads}}
    %value = tt.load %addr : tensor<32x!tt.ptr<f8E5M2>, #blocked1>
    tt.return
  }
}

// -----

// The same byte-ABI guard applies to scalar E5M2 loads.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @residual_scalar_e5m2_load_unsupported(%ptr: !tt.ptr<f8E5M2>) {
    // expected-error@+1 {{'tt.load' op Metal backend: FP8 loads are supported only as fully consumed tt.dot_scaled payloads}}
    %value = tt.load %ptr : !tt.ptr<f8E5M2>
    tt.return
  }
}

// -----

// E4M3FN uses the same byte-ABI guard as E5M2.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @residual_scalar_e4m3_load_unsupported(%ptr: !tt.ptr<f8E4M3FN>) {
    // expected-error@+1 {{'tt.load' op Metal backend: FP8 loads are supported only as fully consumed tt.dot_scaled payloads}}
    %value = tt.load %ptr : !tt.ptr<f8E4M3FN>
    tt.return
  }
}

// -----

// A one-group scale can omit a column contribution, but it still cannot hide
// a constant slice offset inside its row contribution.  scalar_dot reconstructs
// scale addresses from the base plus row * stride, so accepting this pointer
// would silently read the preceding scale element for every row.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#slice_row = #ttg.slice<{dim = 1, parent = #blocked}>
#slice_col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_scaled_offset_scale_unsupported(
      %a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>,
      %scale_ptr: !tt.ptr<i8>) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %k_col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_col>
    %k_row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_row>
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

    %c16 = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16 : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<f16>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<f16>, #blocked>

    %c1 = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %scale_row = arith.muli %m_2d, %c1 : tensor<16x1xi32, #blocked>
    %scale_off = arith.addi %scale_row, %c1 : tensor<16x1xi32, #blocked>
    %scale_base = tt.splat %scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %scale_addr = tt.addptr %scale_base, %scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %a_scale = tt.load %scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // expected-error@+1 {{'tt.dot_scaled' op Metal backend: requires direct unmasked row-major i8 scale loads}}
    %result = tt.dot_scaled %a scale %a_scale, %b, %acc lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x32xf16, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xf16, #blocked> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// A dynamic scale-row origin is also outside the current exact envelope.  It
// cannot be inferred from the output coordinates and is not carried by
// scalar_dot, so preflight must reject it instead of dropping it.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#slice_row = #ttg.slice<{dim = 1, parent = #blocked}>
#slice_col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_scaled_dynamic_origin_scale_unsupported(
      %a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>,
      %scale_ptr: !tt.ptr<i8>, %scale_origin: i32) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %k_col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_col>
    %k_row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_row>
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

    %c16 = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16 : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<f16>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<f16>, #blocked>

    %origin = tt.splat %scale_origin : i32 -> tensor<16x1xi32, #blocked>
    %scale_off = arith.addi %m_2d, %origin : tensor<16x1xi32, #blocked>
    %scale_base = tt.splat %scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %scale_addr = tt.addptr %scale_base, %scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %a_scale = tt.load %scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // expected-error@+1 {{'tt.dot_scaled' op Metal backend: requires direct unmasked row-major i8 scale loads}}
    %result = tt.dot_scaled %a scale %a_scale, %b, %acc lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x32xf16, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xf16, #blocked> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// Payload loads are reconstructed by scalar_dot too.  A shaped K-axis slice
// must therefore be rejected instead of silently dropping the hidden +8.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#slice_row = #ttg.slice<{dim = 1, parent = #blocked}>
#slice_col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_scaled_offset_payload_unsupported(
      %a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>,
      %scale_ptr: !tt.ptr<i8>) {
    %m = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_row>
    %n = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #slice_col>
    %k_col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_col>
    %k_row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice_row>
    %m_2d = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32, #slice_row> -> tensor<16x1xi32, #blocked>
    %n_2d = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32, #slice_col> -> tensor<1x16xi32, #blocked>
    %k_a_2d = tt.expand_dims %k_col {axis = 0 : i32} : tensor<32xi32, #slice_col> -> tensor<1x32xi32, #blocked>
    %k_b_2d = tt.expand_dims %k_row {axis = 1 : i32} : tensor<32xi32, #slice_row> -> tensor<32x1xi32, #blocked>

    %c32 = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %a_row = arith.muli %m_2d, %c32 : tensor<16x1xi32, #blocked>
    %a_row_bc = tt.broadcast %a_row : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %k_a_bc = tt.broadcast %k_a_2d : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %c8 = arith.constant dense<8> : tensor<16x32xi32, #blocked>
    %a_k_slice = arith.addi %k_a_bc, %c8 : tensor<16x32xi32, #blocked>
    %a_off = arith.addi %a_row_bc, %a_k_slice : tensor<16x32xi32, #blocked>
    %a_base = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<16x32x!tt.ptr<f16>, #blocked>
    %a_addr = tt.addptr %a_base, %a_off : tensor<16x32x!tt.ptr<f16>, #blocked>, tensor<16x32xi32, #blocked>
    %a = tt.load %a_addr : tensor<16x32x!tt.ptr<f16>, #blocked>

    %c16 = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16 : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<f16>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<f16>, #blocked>

    %c1 = arith.constant dense<1> : tensor<16x1xi32, #blocked>
    %scale_off = arith.muli %m_2d, %c1 : tensor<16x1xi32, #blocked>
    %scale_base = tt.splat %scale_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>, #blocked>
    %scale_addr = tt.addptr %scale_base, %scale_off : tensor<16x1x!tt.ptr<i8>, #blocked>, tensor<16x1xi32, #blocked>
    %a_scale = tt.load %scale_addr : tensor<16x1x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // expected-error@+1 {{'tt.dot_scaled' op Metal backend: requires canonical row-major matrix pointer arithmetic}}
    %result = tt.dot_scaled %a scale %a_scale, %b, %acc lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x32xf16, #blocked>, tensor<16x1xi8, #blocked> * tensor<32x16xf16, #blocked> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

// The scale tensor shape is valid, but its physical K/S dimension is loaded
// with stride 2.  The scalar lowering reconstructs contiguous scale-group
// addresses, so this adjacent layout must be rejected rather than miscompiled.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#slice_row = #ttg.slice<{dim = 1, parent = #blocked}>
#slice_col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_scaled_strided_scale_unsupported(
      %a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>,
      %scale_ptr: !tt.ptr<i8>) {
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

    %c16 = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %b_row = arith.muli %k_b_2d, %c16 : tensor<32x1xi32, #blocked>
    %b_row_bc = tt.broadcast %b_row : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %n_bc = tt.broadcast %n_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %b_off = arith.addi %b_row_bc, %n_bc : tensor<32x16xi32, #blocked>
    %b_base = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x16x!tt.ptr<f16>, #blocked>
    %b_addr = tt.addptr %b_base, %b_off : tensor<32x16x!tt.ptr<f16>, #blocked>, tensor<32x16xi32, #blocked>
    %b = tt.load %b_addr : tensor<32x16x!tt.ptr<f16>, #blocked>

    %scale_k_2d = tt.expand_dims %scale_k {axis = 0 : i32} : tensor<2xi32, #slice_col> -> tensor<1x2xi32, #blocked>
    %c4 = arith.constant dense<4> : tensor<16x1xi32, #blocked>
    %scale_row = arith.muli %m_2d, %c4 : tensor<16x1xi32, #blocked>
    %c2 = arith.constant dense<2> : tensor<1x2xi32, #blocked>
    %scale_col = arith.muli %scale_k_2d, %c2 : tensor<1x2xi32, #blocked>
    %scale_row_bc = tt.broadcast %scale_row : tensor<16x1xi32, #blocked> -> tensor<16x2xi32, #blocked>
    %scale_col_bc = tt.broadcast %scale_col : tensor<1x2xi32, #blocked> -> tensor<16x2xi32, #blocked>
    %scale_off = arith.addi %scale_row_bc, %scale_col_bc : tensor<16x2xi32, #blocked>
    %scale_base = tt.splat %scale_ptr : !tt.ptr<i8> -> tensor<16x2x!tt.ptr<i8>, #blocked>
    %scale_addr = tt.addptr %scale_base, %scale_off : tensor<16x2x!tt.ptr<i8>, #blocked>, tensor<16x2xi32, #blocked>
    %a_scale = tt.load %scale_addr : tensor<16x2x!tt.ptr<i8>, #blocked>

    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // expected-error@+1 {{'tt.dot_scaled' op Metal backend: requires direct unmasked row-major i8 scale loads}}
    %result = tt.dot_scaled %a scale %a_scale, %b, %acc lhs = fp16 rhs = fp16 {fastMath = true} : tensor<16x32xf16, #blocked>, tensor<16x2xi8, #blocked> * tensor<32x16xf16, #blocked> -> tensor<16x16xf32, #blocked>
    tt.return
  }
}
