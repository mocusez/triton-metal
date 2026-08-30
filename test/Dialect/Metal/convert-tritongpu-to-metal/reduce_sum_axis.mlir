// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// Rank-2 axis=1 `tt.reduce` (combine ∈ {arith.addf, arith.addi}, T ∈
// {f32, i32}) lowers to the self-contained per-row reduction body
// (L3a-tileloop-2): walk back to the producing `tt.load`, allocate a
// per-row threadgroup buffer `rowBuf[M]`, and have each thread reduce a
// grid-strided set of whole rows directly from device memory into rowBuf;
// after a barrier each thread reads the row its downstream store targets.
// The staging is hoisted above any FuncOpLowering tile loop so it runs once.

// -----
// f32 / arith.addf, shape <8x16xf32>, axis=1. M=8 < tpb=128, so there is no
// outer tile loop and the M<tpb output store is wrapped in a bounds guard.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_sum_axis1_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      tt.reduce.return %add : f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<f32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xf32, #slice1> -> tensor<8xf32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<f32>, #blocked1>
    tt.return
  }

  tt.func public @reduce_product_axis1_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %product = arith.mulf %a, %b : f32
      tt.reduce.return %product : f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<f32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xf32, #slice1> -> tensor<8xf32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<f32>, #blocked1>
    tt.return
  }

  // A user loop with an iter_arg folds the per-program and per-trip bases into
  // the load's tensor offset. The cooperative row scan must replay that full
  // address instead of replacing it with `(row + tgid*tpb) * N`.
  tt.func public @reduce_sum_axis1_loop_iterarg(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %T: i32, %initial: f32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    %tile = arith.constant 128 : i32
    %half = arith.constant dense<5.000000e-01> : tensor<8xf32, #slice1>
    %h0 = tt.splat %initial : f32 -> tensor<8xf32, #slice1>
    %pid = tt.get_program_id x : i32
    %pid_t = arith.muli %pid, %T : i32
    %program_base = arith.muli %pid_t, %tile : i32
    %row_scale = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %row_offsets = arith.muli %rows_2d, %row_scale : tensor<8x1xi32, #blocked>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %cols_full = tt.broadcast %cols_2d : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %x_base = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %out_program = arith.muli %pid_t, %c8 : i32
    %out_base = tt.addptr %out_ptr, %out_program : !tt.ptr<f32>, i32
    %result = scf.for %t = %c0 to %T step %c1 iter_args(%h = %h0) -> (tensor<8xf32, #slice1>) : i32 {
      %trip_base = arith.muli %t, %tile : i32
      %tile_base = arith.addi %program_base, %trip_base : i32
      %tile_base_2d = tt.splat %tile_base : i32 -> tensor<8x1xi32, #blocked>
      %row_bases = arith.addi %tile_base_2d, %row_offsets : tensor<8x1xi32, #blocked>
      %row_bases_full = tt.broadcast %row_bases : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
      %offsets = arith.addi %row_bases_full, %cols_full : tensor<8x16xi32, #blocked>
      %x_addr = tt.addptr %x_base, %offsets : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
      %x = tt.load %x_addr : tensor<8x16x!tt.ptr<f32>, #blocked>
      %sum = "tt.reduce"(%x) ({
      ^bb0(%a: f32, %b: f32):
        %add = arith.addf %a, %b : f32
        tt.reduce.return %add : f32
      }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<8xf32, #slice1>
      %decayed = arith.mulf %h, %half : tensor<8xf32, #slice1>
      %next = arith.addf %decayed, %sum : tensor<8xf32, #slice1>
      %out_trip = arith.muli %t, %c8 : i32
      %out_trip_base = tt.addptr %out_base, %out_trip : !tt.ptr<f32>, i32
      %out_splat = tt.splat %out_trip_base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked1>
      %out_rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
      %out_addr = tt.addptr %out_splat, %out_rows : tensor<8x!tt.ptr<f32>, #blocked1>, tensor<8xi32, #blocked1>
      %next_blocked = ttg.convert_layout %next : tensor<8xf32, #slice1> -> tensor<8xf32, #blocked1>
      tt.store %out_addr, %next_blocked : tensor<8x!tt.ptr<f32>, #blocked1>
      scf.yield %next : tensor<8xf32, #slice1>
    }
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_sum_axis1_f32
// rowBuf is M (= 8) elements, not M*N.
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// Grid-stride row loop + per-row `r < M` guard + per-row column scan.
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.get_element
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK: scf.yield
// CHECK: metal.store {{.*}} !metal.memref<8 x f32>
// CHECK: metal.barrier
// Per-row result read + sub-tpb (M < tpb) output store bounds guard.
// CHECK: metal.get_element {{.*}}memref<8 x f32>
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: metal.store
// CHECK: metal.return
// CHECK-LABEL: metal.kernel reduce_product_axis1_f32
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// CHECK: arith.constant 1.000000e+00 : f32
// CHECK: metal.binary_exp {{.*}}, {{.*}}, mulOp : (f32, f32) -> f32
// CHECK: metal.return
// CHECK-LABEL: metal.kernel reduce_sum_axis1_loop_iterarg
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// CHECK: %[[T_RAW:.*]] = metal.get_element %arg2
// CHECK-NEXT: %[[T:.*]] = builtin.unrealized_conversion_cast %[[T_RAW]] : ui32 to i32
// CHECK: scf.for %[[TRIP:.*]] = {{.*}} to %[[T]] {{.*}} iter_args
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: scf.for %[[COL:[^ ]+]] =
// The row scan address must contain both pid*T*tile and trip*tile.
// CHECK: %[[PID_T:.*]] = arith.muli {{.*}}, %[[T]] : i32
// CHECK: %[[PID_BASE:.*]] = arith.muli %[[PID_T]], {{.*}} : i32
// CHECK: %[[TRIP_BASE:.*]] = arith.muli %[[TRIP]], {{.*}} : i32
// CHECK: %[[TILE_BASE:.*]] = arith.addi %[[PID_BASE]], %[[TRIP_BASE]] : i32
// CHECK: %[[ROW_BASE:.*]] = arith.addi %[[TILE_BASE]], {{.*}} : i32
// CHECK: %[[ADDR:.*]] = arith.addi %[[ROW_BASE]], %[[COL]] : i32
// CHECK: %[[ADDR_UI32:.*]] = builtin.unrealized_conversion_cast %[[ADDR]] : i32 to ui32
// CHECK: metal.get_element %arg0[%[[ADDR_UI32]]]
// CHECK: metal.return

// -----
// i32 / arith.addi, shape <8x16xi32>, axis=1. The i32 path routes the row
// buffer + column scan through ui32 storage (Metal_Type rejects signless
// i32) and emits the column scan UNROLLED (the translator only threads f32
// scf.for iter_args).
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_sum_axis1_i32(%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<i32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<i32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %add = arith.addi %a, %b : i32
      tt.reduce.return %add : i32
    }) {axis = 1 : i32} : (tensor<8x16xi32, #blocked>) -> tensor<8xi32, #slice1>
    %osp = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #blocked1>
    %oap = tt.addptr %osp, %offs_m : tensor<8x!tt.ptr<i32>, #blocked1>, tensor<8xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<8xi32, #slice1> -> tensor<8xi32, #blocked1>
    tt.store %oap, %cv : tensor<8x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_sum_axis1_i32
// i32 row buffer routed through ui32 storage.
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x ui32>
// Unrolled ui32 column scan.
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp : (ui32, ui32) -> ui32
// CHECK: metal.barrier
// CHECK: metal.return

// -----
// i32 / arith.muli, shape <8x16xi32>, axis=0. Each output thread owns one
// column and scans its rows locally. Product uses ui32 internally so signed
// i32 overflow retains Triton's modulo-2^32 bit pattern; masked rows contribute
// multiplicative identity one.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_product_axis0_i32(%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_n = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice0>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #slice0> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<i32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<i32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %product = arith.muli %a, %b : i32
      tt.reduce.return %product : i32
    }) {axis = 0 : i32} : (tensor<8x16xi32, #blocked>) -> tensor<16xi32, #slice0>
    %osp = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked1>
    %oap = tt.addptr %osp, %offs_n : tensor<16x!tt.ptr<i32>, #blocked1>, tensor<16xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<16xi32, #slice0> -> tensor<16xi32, #blocked1>
    tt.store %oap, %cv : tensor<16x!tt.ptr<i32>, #blocked1>
    tt.return
  }

  tt.func public @reduce_product_axis0_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_n = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice0>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #slice0> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %product = arith.mulf %a, %b : f32
      tt.reduce.return %product : f32
    }) {axis = 0 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<16xf32, #slice0>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_n : tensor<16x!tt.ptr<f32>, #blocked1>, tensor<16xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<16xf32, #slice0> -> tensor<16xf32, #blocked1>
    tt.store %oap, %cv : tensor<16x!tt.ptr<f32>, #blocked1>
    tt.return
  }

  tt.func public @reduce_min_axis0_f32(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<16> : tensor<8x1xi32, #blocked>
    %offs_n = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked1>
    %r0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #slice1>
    %r1 = tt.expand_dims %r0 {axis = 1 : i32} : tensor<8xi32, #slice1> -> tensor<8x1xi32, #blocked>
    %r2 = arith.muli %r1, %cst : tensor<8x1xi32, #blocked>
    %r3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice0>
    %r4 = tt.expand_dims %r3 {axis = 0 : i32} : tensor<16xi32, #slice0> -> tensor<1x16xi32, #blocked>
    %r5 = tt.broadcast %r2 : tensor<8x1xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r6 = tt.broadcast %r4 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %r7 = arith.addi %r5, %r6 : tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %r7 : tensor<8x16x!tt.ptr<f32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<f32>, #blocked>
    %s = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %minimum = arith.minnumf %a, %b : f32
      tt.reduce.return %minimum : f32
    }) {axis = 0 : i32} : (tensor<8x16xf32, #blocked>) -> tensor<16xf32, #slice0>
    %osp = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked1>
    %oap = tt.addptr %osp, %offs_n : tensor<16x!tt.ptr<f32>, #blocked1>, tensor<16xi32, #blocked1>
    %cv = ttg.convert_layout %s : tensor<16xf32, #slice0> -> tensor<16xf32, #blocked1>
    tt.store %oap, %cv : tensor<16x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
// CHECK-LABEL: metal.kernel reduce_product_axis0_i32
// CHECK-NOT: metal.threadgroup_alloca
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (ui32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, mulOp : (ui32, ui32) -> ui32
// CHECK: metal.return
// CHECK-LABEL: metal.kernel reduce_product_axis0_f32
// CHECK-NOT: metal.threadgroup_alloca
// CHECK: arith.constant 1.000000e+00 : f32
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, mulOp : (f32, f32) -> f32
// CHECK: metal.return
// CHECK-LABEL: metal.kernel reduce_min_axis0_f32
// CHECK-NOT: metal.threadgroup_alloca
// CHECK: arith.constant 0x7F800000 : f32
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.binary_exp {{.*}}, {{.*}}, minOp : (f32, f32) -> f32
// CHECK: metal.return
