// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Session L3 (`.omc/specs/deep-interview-leet-triton-l3-reduce-axis-2d.md`)
// negative pre-pass: tt.reduce inputs that fall outside the L3 envelope are
// rejected with the spec-mandated error strings. Phase C (the actual tree-
// reduction lowering) is deferred to L3a per the spec's honest divergence
// policy; this fixture exercises the pre-pass rejection gate.

// -----
// Rank-3 input → "multi-axis or rank>2".
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#slice3d = #ttg.slice<{dim = 0, parent = #blocked3d}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank3(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<8x4x4xf32, #blocked3d>
    // expected-error @+1 {{reduce with rank not in {1, 2} not supported (rank-1 added in Option β; rank>2 requires Session L3b)}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<8x4x4xf32, #blocked3d>) -> tensor<4x4xf32, #slice3d>
    tt.return
  }
}

// -----
// A two-source reducer that is close to argmax but selects the HIGHER index on
// ties must not be claimed by the canonical argmax lowering.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_argmax_near_miss(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked>
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    // expected-error @+1 {{reduce combine requires Session L3c (future) — got arith.cmpf}}
    %r:2 = "tt.reduce"(%x, %idx) ({
    ^bb0(%lhs_value: f32, %lhs_index: i32, %rhs_value: f32, %rhs_index: i32):
      %equal = arith.cmpf oeq, %lhs_value, %rhs_value : f32
      %higher_index = arith.cmpi sgt, %lhs_index, %rhs_index : i32
      %equal_and_higher = arith.andi %equal, %higher_index : i1
      %greater = arith.cmpf ogt, %lhs_value, %rhs_value : f32
      %take_left = arith.ori %greater, %equal_and_higher : i1
      %value = arith.select %take_left, %lhs_value, %rhs_value : f32
      %index = arith.select %take_left, %lhs_index, %rhs_index : i32
      tt.reduce.return %value, %index : f32, i32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked>, tensor<32xi32, #blocked>) -> (f32, i32)
    tt.return
  }
}

// -----
// Unsupported combine op → "reduce combine requires Session L3c". Signed
// i32 max/min are now accepted for rank-1 and rank-2, so integer product remains
// as the representative unsupported combine.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_muli(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<0> : tensor<16x32xi32, #blocked>
    // expected-error @+1 {{reduce combine requires Session L3c (future) — got arith.muli}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %m = arith.muli %a, %b : i32
      tt.reduce.return %m : i32
    }) {axis = 1 : i32} : (tensor<16x32xi32, #blocked>) -> tensor<16xi32, #slice1>
    tt.return
  }
}

// -----
// Rank-2 axis=1 i32 extrema currently support a direct unmasked load only.
// Keep the adjacent computed-cone shape behind a stable pre-pass diagnostic.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_maxsi_computed_deferred(%x_ptr: !tt.ptr<i32>) {
    %r0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r1 = tt.expand_dims %r0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %offs = tt.broadcast %r1 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<8x16x!tt.ptr<i32>, #blocked>
    %ap = tt.addptr %sp, %offs : tensor<8x16x!tt.ptr<i32>, #blocked>, tensor<8x16xi32, #blocked>
    %x = tt.load %ap : tensor<8x16x!tt.ptr<i32>, #blocked>
    %one = arith.constant dense<1> : tensor<8x16xi32, #blocked>
    %computed = arith.addi %x, %one : tensor<8x16xi32, #blocked>
    // expected-error @+1 {{rank-2 axis=1 i32 max/min requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%computed) ({
    ^bb0(%a: i32, %b: i32):
      %m = arith.maxsi %a, %b : i32
      tt.reduce.return %m : i32
    }) {axis = 1 : i32} : (tensor<8x16xi32, #blocked>) -> tensor<8xi32, #slice1>
    tt.return
  }
}

// -----
// f16 dtype on supported combine op (addf) → "reduce dtype must be f32 or i32 in Session L3".
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_f16(%x_ptr: !tt.ptr<f16>) {
    %x = arith.constant dense<0.0> : tensor<16x32xf16, #blocked>
    // expected-error @+1 {{reduce dtype must be f32 or i32 in Session L3}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f16, %b: f16):
      %s = arith.addf %a, %b : f16
      tt.reduce.return %s : f16
    }) {axis = 1 : i32} : (tensor<16x32xf16, #blocked>) -> tensor<16xf16, #slice1>
    tt.return
  }
}

// -----
// axis=0 reduce — f32 sum/max and i32 sum/max/min now ship (Session L3a2,
// lowerRank2Axis0Reduce), but other combines (here f32 PRODUCT) stay deferred.
// The pre-pass axis check rejects the unsupported combine.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_mul_axis0_deferred(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<16x32xf32, #blocked>
    // expected-error @+1 {{tt.reduce axis=0 reduce supports f32 sum/max and i32 sum/max/min (Session L3a2)}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.mulf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<16x32xf32, #blocked>) -> tensor<32xf32, #slice0>
    tt.return
  }
}
