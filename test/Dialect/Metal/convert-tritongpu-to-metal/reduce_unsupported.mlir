// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Session L3 (the implementation notes)
// negative pre-pass: tt.reduce inputs that fall outside the L3 envelope are
// rejected with the spec-mandated error strings. Phase C (the actual tree-
// reduction lowering) is deferred to L3a per the spec's honest divergence
// policy; this fixture exercises the pre-pass rejection gate.

// -----
// Rank >= 3 IS implemented now, by the staged fallback, but only for a tile it
// can publish in one pass. 16x8x8 is 1024 elements over 128 threads, so it
// stays rejected — and the message names the envelope instead of a session.
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#slice3d = #ttg.slice<{dim = 0, parent = #blocked3d}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank3(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<16x8x8xf32, #blocked3d>
    // expected-error @+1 {{rank >= 3 reduce is implemented for a blocked tile of at most one element per thread}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<16x8x8xf32, #blocked3d>) -> tensor<8x8xf32, #slice3d>
    tt.return
  }
}

// -----
// Rank-1 f32 product starts with a direct-unmasked load envelope. A computed
// source remains behind a stable preflight diagnostic.
#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_mulf_rank1_computed_deferred() {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked1d>
    // expected-error @+1 {{rank-1 f32 product requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %product = arith.mulf %a, %b : f32
      tt.reduce.return %product : f32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked1d>) -> f32
    tt.return
  }
}

// -----
// Masked rank-1 loads remain outside the initial f32 product slice.
#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_mulf_rank1_masked_deferred(%x_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1d>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1d>
    %ap = tt.addptr %sp, %offsets : tensor<32x!tt.ptr<f32>, #blocked1d>, tensor<32xi32, #blocked1d>
    %mask = arith.constant dense<true> : tensor<32xi1, #blocked1d>
    %other = arith.constant dense<1.0> : tensor<32xf32, #blocked1d>
    %x = tt.load %ap, %mask, %other : tensor<32x!tt.ptr<f32>, #blocked1d>
    // expected-error @+1 {{rank-1 f32 product requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %product = arith.mulf %a, %b : f32
      tt.reduce.return %product : f32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked1d>) -> f32
    tt.return
  }
}

// -----
// A two-source reducer that is close to argmax but selects the HIGHER index on
// ties must not be claimed by the canonical argmax lowering.
//
// It is now refused one gate earlier, as a multi-value reduce, rather than by
// the single-result combine allowlist noticing `arith.cmpf`. Same verdict, and
// the reason is the accurate one: the shape has two sources, which is what the
// lowering cannot do. The allowlist only ever caught this by accident -- see
// the `maxnumf` pair at the end of this file, whose first region op IS on the
// allowlist and which therefore crashed the process instead.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_argmax_near_miss(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked>
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    // expected-error @+1 {{multi-value tt.reduce is implemented only for the canonical rank-1 argmax and rank-2 axis=1 argmin pairs}}
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
// Direct rank-2 axis=1 i32 product is supported. Keep its adjacent computed
// source behind the same deterministic direct-load envelope as i32 extrema.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_muli(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<0> : tensor<16x32xi32, #blocked>
    // expected-error @+1 {{rank-2 axis=1 i32 product/max/min/bitwise requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %m = arith.muli %a, %b : i32
      tt.reduce.return %m : i32
    }) {axis = 1 : i32} : (tensor<16x32xi32, #blocked>) -> tensor<16xi32, #slice1>
    tt.return
  }
}

// -----
// Rank-2 axis=1 i32 product/extrema over a COMPUTED cone: the row scanner wants
// a direct unmasked load, and the staged fallback that now covers the shape
// needs the tile to fit one element per thread. 32x16 is 512 over 128 threads,
// so neither takes it and the row scanner's diagnostic stands.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_maxsi_computed_deferred(%x_ptr: !tt.ptr<i32>) {
    %r0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r1 = tt.expand_dims %r0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %offs = tt.broadcast %r1 : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<32x16x!tt.ptr<i32>, #blocked>
    %ap = tt.addptr %sp, %offs : tensor<32x16x!tt.ptr<i32>, #blocked>, tensor<32x16xi32, #blocked>
    %x = tt.load %ap : tensor<32x16x!tt.ptr<i32>, #blocked>
    %one = arith.constant dense<1> : tensor<32x16xi32, #blocked>
    %computed = arith.addi %x, %one : tensor<32x16xi32, #blocked>
    // expected-error @+1 {{rank-2 axis=1 i32 product/max/min/bitwise requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%computed) ({
    ^bb0(%a: i32, %b: i32):
      %m = arith.maxsi %a, %b : i32
      tt.reduce.return %m : i32
    }) {axis = 1 : i32} : (tensor<32x16xi32, #blocked>) -> tensor<32xi32, #slice1>
    tt.return
  }
}

// -----
// f16 addf now reduces in its own type, but only over a DIRECT unmasked load —
// the computed-cone re-emitter is f32-only. A constant source is a computed
// cone, so this stays rejected, with the message naming the real restriction.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_f16(%x_ptr: !tt.ptr<f16>) {
    %x = arith.constant dense<0.0> : tensor<16x32xf16, #blocked>
    // expected-error @+1 {{rank-2 axis=1 f16/bf16 reduce requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f16, %b: f16):
      %s = arith.addf %a, %b : f16
      tt.reduce.return %s : f16
    }) {axis = 1 : i32} : (tensor<16x32xf16, #blocked>) -> tensor<16xf16, #slice1>
    tt.return
  }
}

// -----
// Axis=0 f32 product starts with the direct-unmasked load envelope. A constant
// source remains behind a stable preflight diagnostic.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_mul_axis0_deferred(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<16x32xf32, #blocked>
    // expected-error @+1 {{rank-2 axis=0 f32 product requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.mulf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<16x32xf32, #blocked>) -> tensor<32xf32, #slice0>
    tt.return
  }
}

// -----
// i32 axis=0 product deliberately starts with the direct-unmasked load
// envelope. A constant tile stays behind a stable preflight diagnostic.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_muli_axis0_computed_deferred() {
    %x = arith.constant dense<1> : tensor<16x32xi32, #blocked>
    // expected-error @+1 {{rank-2 axis=0 i32 product requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %product = arith.muli %a, %b : i32
      tt.reduce.return %product : i32
    }) {axis = 0 : i32} : (tensor<16x32xi32, #blocked>) -> tensor<32xi32, #slice0>
    tt.return
  }
}

// -----
// A direct but masked load remains outside the initial axis=0 product slice.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_muli_axis0_masked_deferred(%x_ptr: !tt.ptr<i32>) {
    %offsets = arith.constant dense<0> : tensor<16x32xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<16x32xi1, #blocked>
    %other = arith.constant dense<1> : tensor<16x32xi32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<16x32x!tt.ptr<i32>, #blocked>
    %ap = tt.addptr %sp, %offsets : tensor<16x32x!tt.ptr<i32>, #blocked>, tensor<16x32xi32, #blocked>
    %x = tt.load %ap, %mask, %other : tensor<16x32x!tt.ptr<i32>, #blocked>
    // expected-error @+1 {{rank-2 axis=0 i32 product requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: i32, %b: i32):
      %product = arith.muli %a, %b : i32
      tt.reduce.return %product : i32
    }) {axis = 0 : i32} : (tensor<16x32xi32, #blocked>) -> tensor<32xi32, #slice0>
    tt.return
  }
}

// -----
// Rank-2 axis=1 f32 product starts with the same direct-unmasked envelope as
// i32 product. A computed source is rejected before conversion mutates it.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_mulf_axis1_computed_deferred() {
    %x = arith.constant dense<1.0> : tensor<16x32xf32, #blocked>
    // expected-error @+1 {{rank-2 axis=1 f32 product requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %product = arith.mulf %a, %b : f32
      tt.reduce.return %product : f32
    }) {axis = 1 : i32} : (tensor<16x32xf32, #blocked>) -> tensor<16xf32, #slice1>
    tt.return
  }
}

// -----
// A masked load remains outside the initial axis=1 f32 product slice.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_mulf_axis1_masked_deferred(%x_ptr: !tt.ptr<f32>) {
    %offsets = arith.constant dense<0> : tensor<16x32xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<16x32xi1, #blocked>
    %other = arith.constant dense<1.0> : tensor<16x32xf32, #blocked>
    %sp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked>
    %ap = tt.addptr %sp, %offsets : tensor<16x32x!tt.ptr<f32>, #blocked>, tensor<16x32xi32, #blocked>
    %x = tt.load %ap, %mask, %other : tensor<16x32x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{rank-2 axis=1 f32 product requires a direct unmasked tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %product = arith.mulf %a, %b : f32
      tt.reduce.return %product : f32
    }) {axis = 1 : i32} : (tensor<16x32xf32, #blocked>) -> tensor<16xf32, #slice1>
    tt.return
  }
}

// -----
// A multi-value `tl.reduce((a, b), axis, combine)` that is neither of the two
// canonical paired forms (rank-1 argmax, rank-2 axis=1 argmin) has no lowering:
// `ReduceLowering::matchAndRewrite` bails on `op.getSrcs().size() != 1`.
//
// That decline used to happen INSIDE `applyFullConversion`, which SIGSEGVs the
// process during the failed conversion's rollback. The shape below is the one
// that reached it: a tuple combine whose FIRST region op is `arith.maxnumf`
// looks exactly like a supported single-value max reduce to the combine
// allowlist, so it sailed past every earlier gate. The same reduce written with
// `tl.where` was rejected all along, by accident, because `arith.cmpf` is not
// an admitted combine -- i.e. the crashing shape was precisely the one that
// looked supported.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_tuple_maxnum_pair(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %y = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    // expected-error @+1 {{multi-value tt.reduce is implemented only for the canonical rank-1 argmax and rank-2 axis=1 argmin pairs}}
    %r:2 = "tt.reduce"(%x, %y) ({
    ^bb0(%a1: f32, %b1: f32, %a2: f32, %b2: f32):
      %m1 = arith.maxnumf %a1, %a2 : f32
      %m2 = arith.maxnumf %b1, %b2 : f32
      tt.reduce.return %m1, %m2 : f32, f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> (f32, f32)
    tt.return
  }
}
