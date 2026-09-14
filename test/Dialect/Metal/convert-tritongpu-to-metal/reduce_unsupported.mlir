// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Session L3 (the implementation notes)
// negative pre-pass: tt.reduce inputs that fall outside the L3 envelope are
// rejected with the spec-mandated error strings. Phase C (the actual tree-
// reduction lowering) is deferred to L3a per the spec's honest divergence
// policy; this fixture exercises the pre-pass rejection gate.

// -----
// Multi-band rank-N reduce is implemented through full-tile phases, but the
// typed ping-pong pool remains bounded by Metal's 32 KiB threadgroup budget.
// This one f32 materialization alone needs 64 KiB, so it is rejected with the
// required and limit byte counts instead of reaching legalization.
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#slice3d = #ttg.slice<{dim = 0, parent = #blocked3d}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{full-tile exchange scratch needs 65536 bytes, exceeding the 32768-byte budget}}
  tt.func public @reduce_rank3_over_budget(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<0.0> : tensor<128x64x2xf32, #blocked3d>
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 2 : i32} : (tensor<128x64x2xf32, #blocked3d>) -> tensor<128x64xf32, #ttg.slice<{dim = 2, parent = #blocked3d}>>
    tt.return
  }
}

// -----
// A rank-1 f32 product folds straight out of device memory, so a COMPUTED
// source remains behind a stable preflight diagnostic. A masked load is no
// longer one of these: it is covered as a positive in
// rank1_reduce_addf_block32.mlir.
#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_mulf_rank1_computed_deferred() {
    %x = arith.constant dense<1.0> : tensor<32xf32, #blocked1d>
    // expected-error @+1 {{rank-1 f32 product requires a direct tt.load}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %product = arith.mulf %a, %b : f32
      tt.reduce.return %product : f32
    }) {axis = 0 : i32} : (tensor<32xf32, #blocked1d>) -> f32
    tt.return
  }
}

// -----
// A tuple reduce whose per-thread band is longer than the fold is implemented
// for. Above one element per thread a thread folds its own band first (see
// rank1_reduce_tuple.mlir), and that band is capped: a four-thousand-element
// block over one warp gives each thread a hundred and twenty-eight elements.
//
// A near-miss argmax that breaks ties towards the HIGHER index used to sit
// here, and so did a hundred-and-twenty-eight-element block over one warp.
// Both are lowered now, which is the point of that fixture.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_tuple_band_too_long(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<4096xf32, #blocked>
    %y = arith.constant dense<2.0> : tensor<4096xf32, #blocked>
    // expected-error @+1 {{a general tuple reduce is implemented up to sixty-four elements per thread}}
    %r:2 = "tt.reduce"(%x, %y) ({
    ^bb0(%a1: f32, %b1: f32, %a2: f32, %b2: f32):
      %m1 = arith.minnumf %a1, %a2 : f32
      %m2 = arith.maxnumf %b1, %b2 : f32
      tt.reduce.return %m1, %m2 : f32, f32
    }) {axis = 0 : i32} : (tensor<4096xf32, #blocked>, tensor<4096xf32, #blocked>) -> (f32, f32)
    tt.return
  }
}

// -----
// A tuple reduce whose slots are f16. Each lockstep buffer is one 32-bit slot,
// and only f32 and i32 have a spelling on both sides of its ui32 storage
// bridge; a narrower type would need its own staging decision, so it stays
// refused rather than silently widened.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_tuple_f16_deferred(%x_ptr: !tt.ptr<f16>) {
    %x = arith.constant dense<1.0> : tensor<64xf16, #blocked>
    %y = arith.constant dense<2.0> : tensor<64xf16, #blocked>
    // expected-error @+1 {{a general tuple reduce carries f32 and i32 values only}}
    %r:2 = "tt.reduce"(%x, %y) ({
    ^bb0(%a1: f16, %b1: f16, %a2: f16, %b2: f16):
      %m1 = arith.minnumf %a1, %a2 : f16
      %m2 = arith.maxnumf %b1, %b2 : f16
      tt.reduce.return %m1, %m2 : f16, f16
    }) {axis = 0 : i32} : (tensor<64xf16, #blocked>, tensor<64xf16, #blocked>) -> (f16, f16)
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
// A direct axis=1 sum inside a state-carrying user loop is supported only when
// its complete pointer-offset cone can be replayed at logical (row, column).
// `arith.maxsi` has an ordinary Metal tensor lowering but is deliberately not
// part of that address evaluator, so this adjacent shape must fail in the
// preflight instead of declining after full conversion has mutated the module.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_axis1_loop_address_unsupported(%x_ptr: !tt.ptr<f32>, %T: i32, %initial: f32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %tile = arith.constant 512 : i32
    %h0 = tt.splat %initial : f32 -> tensor<32xf32, #slice1>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice1>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #slice1> -> tensor<32x1xi32, #blocked>
    %row_scale = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %row_offsets = arith.muli %rows_2d, %row_scale : tensor<32x1xi32, #blocked>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %cols_full = tt.broadcast %cols_2d : tensor<1x16xi32, #blocked> -> tensor<32x16xi32, #blocked>
    %x_base = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>, #blocked>
    %result = scf.for %t = %c0 to %T step %c1 iter_args(%h = %h0) -> (tensor<32xf32, #slice1>) : i32 {
      %trip_base = arith.muli %t, %tile : i32
      %trip_splat = tt.splat %trip_base : i32 -> tensor<32x1xi32, #blocked>
      %row_bases = arith.addi %trip_splat, %row_offsets : tensor<32x1xi32, #blocked>
      %row_bases_full = tt.broadcast %row_bases : tensor<32x1xi32, #blocked> -> tensor<32x16xi32, #blocked>
      %offsets = arith.addi %row_bases_full, %cols_full : tensor<32x16xi32, #blocked>
      %zero = arith.constant dense<0> : tensor<32x16xi32, #blocked>
      %unsupported = arith.maxsi %offsets, %zero : tensor<32x16xi32, #blocked>
      %x_addr = tt.addptr %x_base, %unsupported : tensor<32x16x!tt.ptr<f32>, #blocked>, tensor<32x16xi32, #blocked>
      %x = tt.load %x_addr : tensor<32x16x!tt.ptr<f32>, #blocked>
      // expected-error @+1 {{rank-2 axis=1 loop-varying direct-load address has an unsupported producer}}
      %sum = "tt.reduce"(%x) ({
      ^bb0(%a: f32, %b: f32):
        %add = arith.addf %a, %b : f32
        tt.reduce.return %add : f32
      }) {axis = 1 : i32} : (tensor<32x16xf32, #blocked>) -> tensor<32xf32, #slice1>
      %next = arith.addf %h, %sum : tensor<32xf32, #slice1>
      scf.yield %next : tensor<32xf32, #slice1>
    }
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
// A multi-value reduce along axis 1. The tuple lowering is a cross-THREAD tree
// over one rank-1 tile; a rank-2 axis=1 pair is a per-row fold, which only the
// canonical argmin form has. This is the shape whose single-value sibling the
// row scanner takes, so the refusal has to name the axis, not the arity.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_tuple_axis1_deferred(%x_ptr: !tt.ptr<f32>) {
    %x = arith.constant dense<1.0> : tensor<8x16xf32, #blocked>
    %y = arith.constant dense<2.0> : tensor<8x16xf32, #blocked>
    // expected-error @+1 {{a general tuple reduce is implemented for axis 0 only}}
    %r:2 = "tt.reduce"(%x, %y) ({
    ^bb0(%a1: f32, %b1: f32, %a2: f32, %b2: f32):
      %m1 = arith.minnumf %a1, %a2 : f32
      %m2 = arith.maxnumf %b1, %b2 : f32
      tt.reduce.return %m1, %m2 : f32, f32
    }) {axis = 1 : i32} : (tensor<8x16xf32, #blocked>, tensor<8x16xf32, #blocked>) -> (tensor<8xf32, #slice1>, tensor<8xf32, #slice1>)
    tt.return
  }
}

// -----
// A tuple combine containing an op the replayer has no scalar spelling for.
// The tree clones the region at every merge, so an op it cannot clone is not a
// combine it can perform -- and the pre-pass has to say so, because declining
// inside `applyFullConversion` SIGSEGVs the process during the failed
// conversion's rollback rather than raising.
//
// A tuple combine whose FIRST region op is `arith.maxnumf` used to reach
// exactly that kill: it looked like a supported single-value max reduce to the
// combine allowlist and sailed past every earlier gate. It is now lowered --
// see `rank1_reduce_tuple.mlir` -- so the crash-safety case is carried by a
// region that genuinely has no lowering.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_tuple_unreplayable_combine(%x_ptr: !tt.ptr<i32>) {
    %x = arith.constant dense<1> : tensor<128xi32, #blocked>
    %y = arith.constant dense<2> : tensor<128xi32, #blocked>
    // expected-error @+1 {{the combine region must be one block of replayable scalar ops over its own arguments}}
    %r:2 = "tt.reduce"(%x, %y) ({
    ^bb0(%a1: i32, %b1: i32, %a2: i32, %b2: i32):
      %pop = math.ctpop %a1 : i32
      %m1 = arith.addi %pop, %a2 : i32
      %m2 = arith.addi %b1, %b2 : i32
      tt.reduce.return %m1, %m2 : i32, i32
    }) {axis = 0 : i32} : (tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> (i32, i32)
    tt.return
  }
}


// -----
// A rank-2 axis=0 reduce with more output columns than threads, pushed back
// into a 2-D tile (`x - sum[None, :]`) and NEVER becoming a rank-1 value.
// Re-expansion itself is lowered now -- see
// rank2_axis0_reduce_reexpanded.mlir -- but the per-thread output geometry
// still has to come from a rank-1 consumer carrying exactly these columns, and
// this kernel has none.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @colsum_reexpanded_without_vector(%in: !tt.ptr<f32>) {
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %cN = arith.constant dense<64> : tensor<8x1xi32, #blocked>
    %rowsN = arith.muli %rows2, %cN : tensor<8x1xi32, #blocked>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice0>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #slice0> -> tensor<1x64xi32, #blocked>
    %sp = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #blocked>
    %ap0 = tt.addptr %sp, %rowsN : tensor<8x1x!tt.ptr<f32>, #blocked>, tensor<8x1xi32, #blocked>
    %apb = tt.broadcast %ap0 : tensor<8x1x!tt.ptr<f32>, #blocked> -> tensor<8x64x!tt.ptr<f32>, #blocked>
    %colsb = tt.broadcast %cols2 : tensor<1x64xi32, #blocked> -> tensor<8x64xi32, #blocked>
    %ap = tt.addptr %apb, %colsb : tensor<8x64x!tt.ptr<f32>, #blocked>, tensor<8x64xi32, #blocked>
    %tile = tt.load %ap : tensor<8x64x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{rank-1 consumer carrying exactly those columns}}
    %s = "tt.reduce"(%tile) ({
    ^bb0(%a: f32, %b: f32):
      %add = arith.addf %a, %b : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<8x64xf32, #blocked>) -> tensor<64xf32, #slice0>
    %s2 = tt.expand_dims %s {axis = 0 : i32} : tensor<64xf32, #slice0> -> tensor<1x64xf32, #blocked>
    %sb = tt.broadcast %s2 : tensor<1x64xf32, #blocked> -> tensor<8x64xf32, #blocked>
    %dev = arith.subf %tile, %sb : tensor<8x64xf32, #blocked>
    tt.store %ap, %dev : tensor<8x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----
// Canonical blocked rank-1 single-result reducers keep the direct tree bounded
// to E=BLOCK/tpb <= 64. E=128 must be rejected before conversion declines and
// leaves `tt.reduce` behind for failed legalization rollback.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank1_e128_geometry_deferred(%x_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
    %x = tt.load %x_addr : tensor<4096x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{rank-1 reduce E = BLOCK/tpb outside supported envelope [1, 64]}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<4096xf32, #blocked>) -> f32
    tt.return
  }
}

// -----
// The legacy rank-1 masked path only replays the canonical loop-offset
// `offset < runtime_bound` predicate. A compound max mask must fail in the
// preflight with the named reason instead of reaching conversion rollback.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank1_masked_compound_max_deferred(%x_ptr: !tt.ptr<f32>, %n: i32, %row: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %col_ok = arith.cmpi slt, %offsets, %n_splat : tensor<128xi32, #blocked>
    %zero_i32 = arith.constant 0 : i32
    %row_ok_scalar = arith.cmpi sge, %row, %zero_i32 : i32
    %row_ok = tt.splat %row_ok_scalar : i1 -> tensor<128xi1, #blocked>
    %mask = arith.andi %col_ok, %row_ok : tensor<128xi1, #blocked>
    %other = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{rank-1 reduce B2.3 masked: mask not cmpi slt}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}

// -----
// The legacy single-load-chain rank-1 reduce replays addresses as a contiguous
// logical range plus scalar pointer offsets. A tensor stride in the load offset
// must be rejected before the direct load-chain path silently reads `offsets`.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank1_strided_address_deferred(%x_ptr: !tt.ptr<f32>, %stride: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<128xi32, #blocked>
    %strided = arith.muli %offsets, %stride_splat : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %strided : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr : tensor<128x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{rank-1 reduce address cannot be replayed by the single-load-chain path}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}

// -----
// Looking only at the outer addptr is not enough: this spells a contiguous
// outer range on top of an inner tensor offset, which the legacy path would
// otherwise drop when it accumulates only scalar base terms.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank1_nested_tensor_address_deferred(%x_ptr: !tt.ptr<f32>, %stride: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<128xi32, #blocked>
    %inner_offsets = arith.muli %offsets, %stride_splat : tensor<128xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %inner_addr = tt.addptr %x_splat, %inner_offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_addr = tt.addptr %inner_addr, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr : tensor<128x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{rank-1 reduce address cannot be replayed by the single-load-chain path}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}

// -----
// A bare splat pointer has no per-element addptr offset for the legacy address
// replay formula.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank1_bare_splat_address_deferred(%x_ptr: !tt.ptr<f32>) {
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x = tt.load %x_splat : tensor<128x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{rank-1 reduce address cannot be replayed by the single-load-chain path}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}

// -----
// The contiguous masked sum specialization replays each load lane from the
// original index cone. Bitwise tensor index producers must fail preflight with
// a named diagnostic rather than reaching scalarization asserts.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank1_contiguous_masked_load_bitand_deferred(%x_ptr: !tt.ptr<f32>, %n: i32, %k: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %k_splat = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %load_index = arith.andi %offsets, %k_splat : tensor<128xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<128xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %load_index : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{contiguous masked reduce cannot replay its load index}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}

// -----
// The same contiguous masked sum path also replays the mask index. A non-linear
// mask index must be rejected explicitly before conversion attempts rollback.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank1_contiguous_masked_mask_bitand_deferred(%x_ptr: !tt.ptr<f32>, %n: i32, %k: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %k_splat = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %mask_index = arith.andi %offsets, %k_splat : tensor<128xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %mask_index, %n_splat : tensor<128xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{contiguous masked reduce cannot replay its mask index}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}

// -----
// Every level of the pointer chain is proved, not just the outermost one. The
// inner level's bitwise index has no scalarization, so the whole chain must be
// refused by name instead of silently dropping that level's term.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_rank1_contiguous_masked_inner_level_bitand_deferred(%x_ptr: !tt.ptr<f32>, %n: i32, %k: i32) {
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %k_splat = tt.splat %k : i32 -> tensor<128xi32, #blocked>
    %row_off = arith.andi %offsets, %k_splat : tensor<128xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<128xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<128xf32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %row_addr = tt.addptr %x_splat, %row_off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x_addr = tt.addptr %row_addr, %offsets : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %x_addr, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    // expected-error @+1 {{contiguous masked reduce cannot replay its load index}}
    %r = "tt.reduce"(%x) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<128xf32, #blocked>) -> f32
    tt.return
  }
}
