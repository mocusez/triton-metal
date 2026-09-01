// RUN: triton-metal-opt --convert-tritongpu-to-metal --verify-diagnostics --split-input-file %s
//
// Triton ops the Metal backend does not implement must be rejected up front by
// `validateUnsupportedOpsRejected`, BEFORE any conversion runs.
//
// This is not cosmetic. Reaching `applyFullConversion` with no pattern printed
// "failed to legalize operation ..." and then took the PROCESS down in
// failed-conversion teardown: tt.assert and tt.gather with SIGSEGV (exit 139),
// tt.join with an abort (exit 134). A crash gives a kernel author no way to
// tell "this backend can't do that yet" from "the compiler is broken", and it
// kills the whole pytest process along the way.
//
// Each case below therefore pins BOTH that the construct is refused and the
// wording that tells the author what to do instead. Implementing one of these
// means deleting its entry in the validator and its case here.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_map_elementwise_pack2() {
    %v = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    // expected-error @+1 {{tl.map_elementwise is implemented for pack=1 only}}
    %r = "tt.map_elementwise"(%v, %v) <{pack = 2 : i32}> ({
    ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
      %x = arith.addf %a0, %b0 : f32
      %y = arith.addf %a1, %b1 : f32
      tt.map_elementwise.return %x, %y : f32, f32
    }) : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.print "r: " {hex = false, isSigned = array<i32: 0>} : %r : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_map_elementwise_multiblock() {
    %v = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    // expected-error @+1 {{tl.map_elementwise requires a single-block scalar region}}
    %r = "tt.map_elementwise"(%v, %v) <{pack = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %take_a = arith.cmpf ogt, %a, %b : f32
      cf.cond_br %take_a, ^bb1(%a : f32), ^bb1(%b : f32)
    ^bb1(%selected: f32):
      tt.map_elementwise.return %selected : f32
    }) : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.print "r: " {hex = false, isSigned = array<i32: 0>} : %r : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_raw_descriptor_reduce(%desc: !tt.tensordesc<16xi32>, %src: tensor<16xi32, #blocked>) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{tt.descriptor_reduce must be eliminated by triton-rewrite-tensor-descriptor-to-pointer before Metal conversion}}
    tt.descriptor_reduce add, %desc[%c0], %src : !tt.tensordesc<16xi32>, tensor<16xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // A pointer loaded out of memory and dereferenced -- `p = tl.load(pp)` then
  // `tl.load(p.to(tl.pointer_type(tl.float32)) + i)`. This one reached the
  // conversion because the scalar i64 load it starts with is supported; only
  // the cast has no pattern.
  tt.func public @reject_int_to_ptr(%pp: !tt.ptr<i64>) {
    %p = tt.load %pp : !tt.ptr<i64>
    // expected-error @+1 {{casting an integer to a pointer is not supported}}
    %q = tt.int_to_ptr %p : i64 -> !tt.ptr<f32>
    %v = tt.load %q : !tt.ptr<f32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_ptr_to_int(%x: !tt.ptr<f32>, %o: !tt.ptr<i64>) {
    // expected-error @+1 {{casting a pointer to an integer is not supported}}
    %n = tt.ptr_to_int %x : !tt.ptr<f32> -> i64
    tt.store %o, %n : !tt.ptr<i64>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_join(%x: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    // expected-error @+1 {{tt.join is implemented for a rank-1 pair joined into an [N, 2] tile of at most one element per thread}}
    %j = tt.join %v, %v : tensor<128xf32, #blocked> -> tensor<128x2xf32, #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_split(%x: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<128x2xf32, #blocked>
    // expected-error @+1 {{tt.split is implemented for an [N, 2] tile of at most one element per thread}}
    %a, %b = tt.split %v : tensor<128x2xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----

// A rank-1 axis-0 gather IS implemented (GatherLowering); a rank-2 one is not,
// and the message says which part is missing rather than claiming tt.gather as
// a whole is absent.
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_rank2_gather(%x: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<8x32xf32, #blocked2>
    %i = arith.constant dense<0> : tensor<8x32xi32, #blocked2>
    // expected-error @+1 {{tl.gather is implemented for a rank-1 gather along axis 0 only}}
    %g = tt.gather %v[%i] {axis = 1 : i32} : (tensor<8x32xf32, #blocked2>, tensor<8x32xi32, #blocked2>) -> tensor<8x32xf32, #blocked2>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_unknown_extern(%out: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    // expected-error @+1 {{tt.extern_elementwise symbol '__metal_missing' has no Metal intrinsic}}
    %r = tt.extern_elementwise %v {libname = "", libpath = "", pure = true, symbol = "__metal_missing"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %ptr, %r : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_extern_arity(%out: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    // expected-error @+1 {{tt.extern_elementwise symbol '__metal_atan2' expects 2 operands, got 1}}
    %r = tt.extern_elementwise %v {libname = "", libpath = "", pure = true, symbol = "__metal_atan2"} : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %ptr, %r : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #parent}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_rank1_slice_scan(%out: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<32xf32, #slice>
    // expected-error @+1 {{Metal backend: rank-1 scan requires a blocked layout}}
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.scan.return %sum : f32
    }) : (tensor<32xf32, #slice>) -> tensor<32xf32, #slice>
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #slice>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #slice>
    %ptr = tt.addptr %base, %range : tensor<32x!tt.ptr<f32>, #slice>, tensor<32xi32, #slice>
    tt.store %ptr, %scan : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #parent}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_rank1_slice_gather() {
    %v = arith.constant dense<1.0> : tensor<32xf32, #slice>
    %i = arith.constant dense<0> : tensor<32xi32, #slice>
    // expected-error @+1 {{Metal backend: rank-1 gather requires a blocked layout}}
    %g = tt.gather %v[%i] {axis = 0 : i32} : (tensor<32xf32, #slice>, tensor<32xi32, #slice>) -> tensor<32xf32, #slice>
    tt.print "g: " {hex = false, isSigned = array<i32: 0>} : %g : tensor<32xf32, #slice>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_rank1_gather_i64_indices() {
    %v = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    %i = arith.constant dense<0> : tensor<128xi64, #blocked>
    // expected-error @+1 {{Metal backend: rank-1 gather requires i32 indices}}
    %g = tt.gather %v[%i] {axis = 0 : i32} : (tensor<128xf32, #blocked>, tensor<128xi64, #blocked>) -> tensor<128xf32, #blocked>
    tt.print "g: " {hex = false, isSigned = array<i32: 0>} : %g : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_rank1_gather_extent_mismatch() {
    %v = arith.constant dense<1.0> : tensor<64xf32, #blocked>
    %i = arith.constant dense<0> : tensor<32xi32, #blocked>
    // expected-error @+1 {{Metal backend: rank-1 gather source, indices, and result must share one non-empty extent}}
    %g = tt.gather %v[%i] {axis = 0 : i32} : (tensor<64xf32, #blocked>, tensor<32xi32, #blocked>) -> tensor<32xf32, #blocked>
    tt.print "g: " {hex = false, isSigned = array<i32: 0>} : %g : tensor<32xf32, #blocked>
    tt.return
  }
}

// -----

// A `#ttg.linear` `tt.make_range` USED to be refused here, because
// MakeRangeLowering could only decompose a blocked layout (rank-1), a
// slice-of-blocked (rank-2) or a slice-of-slice-of-blocked (rank-3), and
// emitting a constant 0 for anything else made every element of `tl.arange(0,2)`
// reshaped into a rank-3 axis read back as 0 — a silent wrong answer.
//
// It is no longer refused: `planLinearRange` evaluates the layout's basis
// vectors directly, which is what a rank-3 `tl.permute` needs (Triton folds the
// whole permutation into the load's index layout and leaves the reshape and the
// transpose as flat identities). Coverage moved to
// `make_range_linear_layout.mlir`; the rejection above it stays for a layout
// neither path can decode.
