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
  tt.func public @reject_join(%x: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    // expected-error @+1 {{tt.join is not implemented}}
    %j = tt.join %v, %v : tensor<128xf32, #blocked> -> tensor<128x2xf32, #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_split(%x: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<128x2xf32, #blocked>
    // expected-error @+1 {{tt.split is not implemented}}
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
  tt.func public @reject_assert(%x: !tt.ptr<f32>) {
    %c = arith.constant dense<true> : tensor<128xi1, #blocked>
    // expected-error @+1 {{tl.device_assert is not implemented}}
    tt.assert %c, "boom" : tensor<128xi1, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_print(%x: !tt.ptr<f32>) {
    %v = arith.constant dense<1.0> : tensor<128xf32, #blocked>
    // expected-error @+1 {{tl.device_print is not implemented}}
    tt.print " v: " {hex = false, isSigned = array<i32: 0>} : %v : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// A `torch.bool` output arrives as `!tt.ptr<i1>`, which Triton bitcasts to
// `!tt.ptr<i8>` at every access. `findBaseMemref` chases through that bitcast
// deliberately (the f32 atomic min/max expansion needs the ORIGINAL buffer to
// pick its MSL atomic pointer type), so the store lands on an i1-element memref
// holding an i8 value and fails to legalize — taking the process with it.
// Rejecting the ARGUMENT type is narrower than rejecting the bitcast: it names
// the dtype the caller chose, and cannot catch the atomic bitcasts, whose
// arguments are `!tt.ptr<f32>`.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{argument #0 is a bool (i1) tensor, which is not supported}}
  tt.func public @reject_bool_arg(%o: !tt.ptr<i1>) {
    tt.return
  }
}

// -----

// `tt.make_range` is a per-element index. MakeRangeLowering can decompose one
// out of a blocked layout (rank-1), a slice-of-blocked (rank-2) or a
// slice-of-slice-of-blocked (rank-3); a `#ttg.linear` layout — what reshaping
// an arange into a hypercube axis produces — has no per-element index at all.
//
// This used to emit a constant 0 and compile clean, so every element of
// `tl.arange(0, 2)` reshaped into a rank-3 axis read back as 0. A silent wrong
// answer, not a missing feature.
#linear = #ttg.linear<{register = [], lane = [[0], [1], [0], [0], [0]], warp = [[0], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_linear_layout_make_range(%o: !tt.ptr<i32>) {
    // expected-error @+1 {{tl.arange has no per-element index under this layout}}
    %r = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear>
    tt.return
  }
}
