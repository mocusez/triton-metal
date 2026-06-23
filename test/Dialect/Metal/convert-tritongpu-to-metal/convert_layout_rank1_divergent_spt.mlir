// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Mixed pointer alignment in one elementwise kernel: aligned x/y
// (tt.divisibility = 16 -> sizePerThread = [4]) and an unaligned out
// (sizePerThread = [1], e.g. an MPS tensor sliced to base[7:]). The Triton
// frontend bridges the spt=4 compute value to the spt=1 store with
// `ttg.convert_layout #blocked -> #blocked1`. Before normalizeRank1DivergentCvts
// this hit the L1d3 "broader staged-transpose deferred" hard error; now the
// cvt's producer cone (loads, addf, addptr/splat/make_range, mask cmpi) is
// rewritten to the spt=1 encoding so loads and store share strided indexing and
// the cvt collapses to an identity (passthrough). See driver.py / the MPS
// storage-offset test for the runtime motivation.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_mixed_align(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32>, %n: i32 {tt.divisibility = 16 : i32}) {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %off = arith.muli %pid, %c1024_i32 : i32
    %r0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %r1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %s0 = tt.splat %off : i32 -> tensor<1024xi32, #blocked>
    %s1 = tt.splat %off : i32 -> tensor<1024xi32, #blocked1>
    %o0 = arith.addi %s0, %r0 : tensor<1024xi32, #blocked>
    %o1 = arith.addi %s1, %r1 : tensor<1024xi32, #blocked1>
    %ns = tt.splat %n : i32 -> tensor<1024xi32, #blocked>
    %ns1 = tt.splat %n : i32 -> tensor<1024xi32, #blocked1>
    %m0 = arith.cmpi slt, %o0, %ns : tensor<1024xi32, #blocked>
    %m1 = arith.cmpi slt, %o1, %ns1 : tensor<1024xi32, #blocked1>
    %xs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %xp = tt.addptr %xs, %o0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %xv = tt.load %xp, %m0 : tensor<1024x!tt.ptr<f32>, #blocked>
    %ys = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %yp = tt.addptr %ys, %o0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %yv = tt.load %yp, %m0 : tensor<1024x!tt.ptr<f32>, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
    %op = tt.addptr %os, %o1 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
    %sum = arith.addf %xv, %yv : tensor<1024xf32, #blocked>
    %cvt = ttg.convert_layout %sum : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
    tt.store %op, %cvt, %m1 : tensor<1024x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// The pass must succeed (previously a hard error) and emit a Metal kernel; no
// ttg.convert_layout survives the normalization + passthrough.
// CHECK: metal.kernel
// CHECK-NOT: ttg.convert_layout
// CHECK: metal.return
