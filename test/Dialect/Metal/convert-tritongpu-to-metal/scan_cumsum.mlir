// RUN: triton-metal-opt --convert-tritongpu-to-metal --split-input-file %s | FileCheck %s
//
// W-C: `tt.scan` (cumsum) lowering. `tl.cumsum(v, axis=0)` feeding an inverse-CDF
// `tl.min` reduce lowers to a threadgroup prefix-sum: two `metal.threadgroup_alloca`
// f32 buffers (inbuf + scanbuf) filled from the scan-input cone, a
// `metal.threadgroup_prefix_sum` (Hillis-Steele + iv-carry template), and the
// consuming min-reduce reads scanbuf[idx] per element. See ScanLowering and
// metal-speculative-decoding-plan.md (W-C).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#loc = loc("/tmp/dump_scan_ttgir.py":6:1)
#loc8 = loc("/tmp/dump_scan_ttgir.py":11:10)
#loc9 = loc(unknown)
#loc16 = loc("/tmp/dump_scan_ttgir.py":14:27)
#loc19 = loc("in_ptr"(#loc))
#loc20 = loc("out_ptr"(#loc))
#loc21 = loc("V"(#loc))
#loc22 = loc("target"(#loc))
#loc29 = loc("cs"(#loc8))
#loc34 = loc(callsite(#loc9 at #loc16))
#loc36 = loc(callsite(#loc9 at #loc29))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @k(%in_ptr: !tt.ptr<f32> loc("in_ptr"(#loc)), %out_ptr: !tt.ptr<i32> loc("out_ptr"(#loc)), %V: i32 loc("V"(#loc)), %target: f32 loc("target"(#loc))) attributes {noinline = false} {
    %v = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked> loc(#loc23)
    %b = tt.get_program_id x : i32 loc(#loc24)
    %idx = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked> loc(#loc25)
    %mask = tt.splat %V : i32 -> tensor<1024xi32, #blocked> loc(#loc26)
    %mask_0 = arith.cmpi slt, %idx, %mask : tensor<1024xi32, #blocked> loc(#loc26)
    %v_1 = arith.muli %b, %V : i32 loc(#loc27)
    %v_2 = tt.addptr %in_ptr, %v_1 : !tt.ptr<f32>, i32 loc(#loc28)
    %v_3 = tt.splat %v_2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc28)
    %v_4 = tt.addptr %v_3, %idx : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked> loc(#loc28)
    %v_5 = tt.load %v_4, %mask_0, %v : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc23)
    %cs = "tt.scan"(%v_5) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%cs_8: f32 loc(callsite(#loc9 at #loc29)), %cs_9: f32 loc(callsite(#loc9 at #loc29))):
      %cs_10 = arith.addf %cs_8, %cs_9 : f32 loc(#loc38)
      tt.scan.return %cs_10 : f32 loc(#loc35)
    }) : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked> loc(#loc35)
    %cond = tt.splat %target : f32 -> tensor<1024xf32, #blocked> loc(#loc30)
    %cond_6 = arith.cmpf oge, %cs, %cond : tensor<1024xf32, #blocked> loc(#loc30)
    %cond_7 = arith.andi %cond_6, %mask_0 : tensor<1024xi1, #blocked> loc(#loc31)
    %sel = arith.select %cond_7, %idx, %mask : tensor<1024xi1, #blocked>, tensor<1024xi32, #blocked> loc(#loc32)
    %0 = tt.addptr %out_ptr, %b : !tt.ptr<i32>, i32 loc(#loc14)
    %1 = "tt.reduce"(%sel) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32 loc(callsite(#loc9 at #loc16)), %arg5: i32 loc(callsite(#loc9 at #loc16))):
      %2 = arith.minsi %arg4, %arg5 : i32 loc(#loc37)
      tt.reduce.return %2 : i32 loc(#loc33)
    }) : (tensor<1024xi32, #blocked>) -> i32 loc(#loc33)
    tt.store %0, %1 : !tt.ptr<i32> loc(#loc18)
    tt.return loc(#loc)
  } loc(#loc)

  // Sub-tpb tile: BLOCK=16 with tpb=128 (num_warps=4). The buffers are sized to
  // a full tpb window and the tail [16, 128) is filled with the addf identity,
  // so the prefix-sum template is reused verbatim — no partial-window variant,
  // hence no threadgroup_barrier inside divergent control flow (UB on Metal).
  // Before this, BLOCK % tpb != 0 failed to legalize, which meant num_warps (a
  // perf knob) decided whether a small cumsum compiled at all, and a sub-warp
  // BLOCK was unreachable at every num_warps.
  tt.func public @k_sub_tpb(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    %off = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %m = tt.splat %n : i32 -> tensor<16xi32, #blocked>
    %m_0 = arith.cmpi slt, %off, %m : tensor<16xi32, #blocked>
    %x_1 = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %x_2 = tt.addptr %x_1, %off : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %x_3 = tt.load %x_2, %m_0, %x : tensor<16x!tt.ptr<f32>, #blocked>
    %o = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %o_0 = tt.addptr %o, %off : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %cs = "tt.scan"(%x_3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }) : (tensor<16xf32, #blocked>) -> tensor<16xf32, #blocked>
    tt.store %o_0, %cs, %m_0 : tensor<16x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // Same one-input/one-result add scan envelope, but reverse=true. The result
  // remains indexed in the original logical order: out[i] = sum(in[i:]).
  tt.func public @k_reverse(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %off = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %i = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %i_0 = tt.addptr %i, %off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x = tt.load %i_0 : tensor<1024x!tt.ptr<f32>, #blocked>
    %cs = "tt.scan"(%x) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.scan.return %s : f32
    }) : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_0 = tt.addptr %o, %off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_0, %cs : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // BFS frontier compaction shape: the scan input is computed from the OLD
  // value returned by a tensor atomic exchange. Replaying that cone would emit
  // a second atomic exchange; BLOCK==tpb lets the lowering use the already
  // converted per-lane scalar instead.
  tt.func public @k_atomic_old_value_i32_cumsum(%visited: !tt.ptr<i32>, %out: !tt.ptr<i32>, %n: i32) attributes {noinline = false} {
    %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %one = arith.constant dense<1> : tensor<128xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<128xi32, #blocked>
    %n_s = tt.splat %n : i32 -> tensor<128xi32, #blocked>
    %mask = arith.cmpi slt, %off, %n_s : tensor<128xi32, #blocked>
    %base = tt.splat %visited : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %addr = tt.addptr %base, %off : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    %old = tt.atomic_rmw exch, acq_rel, gpu, %addr, %one, %mask : (tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>, tensor<128xi1, #blocked>) -> tensor<128xi32, #blocked>
    %claimed = arith.cmpi eq, %old, %zero : tensor<128xi32, #blocked>
    %claimed_i32 = arith.extui %claimed : tensor<128xi1, #blocked> to tensor<128xi32, #blocked>
    %cs = "tt.scan"(%claimed_i32) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %s = arith.addi %a, %b : i32
      tt.scan.return %s : i32
    }) : (tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    %dst = tt.splat %out : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %dst_addr = tt.addptr %dst, %off : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %dst_addr, %cs, %mask : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
} loc(#loc)
#loc1 = loc("/tmp/dump_scan_ttgir.py":10:9)
#loc2 = loc("/tmp/dump_scan_ttgir.py":7:9)
#loc3 = loc("/tmp/dump_scan_ttgir.py":8:11)
#loc4 = loc("/tmp/dump_scan_ttgir.py":9:12)
#loc5 = loc("/tmp/dump_scan_ttgir.py":10:26)
#loc6 = loc("/tmp/dump_scan_ttgir.py":10:17)
#loc7 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":343:12)
#loc10 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":263:12)
#loc11 = loc("/tmp/dump_scan_ttgir.py":12:13)
#loc12 = loc("/tmp/dump_scan_ttgir.py":12:12)
#loc13 = loc("/tmp/dump_scan_ttgir.py":13:11)
#loc14 = loc("/tmp/dump_scan_ttgir.py":14:14)
#loc15 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":250:16)
#loc17 = loc("/Users/mocus/Code/triton/python/triton/language/standard.py":229:12)
#loc18 = loc("/tmp/dump_scan_ttgir.py":14:5)
#loc23 = loc("v"(#loc1))
#loc24 = loc("b"(#loc2))
#loc25 = loc("idx"(#loc3))
#loc26 = loc("mask"(#loc4))
#loc27 = loc("v"(#loc5))
#loc28 = loc("v"(#loc6))
#loc30 = loc("cond"(#loc11))
#loc31 = loc("cond"(#loc12))
#loc32 = loc("sel"(#loc13))
#loc33 = loc(callsite(#loc15 at #loc16))
#loc35 = loc(callsite(#loc7 at #loc29))
#loc37 = loc(callsite(#loc17 at #loc33))
#loc38 = loc(callsite(#loc10 at #loc35))

// CHECK-LABEL: metal.kernel k
// CHECK: metal.threadgroup_alloca : !metal.memref<1024 x f32>
// CHECK: metal.threadgroup_alloca : !metal.memref<1024 x f32>
// CHECK: metal.threadgroup_prefix_sum {{.*}} {block = 1024 : i64, tpb = 128 : i64}

// Sub-tpb: buffers and the prefix-sum window are the padded tpb size (128), not
// the tile size (16), and the fill selects the addf identity past element 16.
// CHECK-LABEL: metal.kernel k_sub_tpb
// CHECK: metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: arith.select
// CHECK: metal.threadgroup_prefix_sum {{.*}} {block = 128 : i64, tpb = 128 : i64}
// CHECK-NOT: tt.scan

// CHECK-LABEL: metal.kernel k_reverse
// CHECK: metal.threadgroup_prefix_sum {{.*}} {block = 1024 : i64, reverse, tpb = 128 : i64}
// CHECK-NOT: tt.scan

// CHECK-LABEL: metal.kernel k_atomic_old_value_i32_cumsum
// CHECK-COUNT-1: metal.atomic_rmw Xchg
// CHECK: metal.threadgroup_alloca : !metal.memref<128 x ui32>
// CHECK: metal.threadgroup_prefix_sum {{.*}} {block = 128 : i64, tpb = 128 : i64}
// CHECK-NOT: tt.scan

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @segmented_scan(%values_ptr: !tt.ptr<f32>, %flags_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<f32>, %flag_out_ptr: !tt.ptr<i32>, %n: i32) attributes {noinline = false} {
    %zero = arith.constant dense<0.000000e+00> : tensor<4096xf32, #blocked>
    %zero_i32 = arith.constant dense<0> : tensor<4096xi32, #blocked>
    %one_i32 = arith.constant dense<1> : tensor<4096xi32, #blocked>
    %offs = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %offs, %n_splat : tensor<4096xi32, #blocked>
    %vptr = tt.splat %values_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
    %vptr_0 = tt.addptr %vptr, %offs : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
    %fptr = tt.splat %flags_ptr : !tt.ptr<i32> -> tensor<4096x!tt.ptr<i32>, #blocked>
    %fptr_0 = tt.addptr %fptr, %offs : tensor<4096x!tt.ptr<i32>, #blocked>, tensor<4096xi32, #blocked>
    %vals = tt.load %vptr_0, %mask, %zero : tensor<4096x!tt.ptr<f32>, #blocked>
    %flags_i32 = tt.load %fptr_0, %mask, %zero_i32 : tensor<4096x!tt.ptr<i32>, #blocked>
    %flags = arith.cmpi eq, %flags_i32, %one_i32 : tensor<4096xi32, #blocked>
    %scan:2 = "tt.scan"(%vals, %flags) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%lhs_v: f32, %lhs_f: i1, %rhs_v: f32, %rhs_f: i1):
      %sum = arith.addf %lhs_v, %rhs_v : f32
      %out_v = arith.select %rhs_f, %rhs_v, %sum : f32
      %out_f = arith.ori %lhs_f, %rhs_f : i1
      tt.scan.return %out_v, %out_f : f32, i1
    }) : (tensor<4096xf32, #blocked>, tensor<4096xi1, #blocked>) -> (tensor<4096xf32, #blocked>, tensor<4096xi1, #blocked>)
    %optr = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
    %optr_0 = tt.addptr %optr, %offs : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
    %flag_i32 = arith.extui %scan#1 : tensor<4096xi1, #blocked> to tensor<4096xi32, #blocked>
    %flag_optr = tt.splat %flag_out_ptr : !tt.ptr<i32> -> tensor<4096x!tt.ptr<i32>, #blocked>
    %flag_optr_0 = tt.addptr %flag_optr, %offs : tensor<4096x!tt.ptr<i32>, #blocked>, tensor<4096xi32, #blocked>
    tt.store %optr_0, %scan#0, %mask : tensor<4096x!tt.ptr<f32>, #blocked>
    tt.store %flag_optr_0, %flag_i32, %mask : tensor<4096x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel segmented_scan
// CHECK: metal.threadgroup_alloca : !metal.memref<4096 x f32>
// CHECK: metal.threadgroup_alloca : !metal.memref<4096 x i1>
// CHECK-NOT: metal.threadgroup_alloca
// CHECK: metal.threadgroup_segmented_prefix_sum {{.*}} {block = 4096 : i64, tpb = 256 : i64}
// CHECK-NOT: tt.scan

// -----

// The other two monoids the scan lowering accepts, both from
// `leet-triton/medium-parallel_reverse_scan_gae.py`: a single-operand MUL scan
// (`tl.associative_scan(c, combine_fn=lambda l, r: l * r)`, the per-lane
// `c^(block_end - t)` carry coefficient) and the two-state affine scan run
// BACKWARDS (the GAE recurrence `A_t = delta_t + c * A_(t+1)`). Both were
// `failed to legalize operation 'tt.scan'` before: the single-operand path
// hardcoded add, and `isCanonicalAffineScan` rejected `reverse`.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @k_cumprod(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %off = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %i = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %i_0 = tt.addptr %i, %off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x = tt.load %i_0 : tensor<1024x!tt.ptr<f32>, #blocked>
    %cp = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %p = arith.mulf %a, %b : f32
      tt.scan.return %p : f32
    }) : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_0 = tt.addptr %o, %off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_0, %cp : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @k_cumprod_reverse(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %off = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %i = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %i_0 = tt.addptr %i, %off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x = tt.load %i_0 : tensor<1024x!tt.ptr<f32>, #blocked>
    %cp = "tt.scan"(%x) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%a: f32, %b: f32):
      %p = arith.mulf %a, %b : f32
      tt.scan.return %p : f32
    }) : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_0 = tt.addptr %o, %off : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_0, %cp : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // Sub-tpb MUL scan: the padded tail [16, 128) must select the MUL identity.
  // Padding a cumprod with the addf identity would zero every prefix instead.
  tt.func public @k_cumprod_sub_tpb(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    %off = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %m = tt.splat %n : i32 -> tensor<16xi32, #blocked>
    %m_0 = arith.cmpi slt, %off, %m : tensor<16xi32, #blocked>
    %x_1 = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %x_2 = tt.addptr %x_1, %off : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    %x_3 = tt.load %x_2, %m_0, %x : tensor<16x!tt.ptr<f32>, #blocked>
    %cp = "tt.scan"(%x_3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %p = arith.mulf %a, %b : f32
      tt.scan.return %p : f32
    }) : (tensor<16xf32, #blocked>) -> tensor<16xf32, #blocked>
    %o = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %o_0 = tt.addptr %o, %off : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
    tt.store %o_0, %cp, %m_0 : tensor<16x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @k_affine_reverse(%a_ptr: !tt.ptr<f32>, %x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %off = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %ap = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ap_0 = tt.addptr %ap, %off : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %av = tt.load %ap_0 : tensor<256x!tt.ptr<f32>, #blocked>
    %xp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %xp_0 = tt.addptr %xp, %off : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %xv = tt.load %xp_0 : tensor<256x!tt.ptr<f32>, #blocked>
    %sc:2 = "tt.scan"(%av, %xv) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%la: f32, %lx: f32, %ra: f32, %rx: f32):
      %ca = arith.mulf %la, %ra : f32
      %cm = arith.mulf %ra, %lx : f32
      %cx = arith.addf %cm, %rx : f32
      tt.scan.return %ca, %cx : f32, f32
    }) : (tensor<256xf32, #blocked>, tensor<256xf32, #blocked>) -> (tensor<256xf32, #blocked>, tensor<256xf32, #blocked>)
    %o = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %o_0 = tt.addptr %o, %off : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %o_0, %sc#1 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// An absent `combine` still means add, so the cumsum kernels above keep their
// attribute-free spelling; only the mul scans carry it.
// CHECK-LABEL: metal.kernel k_cumprod
// CHECK: metal.threadgroup_prefix_sum {{.*}} combine mulOp {block = 1024 : i64, tpb = 128 : i64}
// CHECK-NOT: tt.scan

// CHECK-LABEL: metal.kernel k_cumprod_reverse
// CHECK: metal.threadgroup_prefix_sum {{.*}} combine mulOp {block = 1024 : i64, reverse, tpb = 128 : i64}
// CHECK-NOT: tt.scan

// CHECK-LABEL: metal.kernel k_cumprod_sub_tpb
// CHECK: %[[UNIT:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: arith.select {{.*}}, %[[UNIT]] : f32
// CHECK: metal.threadgroup_prefix_sum {{.*}} combine mulOp {block = 128 : i64, tpb = 128 : i64}
// CHECK-NOT: tt.scan

// CHECK-LABEL: metal.kernel k_affine_reverse
// CHECK: metal.threadgroup_affine_prefix_scan {{.*}} {block = 256 : i64, reverse, tpb = 128 : i64}
// CHECK-NOT: tt.scan

// -----

// Staged XOR uses the same publication barrier as add/mul, a zero identity,
// and ui32 storage. Swapped combine operands are equivalent for this monoid.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "metal:0", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @staged_xor_reverse(%out: !tt.ptr<i32>, %x: i32) {
    %v = tt.splat %x : i32 -> tensor<4x8xi32, #blocked>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %r = arith.xori %rhs, %lhs : i32
      tt.scan.return %r : i32
    }) : (tensor<4x8xi32, #blocked>) -> tensor<4x8xi32, #blocked>
    %row = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #row>
    %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #col>
    %eight = arith.constant dense<8> : tensor<4xi32, #row>
    %rowOff = arith.muli %row, %eight : tensor<4xi32, #row>
    %r = tt.expand_dims %rowOff {axis = 1 : i32} : tensor<4xi32, #row> -> tensor<4x1xi32, #blocked>
    %c = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #col> -> tensor<1x8xi32, #blocked>
    %rr = tt.broadcast %r : tensor<4x1xi32, #blocked> -> tensor<4x8xi32, #blocked>
    %cc = tt.broadcast %c : tensor<1x8xi32, #blocked> -> tensor<4x8xi32, #blocked>
    %off = arith.addi %rr, %cc : tensor<4x8xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>, #blocked>
    %ptr = tt.addptr %base, %off : tensor<4x8x!tt.ptr<i32>, #blocked>, tensor<4x8xi32, #blocked>
    tt.store %ptr, %scan : tensor<4x8x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel staged_xor_reverse
// CHECK: %[[BUF:.*]] = metal.threadgroup_alloca : !metal.memref<32 x ui32>
// CHECK: metal.barrier
// CHECK: metal.store {{.*}}, %[[BUF]][
// CHECK: metal.barrier
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: metal.tg_load_indexed %[[BUF]][
// CHECK: arith.cmpi sge
// CHECK: arith.select {{.*}}, %[[ZERO]] : i32
// CHECK: bitXorOp
// CHECK-NOT: tt.scan

// -----

// A full one-element-per-thread scan consumes the scalarized FMA directly.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_fma_exact_thread_count(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %off = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %x = tt.load %ptr : tensor<128x!tt.ptr<f32>, #blocked>
    %src = math.fma %x, %x, %x : tensor<128xf32, #blocked>
    %scan = "tt.scan"(%src) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.scan.return %sum : f32
    }) : (tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    %out = tt.splat %output : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %dst = tt.addptr %out, %off : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %dst, %scan : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_fma_exact_thread_count
// CHECK-NOT: tt.scan
// CHECK: %[[FMA:.*]] = metal.fma
// CHECK-NOT: tt.scan
// CHECK: metal.store %[[FMA]],
// CHECK: metal.threadgroup_prefix_sum {{.*}} {block = 128 : i64, tpb = 128 : i64}
// CHECK-NOT: tt.scan

// -----

// Rank-2 staged scan above one element per thread: BLOCK=64 with tpb=32, so
// every thread publishes two logical tile positions before reading axis
// prefixes from the completed tile.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "metal:0", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @staged_rank2_scan_multiband(%out: !tt.ptr<f32>, %x: f32) {
    %v = tt.splat %x : f32 -> tensor<8x8xf32, #blocked>
    %scan = "tt.scan"(%v) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %r = arith.addf %lhs, %rhs : f32
      tt.scan.return %r : f32
    }) : (tensor<8x8xf32, #blocked>) -> tensor<8x8xf32, #blocked>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #row>
    %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #col>
    %eight = arith.constant dense<8> : tensor<8xi32, #row>
    %rowOff = arith.muli %row, %eight : tensor<8xi32, #row>
    %r = tt.expand_dims %rowOff {axis = 1 : i32} : tensor<8xi32, #row> -> tensor<8x1xi32, #blocked>
    %c = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #col> -> tensor<1x8xi32, #blocked>
    %rr = tt.broadcast %r : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %cc = tt.broadcast %c : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %off = arith.addi %rr, %cc : tensor<8x8xi32, #blocked>
    %ptr = tt.splat %out : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, #blocked>
    %outp = tt.addptr %ptr, %off : tensor<8x8x!tt.ptr<f32>, #blocked>, tensor<8x8xi32, #blocked>
    // An independent store after the scan must stay on the read side.
    tt.store %outp, %v : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.store %outp, %scan : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel staged_rank2_scan_multiband
// CHECK: %[[BUF:.*]] = metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK: scf.for
// CHECK-NOT: metal.store
// CHECK: metal.tg_store_indexed %[[BUF]][
// CHECK-NEXT: }
// CHECK-NEXT: metal.barrier
// CHECK: metal.tg_load_indexed %[[BUF]][
// CHECK: addOp
// CHECK: metal.store
// CHECK: metal.store
// CHECK-NOT: tt.scan

// -----

// Staged XOR uses the same publication barrier as add/mul, a zero identity,
// and ui32 storage. Swapped combine operands are equivalent for this monoid.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#col = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "metal:0", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @staged_xor_reverse_multiband(%out: !tt.ptr<i32>, %x: i32) {
    %v = tt.splat %x : i32 -> tensor<8x8xi32, #blocked>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %r = arith.xori %rhs, %lhs : i32
      tt.scan.return %r : i32
    }) : (tensor<8x8xi32, #blocked>) -> tensor<8x8xi32, #blocked>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #row>
    %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #col>
    %eight = arith.constant dense<8> : tensor<8xi32, #row>
    %rowOff = arith.muli %row, %eight : tensor<8xi32, #row>
    %r = tt.expand_dims %rowOff {axis = 1 : i32} : tensor<8xi32, #row> -> tensor<8x1xi32, #blocked>
    %c = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #col> -> tensor<1x8xi32, #blocked>
    %rr = tt.broadcast %r : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %cc = tt.broadcast %c : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %off = arith.addi %rr, %cc : tensor<8x8xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x8x!tt.ptr<i32>, #blocked>
    %ptr = tt.addptr %base, %off : tensor<8x8x!tt.ptr<i32>, #blocked>, tensor<8x8xi32, #blocked>
    tt.store %ptr, %scan : tensor<8x8x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel staged_xor_reverse_multiband
// CHECK: %[[XBUF:.*]] = metal.threadgroup_alloca : !metal.memref<64 x ui32>
// CHECK: scf.for
// CHECK: metal.tg_store_indexed %[[XBUF]][
// CHECK-NEXT: }
// CHECK-NEXT: metal.barrier
// CHECK: scf.for
// CHECK: metal.tg_load_indexed %[[XBUF]][
// CHECK: arith.cmpi sge
// CHECK: bitXorOp
// CHECK-NOT: tt.scan

// -----
// Each scan closes its register-band phase after its own consumers. A later
// single-band scan must not absorb the first scan's store into an E=1 loop.
#mixed2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mixed1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#mixed3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mixed_staged_scan_bands(%X: !tt.ptr<f32>, %O: !tt.ptr<f32>, %Q: !tt.ptr<f32>) {
    %c4 = arith.constant dense<4> : tensor<4x1xi32, #mixed2>
    %f = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #mixed1>
    %xp = tt.splat %X : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #mixed1>
    %xa = tt.addptr %xp, %f : tensor<128x!tt.ptr<f32>, #mixed1>, tensor<128xi32, #mixed1>
    %flat = tt.load %xa : tensor<128x!tt.ptr<f32>, #mixed1>
    %v = tt.reshape %flat : tensor<128xf32, #mixed1> -> tensor<4x4x8xf32, #mixed3>
    %op = tt.splat %O : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #mixed1>
    %oa = tt.addptr %op, %f : tensor<128x!tt.ptr<f32>, #mixed1>, tensor<128xi32, #mixed1>
    %s3 = "tt.scan"(%v) <{axis = 2 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<4x4x8xf32, #mixed3>) -> tensor<4x4x8xf32, #mixed3>
    %s3f = tt.reshape %s3 : tensor<4x4x8xf32, #mixed3> -> tensor<128xf32, #mixed1>
    tt.store %oa, %s3f : tensor<128x!tt.ptr<f32>, #mixed1>
    %i = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #mixed2}>>
    %ie = tt.expand_dims %i {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #mixed2}>> -> tensor<4x1xi32, #mixed2>
    %io = arith.muli %ie, %c4 : tensor<4x1xi32, #mixed2>
    %x2p = tt.splat %X : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>, #mixed2>
    %x2a = tt.addptr %x2p, %io : tensor<4x1x!tt.ptr<f32>, #mixed2>, tensor<4x1xi32, #mixed2>
    %j = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #mixed2}>>
    %je = tt.expand_dims %j {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #mixed2}>> -> tensor<1x4xi32, #mixed2>
    %x2ab = tt.broadcast %x2a : tensor<4x1x!tt.ptr<f32>, #mixed2> -> tensor<4x4x!tt.ptr<f32>, #mixed2>
    %jb = tt.broadcast %je : tensor<1x4xi32, #mixed2> -> tensor<4x4xi32, #mixed2>
    %x2addr = tt.addptr %x2ab, %jb : tensor<4x4x!tt.ptr<f32>, #mixed2>, tensor<4x4xi32, #mixed2>
    %w = tt.load %x2addr : tensor<4x4x!tt.ptr<f32>, #mixed2>
    %q2p = tt.splat %Q : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>, #mixed2>
    %q2a = tt.addptr %q2p, %io : tensor<4x1x!tt.ptr<f32>, #mixed2>, tensor<4x1xi32, #mixed2>
    %q2ab = tt.broadcast %q2a : tensor<4x1x!tt.ptr<f32>, #mixed2> -> tensor<4x4x!tt.ptr<f32>, #mixed2>
    %qaddr = tt.addptr %q2ab, %jb : tensor<4x4x!tt.ptr<f32>, #mixed2>, tensor<4x4xi32, #mixed2>
    %s2 = "tt.scan"(%w) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<4x4xf32, #mixed2>) -> tensor<4x4xf32, #mixed2>
    tt.store %qaddr, %s2 : tensor<4x4x!tt.ptr<f32>, #mixed2>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel mixed_staged_scan_bands
// CHECK: %[[BUF3:.*]] = metal.threadgroup_alloca : !metal.memref<128 x f32>
// CHECK: %[[BUF2:.*]] = metal.threadgroup_alloca : !metal.memref<16 x f32>
// CHECK: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C4]]
// CHECK: metal.tg_store_indexed %[[BUF3]]
// CHECK: metal.barrier
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C4]]
// CHECK: metal.tg_load_indexed %[[BUF3]]
// CHECK: metal.store
// CHECK: metal.barrier
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C1]]
// CHECK: metal.tg_store_indexed %[[BUF2]]
// CHECK: metal.barrier
// CHECK: %[[READ_C1:.*]] = arith.constant 1 : i32
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[READ_C1]]
// CHECK: metal.tg_load_indexed %[[BUF2]]
// CHECK: metal.store
// CHECK-NOT: tt.scan

// -----
// A reduction of the scan result reads a second publication emitted after
// the scan, rather than replaying the scan's source cone.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_reduce(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8x64xf32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8xf32, #scan_result_row>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<f32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xf32, #scan_result_row> -> tensor<8xf32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<f32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_reduce
// CHECK: %[[INPUT:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: %[[RESULT:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: metal.tg_store_indexed %[[INPUT]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT]]
// CHECK: metal.tg_store_indexed %[[RESULT]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// Number-preferring extrema use the same post-scan publication and preserve
// the reduce's NaN policy in the exact fold.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_maxnum(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8x64xf32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maxnumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8xf32, #scan_result_row>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<f32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xf32, #scan_result_row> -> tensor<8xf32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<f32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_maxnum
// CHECK: %[[INPUT_MAX:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: %[[RESULT_MAX:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: metal.tg_store_indexed %[[INPUT_MAX]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_MAX]]
// CHECK: metal.tg_store_indexed %[[RESULT_MAX]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_MAX]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// A product reduction likewise consumes the post-scan publication.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_product(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8x64xf32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %product = arith.mulf %a, %b : f32
      tt.reduce.return %product : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8xf32, #scan_result_row>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<f32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xf32, #scan_result_row> -> tensor<8xf32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<f32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_product
// CHECK: %[[INPUT_PRODUCT:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: %[[RESULT_PRODUCT:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: metal.tg_store_indexed %[[INPUT_PRODUCT]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_PRODUCT]]
// CHECK: metal.tg_store_indexed %[[RESULT_PRODUCT]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_PRODUCT]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, mulOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// Integer sums publish unchanged 32-bit values through the ui32 storage bridge.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_i32_sum(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<i32> -> tensor<8x1x!tt.ptr<i32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<i32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<i32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8x64xi32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.reduce.return %sum : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8xi32, #scan_result_row>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<i32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xi32, #scan_result_row> -> tensor<8xi32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<i32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_i32_sum
// CHECK: %[[INPUT_I32:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: %[[RESULT_I32:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: metal.tg_store_indexed %[[INPUT_I32]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_I32]]
// CHECK: metal.tg_store_indexed %[[RESULT_I32]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_I32]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// Integer products also wrap through the unchanged ui32 publication.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_i32_product(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<i32> -> tensor<8x1x!tt.ptr<i32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<i32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<i32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8x64xi32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %product = arith.muli %a, %b : i32
      tt.reduce.return %product : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8xi32, #scan_result_row>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<i32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xi32, #scan_result_row> -> tensor<8xi32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<i32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_i32_product
// CHECK: %[[INPUT_I32_PRODUCT:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: %[[RESULT_I32_PRODUCT:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: metal.tg_store_indexed %[[INPUT_I32_PRODUCT]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_I32_PRODUCT]]
// CHECK: metal.tg_store_indexed %[[RESULT_I32_PRODUCT]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_I32_PRODUCT]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, mulOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// Signed extrema read unchanged bits before using a signed comparison.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_i32_smax(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<i32> -> tensor<8x1x!tt.ptr<i32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<i32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<i32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8x64xi32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %max = arith.maxsi %a, %b : i32
      tt.reduce.return %max : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8xi32, #scan_result_row>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<i32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xi32, #scan_result_row> -> tensor<8xi32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<i32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_i32_smax
// CHECK: %[[INPUT_SMAX:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: %[[RESULT_SMAX:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: metal.tg_store_indexed %[[INPUT_SMAX]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_SMAX]]
// CHECK: metal.tg_store_indexed %[[RESULT_SMAX]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_SMAX]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// Unsigned extrema use the same publication without signed comparison.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_i32_umin(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<i32> -> tensor<8x1x!tt.ptr<i32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<i32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<i32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8x64xi32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %min = arith.minui %a, %b : i32
      tt.reduce.return %min : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8xi32, #scan_result_row>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<i32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xi32, #scan_result_row> -> tensor<8xi32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<i32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_i32_umin
// CHECK: %[[INPUT_UMIN:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: %[[RESULT_UMIN:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: metal.tg_store_indexed %[[INPUT_UMIN]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_UMIN]]
// CHECK: metal.tg_store_indexed %[[RESULT_UMIN]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_UMIN]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, minOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// Bitwise folds read the same unchanged ui32 scan publication.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_i32_xor(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<i32> -> tensor<8x1x!tt.ptr<i32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<i32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<i32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8x64xi32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %xor = arith.xori %a, %b : i32
      tt.reduce.return %xor : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8xi32, #scan_result_row>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<i32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xi32, #scan_result_row> -> tensor<8xi32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<i32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_i32_xor
// CHECK: %[[INPUT_XOR:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: %[[RESULT_XOR:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: metal.tg_store_indexed %[[INPUT_XOR]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_XOR]]
// CHECK: metal.tg_store_indexed %[[RESULT_XOR]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_XOR]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, bitXorOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// A keep-dims broadcast consumes the post-scan reduction at each tile row.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_sum_mixed(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %rowout: !tt.ptr<f32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8x64xf32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8xf32, #scan_result_row>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<8xf32, #scan_result_row> -> tensor<8x1xf32, #scan_result_tile>
    %b = tt.broadcast %e : tensor<8x1xf32, #scan_result_tile> -> tensor<8x64xf32, #scan_result_tile>
    %result = arith.addf %scan, %b : tensor<8x64xf32, #scan_result_tile>
    %obase = tt.splat %out : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %orow = tt.addptr %obase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %orowb = tt.broadcast %orow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %oaddr = tt.addptr %orowb, %cb : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    tt.store %oaddr, %result : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %rowout : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<f32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xf32, #scan_result_row> -> tensor<8xf32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<f32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_sum_mixed
// CHECK: %[[INPUT_BCAST:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: %[[RESULT_BCAST:.*]] = metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: metal.tg_store_indexed %[[INPUT_BCAST]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_BCAST]]
// CHECK: metal.tg_store_indexed %[[RESULT_BCAST]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_BCAST]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// Mixed direct-row and broadcast consumers also reuse the published i32 scan.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scan_result_row_i32_product_mixed(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>, %rowout: !tt.ptr<i32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<i32> -> tensor<8x1x!tt.ptr<i32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<i32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<i32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8x64xi32, #scan_result_tile>
    %r = "tt.reduce"(%scan) <{axis = 1 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %product = arith.muli %a, %b : i32
      tt.reduce.return %product : i32
    }) : (tensor<8x64xi32, #scan_result_tile>) -> tensor<8xi32, #scan_result_row>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %b = tt.broadcast %e : tensor<8x1xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %obase = tt.splat %out : !tt.ptr<i32> -> tensor<8x1x!tt.ptr<i32>, #scan_result_tile>
    %orow = tt.addptr %obase, %ro : tensor<8x1x!tt.ptr<i32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %orowb = tt.broadcast %orow : tensor<8x1x!tt.ptr<i32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %oaddr = tt.addptr %orowb, %cb : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    tt.store %oaddr, %b : tensor<8x64x!tt.ptr<i32>, #scan_result_tile>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %rowout : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<i32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xi32, #scan_result_row> -> tensor<8xi32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<i32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel scan_result_row_i32_product_mixed
// CHECK: %[[INPUT_PRODUCT_MIXED:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: %[[RESULT_PRODUCT_MIXED:.*]] = metal.threadgroup_alloca : !metal.memref<512 x ui32>
// CHECK: metal.tg_store_indexed %[[INPUT_PRODUCT_MIXED]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[INPUT_PRODUCT_MIXED]]
// CHECK: metal.tg_store_indexed %[[RESULT_PRODUCT_MIXED]]
// CHECK: metal.barrier
// CHECK: metal.tg_load_indexed %[[RESULT_PRODUCT_MIXED]]
// CHECK: metal.binary_exp {{.*}}, {{.*}}, mulOp
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// Distinct scan and reduction inputs share their own publications across both folds.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @distinct_input_row_sum_mixed(%in: !tt.ptr<f32>, %other: !tt.ptr<f32>, %out: !tt.ptr<f32>, %rowout: !tt.ptr<f32>) {
    %c64 = arith.constant dense<64> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c64 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<64xi32, #scan_result_col> -> tensor<1x64xi32, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x64xi32, #scan_result_tile> -> tensor<8x64xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %wbase = tt.splat %other : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %wrow = tt.addptr %wbase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %wrowb = tt.broadcast %wrow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %waddr = tt.addptr %wrowb, %cb : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    %w = tt.load %waddr : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8x64xf32, #scan_result_tile>
    %r = "tt.reduce"(%w) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8x64xf32, #scan_result_tile>) -> tensor<8xf32, #scan_result_row>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<8xf32, #scan_result_row> -> tensor<8x1xf32, #scan_result_tile>
    %b = tt.broadcast %e : tensor<8x1xf32, #scan_result_tile> -> tensor<8x64xf32, #scan_result_tile>
    %result = arith.addf %scan, %b : tensor<8x64xf32, #scan_result_tile>
    %obase = tt.splat %out : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %orow = tt.addptr %obase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %orowb = tt.broadcast %orow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %oaddr = tt.addptr %orowb, %cb : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>, tensor<8x64xi32, #scan_result_tile>
    tt.store %oaddr, %result : tensor<8x64x!tt.ptr<f32>, #scan_result_tile>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %rowout : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<f32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xf32, #scan_result_row> -> tensor<8xf32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<f32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel distinct_input_row_sum_mixed
// CHECK: metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: metal.threadgroup_alloca : !metal.memref<512 x f32>
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// A square original tile also accepts a cross-axis broadcast beside its row store.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cross_axis_row_sum_mixed(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %rowout: !tt.ptr<f32>) {
    %c8 = arith.constant dense<8> : tensor<8x1xi32, #scan_result_tile>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c8 : tensor<8x1xi32, #scan_result_tile>
    %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #scan_result_col> -> tensor<1x8xi32, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x8xi32, #scan_result_tile> -> tensor<8x8xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x8x!tt.ptr<f32>, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x8x!tt.ptr<f32>, #scan_result_tile>, tensor<8x8xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x8x!tt.ptr<f32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<8x8xf32, #scan_result_tile>) -> tensor<8x8xf32, #scan_result_tile>
    %r = "tt.reduce"(%v) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8x8xf32, #scan_result_tile>) -> tensor<8xf32, #scan_result_row>
    %rc = ttg.convert_layout %r : tensor<8xf32, #scan_result_row> -> tensor<8xf32, #scan_result_col>
    %e = tt.expand_dims %rc {axis = 0 : i32} : tensor<8xf32, #scan_result_col> -> tensor<1x8xf32, #scan_result_tile>
    %b = tt.broadcast %e : tensor<1x8xf32, #scan_result_tile> -> tensor<8x8xf32, #scan_result_tile>
    %result = arith.addf %scan, %b : tensor<8x8xf32, #scan_result_tile>
    %obase = tt.splat %out : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %orow = tt.addptr %obase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %orowb = tt.broadcast %orow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x8x!tt.ptr<f32>, #scan_result_tile>
    %oaddr = tt.addptr %orowb, %cb : tensor<8x8x!tt.ptr<f32>, #scan_result_tile>, tensor<8x8xi32, #scan_result_tile>
    tt.store %oaddr, %result : tensor<8x8x!tt.ptr<f32>, #scan_result_tile>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %rowout : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<f32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xf32, #scan_result_row> -> tensor<8xf32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<f32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel cross_axis_row_sum_mixed
// CHECK: metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce

// -----
// A wider same-axis broadcast can share the row reduction's publication.
#scan_result_tile = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_wide = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scan_result_row = #ttg.slice<{dim = 1, parent = #scan_result_tile}>
#scan_result_col = #ttg.slice<{dim = 0, parent = #scan_result_tile}>
#scan_result_wide_row = #ttg.slice<{dim = 1, parent = #scan_result_wide}>
#scan_result_wide_col = #ttg.slice<{dim = 0, parent = #scan_result_wide}>
#scan_result_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wider_same_axis_row_sum_mixed(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %wideout: !tt.ptr<f32>, %rowout: !tt.ptr<f32>) {
    %c8 = arith.constant dense<8> : tensor<8x1xi32, #scan_result_tile>
    %c16 = arith.constant dense<16> : tensor<8x1xi32, #scan_result_wide>
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_row>
    %re = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #scan_result_row> -> tensor<8x1xi32, #scan_result_tile>
    %ro = arith.muli %re, %c8 : tensor<8x1xi32, #scan_result_tile>
    %wrow = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_wide_row>
    %wre = tt.expand_dims %wrow {axis = 1 : i32} : tensor<8xi32, #scan_result_wide_row> -> tensor<8x1xi32, #scan_result_wide>
    %wro = arith.muli %wre, %c16 : tensor<8x1xi32, #scan_result_wide>
    %col = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_col>
    %ce = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #scan_result_col> -> tensor<1x8xi32, #scan_result_tile>
    %cb = tt.broadcast %ce : tensor<1x8xi32, #scan_result_tile> -> tensor<8x8xi32, #scan_result_tile>
    %xbase = tt.splat %in : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %xrow = tt.addptr %xbase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %xrowb = tt.broadcast %xrow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x8x!tt.ptr<f32>, #scan_result_tile>
    %xaddr = tt.addptr %xrowb, %cb : tensor<8x8x!tt.ptr<f32>, #scan_result_tile>, tensor<8x8xi32, #scan_result_tile>
    %v = tt.load %xaddr : tensor<8x8x!tt.ptr<f32>, #scan_result_tile>
    %scan = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<8x8xf32, #scan_result_tile>) -> tensor<8x8xf32, #scan_result_tile>
    %r = "tt.reduce"(%v) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8x8xf32, #scan_result_tile>) -> tensor<8xf32, #scan_result_row>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<8xf32, #scan_result_row> -> tensor<8x1xf32, #scan_result_tile>
    %b = tt.broadcast %e : tensor<8x1xf32, #scan_result_tile> -> tensor<8x16xf32, #scan_result_wide>
    %widecol = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #scan_result_wide_col>
    %wce = tt.expand_dims %widecol {axis = 0 : i32} : tensor<16xi32, #scan_result_wide_col> -> tensor<1x16xi32, #scan_result_wide>
    %wcb = tt.broadcast %wce : tensor<1x16xi32, #scan_result_wide> -> tensor<8x16xi32, #scan_result_wide>
    %wcf = arith.sitofp %wcb : tensor<8x16xi32, #scan_result_wide> to tensor<8x16xf32, #scan_result_wide>
    %wider = arith.addf %b, %wcf : tensor<8x16xf32, #scan_result_wide>
    %obase = tt.splat %out : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_tile>
    %orow = tt.addptr %obase, %ro : tensor<8x1x!tt.ptr<f32>, #scan_result_tile>, tensor<8x1xi32, #scan_result_tile>
    %orowb = tt.broadcast %orow : tensor<8x1x!tt.ptr<f32>, #scan_result_tile> -> tensor<8x8x!tt.ptr<f32>, #scan_result_tile>
    %oaddr = tt.addptr %orowb, %cb : tensor<8x8x!tt.ptr<f32>, #scan_result_tile>, tensor<8x8xi32, #scan_result_tile>
    tt.store %oaddr, %scan : tensor<8x8x!tt.ptr<f32>, #scan_result_tile>
    %wbase = tt.splat %wideout : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>, #scan_result_wide>
    %wwrow = tt.addptr %wbase, %wro : tensor<8x1x!tt.ptr<f32>, #scan_result_wide>, tensor<8x1xi32, #scan_result_wide>
    %wwrowb = tt.broadcast %wwrow : tensor<8x1x!tt.ptr<f32>, #scan_result_wide> -> tensor<8x16x!tt.ptr<f32>, #scan_result_wide>
    %waddr = tt.addptr %wwrowb, %wcb : tensor<8x16x!tt.ptr<f32>, #scan_result_wide>, tensor<8x16xi32, #scan_result_wide>
    tt.store %waddr, %wider : tensor<8x16x!tt.ptr<f32>, #scan_result_wide>
    %out_index = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #scan_result_flat>
    %base = tt.splat %rowout : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #scan_result_flat>
    %ptr = tt.addptr %base, %out_index : tensor<8x!tt.ptr<f32>, #scan_result_flat>, tensor<8xi32, #scan_result_flat>
    %flat = ttg.convert_layout %r : tensor<8xf32, #scan_result_row> -> tensor<8xf32, #scan_result_flat>
    tt.store %ptr, %flat : tensor<8x!tt.ptr<f32>, #scan_result_flat>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel wider_same_axis_row_sum_mixed
// CHECK: metal.threadgroup_alloca : !metal.memref<64 x f32>
// CHECK: metal.threadgroup_alloca : !metal.memref<8 x f32>
// CHECK-NOT: tt.scan
// CHECK-NOT: tt.reduce
