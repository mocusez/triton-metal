// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
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
// CHECK-NOT: tt.scan
