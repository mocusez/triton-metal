// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s --implicit-check-not=metal.barrier --implicit-check-not=ttg.convert_layout
//
// L1d3 positive fixture: a self-contained rank-2 blocked↔blocked transpose cvt
// with sizePerThread > 1 (the canonical masked-transpose / matmul-pre-transpose
// shape, E>1) is normalized away by `normalizeBlockedDivergentCvts` rather than
// rejected. The pre-pass rewrites the load's producer cone from the row-major
// #blocked2a (order [1,0]) to the column-major #blocked2b (order [0,1]) store
// layout, so the cvt collapses to an identity and the kernel lowers to a direct
// gather (metal.get_element) + scatter (metal.store) inside the E>1 tile loop —
// NO layout-conversion staging (the --implicit-check-not guards above assert
// that no barrier or cvt remains). The shared-source fixture below has a
// masked store, so its independent masked-store sentinel legitimately uses
// threadgroup scratch without a barrier. Mirrors the runtime case
// `test_metal_backend_masked_store_sweep.py::...[32x32_nw4]`.

#blocked2a = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2b = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_rank2_transpose_normalized(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>) {
    %offs = arith.constant dense<0> : tensor<16x16xi32, #blocked2a>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %x_addr = tt.addptr %x_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked2a>, tensor<16x16xi32, #blocked2a>
    %x_val = tt.load %x_addr : tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %x_cvt = ttg.convert_layout %x_val : tensor<16x16xf32, #blocked2a> -> tensor<16x16xf32, #blocked2b>
    %offs_b = arith.constant dense<0> : tensor<16x16xi32, #blocked2b>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2b>
    %y_addr = tt.addptr %y_splat, %offs_b : tensor<16x16x!tt.ptr<f32>, #blocked2b>, tensor<16x16xi32, #blocked2b>
    tt.store %y_addr, %x_cvt : tensor<16x16x!tt.ptr<f32>, #blocked2b>
    tt.return
  }

  tt.func public @cvt_rank2_shared_source_store_side_normalized(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %z_ptr: !tt.ptr<f32>) {
    %offs = arith.constant dense<0> : tensor<16x16xi32, #blocked2a>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %x_addr = tt.addptr %x_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked2a>, tensor<16x16xi32, #blocked2a>
    %x_val = tt.load %x_addr : tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %z_splat = tt.splat %z_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %z_addr = tt.addptr %z_splat, %offs : tensor<16x16x!tt.ptr<f32>, #blocked2a>, tensor<16x16xi32, #blocked2a>
    tt.store %z_addr, %x_val : tensor<16x16x!tt.ptr<f32>, #blocked2a>
    %x_cvt = ttg.convert_layout %x_val : tensor<16x16xf32, #blocked2a> -> tensor<16x16xf32, #blocked2b>
    %offs_b = arith.constant dense<0> : tensor<16x16xi32, #blocked2b>
    %bound_b = arith.constant dense<128> : tensor<16x16xi32, #blocked2b>
    %mask_b = arith.cmpi slt, %offs_b, %bound_b : tensor<16x16xi32, #blocked2b>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked2b>
    %y_addr = tt.addptr %y_splat, %offs_b : tensor<16x16x!tt.ptr<f32>, #blocked2b>, tensor<16x16xi32, #blocked2b>
    tt.store %y_addr, %x_cvt, %mask_b : tensor<16x16x!tt.ptr<f32>, #blocked2b>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel cvt_rank2_transpose_normalized
// CHECK-NOT: metal.threadgroup_alloca
// CHECK-NOT: metal.tg_store_indexed
// CHECK-NOT: metal.tg_load_indexed
// CHECK: scf.for
// CHECK: metal.get_element
// CHECK: metal.store
// CHECK-NOT: metal.threadgroup_alloca
// CHECK-NOT: metal.tg_store_indexed
// CHECK-NOT: metal.tg_load_indexed
// CHECK-LABEL: metal.kernel cvt_rank2_shared_source_store_side_normalized
// CHECK: metal.get_element
// CHECK: metal.store
// CHECK: metal.store
