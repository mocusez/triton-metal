// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Wall 9 lit fixture: rank-1 arith.subf lowering via ArithSubFLowering.
// Mirrors vector_add_spt_unchanged.mlir but uses arith.subf in place of
// arith.addf. Before Wall 9 ships, this fixture FAILS with
// "failed to legalize operation 'arith.subf'". After Wall 9 ships, the
// op lowers to `metal.binary_exp ..., ..., subOp`.
// See the implementation notes AC1-AC2.

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sub_kernel_spt8_unmasked(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_val = tt.load %y_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %diff = arith.subf %x_val, %y_val : tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %diff : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @map_elementwise_sub_pack1(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_val = tt.load %y_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %diff = "tt.map_elementwise"(%x_val, %y_val) <{pack = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %value = arith.subf %x, %y : f32
      tt.map_elementwise.return %value : f32
    }) : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %diff : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @map_elementwise_cfg_pack1(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_val = tt.load %y_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %diff = "tt.map_elementwise"(%x_val, %y_val) <{pack = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %take_x = arith.cmpf ogt, %x, %y : f32
      cf.cond_br %take_x, ^bb1(%x : f32), ^bb1(%y : f32)
    ^bb1(%selected: f32):
      tt.map_elementwise.return %selected : f32
    }) : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %diff : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @map_elementwise_pack2(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %x_val = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %y_addr = tt.addptr %y_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %y_val = tt.load %y_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %result = "tt.map_elementwise"(%x_val, %y_val) <{pack = 2 : i32}> ({
    ^bb0(%x0: f32, %x1: f32, %y0: f32, %y1: f32):
      %first = arith.mulf %x0, %y1 : f32
      %out0 = arith.addf %first, %x1 : f32
      %out1 = arith.subf %y0, %x0 : f32
      tt.map_elementwise.return %out0, %out1 : f32, f32
    }) : (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %result : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel sub_kernel_spt8_unmasked
// One get_element per operand, then a single subOp per lane.
// CHECK: metal.get_element %arg0
// CHECK: metal.get_element %arg1
// CHECK: metal.binary_exp {{.*}}, {{.*}}, subOp
// CHECK: metal.store
// CHECK: metal.return

// CHECK-LABEL: metal.kernel map_elementwise_sub_pack1
// The region is gone and its scalar arithmetic remains inside the normal
// per-element path.
// CHECK-NOT: tt.map_elementwise
// CHECK: %[[DIFF:.*]] = arith.subf
// CHECK: metal.store %[[DIFF]]
// CHECK: metal.return

// CHECK-LABEL: metal.kernel map_elementwise_cfg_pack1
// CHECK-NOT: tt.map_elementwise
// CHECK: scf.if
// CHECK-NOT: cf.cond_br
// CHECK: metal.store
// CHECK: metal.return

// CHECK-LABEL: metal.kernel map_elementwise_pack2
// CHECK-NOT: tt.map_elementwise
// CHECK: scf.for
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: metal.barrier
// CHECK: scf.for
// CHECK: metal.tg_load_indexed
// CHECK: metal.store
// CHECK: metal.return
