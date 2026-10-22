// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | triton-metal-translate --mlir-to-msl | FileCheck %s

// `tl.topk(x, 4)` over a 4x32 tile at one warp. The range layouts omit
// redundant zero register bases, as blocked and slice layouts do when converted
// to linear layouts, so the reshapes preserve the exact register mapping.
//
// When a staged phase rematerialised an exchange read it copied the read's plan
// with `g_tileExchange[clone] = planIt->second`. Both are DenseMaps: the insert
// may grow the table and, under C++17, the right-hand side is a reference into
// that same table evaluated FIRST -- so the copy came from a moved-from plan
// whose index terms were empty, and the clone read scratch slot 0 in every lane.
// The kernel returned values that were not in its input.
//
// Whether the insert grows depends on how many entries the map already holds,
// which inside one Python process depends on every kernel compiled before --
// a pytest cannot pin it. A fresh `triton-metal-opt` on this exact IR does.
// No scratch read in this kernel has a constant index.

// CHECK: kernel void k(
// CHECK-NOT: = v{{[0-9]+}}[0];
// CHECK: return;

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 2, 2], threadsPerWarp = [2, 2, 2, 2, 2, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1, 1, 1], order = [6, 5, 4, 3, 2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1]], lane = [[0], [0], [0], [0], [0]], warp = [], block = []}>
#linear1 = #ttg.linear<{register = [], lane = [[1], [0], [0], [0], [0]], warp = [], block = []}>
#linear2 = #ttg.linear<{register = [], lane = [[0], [0], [1], [0], [0]], warp = [], block = []}>
#linear3 = #ttg.linear<{register = [], lane = [[0], [1], [0], [0], [0]], warp = [], block = []}>
#linear4 = #ttg.linear<{register = [[1]], lane = [[0], [0], [0], [0], [0]], warp = [], block = []}>
#linear5 = #ttg.linear<{register = [[1]], lane = [[0], [0], [0], [0], [0]], warp = [], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[0, 0], [0, 0], [0, 0], [1, 0], [2, 0]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @k(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<1x1x1x1x1x2x2xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<1x1x1x1x2x1x2xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<1x1x1x1x2x2x1xi32, #blocked>
    %cst_2 = arith.constant dense<0> : tensor<1x1x1x2x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %cst_3 = arith.constant dense<0> : tensor<1x1x1x2x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %cst_4 = arith.constant dense<0> : tensor<1x1x2x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %cst_5 = arith.constant dense<0> : tensor<1x1x2x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %cst_6 = arith.constant dense<1> : tensor<1x1x1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %cst_7 = arith.constant dense<1> : tensor<1x1x2x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %cst_8 = arith.constant dense<0> : tensor<1x1x1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %cst_9 = arith.constant dense<0> : tensor<1x1x2x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %cst_10 = arith.constant dense<4> : tensor<4x1xi32, #blocked1>
    %cst_11 = arith.constant dense<32> : tensor<4x1xi32, #blocked2>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<4x1xi32, #blocked2>
    %3 = tt.expand_dims %1 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<4x1xi32, #blocked1>
    %4 = arith.muli %2, %cst_11 : tensor<4x1xi32, #blocked2>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>, #blocked2>
    %6 = tt.addptr %5, %4 : tensor<4x1x!tt.ptr<f32>, #blocked2>, tensor<4x1xi32, #blocked2>
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
    %9 = tt.broadcast %6 : tensor<4x1x!tt.ptr<f32>, #blocked2> -> tensor<4x32x!tt.ptr<f32>, #blocked2>
    %10 = tt.broadcast %8 : tensor<1x32xi32, #blocked2> -> tensor<4x32xi32, #blocked2>
    %11 = tt.addptr %9, %10 : tensor<4x32x!tt.ptr<f32>, #blocked2>, tensor<4x32xi32, #blocked2>
    %12 = tt.load %11 : tensor<4x32x!tt.ptr<f32>, #blocked2>
    %13 = arith.muli %3, %cst_10 : tensor<4x1xi32, #blocked1>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>, #blocked1>
    %15 = tt.addptr %14, %13 : tensor<4x1x!tt.ptr<f32>, #blocked1>, tensor<4x1xi32, #blocked1>
    %16 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %17 = tt.expand_dims %16 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x4xi32, #blocked1>
    %18 = tt.broadcast %15 : tensor<4x1x!tt.ptr<f32>, #blocked1> -> tensor<4x4x!tt.ptr<f32>, #blocked1>
    %19 = tt.broadcast %17 : tensor<1x4xi32, #blocked1> -> tensor<4x4xi32, #blocked1>
    %20 = tt.addptr %18, %19 : tensor<4x4x!tt.ptr<f32>, #blocked1>, tensor<4x4xi32, #blocked1>
    %21 = tt.reshape %12 : tensor<4x32xf32, #blocked2> -> tensor<2x2x2x2x2x2x2xf32, #blocked>
    %22 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear>
    %23 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear1>
    %24 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear2>
    %25 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear3>
    %26 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear4>
    %27 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #linear5>
    %28 = tt.reshape %22 : tensor<2xi32, #linear> -> tensor<1x1x1x1x1x2x1xi32, #blocked>
    %29 = tt.bitcast %21 : tensor<2x2x2x2x2x2x2xf32, #blocked> -> tensor<2x2x2x2x2x2x2xi32, #blocked>
    %30 = "tt.reduce"(%29) <{axis = 6 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #blocked>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #blocked}>>
    %31 = tt.expand_dims %30 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #blocked}>> -> tensor<2x2x2x2x2x2x1xi32, #blocked>
    %32 = tt.broadcast %31 : tensor<2x2x2x2x2x2x1xi32, #blocked> -> tensor<2x2x2x2x2x2x2xi32, #blocked>
    %33 = arith.xori %29, %32 : tensor<2x2x2x2x2x2x2xi32, #blocked>
    %34 = tt.bitcast %33 : tensor<2x2x2x2x2x2x2xi32, #blocked> -> tensor<2x2x2x2x2x2x2xf32, #blocked>
    %35 = tt.reshape %26 : tensor<2xi32, #linear4> -> tensor<1x1x1x1x1x1x2xi32, #blocked>
    %36 = arith.cmpf ogt, %21, %34 : tensor<2x2x2x2x2x2x2xf32, #blocked>
    %37 = tt.broadcast %28 : tensor<1x1x1x1x1x2x1xi32, #blocked> -> tensor<1x1x1x1x1x2x2xi32, #blocked>
    %38 = tt.broadcast %35 : tensor<1x1x1x1x1x1x2xi32, #blocked> -> tensor<1x1x1x1x1x2x2xi32, #blocked>
    %39 = arith.xori %37, %38 : tensor<1x1x1x1x1x2x2xi32, #blocked>
    %40 = arith.cmpi ne, %39, %cst : tensor<1x1x1x1x1x2x2xi32, #blocked>
    %41 = tt.broadcast %40 : tensor<1x1x1x1x1x2x2xi1, #blocked> -> tensor<2x2x2x2x2x2x2xi1, #blocked>
    %42 = arith.cmpi ne, %36, %41 : tensor<2x2x2x2x2x2x2xi1, #blocked>
    %43 = arith.select %42, %34, %21 : tensor<2x2x2x2x2x2x2xi1, #blocked>, tensor<2x2x2x2x2x2x2xf32, #blocked>
    %44 = tt.reshape %23 : tensor<2xi32, #linear1> -> tensor<1x1x1x1x2x1x1xi32, #blocked>
    %45 = tt.bitcast %43 : tensor<2x2x2x2x2x2x2xf32, #blocked> -> tensor<2x2x2x2x2x2x2xi32, #blocked>
    %46 = "tt.reduce"(%45) <{axis = 5 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #blocked>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #blocked}>>
    %47 = tt.expand_dims %46 {axis = 5 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #blocked}>> -> tensor<2x2x2x2x2x1x2xi32, #blocked>
    %48 = tt.broadcast %47 : tensor<2x2x2x2x2x1x2xi32, #blocked> -> tensor<2x2x2x2x2x2x2xi32, #blocked>
    %49 = arith.xori %45, %48 : tensor<2x2x2x2x2x2x2xi32, #blocked>
    %50 = tt.bitcast %49 : tensor<2x2x2x2x2x2x2xi32, #blocked> -> tensor<2x2x2x2x2x2x2xf32, #blocked>
    %51 = arith.cmpf ogt, %43, %50 : tensor<2x2x2x2x2x2x2xf32, #blocked>
    %52 = tt.broadcast %44 : tensor<1x1x1x1x2x1x1xi32, #blocked> -> tensor<1x1x1x1x2x2x1xi32, #blocked>
    %53 = tt.broadcast %28 : tensor<1x1x1x1x1x2x1xi32, #blocked> -> tensor<1x1x1x1x2x2x1xi32, #blocked>
    %54 = arith.xori %52, %53 : tensor<1x1x1x1x2x2x1xi32, #blocked>
    %55 = arith.cmpi ne, %54, %cst_1 : tensor<1x1x1x1x2x2x1xi32, #blocked>
    %56 = tt.broadcast %55 : tensor<1x1x1x1x2x2x1xi1, #blocked> -> tensor<2x2x2x2x2x2x2xi1, #blocked>
    %57 = arith.cmpi ne, %51, %56 : tensor<2x2x2x2x2x2x2xi1, #blocked>
    %58 = arith.select %57, %50, %43 : tensor<2x2x2x2x2x2x2xi1, #blocked>, tensor<2x2x2x2x2x2x2xf32, #blocked>
    %59 = tt.bitcast %58 : tensor<2x2x2x2x2x2x2xf32, #blocked> -> tensor<2x2x2x2x2x2x2xi32, #blocked>
    %60 = "tt.reduce"(%59) <{axis = 6 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2x2x2x2xi32, #blocked>) -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #blocked}>>
    %61 = tt.expand_dims %60 {axis = 6 : i32} : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 6, parent = #blocked}>> -> tensor<2x2x2x2x2x2x1xi32, #blocked>
    %62 = tt.broadcast %61 : tensor<2x2x2x2x2x2x1xi32, #blocked> -> tensor<2x2x2x2x2x2x2xi32, #blocked>
    %63 = arith.xori %59, %62 : tensor<2x2x2x2x2x2x2xi32, #blocked>
    %64 = tt.bitcast %63 : tensor<2x2x2x2x2x2x2xi32, #blocked> -> tensor<2x2x2x2x2x2x2xf32, #blocked>
    %65 = arith.cmpf ogt, %58, %64 : tensor<2x2x2x2x2x2x2xf32, #blocked>
    %66 = tt.broadcast %44 : tensor<1x1x1x1x2x1x1xi32, #blocked> -> tensor<1x1x1x1x2x1x2xi32, #blocked>
    %67 = tt.broadcast %35 : tensor<1x1x1x1x1x1x2xi32, #blocked> -> tensor<1x1x1x1x2x1x2xi32, #blocked>
    %68 = arith.xori %66, %67 : tensor<1x1x1x1x2x1x2xi32, #blocked>
    %69 = arith.cmpi ne, %68, %cst_0 : tensor<1x1x1x1x2x1x2xi32, #blocked>
    %70 = tt.broadcast %69 : tensor<1x1x1x1x2x1x2xi1, #blocked> -> tensor<2x2x2x2x2x2x2xi1, #blocked>
    %71 = arith.cmpi ne, %65, %70 : tensor<2x2x2x2x2x2x2xi1, #blocked>
    %72 = arith.select %71, %64, %58 : tensor<2x2x2x2x2x2x2xi1, #blocked>, tensor<2x2x2x2x2x2x2xf32, #blocked>
    %73 = "tt.reduce"(%72) <{axis = 4 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %166 = arith.maxnumf %arg2, %arg3 : f32
      tt.reduce.return %166 : f32
    }) : (tensor<2x2x2x2x2x2x2xf32, #blocked>) -> tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %74 = tt.reshape %25 : tensor<2xi32, #linear3> -> tensor<1x1x1x2x1x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %75 = tt.bitcast %73 : tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %76 = "tt.reduce"(%75) <{axis = 4 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>) -> tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %77 = tt.expand_dims %76 {axis = 4 : i32} : tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %78 = tt.broadcast %77 : tensor<2x2x2x2x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %79 = arith.xori %75, %78 : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %80 = tt.bitcast %79 : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %81 = tt.reshape %27 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %82 = arith.cmpf ogt, %73, %80 : tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %83 = tt.broadcast %74 : tensor<1x1x1x2x1x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<1x1x1x2x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %84 = tt.broadcast %81 : tensor<1x1x1x1x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<1x1x1x2x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %85 = arith.xori %83, %84 : tensor<1x1x1x2x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %86 = arith.cmpi ne, %85, %cst_3 : tensor<1x1x1x2x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %87 = tt.broadcast %86 : tensor<1x1x1x2x2x1xi1, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<2x2x2x2x2x2xi1, #ttg.slice<{dim = 4, parent = #blocked}>>
    %88 = arith.cmpi ne, %82, %87 : tensor<2x2x2x2x2x2xi1, #ttg.slice<{dim = 4, parent = #blocked}>>
    %89 = arith.select %88, %80, %73 : tensor<2x2x2x2x2x2xi1, #ttg.slice<{dim = 4, parent = #blocked}>>, tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %90 = tt.bitcast %89 : tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %91 = "tt.reduce"(%90) <{axis = 5 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>) -> tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %92 = tt.expand_dims %91 {axis = 5 : i32} : tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 5, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %93 = tt.broadcast %92 : tensor<2x2x2x2x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %94 = arith.xori %90, %93 : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %95 = tt.bitcast %94 : tensor<2x2x2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %96 = tt.reshape %27 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %97 = arith.cmpf ogt, %89, %95 : tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %98 = tt.broadcast %74 : tensor<1x1x1x2x1x1xi32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<1x1x1x2x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %99 = tt.broadcast %96 : tensor<1x1x1x1x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<1x1x1x2x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %100 = arith.xori %98, %99 : tensor<1x1x1x2x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %101 = arith.cmpi ne, %100, %cst_2 : tensor<1x1x1x2x1x2xi32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %102 = tt.broadcast %101 : tensor<1x1x1x2x1x2xi1, #ttg.slice<{dim = 4, parent = #blocked}>> -> tensor<2x2x2x2x2x2xi1, #ttg.slice<{dim = 4, parent = #blocked}>>
    %103 = arith.cmpi ne, %97, %102 : tensor<2x2x2x2x2x2xi1, #ttg.slice<{dim = 4, parent = #blocked}>>
    %104 = arith.select %103, %95, %89 : tensor<2x2x2x2x2x2xi1, #ttg.slice<{dim = 4, parent = #blocked}>>, tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>>
    %105 = "tt.reduce"(%104) <{axis = 3 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %166 = arith.maxnumf %arg2, %arg3 : f32
      tt.reduce.return %166 : f32
    }) : (tensor<2x2x2x2x2x2xf32, #ttg.slice<{dim = 4, parent = #blocked}>>) -> tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %106 = tt.reshape %24 : tensor<2xi32, #linear2> -> tensor<1x1x2x1x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %107 = tt.bitcast %105 : tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %108 = "tt.reduce"(%107) <{axis = 3 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>) -> tensor<2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %109 = tt.expand_dims %108 {axis = 3 : i32} : tensor<2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %110 = tt.broadcast %109 : tensor<2x2x2x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %111 = arith.xori %107, %110 : tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %112 = tt.bitcast %111 : tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %113 = tt.reshape %27 : tensor<2xi32, #linear5> -> tensor<1x1x1x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %114 = arith.cmpf ogt, %105, %112 : tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %115 = tt.broadcast %106 : tensor<1x1x2x1x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<1x1x2x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %116 = tt.broadcast %113 : tensor<1x1x1x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<1x1x2x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %117 = arith.xori %115, %116 : tensor<1x1x2x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %118 = arith.cmpi ne, %117, %cst_5 : tensor<1x1x2x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %119 = tt.broadcast %118 : tensor<1x1x2x2x1xi1, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2xi1, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %120 = arith.cmpi ne, %114, %119 : tensor<2x2x2x2x2xi1, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %121 = arith.select %120, %112, %105 : tensor<2x2x2x2x2xi1, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>, tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %122 = tt.bitcast %121 : tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %123 = "tt.reduce"(%122) <{axis = 4 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>) -> tensor<2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %124 = tt.expand_dims %123 {axis = 4 : i32} : tensor<2x2x2x2xi32, #ttg.slice<{dim = 4, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %125 = tt.broadcast %124 : tensor<2x2x2x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %126 = arith.xori %122, %125 : tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %127 = tt.bitcast %126 : tensor<2x2x2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %128 = tt.reshape %27 : tensor<2xi32, #linear5> -> tensor<1x1x1x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %129 = arith.cmpf ogt, %121, %127 : tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %130 = tt.broadcast %106 : tensor<1x1x2x1x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<1x1x2x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %131 = tt.broadcast %128 : tensor<1x1x1x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<1x1x2x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %132 = arith.xori %130, %131 : tensor<1x1x2x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %133 = arith.cmpi ne, %132, %cst_4 : tensor<1x1x2x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %134 = tt.broadcast %133 : tensor<1x1x2x1x2xi1, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<2x2x2x2x2xi1, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %135 = arith.cmpi ne, %129, %134 : tensor<2x2x2x2x2xi1, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %136 = arith.select %135, %127, %121 : tensor<2x2x2x2x2xi1, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>, tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>
    %137 = "tt.reduce"(%136) <{axis = 2 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %166 = arith.maxnumf %arg2, %arg3 : f32
      tt.reduce.return %166 : f32
    }) : (tensor<2x2x2x2x2xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>) -> tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %138 = tt.bitcast %137 : tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %139 = "tt.reduce"(%138) <{axis = 2 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>) -> tensor<2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>}>>
    %140 = tt.expand_dims %139 {axis = 2 : i32} : tensor<2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>}>> -> tensor<2x2x1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %141 = tt.broadcast %140 : tensor<2x2x1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %142 = arith.xori %138, %141 : tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %143 = tt.bitcast %142 : tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %144 = tt.reshape %27 : tensor<2xi32, #linear5> -> tensor<1x1x2x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %145 = arith.cmpf ogt, %137, %143 : tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %146 = arith.xori %144, %cst_7 : tensor<1x1x2x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %147 = arith.cmpi ne, %146, %cst_9 : tensor<1x1x2x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %148 = tt.broadcast %147 : tensor<1x1x2x1xi1, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2xi1, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %149 = arith.cmpi ne, %145, %148 : tensor<2x2x2x2xi1, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %150 = arith.select %149, %143, %137 : tensor<2x2x2x2xi1, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>, tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %151 = tt.bitcast %150 : tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %152 = "tt.reduce"(%151) <{axis = 3 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %166 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %166 : i32
    }) : (tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>) -> tensor<2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>}>>
    %153 = tt.expand_dims %152 {axis = 3 : i32} : tensor<2x2x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>}>> -> tensor<2x2x2x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %154 = tt.broadcast %153 : tensor<2x2x2x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %155 = arith.xori %151, %154 : tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %156 = tt.bitcast %155 : tensor<2x2x2x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %157 = tt.reshape %27 : tensor<2xi32, #linear5> -> tensor<1x1x1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %158 = arith.cmpf ogt, %150, %156 : tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %159 = arith.xori %157, %cst_6 : tensor<1x1x1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %160 = arith.cmpi ne, %159, %cst_8 : tensor<1x1x1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %161 = tt.broadcast %160 : tensor<1x1x1x2xi1, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<2x2x2x2xi1, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %162 = arith.cmpi ne, %158, %161 : tensor<2x2x2x2xi1, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %163 = arith.select %162, %156, %150 : tensor<2x2x2x2xi1, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>, tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>>
    %164 = tt.reshape %163 : tensor<2x2x2x2xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>}>> -> tensor<4x4xf32, #linear6>
    %165 = ttg.convert_layout %164 : tensor<4x4xf32, #linear6> -> tensor<4x4xf32, #blocked1>
    tt.store %20, %165 : tensor<4x4x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

