// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Unary math inside a reduced cone.
//
// The cone replayers re-derive a reduce's source one element at a time, and
// each is gated by a predicate listing the ops it can replay. exp2/log2/absf/
// absi/floor/ceil and arith.negf were in neither list even though the
// elementwise path already emitted each of them, so a reduce over such a cone
// matched NO tier: it failed to legalize after FuncOpLowering had rewritten
// the function signature, i.e. a process kill rather than a diagnostic.
// `coneUnaryExpOperator` / `coneClonableUnaryOp` / `emitMetalAbsF` are now the
// single tables both the predicates and the evaluators consult.
//
// BLOCK=1024, num_warps=8 → tpb=256, four elements per thread.
//
// MLIR only: math.floor / math.ceil / math.absi stay LEGAL through the
// conversion, so the original tensor-typed op survives as dead IR next to the
// scalar replay. That is harmless in the compiler's own pipeline (the module
// never leaves memory, and the emitter walks live values), but it keeps a
// `#ttg.blocked` alias in the printed module, which `triton-metal-translate`
// cannot re-parse. The MSL spellings are covered on device by
// `test_metal_backend_reduce_rank1.py::test_reduce_rank1_unary_math_cone`.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked_i = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // metal.unary_exp shapes: exp2 and log2 have their own operators, kept
  // distinct from exp/log because the rounding differs.
  tt.func public @reduce_sum_exp2(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %e = math.exp2 %row : tensor<1024xf32, #blocked>
    %sum = "tt.reduce"(%e) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %sum_t = tt.splat %sum : f32 -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %sum_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // math.floor stays LEGAL through conversion — the translator spells it —
  // so the cone replays it by cloning the op onto the evaluated element.
  tt.func public @reduce_sum_floor(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %f = math.floor %row : tensor<1024xf32, #blocked>
    %sum = "tt.reduce"(%f) ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %sum_t = tt.splat %sum : f32 -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %sum_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // f32 abs is metal.unary_exp absOp; a negation is arith.negf, kept as such.
  tt.func public @reduce_max_absf_neg(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f32>, #blocked>
    %n = arith.negf %row : tensor<1024xf32, #blocked>
    %a = math.absf %n : tensor<1024xf32, #blocked>
    %m = "tt.reduce"(%a) ({
    ^bb0(%p: f32, %q: f32):
      %s = arith.maxnumf %p, %q : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<1024xf32, #blocked>) -> f32
    %m_t = tt.splat %m : f32 -> tensor<1024xf32, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %m_t : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // Integer abs is math.absi, which the translator re-reads as SIGNED: a
  // signless device buffer is uint32_t, on which abs is the identity.
  tt.func public @reduce_sum_absi(%x_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked_i>
    %x_splat = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked_i>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<i32>, #blocked_i>, tensor<1024xi32, #blocked_i>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<i32>, #blocked_i>
    %a = math.absi %row : tensor<1024xi32, #blocked_i>
    %sum = "tt.reduce"(%a) ({
    ^bb0(%p: i32, %q: i32):
      %s = arith.addi %p, %q : i32
      tt.reduce.return %s : i32
    }) {axis = 0 : i32} : (tensor<1024xi32, #blocked_i>) -> i32
    %sum_t = tt.splat %sum : i32 -> tensor<1024xi32, #blocked_i>
    %o_splat = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked_i>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<i32>, #blocked_i>, tensor<1024xi32, #blocked_i>
    tt.store %o_addr, %sum_t : tensor<1024x!tt.ptr<i32>, #blocked_i>
    tt.return
  }

  // f16 abs must stay the bit-preserving sign-mask, NOT a float conversion:
  // the cone and the elementwise pattern share one emitter so a NaN payload
  // and a signed zero cannot diverge between them.
  tt.func public @reduce_max_absf_f16(%x_ptr: !tt.ptr<f16>, %out_ptr: !tt.ptr<f16>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %row = tt.load %x_addr : tensor<1024x!tt.ptr<f16>, #blocked>
    %a = math.absf %row : tensor<1024xf16, #blocked>
    %m = "tt.reduce"(%a) ({
    ^bb0(%p: f16, %q: f16):
      %s = arith.maxnumf %p, %q : f16
      tt.reduce.return %s : f16
    }) {axis = 0 : i32} : (tensor<1024xf16, #blocked>) -> f16
    %m_t = tt.splat %m : f16 -> tensor<1024xf16, #blocked>
    %o_splat = tt.splat %out_ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %o_addr = tt.addptr %o_splat, %offsets : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %o_addr, %m_t : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: metal.kernel reduce_sum_exp2
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: metal.unary_exp {{.*}}, exp2Op
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp

// CHECK-LABEL: metal.kernel reduce_sum_floor
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: math.floor
// CHECK: metal.binary_exp {{.*}}, {{.*}}, addOp

// CHECK-LABEL: metal.kernel reduce_max_absf_neg
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (f32)
// CHECK: arith.negf
// CHECK: metal.unary_exp {{.*}}, absOp
// CHECK: metal.binary_exp {{.*}}, {{.*}}, maxOp

// CHECK-LABEL: metal.kernel reduce_sum_absi
// CHECK: math.absi

// The 16-bit abs is a bitcast + mask, never a unary_exp absOp.
// CHECK-LABEL: metal.kernel reduce_max_absf_f16
// CHECK: metal.bitcast
// CHECK: metal.binary_exp {{.*}}, {{.*}}, bitAndOp
// CHECK: metal.bitcast
