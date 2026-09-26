// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s

// Pruning the causal half of the key sweep.
//
// A causal kernel puts its mask INSIDE the score region and leaves its key LOOP
// over the whole range. The emitter reproduces the source's own loop bound, so
// it used to sweep every key block; for the blocks above the diagonal every
// score is the region's false arm, which under an online softmax is a -inf
// logit -- exactly zero weight, and the running (max, sum, accumulator) is left
// alone. Those blocks are dead work: half the matrix work and half the global
// traffic, measured at 1.38-1.67x on the corpus causal kernel.
//
// `fusedAttentionKeysAboveRowAreInert` decides this structurally: the yielded
// value is a `select` whose FALSE arm is the inert value for the norm, and
// whose condition is an AND-chain carrying `row >= key` (or `row > key`, or the
// mirrored `key <= row` / `key < row`). One false conjunct falsifies the whole
// condition, so the other conjuncts need not be understood.
//
// The kernels below are the spellings that MUST prune -- including a two-phase
// op whose every phase is causal -- and the shapes that MUST NOT. Two of those
// would be a silent wrong answer, not a missed optimisation.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    // `row >= key` as one conjunct of an AND chain -- the shape a Triton
    // causal kernel's `tl.where((qo[:, None] >= so[None, :]) & mask, s, -inf)`
    // lowers to.
    metal.kernel causal_and_chain_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %zero = metal.constant 0 : si32
        %causal = metal.binary_exp %row, %key, geOp : (si32, si32) -> i1
        %nonneg = metal.binary_exp %key, %zero, geOp : (si32, si32) -> i1
        %keep = metal.binary_exp %causal, %nonneg, andOp : (i1, i1) -> i1
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        %ninf = metal.constant 0xFF800000 : f32
        %out = arith.select %keep, %scaled, %ninf : f32
        metal.score_yield %out : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
    // The mirrored spelling, and with no other conjunct at all.
    metal.kernel causal_mirrored_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %keep = metal.binary_exp %key, %row, leOp : (si32, si32) -> i1
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        %ninf = metal.constant 0xFF800000 : f32
        %out = arith.select %keep, %scaled, %ninf : f32
        metal.score_yield %out : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
    // No causal conjunct: a missed optimisation if pruned, nothing worse.
    metal.kernel plain_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %zero = metal.constant 0 : si32
        %keep = metal.binary_exp %key, %zero, geOp : (si32, si32) -> i1
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        %ninf = metal.constant 0xFF800000 : f32
        %out = arith.select %keep, %scaled, %ninf : f32
        metal.score_yield %out : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
    // ⚠️ The dangerous one. `row >= key` is there, but the false arm is a ZERO
    // LOGIT, not -inf -- under an online softmax that is a weight of
    // `exp(0 - m)`, which is emphatically not "contributes nothing". Pruning
    // here would be a SILENT WRONG ANSWER, so the false arm is checked, not
    // just the condition.
    metal.kernel causal_zero_false_arm_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %keep = metal.binary_exp %row, %key, geOp : (si32, si32) -> i1
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        %zerof = metal.constant 0.000000e+00 : f32
        %out = arith.select %keep, %scaled, %zerof : f32
        metal.score_yield %out : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        metal.key_bounds_yield %zero, %n2
      }
      metal.return
    }
    // Two phases, the attention-with-sinks shape: the region yields
    // `select(phase == 0, sink, window)` and EACH arm is its own causal select
    // with a -inf false arm. Neither arm's condition mentions the phase, and
    // the outer select's condition is not causal at all -- it is inert above
    // the row because both of its arms are, so every phase is clamped.
    metal.kernel two_phase_causal_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, num_phases = 2 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %zero = metal.constant 0 : si32
        %eight = metal.constant 8 : si32
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        %ninf = metal.constant 0xFF800000 : f32
        %sinkCausal = metal.binary_exp %key, %row, leOp : (si32, si32) -> i1
        %isSinkCol = metal.binary_exp %key, %eight, ltOp : (si32, si32) -> i1
        %sinkKeep = metal.binary_exp %isSinkCol, %sinkCausal, andOp : (i1, i1) -> i1
        %sink = arith.select %sinkKeep, %scaled, %ninf : f32
        %winCausal = metal.binary_exp %row, %key, geOp : (si32, si32) -> i1
        %pastSinks = metal.binary_exp %key, %eight, geOp : (si32, si32) -> i1
        %winKeep = metal.binary_exp %winCausal, %pastSinks, andOp : (i1, i1) -> i1
        %window = arith.select %winKeep, %scaled, %ninf : f32
        %isSink = metal.binary_exp %phase, %zero, eqOp : (si32, si32) -> i1
        %out = arith.select %isSink, %sink, %window : f32
        metal.score_yield %out : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        %eight = metal.constant 8 : si32
        %isSink = metal.binary_exp %ph, %zero, eqOp : (si32, si32) -> i1
        %beg = arith.select %isSink, %zero, %eight : si32
        %end = arith.select %isSink, %eight, %n2 : si32
        metal.key_bounds_yield %beg, %end
      }
      metal.return
    }
    // ⚠️ The same two phases with the window phase NOT causal. The sink arm
    // alone is inert above the row, so a proof that looked at one phase -- or
    // clamped each phase on the strength of another -- would silently drop the
    // window phase's keys above the diagonal. The clamp must not appear in
    // EITHER phase: the bound is emitted per phase from a single answer.
    metal.kernel two_phase_one_open_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, num_phases = 2 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32):
        %zero = metal.constant 0 : si32
        %eight = metal.constant 8 : si32
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        %ninf = metal.constant 0xFF800000 : f32
        %sinkCausal = metal.binary_exp %key, %row, leOp : (si32, si32) -> i1
        %isSinkCol = metal.binary_exp %key, %eight, ltOp : (si32, si32) -> i1
        %sinkKeep = metal.binary_exp %isSinkCol, %sinkCausal, andOp : (i1, i1) -> i1
        %sink = arith.select %sinkKeep, %scaled, %ninf : f32
        %winKeep = metal.binary_exp %key, %eight, geOp : (si32, si32) -> i1
        %window = arith.select %winKeep, %scaled, %ninf : f32
        %isSink = metal.binary_exp %phase, %zero, eqOp : (si32, si32) -> i1
        %out = arith.select %isSink, %sink, %window : f32
        metal.score_yield %out : f32
      } bounds {
      ^bb0(%blk: si32, %ph: si32, %m2: si32, %n2: si32):
        %zero = metal.constant 0 : si32
        %eight = metal.constant 8 : si32
        %isSink = metal.binary_exp %ph, %zero, eqOp : (si32, si32) -> i1
        %beg = arith.select %isSink, %zero, %eight : si32
        %end = arith.select %isSink, %eight, %n2 : si32
        metal.key_bounds_yield %beg, %end
      }
      metal.return
    }
  }
}

// CHECK-LABEL: kernel void causal_and_chain_kernel
// CHECK: uint _fa_kend =
// CHECK-NEXT: _fa_kend = min(_fa_kend, _fa_rowoff + 32u);

// CHECK-LABEL: kernel void causal_mirrored_kernel
// CHECK: uint _fa_kend =
// CHECK-NEXT: _fa_kend = min(_fa_kend, _fa_rowoff + 32u);

// Both phases of the two-phase kernel, each in its own key sweep.
// CHECK-LABEL: kernel void two_phase_causal_kernel
// CHECK: uint _fa_kend =
// CHECK-NEXT: _fa_kend = min(_fa_kend, _fa_rowoff + 32u);
// CHECK: uint _fa_kend =
// CHECK-NEXT: _fa_kend = min(_fa_kend, _fa_rowoff + 32u);

// The two that must keep the source's own key range.
// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s --check-prefix=KEEP
// KEEP: kernel void plain_kernel
// KEEP-NOT: _fa_kend = min(_fa_kend, _fa_rowoff
// KEEP: kernel void causal_zero_false_arm_kernel
// KEEP-NOT: _fa_kend = min(_fa_kend, _fa_rowoff
// KEEP: return;
// KEEP: kernel void two_phase_one_open_kernel
// KEEP-NOT: _fa_kend = min(_fa_kend, _fa_rowoff
// KEEP: return;
