// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s
//
// metal.fused_attention: the generalized `Q@K^T -> score transform -> P@V`
// body. What used to require a NEW op per attention variant is here a REGION
// carrying the score transform, so both kernels below share one emitter.
//
//   decay_kernel   — norm = None. The region is the causal decay of
//                    leet-triton/medium-decaying_causal_attention.py:
//                    s * scale * exp2((row-key) >= 0 ? (row-key)*log2(gamma)
//                                                    : -inf)
//                    Note there is NO separate mask operand: masking is what
//                    the region computes, exactly as the source kernel writes
//                    it.
//   softmax_kernel — norm = OnlineSoftmax with an EMPTY score_params list.
//                    The scale is a constant Triton folded out of the
//                    signature, and under this op a folded constant needs no
//                    operand at all — it is just `metal.constant` in the
//                    region. That is what retires the `Optional<operand> +
//                    *_const attribute` pair the predecessor ops carried per
//                    parameter.
//
// Buffers: v0..v3 = q/k/v/out, v4 = m (query rows), v5 = n (keys),
// v6 = d_head, v7..v10 = the four row strides, v11 = log2(gamma).

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  metal.module {
    metal.kernel decay_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>, %lg: !metal.memref<? x f32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so params %lg {bm = 64 : i64, bn = 64 : i64, bd = 64 : i64, norm = 0 : i32} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x f32> {
      ^bb0(%score: f32, %row: si32, %key: si32, %log2g: f32):
        %diff = metal.binary_exp %row, %key, subOp : (si32, si32) -> si32
        %zero = metal.constant 0 : si32
        %ge = metal.binary_exp %diff, %zero, geOp : (si32, si32) -> i1
        %difff = metal.cast %diff : (si32) -> f32
        %expo = metal.binary_exp %difff, %log2g, mulOp : (f32, f32) -> f32
        %ninf = metal.constant 0xFF800000 : f32
        %sel = arith.select %ge, %expo, %ninf : f32
        %w = metal.unary_exp %sel, exp2Op : (f32) -> f32
        %out = metal.binary_exp %score, %w, mulOp : (f32, f32) -> f32
        metal.score_yield %out : f32
      }
      metal.return
    }
    metal.kernel softmax_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dh: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dh strides %sq, %sk, %sv, %so {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32):
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        metal.score_yield %scaled : f32
      }
      metal.return
    }
    metal.kernel head_split_kernel address_space_device [true, true, true, true, true, true, true, true, true, true, true, true] {
    ^bb0(%q: !metal.memref<? x f32>, %k: !metal.memref<? x f32>, %v: !metal.memref<? x f32>, %o: !metal.memref<? x f32>, %m: !metal.memref<? x ui32>, %n: !metal.memref<? x ui32>, %dm: !metal.memref<? x ui32>, %sq: !metal.memref<? x ui32>, %sk: !metal.memref<? x ui32>, %sv: !metal.memref<? x ui32>, %so: !metal.memref<? x ui32>, %h: !metal.memref<? x ui32>):
      metal.fused_attention %q, %k, %v, %o, %m, %n, %dm strides %sq, %sk, %sv, %so heads %h {bm = 32 : i64, bn = 32 : i64, bd = 16 : i64, norm = 1 : i32, softmax_natural_exp} : !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x f32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32>, !metal.memref<? x ui32> {
      ^bb0(%score: f32, %row: si32, %key: si32):
        %scale = metal.constant 2.500000e-01 : f32
        %scaled = metal.binary_exp %score, %scale, mulOp : (f32, f32) -> f32
        metal.score_yield %scaled : f32
      }
      metal.return
    }
  }
}

// --- decay kernel: norm = None. At 64x64x64 the simdgroup body would need
// 3*64*64 + 2*64*64 + 2*64*64 = 28672 floats against a 8192 budget, so this one
// takes the SCALAR floor instead of the op declining — that is what keeping two
// bodies buys.
// CHECK-LABEL: kernel void decay_kernel
// CHECK: threadgroup float _fa_qbuf[2048]
// CHECK: threadgroup float _fa_obuf[2048]
// CHECK-NOT: _fa_rmax
// bm=64 with bd=64 exceeds a single 32-lane pass, so the emitter chunks the
// query rows (2 x 32) instead of declining the way flash_attention's bm<=32
// gate did.
// CHECK: for (uint _fa_c0 = 0u; _fa_c0 < 64u; _fa_c0 += 32u)
// CHECK: _fa_a += _fa_qbuf
// --- the region, emitted inline as let-bindings with score/row/key bound ---
// CHECK: = _fa_a;
// CHECK: = (int)_fa_row;
// CHECK: = (int)_fa_key;
// CHECK: [0];
// CHECK: exp2(
// CHECK: _fa_obuf{{.*}} +=
// CHECK-NOT: _fa_den

// --- softmax kernel: same op and same region, but its working set
// (3*32*16 + 2*16*32 + 2*32*32 + 2*32 = 4672 floats) fits Apple's 8192, so the
// emitter picks the SIMDGROUP body. Both matmuls run on the matrix unit and the
// region is evaluated over the staged S tile — which is the whole point: one
// emitter, two bodies, no per-variant code.
// CHECK-LABEL: kernel void softmax_kernel
// CHECK: threadgroup float _fa_sbuf[1024]
// CHECK: threadgroup float _fa_pbuf[1024]
// CHECK: threadgroup float _fa_rmax[32]
// CHECK: simdgroup_multiply_accumulate
// the region, inlined over the staged score tile
// CHECK: = _fa_sbuf[q*32u + kk];
// CHECK: _fa_pbuf[q*32u + kk] =
// running state + rescale + epilogue divide
// CHECK: float m_new = max(m_old, m_cur)
// CHECK: _fa_rsum[q] = _fa_rsum[q]*scaler + denom
// CHECK: simdgroup_multiply_accumulate
// CHECK: / denom

// --- head-split kernel: `heads` present, so the per-head feature width is
// DERIVED (`d_model / h`) rather than read from a buffer, and the feature
// column is offset by the grid's y dimension. A head-split Triton kernel
// computes `d_head = d_model // h` itself, so there is no kernel argument for a
// d_head buffer to point at -- carrying `h` and dividing here is what let this
// family be claimed at all (it moved the parity bar 10 -> 24 on its own).
// CHECK-LABEL: kernel void head_split_kernel
// CHECK: uint3 tgid {{\[\[}}threadgroup_position_in_grid]]
// CHECK: uint3 ltid {{\[\[}}thread_position_in_threadgroup]]
// CHECK: uint _fa_dh = v6[0] / v11[0];
// CHECK: uint _fa_col = tgid.y * _fa_dh;
// The padded feature columns d >= d_head must stage as zero, or the head slice
// bleeds into its neighbour.
// CHECK: _fa_qbuf[c] = (row < _fa_M && d < _fa_dh) ? v0[row * _fa_sq + _fa_col + d] : 0.0f;
//
// `softmax_natural_exp` picks base-e. It is not cosmetic: a kernel that folds
// log2(e) into its scale uses exp2 instead, and the region has already produced
// the logit, so the emitter cannot rescale after the fact.
// CHECK: float scaler = (m_old == m_new) ? 1.0f : exp(m_old - m_new);
// CHECK-NOT: exp2(
