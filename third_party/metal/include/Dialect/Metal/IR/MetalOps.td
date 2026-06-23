//===--- MetalOps.td - Metal dialect ops -------------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_OPS
#define METAL_OPS

include "Dialect/Metal/IR/MetalTypes.td"
include "Dialect/Metal/IR/MetalEnums.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Metal_Type : AnyTypeOf<[I1, I8, UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64,
                            F16, F32, BF16]>;

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

def ModuleOp : Metal_Op<"module", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ModuleEndOp">]> {
  let summary = "Module operation";

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins)>];

  let hasCustomAssemblyFormat = 1;
}

def ModuleEndOp : Metal_Op<"module_end", [
    Terminator, HasParent<"ModuleOp">]> {
  let summary = "Module end operation";

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

def KernelOp : Metal_Op<"kernel", [SingleBlockImplicitTerminator<"ReturnOp">,
    HasParent<"ModuleOp">]> {
  let summary = "Kernel operation";

  let arguments = (ins StrAttr:$name, ArrayAttr:$address_space_device);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "llvm::SmallVectorImpl<Type>&":$buffers,
                      "llvm::SmallVectorImpl<bool>&":$isAddressSpaceDevice)>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    mlir::Block& getEntryBlock();
    mlir::Value getBuffer(uint32_t index);
    mlir::MutableArrayRef<mlir::BlockArgument> getBuffers();
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Metal_Op<"return", [Pure, Terminator]> {
  let summary = "Return operation";

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Metal_Op<"constant", [ConstantLike, Pure, AllTypesMatch<["value", "result"]>,]> {
  let summary = "Constant operation";

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Metal_Type:$result);

  let builders = [
    OpBuilder<(ins "TypedAttr":$attr)>
  ];

  let hasFolder = 1;

  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def AllocaOp : Metal_Op<"alloca"> {
  let summary = "Alloca operation";

  let results = (outs Metal_MemRefType);

  let builders = [
    OpBuilder<(ins "MetalMemRefType":$type), [{
      $_state.addTypes(type);
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// ThreadgroupAllocaOp
//===----------------------------------------------------------------------===//

def ThreadgroupAllocaOp : Metal_Op<"threadgroup_alloca"> {
  let summary = "Threadgroup-shared memory allocation operation";
  let description = [{
    Allocates a threadgroup-shared array of the result memref's element type
    and static size. Emits an MSL `threadgroup <elem> _alloca_<id>[<N>];`
    declaration inside the kernel body. Shared with §3.5 staged-transpose;
    introduced by §3.3 Session L3.
  }];

  let results = (outs Metal_MemRefType);

  let builders = [
    OpBuilder<(ins "MetalMemRefType":$type), [{
      $_state.addTypes(type);
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// BarrierOp
//===----------------------------------------------------------------------===//

def BarrierOp : Metal_Op<"barrier",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Threadgroup memory barrier";
  let description = [{
    Emits `threadgroup_barrier(mem_flags::mem_threadgroup);` in MSL.
    Synchronizes all threads in the threadgroup with respect to threadgroup
    memory. Shared with §3.5 staged-transpose; introduced by §3.3 Session L3.

    Carries an effects=write trait so MLIR's canonicalizer/CSE cannot fold
    two barriers into one. The L1d2 staged-transpose body emits two barriers
    per cvt (one after store, one after load) — the trailing barrier is
    paranoid-safe for multi-cvt kernels where a follow-up cvt's alloca could
    alias the in-flight buffer. Without this trait the second barrier is
    CSE-elided.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// TgStoreIndexedOp
//===----------------------------------------------------------------------===//

def TgStoreIndexedOp : Metal_Op<"tg_store_indexed"> {
  let summary = "Store a scalar into a threadgroup-memory buffer at an index.";
  let description = [{
    Emits MSL `<bufVar>[<idx>] = <val>;` for an indexed store into a
    threadgroup-shared buffer (typically produced by `metal.threadgroup_alloca`).
    Phase A infrastructure for §3.5 staged-transpose (L1d2). Introduced by
    Session L1d Phase A.
  }];

  let arguments = (ins Metal_MemRefType:$buffer, UI32:$index, Metal_Type:$value);

  let hasVerifier = 1;

  let assemblyFormat = "$buffer `[` $index `]` `,` $value attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// TgLoadIndexedOp
//===----------------------------------------------------------------------===//

def TgLoadIndexedOp : Metal_Op<"tg_load_indexed", [Pure]> {
  let summary = "Load a scalar from a threadgroup-memory buffer at an index.";
  let description = [{
    Emits MSL `<bufVar>[<idx>]` as a value expression for an indexed load from
    a threadgroup-shared buffer (typically produced by `metal.threadgroup_alloca`).
    Phase A infrastructure for §3.5 staged-transpose (L1d2). Introduced by
    Session L1d Phase A.
  }];

  let arguments = (ins Metal_MemRefType:$buffer, UI32:$index);
  let results = (outs Metal_Type:$result);

  let hasVerifier = 1;

  let assemblyFormat = "$buffer `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : Metal_Op<"store"> {
  let summary = "Store operation";

  let arguments = (ins Metal_Type:$value, Metal_MemRefType:$memref, UI32:$index);

  let hasVerifier = 1;

  let assemblyFormat = "$value `,` $memref `[` $index `]`  attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

def GetElementOp : Metal_Op<"get_element", [Pure]> {
  let summary = "Get element operation";

  let arguments = (ins Metal_MemRefType:$memref, UI32:$index);
  let results = (outs Metal_Type:$result);

  let builders = [OpBuilder<(ins "Value":$memref, "Value":$index)>];

  let hasVerifier = 1;

  let assemblyFormat = "$memref `[` $index `]`  attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// AtomicRmwOp
//===----------------------------------------------------------------------===//

def AtomicRmwOp : Metal_Op<"atomic_rmw"> {
  let summary = "Atomic read-modify-write (fadd) on a device-memory element.";
  let description = [{
    Atomically adds `value` into `memref[index]` and yields the old value.
    Lowers to MSL `atomic_fetch_add_explicit(..., memory_order_relaxed)`.
    Models Triton `tt.atomic_rmw fadd` (float add only). NOT `Pure` — it has
    a device-memory side effect, so it survives dead-code elimination even when
    its result is unused.
  }];

  let arguments = (ins Metal_Type:$value, Metal_MemRefType:$memref, UI32:$index);
  let results = (outs Metal_Type:$result);

  let assemblyFormat = "$value `,` $memref `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

def ThreadIdOp : Metal_Op<"thread_id", [Pure]> {
  let summary = "Thread id operation";

  let arguments = (ins StrAttr:$dimension);
  let results = (outs UI32);

  let builders = [OpBuilder<(ins "StringRef":$dimension)>];

  let hasVerifier = 1;

  let assemblyFormat = "$dimension attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// ThreadgroupIdOp - maps to MSL [[threadgroup_position_in_grid]]
//===----------------------------------------------------------------------===//

def ThreadgroupIdOp : Metal_Op<"threadgroup_id", [Pure]> {
  let summary = "Threadgroup position in grid (a.k.a. program id)";

  let arguments = (ins StrAttr:$dimension);
  let results = (outs UI32);

  let builders = [OpBuilder<(ins "StringRef":$dimension)>];

  let hasVerifier = 1;

  let assemblyFormat = "$dimension attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// ThreadgroupsPerGridOp - maps to MSL [[threadgroups_per_grid]]
//===----------------------------------------------------------------------===//

def ThreadgroupsPerGridOp : Metal_Op<"threadgroups_per_grid", [Pure]> {
  let summary = "Number of threadgroups in grid along a dimension (a.k.a. num_programs)";

  let arguments = (ins StrAttr:$dimension);
  let results = (outs UI32);

  let builders = [OpBuilder<(ins "StringRef":$dimension)>];

  let hasVerifier = 1;

  let assemblyFormat = "$dimension attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Metal_Op<"cast", [Pure]> {
  let summary = "Cast operation";

  let arguments = (ins Metal_Type:$argument);
  let results = (outs Metal_Type);

  let assemblyFormat = "$argument  attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// UnaryExpOp
//===----------------------------------------------------------------------===//

def UnaryExpOperatorNot : I64EnumAttrCase<"notOp", 0>;
def UnaryExpOperatorMinus : I64EnumAttrCase<"minusOp", 1>;
def UnaryExpOperatorExp : I64EnumAttrCase<"expOp", 2>;
def UnaryExpOperatorSqrt : I64EnumAttrCase<"sqrtOp", 3>;
def UnaryExpOperatorErf : I64EnumAttrCase<"erfOp", 4>;
def UnaryExpOperatorLog : I64EnumAttrCase<"logOp", 5>;
def UnaryExpOperatorRsqrt : I64EnumAttrCase<"rsqrtOp", 6>;
def UnaryExpOperatorSin : I64EnumAttrCase<"sinOp", 7>;
def UnaryExpOperatorCos : I64EnumAttrCase<"cosOp", 8>;

def UnaryExpOperator : I64EnumAttr<
    "UnaryExpOperator", "",
    [UnaryExpOperatorNot, UnaryExpOperatorMinus, UnaryExpOperatorExp,
     UnaryExpOperatorSqrt, UnaryExpOperatorErf, UnaryExpOperatorLog,
     UnaryExpOperatorRsqrt, UnaryExpOperatorSin, UnaryExpOperatorCos]> {
  let cppNamespace = "mlir::triton::metal";
}

def UnaryExpOp : Metal_Op<"unary_exp", [Pure]> {
  let summary = "Unary expression operation";

  let arguments = (ins UnaryExpOperator:$unaryOperator, Metal_Type:$argument);
  let results = (outs Metal_Type);

  let builders = [
    OpBuilder<(ins "UnaryExpOperator":$unaryOperator, "Value":$argument)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$argument `,` $unaryOperator attr-dict `:` functional-type(operands, results)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// BinaryExpOp
//===----------------------------------------------------------------------===//

def BinaryExpOperatorAdd : I64EnumAttrCase<"addOp", 0>;
def BinaryExpOperatorSub : I64EnumAttrCase<"subOp", 1>;
def BinaryExpOperatorMul : I64EnumAttrCase<"mulOp", 2>;
def BinaryExpOperatorDiv : I64EnumAttrCase<"divOp", 3>;
def BinaryExpOperatorRem : I64EnumAttrCase<"remOp", 4>;
def BinaryExpOperatorEq  : I64EnumAttrCase<"eqOp", 5>;
def BinaryExpOperatorNe  : I64EnumAttrCase<"neOp", 6>;
def BinaryExpOperatorLt : I64EnumAttrCase<"ltOp", 7>;
def BinaryExpOperatorLe : I64EnumAttrCase<"leOp", 8>;
def BinaryExpOperatorGt : I64EnumAttrCase<"gtOp", 9>;
def BinaryExpOperatorGe : I64EnumAttrCase<"geOp", 10>;
def BinaryExpOperatorAnd : I64EnumAttrCase<"andOp", 11>;
def BinaryExpOperatorOr : I64EnumAttrCase<"orOp", 12>;
// Rank-1 reduce extension (spec
// `.omc/specs/deep-interview-metal-rank1-reduce.md`): emit MSL `max(a, b)`
// via the function-call form so combinators like `arith.maximumf` / `arith.
// maxsi` can lower to a single BinaryExp node, mirroring how `addOp`
// lowers to `(a) + (b)` for `arith.addf` / `arith.addi`.
def BinaryExpOperatorMax : I64EnumAttrCase<"maxOp", 13>;

def BinaryExpOperator : I64EnumAttr<
    "BinaryExpOperator", "",
    [BinaryExpOperatorAdd,
     BinaryExpOperatorSub,
     BinaryExpOperatorMul,
     BinaryExpOperatorDiv,
     BinaryExpOperatorRem,
     BinaryExpOperatorEq,
     BinaryExpOperatorNe,
     BinaryExpOperatorLt,
     BinaryExpOperatorLe,
     BinaryExpOperatorGt,
     BinaryExpOperatorGe,
     BinaryExpOperatorAnd,
     BinaryExpOperatorOr,
     BinaryExpOperatorMax]> {
  let cppNamespace = "mlir::triton::metal";
}

def BinaryExpOp : Metal_Op<"binary_exp", [Pure]> {
  let summary = "Binary expression operation";

  let arguments = (ins BinaryExpOperator:$binaryOperator, Metal_Type:$lhs, Metal_Type:$rhs);
  let results = (outs Metal_Type);

  let builders = [
    OpBuilder<(ins "BinaryExpOperator":$binaryOperator, "Value":$lhs, "Value":$rhs)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$lhs `,` $rhs `,` $binaryOperator  attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : Metal_Op<"matmul"> {
  let summary = "Stage-1 dense f32 row-contiguous matmul";
  let description = [{
    Stage-1 dense matmul: `C[M,N] = A[M,K] * B[K,N]`. Operands are flat
    `!metal.memref<? x f32>` (or static-size equivalents) interpreted as
    row-contiguous 2D matrices using the static `m`, `n`, `k` attributes.
    Not MLX Steel parity: this is a correctness-only Stage 1 op. GEMV
    (degenerate `M==1` or `N==1`) and quantized variants are explicitly
    out of scope.
  }];

  let arguments = (ins Metal_MemRefType:$lhs, Metal_MemRefType:$rhs,
                       Metal_MemRefType:$out,
                       I64Attr:$m, I64Attr:$n, I64Attr:$k,
                       DefaultValuedAttr<MatmulKindAttr,
                         "::mlir::triton::metal::MatmulKind::Scalar">:$kind);

  let hasVerifier = 1;

  let assemblyFormat = "$lhs `,` $rhs `,` $out attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// GemvOp
//===----------------------------------------------------------------------===//

def GemvOp : Metal_Op<"gemv"> {
  let summary = "Stage-2 dense matrix-vector multiply (A[M,K] * x[K] -> y[M])";
  let description = [{
    Stage-2 dense GEMV: `y[M] = A[M,K] * x[K]`. Operands are flat
    `!metal.memref<? x T>` (or static-size equivalents) where `T` is f16,
    f32, or bf16, interpreted as row-contiguous via the static `m` and `k`
    attributes. Naive one-thread-per-row scalar implementation; not Steel
    parity. Output `y` must be a device-address-space block argument of the
    enclosing metal.kernel.
  }];

  let arguments = (ins Metal_MemRefType:$lhs, Metal_MemRefType:$rhs,
                       Metal_MemRefType:$out,
                       I64Attr:$m, I64Attr:$k);

  let hasVerifier = 1;

  let assemblyFormat = "$lhs `,` $rhs `,` $out attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// QmvOp
//===----------------------------------------------------------------------===//

def QmvOp : Metal_Op<"qmv"> {
  let summary = "Stage-3 quantized matrix-vector multiply (MLX-compatible packed weights)";
  let description = [{
    Stage-3 quantized GEMV: `y[M] = dequant(Wq, scales, biases)[M,K] * x[K]`.
    Operands are flat `!metal.memref<? x T>` where Wq has element type
    `ui32` for bits in {2,4,8} or `ui8` for bits in {3,5,6}; scales/biases/x/y
    share element type T in {f16, f32, bf16}. Packed layout matches MLX
    `quantized.h` exactly. Naive scalar kernel.
  }];

  let arguments = (ins Metal_MemRefType:$wq, Metal_MemRefType:$scales,
                       Metal_MemRefType:$biases, Metal_MemRefType:$x,
                       Metal_MemRefType:$out,
                       I64Attr:$m, I64Attr:$k,
                       I64Attr:$bits, I64Attr:$group_size);

  let hasVerifier = 1;

  let assemblyFormat = "$wq `,` $scales `,` $biases `,` $x `,` $out attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// QmmOp
//===----------------------------------------------------------------------===//

def QmmOp : Metal_Op<"qmm"> {
  let summary = "Stage-3 quantized matrix-matrix multiply (MLX-compatible packed weights)";
  let description = [{
    Stage-3 quantized GEMM: `Y[M,N] = X[M,K] * dequant(Wq, scales, biases)[K,N]`
    where Wq is row-major over the output dim (K-major fast axis inside each
    output row). Operand element-type rules identical to `metal.qmv`. Naive
    scalar kernel.
  }];

  let arguments = (ins Metal_MemRefType:$wq, Metal_MemRefType:$scales,
                       Metal_MemRefType:$biases, Metal_MemRefType:$x,
                       Metal_MemRefType:$out,
                       I64Attr:$m, I64Attr:$n, I64Attr:$k,
                       I64Attr:$bits, I64Attr:$group_size);

  let hasVerifier = 1;

  let assemblyFormat = "$wq `,` $scales `,` $biases `,` $x `,` $out attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

def ReduceOp : Metal_Op<"reduce"> {
  let summary = "Stage-4 last-axis reduction (sum / max / mean)";
  let description = [{
    Stage-4 last-axis reduction: `y[m_outer] = reduce(x[m_outer, r], axis = -1)`
    where reduction is sum/max/mean per the `kind` attribute. Operands are flat
    `!metal.memref<? x T>` with T in {f16, f32, bf16}. Output is rank-1
    (flat size = m_outer). MSL emission uses MLX-native primitives (simd_sum /
    simd_max) — Stage 4 lifts the prior simdgroup ban for these ops only.
  }];

  let arguments = (ins Metal_MemRefType:$in, Metal_MemRefType:$out,
                       MetalReduceKindAttr:$kind,
                       I64Attr:$m_outer, I64Attr:$r);

  let hasVerifier = 1;

  let assemblyFormat = "$in `,` $out attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// ArgmaxOp
//===----------------------------------------------------------------------===//

def ArgmaxOp : Metal_Op<"argmax"> {
  let summary = "Stage-4 last-axis argmax (lowest-index-wins on ties)";
  let description = [{
    Stage-4 last-axis argmax: `out[m_outer] = argmax(in[m_outer, r], axis = -1)`.
    Input T ∈ {f16, f32, bf16}. Output element type is fixed to signless i32.
    Tie-break: lowest index wins (mirrors MLX ArgMax::reduce). MSL uses
    simd_shuffle_down over IndexValPair{uint32_t index; T val;}.
  }];

  let arguments = (ins Metal_MemRefType:$in, Metal_MemRefType:$out,
                       I64Attr:$m_outer, I64Attr:$r);

  let hasVerifier = 1;

  let assemblyFormat = "$in `,` $out attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

def SoftmaxOp : Metal_Op<"softmax"> {
  let summary = "Stage-4 last-axis softmax (numerically stable, AccT=float)";
  let description = [{
    Stage-4 last-axis softmax: `out[m_outer, r] = softmax(in[m_outer, r], axis = -1)`.
    Output shape == input shape. T in {f16, f32, bf16}. Numerically stable
    (subtract row max before exp). Accumulation in float regardless of T,
    matching MLX block_softmax_precise_.
  }];

  let arguments = (ins Metal_MemRefType:$in, Metal_MemRefType:$out,
                       I64Attr:$m_outer, I64Attr:$r);

  let hasVerifier = 1;

  let assemblyFormat = "$in `,` $out attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// LogsumexpOp
//===----------------------------------------------------------------------===//

def LogsumexpOp : Metal_Op<"logsumexp"> {
  let summary = "Stage-4 last-axis logsumexp (numerically stable, AccT=float)";
  let description = [{
    Stage-4 last-axis logsumexp: `out[m_outer] = log(sum_i exp(in[m_outer, i] - maxval)) + maxval`.
    Output rank-1 (one scalar per row). T in {f16, f32, bf16}. AccT=float.
    `isinf(maxval)` short-circuit matches MLX 0.31.2 semantics. Reuses softmax's
    max+sum body via the shared `emitMaxSumBody` helper.
  }];

  let arguments = (ins Metal_MemRefType:$in, Metal_MemRefType:$out,
                       I64Attr:$m_outer, I64Attr:$r);

  let hasVerifier = 1;

  let assemblyFormat = "$in `,` $out attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// SdpaOp
//===----------------------------------------------------------------------===//

def SdpaOp : Metal_Op<"sdpa"> {
  let summary = "Stage-5 single-pass scaled-dot-product attention (decode + small prefill)";
  let description = [{
    Stage-6 SDPA-vector subset of MLX 0.31.2 `sdpa_vector<T, D>`:
    O[n,d] = softmax((Q[n,d] · Kᵀ[d,k_len]) * scale + bias, axis=-1) · V[k_len,d].
    Single attention head; no GQA; no 2-pass. D ∈ {64,128}; N ∈ {1,8};
    T ∈ {f16,f32,bf16}. AccT=float invariant. One SIMD group per output row.
    Five mutually exclusive modes (Causal/BoolMask/FloatMask/Sinks/NonCausal)
    controlled by the `mode` attribute; see MaskSinkMode.
  }];
  let arguments = (ins Metal_MemRefType:$q, Metal_MemRefType:$k,
                       Metal_MemRefType:$v, Metal_MemRefType:$out,
                       Variadic<Metal_MemRefType>:$extras,
                       I64Attr:$d, I64Attr:$n, I64Attr:$k_len,
                       F32Attr:$scale, BoolAttr:$has_causal,
                       DefaultValuedAttr<MaskSinkModeAttr,
                         "::mlir::triton::metal::MaskSinkMode::Causal">:$mode);

  let hasVerifier = 1;

  let assemblyFormat =
      "$q `,` $k `,` $v `,` $out (`,` $extras^)? attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// RmsNormOp
//===----------------------------------------------------------------------===//

def RmsNormOp : Metal_Op<"rms_norm"> {
  let summary = "Stage-9 Llama-style RMS normalization (single-warp, AccT=float)";
  let description = [{
    Stage-9 Llama-style RMS normalization (forward):
    `y[i, c] = x[i, c] * gamma[c] * rsqrt(mean_c(x[i, c]^2) + eps)`.
    2D logical layout [n, d]; reduction along the last (feature) axis.
    T in {f16, f32, bf16}; gamma and out share T with x. AccT=float invariant
    (sum-of-squares + eps + rsqrt all in f32; only the final store is cast
    back to T). Single-warp simd_sum reduction; D in {64, 128, 256}; D % 32 == 0.
    eps is added INSIDE the rsqrt (Llama-2/3 / Mistral / Qwen convention).
  }];

  let arguments = (ins Metal_MemRefType:$x, Metal_MemRefType:$gamma,
                       Metal_MemRefType:$o,
                       I64Attr:$n, I64Attr:$d, F32Attr:$eps);

  let hasVerifier = 1;

  let assemblyFormat = "$x `,` $gamma `,` $o attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : Metal_Op<"yield", [Pure, Terminator]> {
  let summary = "Yield operation";

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : Metal_Op<"if", [RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "If-then-else operation";

  let arguments = (ins I1:$condition);

  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let builders = [
    OpBuilder<(ins "Value":$cond, "function_ref<void(OpBuilder &, Location)>":$thenBuilder,
               CArg<"function_ref<void(OpBuilder &, Location)>", "nullptr">:$elseBuilder)>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : Metal_Op<"while", [RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "While operation";

  let regions = (region SizedRegion<1>:$conditionRegion, SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "function_ref<void(OpBuilder &, Location)>":$conditionBuilder,
                      "function_ref<void(OpBuilder &, Location)>":$bodyBuilder)>
  ];

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;
}

def YieldWhileOp : Metal_Op<"while_yield", [Pure, Terminator,
                                            HasParent<"WhileOp">]> {
  let summary = "Yield while operation";

  let arguments = (ins I1:$condition);

  let assemblyFormat = "$condition attr-dict ";
}


//===----------------------------------------------------------------------===//
// ReleaseOp
//===----------------------------------------------------------------------===//

def ReleaseOp : Metal_Op<"release"> {
  let summary = "Release operation";

  let arguments = (ins Index:$ref);

  let assemblyFormat = "$ref attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// Runtime - Device
//===----------------------------------------------------------------------===//

def DeviceMakeDefaultOp : Metal_Op<"device_make_default"> {
  let summary = "Device make operation";

  let results = (outs Index:$device);

  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "attr-dict `:` type(results)";
}

def DeviceMakeCommandQueueOp : Metal_Op<"device_make_command_queue"> {
  let summary = "Device make command queue operation";

  let arguments = (ins Index:$device);
  let results = (outs Index:$commandQueue);

  let builders = [
    OpBuilder<(ins "Value":$device)>
  ];

  let assemblyFormat = "$device attr-dict `:` functional-type(operands, results)";
}

def DeviceMakeBufferOp : Metal_Op<"device_make_buffer"> {
  let summary = "Device make buffer operation";

  let arguments = (ins Index:$device, I1:$isStorageModeManaged,
                       I64:$count, I64:$sizeType);

  let results = (outs Index:$bufferOp);

  let builders = [
    OpBuilder<(ins "Value":$device, "Value":$isStorageModeManaged, "Value":$count,
                      "Value":$sizeType)>
  ];

  let assemblyFormat = "$device `,`$isStorageModeManaged `,` $count `,` $sizeType "
                       "attr-dict `:` functional-type(operands, results)";
}



//===----------------------------------------------------------------------===//
// Runtime - Buffer
//===----------------------------------------------------------------------===//

def BufferGetContentsOp : Metal_Op<"buffer_get_contents"> {
  let summary = "Buffer get contents operation";

  let arguments = (ins Index:$device);

  let results = (outs AnyMemRef);

  let builders = [
    OpBuilder<(ins "Value":$device, "Type":$elementType)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$device attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Runtime - Debug Print
//===----------------------------------------------------------------------===//

def PrintOp : Metal_Op<"print"> {
  let summary = "Host-side debug print of the first n elements of a CPU-visible memref.";
  let description = [{
    Prints the first `$n` elements of `$mem` to stdout as comma-separated
    `%f` values followed by `\n`. Only legal in non-kernel (host) regions,
    typically inside `func.func @main()` after `metal.buffer_get_contents`
    and `metal.command_buffer_wait_until_completed`. Element type must be
    one of f32, f16, bf16. Operand must be rank-1 `memref<?xT>` (standard
    MLIR memref, not `!metal.memref<? x T>`).

    n <= 0 is a no-op (prints nothing). n > buffer-size is undefined
    behavior (user responsibility, matching the contract of
    `metal.buffer_get_contents`).
  }];

  let arguments = (ins AnyMemRef:$mem, I64:$n);

  let hasVerifier = 1;

  let assemblyFormat = "$mem `,` $n attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// Runtime - CommandQueue
//===----------------------------------------------------------------------===//

def CommandQueueMakeCommandBufferOp : Metal_Op<"command_queue_make_command_buffer"> {
  let summary = "Command queue make command buffer operation";

  let arguments = (ins Index:$commandQueue, I64:$width, I64:$height, I64:$depth,
                       StrAttr:$functionName);
  let results = (outs Index);

  let builders = [
    OpBuilder<(ins "Value":$commandQueue, "StringRef":$functionName,
                      "Value":$width, "Value":$height, "Value":$depth)>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Runtime - CommandBuffer
//===----------------------------------------------------------------------===//

def CommandBufferAddBufferOp : Metal_Op<"command_buffer_add_buffer"> {
  let summary = "Command buffer add buffer operation";

  let arguments = (ins Index:$commandBuffer, Index:$bufferRef, I64:$index);

  let assemblyFormat = "$commandBuffer `,`$bufferRef `,` $index "
                        "attr-dict `:` functional-type(operands, results)";
}

def CommandBufferCommitOp : Metal_Op<"command_buffer_commit"> {
  let summary = "Command buffer commit operation";

  let arguments = (ins Index:$commandBuffer);

  let assemblyFormat = "$commandBuffer attr-dict `:` type(operands)";
}

def CommandBufferWaitUntilCompletedOp : Metal_Op<"command_buffer_wait_until_completed"> {
  let summary = "Command buffer wait until completed operation";

  let arguments = (ins Index:$commandBuffer);

  let assemblyFormat = "$commandBuffer attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// SIMD-group matrix ops (scaffolding for tt.dot lowering).
// See `.omc/specs/deep-interview-metal-matmul-session2-simdgroup-scaffold.md`.
//===----------------------------------------------------------------------===//

def SimdgroupLoadDeviceStagedOp : Metal_Op<"simdgroup_load_device_staged"> {
  let summary = "Stage a tile from device memory through threadgroup memory, then simdgroup_load";
  let description = [{
    Cooperative copy of an 8x8 tile from `device T*` into a threadgroup-shared
    buffer, followed by `simdgroup_load` from that buffer. Apple's matmul +
    flash-attention samples (and the in-tree FA emitter at
    ModuleTranslation.cpp:1340-1395) only chain `simdgroup_multiply_accumulate`
    on simdgroup_matrix values loaded from `threadgroup T*`. Direct chained
    sgmma on values loaded from `const device T*` silently drops output-column
    contributions on Apple GPU family 9 / Metal 17.5 (iter-8 root cause for
    [16,16,16] K_TILES>=2). This op encapsulates the staging block so the
    matcher emits a single IR node and the emitter expands it into:
      threadgroup float _stage_N[64];
      for (uint c = (id.x & 31u); c < 64u; c += 32u) {
        _stage_N[c] = src[(origin_row + c/8u) * stride + (origin_col + c%8u)];
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_<elem><M>x<N> vN;
      simdgroup_load(vN, &_stage_N[0], 8);
  }];

  let arguments = (ins Metal_MemRefType:$memref, UI32:$origin_row,
                       UI32:$origin_col, UI32:$stride,
                       Variadic<UI32>:$warp_index);
  let results = (outs Metal_SimdgroupMatrixType:$result);

  let assemblyFormat = "$memref `[` $origin_row `,` $origin_col `]` `,` $stride"
                       " (`warp` `[` $warp_index^ `]`)?"
                       " attr-dict `:` functional-type(operands, results)";

  let hasVerifier = 1;
}

def SimdgroupIndexOp : Metal_Op<"simdgroup_index", [Pure]> {
  let summary = "Threadgroup-local SIMD-group (warp) index";
  let description = [{
    Returns Apple's `simdgroup_index_in_threadgroup` builtin (UI32).
    The value is invariant over the kernel region and CSE'able across
    uses within the same kernel; `Pure` is conservatively correct.
  }];

  let arguments = (ins);
  let results = (outs UI32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def SimdgroupMatrixZeroOp : Metal_Op<"simdgroup_matrix_zero", [Pure]> {
  let summary = "SIMD-group matrix initialized to zero via constructor";
  let description = [{
    Materializes `simdgroup_matrix<T, M, N>(0.0)` directly. The Apple
    matmul / flash-attention samples initialize the accumulator this way
    instead of via `simdgroup_load` from a pre-zeroed device buffer; the
    constructor-init pattern is the only form that chained
    `simdgroup_multiply_accumulate` calls accumulate correctly across all
    output columns on Apple GPU family 9 / Metal 17.5.
  }];

  let arguments = (ins);
  let results = (outs Metal_SimdgroupMatrixType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def SimdgroupLoadOp : Metal_Op<"simdgroup_load", [Pure]> {
  let summary = "Cooperative SIMD-group load of a matrix tile from device memory";

  let arguments = (ins Metal_MemRefType:$memref, UI32:$origin_row,
                       UI32:$origin_col, UI32:$stride);
  let results = (outs Metal_SimdgroupMatrixType:$result);

  let assemblyFormat = "$memref `[` $origin_row `,` $origin_col `]` `,` $stride"
                       " attr-dict `:` functional-type(operands, results)";
}

def SimdgroupMultiplyAccumulateOp
    : Metal_Op<"simdgroup_multiply_accumulate", [Pure]> {
  let summary = "SIMD-group fused multiply-accumulate: C + A * B";

  let arguments = (ins Metal_SimdgroupMatrixType:$c,
                       Metal_SimdgroupMatrixType:$a,
                       Metal_SimdgroupMatrixType:$b);
  let results = (outs Metal_SimdgroupMatrixType:$result);

  let assemblyFormat = "$c `,` $a `,` $b attr-dict"
                       " `:` functional-type(operands, results)";
}

def SimdgroupStoreOp : Metal_Op<"simdgroup_store"> {
  let summary = "Cooperative SIMD-group store of a matrix tile to device memory";

  let description = [{
    Stores an 8x8 SIMD-group matrix tile to a memref. When the variadic
    `partial_extents` operand list is non-empty (must be exactly 2 UI32
    values `[m_extent, n_extent]`), the lowering emits a two-stage
    epilogue: simdgroup_store -> per-threadgroup scratch, then a
    cooperative coop-loop that writes only the lanes whose global
    `(gi, gj)` satisfy `gi < m_extent && gj < n_extent` back to the
    destination memref. This is the AC2 masked-tail path for kernels
    whose `tt.store` carries a `(offs_m < M) & (offs_n < N)` mask.
  }];

  let arguments = (ins Metal_SimdgroupMatrixType:$matrix,
                       Metal_MemRefType:$memref, UI32:$origin_row,
                       UI32:$origin_col, UI32:$stride,
                       Variadic<UI32>:$partial_extents);

  let assemblyFormat = "$matrix `,` $memref `[` $origin_row `,` $origin_col `]`"
                       " `,` $stride (`partial` `[` $partial_extents^ `]`)?"
                       " attr-dict `:` type(operands)";

  let hasVerifier = 1;
}

#endif // METAL_OPS
