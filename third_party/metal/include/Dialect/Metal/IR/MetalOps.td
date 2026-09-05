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
// Fp8ConvertOp
//===----------------------------------------------------------------------===//

def Fp8ConvertOp : Metal_Op<"fp8_convert", [Pure]> {
  let summary = "Convert between f32 and an 8-bit float format.";
  let description = [{
    What a standalone `tt.fp_to_fp` involving e4m3fn or e5m2 lowers to. MSL has
    no 8-bit float type, so the payload travels as `i8` bits and the emitter
    calls a preamble helper that does the conversion in software.

    `to_fp8` picks the direction: set, the operand is f32 and the result is the
    8-bit pattern; clear, the reverse. `e5m2` picks the format — the two differ
    in exponent width, bias, and whether they have an infinity (e4m3fn does
    not, so it overflows to NaN where e5m2 saturates to inf).

    Rounding is round-to-nearest-even, applied ONCE straight off the f32 bit
    pattern. Rounding through f16 on the way would double-round: 119 of 60032
    fuzz values came out a ulp away from what `torch.Tensor.to` produces.
  }];

  let arguments = (ins AnyTypeOf<[F32, I8]>:$value, UnitAttr:$to_fp8,
                       UnitAttr:$e5m2);
  let results = (outs AnyTypeOf<[F32, I8]>:$result);

  let assemblyFormat =
      "$value attr-dict `:` type($value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// DebugRecordOp
//===----------------------------------------------------------------------===//

def DebugRecordOp : Metal_Op<"debug_record", [AttrSizedOperandSegments]> {
  let summary = "Append one record to the kernel's host-readable debug buffer.";
  let description = [{
    What `tl.device_print` and `tl.device_assert` lower to. MSL has no printf
    and no host-visible assert channel, so the kernel writes fixed-size records
    into a trailing `device uint32_t*` buffer the launcher binds, and the host
    formats them after the dispatch completes.

    `kind` is 0 for a print and 1 for an assert; `msg_id` indexes the module's
    `metal.debug_messages` table, which the launcher receives alongside the MSL
    text. `pred`, when present, gates the record — an assert records only when
    its condition is FALSE. `value` is the scalar being printed; `dtype_tag`
    tells the host how to read its bits back (0 = i32, 1 = u32, 2 = f32,
    3 = f16, 4 = other).

    The record slot comes from an atomic increment of word 0, so a full buffer
    drops records rather than overwriting: the host reports the count it saw
    against the cap.
  }];

  let arguments = (ins I32Attr:$kind, I32Attr:$msg_id, I32Attr:$dtype_tag,
                       Optional<Metal_Type>:$value, Optional<I1>:$pred,
                       Optional<I32>:$lane);

  // Each optional operand gets its own keyword and no leading comma: with a
  // shared separator the parser commits to the first optional group and then
  // reports "expected 'if'" on a record that only carries a lane.
  let assemblyFormat = [{
    `kind` `=` $kind `,` `msg` `=` $msg_id (`value` $value^ `:` type($value))?
    (`if` $pred^)? (`lane` $lane^)? attr-dict
  }];
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
// ContiguousVectorAddOp
//===----------------------------------------------------------------------===//

def ContiguousVectorAddOp : Metal_Op<"contiguous_vector_add"> {
  let summary = "Aligned contiguous rank-1 vector addition with a scalar tail";
  let description = [{
    Computes `out[i] = x[i] + y[i]` for the canonical rank-1 masked Triton
    vector-add envelope. Each hardware thread owns `elements_per_thread`
    adjacent f32 elements. Full groups use aligned MSL vector transactions;
    the final partial group falls back to scalar guarded accesses, preserving
    the source `i < n` mask without reading or writing outside the buffers.

    This op is introduced only by a whole-function structural prepass that
    proves common contiguous indices, a shared canonical mask, and 16-byte
    alignment for all three buffer arguments. Kernels outside that exact
    envelope retain the ordinary scalar load/store lowering.
  }];

  let arguments = (ins Metal_MemRefType:$x, Metal_MemRefType:$y,
                       Metal_MemRefType:$out, I32:$n,
                       I64Attr:$elements_per_thread,
                       I64Attr:$vector_width);

  let hasVerifier = 1;

  let assemblyFormat = "$x `,` $y `,` $out `,` $n attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// AtomicRmwOp
//===----------------------------------------------------------------------===//

def AtomicRmwOp : Metal_Op<"atomic_rmw"> {
  let summary = "Atomic read-modify-write on a device-memory element.";
  let description = [{
    Applies `kind` to `value` and `memref[index]`, yielding the old value.
    `add` models Triton `tt.atomic_rmw fadd/add`; `min`/`max` and
    `umin`/`umax` preserve Triton's signed and unsigned integer ordering.
    The 32-bit integer kinds also model the ordered-bit expansion used by
    Triton's f32 atomic min/max. Unsigned 64-bit min/max are permitted only
    when the old-value result is unused because MSL's atomic_ulong modify
    functions return void. NOT `Pure` — it has a device-memory side effect,
    so it survives dead-code elimination even when its result is unused.
  }];

  let arguments = (ins AtomicRmwKindAttr:$kind, Metal_Type:$value,
                       Metal_MemRefType:$memref, UI32:$index);
  let results = (outs Metal_Type:$result);

  let hasVerifier = 1;

  let assemblyFormat = "$kind $value `,` $memref `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// AtomicCasOp
//===----------------------------------------------------------------------===//

def AtomicCasOp : Metal_Op<"atomic_cas"> {
  let summary = "Atomic compare-and-swap on a 32-bit device-memory element.";
  let description = [{
    Atomically compares the 32-bit object representation at `memref[index]`
    against `cmp` and stores `value` when equal, yielding the old bits observed
    at the memory location. Integer storage uses its native representation;
    f32 storage is accessed through a ui32 atomic view. NOT `Pure`: it has a
    device-memory side effect and must be emitted even when its result is
    unused.
  }];

  let arguments = (ins Metal_Type:$cmp, Metal_Type:$value,
                       Metal_MemRefType:$memref, UI32:$index);
  let results = (outs Metal_Type:$result);

  let hasVerifier = 1;

  let assemblyFormat = "$cmp `,` $value `,` $memref `[` $index `]` attr-dict `:` functional-type(operands, results)";
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

def BitcastOp : Metal_Op<"bitcast", [Pure]> {
  let summary = "Bit-preserving scalar reinterpret cast";

  let arguments = (ins Metal_Type:$argument);
  let results = (outs Metal_Type);

  let assemblyFormat = "$argument attr-dict `:` functional-type(operands, results)";
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
// Base-2 exponential. Distinct from expOp on purpose: the attention kernels
// fold log2(e) into their scale and use exp2, and a decay mask depends on
// exp2(-inf) == 0 exactly.
def UnaryExpOperatorExp2 : I64EnumAttrCase<"exp2Op", 9>;
def UnaryExpOperatorLog2 : I64EnumAttrCase<"log2Op", 10>;
def UnaryExpOperatorAbs : I64EnumAttrCase<"absOp", 11>;

def UnaryExpOperator : I64EnumAttr<
    "UnaryExpOperator", "",
    [UnaryExpOperatorNot, UnaryExpOperatorMinus, UnaryExpOperatorExp,
     UnaryExpOperatorSqrt, UnaryExpOperatorErf, UnaryExpOperatorLog,
     UnaryExpOperatorRsqrt, UnaryExpOperatorSin, UnaryExpOperatorCos,
     UnaryExpOperatorExp2, UnaryExpOperatorLog2, UnaryExpOperatorAbs]> {
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
// the implementation notes): emit MSL `max(a, b)`
// via the function-call form so combinators like `arith.maximumf` / `arith.
// maxsi` can lower to a single BinaryExp node, mirroring how `addOp`
// lowers to `(a) + (b)` for `arith.addf` / `arith.addi`.
def BinaryExpOperatorMax : I64EnumAttrCase<"maxOp", 13>;
// Rank-1 reduce min extension: emit MSL `min(a, b)` via the function-call form,
// mirroring maxOp, so `arith.minsi` / `arith.minnumf` reduce combines lower to a
// single BinaryExp node (identity INT32_MAX / +FLT_MAX in lowerRank1Reduce).
def BinaryExpOperatorMin : I64EnumAttrCase<"minOp", 14>;
// BITWISE and/or/xor, distinct from andOp/orOp above — those are the LOGICAL
// `&&` / `||` and produce i1. These emit MSL `&` / `|` / `^` and keep the
// operand width, so integer `tl.reduce` combines (`arith.andi` / `ori` / `xori`,
// which is what `tl.xor_sum` and bitwise-mask reductions lower to) can be a
// single BinaryExp node like addOp is. Identities: and → ~0, or/xor → 0.
def BinaryExpOperatorBitAnd : I64EnumAttrCase<"bitAndOp", 15>;
def BinaryExpOperatorBitOr  : I64EnumAttrCase<"bitOrOp", 16>;
def BinaryExpOperatorBitXor : I64EnumAttrCase<"bitXorOp", 17>;

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
     BinaryExpOperatorMax,
     BinaryExpOperatorMin,
     BinaryExpOperatorBitAnd,
     BinaryExpOperatorBitOr,
     BinaryExpOperatorBitXor]> {
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
// FmaOp
//===----------------------------------------------------------------------===//

def FmaOp : Metal_Op<"fma", [Pure]> {
  let summary = "Fused f32 multiply-add expression";
  let description = [{
    Computes `fma(a, b, c)` with one fused rounding. This is first-class rather
    than represented as `a * b + c` so Triton `math.fma` can preserve fused
    semantics through Metal emission.
  }];

  let arguments = (ins F32:$a, F32:$b, F32:$c);
  let results = (outs F32:$result);

  let builders = [
    OpBuilder<(ins "Value":$a, "Value":$b, "Value":$c)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// MathIntrinsicOp
//===----------------------------------------------------------------------===//

def MathIntrinsicOp : Metal_Op<"math_intrinsic", [Pure]> {
  let summary = "Call a named MSL math intrinsic";
  let description = [{
    Emits `<callee>(<args>...)` verbatim into MSL. This backs
    `tt.extern_elementwise`, i.e. `triton.language.extra.libdevice`, whose
    surface is far too wide to give each function its own dialect op or
    `UnaryExpOperator` case.

    `callee` is written into the generated shader unchanged, so it must only
    ever come from the backend's own symbol table in `ExternElementwiseLowering`
    — never from a user-supplied string. An unknown libdevice symbol is
    rejected there rather than reaching this op.
  }];

  let arguments = (ins StrAttr:$callee, Variadic<Metal_Type>:$args);
  let results = (outs Metal_Type:$result);

  let assemblyFormat =
      "$callee `(` $args `)` attr-dict `:` functional-type($args, $result)";
}

//===----------------------------------------------------------------------===//
// ClampFOp
//===----------------------------------------------------------------------===//

def ClampFOp : Metal_Op<"clampf", [Pure]> {
  let summary = "Clamp an f32 value with explicit input-NaN propagation";
  let description = [{
    Computes `fmin(fmax(x, min), max)`. When `propagateNan` is true, an NaN
    value of `x` is returned directly instead of being consumed by the
    number-preferring min/max operations. Triton leaves NaN bounds undefined,
    so only `x` participates in the explicit propagation rule.
  }];

  let arguments = (ins F32:$x, F32:$min, F32:$max,
                        BoolAttr:$propagateNan);
  let results = (outs F32:$result);

  let builders = [
    OpBuilder<(ins "Value":$x, "Value":$min, "Value":$max,
                   "bool":$propagateNan)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$x `,` $min `,` $max `,` `propagate_nan` `=` $propagateNan attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// MulHiUIOp
//===----------------------------------------------------------------------===//

def MulHiUIOp : Metal_Op<"mulhi_ui", [Pure]> {
  let summary = "High half of an unsigned 32x32-bit or 64x64-bit product";
  let description = [{
    For ui32, zero-extends both operands to 64 bits and returns bits 32..63.
    For ui64, uses an exact 32-bit-limb decomposition and returns bits 64..127
    without requiring a 128-bit MSL type. Operands/results are explicitly
    unsigned so signless Triton integers cannot be sign-extended.
  }];

  let arguments = (ins AnyUnsignedInteger:$x, AnyUnsignedInteger:$y);
  let results = (outs AnyUnsignedInteger:$result);

  let builders = [OpBuilder<(ins "Value":$x, "Value":$y)>];

  let hasVerifier = 1;

  let assemblyFormat = "$x `,` $y attr-dict `:` functional-type(operands, results)";
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
// Int4WeightOnlyMatmulOp
//===----------------------------------------------------------------------===//

def Int4WeightOnlyMatmulOp : Metal_Op<"int4_weight_only_matmul"> {
  let summary = "Correctness-first int4 weight-only matmul fallback";
  let description = [{
    Specialized fallback for leet-triton's int4 weight-only GEMM:
    `Y[M,N] = X[M,K] * dequant(Wq, scales)[K,N]`.

    Wq is laid out by output column first. Each byte stores two K values:
    even K in the high nibble, odd K in the low nibble. The dequantized value
    is `(q - 8) * scales[n * stride_sn + k / group_size]`.

    The op is intentionally narrow. It carries runtime extents and leading
    strides from the original Triton signature, but assumes the inner K/N
    strides are one, matching the source kernel this fallback claims.
  }];

  let arguments = (ins Metal_MemRefType:$x, Metal_MemRefType:$wq,
                       Metal_MemRefType:$scales, Metal_MemRefType:$out,
                       Metal_MemRefType:$m, Metal_MemRefType:$n,
                       Metal_MemRefType:$k, Metal_MemRefType:$stride_xm,
                       Metal_MemRefType:$stride_wqn,
                       Metal_MemRefType:$stride_sn,
                       Metal_MemRefType:$stride_ym,
                       I64Attr:$group_size, I64Attr:$bm, I64Attr:$bn);

  let hasVerifier = 1;

  let assemblyFormat = "$x `,` $wq `,` $scales `,` $out `,` $m `,` $n `,` $k "
                       "`strides` $stride_xm `,` $stride_wqn `,` $stride_sn "
                       "`,` $stride_ym attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// Int8QuantizedMatmulOp
//===----------------------------------------------------------------------===//

def Int8QuantizedMatmulOp : Metal_Op<"int8_quantized_matmul"> {
  let summary = "Correctness-first asymmetric int8 matmul fallback";
  let description = [{
    Specialized fallback for leet-triton's asymmetric int8 GEMM. It computes
    the int32 matrix product and row/column sums, applies the three zero-point
    correction terms, rescales in f32, rounds with floor(x + 0.5), clamps to
    [-128, 127], and stores int8 output.

    The op is intentionally narrow: A, B, and C are contiguous row-major
    matrices, runtime extents come from the original Triton signature, and the
    currently supported source tile is 64x64 with a 64-element K step.
  }];

  let arguments = (ins Metal_MemRefType:$a, Metal_MemRefType:$b,
                       Metal_MemRefType:$out, Metal_MemRefType:$m,
                       Metal_MemRefType:$n, Metal_MemRefType:$k,
                       Metal_MemRefType:$scale_a, Metal_MemRefType:$scale_b,
                       Metal_MemRefType:$scale_c,
                       Metal_MemRefType:$zero_point_a,
                       Metal_MemRefType:$zero_point_b,
                       Metal_MemRefType:$zero_point_c,
                       I64Attr:$bm, I64Attr:$bn);

  let hasVerifier = 1;

  let assemblyFormat = "$a `,` $b `,` $out `,` $m `,` $n `,` $k "
                       "`scales` $scale_a `,` $scale_b `,` $scale_c "
                       "`zero_points` $zero_point_a `,` $zero_point_b `,` "
                       "$zero_point_c attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// LinearAttentionPreprocessOp / LinearAttentionApplyOp
//===----------------------------------------------------------------------===//

def LinearAttentionPreprocessOp : Metal_Op<"linear_attention_preprocess"> {
  let summary = "Correctness-first hard linear-attention preprocessing";
  let description = [{
    Specialized fallback for the first leet-triton hard linear-attention
    kernel.  For the source program's 256-row chunk it atomically accumulates
    `KV += phi(K)^T * V` and `Ksum += sum(phi(K), axis=0)`, where
    `phi(x) = x + 1` for positive x and `exp(x)` otherwise.

    The op is intentionally produced only by a conservative whole-kernel
    matcher.  It consumes the computed dot operand, its side reduction, and
    the rank-2/rank-1 atomic publication before those operations reach the
    generic layout and atomic lowerings.
  }];

  let arguments = (ins Metal_MemRefType:$kt, Metal_MemRefType:$v,
                       Metal_MemRefType:$kv, Metal_MemRefType:$ksum,
                       Metal_MemRefType:$m, Metal_MemRefType:$d);

  let hasVerifier = 1;

  let assemblyFormat = "$kt `,` $v `,` $kv `,` $ksum `,` $m `,` $d "
                       "attr-dict `:` type(operands)";
}

def LinearAttentionApplyOp : Metal_Op<"linear_attention_apply"> {
  let summary = "Correctness-first hard linear-attention output kernel";
  let description = [{
    Specialized fallback for the second leet-triton hard linear-attention
    kernel.  Each source program covers a 64x64 output tile and computes
    `phi(Q) * KV / (phi(Q) * Ksum + 1e-5)` using runtime M and D bounds.

    The op is intentionally produced only by a conservative whole-kernel
    matcher so unrelated dot/reduce kernels keep using the generic paths.
  }];

  let arguments = (ins Metal_MemRefType:$q, Metal_MemRefType:$kv,
                       Metal_MemRefType:$ksum, Metal_MemRefType:$out,
                       Metal_MemRefType:$m, Metal_MemRefType:$d);

  let hasVerifier = 1;

  let assemblyFormat = "$q `,` $kv `,` $ksum `,` $out `,` $m `,` $d "
                       "attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// SoftmaxAttentionBackward*Op
//===----------------------------------------------------------------------===//

def AttentionBackwardScoreYieldOp : Metal_Op<
    "attention_backward_score_yield",
    [Pure, Terminator, HasParent<"SoftmaxAttentionBackwardPreOp">]> {
  let summary = "Terminator of a backward-pre score-transform region";
  let description = [{
    Yields the stored logit for one (query row, key) pair. The parent binds the
    region arguments to the raw Q@K value, global row/key indices, N_true, and
    sm_scale. Keeping this scalar transform in IR lets one MMA schedule preserve
    pure score variants without enumerating them in the emitter.
  }];

  let arguments = (ins F32:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def SoftmaxAttentionBackwardPreOp : Metal_Op<"attention_backward_pre"> {
  let summary = "Regionized softmax attention backward prepass";
  let description = [{
    Computes Q@K and dO@V with the source tile geometry, then runs `score` once
    per logical (row, key) element before the fixed online-softmax recurrence.
    The score region has the signature

        (raw_score: f32, row: si32, key: si32,
         n_true: ui32, sm_scale: f32) -> f32

    and must end in `metal.attention_backward_score_yield`. The recurrence,
    scratch addresses, and final state stores remain part of the op contract;
    only the pure elementwise score transform is data-driven.
  }];
  let arguments = (ins Metal_MemRefType:$q, Metal_MemRefType:$kt,
                       Metal_MemRefType:$vt, Metal_MemRefType:$dO,
                       Metal_MemRefType:$s, Metal_MemRefType:$st,
                       Metal_MemRefType:$dp, Metal_MemRefType:$dpt,
                       Metal_MemRefType:$mp, Metal_MemRefType:$lp,
                       Metal_MemRefType:$deltap,
                       Metal_MemRefType:$stride_qm,
                       Metal_MemRefType:$stride_kt,
                       Metal_MemRefType:$stride_vt,
                       Metal_MemRefType:$stride_dom,
                       Metal_MemRefType:$stride_sm,
                       Metal_MemRefType:$stride_stn,
                       Metal_MemRefType:$stride_dpm,
                       Metal_MemRefType:$stride_dptn,
                       Metal_MemRefType:$m, Metal_MemRefType:$n_true,
                       Metal_MemRefType:$d, Metal_MemRefType:$sm_scale,
                       I64Attr:$bm, I64Attr:$bn, I64Attr:$bd);
  let regions = (region SizedRegion<1>:$score);
  let hasVerifier = 1;
  let assemblyFormat = "$q `,` $kt `,` $vt `,` $dO `,` $s `,` $st `,` "
                       "$dp `,` $dpt `,` $mp `,` $lp `,` $deltap `,` "
                       "$stride_qm `,` $stride_kt `,` $stride_vt `,` "
                       "$stride_dom `,` $stride_sm `,` $stride_stn `,` "
                       "$stride_dpm `,` $stride_dptn `,` $m `,` "
                       "$n_true `,` $d `,` $sm_scale attr-dict `:` "
                       "type(operands) $score";
}

def AttentionBackwardGradientYieldOp : Metal_Op<
    "attention_backward_gradient_yield", [Pure, Terminator]> {
  let summary = "Terminator of a backward probability/gradient region";
  let description = [{
    Yields the scalar softmax probability and score gradient for one logical
    attention element. The parent DQ or DKDV op binds the region arguments to
    score, dp, m, l_safe, delta, and sm_scale respectively. Parent verifiers
    restrict this terminator to those two operations.
  }];

  let arguments = (ins F32:$p, F32:$ds);
  let hasVerifier = 1;
  let assemblyFormat = "$p `,` $ds attr-dict `:` type($p) `,` type($ds)";
}

def SoftmaxAttentionBackwardDqOp : Metal_Op<"attention_backward_dq"> {
  let summary = "Regionized softmax attention backward dQ";
  let description = [{
    Computes dQ using the source tile geometry. The `gradient` region has the
    signature

        (score: f32, dp: f32, m: f32, l_safe: f32,
         delta: f32, sm_scale: f32) -> (p: f32, ds: f32)

    and must end in `metal.attention_backward_gradient_yield`. The converter
    admits only proven rearrangements of the canonical probability and score
    gradient formulas; address, dot, and accumulator contracts remain fixed.
  }];
  let arguments = (ins Metal_MemRefType:$k, Metal_MemRefType:$s,
                       Metal_MemRefType:$dp, Metal_MemRefType:$mp,
                       Metal_MemRefType:$lp, Metal_MemRefType:$deltap,
                       Metal_MemRefType:$dq, Metal_MemRefType:$stride_kn,
                       Metal_MemRefType:$stride_sm,
                       Metal_MemRefType:$stride_dpm,
                       Metal_MemRefType:$stride_dqm, Metal_MemRefType:$m,
                       Metal_MemRefType:$d, Metal_MemRefType:$sm_scale,
                       I64Attr:$bm, I64Attr:$bn, I64Attr:$bd);
  let regions = (region SizedRegion<1>:$gradient);
  let hasVerifier = 1;
  let assemblyFormat = "$k `,` $s `,` $dp `,` $mp `,` $lp `,` $deltap "
                       "`,` $dq `,` $stride_kn `,` $stride_sm `,` "
                       "$stride_dpm `,` $stride_dqm `,` $m `,` $d `,` "
                       "$sm_scale attr-dict `:` type(operands) $gradient";
}

def SoftmaxAttentionBackwardDkdvOp : Metal_Op<"attention_backward_dkdv"> {
  let summary = "Regionized softmax attention backward dK/dV";
  let description = [{
    Computes dK and dV using the source tile geometry and a shared `gradient`
    region with the same ABI as `metal.attention_backward_dq`. The yielded p
    feeds dV while ds feeds dK. Source load/store addresses, dot roles, and the
    output accumulation condition remain part of the fixed op contract.
  }];
  let arguments = (ins Metal_MemRefType:$q, Metal_MemRefType:$dO,
                       Metal_MemRefType:$st, Metal_MemRefType:$dpt,
                       Metal_MemRefType:$mp, Metal_MemRefType:$lp,
                       Metal_MemRefType:$deltap, Metal_MemRefType:$dk,
                       Metal_MemRefType:$dv, Metal_MemRefType:$stride_qm,
                       Metal_MemRefType:$stride_dom,
                       Metal_MemRefType:$stride_stn,
                       Metal_MemRefType:$stride_dptn,
                       Metal_MemRefType:$stride_dkn,
                       Metal_MemRefType:$stride_dvn, Metal_MemRefType:$m,
                       Metal_MemRefType:$n, Metal_MemRefType:$d,
                       Metal_MemRefType:$sm_scale,
                       Metal_MemRefType:$accumulate,
                       I64Attr:$accumulate_mode, I64Attr:$bn,
                       I64Attr:$bm, I64Attr:$bd);
  let regions = (region SizedRegion<1>:$gradient);
  let hasVerifier = 1;
  let assemblyFormat = "$q `,` $dO `,` $st `,` $dpt `,` $mp `,` $lp "
                       "`,` $deltap `,` $dk `,` $dv `,` $stride_qm "
                       "`,` $stride_dom `,` $stride_stn `,` $stride_dptn "
                       "`,` $stride_dkn `,` $stride_dvn `,` $m `,` $n "
                       "`,` $d `,` $sm_scale `,` $accumulate attr-dict "
                       "`:` type(operands) $gradient";
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
// FusedAttentionOp
//===----------------------------------------------------------------------===//

def ScoreYieldOp : Metal_Op<"score_yield", [Pure, Terminator,
                                            HasParent<"FusedAttentionOp">]> {
  let summary = "Terminator of a metal.fused_attention score-transform region";
  let description = [{
    Yields the transformed score for one (query row, key) pair. Exactly one
    f32 operand; the parent op's `norm` attribute decides whether that value is
    a weight (`none`) or a logit (`online_softmax`).
  }];

  let arguments = (ins F32:$value);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def KeyBoundsYieldOp : Metal_Op<"key_bounds_yield", [Pure, Terminator,
                                                     HasParent<"FusedAttentionOp">]> {
  let summary = "Terminator of a metal.fused_attention key-bounds region";
  let description = [{
    Yields the half-open key range `[start, end)` that one phase of the parent
    op visits, as two si32 values.

    This is how the op REPRODUCES a source kernel's key set instead of proving
    something about it. A causal loop stops at `min((pid+1)*bm, n)`; an
    attention-sinks kernel visits a sink block and then a sliding window whose
    start is `max(pid*bm - window + 1, num_sinks)`. Both are ordinary scalar
    arithmetic over the block index and the kernel's own arguments, so both are
    expressible here -- which is what retired the `causal_key_bound` unit
    attribute rather than growing a second one beside it.
  }];

  let arguments = (ins SI32:$start, SI32:$end);

  let assemblyFormat = "$start `,` $end attr-dict";
}

// NOTE: deliberately NOT SingleBlockImplicitTerminator<"ScoreYieldOp">. That
// trait auto-inserts the terminator and therefore needs one that is buildable
// with no operands; `metal.score_yield` carries the transformed score, so the
// region's terminator is required and type-checked in `verify()` instead.
def FusedAttentionOp : Metal_Op<"fused_attention", [AttrSizedOperandSegments]> {
  let summary = "Fused Q@K^T -> score transform -> P@V attention body";

  let description = [{
    Self-contained attention body for one query-row block, generalizing the
    `dot -> elementwise -> dot` chain that every attention-family Triton kernel
    compiles to. Replaced `metal.flash_attention` and `metal.sink_attention`,
    both now deleted, whose semantics were baked into hand-printed MSL and
    therefore needed a NEW op — matcher and emitter both — for every new
    masking or weighting scheme.

    What varies between those variants is exactly one thing: the elementwise
    function applied to each raw `Q@K^T` score before it multiplies V. Here that
    function is DATA, carried in the `score` region, instead of code in the
    emitter:

        metal.fused_attention ... {
        ^bb0(%score: f32, %row: si32, %key: si32, %phase: si32, %p0: f32, ...):
          // arbitrary scalar arith/math on the score and its indices
          metal.score_yield %transformed : f32
        } bounds {
        ^bb0(%blk: si32, %phase: si32, %m: si32, %n: si32, %p0: f32, ...):
          metal.key_bounds_yield %start, %end
        }

    The score region is entered once per (query row, key) pair with `%score`
    bound to the raw dot product, `%row`/`%key` bound to the GLOBAL (not
    tile-relative) indices, `%phase` to the index of the key phase being swept,
    and the trailing block args bound to element 0 of each `score_params`
    buffer. Masking is expressed the way the source kernel writes
    it — yield `-inf` (or `0` under `norm = none`) for a key that must not
    contribute — rather than as an enumerated mode attribute.

    Two consequences worth naming, because both were recurring costs in the ops
    this replaces:

    - A kernel scalar that Triton FOLDED to a constant (it drops an argument
      equal to 1 from the signature) needs no operand at all: it is simply an
      `arith.constant` inside the region. The `Optional<operand> + *_const
      attribute` pair that both predecessor ops carried per parameter — and the
      "exactly one of the two is set" invariant that came with it — disappears.
    - A matcher targeting this op does not enumerate known score shapes. It
      absorbs every op between the two dots into the region and declines if any
      op resists absorption, so coverage is established by CONSTRUCTION. The
      predecessor design established it by pattern enumeration, which is what
      let a sliding-window / causal / ALiBi / softcap kernel be claimed as plain
      attention and silently miscompiled (max abs err 0.95-2.4).

    `norm` selects the epilogue: `none` accumulates `sum_key transformed * V`
    and stores it; `online_softmax` treats the transform's output as a logit and
    keeps the running (max, sum) rescale, dividing by the sum on the way out.

    `q`/`k`/`v`/`out` are kernel-argument buffers. Normally they are row-major
    device f32 with per-tensor row strides. Scalar address parameters are
    either kernel-argument ui32 buffers or literal ui32 values when Triton
    specialized a constexpr away. `m` is the query row count, `n` the key
    count, `d_head` the feature width. `h` is OPTIONAL: present when the grid's
    y dimension selects a head.

    `head_params` is the independent-head-storage alternative. When present it
    contains exactly `[stride_qh, stride_kh, stride_vh, stride_oh, groups]`,
    with each entry following the same buffer-or-literal scalar contract.
    The grid's y dimension selects query head `y`, while K/V use grouped head
    `y / groups`; the four strides address tensors stored as
    `[head, sequence, feature]`. This is deliberately distinct from `h`, whose
    heads occupy feature slices of one `[sequence, d_model]` tensor.

    `base_offsets`, when present, contains exactly four literal element offsets
    `[q, k, v, out]`. It represents packed storage such as one QKV allocation
    with K and V beginning at fixed column offsets. Keeping these offsets
    explicit prevents the address matcher from erasing scalar `tt.addptr`
    operations that the emitter would otherwise fail to reproduce.

    `feature_tiled` is the deliberately narrow alternative used by kernels
    whose y grid tiles the OUTPUT feature dimension rather than selecting a
    head. In that mode each program writes
    `[tgid.y * bd, min((tgid.y + 1) * bd, d_head))`, while Q@K reduces over the
    full `d_head`. `k_feature_major` states that K is already materialized as
    `[feature, key]`, so its address is `feature * stride_k + key`. The verifier
    currently requires those attributes together and rejects `h`: keeping the
    storage and grid contracts explicit prevents a transposed-K kernel from
    being silently emitted with the ordinary row-major/head-split indexing.

    `safe_denominator_one` reproduces the exact epilogue
    `acc / select(sum == 0, 1, sum)`. It is meaningful only for online softmax.

    `softmax_natural_exp` selects base-e for the `online_softmax` running
    state instead of the default base-2. Both spellings occur and they are NOT
    interchangeable: the kernels that fold `log2(e)` into their scale use
    `exp2`, while a kernel that scales by plain `1/sqrt(d)` uses `exp`. The
    exponential must match the source's, since the region has already produced
    the logit and the emitter cannot retroactively rescale it.

    The `key_bounds` region gives the half-open key range `[start, end)` that
    each phase visits, as scalar arithmetic over the block index, `m`/`n` and
    the same params the score region sees. Agreement with the source kernel is
    REPRODUCED here, not proven: a causally-bounded loop never visits keys past
    `min((pid+1)*bm, n)`, and that agrees with a `[0, n)` sweep only because the
    region happens to send `key > row` to zero — undecidable for an arbitrary
    region, and pattern-matching the causal shape to decide it would reintroduce
    exactly the enumeration this op exists to remove.

    `num_phases` is how many times that repeats. Both regions take a `phase`
    argument, so a kernel that visits several disjoint key ranges with a
    DIFFERENT mask on each — attention sinks being the motivating case: a sink
    block, then a sliding window that starts at
    `max(pid*bm - window + 1, num_sinks)` — is one op with `num_phases = 2`
    rather than a new op. The phases run in source order against one running
    (accumulator, sum, max) state, which is why they compose: an online softmax
    merges any partition of the key set.

    Collapsing several phases into one masked sweep is deliberately NOT done.
    It would be valid only when the ranges cover everything the masks keep, and
    for a sliding window that is a RUNTIME property (`n_local_blocks*bn >=
    window + bm - 1`, with `window` a kernel argument), so the matcher cannot
    check it. Reproducing costs nothing and needs no such argument.

    `bm` is the source kernel's query-row block and is BINDING — it fixes the
    grid, so this program owns rows `[tgid.x*bm, tgid.x*bm + bm)`. `bn`/`bd` are
    the source's key/feature tiles, recorded for the verifier; the emitter picks
    its own key and feature chunking to fit Apple's 32 KiB threadgroup budget
    and is therefore NOT capped by them. Online softmax is exact at any key
    blocking, so re-blocking changes only summation order, not the result.
  }];

  let arguments = (ins Metal_MemRefType:$q, Metal_MemRefType:$k,
                       Metal_MemRefType:$v, Metal_MemRefType:$out,
                       AnyType:$m, AnyType:$n, AnyType:$d_head,
                       AnyType:$stride_q, AnyType:$stride_k,
                       AnyType:$stride_v, AnyType:$stride_o,
                       Variadic<AnyType>:$head_params,
                       Variadic<UI32>:$base_offsets,
                       Optional<Metal_MemRefType>:$h,
                       Variadic<Metal_MemRefType>:$score_params,
                       I64Attr:$bm, I64Attr:$bn, I64Attr:$bd,
                       DefaultValuedAttr<I64Attr, "1">:$num_phases,
                       AttnNormAttr:$norm,
                       UnitAttr:$softmax_natural_exp,
                       UnitAttr:$safe_denominator_one,
                       UnitAttr:$feature_tiled,
                       UnitAttr:$k_feature_major);

  let regions = (region SizedRegion<1>:$score,
                        SizedRegion<1>:$key_bounds);

  let hasVerifier = 1;

  let assemblyFormat =
      "$q `,` $k `,` $v `,` $out `,` $m `,` $n `,` $d_head"
      " `strides` $stride_q `,` $stride_k `,` $stride_v `,` $stride_o"
      " (`head_params` $head_params^)?"
      " (`base_offsets` `[` $base_offsets^ `]`)?"
      " (`heads` $h^)? (`params` $score_params^)? attr-dict"
      " `:` type(operands) $score `bounds` $key_bounds";
}

//===----------------------------------------------------------------------===//
// ThreadgroupPrefixSumOp
//===----------------------------------------------------------------------===//

def ThreadgroupPrefixSumOp : Metal_Op<"threadgroup_prefix_sum",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Inclusive prefix scan (cumsum / cumprod) over a threadgroup buffer";
  let description = [{
    Reads `inbuf[0..block)` and writes the inclusive prefix scan to
    `outbuf[0..block)`, both `metal.threadgroup_alloca` buffers of size `block`.
    Tiled over `E = block/tpb` cyclic iv-blocks (logical position
    `pos = iv*tpb + tid`): each iv-block runs a double-barriered Hillis-Steele
    inclusive scan across the `tpb` threads via an internal scratch buffer, with
    a running `carry` combining all prior iv-blocks. Validated bit-close to
    torch.cumsum (scan_spike.py). Side-effecting (not Pure) so it is neither
    DCE'd nor CSE-merged; barriers inside must be reached by all threads (caller
    ensures uniform control flow).

    `combine` selects the monoid: `addOp` (cumsum, identity 0) or `mulOp`
    (cumprod, identity 1); absent means `addOp`, which keeps every pre-existing
    cumsum spelling of this op unchanged. Both are commutative, so the
    Hillis-Steele step order is free. The emitter rejects any other operator.

    `reverse` runs the scan from the high end: chunks are visited from the last
    one down, and each chunk is read/written through a mirrored index so the
    forward template runs verbatim over the reversed sequence. That read is
    cross-thread (thread `t` reads the slot thread `tpb-1-t` filled), so the
    emitter barriers before it.
  }];

  let arguments = (ins Metal_MemRefType:$inbuf, Metal_MemRefType:$outbuf,
                       I64Attr:$block, I64Attr:$tpb,
                       OptionalAttr<UnitAttr>:$reverse,
                       OptionalAttr<BinaryExpOperator>:$combine);

  // `combine` prints symbolically and only when present, so every pre-existing
  // cumsum spelling of this op round-trips unchanged.
  let assemblyFormat =
      "$inbuf `,` $outbuf (`combine` $combine^)? attr-dict `:` type($inbuf) "
      "`,` type($outbuf)";
}

//===----------------------------------------------------------------------===//
// ThreadgroupSegmentedPrefixSumOp
//===----------------------------------------------------------------------===//

def ThreadgroupSegmentedPrefixSumOp
    : Metal_Op<"threadgroup_segmented_prefix_sum",
        [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "In-place segmented inclusive prefix-sum over threadgroup buffers";
  let description = [{
    Reads and writes two threadgroup buffers in place: `values[0..block)` is an
    f32 buffer and `flags[0..block)` is an i1 buffer. The op computes the
    inclusive associative scan of pairs `(value, flag)` using
    `combine((v1, f1), (v2, f2)) = (f2 ? v2 : v1 + v2, f1 || f2)`.

    The MSL emitter tiles over `E = block/tpb` chunks. Each chunk runs a
    barrier-uniform Hillis-Steele pair scan across `tpb` threads, then combines
    the chunk-local prefix with a carry pair from all previous chunks. Barriers
    are emitted outside per-lane conditionals; callers must place this op only
    in uniform control flow.
  }];

  let arguments = (ins Metal_MemRefType:$values, Metal_MemRefType:$flags,
                       I64Attr:$block, I64Attr:$tpb);

  let assemblyFormat =
      "$values `,` $flags attr-dict `:` type($values) `,` type($flags)";
}

//===----------------------------------------------------------------------===//
// ThreadgroupAffinePrefixScanOp
//===----------------------------------------------------------------------===//

def ThreadgroupAffinePrefixScanOp
    : Metal_Op<"threadgroup_affine_prefix_scan",
        [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "In-place affine-composition prefix scan over threadgroup buffers";
  let description = [{
    Reads and writes two f32 threadgroup buffers in place.  Each pair `(a, x)`
    represents the affine transform `h -> a*h + x`; the op computes the
    inclusive associative scan

    `combine((a_l, x_l), (a_r, x_r)) =
        (a_l*a_r, a_r*x_l + x_r)`.

    The MSL emitter tiles over `E = block/tpb` chunks.  Each chunk runs a
    barrier-uniform Hillis-Steele pair scan across `tpb` threads, then composes
    the chunk-local prefixes with the carry transform from prior chunks.
    Barriers are emitted outside per-lane conditionals; callers must place this
    op only in uniform control flow.

    `reverse` scans from the high end, i.e. `out[i] = src[block-1] * ... * src[i]`
    in the combine above — the composition `T_src[i] o ... o T_src[block-1]`,
    which is the reverse recurrence `A_i = x_i + a_i * A_(i+1)`.  The combine is
    NOT commutative, so this is implemented the only order-safe way: each chunk
    is mirrored IN PLACE (thread `t` swaps with thread `tpb-1-t`, barriered on
    both sides), the forward template runs verbatim over it, and the chunk is
    mirrored back.  Chunks are visited from the last one down, so the carry
    composes in reversed order too.
  }];

  let arguments = (ins Metal_MemRefType:$a, Metal_MemRefType:$x,
                       I64Attr:$block, I64Attr:$tpb,
                       OptionalAttr<UnitAttr>:$reverse);

  let assemblyFormat =
      "$a `,` $x attr-dict `:` type($a) `,` type($x)";
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
// See the implementation notes.
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
                       Variadic<UI32>:$warp_index, UnitAttr:$transposed);
  let results = (outs Metal_SimdgroupMatrixType:$result);

  let assemblyFormat = "$memref `[` $origin_row `,` $origin_col `]` `,` $stride"
                       " (`warp` `[` $warp_index^ `]`)?"
                       " attr-dict `:` functional-type(operands, results)";

  let hasVerifier = 1;
}

def SimdgroupLoadDeviceStagedMaskedOp
    : Metal_Op<"simdgroup_load_device_staged_masked"> {
  let summary = "Masked staged load: out-of-bounds elements read as 0";

  let description = [{
    Masked variant of `simdgroup_load_device_staged`. During the cooperative
    device->threadgroup copy each lane loads `memref[(origin_row+si)*stride +
    origin_col+sj]` only when `(origin_row+si) < row_extent && (origin_col+sj) <
    col_extent`, writing 0 otherwise. Handles the `tl.load(..., mask=(offs_row <
    ROW) & (offs_k < K), other=0.0)` pattern (K-tail + row/col edges). The
    `transposed` attr swaps the staging destination index (W1). Single-warp.
  }];

  let arguments = (ins Metal_MemRefType:$memref, UI32:$origin_row,
                       UI32:$origin_col, UI32:$stride, UI32:$row_extent,
                       UI32:$col_extent, Variadic<UI32>:$warp_index,
                       UnitAttr:$transposed);
  let results = (outs Metal_SimdgroupMatrixType:$result);

  let assemblyFormat = "$memref `[` $origin_row `,` $origin_col `]` `,` $stride"
                       " `mask` `[` $row_extent `,` $col_extent `]`"
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

def SimdgroupStoreOp
    : Metal_Op<"simdgroup_store", [AttrSizedOperandSegments]> {
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
                       Variadic<UI32>:$partial_extents,
                       Variadic<UI32>:$warp_index);

  let assemblyFormat = "$matrix `,` $memref `[` $origin_row `,` $origin_col `]`"
                       " `,` $stride (`partial` `[` $partial_extents^ `]`)?"
                       " (`warp` `[` $warp_index^ `]`)?"
                       " attr-dict `:` type(operands)";

  let hasVerifier = 1;
}

def SimdgroupFusedStoreOp
    : Metal_Op<"simdgroup_fused_store", [AttrSizedOperandSegments]> {
  let summary = "Store base + scale*delta (fused matmul epilogue) to device memory";

  let description = [{
    W2c fused-LoRA epilogue. Stores `base + scale * delta` for two 8x8
    SIMD-group matrices `base` / `delta` and a runtime f32 `scale`. Both
    matrices are staged to per-threadgroup scratch, then a cooperative coop-loop
    writes `scratch_base[c] + scale * scratch_delta[c]` back to the destination.
    When `partial_extents = [m_extent, n_extent]` is present the coop-loop masks
    lanes to `gi < m_extent && gj < n_extent` (the masked-tail path).
  }];

  let arguments = (ins Metal_SimdgroupMatrixType:$base,
                       Metal_SimdgroupMatrixType:$delta, F32:$scale,
                       Metal_MemRefType:$memref, UI32:$origin_row,
                       UI32:$origin_col, UI32:$stride,
                       Variadic<UI32>:$partial_extents,
                       Variadic<UI32>:$warp_index);

  let assemblyFormat = "$base `,` $delta `,` $scale `,` $memref"
                       " `[` $origin_row `,` $origin_col `]` `,` $stride"
                       " (`partial` `[` $partial_extents^ `]`)?"
                       " (`warp` `[` $warp_index^ `]`)?"
                       " attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// ScalarDotOp
//===----------------------------------------------------------------------===//

def ScalarDotOp : Metal_Op<"scalar_dot", [AttrSizedOperandSegments]> {
  let summary = "Per-thread scalar GEMM (correctness fallback for tt.dot).";

  let description = [{
    Correctness-first lowering of a standalone `tt.dot` whose fused-epilogue
    shape the SIMD-group matmul matchers cannot consume (e.g. a `dot` feeding
    an `alpha*out + beta*C` epilogue rather than a bare `tt.store`). The result
    is a rank-2 `tensor<MxNxf32, #blocked>` value, or a rank-3 batched
    `tensor<BxMxNxf32, #blocked>` value. It flows into the ordinary per-thread
    tile-loop model:
    `ScalarDotLowering` scalarizes it to one output element per (thread,
    tile-loop iteration) and emits an `scf.for` reduction

      acc = c_init;
      for (local_k = 0; local_k < reduction_extent; ++local_k) {
        k = reduction_start + local_k;
        acc += a_buf[a_base_offset + a_batch_offset +
                     gRow*stride_a + k*a_inner_stride]
             * b_buf[b_base_offset + b_batch_offset +
                     k*stride_b + gCol*b_inner_stride];
      }

    where `(localBatch, linRow, linCol)` are recovered from the blocked tile
    geometry and `(gRow, gCol) = (row_origin + linRow, col_origin + linCol)`.
    When `output_indices` is present, its scalarized row contribution, column
    contribution, and row stride instead recover the exact rank-2 output
    coordinates owned by the current store lane. Rank-3 batch tails prepend
    the batch mask coordinate to the same triple. This keeps scalar-dot
    evaluation and its tail guard aligned with a folded result layout
    conversion whose blocked distribution differs from the function-wide tile
    loop.
    `a_batch_offset`/`b_batch_offset` are scalarized shaped pointer
    contributions for the current batch lane. The geometry is pinned to the
    function-wide tile loop so it remains aligned with the store when a folded
    result layout conversion uses a different blocked encoding. The reduction
    trip count is `reduction_extent` (normally equal to logical K) and starts
    at `reduction_start` (normally zero).  A
    `transpose_a` unit attribute changes A's physical index to
    `k*stride_a + gRow*a_inner_stride`; a `transpose_b` unit attribute likewise
    changes B's to `gCol*stride_b + k*b_inner_stride`, which is what
    `tl.dot(a, tl.trans(b))` means when b is stored `[N, K]` (an `nn.Linear`
    weight, among others). Without it the
    matcher's peel through `tt.trans` silently dropped the transpose and indexed
    B as if it were `[K, N]`; an optional `weight_buf` multiplies each product by
    `weight_buf[k]`.  Together these two narrow extensions represent the
    weighted Gram matrix `X.T @ diag(W) @ X` used by logistic-regression
    Hessians without pretending its dot-operand layout conversions are generic
    scalar identities. When `partial_extents = [m_extent, n_extent]` is present,
    the reduction is guarded by the matrix bounds. Rank-3 multi-batch tiles use
    `[batch_extent, m_extent, n_extent]` and additionally guard
    `batch_origin + localBatch < batch_extent`, so no tail lane issues an
    out-of-bounds device read.

    `a_base_offset`/`b_base_offset` preserve scalar `tt.addptr` offsets applied
    before the tiled pointer chain, such as a batched matrix's per-program base.
    The optional `scale_bufs`/`scale_params` tail carries the first exact
    software `tt.dot_scaled` slice.  It contains two buffers (A then B, with an
    unused matrix buffer as the placeholder for an unscaled side) and seven
    ui32 parameters: A/B scale base offsets, A/B scale row strides, A/B scale
    batch strides, and the shared 16- or 32-element scale factor. The batch
    strides are exact operands, so zero retains batch-shared scales while
    static padding and runtime strides remain observable. Unit attributes
    identify which sides are scaled and whether scale-NaN masking may be
    skipped. The lowering decodes each E8M0 byte into the operand compute type;
    optional markers identify E2M1 (packed along K or the outer matrix
    dimension) or byte-backed E4M3FN/E5M2 payloads. E2M1 is always shared by
    both operands; FP8 markers identify each operand independently so mixed
    E4M3FN/E5M2 pairs retain their distinct decoders. Conflicting or one-sided
    packed-payload markers are rejected. Every finite payload is widened
    exactly to bf16 before scaling; NaN class is preserved without promising
    payload bits. The scaled operand is rounded at that compute type and only
    then extended for f32 accumulation. Exact
    cross-backend numerical parity excludes denormal products or accumulation
    results on targets whose native floating-point arithmetic flushes them.
    NOTE: this is O(M*N*K) scalar work with no matrix-unit reuse — a correctness
    path, not a performance path. `a_buf`/`b_buf` are the device kernel-arg
    buffers; `c_init` is the dot accumulator init (typically `dense<0.0>`).
  }];

  let arguments = (ins Metal_MemRefType:$a_buf, Metal_MemRefType:$b_buf,
                       UI32:$a_base_offset, UI32:$b_base_offset,
                       AnyType:$a_batch_offset, AnyType:$b_batch_offset,
                       UI32:$batch_origin, UI32:$row_origin, UI32:$col_origin,
                       UI32:$stride_a, UI32:$stride_b,
                       UI32:$a_inner_stride, UI32:$b_inner_stride,
                       UI32:$reduction_start, UI32:$reduction_extent,
                       AnyType:$c_init,
                       Variadic<AnyType>:$output_indices,
                       Optional<Metal_MemRefType>:$weight_buf,
                       Variadic<UI32>:$partial_extents,
                       Variadic<Metal_MemRefType>:$scale_bufs,
                       Variadic<UI32>:$scale_params);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$a_buf `,` $b_buf `offset` `[` $a_base_offset `,`"
                       "$b_base_offset `]` `batch_offset` `[` $a_batch_offset"
                       " `,` $b_batch_offset `]` `origin` `[` $batch_origin"
                       " `,` $row_origin `,` $col_origin `]`"
                       " `,` $stride_a `,` $stride_b `,` $a_inner_stride"
                       " `,` $b_inner_stride `,` $reduction_start"
                       " `,` $reduction_extent"
                       " `,` $c_init"
                       " (`output` `[` $output_indices^ `]`)?"
                       " (`weight` $weight_buf^)?"
                       " (`partial` `[` $partial_extents^ `]`)?"
                       " (`scales` `[` $scale_bufs^ `]` `params` `[`"
                       " $scale_params `]`)?"
                       " attr-dict `:` functional-type(operands, results)";
}

#endif // METAL_OPS
