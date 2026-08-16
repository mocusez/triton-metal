//===--- MetalEnums.td - Metal dialect enums --------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_ENUMS
#define METAL_ENUMS

include "mlir/IR/EnumAttr.td"

def MetalReduceKind_Sum  : I64EnumAttrCase<"sum",  0>;
def MetalReduceKind_Max  : I64EnumAttrCase<"max",  1>;
def MetalReduceKind_Mean : I64EnumAttrCase<"mean", 2>;

def MetalReduceKindAttr : I64EnumAttr<
    "MetalReduceKind", "Reduction kind for metal.reduce",
    [MetalReduceKind_Sum, MetalReduceKind_Max, MetalReduceKind_Mean]> {
  let cppNamespace = "mlir::triton::metal";
}

def AtomicRmwKind_Add  : I32EnumAttrCase<"Add",  0>;
def AtomicRmwKind_Max  : I32EnumAttrCase<"Max",  1>;
def AtomicRmwKind_UMin : I32EnumAttrCase<"UMin", 2>;
def AtomicRmwKind_Min  : I32EnumAttrCase<"Min",  3>;
def AtomicRmwKind_UMax : I32EnumAttrCase<"UMax", 4>;
// Bitwise and exchange. MSL has these for 32-bit atomics only
// (`atomic_fetch_and_explicit` has no atomic_ulong overload), and exchange
// additionally works on atomic_float — all four verified by compiling them
// through `torch.mps.compile_shader`, not from the spec.
def AtomicRmwKind_And  : I32EnumAttrCase<"And",  5>;
def AtomicRmwKind_Or   : I32EnumAttrCase<"Or",   6>;
def AtomicRmwKind_Xor  : I32EnumAttrCase<"Xor",  7>;
def AtomicRmwKind_Xchg : I32EnumAttrCase<"Xchg", 8>;

def AtomicRmwKindAttr : I32EnumAttr<
    "AtomicRmwKind", "Atomic read-modify-write operation",
    [AtomicRmwKind_Add, AtomicRmwKind_Max, AtomicRmwKind_UMin,
     AtomicRmwKind_Min, AtomicRmwKind_UMax, AtomicRmwKind_And,
     AtomicRmwKind_Or, AtomicRmwKind_Xor, AtomicRmwKind_Xchg]> {
  let cppNamespace = "mlir::triton::metal";
}

def MaskSinkMode_Causal     : I32EnumAttrCase<"Causal",     0>;
def MaskSinkMode_BoolMask   : I32EnumAttrCase<"BoolMask",   1>;
def MaskSinkMode_FloatMask  : I32EnumAttrCase<"FloatMask",  2>;
def MaskSinkMode_Sinks      : I32EnumAttrCase<"Sinks",      3>;
def MaskSinkMode_NonCausal  : I32EnumAttrCase<"NonCausal",  4>;

def MaskSinkModeAttr : I32EnumAttr<
    "MaskSinkMode", "SDPA mask/sink discriminator (mutually exclusive)",
    [MaskSinkMode_Causal, MaskSinkMode_BoolMask, MaskSinkMode_FloatMask,
     MaskSinkMode_Sinks, MaskSinkMode_NonCausal]> {
  let cppNamespace = "mlir::triton::metal";
}

def MatmulKind_Scalar : I32EnumAttrCase<"Scalar", 0>;
def MatmulKind_Mma    : I32EnumAttrCase<"Mma",    1>;

def MatmulKindAttr : I32EnumAttr<
    "MatmulKind", "Matmul emitter discriminator (mutually exclusive)",
    [MatmulKind_Scalar, MatmulKind_Mma]> {
  let cppNamespace = "mlir::triton::metal";
}

// How `metal.fused_attention` turns transformed scores into output weights.
//
// `none`: the transformed score IS the weight. The P@V accumulator is the
//   result and there is no denominator — `medium-decaying_causal_attention.py`,
//   where the score cone already applies the causal decay `gamma^(row-key)` and
//   masks the future to exactly 0.
// `online_softmax`: the transformed score is a logit. The emitter keeps the
//   running (max, sum) state per query row, rescales the accumulator on every
//   key block, and divides by the running sum in the epilogue — the classic
//   flash-attention formulation.
def AttnNorm_None         : I32EnumAttrCase<"None",         0>;
def AttnNorm_OnlineSoftmax : I32EnumAttrCase<"OnlineSoftmax", 1>;

def AttnNormAttr : I32EnumAttr<
    "AttnNorm", "How fused_attention normalizes transformed scores",
    [AttnNorm_None, AttnNorm_OnlineSoftmax]> {
  let cppNamespace = "mlir::triton::metal";
}

#endif // METAL_ENUMS
