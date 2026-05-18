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

#endif // METAL_ENUMS
