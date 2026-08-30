//===- MetalTypes.td - Metal dialect types ---------------*- tablegen -*-===//
//
// This source file is part of the Metal open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_TYPES
#define METAL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Dialect/Metal/IR/MetalDialect.td"

//===----------------------------------------------------------------------===//
// Metal type definitions
//===----------------------------------------------------------------------===//

class Metal_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Metal_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Metal_MemRefType : Metal_Type<"MetalMemRef", "memref"> {
    let summary = "Metal memref type";
    let parameters = (ins "Type":$type, "unsigned":$size);
    let hasCustomAssemblyFormat = 1;
}

// `metal.simdgroup_matrix<rows x cols x elem>` — a SIMD-group hardware
// matrix register tile. Apple Silicon supports 8x8 tiles natively but the
// type is parameterized for fp16/bf16/16x16 future-proofing. See
// the implementation notes.
def Metal_SimdgroupMatrixType
    : Metal_Type<"MetalSimdgroupMatrix", "simdgroup_matrix"> {
    let summary = "Metal SIMD-group hardware matrix register tile";
    let parameters = (ins "uint32_t":$rows, "uint32_t":$cols, "Type":$elem);
    let hasCustomAssemblyFormat = 1;
}

#endif // METAL_TYPES
