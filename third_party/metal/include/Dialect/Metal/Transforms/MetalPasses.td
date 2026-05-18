//===--- MetalPasses.td - Metal passes ---------------------*- tablegen -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_PASSES_TD
#define METAL_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertMetalToLLVM : Pass<"convert-metal-to-llvm"> {
  let summary = "Convert Metal operations into the LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def ConvertFuncsToMetalKernels : Pass<"convert-funcs-to-metal-kernels", "::mlir::ModuleOp"> {
  let summary = "Wrap annotated func.funcs into metal.module/metal.kernel skeletons";
  let description = [{
    For each `func.func` carrying the attribute `metal.kernel`, this pass
    replaces it with a top-level `metal.module { metal.kernel @<name>
    address_space_device [true, ...] { ... } }`. The kernel block-arg types
    are converted from `memref<...>` to `!metal.memref<? x T>`. To keep the
    body verifiable while linalg ops are still present, the pass inserts
    `builtin.unrealized_conversion_cast` ops at the top of the kernel body
    that bridge the new MetalMemRef block args back to standard memrefs;
    `convert-linalg-to-metal` (run after this pass) folds these casts away
    when it rewrites the linalg ops.

    Funcs without the `metal.kernel` attribute are left untouched.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::triton::metal::MetalDialect"
  ];
}

def ConvertLinalgToMetal : Pass<"convert-linalg-to-metal", "::mlir::ModuleOp"> {
  let summary = "Lower a curated subset of linalg named ops into the metal dialect";
  let description = [{
    Lowers `linalg.softmax` (innermost-axis) and `linalg.reduce` (with
    arith.addf or arith.maximumf combiner, innermost-axis) onto their
    metal-dialect counterparts, and converts standard `memref<...>` operand
    types to `!metal.memref<? x T>`. Input must be bufferized (memref form)
    with fully static shapes and element types in {f32, f16, bf16}. Other
    linalg ops (matmul, matvec, reduce-mean, generic, etc.) are out of scope
    for this pass.
  }];
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::triton::metal::MetalDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // METAL_PASSES_TD
