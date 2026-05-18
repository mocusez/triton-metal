add_mlir_dialect_library(MLIRTritonMetal
  MetalDialect.cpp
  MetalOps.cpp
  MetalTypes.cpp

  DEPENDS
  MLIRMetalOpsIncGen
  MLIRMetalOpsEnumsGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
)

# LLVM 22→23 drift: many `builder.create<OpTy>(...)` calls in the copied
# metal-dialect sources are deprecated under LLVM 23 (replaced by
# `OpTy::create(builder, ...)`). Triton builds with `-Werror`, turning the
# deprecation into a hard error. Suppress just the error-promotion (the
# warning stays visible) until the call sites are migrated. Tracked in
# `.omc/specs/deep-interview-metal-inprocess-pivot.md` open question.
target_compile_options(obj.MLIRTritonMetal PRIVATE -Wno-error=deprecated-declarations)
