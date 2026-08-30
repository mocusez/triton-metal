add_mlir_dialect_library(MLIRTritonMetalTarget
  ConvertToMetalShadingLanguage.cpp
  ModuleTranslation.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTritonMetal
  # AC4 v6: ModuleTranslation calls triton::gpu::lookupNumWarps when
  # lowering SimdgroupLoadDeviceStagedOp with a warp_index operand.
  TritonGPUIR
)

target_compile_options(obj.MLIRTritonMetalTarget PRIVATE -Wno-error=deprecated-declarations)
