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
)

target_compile_options(obj.MLIRTritonMetalTarget PRIVATE -Wno-error=deprecated-declarations)
