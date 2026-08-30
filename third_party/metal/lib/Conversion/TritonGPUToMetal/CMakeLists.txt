add_mlir_dialect_library(MLIRTritonGPUToMetal
  TritonGPUToMetal.cpp

  DEPENDS
  TritonGPUToMetalConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTritonMetal
  TritonIR
  TritonGPUIR
)

# Same -Wno-error=deprecated-declarations rationale as the other metal
# subtree targets — LLVM 22→23 builder.create<>(...) deprecations still
# present in some MLIR header inlines reached from here.
target_compile_options(obj.MLIRTritonGPUToMetal PRIVATE -Wno-error=deprecated-declarations)
