get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_dialect_library(MLIRTritonMetalTransforms
  ConvertFuncsToMetalKernels.cpp
  ConvertLinalgToMetal.cpp
  ConvertMetalToLLVM.cpp
  LinalgToMetal.cpp
  MetalToLLVM.cpp

  DEPENDS
  MLIRMetalConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTritonMetal
  ${conversion_libs}
)

target_compile_options(obj.MLIRTritonMetalTransforms PRIVATE
  -Wno-error=deprecated-declarations
  # LLVM 23 raised -Wdangling on initializer-list temporaries (e.g.
  # MetalToLLVM.cpp:186/196/231 newOperands ArrayRef construction).
  # The legacy code is on the unused-pass path; warning kept visible but
  # not error-promoted.
  -Wno-error=dangling)
