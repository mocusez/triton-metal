get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# Land the binaries alongside triton-opt under ${triton_obj_root}/bin/ so
# lit's existing `triton_tools_dir` substitution mechanism picks them up
# without further PATH plumbing.
set(_metal_bin_dir "${CMAKE_BINARY_DIR}/bin")

add_executable(triton-metal-opt triton-metal-opt/triton-metal-opt.cpp)
set_target_properties(triton-metal-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${_metal_bin_dir}")
target_compile_options(triton-metal-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-metal-opt PRIVATE
  MLIRTritonMetal
  MLIRTritonMetalTransforms
  MLIRTritonMetalTarget
  MLIRTritonGPUToMetal
  # Bring in ALL of Triton's MLIR libs via the global property used by
  # `bin/CMakeLists.txt`'s `triton-opt` — TritonGPU's verify path needs
  # TritonAnalysis + TritonGPUTransforms symbols not in TritonGPUIR alone.
  ${triton_libs}
  # MLIR core
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)
mlir_check_all_link_libraries(triton-metal-opt)

add_executable(triton-metal-translate triton-metal-translate/triton-metal-translate.cpp)
set_target_properties(triton-metal-translate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${_metal_bin_dir}")
target_compile_options(triton-metal-translate PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-metal-translate PRIVATE
  MLIRTritonMetal
  MLIRTritonMetalTarget
  # MLIR translate infrastructure
  MLIRIR
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTranslateLib
  MLIRRegisterAllDialects
)
mlir_check_all_link_libraries(triton-metal-translate)
