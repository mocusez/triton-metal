# TableGen for the metal dialect, mirroring third_party/nvidia/include/Dialect/...
# Generated .inc files land under ${CMAKE_CURRENT_BINARY_DIR}/MetalOps{,Dialect,Types,Enums}.{h,cpp}.inc
# and are #include'd by lib/Dialect/Metal/IR/MetalDialect.cpp etc. via the
# "Dialect/Metal/IR/<file>.inc" path made findable by the parent include_directories().

add_mlir_dialect(MetalOps metal)
set(LLVM_TARGET_DEFINITIONS MetalOps.td)
mlir_tablegen(MetalOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(MetalOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMetalOpsEnumsGen)
