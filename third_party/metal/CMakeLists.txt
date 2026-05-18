# Triton's Metal backend: in-process Metal dialect + MSL emitter, ported
# from the standalone `reference/metal-dialect/` repo into Triton against
# Triton's vendored LLVM 23. See:
#   - .omc/specs/deep-interview-metal-inprocess-pivot.md
#   - .omc/plans/triton-metal-backend-v1.md
# for the design.
#
# The pybind shim `triton_metal.cc` satisfies main.cc's `init_triton_metal`
# link contract; the heavy MLIR lifting lives in lib/Dialect/Metal/IR
# (the dialect), lib/Dialect/Metal/Transforms (conversion passes:
# Linalg→Metal, Funcs→Kernels, Metal→LLVM), and lib/Target/Metal
# (MLIR→MSL emitter).

# Expose the source-tree headers (for hand-written .h) AND the build-tree
# headers (for TableGen-generated .inc files) under "Dialect/Metal/..."
# and "Target/Metal/..." include roots.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(cli)

if(TRITON_BUILD_PYTHON_MODULE)
  set(_TRITON_METAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/triton_metal.cc)
  # On Darwin, link the Metal-runtime layer (Objective-C++ wrapping
  # MTLDevice / MTLBuffer / dispatch). Non-Darwin builds omit it; the
  # Python-side gate is `hasattr(libmetal, "launch_kernel")`.
  # See `.omc/specs/deep-interview-metal-gpu-launch.md`.
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    list(APPEND _TRITON_METAL_SOURCES
         ${CMAKE_CURRENT_SOURCE_DIR}/lib/Runtime/Runtime.mm)
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/Runtime/Runtime.mm
      PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  endif()
  add_triton_plugin(TritonMetal ${_TRITON_METAL_SOURCES})
  target_link_libraries(TritonMetal PRIVATE
    Python3::Module
    pybind11::headers
    MLIRTritonMetal
    MLIRTritonMetalTarget
    MLIRTritonMetalTransforms
    MLIRTritonGPUToMetal
  )
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(TritonMetal PRIVATE
      "-framework Metal"
      "-framework Foundation"
    )
  endif()
endif()
