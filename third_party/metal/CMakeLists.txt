# Triton's Metal backend: in-process Metal dialect + MSL emitter, ported
# from the standalone `reference/metal-dialect/` repo into Triton against
# Triton's vendored LLVM 23. See:
#   - the implementation notes
#   - the implementation notes
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
  # MPS-only backend: the pybind module exposes just the compile path
  # (load_dialects + ttgir_to_msl). Kernels launch via torch.mps.compile_shader
  # on the Python side (backends/metal/driver.py), so there is no native Metal
  # runtime layer to compile or link — the legacy lib/Runtime/Runtime.cpp (and
  # its `-framework Metal/Foundation` + metal-cpp dependency) was removed.
  add_triton_plugin(TritonMetal ${CMAKE_CURRENT_SOURCE_DIR}/triton_metal.cc)
  target_link_libraries(TritonMetal PRIVATE
    Python3::Module
    nanobind-static
    MLIRTritonMetal
    MLIRTritonMetalTarget
    MLIRTritonMetalTransforms
    MLIRTritonGPUToMetal
  )
endif()
