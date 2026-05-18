// Runtime.h - Metal runtime entry points for the Triton Metal backend.
//
// Darwin-only. The implementation lives in
// `third_party/metal/lib/Runtime/Runtime.mm` (Objective-C++ against the
// Metal framework). The pybind shim in `triton_metal.cc` registers
// these functions on the `triton._C.libtriton.metal` Python module.
//
// See `.omc/specs/deep-interview-metal-gpu-launch.md`.

#ifndef TRITON_METAL_RUNTIME_H
#define TRITON_METAL_RUNTIME_H

#include <pybind11/pybind11.h>

namespace mlir {
namespace triton {
namespace metal {

// Register the 6 runtime callables (compile_msl_to_metallib, alloc_buffer,
// free_buffer, copy_h2d, copy_d2h, launch_kernel) on the given Python
// module. Called from `init_triton_metal` on Darwin; the non-Darwin
// build provides a no-op definition.
void registerMetalRuntime(pybind11::module &m);

} // namespace metal
} // namespace triton
} // namespace mlir

#endif // TRITON_METAL_RUNTIME_H
