// RUN: triton-metal-opt --convert-funcs-to-metal-kernels --verify-diagnostics --split-input-file %s

// AC-W-N1.1: dynamic-shape arg is rejected.

// expected-error @+1 {{convert-funcs-to-metal-kernels: dynamic shapes are not supported}}
func.func @dyn(%a: memref<?xf32>) attributes {metal.kernel} {
  return
}

// -----

// AC-W-N1.2: unsupported element type is rejected.

// expected-error @+1 {{convert-funcs-to-metal-kernels: unsupported element type}}
func.func @bad_et(%a: memref<4xi32>) attributes {metal.kernel} {
  return
}

// -----

// AC-W-N1.3: non-memref arg is rejected.

// expected-error @+1 {{convert-funcs-to-metal-kernels: arguments must be memrefs}}
func.func @non_memref(%a: i64) attributes {metal.kernel} {
  return
}

// -----

// AC-W-N1.4: multi-block func body is rejected.

// expected-error @+1 {{convert-funcs-to-metal-kernels: multi-block func bodies are not supported}}
func.func @multi_block(%a: memref<4xf32>) attributes {metal.kernel} {
  cf.br ^bb1
^bb1:
  return
}
