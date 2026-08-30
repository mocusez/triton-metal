// RUN: triton-metal-translate --mlir-to-msl %s 2>&1 | FileCheck %s

// A @triton.jit function literally named `kernel` (e.g.
// leet-triton/medium-monte_carlo_integration.py) reaches the emitter with
// entry name `kernel`. `kernel` is a reserved MSL keyword, so emitting
// `kernel void kernel(...)` is a hard xcrun / torch.mps.compile_shader syntax
// error. sanitizeKernelName() mangles a reserved entry name to `triton_<name>`;
// the launcher's metadata["name"] (compiler.py make_msl re-greps this text) and
// its getattr(lib, name) (driver.py load_binary) both follow.

module {
  metal.module {
    metal.kernel kernel address_space_device [true] {
    ^bb0(%arg0: !metal.memref<? x f32>):
      metal.return
    }
    metal.module_end
  }
}

// CHECK-NOT: kernel void kernel(
// CHECK: kernel void triton_kernel(
