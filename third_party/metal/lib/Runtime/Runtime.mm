// Runtime.mm - Objective-C++ Metal runtime for the Triton Metal backend.
//
// Implements the 6 pybind callables that drive `@triton.jit` → MSL →
// `.metallib` → MTLDevice → dispatch → host result on Apple GPUs. See
// `.omc/specs/deep-interview-metal-gpu-launch.md`.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "Runtime/Runtime.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

// Singleton MTLDevice + MTLCommandQueue. Created lazily on first use.
// The Metal framework returns the same default device handle for the
// lifetime of the process; reusing one command queue is the typical
// pattern.
struct MetalRuntime {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;

  static MetalRuntime &get() {
    static MetalRuntime r;
    return r;
  }

  void ensureInit() {
    if (device == nil) {
      device = MTLCreateSystemDefaultDevice();
      if (device == nil)
        throw std::runtime_error(
            "Metal runtime: MTLCreateSystemDefaultDevice() returned nil "
            "(no Metal-capable GPU available?)");
      queue = [device newCommandQueue];
      if (queue == nil)
        throw std::runtime_error(
            "Metal runtime: [device newCommandQueue] returned nil");
    }
  }
};

// Run `xcrun -sdk macosx metal <input.metal> -o <output.metallib>`
// synchronously and capture the resulting metallib bytes.
py::bytes compileMslToMetallib(const std::string &mslText) {
  @autoreleasepool {
    // Write the MSL to a temp file. NSTemporaryDirectory() is per-user
    // and persists per session; uniqueness via NSProcessInfo.globallyUniqueString.
    NSString *uniq = [[NSProcessInfo processInfo] globallyUniqueString];
    NSString *metalPath = [NSString
        stringWithFormat:@"%@triton-metal-%@.metal", NSTemporaryDirectory(),
                         uniq];
    NSString *libPath = [NSString
        stringWithFormat:@"%@triton-metal-%@.metallib",
                         NSTemporaryDirectory(), uniq];

    NSString *msl = [NSString stringWithUTF8String:mslText.c_str()];
    NSError *err = nil;
    if (![msl writeToFile:metalPath
              atomically:YES
                encoding:NSUTF8StringEncoding
                   error:&err]) {
      throw std::runtime_error(
          std::string("Metal runtime: writing MSL temp file failed: ") +
          [[err localizedDescription] UTF8String]);
    }

    NSTask *task = [[NSTask alloc] init];
    [task setLaunchPath:@"/usr/bin/xcrun"];
    [task setArguments:@[
      @"-sdk", @"macosx", @"metal", metalPath, @"-o", libPath
    ]];
    NSPipe *stderrPipe = [NSPipe pipe];
    [task setStandardError:stderrPipe];
    NSError *launchErr = nil;
    if (![task launchAndReturnError:&launchErr]) {
      [[NSFileManager defaultManager] removeItemAtPath:metalPath error:nil];
      throw std::runtime_error(
          std::string("Metal runtime: xcrun launch failed (Xcode Command "
                      "Line Tools installed?): ") +
          (launchErr ? [[launchErr localizedDescription] UTF8String]
                     : "unknown"));
    }
    [task waitUntilExit];
    int status = [task terminationStatus];
    if (status != 0) {
      NSData *errData =
          [[stderrPipe fileHandleForReading] readDataToEndOfFile];
      NSString *errStr =
          [[NSString alloc] initWithData:errData encoding:NSUTF8StringEncoding];
      [[NSFileManager defaultManager] removeItemAtPath:metalPath error:nil];
      [[NSFileManager defaultManager] removeItemAtPath:libPath error:nil];
      throw std::runtime_error(
          std::string("Metal runtime: xcrun metal failed (exit ") +
          std::to_string(status) + "):\n" +
          (errStr ? [errStr UTF8String] : ""));
    }

    NSData *libData = [NSData dataWithContentsOfFile:libPath];
    [[NSFileManager defaultManager] removeItemAtPath:metalPath error:nil];
    [[NSFileManager defaultManager] removeItemAtPath:libPath error:nil];
    if (libData == nil)
      throw std::runtime_error(
          "Metal runtime: xcrun produced no .metallib output");
    return py::bytes(reinterpret_cast<const char *>([libData bytes]),
                     [libData length]);
  }
}

// Buffer handles are __bridge_retained id<MTLBuffer> reinterpreted as
// uintptr_t. allocBuffer creates +1 retain count; freeBuffer transfers
// ownership back and releases.
uintptr_t allocBuffer(std::size_t nbytes) {
  @autoreleasepool {
    MetalRuntime::get().ensureInit();
    id<MTLBuffer> buf =
        [MetalRuntime::get().device newBufferWithLength:nbytes
                                                options:MTLResourceStorageModeShared];
    if (buf == nil)
      throw std::runtime_error("Metal runtime: newBufferWithLength returned nil");
    return reinterpret_cast<uintptr_t>((__bridge_retained void *)buf);
  }
}

void freeBuffer(uintptr_t handle) {
  @autoreleasepool {
    if (handle == 0)
      return;
    id<MTLBuffer> buf =
        (__bridge_transfer id<MTLBuffer>)reinterpret_cast<void *>(handle);
    (void)buf; // ARC releases on scope exit.
  }
}

void copyH2D(uintptr_t handle, const py::bytes &src) {
  @autoreleasepool {
    id<MTLBuffer> buf =
        (__bridge id<MTLBuffer>)reinterpret_cast<void *>(handle);
    std::string s = src;
    if (s.size() > [buf length])
      throw std::runtime_error(
          "Metal runtime: copy_h2d source larger than buffer (" +
          std::to_string(s.size()) + " > " +
          std::to_string([buf length]) + ")");
    memcpy([buf contents], s.data(), s.size());
  }
}

py::bytes copyD2H(uintptr_t handle, std::size_t nbytes) {
  @autoreleasepool {
    id<MTLBuffer> buf =
        (__bridge id<MTLBuffer>)reinterpret_cast<void *>(handle);
    if (nbytes > [buf length])
      throw std::runtime_error(
          "Metal runtime: copy_d2h size larger than buffer (" +
          std::to_string(nbytes) + " > " + std::to_string([buf length]) +
          ")");
    return py::bytes(static_cast<const char *>([buf contents]), nbytes);
  }
}

// Load a metallib + resolve a kernel function. Returns (lib_handle,
// fn_handle, pipelineState_handle, maxThreadsPerThreadgroup). The
// pipelineState is also pre-built so launch is just dispatch.
std::tuple<uintptr_t, uintptr_t, uintptr_t, std::size_t>
loadMetallib(const py::bytes &metallib, const std::string &kernelName) {
  @autoreleasepool {
    MetalRuntime::get().ensureInit();
    id<MTLDevice> device = MetalRuntime::get().device;

    std::string libBytes = metallib;
    dispatch_data_t dispatchData = dispatch_data_create(
        libBytes.data(), libBytes.size(), nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *err = nil;
    id<MTLLibrary> library = [device newLibraryWithData:dispatchData
                                                   error:&err];
    if (library == nil)
      throw std::runtime_error(
          std::string("Metal runtime: newLibraryWithData failed: ") +
          (err ? [[err localizedDescription] UTF8String] : "unknown"));

    NSString *nsName = [NSString stringWithUTF8String:kernelName.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:nsName];
    if (function == nil)
      throw std::runtime_error(
          std::string("Metal runtime: kernel function not found in metallib: ") +
          kernelName);

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&err];
    if (pso == nil)
      throw std::runtime_error(
          std::string("Metal runtime: newComputePipelineStateWithFunction failed: ") +
          (err ? [[err localizedDescription] UTF8String] : "unknown"));

    std::size_t maxThreads = (std::size_t)[pso maxTotalThreadsPerThreadgroup];
    return std::make_tuple(
        reinterpret_cast<uintptr_t>((__bridge_retained void *)library),
        reinterpret_cast<uintptr_t>((__bridge_retained void *)function),
        reinterpret_cast<uintptr_t>((__bridge_retained void *)pso),
        maxThreads);
  }
}

void freeLibrary(uintptr_t handle) {
  @autoreleasepool {
    if (handle == 0) return;
    id<MTLLibrary> lib =
        (__bridge_transfer id<MTLLibrary>)reinterpret_cast<void *>(handle);
    (void)lib;
  }
}

void freeFunction(uintptr_t handle) {
  @autoreleasepool {
    if (handle == 0) return;
    id<MTLFunction> fn =
        (__bridge_transfer id<MTLFunction>)reinterpret_cast<void *>(handle);
    (void)fn;
  }
}

void freePipeline(uintptr_t handle) {
  @autoreleasepool {
    if (handle == 0) return;
    id<MTLComputePipelineState> pso =
        (__bridge_transfer id<MTLComputePipelineState>)reinterpret_cast<void *>(
            handle);
    (void)pso;
  }
}

// Dispatch using a pre-built MTLComputePipelineState (faster than
// launchKernel which reloads the metallib each call).
void launchKernelWithPipeline(uintptr_t psoHandle,
                               const std::vector<uintptr_t> &bufferHandles,
                               std::tuple<int, int, int> grid,
                               std::tuple<int, int, int> threadgroup) {
  @autoreleasepool {
    MetalRuntime::get().ensureInit();
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)reinterpret_cast<void *>(
            psoHandle);
    id<MTLCommandBuffer> cmdBuf = [MetalRuntime::get().queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (std::size_t i = 0; i < bufferHandles.size(); ++i) {
      id<MTLBuffer> buf =
          (__bridge id<MTLBuffer>)reinterpret_cast<void *>(bufferHandles[i]);
      [enc setBuffer:buf offset:0 atIndex:static_cast<NSUInteger>(i)];
    }
    MTLSize gridSize = MTLSizeMake(std::get<0>(grid), std::get<1>(grid),
                                    std::get<2>(grid));
    MTLSize tgSize = MTLSizeMake(std::get<0>(threadgroup),
                                  std::get<1>(threadgroup),
                                  std::get<2>(threadgroup));
    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    if ([cmdBuf error] != nil) {
      throw std::runtime_error(
          std::string("Metal runtime: command buffer error: ") +
          [[[cmdBuf error] localizedDescription] UTF8String]);
    }
  }
}

void launchKernel(const py::bytes &metallib, const std::string &kernelName,
                   const std::vector<uintptr_t> &bufferHandles,
                   std::tuple<int, int, int> grid,
                   std::tuple<int, int, int> threadgroup) {
  @autoreleasepool {
    MetalRuntime::get().ensureInit();
    id<MTLDevice> device = MetalRuntime::get().device;

    std::string libBytes = metallib;
    dispatch_data_t dispatchData = dispatch_data_create(
        libBytes.data(), libBytes.size(), nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *err = nil;
    id<MTLLibrary> library = [device newLibraryWithData:dispatchData
                                                   error:&err];
    // dispatch_data_t is ARC-managed; no explicit release needed.
    if (library == nil)
      throw std::runtime_error(
          std::string("Metal runtime: newLibraryWithData failed: ") +
          (err ? [[err localizedDescription] UTF8String] : "unknown"));

    NSString *nsName = [NSString stringWithUTF8String:kernelName.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:nsName];
    if (function == nil)
      throw std::runtime_error(
          std::string("Metal runtime: kernel function not found in metallib: ") +
          kernelName);

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&err];
    if (pso == nil)
      throw std::runtime_error(
          std::string("Metal runtime: newComputePipelineStateWithFunction failed: ") +
          (err ? [[err localizedDescription] UTF8String] : "unknown"));

    id<MTLCommandBuffer> cmdBuf = [MetalRuntime::get().queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (std::size_t i = 0; i < bufferHandles.size(); ++i) {
      id<MTLBuffer> buf =
          (__bridge id<MTLBuffer>)reinterpret_cast<void *>(bufferHandles[i]);
      [enc setBuffer:buf offset:0 atIndex:static_cast<NSUInteger>(i)];
    }
    MTLSize gridSize = MTLSizeMake(std::get<0>(grid), std::get<1>(grid),
                                    std::get<2>(grid));
    MTLSize tgSize = MTLSizeMake(std::get<0>(threadgroup),
                                  std::get<1>(threadgroup),
                                  std::get<2>(threadgroup));
    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    if ([cmdBuf error] != nil) {
      throw std::runtime_error(
          std::string("Metal runtime: command buffer error: ") +
          [[[cmdBuf error] localizedDescription] UTF8String]);
    }
  }
}

// Walks MTLGPUFamilyAppleN from highest known down to 1; returns the
// highest N supported by the system default device, or 0 if none.
//
// SDK guarding: MTLGPUFamilyApple{N} are NS_ENUM(NSInteger) values, not
// preprocessor macros, so `#ifdef MTLGPUFamilyApple10` is dead code. We
// gate the Apple9 row (introduced in macOS 14 / iOS 17) with
// __MAC_OS_X_VERSION_MAX_ALLOWED for build-SDK gating and @available for
// runtime OS gating. When Apple10-capable hardware ships and Xcode SDKs
// uniformly declare the enum, add a guarded row above the Apple9 block.
// Do NOT invent numeric casts like (MTLGPUFamily)1010 — rely on the
// named enum identifier or skip the row.
static int getAppleGpuFamily() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return 0;

  if (@available(macOS 14.0, iOS 17.0, *)) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
    if ([device supportsFamily:MTLGPUFamilyApple9]) return 9;
#endif
  }

  const struct { MTLGPUFamily family; int n; } table[] = {
    {MTLGPUFamilyApple8,  8},
    {MTLGPUFamilyApple7,  7},
    {MTLGPUFamilyApple6,  6},
    {MTLGPUFamilyApple5,  5},
    {MTLGPUFamilyApple4,  4},
    {MTLGPUFamilyApple3,  3},
    {MTLGPUFamilyApple2,  2},
    {MTLGPUFamilyApple1,  1},
  };
  for (size_t i = 0; i < sizeof(table)/sizeof(table[0]); ++i) {
    if ([device supportsFamily:table[i].family]) return table[i].n;
  }
  return 0;
}

} // namespace

namespace mlir {
namespace triton {
namespace metal {

void registerMetalRuntime(py::module &m) {
  m.def("compile_msl_to_metallib", &compileMslToMetallib);
  m.def("alloc_buffer", &allocBuffer);
  m.def("free_buffer", &freeBuffer);
  m.def("copy_h2d", &copyH2D);
  m.def("copy_d2h", &copyD2H);
  m.def("launch_kernel", &launchKernel);
  m.def("load_metallib", &loadMetallib);
  m.def("free_library", &freeLibrary);
  m.def("free_function", &freeFunction);
  m.def("free_pipeline", &freePipeline);
  m.def("launch_kernel_with_pipeline", &launchKernelWithPipeline);
  m.def("get_apple_gpu_family", &getAppleGpuFamily);
}

} // namespace metal
} // namespace triton
} // namespace mlir
