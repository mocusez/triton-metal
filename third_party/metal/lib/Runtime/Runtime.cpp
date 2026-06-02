// Runtime.cpp - C++ Metal runtime for the Triton Metal backend.
//
// Implements the pybind callables that drive `@triton.jit` -> MSL ->
// `.metallib` -> MTLDevice -> dispatch -> host result on Apple GPUs. See
// `.omc/specs/deep-interview-metal-gpu-launch.md`.
//
// Metal objects are accessed through the header-only metal-cpp bindings
// (third_party/metal/include/metal-cpp). metal-cpp is NOT ARC: objects
// returned by `new*`/`Create*` carry a +1 retain count that this file
// transfers into the opaque uintptr_t handles handed to Python; the
// matching `free*` callable calls `release()`. Foundation's process and
// filesystem facilities (NSTask/NSData/NSFileManager) are not part of
// metal-cpp, so the MSL->metallib compile step uses std::filesystem +
// posix_spawn instead.

#include "Runtime/Runtime.h"

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "metal-cpp/Metal/Metal.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dispatch/dispatch.h>

#include <crt_externs.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

// RAII drain for an NS::AutoreleasePool. metal-cpp objects created by
// methods that do not begin with alloc/new/copy/Create are autoreleased;
// this guard scopes their lifetime to the enclosing callable and drains
// even when a std::runtime_error unwinds out of it.
struct AutoreleasePoolGuard {
  NS::AutoreleasePool *pool;
  AutoreleasePoolGuard() : pool(NS::AutoreleasePool::alloc()->init()) {}
  ~AutoreleasePoolGuard() { pool->release(); }
  AutoreleasePoolGuard(const AutoreleasePoolGuard &) = delete;
  AutoreleasePoolGuard &operator=(const AutoreleasePoolGuard &) = delete;
};

// Pull a UTF-8 message out of an NS::Error, or a fallback when nil.
std::string errorMessage(NS::Error *err, const char *fallback) {
  if (err == nullptr || err->localizedDescription() == nullptr)
    return fallback;
  return err->localizedDescription()->utf8String();
}

// Singleton MTLDevice + MTLCommandQueue. Created lazily on first use.
// The Metal framework returns the same default device handle for the
// lifetime of the process; reusing one command queue is the typical
// pattern.
struct MetalRuntime {
  MTL::Device *device = nullptr;
  MTL::CommandQueue *queue = nullptr;

  static MetalRuntime &get() {
    static MetalRuntime r;
    return r;
  }

  void ensureInit() {
    if (device == nullptr) {
      device = MTL::CreateSystemDefaultDevice();
      if (device == nullptr)
        throw std::runtime_error(
            "Metal runtime: MTLCreateSystemDefaultDevice() returned nil "
            "(no Metal-capable GPU available?)");
      queue = device->newCommandQueue();
      if (queue == nullptr)
        throw std::runtime_error(
            "Metal runtime: [device newCommandQueue] returned nil");
    }
  }
};

// Monotonic thread-local GPU-time accumulator (nanoseconds). Updated inside
// launchKernelWithPipeline after each successful waitUntilCompleted; read
// (without reset) via readGpuTimeNsTotal. Consumed by Python-side
// _PerfCounterEvent.record() so that triton.testing.do_bench reports
// kernel-only GPU time, matching cuda.Event(enable_timing=True) semantics.
thread_local int64_t g_gpu_time_ns_accum = 0;

// Run `xcrun -sdk macosx metal <input.metal> -o <output.metallib>`
// synchronously and capture the resulting metallib bytes.
py::bytes compileMslToMetallib(const std::string &mslText) {
  namespace fs = std::filesystem;
  AutoreleasePoolGuard arp;

  // Write the MSL to a temp file. temp_directory_path() is per-user and
  // persists per session; uniqueness via NSProcessInfo.globallyUniqueString.
  std::string uniq =
      NS::ProcessInfo::processInfo()->globallyUniqueString()->utf8String();
  fs::path metalPath = fs::temp_directory_path() /
                       ("triton-metal-" + uniq + ".metal");
  fs::path libPath = fs::temp_directory_path() /
                     ("triton-metal-" + uniq + ".metallib");

  {
    std::ofstream ofs(metalPath, std::ios::binary | std::ios::trunc);
    if (!ofs)
      throw std::runtime_error(
          "Metal runtime: writing MSL temp file failed: " +
          metalPath.string());
    ofs.write(mslText.data(),
              static_cast<std::streamsize>(mslText.size()));
    ofs.close();
    if (!ofs)
      throw std::runtime_error(
          "Metal runtime: writing MSL temp file failed: " +
          metalPath.string());
  }

  // Spawn xcrun with its stderr redirected into a pipe we drain below.
  int errPipe[2];
  if (pipe(errPipe) != 0) {
    fs::remove(metalPath);
    throw std::runtime_error(
        std::string("Metal runtime: pipe() failed: ") + std::strerror(errno));
  }

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, errPipe[1], STDERR_FILENO);
  posix_spawn_file_actions_addclose(&actions, errPipe[0]);
  posix_spawn_file_actions_addclose(&actions, errPipe[1]);

  std::string metalArg = metalPath.string();
  std::string libArg = libPath.string();
  const char *argv[] = {"/usr/bin/xcrun", "-sdk",   "macosx",
                        "metal",          metalArg.c_str(), "-o",
                        libArg.c_str(),   nullptr};

  pid_t pid = 0;
  int rc = posix_spawn(&pid, "/usr/bin/xcrun", &actions, nullptr,
                       const_cast<char *const *>(argv), *_NSGetEnviron());
  posix_spawn_file_actions_destroy(&actions);
  close(errPipe[1]);
  if (rc != 0) {
    close(errPipe[0]);
    fs::remove(metalPath);
    throw std::runtime_error(
        std::string("Metal runtime: xcrun launch failed (Xcode Command "
                    "Line Tools installed?): ") +
        std::strerror(rc));
  }

  // Drain stderr to EOF (child is the only writer; bounded compiler output),
  // then reap.
  std::string errStr;
  char buf[4096];
  ssize_t n;
  while ((n = read(errPipe[0], buf, sizeof(buf))) > 0)
    errStr.append(buf, static_cast<std::size_t>(n));
  close(errPipe[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  int exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  if (exitCode != 0) {
    fs::remove(metalPath);
    fs::remove(libPath);
    throw std::runtime_error(
        std::string("Metal runtime: xcrun metal failed (exit ") +
        std::to_string(exitCode) + "):\n" + errStr);
  }

  std::ifstream ifs(libPath, std::ios::binary | std::ios::ate);
  bool ok = static_cast<bool>(ifs);
  std::string libData;
  if (ok) {
    std::streamsize sz = ifs.tellg();
    ifs.seekg(0);
    libData.resize(static_cast<std::size_t>(sz));
    if (sz > 0)
      ifs.read(&libData[0], sz);
    ok = static_cast<bool>(ifs);
  }
  fs::remove(metalPath);
  fs::remove(libPath);
  if (!ok)
    throw std::runtime_error(
        "Metal runtime: xcrun produced no .metallib output");
  return py::bytes(libData.data(), libData.size());
}

// Buffer handles are owned MTL::Buffer* reinterpreted as uintptr_t.
// allocBuffer transfers the newBuffer +1 retain into the handle; freeBuffer
// releases it.
uintptr_t allocBuffer(std::size_t nbytes) {
  AutoreleasePoolGuard arp;
  MetalRuntime::get().ensureInit();
  MTL::Buffer *buf = MetalRuntime::get().device->newBuffer(
      nbytes, MTL::ResourceStorageModeShared);
  if (buf == nullptr)
    throw std::runtime_error(
        "Metal runtime: newBufferWithLength returned nil");
  return reinterpret_cast<uintptr_t>(buf);
}

void freeBuffer(uintptr_t handle) {
  if (handle == 0)
    return;
  AutoreleasePoolGuard arp;
  reinterpret_cast<MTL::Buffer *>(handle)->release();
}

void copyH2D(uintptr_t handle, const py::bytes &src) {
  AutoreleasePoolGuard arp;
  MTL::Buffer *buf = reinterpret_cast<MTL::Buffer *>(handle);
  std::string s = src;
  if (s.size() > buf->length())
    throw std::runtime_error(
        "Metal runtime: copy_h2d source larger than buffer (" +
        std::to_string(s.size()) + " > " + std::to_string(buf->length()) +
        ")");
  memcpy(buf->contents(), s.data(), s.size());
}

py::bytes copyD2H(uintptr_t handle, std::size_t nbytes) {
  AutoreleasePoolGuard arp;
  MTL::Buffer *buf = reinterpret_cast<MTL::Buffer *>(handle);
  if (nbytes > buf->length())
    throw std::runtime_error(
        "Metal runtime: copy_d2h size larger than buffer (" +
        std::to_string(nbytes) + " > " + std::to_string(buf->length()) +
        ")");
  return py::bytes(static_cast<const char *>(buf->contents()), nbytes);
}

// Load a metallib + resolve a kernel function. Returns (lib_handle,
// fn_handle, pipelineState_handle, maxThreadsPerThreadgroup). The
// pipelineState is also pre-built so launch is just dispatch. The three
// returned objects keep their +1 retain and are released by the matching
// free* callables.
std::tuple<uintptr_t, uintptr_t, uintptr_t, std::size_t>
loadMetallib(const py::bytes &metallib, const std::string &kernelName) {
  AutoreleasePoolGuard arp;
  MetalRuntime::get().ensureInit();
  MTL::Device *device = MetalRuntime::get().device;

  std::string libBytes = metallib;
  dispatch_data_t dispatchData = dispatch_data_create(
      libBytes.data(), libBytes.size(), nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  NS::Error *err = nullptr;
  MTL::Library *library = device->newLibrary(dispatchData, &err);
  // dispatch_data_t is a C dispatch object here (no ObjC ARC); release it
  // once newLibrary has copied what it needs.
  dispatch_release(dispatchData);
  if (library == nullptr)
    throw std::runtime_error(
        "Metal runtime: newLibraryWithData failed: " +
        errorMessage(err, "unknown"));

  NS::String *nsName =
      NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);
  MTL::Function *function = library->newFunction(nsName);
  if (function == nullptr) {
    library->release();
    throw std::runtime_error(
        "Metal runtime: kernel function not found in metallib: " +
        kernelName);
  }

  NS::Error *psoErr = nullptr;
  MTL::ComputePipelineState *pso =
      device->newComputePipelineState(function, &psoErr);
  if (pso == nullptr) {
    function->release();
    library->release();
    throw std::runtime_error(
        "Metal runtime: newComputePipelineStateWithFunction failed: " +
        errorMessage(psoErr, "unknown"));
  }

  std::size_t maxThreads =
      static_cast<std::size_t>(pso->maxTotalThreadsPerThreadgroup());
  return std::make_tuple(reinterpret_cast<uintptr_t>(library),
                         reinterpret_cast<uintptr_t>(function),
                         reinterpret_cast<uintptr_t>(pso), maxThreads);
}

void freeLibrary(uintptr_t handle) {
  if (handle == 0)
    return;
  AutoreleasePoolGuard arp;
  reinterpret_cast<MTL::Library *>(handle)->release();
}

void freeFunction(uintptr_t handle) {
  if (handle == 0)
    return;
  AutoreleasePoolGuard arp;
  reinterpret_cast<MTL::Function *>(handle)->release();
}

void freePipeline(uintptr_t handle) {
  if (handle == 0)
    return;
  AutoreleasePoolGuard arp;
  reinterpret_cast<MTL::ComputePipelineState *>(handle)->release();
}

// Encode the buffer bindings + a single threadgroups dispatch onto a fresh
// command buffer from the shared queue, run it synchronously, and return the
// command buffer so the caller can inspect error()/GPU times. The returned
// buffer is autoreleased into the caller's AutoreleasePoolGuard scope.
MTL::CommandBuffer *dispatchSync(MTL::ComputePipelineState *pso,
                                 const std::vector<uintptr_t> &bufferHandles,
                                 std::tuple<int, int, int> grid,
                                 std::tuple<int, int, int> threadgroup) {
  MTL::CommandBuffer *cmdBuf = MetalRuntime::get().queue->commandBuffer();
  MTL::ComputeCommandEncoder *enc = cmdBuf->computeCommandEncoder();
  enc->setComputePipelineState(pso);
  for (std::size_t i = 0; i < bufferHandles.size(); ++i) {
    MTL::Buffer *buf = reinterpret_cast<MTL::Buffer *>(bufferHandles[i]);
    enc->setBuffer(buf, 0, static_cast<NS::UInteger>(i));
  }
  MTL::Size gridSize = MTL::Size::Make(std::get<0>(grid), std::get<1>(grid),
                                       std::get<2>(grid));
  MTL::Size tgSize =
      MTL::Size::Make(std::get<0>(threadgroup), std::get<1>(threadgroup),
                      std::get<2>(threadgroup));
  enc->dispatchThreadgroups(gridSize, tgSize);
  enc->endEncoding();
  cmdBuf->commit();
  cmdBuf->waitUntilCompleted();
  return cmdBuf;
}

// Dispatch using a pre-built MTL::ComputePipelineState (faster than
// launchKernel which reloads the metallib each call).
void launchKernelWithPipeline(uintptr_t psoHandle,
                              const std::vector<uintptr_t> &bufferHandles,
                              std::tuple<int, int, int> grid,
                              std::tuple<int, int, int> threadgroup) {
  AutoreleasePoolGuard arp;
  MetalRuntime::get().ensureInit();
  MTL::ComputePipelineState *pso =
      reinterpret_cast<MTL::ComputePipelineState *>(psoHandle);
  MTL::CommandBuffer *cmdBuf =
      dispatchSync(pso, bufferHandles, grid, threadgroup);
  NS::Error *cbErr = cmdBuf->error();
  if (cbErr == nullptr) {
    // GPUStartTime/GPUEndTime are CFTimeInterval (double, seconds since
    // boot), valid only after waitUntilCompleted on a non-errored CB.
    // Convert to ns and add to the thread-local monotonic counter read
    // by _PerfCounterEvent. Counter never resets; consumers subtract
    // two reads - matches cudaEventElapsedTime semantics.
    //
    // INVARIANTS:
    // (a) Synchronous launcher: GPU times are read AFTER waitUntilCompleted
    //     on the calling thread. If/when an async launcher lands, move this
    //     update into an addCompletedHandler completion block on the same
    //     MTLCommandBuffer (which runs on Metal's internal serial dispatch
    //     queue).
    // (b) Single thread per record: g_gpu_time_ns_accum is thread_local
    //     because do_bench calls _PerfCounterEvent.record() and
    //     launchKernelWithPipeline on the same Python thread. If a future
    //     harness records on thread A and launches on thread B, the
    //     recording thread reads 0 - migrate to std::atomic<int64_t> keyed
    //     by issuing-thread id (or a completion-block accumulator) before
    //     introducing cross-thread launches.
    CFTimeInterval start = cmdBuf->GPUStartTime();
    CFTimeInterval end = cmdBuf->GPUEndTime();
    g_gpu_time_ns_accum += static_cast<int64_t>((end - start) * 1e9);
  } else {
    throw std::runtime_error(
        "Metal runtime: command buffer error: " +
        errorMessage(cbErr, "unknown"));
  }
}

// Read the thread-local monotonic GPU-time counter (ns). Counter never
// resets; consumers subtract two reads for an interval.
int64_t readGpuTimeNsTotal() { return g_gpu_time_ns_accum; }

void launchKernel(const py::bytes &metallib, const std::string &kernelName,
                  const std::vector<uintptr_t> &bufferHandles,
                  std::tuple<int, int, int> grid,
                  std::tuple<int, int, int> threadgroup) {
  AutoreleasePoolGuard arp;
  MetalRuntime::get().ensureInit();
  MTL::Device *device = MetalRuntime::get().device;

  std::string libBytes = metallib;
  dispatch_data_t dispatchData = dispatch_data_create(
      libBytes.data(), libBytes.size(), nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  NS::Error *err = nullptr;
  // These locals are released on scope exit (including unwinding) via
  // NS::SharedPtr; this launcher reloads the metallib each call.
  NS::SharedPtr<MTL::Library> library =
      NS::TransferPtr(device->newLibrary(dispatchData, &err));
  dispatch_release(dispatchData);
  if (!library)
    throw std::runtime_error(
        "Metal runtime: newLibraryWithData failed: " +
        errorMessage(err, "unknown"));

  NS::String *nsName =
      NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);
  NS::SharedPtr<MTL::Function> function =
      NS::TransferPtr(library->newFunction(nsName));
  if (!function)
    throw std::runtime_error(
        "Metal runtime: kernel function not found in metallib: " +
        kernelName);

  NS::Error *psoErr = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> pso =
      NS::TransferPtr(device->newComputePipelineState(function.get(), &psoErr));
  if (!pso)
    throw std::runtime_error(
        "Metal runtime: newComputePipelineStateWithFunction failed: " +
        errorMessage(psoErr, "unknown"));

  MTL::CommandBuffer *cmdBuf =
      dispatchSync(pso.get(), bufferHandles, grid, threadgroup);
  if (cmdBuf->error() != nullptr)
    throw std::runtime_error(
        "Metal runtime: command buffer error: " +
        errorMessage(cmdBuf->error(), "unknown"));
}

// Walks MTL::GPUFamilyAppleN from highest known down to 1; returns the
// highest N supported by the system default device, or 0 if none.
//
// metal-cpp's supportsFamily() is selector-safe: it returns false when the
// running OS lacks the selector, and the GPUFamilyApple{N} enumerators are
// declared unconditionally by the bindings. That removes the need for the
// @available / __MAC_OS_X_VERSION_MAX_ALLOWED gating the Objective-C version
// required. When Apple11-capable hardware ships, add a row above Apple10.
static int getAppleGpuFamily() {
  AutoreleasePoolGuard arp;
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr)
    return 0;

  const struct {
    MTL::GPUFamily family;
    int n;
  } table[] = {
      {MTL::GPUFamilyApple10, 10}, {MTL::GPUFamilyApple9, 9},
      {MTL::GPUFamilyApple8, 8},   {MTL::GPUFamilyApple7, 7},
      {MTL::GPUFamilyApple6, 6},   {MTL::GPUFamilyApple5, 5},
      {MTL::GPUFamilyApple4, 4},   {MTL::GPUFamilyApple3, 3},
      {MTL::GPUFamilyApple2, 2},   {MTL::GPUFamilyApple1, 1},
  };
  int result = 0;
  for (const auto &row : table) {
    if (device->supportsFamily(row.family)) {
      result = row.n;
      break;
    }
  }
  device->release();
  return result;
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
  m.def("read_gpu_time_ns_total", &readGpuTimeNsTotal,
        "Read the thread-local monotonic GPU-time counter (ns), updated by "
        "launchKernelWithPipeline after each successful waitUntilCompleted "
        "from MTLCommandBuffer.GPUStartTime/GPUEndTime. Counter never resets; "
        "consumers subtract two reads for an interval.");
}

} // namespace metal
} // namespace triton
} // namespace mlir
