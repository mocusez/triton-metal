"""Metal driver: standard Triton launch protocol via torch tensors.

The driver fills the launcher_cls + utils.load_binary surface so that the
canonical Triton ergonomic `kernel[grid](*tensors)` works end-to-end on
Apple Metal. Per-launch arg marshalling bridges torch CPU tensors to
`MTLBuffer`s via `libmetal`. See
`.omc/specs/deep-interview-metal-standard-launch.md`.
"""

import functools
import os
import platform

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


@functools.lru_cache(maxsize=1)
def _use_mps_runtime() -> bool:
    """True when the launch path routes through ``torch.mps.compile_shader``
    for zero-copy dispatch on PyTorch MPS tensors (no host staging, no buffer
    alloc/free, ordered on the MPS stream). Opt out with
    ``TRITON_METAL_USE_MPS=0``. Mirror of ``compiler._use_mps_runtime`` — keep
    the two in lockstep so the compiler's binary_ext and the driver's launch
    path agree on which runtime is active."""
    if os.environ.get("TRITON_METAL_USE_MPS", "1") == "0":
        return False
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _install_torch_stream_shim():
    """Inject no-op `Stream` / `set_stream` on the active-device namespace(s).

    CUDA-style tutorials (e.g. `leet-triton/tutorials_python/02-fused-softmax.py`)
    call `getattr(torch, DEVICE.type).Stream()` and `set_stream(...)`. Metal's
    `get_active_torch_device()` returns `torch.device("mps", 0)` on the MPS path
    (or `torch.device("cpu")` on the legacy path), but neither `torch.mps` nor
    `torch.cpu` reliably exposes `Stream`/`set_stream` (this torch build has
    `torch.mps.synchronize`/`Event` but no `Stream`). The shim is a no-op
    because `triton.testing.do_bench` already drives synchronization via
    `driver.active.get_device_interface().synchronize()`.

    Both namespaces are shimmed so the tutorials work regardless of which launch
    path is active. Function-local `import torch` matches the convention in this
    module. Each attribute is `hasattr`-guarded so a future real
    `torch.{mps,cpu}.Stream` is never overwritten.
    See `.omc/plans/tutorial02-fused-softmax-fix-consensus.md` AC7/AC7a/R3.
    """
    import torch

    class _NoopStream:
        def synchronize(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    for ns in (getattr(torch, "cpu", None), getattr(torch, "mps", None)):
        if ns is None:
            continue
        if not hasattr(ns, "Stream"):
            ns.Stream = _NoopStream
        if not hasattr(ns, "set_stream"):
            ns.set_stream = lambda _stream: None


_install_torch_stream_shim()




# Floor for the MPS-path timer. Larger than the legacy 0.1 ms because each
# do_bench repeat allocates a REAL torch.mps.Event (not a counter read), so the
# floor also bounds n_repeat = rep / (estimate_ms) and keeps the event count
# modest. See _MPSFlooredEvent.
_MPS_ELAPSED_TIME_FLOOR_MS = 0.5


class _MPSFlooredEvent:
    """do_bench timing Event for the MPS path: a thin, robust wrapper over
    torch.mps.Event(enable_timing=True).

    Three adaptations vs. raw torch.mps.Event:
      * elapsed_time() is floored to _MPS_ELAPSED_TIME_FLOOR_MS. For the tiny,
        latency-bound kernels the Metal backend runs, GPU time is sub-µs; an
        unfloored estimate makes do_bench's n_repeat = rep/estimate_ms explode
        into thousands of events. Floored, n_repeat stays modest.
      * elapsed_time() swallows the torch MPS "End event N was not recorded
        after start event M" RuntimeError. torch's MPS event timing is flaky for
        sub-µs spans (the two timestamps collapse to one tick and torch reports
        them out of order) — observed nondeterministically even at small repeat
        counts. Such a span is below the floor anyway, so we report the floor
        instead of crashing the whole benchmark.
      * BOTH synchronize() and elapsed_time() force a DEVICE-WIDE
        torch.mps.synchronize() rather than the per-event
        torch.mps.Event.synchronize().

    That last one is a deadlock guard, not a nicety. On torch 2.10.0 a
    per-event `synchronize()` does NOT leave the event queryable, and the
    following `elapsed_time()` then blocks forever inside
    `at::mps::MPSEventPool::elapsedTime`'s condition_variable — and it blocks
    while HOLDING THE GIL, so every Python thread in the process freezes with
    it. pytest-timeout, faulthandler and signal.alarm are all powerless; only an
    external SIGKILL clears it. Measured: `record(); record(); ev.synchronize()
    x2; elapsed_time()` hangs, and so does the same sequence with real GPU work
    between the records — the discriminator is the sync flavour, not whether
    work was submitted. Swapping in a device-wide sync returns a normal value
    (a sub-floor 0.078 ms for the empty span). Note the deadlock is directional:
    torch *raises* the RuntimeError above for the reversed operand order but
    hangs for the forward one, which is why the `except` alone never saved us.

    A device sync in elapsed_time() is nearly free where it matters: do_bench
    already calls `di.synchronize()` before reading its timestamps, so the syncs
    inside the `times = [s.elapsed_time(e) ...]` loop are no-ops.
    """

    __slots__ = ("_ev",)

    def __init__(self, enable_timing: bool = False):
        import torch
        self._ev = torch.mps.Event(enable_timing=enable_timing)

    def record(self):
        self._ev.record()

    def synchronize(self):
        import torch
        # Device-wide, NOT self._ev.synchronize() — see the class docstring.
        torch.mps.synchronize()

    def elapsed_time(self, other) -> float:
        import torch
        # Unconditional: a caller that never synchronized (or synchronized only
        # per-event) would otherwise wedge the whole process, unkillably.
        torch.mps.synchronize()
        try:
            ms = self._ev.elapsed_time(other._ev)
        except RuntimeError:
            # Sub-µs span torch MPS couldn't order; it's below the floor anyway.
            return _MPS_ELAPSED_TIME_FLOOR_MS
        return max(ms, _MPS_ELAPSED_TIME_FLOOR_MS)


# libmetal exposes the in-process compile path only: `load_dialects` and
# `ttgir_to_msl` (TTGIR -> Metal dialect -> MSL text). The legacy native
# runtime (alloc/copy/launch, metallib compile) was removed — kernels launch
# via torch.mps.compile_shader on the Python side (MetalLauncher, MPS path).
try:
    from triton._C.libtriton import metal as _libmetal
except ImportError:  # pragma: no cover - libtriton always present at runtime
    _libmetal = None


# Apple GPU family tag for GPUTarget.arch. This is a cache-key / identity tag
# ONLY: codegen is decoupled from it (warp_size is fixed at 32 and the MSL
# emitter never branches on GPU family). The native Metal family probe was
# removed with the legacy runtime, so the tag is a stable constant — it only
# needs to be consistent for a given machine's compile cache. M3-class.
_APPLE_GPU_FAMILY = 9


def _get_apple_gpu_family() -> int:
    return _APPLE_GPU_FAMILY


# Decide, at __call__ time, whether each arg is a pointer (takes a torch
# tensor bound zero-copy) or a scalar (bound by value via compile_shader).
def _is_pointer_type(ty: str) -> bool:
    return ty.startswith("*")


class MetalUtils:
    """Driver-level utilities. `load_binary` is the load_binary hook that
    `CompiledKernel._init_handles` calls; it loads the metallib bytes into
    `MTLLibrary` + `MTLFunction` + `MTLComputePipelineState` handles.
    """

    def load_binary(self, name, kernel_bytes, shared_mem, device):
        # MPS-only: kernel_bytes is the MSL text (binary_ext == "metal").
        # torch.mps.compile_shader compiles it once per kernel and returns a
        # ShaderLibrary; getattr(lib, name) is the callable MetalKernel the
        # launcher invokes directly with MPS tensors (zero-copy). The library
        # object is the "module" handle (kept alive for the kernel's lifetime);
        # n_regs=32 clears the CUDA-style occupancy zero-divide and
        # n_max_threads=1024 clears the num_warps*warp_size threads check
        # (compiler.py). No metallib, no pso, no native runtime.
        if not _use_mps_runtime():
            raise RuntimeError(
                "Metal backend is MPS-only: the legacy native runtime was "
                "removed. Launching requires an MPS-enabled PyTorch on Apple "
                "Silicon (and TRITON_METAL_USE_MPS != 0).")
        import torch
        src = (kernel_bytes.decode() if isinstance(kernel_bytes, (bytes, bytearray))
               else kernel_bytes)
        lib = torch.mps.compile_shader(src)
        kernel = getattr(lib, name)
        return lib, kernel, 32, 0, 1024

    def unload_module(self, module):
        # MPS hands back a torch ShaderLibrary; Python GC reclaims it, nothing
        # to free explicitly.
        return

    def get_device_properties(self, device):
        # Triton inspects max_shared_mem; metal's spec varies by GPU but
        # Apple GPUs typically expose 32KB threadgroup memory. We don't
        # use shared memory in the vector_add path; return a generous
        # value to dodge OutOfResources checks.
        return {
            "max_shared_mem": 32 * 1024,
            "multiprocessor_count": 1,
            "max_num_regs": 65536,
            "warpSize": 32,
            "max_threads_per_sm": 1024,
            "sm_clock_rate": 1000000,
            "mem_clock_rate": 1000000,
            "mem_bus_width": 128,
        }


class MetalLauncher:
    """Full-fidelity Triton launcher: `(gridX, gridY, gridZ, stream,
    function, kernel_metadata, launch_metadata, launch_enter_hook,
    launch_exit_hook, *args)`. Matches the contract that
    `CompiledKernel.__getitem__` invokes (compiler/compiler.py:510).
    """

    def __init__(self, src, metadata):
        # The signature dict maps arg-name -> Triton type code. We
        # extract two parallel lists: arg_types (positional, only the
        # non-constexpr args), and pointer flags.
        sig = src.signature
        fn_arg_names = src.fn.arg_names
        # constexpr args are not passed at launch.
        constexpr_names = set()
        for name, ty in sig.items():
            if ty == "constexpr":
                constexpr_names.add(name)
        # Build positional type list excluding constexprs AND specialized
        # args (Triton drops constants like `stride_ak=1` from the kernel
        # signature). The mask MUST be in fn_arg_names order so it can
        # filter the positional `args` at launch time — without it, the
        # later `zip(args, self._arg_types)` silently truncates and
        # mis-aligns runtime args with the MSL buffer slots (iter-8c root
        # cause for `[16,16,16] f32`: stride_bk was read from `v4[0]` but
        # `v4` was actually bound to stride_ak=1, causing `simdgroup_load`
        # for B to use stride 1 instead of the real stride).
        self._arg_mask = [
            (n in sig and n not in constexpr_names) for n in fn_arg_names
        ]
        self._arg_types = [
            sig[n] for n, keep in zip(fn_arg_names, self._arg_mask) if keep
        ]
        # num_warps × warp_size = threadgroup x-dim. Default to 4×32=128.
        # `threads_per_group`, when the compiler set it, overrides that: a
        # kernel that lowers to a single `metal.fused_attention` runs its whole
        # body on one warp, so honouring a source `num_warps=4` there would
        # launch 128 threads and have 96 of them exit immediately (~12%).
        nw = getattr(metadata, "num_warps", 4)
        ws = 32
        tpg = getattr(metadata, "threads_per_group", None)
        self._threadgroup = (tpg if tpg else nw * ws, 1, 1)

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *args):
        # Zero-copy MPS launch. `function` is the torch MetalKernel from
        # MetalUtils.load_binary (a compiled torch.mps.compile_shader entry).
        # Map Triton's grid (= #threadgroups) and threadgroup size onto
        # compile_shader's (threads = total grid threads, group_size = threads
        # per threadgroup): threadgroup_position_in_grid then enumerates
        # 0..gridX-1 == tl.program_id. MPS tensors bind directly (honoring
        # storage_offset); Python scalars bind via setBytes into the 1-element
        # `device T*` slots the emitter declares. No alloc/copy/free, ordered
        # on PyTorch's MPS stream. The MPS path uses only tensor methods, so no
        # `torch` module import is needed here.
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        gx, gy, gz = self._threadgroup
        threads = (int(gridX) * gx, int(gridY) * gy, int(gridZ) * gz)
        group_size = (gx, gy, gz)

        filtered_args = [a for a, keep in zip(args, self._arg_mask) if keep]
        call_args = []
        writeback = []  # (orig, device_tensor) for non-MPS / non-contiguous inputs
        for arg, ty in zip(filtered_args, self._arg_types):
            if _is_pointer_type(ty):
                t = arg
                dev_t = t if (getattr(t, "device", None) is not None
                              and t.device.type == "mps") else t.to("mps")
                if not dev_t.is_contiguous():
                    dev_t = dev_t.contiguous()
                call_args.append(dev_t)
                if dev_t is not t:
                    writeback.append((t, dev_t))
            else:
                call_args.append(arg)

        function(*call_args, threads=threads, group_size=group_size)

        # Pure zero-copy path (all inputs already MPS + contiguous) leaves
        # writeback empty. CPU / non-contiguous inputs are mirrored back in
        # place so the standard "kernel writes its output arg" contract holds.
        for orig, dev_t in writeback:
            orig.copy_(dev_t)

        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class MetalDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher

    @classmethod
    def is_active(cls):
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def map_python_to_cpp_type(self, ty: str) -> str:
        # Triton type code -> C++ type string. Used by some launch-path
        # signature checks; the vector_add path doesn't depend on a
        # specific mapping, so any consistent value works.
        return {
            "i1": "bool",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "half",
            "bf16": "__bf16",
            "fp32": "float",
            "fp64": "double",
        }.get(ty, "void*")

    def get_current_target(self):
        return GPUTarget(backend="metal", arch=_get_apple_gpu_family(), warp_size=32)

    def get_active_torch_device(self):
        # MPS path: tensors live on the MPS device so the launcher binds them
        # zero-copy. Without the MPS runtime, tensors live on CPU and the
        # legacy launcher stages them through MTLBuffers per launch.
        #
        # Return the CONCRETE indexed device ("mps:0", not "mps"): a tensor
        # created with device="mps" reports `.device == mps:0`, so an
        # index-less `torch.device("mps")` fails the canonical tutorial guard
        # `assert x.device == DEVICE` and yields `DEVICE.index is None` (which
        # callers pass to get_device_properties). This mirrors CUDA, whose
        # active device is indexed (cuda:0).
        import torch
        if _use_mps_runtime():
            return torch.device("mps", 0)
        return torch.device("cpu")

    def get_benchmarker(self):
        # Delegate to triton.testing.do_bench so all backends share the same
        # warmup/quantile/cache-clear loop. Lazy import to avoid a circular
        # import during driver construction (triton.testing imports torch
        # and re-enters the driver module on its first call).
        import triton.testing

        def _benchmarker(kernel_call, *, quantiles, **kwargs):
            return triton.testing.do_bench(
                kernel_call,
                quantiles=quantiles,
                **kwargs,
            )

        return _benchmarker

    def get_empty_cache_for_benchmark(self):
        # Apple Silicon uses unified memory with no programmer-visible L2
        # evict primitive analogous to the CUDA pattern. do_bench expects an
        # opaque token here; returning None pairs with the no-op clear_cache
        # below. Faking a flush by zero-filling a large MTLBuffer would only
        # thrash DRAM under UMA and bias the benchmark — see plan ADR.
        return None

    def clear_cache(self, cache):
        # See get_empty_cache_for_benchmark — no-op under UMA. Intentionally
        # ignores `cache` (always None).
        return None

    # ---------------------------------------------------------------------
    # Hooks called by JIT runtime / compile path. The launch path goes
    # through MetalLauncher; these return placeholders sufficient for the
    # vector_add path.
    # ---------------------------------------------------------------------
    def get_current_device(self):
        return 0

    def get_current_stream(self, device):
        return 0

    def get_device_capability(self, device):
        return (8, 0)

    def set_current_device(self, device):
        pass

    def get_device_interface(self):
        # MPS-only: do_bench's timing and sync ride on torch.mps. torch.mps.Event
        # supports record()/elapsed_time()/synchronize() (matching the cuda.Event
        # contract do_bench expects); _MPSFlooredEvent wraps it for the backend's
        # tiny latency-bound kernels (see its docstring).
        import torch

        class _MPSDeviceInterface:
            Event = _MPSFlooredEvent

            @staticmethod
            def empty_cache():
                torch.mps.empty_cache()

            @staticmethod
            def synchronize():
                torch.mps.synchronize()

            @staticmethod
            def get_device_properties(device):
                return MetalUtils().get_device_properties(device)

        return _MPSDeviceInterface()
