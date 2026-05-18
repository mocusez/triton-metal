"""Metal driver: standard Triton launch protocol via torch tensors.

The driver fills the launcher_cls + utils.load_binary surface so that the
canonical Triton ergonomic `kernel[grid](*tensors)` works end-to-end on
Apple Metal. Per-launch arg marshalling bridges torch CPU tensors to
`MTLBuffer`s via `libmetal`. See
`.omc/specs/deep-interview-metal-standard-launch.md`.
"""

import functools
import platform
import struct

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


# libmetal exposes the Darwin runtime callables (load_metallib,
# launch_kernel_with_pipeline, alloc_buffer, copy_h2d/d2h, free_buffer,
# etc.). On non-Darwin builds, only the dialect / MSL callables are
# available; the launch path here will then fail when invoked.
try:
    from triton._C.libtriton import metal as _libmetal
except ImportError:  # pragma: no cover - libtriton always present at runtime
    _libmetal = None


# Apple GPU family sentinel used when libmetal isn't built in (non-Darwin
# CI) or the runtime probe returns 0. M3-class is a reasonable middle.
_APPLE_GPU_FAMILY_FALLBACK = 9


@functools.lru_cache(maxsize=None)
def _get_apple_gpu_family() -> int:
    """Highest MTLGPUFamilyAppleN supported by the system default device.

    Falls back to ``_APPLE_GPU_FAMILY_FALLBACK`` when libmetal is unbuilt
    or the native probe returns 0. Cached for the process lifetime; the
    Apple GPU family cannot change without a process restart.
    """
    if _libmetal is not None and hasattr(_libmetal, "get_apple_gpu_family"):
        try:
            family = int(_libmetal.get_apple_gpu_family())
        except Exception:
            family = 0
        if family > 0:
            return family
    return _APPLE_GPU_FAMILY_FALLBACK


def _has_runtime() -> bool:
    return _libmetal is not None and hasattr(_libmetal, "launch_kernel_with_pipeline")


# Per-position type metadata cached on the launcher. We need to decide,
# at __call__ time, whether each arg should be packed as a scalar (1-elem
# MTLBuffer) or whether it's a pointer arg that takes a torch tensor.
def _is_pointer_type(ty: str) -> bool:
    return ty.startswith("*")


# Map Triton scalar type codes to (struct format, nbytes). Used to pack
# scalar args into 1-element MTLBuffers (matching the metal.kernel
# memref-only arg-type discipline).
_SCALAR_FMT = {
    "i1": ("?", 1),
    "i8": ("b", 1),
    "u8": ("B", 1),
    "i16": ("h", 2),
    "u16": ("H", 2),
    "i32": ("i", 4),
    "u32": ("I", 4),
    "i64": ("q", 8),
    "u64": ("Q", 8),
    "fp16": ("e", 2),
    "bf16": ("e", 2),  # struct doesn't have bf16; reuse fp16 as transport
    "fp32": ("f", 4),
    "fp64": ("d", 8),
}


def _pack_scalar(ty: str, value) -> bytes:
    fmt, _ = _SCALAR_FMT.get(ty, (None, None))
    if fmt is None:
        raise TypeError(f"MetalLauncher: unsupported scalar type {ty!r}")
    return struct.pack(fmt, value)


class MetalUtils:
    """Driver-level utilities. `load_binary` is the load_binary hook that
    `CompiledKernel._init_handles` calls; it loads the metallib bytes into
    `MTLLibrary` + `MTLFunction` + `MTLComputePipelineState` handles.
    """

    def load_binary(self, name, kernel_bytes, shared_mem, device):
        if not _has_runtime():
            raise RuntimeError(
                "Metal runtime not available; standard launch requires the "
                "Darwin libmetal extension to be built into libtriton.")
        if isinstance(kernel_bytes, (bytearray,)):
            kernel_bytes = bytes(kernel_bytes)
        # load_metallib returns (lib_handle, fn_handle, pso_handle, max_threads)
        lib_handle, fn_handle, pso_handle, max_threads = _libmetal.load_metallib(
            kernel_bytes, name)
        # CompiledKernel stores (module, function, n_regs, n_spills, n_max_threads).
        # We use `module` as a tuple of all 3 handles so unload_module can free
        # all of them. The launcher receives `function` = pso_handle and uses
        # it directly to launch.
        module = (lib_handle, fn_handle, pso_handle)
        return module, pso_handle, 0, 0, int(max_threads)

    def unload_module(self, module):
        if module is None or not _has_runtime():
            return
        lib_handle, fn_handle, pso_handle = module
        # Order matters: pipeline first, then function, then library.
        _libmetal.free_pipeline(pso_handle)
        _libmetal.free_function(fn_handle)
        _libmetal.free_library(lib_handle)

    def get_device_properties(self, device):
        # Triton inspects max_shared_mem; metal's spec varies by GPU but
        # Apple GPUs typically expose 32KB threadgroup memory. We don't
        # use shared memory in the vector_add path; return a generous
        # value to dodge OutOfResources checks.
        return {
            "max_shared_mem": 32 * 1024,
            "multiprocessor_count": 1,
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
        # Build positional type list excluding constexprs.
        self._arg_types = [
            sig[n] for n in fn_arg_names
            if n in sig and n not in constexpr_names
        ]
        # num_warps × warp_size = threadgroup x-dim. Default to 4×32=128.
        nw = getattr(metadata, "num_warps", 4)
        ws = 32
        self._threadgroup = (nw * ws, 1, 1)

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *args):
        if not _has_runtime():
            raise RuntimeError(
                "Metal runtime not available; standard launch requires the "
                "Darwin libmetal extension to be built into libtriton.")
        # `function` is the pso_handle from MetalUtils.load_binary.
        pso_handle = function

        # Enter hook.
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        # Marshal *args into MTLBuffer handles. Pointer args (torch
        # tensors) get a fresh MTLBuffer + h2d copy; scalar args get a
        # 1-element MTLBuffer with packed value. Track allocations so we
        # can copy results back + free at the end.
        buffer_handles = []
        tensor_pairs = []  # (handle, tensor) for d2h copy-back
        try:
            for i, (arg, ty) in enumerate(zip(args, self._arg_types)):
                if _is_pointer_type(ty):
                    tensor = arg
                    if hasattr(tensor, "contiguous"):
                        tensor = tensor.contiguous()
                    # Stage through a CPU view for h2d serialization; .numpy()
                    # rejects non-CPU tensors (MPS, CUDA). The d2h copy_ on
                    # the original tensor handles the cross-device return.
                    host_tensor = tensor.cpu() if tensor.device.type != "cpu" else tensor
                    nbytes = host_tensor.numel() * host_tensor.element_size()
                    buf = _libmetal.alloc_buffer(nbytes)
                    _libmetal.copy_h2d(buf, bytes(host_tensor.numpy().tobytes()))
                    buffer_handles.append(buf)
                    tensor_pairs.append((buf, tensor, nbytes))
                else:
                    # Scalar: pack to bytes, alloc 1-elem MTLBuffer.
                    _, nbytes = _SCALAR_FMT[ty]
                    packed = _pack_scalar(ty, arg)
                    buf = _libmetal.alloc_buffer(nbytes)
                    _libmetal.copy_h2d(buf, packed)
                    buffer_handles.append(buf)

            _libmetal.launch_kernel_with_pipeline(
                pso_handle, buffer_handles,
                (int(gridX), int(gridY), int(gridZ)),
                self._threadgroup,
            )

            # Copy pointer args back into their tensors (over-copy, since
            # we can't tell input vs output without an explicit annotation).
            for buf, tensor, nbytes in tensor_pairs:
                raw = _libmetal.copy_d2h(buf, nbytes)
                import numpy as np
                arr = np.frombuffer(raw, dtype=_torch_dtype_to_np(tensor.dtype))
                arr = arr.reshape(tensor.shape)
                # In-place copy into the tensor's storage.
                tensor.copy_(_np_to_torch(arr))
        finally:
            for buf in buffer_handles:
                _libmetal.free_buffer(buf)

        # Exit hook.
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


def _torch_dtype_to_np(dtype):
    import numpy as np
    import torch
    return {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }[dtype]


def _np_to_torch(arr):
    import torch
    # frombuffer arrays are non-writable; .copy() yields a writable copy
    # so the resulting tensor is fully usable downstream.
    return torch.from_numpy(arr.copy())


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
        }.get(ty, f"void*")

    def get_current_target(self):
        return GPUTarget(backend="metal", arch=_get_apple_gpu_family(), warp_size=32)

    def get_active_torch_device(self):
        # Tensors live on CPU; the MetalLauncher copies them to MTLBuffer
        # per launch. MPS-backed tensors are out of scope for this slice.
        import torch
        return torch.device("cpu")

    def get_benchmarker(self):
        raise NotImplementedError(
            "MetalDriver does not yet provide a benchmarker (deferred).")

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
        class _DeviceInterface:
            @staticmethod
            def empty_cache():
                pass

            @staticmethod
            def synchronize():
                pass

            @staticmethod
            def get_device_properties(device):
                return MetalUtils().get_device_properties(device)

        return _DeviceInterface()
