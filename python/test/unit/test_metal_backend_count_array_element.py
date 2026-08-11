"""Counting occurrences of a value on the Metal backend.

Runs the verbatim `leet-triton/medium-count_array_element.py` kernel. It hit two
independent walls, one in the conversion and one in the emitter:

1. `tl.assume(pid >= 0)` lowers to `llvm.intr.assume` (python/src/ir.cc
   `create_assume`), a result-less optimizer hint the conversion had no pattern
   for — so *any* kernel using `tl.assume` failed to legalize. `eraseAssumeHints`
   drops the hint plus the predicate cone it kept alive; dropping only the hint
   would leave a dead-but-legal `arith.cmpi` that reaches the MSL emitter and
   gets printed as a bare statement.

2. Scalar `tl.atomic_add(ptr, i32)` is `tt.atomic_rmw add`, NOT `fadd`:
   `tl.sum(x == K)` reduces a comparison, so the payload is i32.
   `AtomicRmwLowering` accepted FADD/f32 only. The payload now enters
   `metal.atomic_rmw` as the memref's ui32 storage type (signless i32 is not in
   `Metal_Type`), and `ModuleTranslation` picks the atomic pointer type from the
   memref element — `atomic_uint` rather than `atomic_float`. That last part is
   load-bearing and silent if wrong: casting a `device uint32_t*` to
   `atomic_float*` compiles and corrupts the count instead of failing, which is
   why `test_atomic_add_scalar_i32_emits_integer_atomic` pins the MSL text.

Everything else the kernel needs already worked: the masked `other=0` load, the
`tl.sum` of an i1 comparison, `BLOCK_SIZE > threads_per_block` tile-loop
reduction, and the scalar `if sum > 0:` guard around the atomic.

Note on `K == 0`: the kernel's own semantics over-count there, on any backend —
masked-off tail lanes read `other=0.` and compare equal. That is the kernel's
bug, not the backend's, so the cases below use K != 0 and
`test_k_zero_counts_masked_padding` pins the (matching) behaviour explicitly.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


# Verbatim copy of leet-triton/medium-count_array_element.py.
@triton.jit
def _count_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE % 4 == 0)
    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    data = tl.load(input_ptr + offs, mask=mask, other=0.)
    sum = tl.sum(data == K)
    if (sum > 0):
        tl.atomic_add(output_ptr, sum)


def _solve(inp, out, N, K, BLOCK_SIZE=1024, **kwargs):
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    _count_kernel[grid](inp, out, N, K, BLOCK_SIZE=BLOCK_SIZE, **kwargs)
    torch.mps.synchronize()


# Exact tile (1024), sub-tile (N < BLOCK), masked tails on both sides of a tile
# boundary, single element, and multi-program up to ~1k programs.
@pytest.mark.parametrize("N", [1, 7, 1023, 1024, 1025, 5000, 65536, 100000, 1000003])
def test_count_matches_torch(N):
    torch.manual_seed(N)
    K = 3
    inp = torch.randint(0, 8, (N, ), device="mps", dtype=torch.int32)
    out = torch.zeros(1, device="mps", dtype=torch.int32)
    _solve(inp, out, N, K)
    assert out.cpu().item() == (inp.cpu() == K).sum().item()


# tpb = num_warps * 32 sets both the reduce tree and the `localTid == 0` guard
# on the scalar atomic. A guard keyed to the wrong tpb multiplies the count.
@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
def test_count_across_num_warps(num_warps):
    torch.manual_seed(num_warps)
    N, K = 5000, 2
    inp = torch.randint(0, 8, (N, ), device="mps", dtype=torch.int32)
    out = torch.zeros(1, device="mps", dtype=torch.int32)
    _solve(inp, out, N, K, num_warps=num_warps)
    assert out.cpu().item() == (inp.cpu() == K).sum().item()


def test_count_all_hit():
    # Every element matches: the atomic fires in every program, so a
    # per-THREAD (rather than per-program) atomic would show up as an exact
    # tpb-fold overcount.
    N = 5000
    inp = torch.full((N, ), 3, device="mps", dtype=torch.int32)
    out = torch.zeros(1, device="mps", dtype=torch.int32)
    _solve(inp, out, N, 3)
    assert out.cpu().item() == N


def test_count_no_hit_skips_atomic():
    # `if sum > 0:` is false in every program — the atomic is never reached.
    N = 5000
    inp = torch.full((N, ), 3, device="mps", dtype=torch.int32)
    out = torch.zeros(1, device="mps", dtype=torch.int32)
    _solve(inp, out, N, 99)
    assert out.cpu().item() == 0


def test_count_accumulates_into_nonzero_output():
    # The atomic must ADD into the buffer, not overwrite it.
    N, K = 4096, 5
    torch.manual_seed(0)
    inp = torch.randint(0, 8, (N, ), device="mps", dtype=torch.int32)
    out = torch.full((1, ), 1000, device="mps", dtype=torch.int32)
    _solve(inp, out, N, K)
    assert out.cpu().item() == 1000 + (inp.cpu() == K).sum().item()


def test_k_zero_counts_masked_padding():
    # Documented kernel-level (not backend-level) behaviour: with K == 0 the
    # masked tail lanes read `other=0.` and compare equal, so the count is high
    # by exactly the padding width. Pinned so a future backend change that
    # "fixes" this is recognized as a semantic divergence from CUDA.
    N, BLOCK = 100000, 1024
    torch.manual_seed(0)
    inp = torch.randint(1, 8, (N, ), device="mps", dtype=torch.int32)
    out = torch.zeros(1, device="mps", dtype=torch.int32)
    _solve(inp, out, N, 0, BLOCK_SIZE=BLOCK)
    padding = triton.cdiv(N, BLOCK) * BLOCK - N
    assert out.cpu().item() == padding


@triton.jit
def _assume_only_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.assume(pid >= 0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, tl.load(in_ptr + offs, mask=mask, other=0.), mask=mask)


def test_assume_hint_compiles_and_is_inert():
    # `tl.assume` on its own — independent of the atomic — used to fail the
    # conversion. It must now compile and change nothing about the result.
    N, BLOCK = 300, 128
    torch.manual_seed(0)
    inp = torch.randn(N, device="mps")
    out = torch.zeros(N, device="mps")
    _assume_only_kernel[(triton.cdiv(N, BLOCK), )](inp, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), inp.cpu())


@triton.jit
def _assume_shared_pred_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cond = N > 0
    tl.assume(cond)  # hint...
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.)
    y = tl.where(cond, x * 2.0, x)  # ...and the SAME predicate feeds live code
    tl.store(out_ptr + offs, y, mask=mask)


def test_assume_predicate_shared_with_live_code():
    # The only way `eraseAssumeHints`' dead-cone walk could silently break a
    # working kernel is by over-erasing: the predicate here has a second,
    # live consumer, so `isOpTriviallyDead` must keep it.
    N, BLOCK = 300, 128
    torch.manual_seed(0)
    inp = torch.randn(N, device="mps")
    out = torch.zeros(N, device="mps")
    _assume_shared_pred_kernel[(triton.cdiv(N, BLOCK), )](inp, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), inp.cpu() * 2)


@triton.jit
def _atomic_add_i32_kernel(out_ptr, v):
    tl.atomic_add(out_ptr, v)


def test_atomic_add_scalar_i32_emits_integer_atomic():
    # The atomic pointer cast comes from the memref element type. Emitting
    # `atomic_float*` over a `device uint32_t*` buffer COMPILES and silently
    # corrupts the value, so assert on the emitted MSL rather than only on the
    # numbers.
    out = torch.zeros(1, device="mps", dtype=torch.int32)
    compiled = _atomic_add_i32_kernel.warmup(out, 7, grid=(1, ))
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert "atomic_fetch_add_explicit((device atomic_uint*)" in msl
    assert "atomic_float" not in msl


def test_atomic_add_scalar_i32_one_add_per_program():
    G = 64
    out = torch.zeros(1, device="mps", dtype=torch.int32)
    _atomic_add_i32_kernel[(G, )](out, 7)
    torch.mps.synchronize()
    assert out.cpu().item() == 7 * G


def test_atomic_add_scalar_f32_still_float_atomic():
    # Regression guard on the other side of the emitter's type dispatch.
    out = torch.zeros(1, device="mps", dtype=torch.float32)
    compiled = _atomic_add_i32_kernel.warmup(out, 1.5, grid=(1, ))
    msl = compiled.asm["metal"]
    if isinstance(msl, bytes):
        msl = msl.decode()
    assert "atomic_fetch_add_explicit((device atomic_float*)" in msl

    out.zero_()
    _atomic_add_i32_kernel[(16, )](out, 1.5)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tensor([24.0]))
