"""Per-element (rank-1) masked `tl.atomic_add` on the Metal backend.

`tt.atomic_rmw fadd` in tensor form is the lock-free accumulation the
layer-norm backward uses in place of the tutorial's global spin lock (Apple
GPUs give no cross-threadgroup forward-progress guarantee, so a `while
atomic_cas(Lock,0,1): pass` spin lock can deadlock). Many programs atomically
add their partial row into a shared buffer; the atomic serializes the colliding
read-modify-writes.

`AtomicRmwLowering`'s tensor branch models it on the masked device store: the
typeconverter scalarizes value/index/mask to the per-thread cone, the E>1 tile
loop replicates the op in place, and the device atomic is guarded by the mask
cone so masked-off lanes never touch a potentially-OOB (and, zero-copy, live)
address. Covers E==1 / E>1, a real `cols<N` mask, and the exact grouped
`_dw[lock_id*N + cols]` accumulation shape of `_layer_norm_bwd_dx_fused`.
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


@triton.jit
def _scatter_add_rows(In, Out, N, BLOCK: tl.constexpr):
    # grid=(G,): every program atomically adds its row into the shared Out[cols].
    g = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    v = tl.load(In + g * N + cols, mask=mask, other=0.0)
    tl.atomic_add(Out + cols, v, mask=mask)


# E==1 (BLOCK==tpb at 32), E>1 (BLOCK>tpb at 256/1024), non-pow2 N (masked).
@pytest.mark.parametrize("G, N", [(8, 32), (5, 200), (16, 256), (3, 1000),
                                  (7, 1024)])
def test_atomic_add_scatter(G, N):
    torch.manual_seed(G * 100 + N)
    inp = torch.randn(G, N, device="mps")
    out = torch.zeros(N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _scatter_add_rows[(G,)](inp, out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    ref = inp.cpu().sum(0)
    # Float-add reorder across programs is non-bit-deterministic; close in fp32.
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)


@triton.jit
def _grouped_accumulate(In, DW, N, GROUP_SIZE_M, BLOCK: tl.constexpr):
    # Exact shape of the layer-norm backward dw accumulation:
    # DW is [GROUP_SIZE_M, N]; row `r` accumulates into bucket `r % GROUP`.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    lock_id = row % GROUP_SIZE_M
    v = tl.load(In + row * N + cols, mask=mask, other=0.0)
    tl.atomic_add(DW + lock_id * N + cols, v, mask=mask)


@pytest.mark.parametrize("M, N, GROUP", [(8, 128, 4), (16, 256, 4),
                                         (10, 200, 3), (12, 1024, 8)])
def test_atomic_add_grouped(M, N, GROUP):
    torch.manual_seed(M * 1000 + N + GROUP)
    inp = torch.randn(M, N, device="mps")
    dw = torch.zeros(GROUP, N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _grouped_accumulate[(M,)](inp, dw, N, GROUP, BLOCK=BLOCK)
    torch.mps.synchronize()
    # reference: rows folded by (row % GROUP)
    ref = torch.zeros(GROUP, N)
    ic = inp.cpu()
    for r in range(M):
        ref[r % GROUP] += ic[r]
    torch.testing.assert_close(dw.cpu(), ref, atol=1e-4, rtol=1e-4)


@triton.jit
def _atomic_add_ones(Out, N, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    tl.atomic_add(Out + cols, tl.full([BLOCK], 1.0, tl.float32), mask=mask)


@pytest.mark.parametrize("G, N", [(64, 128), (100, 300)])
def test_atomic_add_ones_exact(G, N):
    # Adding 1.0 G times is exact regardless of order -> bit-exact contention check.
    out = torch.zeros(N, device="mps")
    BLOCK = triton.next_power_of_2(N)
    _atomic_add_ones[(G,)](out, N, BLOCK=BLOCK)
    torch.mps.synchronize()
    ref = torch.full((N,), float(G))
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)
