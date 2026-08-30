"""Multi-warp threadgroup reduce nested inside an scf.for on the Metal backend.

Regression test for a loop-carried write-after-read hazard on the reduce
scratch buffer. Both butterfly reduce lowerings — `lowerRank1Reduce` and
`lowerContiguousMaskedReduce` — allocate one `threadgroup` buffer and, when the
reduce sits inside a loop, reuse that same static allocation on every trip. Each
emitted

    buf[tid] = partial;  barrier;  <butterfly>;  read buf[0]

with no barrier *before* the `buf[tid]` write, so iteration t+1's write raced
iteration t's broadcast read of `buf[0]`.

The hazard is invisible at `tpb <= 32`: one SIMD-group runs in lockstep and
cannot drift. It only appears once a threadgroup spans two or more SIMD-groups
(`num_warps >= 2`) and the loop runs long enough, with enough resident
threadgroups, for the groups to slip relative to each other. Every pre-existing
reduce test is either single-warp or has the reduce outside the loop, which is
why this went unnoticed.

Because the race is timing-dependent, this test detects a regression only
PROBABILISTICALLY — measured against the unfixed backend, the multi-warp arms
below trip on roughly 2-3 runs in 8, and the grid sizes here were chosen per arm
to maximise that rate. The deterministic guard is the companion lit fixture
`test/Dialect/Metal/convert-tritongpu-to-metal/rank1_reduce_leading_barrier.mlir`,
which pins the emitted barrier/store order exactly; keep both.

`BLOCK == 32` (num_warps=1) is the control arm: it passed before the fix and
must keep passing. The determinism check is the sharpest runtime signal — a race
shows up as bitwise-varying output across identical runs long before the
numerical error grows large enough to trip a tolerance.
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

TRIPS = 64
REPEATS = 8


@triton.jit
def _rank1_reduce_in_loop_kernel(x_ptr, out_ptr, T, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK)
    h = tl.zeros((BLOCK,), dtype=tl.float32)
    for t in range(T):
        v = tl.load(x_ptr + pid * T * BLOCK + t * BLOCK + tid)
        # Loop-carried per-thread state, then a threadgroup reduce that both
        # reads `buf[0]` this trip and rewrites `buf[tid]` on the next one.
        h = h * 0.5 + v
        tl.store(out_ptr + pid * T + t, tl.sum(h, axis=0))


@triton.jit
def _rank2_reduce_in_loop_kernel(x_ptr, out_ptr, T, BLOCK_M: tl.constexpr,
                                 BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    h = tl.zeros((BLOCK_M,), dtype=tl.float32)
    tile = BLOCK_M * BLOCK_N
    for t in range(T):
        addr = (pid * T * tile + t * tile
                + offs_m[:, None] * BLOCK_N + offs_n[None, :])
        x = tl.load(x_ptr + addr)
        h = h * 0.5 + tl.sum(x, axis=1)
        tl.store(out_ptr + pid * T * BLOCK_M + t * BLOCK_M + offs_m, h)


def _assert_deterministic(outs):
    varying = sum(not torch.equal(o, outs[0]) for o in outs)
    assert varying == 0, (
        f"{varying}/{len(outs)} runs differ bitwise on identical inputs — "
        "threadgroup reduce buffer race"
    )


# (BLOCK, num_warps, grid) — grid tuned per arm for race sensitivity.
@pytest.mark.parametrize("BLOCK, num_warps, grid",
                         [(32, 1, 256), (64, 2, 1024), (128, 4, 256)])
def test_rank1_reduce_in_loop_multiwarp_is_deterministic(BLOCK, num_warps, grid):
    torch.manual_seed(BLOCK)
    x = torch.randn(grid, TRIPS, BLOCK, dtype=torch.float32, device="mps")

    outs = []
    for _ in range(REPEATS):
        out = torch.zeros(grid, TRIPS, dtype=torch.float32, device="mps")
        _rank1_reduce_in_loop_kernel[(grid,)](x, out, TRIPS, BLOCK=BLOCK,
                                              num_warps=num_warps)
        torch.mps.synchronize()
        outs.append(out.cpu())
    _assert_deterministic(outs)

    # Reference: same decayed recurrence in float64 on the host.
    xr = x.cpu().double()
    h = torch.zeros(grid, BLOCK, dtype=torch.float64)
    ref = torch.zeros(grid, TRIPS, dtype=torch.float64)
    for t in range(TRIPS):
        h = h * 0.5 + xr[:, t, :]
        ref[:, t] = h.sum(-1)
    err = (outs[0].double() - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel_err={err:.3e}"


@pytest.mark.xfail(
    strict=True,
    raises=Exception,
    reason="Unsupported shape, rejected at compile time (was: silently wrong). "
    "A rank-2 reduce whose tile address varies with an enclosing loop must be "
    "re-filled per trip, but when that loop ALSO carries iter_args (the `h` "
    "accumulator here) the loop is rebuilt by its own conversion pattern and "
    "the hoisted rowBuf alloca stops dominating the read. The lowering now "
    "bails with a clear message instead of emitting a hoisted (wrong) reduce. "
    "The loop-varying address itself is fixed and covered by "
    "test_rank2_reduce_in_loop_no_iter_args.",
)
@pytest.mark.parametrize("BLOCK_M, num_warps", [(32, 1), (64, 2)])
def test_rank2_reduce_in_loop_multiwarp_is_deterministic(BLOCK_M, num_warps):
    BLOCK_N = 16
    grid = 256
    torch.manual_seed(BLOCK_M)
    x = torch.randn(grid, TRIPS, BLOCK_M, BLOCK_N, dtype=torch.float32,
                    device="mps")

    outs = []
    for _ in range(REPEATS):
        out = torch.zeros(grid, TRIPS * BLOCK_M, dtype=torch.float32,
                          device="mps")
        _rank2_reduce_in_loop_kernel[(grid,)](x, out, TRIPS, BLOCK_M=BLOCK_M,
                                              BLOCK_N=BLOCK_N,
                                              num_warps=num_warps)
        torch.mps.synchronize()
        outs.append(out.cpu())
    _assert_deterministic(outs)

    xr = x.cpu().double()
    h = torch.zeros(grid, BLOCK_M, dtype=torch.float64)
    ref = torch.zeros(grid, TRIPS, BLOCK_M, dtype=torch.float64)
    for t in range(TRIPS):
        h = h * 0.5 + xr[:, t, :, :].sum(-1)
        ref[:, t, :] = h
    got = outs[0].double().reshape(grid, TRIPS, BLOCK_M)
    err = (got - ref).abs().max() / max(ref.abs().max().item(), 1.0)
    assert err <= 2e-5, f"rel_err={err:.3e}"
