"""L1d2c Phase A — 8-cell axis-aligned MSL-shape bisection probe rig.

Per the implementation notes,
each cell varies exactly one of three axes:

  * A1 (store kind): masked vs unconditional
  * A2 (barrier-preceding-store): present vs absent (induced by cvt body)
  * A3 (extra scf.if beyond mask wrap): present vs absent

Shape: 8x8 nw=2 across all cells (threadgroup size 64 = 2 SIMD warps of 32).

Expected MSL shapes per cell (A3="present" = NESTED if beyond mask wrap):
  C0: `v[id] = v6;`                                 (unconditional, no cvt, no extra if)
  C1: `if(c){ v[id] = v6; }`                        (unconditional, no cvt, with python if)
  C2: `barrier; v[id] = v6;`                        (unconditional, cvt body, no extra if)
  C3: `barrier; if(c){ v[id] = v6; }`               (unconditional, cvt body, with python if)
  C4: `if(m){ v[id] = v6; }`                        (masked, no cvt, mask-wrap only)
  C5: `if(c){ if(m){ v[id] = v6; }}`                (masked, no cvt, nested if)
  C6: `barrier; if(m){ v[id] = v6; }`               (masked, cvt body, mask-wrap only) <- L1d2b anchor
  C7: `barrier; if(c){ if(m){ v[id] = v6; }}`       (masked, cvt body, nested if)

Honest divergences from the spec table (will be reconciled in the RCA):
  * The spec's A3="absent" rows for C4/C6 show shapes that contain `if(m)`
    because mask -> MaskedStoreLowering always wraps in scf.if. The "A3"
    axis is therefore re-interpreted as a NESTED scf.if beyond the
    mask-induced wrap.
  * The L1d2b masked transpose maps to C6 (single mask-wrap), not C7.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import triton
import triton.language as tl

libmetal = pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )

BLOCK_N = 8
NUM_WARPS = 2
N_ELEMS = BLOCK_N * BLOCK_N  # 64


# ---------------- C0: unconditional store, no cvt, no scf.if ----------------
@triton.jit
def k_C0_uncond_nobarrier_noif(in_ptr, out_ptr, BLOCK: tl.constexpr):
    """Flat 1D copy: out[i] = in[i]. No mask, no cvt, no extra if."""
    i = tl.arange(0, BLOCK * BLOCK)
    v = tl.load(in_ptr + i)
    tl.store(out_ptr + i, v)


# ---------------- C1: unconditional, no cvt, WITH scf.if --------------------
@triton.jit
def k_C1_uncond_nobarrier_if(in_ptr, out_ptr, BLOCK: tl.constexpr):
    """Flat copy guarded by a runtime gate to force scf.if. The gate uses
    `tl.program_id(0)` (uniformly 0 in a (1,) grid) so the gate is true at
    runtime but opaque to the compiler (no tt.load on a scalar)."""
    pid = tl.program_id(0)
    i = tl.arange(0, BLOCK * BLOCK)
    v = tl.load(in_ptr + i)
    if pid == 0:
        tl.store(out_ptr + i, v)


# ---------------- C2: unconditional store, WITH cvt body (barrier), no extra if ----------------
@triton.jit
def k_C2_uncond_barrier_noif(in_ptr, out_ptr, BLOCK: tl.constexpr):
    """Unmasked transpose: cvt body emits the barrier-preceded-store but
    no scf.if (unconditional)."""
    x = tl.arange(0, BLOCK)
    y = tl.arange(0, BLOCK)
    xi = x[None, :]
    yi = y[:, None]
    tile = tl.load(in_ptr + yi * BLOCK + xi)
    tl.store(out_ptr + xi * BLOCK + yi, tile)


# ---------------- C3: unconditional + cvt + extra scf.if --------------------
@triton.jit
def k_C3_uncond_barrier_if(in_ptr, out_ptr, BLOCK: tl.constexpr):
    """Unmasked transpose wrapped in a program_id(0)==0 gate (forces scf.if)."""
    pid = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    y = tl.arange(0, BLOCK)
    xi = x[None, :]
    yi = y[:, None]
    tile = tl.load(in_ptr + yi * BLOCK + xi)
    if pid == 0:
        tl.store(out_ptr + xi * BLOCK + yi, tile)


# ---------------- C4: masked store, no cvt, mask-wrap only ------------------
@triton.jit
def k_C4_masked_nobarrier_noif(in_ptr, out_ptr, NELEMS, BLOCK: tl.constexpr):
    """1D masked copy: mask induces a single scf.if(mask)."""
    i = tl.arange(0, BLOCK * BLOCK)
    m = i < NELEMS
    v = tl.load(in_ptr + i, mask=m, other=0.0)
    tl.store(out_ptr + i, v, mask=m)


# ---------------- C5: masked store, no cvt, WITH extra scf.if ---------------
@triton.jit
def k_C5_masked_nobarrier_if(in_ptr, out_ptr, NELEMS, BLOCK: tl.constexpr):
    """1D masked copy guarded by program_id(0)==0 (nested scf.if)."""
    pid = tl.program_id(0)
    i = tl.arange(0, BLOCK * BLOCK)
    m = i < NELEMS
    v = tl.load(in_ptr + i, mask=m, other=0.0)
    if pid == 0:
        tl.store(out_ptr + i, v, mask=m)


# ---------------- C6: masked + cvt body, mask-wrap only (L1d2b anchor) ------
@triton.jit
def k_C6_masked_barrier_noif(
    in_ptr, out_ptr, rows, cols, sir, sic, sor, soc, BLOCK: tl.constexpr,
):
    """Masked transpose — exactly the L1d2b xfail pattern."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x = pid_x * BLOCK + tl.arange(0, BLOCK)
    y = pid_y * BLOCK + tl.arange(0, BLOCK)
    xi = x[None, :]
    yi = y[:, None]
    mask = (xi < cols) & (yi < rows)
    tile = tl.load(in_ptr + yi * sir + xi * sic, mask=mask, other=0.0)
    tl.store(out_ptr + xi * sor + yi * soc, tile, mask=mask)


# ---------------- C7: masked + cvt body + extra scf.if ----------------------
@triton.jit
def k_C7_masked_barrier_if(
    in_ptr, out_ptr, rows, cols, sir, sic, sor, soc, BLOCK: tl.constexpr,
):
    """Masked transpose wrapped in a program_id(2)==0 gate (nested scf.if).
    Uses axis-2 program id (always 0 in our (1,1) grid) so the gate does not
    interfere with the pid_x/pid_y tiling along axes 0 and 1."""
    pid_z = tl.program_id(2)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x = pid_x * BLOCK + tl.arange(0, BLOCK)
    y = pid_y * BLOCK + tl.arange(0, BLOCK)
    xi = x[None, :]
    yi = y[:, None]
    mask = (xi < cols) & (yi < rows)
    tile = tl.load(in_ptr + yi * sir + xi * sic, mask=mask, other=0.0)
    if pid_z == 0:
        tl.store(out_ptr + xi * sor + yi * soc, tile, mask=mask)


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _make_input():
    return torch.arange(N_ELEMS, dtype=torch.float32, device=_device()).reshape(
        BLOCK_N, BLOCK_N
    ).contiguous()


def _zeros_2d():
    return torch.zeros(BLOCK_N, BLOCK_N, dtype=torch.float32, device=_device()).contiguous()


def _gate_one():
    return torch.ones(1, dtype=torch.int32, device=_device())


# L1d2c Phase B loop count (AC.T1): each cell runs N_REPEATS times with
# bit-exact assertion every iteration to detect the per-warp-race
# nondeterminism characterized in Phase A's RCA.
N_REPEATS = 10


# RESOLVED 2026-06-03 (XPASS disposition): C6/C7 were marked
# xfail(strict=False) for a hypothesised Apple Metal `tg_load_indexed`
# lane-aliasing miscompile. That miscompile no longer reproduces — the
# current lowering for these single-threadgroup masked-cvt kernels emits no
# cross-lane `tg_load_indexed` + `threadgroup_barrier` reload (dumped MSL
# confirmed), so the Apple bug cannot fire. C6/C7 now pass bit-exact and
# deterministically (30/30). The xfail marks are removed; the cells stay in
# the sweep as regression guards. See
# the implementation notes.
@pytest.mark.parametrize(
    "cell",
    [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
    ],
)
def test_l1d2c_phase_a_probe(cell):
    if _device() == "cpu":
        pytest.skip("MPS device required")

    for iteration in range(N_REPEATS):
        inp = _make_input()
        out = _zeros_2d()

        if cell == "C0":
            k_C0_uncond_nobarrier_noif[(1,)](inp, out, BLOCK_N, num_warps=NUM_WARPS)
            expected = inp.clone()
        elif cell == "C1":
            k_C1_uncond_nobarrier_if[(1,)](inp, out, BLOCK_N, num_warps=NUM_WARPS)
            expected = inp.clone()
        elif cell == "C2":
            k_C2_uncond_barrier_noif[(1,)](inp, out, BLOCK_N, num_warps=NUM_WARPS)
            expected = inp.t().contiguous()
        elif cell == "C3":
            k_C3_uncond_barrier_if[(1,)](inp, out, BLOCK_N, num_warps=NUM_WARPS)
            expected = inp.t().contiguous()
        elif cell == "C4":
            k_C4_masked_nobarrier_noif[(1,)](
                inp, out, N_ELEMS, BLOCK_N, num_warps=NUM_WARPS,
            )
            expected = inp.clone()
        elif cell == "C5":
            k_C5_masked_nobarrier_if[(1,)](
                inp, out, N_ELEMS, BLOCK_N, num_warps=NUM_WARPS,
            )
            expected = inp.clone()
        elif cell == "C6":
            k_C6_masked_barrier_noif[(1, 1)](
                inp, out, BLOCK_N, BLOCK_N, BLOCK_N, 1, BLOCK_N, 1, BLOCK_N,
                num_warps=NUM_WARPS,
            )
            expected = inp.t().contiguous()
        elif cell == "C7":
            k_C7_masked_barrier_if[(1, 1, 1)](
                inp, out, BLOCK_N, BLOCK_N, BLOCK_N, 1, BLOCK_N, 1, BLOCK_N,
                num_warps=NUM_WARPS,
            )
            expected = inp.t().contiguous()
        else:
            pytest.fail(f"unknown cell {cell}")

        ok = torch.equal(out, expected)
        assert ok, (
            f"[{cell}] FAIL on iteration {iteration}/{N_REPEATS}\n"
            f"out=\n{out.cpu()}\nexpected=\n{expected.cpu()}"
        )
