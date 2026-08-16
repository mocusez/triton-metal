"""Parallel reverse-scan GAE on the Metal backend.

Runs the verbatim `leet-triton/medium-parallel_reverse_scan_gae.py` three-pass
solve. Its two scans are the two `tt.scan` shapes the backend could not lower,
and both are REVERSE — the GAE recurrence `A_t = delta_t + c * A_(t+1)` runs
backwards through the sequence:

1. `_gae_local_kernel` scans the affine pairs `(c, delta_t)` with
   `combine((a_l, x_l), (a_r, x_r)) = (a_l*a_r, a_r*x_l + x_r)`. That is exactly
   the monoid `isCanonicalAffineScan` already matched for the medium linear
   recurrence; only its `op.getReverse()` bail rejected this one. The affine
   combine is NOT commutative, so the primitive cannot just run the carry
   backwards: it mirrors each chunk IN PLACE, runs the verbatim forward
   template, and mirrors back (`test_local_kernel_compiles_to_metal`).

2. `_gae_apply_carry_kernel` scans `c` with plain multiplication to get the
   per-lane carry coefficient `c^(block_end - t)`. The single-operand path
   supported `reverse` but hardcoded the ADD combine, so a cumprod had no path
   at all. `metal.threadgroup_prefix_sum` now carries a `combine` attribute
   (absent means add, keeping every cumsum emission unchanged) and the padded
   sub-tpb tail selects the monoid's own identity — padding a cumprod with 0
   would zero every prefix (`test_cumprod_matches_reference`).

`_gae_carry_kernel`, the sequential middle pass, always COMPILED but returned
zeros, which is why compile-only coverage would have missed it. It does
`local = work[b]; work[b] = carry; carry = local + coeff*carry`, and the emitter
inlines a single-use read at its USE rather than its IR position — so the load
was emitted after the store that overwrote the slot and read back the carry it
had just written. Every block carry came out 0 with no diagnostic
(`test_carry_kernel_reads_before_overwriting`).

The reverse cumsum path was already reachable but had never been run: its
mirrored read is cross-thread (thread `t` reads the slot thread `tpb-1-t`
filled) and had no barrier before it, which is silent at num_warps=1 (one
SIMD-group runs in lockstep) and a race above it
(`test_reverse_cumsum_multi_warp_matches_reference`).
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GAE_PATH = REPO_ROOT / "leet-triton" / "medium-parallel_reverse_scan_gae.py"

_spec = importlib.util.spec_from_file_location("gae_kernel_module", GAE_PATH)
gae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gae)

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
)


def _compile_to_msl(kernel, signature, constexprs, *, num_warps):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    compiled = triton.compile(
        src,
        target=GPUTarget(backend="metal", arch=80, warp_size=32),
        options={"num_warps": num_warps},
    )
    raw = compiled.asm["metal"]
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


def _gae_reference(rewards, values, gamma, lam):
    """A_t = r_t + gamma*V_(t+1) - V_t + gamma*lam*A_(t+1), V_S = 0, A_S = 0."""
    B, S = rewards.shape
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros(B, dtype=rewards.dtype)
    for t in range(S - 1, -1, -1):
        next_value = values[:, t + 1] if t + 1 < S else torch.zeros_like(carry)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        carry = delta + gamma * lam * carry
        advantages[:, t] = carry
    return advantages


# ---------------------------------------------------------------------------
# Compilation: the two scan shapes.
# ---------------------------------------------------------------------------


def test_local_kernel_compiles_to_metal():
    msl = _compile_to_msl(
        gae._gae_local_kernel,
        {
            "rewards_ptr": "*fp32",
            "values_ptr": "*fp32",
            "advantages_ptr": "*fp32",
            "work_ptr": "*fp32",
            "gamma": "fp32",
            "c": "fp32",
            "S": "i32",
            "num_blocks": "i32",
            "stride_rb": "i32",
            "stride_rs": "i32",
            "stride_vb": "i32",
            "stride_vs": "i32",
            "stride_ab": "i32",
            "stride_as": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        {"BLOCK_SIZE": 256},
        num_warps=4,
    )
    assert "metal.threadgroup_affine_prefix_scan (reverse)" in msl, msl
    # Mirror in, forward template, mirror back onto the original positions.
    assert "float _aps_mir_a = " in msl, msl
    assert "[_aps_orig] = _aps_out_a;" in msl, msl


def test_apply_carry_kernel_compiles_to_metal():
    msl = _compile_to_msl(
        gae._gae_apply_carry_kernel,
        {
            "advantages_ptr": "*fp32",
            "work_ptr": "*fp32",
            "c": "fp32",
            "S": "i32",
            "num_blocks": "i32",
            "stride_ab": "i32",
            "stride_as": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        {"BLOCK_SIZE": 256},
        num_warps=4,
    )
    assert "metal.threadgroup_prefix_sum (cumprod)" in msl, msl
    assert "float _ps_carry = 1.0f;" in msl, msl
    assert "_ps_carry *= _ps_total;" in msl, msl
    assert "uint _ps_orig = " in msl, msl


def test_carry_kernel_reads_before_overwriting():
    # Regression pin for the emitter's read-after-overwrite guard. Assert the
    # SHAPE, not the values: the kernel returned zeros rather than failing, so
    # only the emission order distinguishes right from wrong.
    msl = _compile_to_msl(
        gae._gae_carry_kernel,
        {"work_ptr": "*fp32", "num_blocks": "i32"},
        {},
        num_warps=1,
    )
    body = msl[msl.index("kernel void _gae_carry_kernel"):]
    store = re.search(r"\bv0\[[^\]]*\] = ", body)
    assert store is not None, body
    after_store = body[store.end():]
    assert "v0[" not in after_store, (
        "the work buffer is read again after being overwritten; the loaded "
        f"value must be let-bound at its IR position instead:\n{body}"
    )


# ---------------------------------------------------------------------------
# Numerics: the original three-pass solve.
# ---------------------------------------------------------------------------


@requires_mps
@pytest.mark.parametrize(
    ("B", "S"),
    [
        (1, 1),
        (1, 7),
        (1, 256),  # exactly one block
        (1, 257),  # ragged second block
        (2, 300),
        (3, 1024),
        (4, 1000),
        (2, 4096),
        (1, 8192),
    ],
)
def test_original_solve_matches_gae_reference(B, S):
    torch.manual_seed(0x6AE + S)
    rewards_cpu = torch.randn((B, S), dtype=torch.float32)
    values_cpu = torch.randn((B, S), dtype=torch.float32)
    gamma, lam = 0.99, 0.95
    expected = _gae_reference(rewards_cpu, values_cpu, gamma, lam)

    rewards = rewards_cpu.to("mps")
    values = values_cpu.to("mps")
    advantages = torch.empty_like(rewards)
    gae.solve(rewards, values, advantages, gamma, lam, B, S)
    torch.mps.synchronize()
    torch.testing.assert_close(advantages.cpu(), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# The two scan primitives on their own.
# ---------------------------------------------------------------------------


@triton.jit
def _mul(left, right):
    return left * right


@triton.jit
def _add(left, right):
    return left + right


@triton.jit
def _scan_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    REVERSE: tl.constexpr,
    MUL: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    if MUL:
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
        y = tl.associative_scan(x, axis=0, combine_fn=_mul, reverse=REVERSE)
    else:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.associative_scan(x, axis=0, combine_fn=_add, reverse=REVERSE)
    tl.store(out_ptr + offsets, y, mask=mask)


def _run_scan(x_cpu, block, reverse, mul, num_warps):
    x = x_cpu.to("mps")
    out = torch.zeros(block, dtype=torch.float32, device="mps")
    _scan_kernel[(1,)](
        x, out, x_cpu.numel(), BLOCK=block, REVERSE=reverse, MUL=mul,
        num_warps=num_warps,
    )
    torch.mps.synchronize()
    return out.cpu()[: x_cpu.numel()]


@requires_mps
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("N", "block", "num_warps"),
    [
        (1024, 1024, 4),  # E = 8 chunks over tpb = 128
        (256, 256, 1),  # E = 8 chunks over tpb = 32
        (128, 128, 4),  # E = 1, a single chunk
        (5, 16, 4),  # sub-tpb: the padded tail must take the MUL identity
    ],
)
def test_cumprod_matches_reference(N, block, num_warps, reverse):
    torch.manual_seed(0xC0FFEE + N)
    # Near 1.0 so a 1024-long product neither overflows nor flushes to zero,
    # and every element still moves the result.
    x_cpu = torch.empty(N, dtype=torch.float32).uniform_(0.98, 1.02)
    got = _run_scan(x_cpu, block, reverse, mul=True, num_warps=num_warps)
    ref = x_cpu.double()
    expected = (
        ref.flip(0).cumprod(0).flip(0) if reverse else ref.cumprod(0)
    ).float()
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@requires_mps
@pytest.mark.parametrize("num_warps", [1, 4, 8])
def test_reverse_cumsum_multi_warp_matches_reference(num_warps):
    # The reverse add scan predates this work but had no runtime coverage. Its
    # mirrored read is cross-thread, so a missing barrier there is invisible at
    # num_warps=1 and a race at every larger threadgroup.
    N = 1024
    torch.manual_seed(0x5CA7)
    x_cpu = torch.randn(N, dtype=torch.float32)
    got = _run_scan(x_cpu, N, reverse=True, mul=False, num_warps=num_warps)
    expected = x_cpu.double().flip(0).cumsum(0).flip(0).float()
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------------------
# The combine matcher must stay structural.
# ---------------------------------------------------------------------------


@triton.jit
def _noncanonical_combine(left, right):
    return (left + right) * 2.0


@triton.jit
def _noncanonical_scan_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.associative_scan(x, axis=0, combine_fn=_noncanonical_combine)
    tl.store(out_ptr + offsets, y)


def test_noncanonical_single_operand_combine_remains_rejected():
    # `(l + r) * 2` is not a monoid this backend implements. Matching on "the
    # first op in the combine region" would take the addf and lower it as a
    # plain cumsum — a silently wrong answer. The match is structural instead:
    # exactly one arithmetic op, over the two block arguments, returned as-is.
    # Conversion failure can crash while cleaning up, so use a child process.
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        path = Path({str(Path(__file__).resolve())!r})
        spec = importlib.util.spec_from_file_location("gae_test_child", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        src = ASTSource(
            fn=module._noncanonical_scan_kernel,
            signature={{"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"}},
            constexprs={{"BLOCK": 256}},
        )
        target = GPUTarget(backend="metal", arch=80, warp_size=32)
        triton.compile(src, target=target, options={{"num_warps": 4}})
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, "non-canonical combine unexpectedly compiled"
    diagnostics = result.stdout + result.stderr
    assert "failed to legalize operation 'tt.scan'" in diagnostics, diagnostics
