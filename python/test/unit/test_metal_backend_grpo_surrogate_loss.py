"""GRPO surrogate loss on the Metal backend.

Runs the verbatim `leet-triton/medium-grpo_surrogate_loss.py` kernel. The single
compiler gap it exposed was elementwise FLOAT MIN: `ArithMinMaxFLowering` covered
`arith.maxnumf` / `arith.maximumf` (`tl.maximum`) and `ArithIntMinMaxLowering`
covered i32 `arith.minsi`, but `arith.minnumf` / `arith.minimumf` (`tl.minimum`)
had no elementwise pattern at all — float min only existed on the *reduce
combine* path (`tl.min`). The PPO/GRPO ratio clip
`tl.minimum(tl.maximum(ratio, lo), hi)` therefore failed to legalize.

Two independent sites needed the min case, and this file pins both:

* the elementwise pattern `ArithMinMaxFLowering<MinNumFOp/MinimumFOp, minOp>`
  (`test_minimum_elementwise` / `test_clamp_idiom`), and
* the reduce CONE evaluator — `evalRank1ValueAt` plus its `rank1ConeSupported`
  dry-run predicate (and the rank-2 pair, kept in sync per the comment there).
  The grpo accumulator `tl.sum(tl.where(mask, l_clip - beta * k, 0.0))` has the
  whole clip chain *inside* the reduce cone, which the reduce lowering re-emits
  per element rather than materializing; a cone op missing from the predicate is
  silently unreachable (`test_minimum_in_reduce_cone` / `test_minimum_in_rank2_cone`).

`metal.binary_exp minOp` already existed in MetalOps.td and ModuleTranslation
already emitted it as the MSL function-call form `min(a, b)`, so nothing
downstream of the conversion needed changing.

The kernel also exercises, all previously working: SCALAR `tl.atomic_add(ptr,
f32)` (one add per program, not per thread), scalar `math.sqrt` of a reduce
result, sub-threadgroup rank-1 reduces (BLOCK_G as small as 1), runtime
`divsi`/`remsi` program-id splitting, select-by-index gather-via-reduce, f32
scalar kernel args, and a loop-carried scalar `acc += tl.sum(...)`.
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


# Verbatim copy of leet-triton/medium-grpo_surrogate_loss.py.
@triton.jit
def _grpo_surrogate_loss_kernel(
    rewards_ptr,
    log_pi_ptr,
    log_pi_old_ptr,
    log_ref_ptr,
    output_ptr,
    clip_eps, beta,
    G, S,
    inv_total,
    BLOCK_G: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // G
    g = pid % G

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    rew = tl.load(rewards_ptr + b * G + offs_g, mask=mask_g, other = 0.0)
    mu = tl.sum(rew, axis = 0) / G
    diff = tl.where(mask_g, rew - mu, 0.0)
    sigma = tl.sqrt(tl.sum(diff * diff, axis = 0) / G)
    adv = (rew - mu) / (sigma + 1e-8)
    a = tl.sum(tl.where(offs_g == g, adv, 0.0), axis = 0)

    lo = 1.0 - clip_eps
    hi = 1.0 + clip_eps

    row_start = pid * S
    acc = 0.0
    for s0 in range(0, S, BLOCK_S):
        offs = s0 + tl.arange(0, BLOCK_S)
        mask = offs < S
        lp = tl.load(log_pi_ptr + row_start + offs, mask = mask, other = 0.0)
        lpo = tl.load(log_pi_old_ptr + row_start + offs, mask = mask, other = 0.0)
        lrf = tl.load(log_ref_ptr + row_start + offs, mask = mask, other = 0.0)

        ratio = tl.exp(lp - lpo)

        ratio_c = tl.minimum(tl.maximum(ratio, lo), hi)
        l_clip = tl.minimum(ratio * a, ratio_c * a)

        d = lrf - lp
        k = tl.exp(d) - d - 1.0

        acc += tl.sum(tl.where(mask, l_clip - beta * k, 0.0), axis = 0)
    tl.atomic_add(output_ptr, -acc * inv_total)


def _solve(rewards, log_pi, log_pi_old, log_ref, output, clip_eps, beta, B, G, S,
           num_warps=4):
    output.zero_()
    _grpo_surrogate_loss_kernel[(B * G,)](
        rewards, log_pi, log_pi_old, log_ref, output,
        clip_eps, beta,
        G, S, 1.0 / (B * G * S),
        BLOCK_G=triton.next_power_of_2(G),
        BLOCK_S=1024,
        num_warps=num_warps,
    )


def _reference(rewards, log_pi, log_pi_old, log_ref, clip_eps, beta, B, G, S):
    # float64 on the CPU: the group standardisation is numerically touchy at
    # small G, and the kernel sums B*G*S terms in fp32.
    rew = rewards.double().view(B, G)
    mu = rew.mean(dim=1, keepdim=True)
    sigma = torch.sqrt(((rew - mu) ** 2).mean(dim=1, keepdim=True))
    adv = ((rew - mu) / (sigma + 1e-8)).reshape(B * G, 1)

    lp = log_pi.double().view(B * G, S)
    lpo = log_pi_old.double().view(B * G, S)
    lrf = log_ref.double().view(B * G, S)

    ratio = torch.exp(lp - lpo)
    ratio_c = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    l_clip = torch.minimum(ratio * adv, ratio_c * adv)

    d = lrf - lp
    k = torch.exp(d) - d - 1.0
    return (-(l_clip - beta * k).sum() / (B * G * S)).reshape(1).float()


def _inputs(B, G, S, seed):
    torch.manual_seed(seed)
    rewards = torch.randn(B, G, device="mps")
    # Log-probs near each other so `ratio` straddles the clip band on both sides.
    log_pi = torch.randn(B * G, S, device="mps") * 0.15 - 1.0
    log_pi_old = torch.randn(B * G, S, device="mps") * 0.15 - 1.0
    log_ref = torch.randn(B * G, S, device="mps") * 0.15 - 1.0
    return rewards, log_pi, log_pi_old, log_ref


# S: 64 (< BLOCK_S), 1000 (ragged), 1024 (exactly one tile), 2048/3000
# (multi-iteration loop). G: 1 (sigma == 0), non-pow2 5/6 (masked BLOCK_G).
@pytest.mark.parametrize("B, G, S", [
    (1, 4, 64),
    (2, 8, 128),
    (3, 5, 1000),
    (2, 4, 1024),
    (4, 8, 2048),
    (2, 6, 3000),
    (5, 1, 512),
    (1, 16, 100),
])
def test_grpo_surrogate_loss(B, G, S):
    rewards, log_pi, log_pi_old, log_ref = _inputs(B, G, S, B * 1000 + G * 10 + S)
    out = torch.zeros(1, device="mps")
    _solve(rewards, log_pi, log_pi_old, log_ref, out, 0.2, 0.04, B, G, S)
    torch.mps.synchronize()
    ref = _reference(rewards.cpu(), log_pi.cpu(), log_pi_old.cpu(), log_ref.cpu(),
                     0.2, 0.04, B, G, S)
    # B*G programs atomically fold into one scalar, so the summation order is
    # not reproducible — close in fp32 rather than bit-exact.
    torch.testing.assert_close(out.cpu(), ref, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_grpo_surrogate_loss_num_warps(num_warps):
    B, G, S = 3, 5, 1500
    rewards, log_pi, log_pi_old, log_ref = _inputs(B, G, S, 77)
    out = torch.zeros(1, device="mps")
    _solve(rewards, log_pi, log_pi_old, log_ref, out, 0.2, 0.04, B, G, S,
           num_warps=num_warps)
    torch.mps.synchronize()
    ref = _reference(rewards.cpu(), log_pi.cpu(), log_pi_old.cpu(), log_ref.cpu(),
                     0.2, 0.04, B, G, S)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("clip_eps", [0.0, 0.05, 0.2, 1.0])
def test_grpo_surrogate_loss_clip_eps(clip_eps):
    # clip_eps sweeps how often the min/max actually bite: 0.0 clamps every
    # ratio to exactly 1.0, 1.0 leaves nearly all of them untouched.
    B, G, S = 2, 8, 777
    rewards, log_pi, log_pi_old, log_ref = _inputs(B, G, S, 5150)
    out = torch.zeros(1, device="mps")
    _solve(rewards, log_pi, log_pi_old, log_ref, out, clip_eps, 0.04, B, G, S)
    torch.mps.synchronize()
    ref = _reference(rewards.cpu(), log_pi.cpu(), log_pi_old.cpu(), log_ref.cpu(),
                     clip_eps, 0.04, B, G, S)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------
# The compiler fix in isolation: elementwise float min, and float min inside a
# reduce cone.
# --------------------------------------------------------------------------
@triton.jit
def _minimum_kernel(A, B, Out, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    tl.store(Out + offs, tl.minimum(a, b), mask=mask)


# E==1 (BLOCK <= tpb) and E>1 (BLOCK > tpb), plus a masked non-pow2 N.
@pytest.mark.parametrize("N, num_warps", [(32, 1), (128, 4), (300, 4), (1024, 4)])
def test_minimum_elementwise(N, num_warps):
    torch.manual_seed(N)
    a = torch.randn(N, device="mps")
    b = torch.randn(N, device="mps")
    out = torch.zeros(N, device="mps")
    _minimum_kernel[(1,)](a, b, out, N, BLOCK=triton.next_power_of_2(N),
                          num_warps=num_warps)
    torch.mps.synchronize()
    # min is exact — no reassociation, so this is bit-exact.
    torch.testing.assert_close(out.cpu(), torch.minimum(a.cpu(), b.cpu()),
                               atol=0, rtol=0)


@triton.jit
def _clamp_kernel(A, Out, lo, hi, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + offs, mask=mask, other=0.0)
    tl.store(Out + offs, tl.minimum(tl.maximum(a, lo), hi), mask=mask)


def test_clamp_idiom():
    # The exact PPO/GRPO ratio clip, standalone.
    N = 512
    torch.manual_seed(0)
    a = torch.randn(N, device="mps") * 2.0
    out = torch.zeros(N, device="mps")
    _clamp_kernel[(1,)](a, out, -0.5, 0.5, N, BLOCK=N, num_warps=4)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), a.cpu().clamp(-0.5, 0.5),
                               atol=0, rtol=0)


@triton.jit
def _min_in_reduce_cone_kernel(A, B, Out, N, BLOCK: tl.constexpr):
    # The min is INSIDE the reduce cone: no tile is materialized, so this
    # exercises evalRank1ValueAt / rank1ConeSupported rather than the
    # elementwise pattern.
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + row * N + offs, mask=mask, other=0.0)
    b = tl.load(B + row * N + offs, mask=mask, other=0.0)
    tl.store(Out + row, tl.sum(tl.where(mask, tl.minimum(a, b), 0.0), axis=0))


@pytest.mark.parametrize("M, N", [(4, 64), (3, 200), (2, 1024)])
def test_minimum_in_reduce_cone(M, N):
    torch.manual_seed(M * 100 + N)
    a = torch.randn(M, N, device="mps")
    b = torch.randn(M, N, device="mps")
    out = torch.zeros(M, device="mps")
    _min_in_reduce_cone_kernel[(M,)](a, b, out, N,
                                     BLOCK=triton.next_power_of_2(N),
                                     num_warps=4)
    torch.mps.synchronize()
    ref = torch.minimum(a.cpu().double(), b.cpu().double()).sum(1).float()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-5)


@triton.jit
def _min_in_rank2_cone_kernel(A, B, Out, N, BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr):
    # Same, one rank up: keeps evalRank2ConeAt / rank2ConeSupported in sync with
    # their rank-1 counterparts (the comment on rank1ConeSupported warns that an
    # op accepted by the evaluator but missing from the predicate is silently
    # unreachable).
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    mask = cols < N
    a = tl.load(A + rows * N + cols, mask=mask, other=0.0)
    b = tl.load(B + rows * N + cols, mask=mask, other=0.0)
    tl.store(Out + tl.arange(0, BLOCK_M),
             tl.sum(tl.where(mask, tl.minimum(a, b), 0.0), axis=1))


@pytest.mark.parametrize("M, N", [(4, 64), (8, 100)])
def test_minimum_in_rank2_cone(M, N):
    torch.manual_seed(M * 7 + N)
    a = torch.randn(M, N, device="mps")
    b = torch.randn(M, N, device="mps")
    out = torch.zeros(M, device="mps")
    _min_in_rank2_cone_kernel[(1,)](a, b, out, N, BLOCK_M=M,
                                    BLOCK_N=triton.next_power_of_2(N),
                                    num_warps=4)
    torch.mps.synchronize()
    ref = torch.minimum(a.cpu().double(), b.cpu().double()).sum(1).float()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-5)
