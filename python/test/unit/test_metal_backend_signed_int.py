"""Signedness of integer operations on the Metal backend.

MSL takes signed-vs-unsigned semantics from the C TYPE of an operand
expression, but in MLIR the signedness lives in the OP: `slt` vs `ult`,
`divsi` vs `divui`, `shrsi` vs `shrui`, `sitofp` vs `uitofp`. Triton spells
both `tl.int32` and `tl.uint32` as a signless i32 and lets the op carry the
distinction, while `metalStorageElementType` declares a signless device buffer
as `uint32_t`. Because the emitter inlines a load at its use site, an uncast
operand silently turned every SIGNED operation on loaded data into an unsigned
one — `x < 0` never fired, `x.to(tl.float32)` produced 2**32, `//`, `%` and
`>>` were all wrong, and `tl.sort` ordered negatives last. `signednessCast` in
`ModuleTranslation.cpp` now spells each op's own signedness onto its operands.

Everything here needs NEGATIVE data to fail: the pre-existing suite passed
throughout because its integer fixtures are almost entirely non-negative, and
i8 was already exempt from the unsigned storage mapping. `i8` is kept in the
dtype sweep as the control that was correct before and must stay correct.

`test_umulhi_constant_operand` and the philox tests cover a second, compounding
defect: a signless i32 CONSTANT is printed as a signed literal, so `0xD2511F53`
came out `-766436013` and `metal.mulhi_ui`'s widening `(uint64_t)(...)`
sign-extended it. Only constants were affected, which is why a `tl.umulhi` on
loaded data was exact while `tl.rand`/`tl.randint`/`tl.randn` — whose philox
round multipliers are constants — silently produced the wrong stream.
`tl.rand` returned values in [0, 2) and `tl.randn` returned NaN.
"""

from __future__ import annotations


import pytest

torch = pytest.importorskip("torch")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

pytest.importorskip(
    "triton._C.libtriton.metal",
    reason="Metal backend pybind module not built into libtriton",
)
if not torch.backends.mps.is_available():
    pytest.skip(
        "Metal backend requires an MPS-enabled PyTorch (Apple Silicon)",
        allow_module_level=True,
    )


NEGATIVE = [-1, -2, -7, -100, 5, 7, 0, -3]


@triton.jit
def _cmp_zero_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, tl.where(tl.load(x_ptr + i) < 0, 1, 0))


@triton.jit
def _to_float_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, tl.load(x_ptr + i).to(tl.float32))


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_signed_compare_against_zero_all_widths(dtype):
    x = torch.tensor(NEGATIVE, device="mps", dtype=dtype)
    out = torch.zeros(len(NEGATIVE), device="mps", dtype=torch.int32)
    _cmp_zero_kernel[(1,)](x, out, len(NEGATIVE))
    torch.mps.synchronize()
    expected = (x.cpu() < 0).to(torch.int32)
    assert torch.equal(out.cpu(), expected), f"{dtype}: {out.cpu().tolist()}"


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_signed_int_to_float_all_widths(dtype):
    x = torch.tensor(NEGATIVE, device="mps", dtype=dtype)
    out = torch.zeros(len(NEGATIVE), device="mps", dtype=torch.float32)
    _to_float_kernel[(1,)](x, out, len(NEGATIVE))
    torch.mps.synchronize()
    # Unsigned conversion produced 2**32 (or 2**16 / 2**64) for every negative.
    assert torch.equal(out.cpu(), x.cpu().to(torch.float32)), out.cpu().tolist()


@triton.jit
def _div_rem_shift_kernel(x_ptr, div_ptr, rem_ptr, shr_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    tl.store(div_ptr + i, x // 2)
    tl.store(rem_ptr + i, x % 4)
    tl.store(shr_ptr + i, x >> 1)


def test_signed_div_rem_shift_of_negatives():
    x = torch.tensor(NEGATIVE, device="mps", dtype=torch.int32)
    outs = [torch.zeros(len(NEGATIVE), device="mps", dtype=torch.int32) for _ in range(3)]
    _div_rem_shift_kernel[(1,)](x, *outs, len(NEGATIVE))
    torch.mps.synchronize()
    ref = x.cpu()
    # Triton's `//` and `%` are arith.divsi/remsi, which truncate toward zero
    # and take the dividend's sign — not torch's floor semantics.
    want_div = torch.tensor([int(v / 2) for v in ref.tolist()], dtype=torch.int32)
    want_rem = torch.tensor([int(v) - 4 * int(v / 4) for v in ref.tolist()], dtype=torch.int32)
    want_shr = ref >> 1  # arithmetic shift: sign-propagating
    assert torch.equal(outs[0].cpu(), want_div), outs[0].cpu().tolist()
    assert torch.equal(outs[1].cpu(), want_rem), outs[1].cpu().tolist()
    assert torch.equal(outs[2].cpu(), want_shr), outs[2].cpu().tolist()


@triton.jit
def _cmp_pair_kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    lt = tl.load(a_ptr + i) < tl.load(b_ptr + i)
    tl.store(out_ptr + i, tl.where(lt, 1, 0))


def test_signed_compare_between_two_loaded_tensors():
    a = torch.tensor([-5, 3, -1, 7, -9, 0, 2, -3], device="mps", dtype=torch.int32)
    b = torch.tensor([2, -4, -1, 1, -2, 5, -6, 8], device="mps", dtype=torch.int32)
    out = torch.zeros(8, device="mps", dtype=torch.int32)
    _cmp_pair_kernel[(1,)](a, b, out, 8)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), (a.cpu() < b.cpu()).to(torch.int32)), out.cpu().tolist()


@triton.jit
def _unsigned_div_shift_kernel(x_ptr, div_ptr, shr_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i).to(tl.uint32, bitcast=True)
    tl.store(div_ptr + i, (x // 2).to(tl.int32, bitcast=True))
    tl.store(shr_ptr + i, (x >> 1).to(tl.int32, bitcast=True))


def test_unsigned_div_and_shift_stay_unsigned():
    """The mirror image: divui/shrui must NOT become signed."""
    x = torch.tensor([-1, -2, -8, 6], device="mps", dtype=torch.int32)
    div = torch.zeros(4, device="mps", dtype=torch.int32)
    shr = torch.zeros(4, device="mps", dtype=torch.int32)
    _unsigned_div_shift_kernel[(1,)](x, div, shr, 4)
    torch.mps.synchronize()
    raw = [v & 0xFFFFFFFF for v in x.cpu().tolist()]
    want = torch.tensor([(v // 2) - (1 << 32) if (v // 2) >= (1 << 31) else v // 2
                         for v in raw], dtype=torch.int32)
    assert torch.equal(div.cpu(), want), div.cpu().tolist()
    assert torch.equal(shr.cpu(), want), shr.cpu().tolist()


@triton.jit
def _sort_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, tl.sort(tl.load(x_ptr + i)))


def test_sort_orders_negative_ints_first():
    """Under unsigned compares this sorted the negatives to the END."""
    x = torch.tensor([-5, 3, -1, 7, -9, 0, 2, -3], device="mps", dtype=torch.int32)
    out = torch.zeros(8, device="mps", dtype=torch.int32)
    _sort_kernel[(1,)](x, out, 8)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), x.cpu().sort().values), out.cpu().tolist()


@triton.jit
def _umulhi_const_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    # 0xD2511F53 is philox's round multiplier and exceeds INT32_MAX, so it is
    # the constant whose signed printing broke the widening inside mulhi_ui.
    tl.store(out_ptr + i, tl.umulhi(tl.load(x_ptr + i), 0xD2511F53))


def test_umulhi_constant_operand():
    raw = [1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678, 0xCD9E8D57, 3]
    x = torch.tensor([v - (1 << 32) if v >= (1 << 31) else v for v in raw],
                     device="mps", dtype=torch.int32)
    out = torch.zeros(len(raw), device="mps", dtype=torch.int32)
    _umulhi_const_kernel[(1,)](x, out, len(raw))
    torch.mps.synchronize()
    got = [v & 0xFFFFFFFF for v in out.cpu().tolist()]
    want = [(v * 0xD2511F53) >> 32 for v in raw]
    assert got == want, [hex(v) for v in got]


# --- philox ------------------------------------------------------------------
#
# A self-contained mirror of the reference in `test_random.py` (which cannot be
# imported here: it pulls in scipy, and its absence is why the RNG defect went
# unnoticed — the whole file fails to collect).

PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85
PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57
U32 = 0xFFFFFFFF


def _philox_reference(seed: int, n: int) -> list[int]:
    """The first `n` outputs of `tl.randint(seed, offset)` for offset 0..n-1."""
    key0, key1 = seed & U32, (seed >> 32) & U32
    out = []
    for offset in range(n):
        c0, c1, c2, c3 = offset, 0, 0, 0
        k0, k1 = key0, key1
        for _ in range(10):
            prod_a = PHILOX_ROUND_A * c0
            prod_b = PHILOX_ROUND_B * c2
            c0, c1, c2, c3 = (
                (prod_b >> 32) ^ c1 ^ k0,
                prod_b & U32,
                (prod_a >> 32) ^ c3 ^ k1,
                prod_a & U32,
            )
            k0, k1 = (k0 + PHILOX_KEY_A) & U32, (k1 + PHILOX_KEY_B) & U32
        out.append(c0)
    return out


@triton.jit
def _randint_kernel(out_ptr, seed, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, tl.randint(seed, i))


@pytest.mark.parametrize("seed", [0, 42, 0xFFFFFFFF, 0x0000000FCAFEB0BA])
def test_randint_matches_philox_reference(seed):
    n = 64
    out = torch.zeros(n, device="mps", dtype=torch.int32)
    _randint_kernel[(1,)](out, seed, n)
    torch.mps.synchronize()
    got = [v & 0xFFFFFFFF for v in out.cpu().tolist()]
    assert got == _philox_reference(seed, n), [hex(v) for v in got[:4]]


@triton.jit
def _rand_kernel(out_ptr, seed, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, tl.rand(seed, i))


def test_rand_stays_within_unit_interval():
    """Sign-extended philox constants put roughly half the values in [1, 2)."""
    n = 4096
    out = torch.zeros(n, device="mps", dtype=torch.float32)
    _rand_kernel[(1,)](out, 12345, n)
    torch.mps.synchronize()
    v = out.cpu()
    assert not v.isnan().any()
    assert int(((v < 0) | (v >= 1)).sum()) == 0, (v.min().item(), v.max().item())
    assert 0.45 < v.mean().item() < 0.55, v.mean().item()


@triton.jit
def _randn_kernel(out_ptr, seed, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out_ptr + i, tl.randn(seed, i))


def test_randn_is_finite_and_standard_normal():
    """Box-Muller takes log() of the uniform, so out-of-range input gave NaN."""
    n = 4096
    out = torch.zeros(n, device="mps", dtype=torch.float32)
    _randn_kernel[(1,)](out, 12345, n)
    torch.mps.synchronize()
    v = out.cpu()
    assert int(v.isnan().sum()) == 0
    assert int(v.isinf().sum()) == 0
    assert abs(v.mean().item()) < 0.1, v.mean().item()
    assert 0.9 < v.std().item() < 1.1, v.std().item()
