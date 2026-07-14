"""Flash attention (online-softmax multi-head attention) on the Metal backend.

The verbatim leet-triton `hard-mult_head_attention.py` kernel: a `scf.for` over
key blocks carrying (2D accumulator, running sum, running max) that does two
`tl.dot`s (Q@K^T and P@V) with a masked online softmax in between. The matmul
track can't absorb the second dot (its A operand is a computed `exp`, not a
load), so `tryFlashAttentionLoop` (TritonGPUToMetal.cpp) recognizes the whole
loop + divide epilogue + masked store and lowers it to one `metal.flash_attention`
op, whose emitter renders the Phase-0-validated simdgroup body (both matmuls on
simdgroup hardware; S/P/O + running state in the threadgroup scalar domain).

Compared against `torch.scaled_dot_product_attention` per head. Envelope:
BLOCKSIZE_N == 32, num_warps == 4 (matches the leet driver), tile dims multiples
of 8, threadgroup working set <= 32 KiB. d_head varies: d_head < BLOCKSIZE_d (=
max(16, d_head)) exercises the padded-column masking (`d < d_head`); d_head ==
BD is the unpadded path. N covers block-aligned, masked-tail, and ragged cases.
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


@triton.jit
def mha_kernel(Q_ptr, K_ptr, V_ptr, output_ptr,
               N, d_model, h,
               BLOCKSIZE_N: tl.constexpr,
               BLOCKSIZE_d: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offset = tl.arange(0, BLOCKSIZE_N)
    d_head = d_model // h
    offset_N = pid0 * BLOCKSIZE_N + offset
    offset_d = pid1 * d_head + tl.arange(0, BLOCKSIZE_d)
    mask_d = tl.arange(0, BLOCKSIZE_d) < d_head
    offset_Q = offset_N[:, None] * d_model + offset_d[None, :]
    mask_Q = (offset_N[:, None] < N) & mask_d[None, :]
    data_Q = tl.load(Q_ptr + offset_Q, mask=mask_Q)
    attention_logit_scale = 1.0 / tl.sqrt(d_head + 0.0)
    accumulator = tl.zeros((BLOCKSIZE_N, BLOCKSIZE_d), dtype=tl.float32)
    softmax_running_sum = tl.zeros([BLOCKSIZE_N], dtype=tl.float32)
    softmax_current_max = tl.full([BLOCKSIZE_N], float("-inf"), dtype=tl.float32)
    for current_index in range(0, N, BLOCKSIZE_N):
        current_N_offset = current_index + offset
        current_N_mask = current_N_offset < N
        offset_K = current_N_offset[:, None] * d_model + offset_d[None, :]
        mask_K = current_N_mask[:, None] & mask_d[None, :]
        data_K = tl.load(K_ptr + offset_K, mask=mask_K, other=0.0)
        offset_V = current_N_offset[:, None] * d_model + offset_d[None, :]
        mask_V = current_N_mask[:, None] & mask_d[None, :]
        data_V = tl.load(V_ptr + offset_V, mask=mask_V, other=0.0)
        attention_logits = tl.dot(data_Q, tl.trans(data_K)) * attention_logit_scale
        attention_logits_mask = (offset_N[:, None] < N) & (current_N_offset[None, :] < N)
        attention_logits = tl.where(attention_logits_mask, attention_logits, float("-inf"))
        current_block_max = tl.max(attention_logits, axis=1)
        max_value = tl.maximum(current_block_max, softmax_current_max)
        softmax_scaler = tl.exp(softmax_current_max - max_value)
        softmax_current_max = max_value
        attention_logits_shift = attention_logits - max_value[:, None]
        softmax_nom = tl.exp(attention_logits_shift)
        softmax_denom = tl.sum(softmax_nom, axis=1)
        softmax_running_sum = tl.fma(softmax_running_sum, softmax_scaler, softmax_denom)
        accumulator = tl.fma(accumulator, softmax_scaler[:, None], tl.dot(softmax_nom, data_V))
    accumulator = accumulator / softmax_running_sum[:, None]
    tl.store(output_ptr + offset_Q, accumulator, mask=mask_Q)


def _reference(Q, K, V, N, d_model, h):
    d_head = d_model // h
    ref = torch.zeros(N, d_model, dtype=torch.float32)
    Qc, Kc, Vc = Q.cpu(), K.cpu(), V.cpu()
    for hi in range(h):
        sl = slice(hi * d_head, (hi + 1) * d_head)
        ref[:, sl] = torch.nn.functional.scaled_dot_product_attention(
            Qc[:, sl].unsqueeze(0), Kc[:, sl].unsqueeze(0), Vc[:, sl].unsqueeze(0)
        ).squeeze(0)
    return ref


@pytest.mark.parametrize(
    "N, d_model, h",
    [
        (64, 64, 4),    # block-aligned, d_head=16 (== BD)
        (48, 64, 4),    # masked tail (48 = 32 + 16)
        (33, 64, 4),    # ragged
        (128, 128, 8),  # more heads, d_head=16
        (96, 32, 2),    # fewer heads, d_head=16
        (256, 64, 4),   # many key blocks
        (64, 64, 8),    # d_head=8 < BD=16 -> padded-column masking
        (64, 64, 2),    # d_head=32 == BD=32 (no padding)
        (96, 96, 3),    # d_head=32, ragged N
    ],
)
def test_flash_attention_online_softmax(N, d_model, h):
    torch.manual_seed(0xF1A5 + N * 7 + d_model + h)
    Q = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    K = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    V = torch.randn(N, d_model, dtype=torch.float32, device="mps").contiguous()
    out = torch.zeros(N, d_model, dtype=torch.float32, device="mps").contiguous()

    BLOCKSIZE_N = 32
    BLOCKSIZE_d = max(16, d_model // h)
    grid = (triton.cdiv(N, BLOCKSIZE_N), h)
    mha_kernel[grid](Q, K, V, out, N, d_model, h, BLOCKSIZE_N, BLOCKSIZE_d, num_warps=4)

    expected = _reference(Q, K, V, N, d_model, h)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)
