"""Adder-transformer autoregressive inference on the Metal backend.

The verbatim leet-triton `medium-adder_transformer_inference.py` kernel: an
autoregressive `for pos` loop that writes/reads a device KV cache and runs a
masked online softmax whose reduce cone roots in the loop-carried `next_token`
(via `scf.if`). This exercises the staged-leaf reduce (Increment 2.5/3:
loop-carried and chained-reduce leaves resolved via getRemappedValue at M<=tpb),
the emitter CSE that keeps the deep per-thread chain from exploding the MSL, and
the multi-program (grid>1) global-row device addressing of the reduce. Toy model
with no numerical oracle, so this is the kernel's own smoke test: it must
compile, run, and produce finite output. batch_size=128 is single-program;
batch_size=256 exercises the two-program path.
"""

from __future__ import annotations

import math

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
def adder_transformer_kernel(
    prompts_ptr, output_ptr, weights_ptr,
    k0_ptr, k1_ptr, v0_ptr, # 显存版 KV Cache
    batch_size, scale, total_steps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = batch_idx < batch_size

    # 只需加载 10 个权重到寄存器
    w0 = tl.load(weights_ptr + 0)
    w1 = tl.load(weights_ptr + 1)
    q0 = tl.load(weights_ptr + 2)
    q1 = tl.load(weights_ptr + 3)
    v0 = tl.load(weights_ptr + 4)
    a  = tl.load(weights_ptr + 5)
    c  = tl.load(weights_ptr + 6)
    carry_w = tl.load(weights_ptr + 7)
    n0 = tl.load(weights_ptr + 8)
    n1 = tl.load(weights_ptr + 9)

    seq_idx = tl.arange(0, 64)
    next_token = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # 动态 total_steps 会阻止 Triton 的灾难性静态循环展开
    for pos in range(total_steps):
        if pos < 31:
            d = tl.load(prompts_ptr + batch_idx * 31 + pos, mask=mask, other=0)
        else:
            d = next_token

        d_f = d.to(tl.float32)

        # 1. 词嵌入
        e0 = w0 - w1 * d_f * d_f
        e1 = -d_f

        # 2. 嵌入层 RMSNorm
        m_e = (e0 * e0 + e1 * e1) * 0.5
        denom_e = tl.sqrt(m_e + 1e-6)
        h0 = e0 / denom_e
        h1 = e1 / denom_e

        # 3. 注意力投影
        q0_val = h0 * q0
        q1_val = h0 * q1
        k0_val = h0
        v0_val = h1 * v0

        # 注意力 RMSNorms
        m_q = (q0_val * q0_val + q1_val * q1_val) * 0.5
        denom_q = tl.sqrt(m_q + 1e-6)
        q0_norm = q0_val / denom_q
        q1_norm = q1_val / denom_q

        m_k = (k0_val * k0_val) * 0.5
        denom_k = tl.sqrt(m_k + 1e-6)
        k0_norm = k0_val / denom_k

        # 旋转位置编码 (RoPE)
        pw = pos * 0.3306939635357677 
        pw_t = tl.full([1], pw, dtype=tl.float32)
        cos_pw = tl.cos(pw_t)
        sin_pw = tl.sin(pw_t)

        q0_rope = q0_norm * cos_pw - q1_norm * sin_pw
        q1_rope = q0_norm * sin_pw + q1_norm * cos_pw
        k0_rope = k0_norm * cos_pw
        k1_rope = k0_norm * sin_pw

        # --- 写入显存 KV Cache ---
        k0_out_ptr = k0_ptr + batch_idx * 64 + pos
        k1_out_ptr = k1_ptr + batch_idx * 64 + pos
        v0_out_ptr = v0_ptr + batch_idx * 64 + pos
        tl.store(k0_out_ptr, k0_rope, mask=mask)
        tl.store(k1_out_ptr, k1_rope, mask=mask)
        tl.store(v0_out_ptr, v0_val, mask=mask)

        # --- 读取至今全部的 KV Cache ---
        # 编译器处理从外部加载 [BLOCK_SIZE, 64] 会极其轻松，告别卡死
        k0_in_ptrs = k0_ptr + batch_idx[:, None] * 64 + seq_idx[None, :]
        k1_in_ptrs = k1_ptr + batch_idx[:, None] * 64 + seq_idx[None, :]
        v0_in_ptrs = v0_ptr + batch_idx[:, None] * 64 + seq_idx[None, :]
        
        k0_seq = tl.load(k0_in_ptrs, mask=mask[:, None], other=0.0)
        k1_seq = tl.load(k1_in_ptrs, mask=mask[:, None], other=0.0)
        v0_seq = tl.load(v0_in_ptrs, mask=mask[:, None], other=0.0)

        # 缩放点积
        score = (q0_rope[:, None] * k0_seq + q1_rope[:, None] * k1_seq) * scale
        score = tl.where(seq_idx[None, :] <= pos, score, float('-inf'))

        # Softmax
        max_score = tl.max(score, axis=1)
        p = tl.exp(score - max_score[:, None])
        p = tl.where(seq_idx[None, :] <= pos, p, 0.0)
        sum_p = tl.sum(p, axis=1)
        p = p / sum_p[:, None]

        attn0 = tl.sum(p * v0_seq, axis=1)

        # 第一次残差
        h_post0 = e0
        h_post1 = e1 + attn0

        # 4. MLP Pre-Norm
        m_post = (h_post0 * h_post0 + h_post1 * h_post1) * 0.5
        denom_post = tl.sqrt(m_post + 1e-6)
        h_mlp_in0 = h_post0 / denom_post
        h_mlp_in1 = h_post1 / denom_post

        # 门控与 SwiGLU 机制
        g0 = h_mlp_in0 * a + h_mlp_in1 * c
        g1 = h_mlp_in0 * (a - c / 1000.0) + h_mlp_in1 * c
        
        mix0 = (g0 * tl.sigmoid(g0)) * h_mlp_in0
        mix1 = (g1 * tl.sigmoid(g1)) * h_mlp_in0

        mlp_out1 = carry_w * (mix1 - mix0)

        # 第二次残差
        h_final0 = h_post0
        h_final1 = h_post1 + mlp_out1

        # 5. Final RMSNorm
        m_final = (h_final0 * h_final0 + h_final1 * h_final1) * 0.5
        denom_final = tl.sqrt(m_final + 1e-6)
        out0 = (h_final0 / denom_final) * n0
        out1 = (h_final1 / denom_final) * n1

        # 6. 自回归生成循环
        if pos >= 30:
            max_logit = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
            best_digit = next_token
            step = pos - 30

            # 强制静态展开小循环：使用 tl.static_range 避开之前张量 float 转换问题
            for digit in tl.static_range(10):
                d_val = digit * 1.0
                E_d0 = w0 - w1 * d_val * d_val
                E_d1 = -d_val
                
                logit_d = out0 * E_d0 + out1 * E_d1
                
                out_ptr_idx = output_ptr + batch_idx * 110 + step * 10 + digit
                tl.store(out_ptr_idx, logit_d, mask=mask)
                
                is_better = logit_d > max_logit
                max_logit = tl.where(is_better, logit_d, max_logit)
                best_digit = tl.where(is_better, tl.full([BLOCK_SIZE], digit, dtype=tl.int32), best_digit)
                
            next_token = best_digit

def _solve(prompts, output, weights, batch_size):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(batch_size, BLOCK_SIZE),)
    w = 2.0 * math.pi / 19.0
    S_sq = math.log(10.0) / (math.sqrt(2.0) * (math.cos(0.3 * w) - math.cos(0.7 * w)))
    scale = S_sq / math.sqrt(2.0)
    dev = prompts.device
    k0 = torch.empty((batch_size, 64), device=dev, dtype=torch.float32)
    k1 = torch.empty((batch_size, 64), device=dev, dtype=torch.float32)
    v0 = torch.empty((batch_size, 64), device=dev, dtype=torch.float32)
    total_steps = 41
    adder_transformer_kernel[grid](
        prompts, output, weights, k0, k1, v0,
        batch_size, scale, total_steps, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)


@pytest.mark.parametrize("batch_size", [128, 256])
def test_adder_transformer_inference_smoke(batch_size):
    torch.manual_seed(0)
    prompts = torch.randint(0, 10, (batch_size, 31), dtype=torch.int32, device="mps")
    weights = torch.randn(10, dtype=torch.float32, device="mps")
    output = torch.zeros(batch_size, 110, dtype=torch.float32, device="mps")
    _solve(prompts, output, weights, batch_size)
    torch.mps.synchronize()
    assert output.shape == (batch_size, 110)
    assert torch.isfinite(output.cpu()).all(), "output contains non-finite values"
