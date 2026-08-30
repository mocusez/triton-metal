import torch
import triton
import triton.language as tl
import math

@triton.jit
def compute_softmax_max(logits, logits_max_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(logits + offs, mask=mask, other=-float("inf"))
    tl.atomic_max(logits_max_ptr, tl.max(x))

@triton.jit
def compute_softmax_denum(logits, logits_max_ptr, logits_denum_ptr, n, BLOCK_SIZE: tl.constexpr):
    logits_max = tl.load(logits_max_ptr)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(logits + offs, mask=mask, other=-float("inf"))
    tl.atomic_add(logits_denum_ptr, tl.sum(tl.exp(x - logits_max)))

@triton.jit
def compute_softmax(logits, logits_max_ptr, logits_denum_ptr, softmax, n, BLOCK_SIZE: tl.constexpr):
    logits_max = tl.load(logits_max_ptr)
    denum = tl.load(logits_denum_ptr)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(logits + offs, mask=mask, other=-float("inf"))
    tl.store(softmax + offs, tl.exp(x - logits_max) / denum, mask=mask)


@triton.jit
def st_1_sort(input, arguments, step, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    stride_pair = 1 << (step + 1)
    stride_half = 1 << step

    group = offs // stride_half
    left = group * stride_pair + (offs & (stride_half - 1))
    right = group * stride_pair + stride_pair - 1 - (offs & (stride_half - 1))

    mask_l = left < n
    mask_r = right < n

    a = tl.load(input + left, mask=mask_l, other=-float("inf"))
    b = tl.load(input + right, mask=mask_r, other=-float("inf"))

    arg_a = tl.load(arguments + left, mask=mask_l, other=0)
    arg_b = tl.load(arguments + right, mask=mask_r, other=0)

    swap = b > a

    tl.store(input + left, tl.where(swap, b, a), mask=mask_l)
    tl.store(input + right, tl.where(swap, a, b), mask=mask_r)
    tl.store(arguments + left, tl.where(swap, arg_b, arg_a), mask=mask_l)
    tl.store(arguments + right, tl.where(swap, arg_a, arg_b), mask=mask_r)

@triton.jit
def st_2_sort_splitter(input, arguments, step, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    stride_pair = 1 << (step + 1)
    stride_half = 1 << step

    group = offs // stride_half
    left = group * stride_pair + (offs & (stride_half - 1))
    right = left + stride_half

    mask_l = left < n
    mask_r = right < n

    a = tl.load(input + left, mask=mask_l, other=-float("inf"))
    b = tl.load(input + right, mask=mask_r, other=-float("inf"))

    arg_a = tl.load(arguments + left, mask=mask_l, other=0)
    arg_b = tl.load(arguments + right, mask=mask_r, other=0)

    swap = b > a

    tl.store(input + left, tl.where(swap, b, a), mask=mask_l)
    tl.store(input + right, tl.where(swap, a, b), mask=mask_r)
    tl.store(arguments + left, tl.where(swap, arg_b, arg_a), mask=mask_l)
    tl.store(arguments + right, tl.where(swap, arg_a, arg_b), mask=mask_r)


@triton.jit
def sum_and_block_sum(data, output, n, sum_blocks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(data + offs, mask=mask, other=0.0)
    tl.store(sum_blocks + pid, tl.sum(x))
    tl.store(output + offs, tl.cumsum(x), mask=mask)

@triton.jit
def prefix_blocks_sum(output, n, sum_block, n_blocks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    local = tl.load(output + offs, mask=mask)
    acc = 0.0
    for i in range(pid):
        acc += tl.load(sum_block + i)
    tl.store(output + offs, local + acc, mask=mask)

@triton.jit
def find_first_last_token(cumsum, n, p, last_idx_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(cumsum + offs, mask=mask, other=0.0)
    idx = tl.where(x >= p, offs, n - 1)
    tl.atomic_min(last_idx_ptr, tl.min(idx))

@triton.jit
def compute_denum(prob, last_idx_ptr, denum_ptr, BLOCK_SIZE: tl.constexpr):
    last_idx = tl.load(last_idx_ptr)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs <= last_idx
    x = tl.load(prob + offs, mask=mask, other=0.0)
    tl.atomic_add(denum_ptr, tl.sum(x))

@triton.jit
def renormalize(prob, last_idx_ptr, n, denum_ptr, BLOCK_SIZE: tl.constexpr):
    last_idx = tl.load(last_idx_ptr)
    denum = tl.load(denum_ptr)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    valid = offs <= last_idx
    invalid = (offs > last_idx) & (offs < n)

    x = tl.load(prob + offs, mask=valid, other=0.0)
    tl.store(prob + offs, x / denum, mask=valid)
    tl.store(prob + offs, 0.0, mask=invalid)


def solve(
    logits: torch.Tensor,
    p: torch.Tensor,
    seed: torch.Tensor,
    sampled_token: torch.Tensor,
    vocab_size: int,
):
    BLOCK = 128

    softmax_max = torch.full((1,), -float("inf"), device=logits.device)
    softmax_denum = torch.zeros((1,), device=logits.device)
    softmax = torch.empty_like(logits)
    grid = (triton.cdiv(vocab_size, BLOCK),)

    compute_softmax_max[grid](logits, softmax_max, vocab_size, BLOCK_SIZE=BLOCK)
    compute_softmax_denum[grid](logits, softmax_max, softmax_denum, vocab_size, BLOCK_SIZE=BLOCK)
    compute_softmax[grid](logits, softmax_max, softmax_denum, softmax, vocab_size, BLOCK_SIZE=BLOCK)

    arguments = torch.arange(vocab_size, device=logits.device)
    soft_probs = softmax.clone()
    padded = triton.next_power_of_2(vocab_size)
    grid_sort = (triton.cdiv(padded // 2, BLOCK),)

    for s in range(int(math.log2(padded))):
        st_1_sort[grid_sort](soft_probs, arguments, s, vocab_size, BLOCK)
        for t in range(s):
            st_2_sort_splitter[grid_sort](
                soft_probs, arguments, s - t - 1, vocab_size, BLOCK
            )

    BLOCK_CUM = 1024
    grid_cum = (triton.cdiv(vocab_size, BLOCK_CUM),)
    cumsum = torch.zeros_like(soft_probs)
    sum_blocks = torch.zeros(grid_cum, device=logits.device)

    sum_and_block_sum[grid_cum](soft_probs, cumsum, vocab_size, sum_blocks, BLOCK_CUM)
    prefix_blocks_sum[grid_cum](cumsum, vocab_size, sum_blocks, grid_cum[0], BLOCK_CUM)

    last_idx = torch.full((), vocab_size - 1, dtype=torch.int32, device=logits.device)
    find_first_last_token[grid_cum](cumsum, vocab_size, p.item(), last_idx, BLOCK_CUM)

    denum = torch.zeros((), device=logits.device)
    grid = (triton.cdiv(vocab_size, BLOCK),)
    compute_denum[grid](soft_probs, last_idx, denum, BLOCK)
    renormalize[grid](soft_probs, last_idx, vocab_size, denum, BLOCK)

    torch.manual_seed(int(seed.item()))
    token = torch.multinomial(soft_probs, 1)
    sampled_token.copy_(arguments[token])
