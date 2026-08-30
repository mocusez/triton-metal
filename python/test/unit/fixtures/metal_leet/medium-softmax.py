import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))

    m_i = -float('inf')
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input + offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        m_i = tl.maximum(m_i, block_max)

    l_i = 0.0
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input + offsets, mask=mask, other=-float('inf'))
        numerator = tl.exp(x - m_i)
        l_i += tl.sum(numerator, axis=0)

    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input + offsets, mask=mask, other=-float('inf'))
        numerator = tl.exp(x - m_i)
        res = numerator / l_i

        tl.store(output + offsets, res, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    softmax_kernel[(1,)](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
