import torch
import triton
import triton.language as tl

@triton.jit
def count_k_kernel(
    input_ptr,
    output_ptr,
    K,
    total_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < total_elements

    data = tl.load(input_ptr + offsets, mask=mask, other=K - 1)

    matches = (data == K)

    block_count = tl.sum(matches.to(tl.int32))

    if block_count > 0:
        tl.atomic_add(output_ptr, block_count)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    total_elements = N * M

    BLOCK_SIZE = 4096

    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

    # 6. 启动 Triton Kernel
    count_k_kernel[grid](
        input,
        output,
        K,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
