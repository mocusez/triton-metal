import torch
import triton
import triton.language as tl

# =====================================================================
# Kernel 1: 统计每个 Block 中 > 0 的元素个数
# =====================================================================
@triton.jit
def count_kernel(A_ptr, Count_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 加载当前 Block 的数据，越界部分补 0.0
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # 筛选条件：A[i] > 0
    valid = a > 0.0

    # 统计当前 Block 内有效元素的总数 (布尔值转为 int32 并求和)
    block_count = tl.sum(valid.to(tl.int32), axis=0)

    # 将统计结果存入全局数组
    tl.store(Count_ptr + pid, block_count)

# =====================================================================
# Kernel 2: 计算局部偏移，并把数据写入到输出数组 (Scatter)
# =====================================================================
@triton.jit
def compact_kernel(A_ptr, Out_ptr, BlockOffsets_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 1. 重新加载当前 Block 的数据
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    valid = a > 0.0
    valid_int = valid.to(tl.int32)

    # 2. 计算块内局部前缀和 (Local Exclusive Prefix Sum)
    # tl.cumsum 是包含自身的（Inclusive），我们需要减去当前元素，变成排他性（Exclusive）
    local_inc_sum = tl.cumsum(valid_int, axis=0)
    local_offsets = local_inc_sum - valid_int

    # 3. 读取该 Block 在全局输出数组中的起始偏移量
    global_offset = tl.load(BlockOffsets_ptr + pid)

    # 4. 计算最终的写入位置: 全局起始位置 + 局部偏移量
    write_offsets = global_offset + local_offsets

    # 5. 写入数据到 out 数组 (仅当 valid 为 True 且没越界时才写入)
    tl.store(Out_ptr + write_offsets, a, mask=valid & mask)

# =====================================================================
# Host 端代码：调度与前缀和计算
# =====================================================================
def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    # 要求：不需要的空位必须是 0.0。直接清零整个输出数组最安全高效。
    out.zero_()

    if N == 0:
        return

    # 定义块大小
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)

    # 分配内存用于存放每个 Block 的计数结果
    block_counts = torch.empty(num_blocks, dtype=torch.int32, device=A.device)

    # ---------------------------------------------------------
    # 步骤 1: Launch Count Kernel
    # ---------------------------------------------------------
    count_kernel[(num_blocks,)](A, block_counts, N, BLOCK_SIZE=BLOCK_SIZE)

    # ---------------------------------------------------------
    # 步骤 2: 全局排他性前缀和 (Global Exclusive Scan)
    # 利用 PyTorch 的原生 GPU 算子计算偏移量
    # 例如：counts = [2, 1, 3] -> offsets = [0, 2, 3]
    # ---------------------------------------------------------
    block_offsets = torch.empty(num_blocks, dtype=torch.int32, device=A.device)
    block_offsets[0] = 0
    if num_blocks > 1:
        block_offsets[1:] = torch.cumsum(block_counts[:-1], dim=0)

    # ---------------------------------------------------------
    # 步骤 3: Launch Compact Kernel
    # ---------------------------------------------------------
    compact_kernel[(num_blocks,)](A, out, block_offsets, N, BLOCK_SIZE=BLOCK_SIZE)


if __name__ == "__main__":
    import sys

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    N = 4113
    input_cpu = (torch.arange(N, dtype=torch.float32) % 17) - 8
    input = input_cpu.to(device)
    output = torch.empty_like(input)

    solve(input, N, output)

    expected = torch.zeros_like(input_cpu)
    kept = input_cpu[input_cpu > 0]
    expected[: kept.numel()] = kept
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    print(f"PASS [{device}] stream_compaction N={N} kept={kept.numel()}")
