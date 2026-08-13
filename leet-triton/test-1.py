import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,      # 输入向量 X 的指针
    y_ptr,      # 输入向量 Y 的指针
    output_ptr, # 输出向量的指针
    n_elements, # 元素总数
    BLOCK_SIZE: tl.constexpr, # 每个程序块处理的元素数量（编译期常量）
):
    # 获取当前程序块的 ID (axis=0 表示一维网格)
    pid = tl.program_id(axis=0)
    
    # 计算当前块处理的元素索引范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 边界保护掩码，防止越界访问
    mask = offsets < n_elements

    # 从 DRAM 加载数据到 SRAM / 寄存器
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 执行并执行并行的向量加法
    output = x + y

    # 将结果写回 DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

# 2. 封装前端调用函数
def triton_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    
    n_elements = output.numel()
    
    # 定义执行网格 (Grid)，计算需要启动多少个程序块
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动核函数
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# 3. 测试与验证
if __name__ == "__main__":
    size = 98432
    x = torch.rand(size)
    y = torch.rand(size)
    
    # PyTorch 原生计算作为基准 (Baseline)
    output_torch = x + y
    
    # 调用我们编写的 Triton 自定义算子
    output_triton = triton_add(x, y)
    
    # 验证结果绝对误差
    max_diff = torch.max(torch.abs(output_torch - output_triton))
    print(f"最大绝对误差: {max_diff.item()}")
    if max_diff < 1e-5:
        print("测试成功：Triton 计算结果与 PyTorch 完全一致！")