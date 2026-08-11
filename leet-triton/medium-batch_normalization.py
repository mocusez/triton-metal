import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    input_ptr, gamma_ptr, beta_ptr,
    output_ptr,
    N, C,
    eps,                                
    BC: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)

    mean = tl.zeros((BC,), dtype=tl.float32)
    for n_start in range(0, N, BN):
        off_n = n_start + tl.arange(0, BN)
        mask_n = off_n < N
        off_c = pid * BC + tl.arange(0, BC)
        mask_c = off_c < C
        input_vals = tl.load(input_ptr + off_n[:, None] * C + off_c[None, :], mask=mask_n[:, None] & mask_c[None, :], other=0.0)
        mean += tl.sum(input_vals, axis=0)
    mean /= N

    var = tl.zeros((BC,), dtype=tl.float32)
    for n_start in range(0, N, BN):
        off_n = n_start + tl.arange(0, BN)
        mask_n = off_n < N
        off_c = pid * BC + tl.arange(0, BC)
        mask_c = off_c < C
        input_vals = tl.load(input_ptr + off_n[:, None] * C + off_c[None, :], mask=mask_n[:, None] & mask_c[None, :], other=0.0)
        input_vals -= mean[None, :]
        input_vals = tl.where(mask_n[:, None] & mask_c[None, :], input_vals, 0.0)
        var += tl.sum(input_vals * input_vals, axis=0)
    var /= N
    
    inv_std_var = 1 / tl.sqrt(var + eps)

    off_c = pid * BC + tl.arange(0, BC)
    mask_c = off_c < C
    
    gamma = tl.load(gamma_ptr + off_c, mask=mask_c, other=0.0)
    beta = tl.load(beta_ptr + off_c, mask=mask_c, other=0.0)
    
    for n_start in range(0, N, BN):
        off_n = n_start + tl.arange(0, BN)
        mask_n = off_n < N
        input_vals = tl.load(input_ptr + off_n[:, None] * C + off_c[None, :], mask=mask_n[:, None] & mask_c[None, :], other=0.0)
        input_vals -= mean[None, :]
        input_vals *= inv_std_var[None, :]
        input_vals = tl.where(mask_n[:, None] & mask_c[None, :], input_vals, 0.0)
        
        output = gamma[None, :] * input_vals + beta[None, :]
        tl.store(output_ptr + off_n[:, None] * C + off_c[None, :], output, mask=mask_n[:, None] & mask_c[None, :])


# input, gamma, beta, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    BN = 16
    BC = 16
    grid = (triton.cdiv(C, BC),)
    
    kernel[grid](
        input, gamma, beta,
        output,
        N, C, eps,
        BC=BC, BN=BN 
    )