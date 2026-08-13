import torch
import triton
import triton.language as tl

# Kernel 1: 计算预测概率 P 和 Hessian 权重 W
@triton.jit
def cal_p_w(X, beta, n_samples, n_features, P, W, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offset_m < n_samples

    # 累加 z = X @ beta
    sum_z = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for step in range(0, n_features, BLOCK_N):
        offset_n = step + tl.arange(0, BLOCK_N)
        mask_n = offset_n < n_features

        vals_x = tl.load(X + offset_m[:, None] * n_features + offset_n[None, :], 
                         mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        vals_beta = tl.load(beta + offset_n, mask=mask_n, other=0.0)

        sum_z += tl.sum(vals_x * vals_beta, axis=1)

    # Sigmoid
    p = 1.0 / (1.0 + tl.exp(-sum_z))
    tl.store(P + offset_m, p, mask=mask_m)
    
    # 计算 W = p * (1 - p)，并施加 1e-8 的下界防止除零
    tl.store(W + offset_m, tl.maximum(p * (1.0 - p), 1e-8), mask=mask_m)


# Kernel 2: 计算一阶梯度 (包含 1e-6 的 L2 正则化惩罚)
@triton.jit
def cal_gradient(X, y, beta, n_samples, n_features, P, gradient, BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    mask_n = offset_n < n_features

    sum_grad = tl.zeros((BLOCK_N2,), dtype=tl.float32)

    for step in range(0, n_samples, BLOCK_M2):
        offset_m = step + tl.arange(0, BLOCK_M2)
        mask_m = offset_m < n_samples

        vals_x = tl.load(X + offset_n[:, None] + offset_m[None, :] * n_features, 
                         mask=(mask_n[:, None] & mask_m[None, :]), other=0.0)
        vals_p = tl.load(P + offset_m, mask=mask_m, other=0.0)
        vals_y = tl.load(y + offset_m, mask=mask_m, other=0.0)

        # X^T @ (P - Y)
        sum_grad += tl.sum(vals_x * (vals_p - vals_y), axis=1)

    # 加上隐藏的 L2 正则化项：1e-6 * beta
    vals_beta = tl.load(beta + offset_n, mask=mask_n, other=0.0)
    sum_grad += vals_beta * 1e-6
    
    tl.store(gradient + offset_n, sum_grad, mask=mask_n)


# Kernel 3: 计算二阶 Hessian 矩阵 (包含 1e-6 的对角线阻尼)
@triton.jit
def cal_hessian(X, n_samples, n_features, W, hessian, BLOCK_M3: tl.constexpr, BLOCK_N3: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offset_row = pid_row * BLOCK_N3 + tl.arange(0, BLOCK_N3)
    offset_col = pid_col * BLOCK_N3 + tl.arange(0, BLOCK_N3)
    mask_row = offset_row < n_features
    mask_col = offset_col < n_features

    sum_h = tl.zeros((BLOCK_N3, BLOCK_N3), dtype=tl.float32)

    for step in range(0, n_samples, BLOCK_M3):
        offset = step + tl.arange(0, BLOCK_M3)
        mask = offset < n_samples
        
        vals_xt = tl.load(X + offset_row[:, None] + offset[None, :] * n_features, 
                          mask=(mask_row[:, None] & mask[None, :]), other=0.0)
        vals_x = tl.load(X + offset[:, None] * n_features + offset_col[None, :], 
                         mask=(mask[:, None] & mask_col[None, :]), other=0.0)
        vals_w = tl.load(W + offset, mask=mask, other=0.0)
        
        # X^T @ W @ X
        sum_h += tl.dot(vals_xt, vals_x * vals_w[:, None])

    # 加上隐藏的 L2 正则化项 (对角线加 1e-6)
    sum_h += tl.where(offset_row[:, None] == offset_col[None, :], 1e-6, 0.0)
    
    tl.store(hessian + offset_row[:, None] * n_features + offset_col[None, :], 
             sum_h, mask=(mask_row[:, None] & mask_col[None, :]))


def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    # 初始化
    beta.zero_()
    tol = 1e-8
    
    for _ in range(15): # 牛顿法通常 10 次以内即可收敛
        # --- 1. 计算 P 和 W ---
        BLOCK_M1, BLOCK_N1 = 32, 32
        grid1 = (triton.cdiv(n_samples, BLOCK_M1),)
        P = torch.zeros((n_samples), device=X.device, dtype=torch.float32)
        W = torch.zeros((n_samples), device=X.device, dtype=torch.float32)
        cal_p_w[grid1](X, beta, n_samples, n_features, P, W, BLOCK_M1, BLOCK_N1)

        # --- 2. 计算 Gradient ---
        BLOCK_M2, BLOCK_N2 = 32, 32
        grid2 = (triton.cdiv(n_features, BLOCK_N2),)
        gradient = torch.zeros((n_features), device=X.device, dtype=torch.float32)
        cal_gradient[grid2](X, y, beta, n_samples, n_features, P, gradient, BLOCK_M2, BLOCK_N2)

        # --- 3. 计算 Hessian ---
        BLOCK_M3, BLOCK_N3 = 32, 32
        grid3 = (triton.cdiv(n_features, BLOCK_N3), triton.cdiv(n_features, BLOCK_N3))
        hessian = torch.zeros((n_features, n_features), device=X.device, dtype=torch.float32)
        cal_hessian[grid3](X, n_samples, n_features, W, hessian, BLOCK_M3, BLOCK_N3)

        # --- 4. 求解牛顿步长 ---
        try:
            delta = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            # Fallback：处理奇异矩阵的最小二乘解
            delta = torch.linalg.lstsq(hessian, gradient.unsqueeze(1)).solution.squeeze()

        beta_new = beta - delta

        # 检查收敛
        if torch.norm(beta_new - beta) < tol:
            beta.copy_(beta_new)
            break

        # 原位更新 beta
        beta.copy_(beta_new)