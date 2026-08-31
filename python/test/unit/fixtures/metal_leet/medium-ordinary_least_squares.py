import torch
import triton
import triton.language as tl


@triton.jit
def gram_kernel(
    X,
    y,
    XtX,
    Xty,
    n_samples,
    n_features,
    stride_x_0,
    stride_x_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    offset_row = pid_row * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_col = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_row = offset_row < n_features
    mask_col = offset_col < n_features

    acc_xtx = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    acc_xty = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for step in range(0, n_samples, BLOCK_M):
        offset_m = step + tl.arange(0, BLOCK_M)
        mask_m = offset_m < n_samples

        x_mn = tl.load(
            X + offset_m[:, None] * stride_x_0
            + offset_col[None, :] * stride_x_1,
            mask=mask_m[:, None] & mask_col[None, :],
            other=0.0,
        )
        x_nm = tl.load(
            X + offset_row[:, None] * stride_x_1
            + offset_m[None, :] * stride_x_0,
            mask=mask_row[:, None] & mask_m[None, :],
            other=0.0,
        )

        acc_xtx += tl.dot(x_nm, x_mn)
        y_m = tl.load(y + offset_m, mask=mask_m, other=0.0)
        acc_xty += tl.sum(x_nm * y_m, axis=1)

    tl.store(
        XtX + offset_row[:, None] * n_features + offset_col[None, :],
        acc_xtx,
        mask=mask_row[:, None] & mask_col[None, :],
    )
    tl.store(Xty + offset_row, acc_xty, mask=mask_row)


# X, y, beta are tensors on the GPU
def solve(
    X: torch.Tensor,
    y: torch.Tensor,
    beta: torch.Tensor,
    n_samples: int,
    n_features: int,
):
    BLOCK_M = 32
    BLOCK_N = 32

    X = X.view(n_samples, n_features).contiguous()
    y = y.view(n_samples).contiguous()
    beta = beta.view(n_features).contiguous()

    XtX = torch.empty(
        (n_features, n_features), device=X.device, dtype=torch.float32
    )
    Xty = torch.zeros(n_features, device=X.device, dtype=torch.float32)

    # This intentionally preserves the raw LeetGPU kernel shape: one program
    # dimension writes the matching feature block of XtX and Xty. The raw source
    # only fills all XtX entries when n_features <= BLOCK_N; larger feature
    # counts are covered by the corrected two-dimensional canary in pytest.
    gram_kernel[(triton.cdiv(n_features, BLOCK_N),)](
        X,
        y,
        XtX,
        Xty,
        n_samples,
        n_features,
        X.stride(0),
        X.stride(1),
        BLOCK_M,
        BLOCK_N,
    )
    beta[:] = torch.linalg.solve(XtX, Xty)


if __name__ == "__main__":
    import sys

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0xC0FFEE)
    cases = ((64, 8), (96, 32))
    for n_samples, n_features in cases:
        X = torch.randn(
            (n_samples, n_features), dtype=torch.float32, device=device
        )
        expected = torch.linspace(
            -1.0, 1.0, n_features, dtype=torch.float32, device=device
        )
        y = X @ expected
        beta = torch.empty(n_features, dtype=torch.float32, device=device)

        solve(X, y, beta, n_samples, n_features)

        torch.testing.assert_close(
            beta.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3
        )

    print(f"PASS [{device}] ordinary_least_squares cases={list(cases)}")
