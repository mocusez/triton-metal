import torch
import triton
import triton.language as tl


@triton.jit
def mse_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    target_vals = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    diff = input_vals - target_vals
    squared_diff = diff * diff

    block_sum = tl.sum(squared_diff)
    tl.atomic_add(output_ptr, block_sum)


# predictions, targets, mse are tensors on the GPU
def solve(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mse: torch.Tensor,
    N: int,
):
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if predictions.numel() < N or targets.numel() < N:
        raise ValueError(
            "predictions and targets must each contain at least N elements"
        )
    if mse.numel() != 1:
        raise ValueError(f"mse must contain exactly one element, got {mse.numel()}")
    if not (
        predictions.device == targets.device == mse.device
        and predictions.dtype == targets.dtype == mse.dtype == torch.float32
    ):
        raise ValueError(
            "predictions, targets, and mse must be float32 tensors on one device"
        )

    input_flat = predictions.contiguous().view(-1)
    target_flat = targets.contiguous().view(-1)

    BLOCK_SIZE = triton.next_power_of_2(min(N, 1024))
    num_blocks = triton.cdiv(N, BLOCK_SIZE)

    # The kernel accumulates partial sums atomically, so make repeated calls
    # safe even when the caller reuses the output tensor.
    mse.zero_()
    mse_kernel[(num_blocks,)](
        input_flat,
        target_flat,
        mse,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    mse.div_(N)


if __name__ == "__main__":
    import sys

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("No GPU device (MPS or CUDA) available")

    torch.manual_seed(0xC0FFEE)
    sizes = (1, 31, 100, 1024, 1025, 4097)
    mse = torch.full((1,), 123.0, dtype=torch.float32, device=device)
    for N in sizes:
        predictions = torch.randn(N, dtype=torch.float32, device=device)
        targets = torch.randn(N, dtype=torch.float32, device=device)

        solve(predictions, targets, mse, N)

        expected = torch.mean((predictions - targets) ** 2).reshape_as(mse)
        torch.testing.assert_close(mse, expected, atol=1e-5, rtol=1e-5)

    print(f"PASS [{device}] mean_squared_error N={list(sizes)}")
