import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def kmeans_kernel(
    data_x_ptr,
    data_y_ptr,
    labels_ptr,
    initial_centroid_x_ptr,
    initial_centroid_y_ptr,
    final_centroid_x_ptr,
    final_centroid_y_ptr,
    n,
    k,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    data_x_ptr = data_x_ptr.to(tl.pointer_type(tl.float32))
    data_y_ptr = data_y_ptr.to(tl.pointer_type(tl.float32))
    labels_ptr = labels_ptr.to(tl.pointer_type(tl.int32))
    initial_centroid_x_ptr = initial_centroid_x_ptr.to(tl.pointer_type(tl.float32))
    initial_centroid_y_ptr = initial_centroid_y_ptr.to(tl.pointer_type(tl.float32))
    final_centroid_x_ptr = final_centroid_x_ptr.to(tl.pointer_type(tl.float32))
    final_centroid_y_ptr = final_centroid_y_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)

    off = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    msk = off < n

    off_k = tl.arange(0, BLOCK_K)
    msk_k = off_k < k

    center_x = tl.load(initial_centroid_x_ptr + off_k, mask = msk_k, other = 0.0)
    center_y = tl.load(initial_centroid_y_ptr + off_k, mask = msk_k, other = 0.0)

    pts_x = tl.load(data_x_ptr + off, mask = msk, other = 0.0)
    pts_y = tl.load(data_y_ptr + off, mask = msk, other = 0.0)

    dx = libdevice.pow(center_x[:, None] - pts_x[None, :], 2)
    dy = libdevice.pow(center_y[:, None] - pts_y[None, :], 2)

    d = dx + dy
    d = tl.where(msk_k[:, None], d, float("inf"))
    idx = tl.argmin(d, axis = 0)

    tl.store(labels_ptr + off, idx, mask = msk)
    tl.atomic_add(final_centroid_x_ptr + idx, pts_x, mask = msk)
    tl.atomic_add(final_centroid_y_ptr + idx, pts_y, mask = msk)

@triton.jit
def postprocess_means(
    labels_ptr,
    init_x_ptr,
    init_y_ptr,
    final_x_ptr,
    final_y_ptr,
    k,
    n,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    labels_ptr = labels_ptr.to(tl.pointer_type(tl.int32))
    init_x_ptr = init_x_ptr.to(tl.pointer_type(tl.float32))
    init_y_ptr = init_y_ptr.to(tl.pointer_type(tl.float32))
    final_x_ptr = final_x_ptr.to(tl.pointer_type(tl.float32))
    final_y_ptr = final_y_ptr.to(tl.pointer_type(tl.float32))

    n_loops = tl.ceil(n / BLOCK_SIZE).to(tl.int32)

    off = tl.arange(0, BLOCK_SIZE)
    off_k = tl.arange(0, BLOCK_K)
    msk_k = off_k < k

    counts = tl.zeros((BLOCK_K,), tl.int32)
    for i in range(n_loops):
        msk = off < n
        idx = tl.load(labels_ptr + off, mask = msk, other = -1)
        idx_count = (idx[:, None] == off_k[None, :]).to(tl.int32)
        count_up = tl.sum(idx_count, axis = 0)
        counts += count_up
        off += BLOCK_SIZE

    centroid_x = tl.load(final_x_ptr + off_k, mask = msk_k)
    centroid_y = tl.load(final_y_ptr + off_k, mask = msk_k)
    counts = counts.to(tl.float32)
    centroid_x /= counts
    centroid_y /= counts
    msk_counts = counts > 0
    tl.store(init_x_ptr + off_k, centroid_x, mask = msk_k & msk_counts)
    tl.store(init_y_ptr + off_k, centroid_y, mask = msk_k & msk_counts)
    tl.store(final_x_ptr + off_k, 0.0, mask = msk_k)
    tl.store(final_y_ptr + off_k, 0.0, mask = msk_k)

@triton.jit
def copy_kernel(
    init_x_ptr,
    init_y_ptr,
    final_x_ptr,
    final_y_ptr,
    k,
    BLOCK_K: tl.constexpr
):
    init_x_ptr = init_x_ptr.to(tl.pointer_type(tl.float32))
    init_y_ptr = init_y_ptr.to(tl.pointer_type(tl.float32))
    final_x_ptr = final_x_ptr.to(tl.pointer_type(tl.float32))
    final_y_ptr = final_y_ptr.to(tl.pointer_type(tl.float32))

    off = tl.arange(0, BLOCK_K)
    msk = off < k
    x = tl.load(init_x_ptr + off, mask = msk)
    y = tl.load(init_y_ptr + off, mask = msk)

    tl.store(final_x_ptr + off, x, mask = msk)
    tl.store(final_y_ptr + off, y, mask = msk)


# data_x, data_y, labels, initial_centroid_x,
# initial_centroid_y, final_centroid_x, final_centroid_y are tensors on the GPU
def solve(
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    labels: torch.Tensor,
    initial_centroid_x: torch.Tensor,
    initial_centroid_y: torch.Tensor,
    final_centroid_x: torch.Tensor,
    final_centroid_y: torch.Tensor,
    sample_size: int,
    k: int,
    max_iterations: int,
):
    BLOCK_SIZE = 128
    BLOCK_K = triton.next_power_of_2(k)

    grid_idx = (triton.cdiv(sample_size, BLOCK_SIZE),)

    for i in range(max_iterations):
        kmeans_kernel[grid_idx](
            data_x,
            data_y,
            labels,
            initial_centroid_x,
            initial_centroid_y,
            final_centroid_x,
            final_centroid_y,
            sample_size,
            k,
            BLOCK_SIZE,
            BLOCK_K
        )
        postprocess_means[(1,)](
            labels,
            initial_centroid_x,
            initial_centroid_y,
            final_centroid_x,
            final_centroid_y,
            k,
            sample_size,
            BLOCK_SIZE,
            BLOCK_K
        )
    copy_kernel[(1,)](
        initial_centroid_x,
        initial_centroid_y,
        final_centroid_x,
        final_centroid_y,
        k,
        BLOCK_K
    )


