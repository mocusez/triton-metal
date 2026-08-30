import torch
import triton
import triton.language as tl

@triton.jit
def nearest_neighbor_3d(points_ptr, indices_ptr,
        N,
        BLOCK_SIZE_N: tl.constexpr,
        TILE_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offset_N = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_N = offset_N < N

    x_offset = offset_N * 3
    y_offset = offset_N * 3 + 1
    z_offset = offset_N * 3 + 2

    main_x = tl.load(points_ptr + x_offset, mask = mask_N, other = 0.0)
    main_y = tl.load(points_ptr + y_offset, mask = mask_N, other = 0.0)
    main_z = tl.load(points_ptr + z_offset, mask = mask_N, other = 0.0)

    distances = tl.full([BLOCK_SIZE_N], value = float("inf"), dtype = tl.float32)
    indices = tl.zeros([BLOCK_SIZE_N], dtype = tl.int32)

    tile_offset = tl.arange(0, TILE_SIZE)

    for tile_start in range(0, N, TILE_SIZE):
        tile_indices = tile_start + tile_offset
        tile_mask = tile_indices < N

        tile_x_offset = tile_indices * 3
        tile_y_offset = tile_indices * 3 + 1
        tile_z_offset = tile_indices * 3 + 2

        tile_x = tl.load(points_ptr + tile_x_offset, mask = tile_mask, other = 0.0)
        tile_y = tl.load(points_ptr + tile_y_offset, mask = tile_mask, other = 0.0)
        tile_z = tl.load(points_ptr + tile_z_offset, mask = tile_mask, other = 0.0)

        dx = main_x[:, None] - tile_x[None, :]
        dy = main_y[:, None] - tile_y[None, :]
        dz = main_z[:, None] - tile_z[None, :]

        squared_distances = dx * dx + dy * dy + dz * dz

        self_mask = offset_N[:, None] == tile_indices[None, :]
        valid_mask = tile_mask[None, :] & ~self_mask

        masked_distances = tl.where(valid_mask, squared_distances, float("inf"))

        min_distances = tl.min(masked_distances, axis = 1)
        min_tile_indices = tl.argmin(masked_distances, axis = 1)

        replacement_mask = min_distances < distances

        distances = tl.where(replacement_mask, min_distances, distances)
        indices = tl.where(replacement_mask, tile_start + min_tile_indices, indices)

    tl.store(indices_ptr + offset_N, indices, mask = mask_N)


# points and indices are tensors on the GPU
def solve(points: torch.Tensor, indices: torch.Tensor, N: int):
    BLOCK_SIZE_N = 16
    TILE_SIZE = 1024

    grid = (triton.cdiv(N, BLOCK_SIZE_N),)
    nearest_neighbor_3d[grid](points, indices, N, BLOCK_SIZE_N, TILE_SIZE)
