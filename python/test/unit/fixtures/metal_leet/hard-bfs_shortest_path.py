import torch
import triton
import triton.language as tl


def _validate_solve_inputs(
    grid: torch.Tensor,
    result: torch.Tensor,
    rows: int,
    cols: int,
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
):
    if not isinstance(grid, torch.Tensor):
        raise ValueError("grid must be a torch.Tensor")
    if not isinstance(result, torch.Tensor):
        raise ValueError("result must be a torch.Tensor")

    for name, value in (
        ("rows", rows),
        ("cols", cols),
        ("start_row", start_row),
        ("start_col", start_col),
        ("end_row", end_row),
        ("end_col", end_col),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if grid.ndim != 2 or tuple(grid.shape) != (rows, cols):
        raise ValueError("grid shape must exactly match (rows, cols)")
    if grid.dtype != torch.int32:
        raise ValueError("grid must have dtype torch.int32")
    if result.dtype != torch.int32:
        raise ValueError("result must have dtype torch.int32")
    if result.numel() < 1:
        raise ValueError("result must contain at least one element")
    if not result.is_contiguous():
        raise ValueError("result must be contiguous")
    if grid.device != result.device:
        raise ValueError("grid and result must be on the same device")
    if not (0 <= start_row < rows and 0 <= start_col < cols):
        raise ValueError("start coordinates must be within the grid")
    if not (0 <= end_row < rows and 0 <= end_col < cols):
        raise ValueError("end coordinates must be within the grid")

    total_cells = rows * cols
    if total_cells > torch.iinfo(torch.int32).max:
        raise ValueError("rows * cols must fit in signed int32")

    return total_cells, start_row * cols + start_col, end_row * cols + end_col


@triton.jit
def _visit_neighbor(
    neighbor,
    allowed,
    grid_ptr,
    visited_ptr,
    next_frontier_ptr,
    counters_ptr,
    end_idx,
):
    is_free = tl.load(
        grid_ptr + neighbor,
        mask=allowed,
        other=1,
    ) == 0
    candidate = allowed & is_free

    old_value = tl.atomic_xchg(
        visited_ptr + neighbor,
        1,
        mask=candidate,
    )
    claimed = candidate & (old_value == 0)
    claimed_int = claimed.to(tl.int32)

    output_pos = tl.atomic_add(
        counters_ptr + tl.zeros_like(claimed_int),
        claimed_int,
        mask=claimed,
    )

    tl.store(
        next_frontier_ptr + output_pos,
        neighbor,
        mask=claimed,
    )

    found = claimed & (neighbor == end_idx)

    tl.atomic_or(
        counters_ptr + 1 + tl.zeros_like(claimed_int),
        found.to(tl.int32),
        mask=found,
    )


@triton.jit
def _bfs_expand_kernel(
    current_frontier_ptr,
    next_frontier_ptr,
    grid_ptr,
    visited_ptr,
    counters_ptr,
    current_count,
    cols,
    rows,
    end_idx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < current_count

    node = tl.load(
        current_frontier_ptr + offsets,
        mask=active,
        other=0,
    )

    row = node // cols
    col = node - row * cols

    _visit_neighbor(
        node - cols,
        active & (row > 0),
        grid_ptr,
        visited_ptr,
        next_frontier_ptr,
        counters_ptr,
        end_idx,
    )

    _visit_neighbor(
        node + cols,
        active & (row + 1 < rows),
        grid_ptr,
        visited_ptr,
        next_frontier_ptr,
        counters_ptr,
        end_idx,
    )

    _visit_neighbor(
        node - 1,
        active & (col > 0),
        grid_ptr,
        visited_ptr,
        next_frontier_ptr,
        counters_ptr,
        end_idx,
    )

    _visit_neighbor(
        node + 1,
        active & (col + 1 < cols),
        grid_ptr,
        visited_ptr,
        next_frontier_ptr,
        counters_ptr,
        end_idx,
    )


# grid, result are tensors on the GPU
def solve(
    grid: torch.Tensor,
    result: torch.Tensor,
    rows: int,
    cols: int,
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
):
    """Return the shortest 4-neighbor path length through free (0) cells.

    The grid/result contract is checked from host-visible tensor metadata before
    launching kernels. Start and end cells are expected to be free (0); blocked
    endpoints are outside this helper's supported input contract.
    """
    total_cells, start_idx, end_idx = _validate_solve_inputs(
        grid,
        result,
        rows,
        cols,
        start_row,
        start_col,
        end_row,
        end_col,
    )
    output = result.view(-1)

    if start_idx == end_idx:
        output[0] = 0
        return 0

    if not grid.is_contiguous():
        grid = grid.contiguous()

    device = grid.device

    visited = torch.zeros(
        total_cells,
        dtype=torch.int32,
        device=device,
    )

    current_frontier = torch.empty(
        total_cells,
        dtype=torch.int32,
        device=device,
    )
    next_frontier = torch.empty(
        total_cells,
        dtype=torch.int32,
        device=device,
    )

    counters = torch.zeros(
        2,
        dtype=torch.int32,
        device=device,
    )

    visited[start_idx] = 1
    current_frontier[0] = start_idx

    current_count = 1
    depth = 0
    BLOCK_SIZE = 128

    while current_count > 0:
        launch_grid = (
            triton.cdiv(current_count, BLOCK_SIZE),
        )

        _bfs_expand_kernel[launch_grid](
            current_frontier,
            next_frontier,
            grid,
            visited,
            counters,
            current_count,
            cols,
            rows,
            end_idx,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        counters_cpu = counters.cpu()
        next_count = int(counters_cpu[0].item())
        found = int(counters_cpu[1].item())

        if found:
            answer = depth + 1
            output[0] = answer
            return answer

        if next_count == 0:
            output[0] = -1
            return -1

        counters.zero_()

        current_frontier, next_frontier = (
            next_frontier,
            current_frontier,
        )
        current_count = next_count
        depth += 1

    output[0] = -1
    return -1
