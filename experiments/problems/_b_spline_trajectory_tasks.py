import re
from collections import deque
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


__all__ = [
    "build_b_spline_trajectory_cache_file",
    "generate_b_spline_trajectory_task_descriptors",
]


def build_b_spline_trajectory_cache_file(
    save_path: str | Path,
    class_name: str,
    num_source_tasks: int,
    dim: int,
    num_evaluation_points: int,
    seed: Optional[int],
) -> Path:
    """Build the class-specific path for one cached trajectory problem."""
    directory_name = re.sub(
        r"(?<!^)(?<!_)(?=[A-Z])",
        "_",
        class_name,
    ).lower()
    cache_directory = Path(save_path).expanduser() / directory_name
    cache_directory.mkdir(parents=True, exist_ok=True)

    return cache_directory / f"K{num_source_tasks}_D{dim}_R{num_evaluation_points}_seed{seed}.pt"


def _has_monotone_grid_path(obstacle_centers: Tensor, obstacle_size: float, grid_size: int) -> bool:
    """Check for a coarse grid path that never moves backwards in x."""
    grid_coordinates = torch.linspace(0.0, 1.0, grid_size, dtype=obstacle_centers.dtype, device="cpu")
    # The first grid axis represents x and the second represents y
    occupied = torch.zeros((grid_size, grid_size), dtype=torch.bool, device="cpu")
    half_size = obstacle_size / 2.0

    for center_x, center_y in obstacle_centers.tolist():
        occupied_x = torch.abs(grid_coordinates - center_x) <= half_size
        occupied_y = torch.abs(grid_coordinates - center_y) <= half_size
        occupied |= occupied_x.unsqueeze(1) & occupied_y.unsqueeze(0) # (G, 1) & (1, G) -> (G, G)

    occupied_grid = occupied.tolist()
    # Ensure the start and goal grid cells are unoccupied by obstacles
    if occupied_grid[0][0] or occupied_grid[-1][-1]:
        return False

    # Use BFS to check for a monotone path from the start to the goal grid cell
    visited = [bytearray(grid_size) for _ in range(grid_size)]
    visited[0][0] = 1
    queue = deque([(0, 0)])
    while queue:
        x_index, y_index = queue.popleft()
        if x_index == grid_size - 1 and y_index == grid_size - 1: # reached the goal
            return True

        neighbors = ( # only allow monotone-x moves, but can move up or down in y
            (x_index + 1, y_index),
            (x_index, y_index + 1),
            (x_index, y_index - 1),
        )
        for next_x, next_y in neighbors:
            if not (0 <= next_x < grid_size and 0 <= next_y < grid_size):
                continue
            if occupied_grid[next_x][next_y] or visited[next_x][next_y]:
                continue

            visited[next_x][next_y] = 1
            queue.append((next_x, next_y))

    return False


def _is_valid_obstacle_layout(
    obstacle_centers: Tensor,
    obstacle_size: float,
    endpoint_clearance: float,
    obstacle_gap: float,
    feasibility_grid_size: int,
) -> bool:
    """Check geometry, direct-path blocking, and coarse path feasibility."""
    half_size = obstacle_size / 2.0
    if torch.any(obstacle_centers < half_size).item(): # ensure all obstacles are fully inside the map
        return False
    if torch.any(obstacle_centers > 1.0 - half_size).item(): # same as above, but for the other side of the map
        return False

    # Ensure all obstacles are sufficiently far from the start and goal endpoints
    endpoint_half_extent = half_size + endpoint_clearance
    near_start = torch.all(torch.abs(obstacle_centers) <= endpoint_half_extent, dim=1)
    near_goal = torch.all(torch.abs(obstacle_centers - 1.0) <= endpoint_half_extent, dim=1)
    if torch.any(near_start | near_goal).item():
        return False

    #
    # Ensure the direct path from start to goal (i.e., the line y = x) is blocked by at least one obstacle.
    # That is, a point (t, t) on the line y = x must be within inteval [c_x - h, c_x + h] \times [c_y - h, c_y + h]
    # In other words, a scalar t must be within intevals [c_x - h, c_x + h] and [c_y - h, c_y + h] at the same time,
    # which is equivalent to there being intersection between the two intevals, inducing |c_x - c_y| < 2h.
    #
    direct_path_blocked = torch.any(
        torch.abs(obstacle_centers[:, 0] - obstacle_centers[:, 1]) < obstacle_size
    ).item()
    if not direct_path_blocked:
        return False

    # Ensure all obstacles are sufficiently far from each other to avoid overlap
    center_distances = torch.abs(obstacle_centers.unsqueeze(1) - obstacle_centers.unsqueeze(0)) # (M,1,2)-(1,M,2)->(M,M,2)
    minimum_center_distance = obstacle_size + obstacle_gap
    obstacles_too_close = (
        (center_distances[..., 0] < minimum_center_distance)
        & (center_distances[..., 1] < minimum_center_distance)
    )
    if torch.any(torch.triu(obstacles_too_close, diagonal=1)).item():
        return False

    return _has_monotone_grid_path(
        obstacle_centers,
        obstacle_size,
        feasibility_grid_size,
    )


def generate_b_spline_trajectory_task_descriptors(
    num_tasks: int,
    num_obstacles: int | tuple[int, int] = 20,
    obstacle_size: float = 0.05,
    endpoint_clearance: float = 0.05,
    obstacle_gap: float = 0.005,
    feasibility_grid_size: int = 256,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[str | torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Generate square-obstacle layouts for B-spline trajectory tasks.

    Each task is defined by square obstacles on the normalized [0, 1]^2 map,
    with the start and goal fixed at (0, 0) and (1, 1). Obstacle-center
    candidates come from a scrambled Sobol sequence. In variable-obstacle-count mode,
    a balanced schedule reproducibly fixes the obstacle-count quotas before sampling.

    Every returned layout keeps its obstacles inside the map, preserves
    clearance around both endpoints, prevents obstacle overlap, blocks the
    direct start-to-goal line, and has a coarse monotone-x grid path. The grid
    check only filters clearly blocked layouts; it does not guarantee that a
    downstream B-spline optimizer will find a feasible trajectory.

    Returns
    -------
    obstacle_centers:
        Tensor of shape [num_tasks, max_obstacles, 2].
        Each row stores active obstacle centers first, followed by padding
        centers when the number of obstacles varies.

    obstacle_mask:
        Boolean tensor of shape [num_tasks, max_obstacles].
        True entries indicate active obstacles; False entries mark padding.
    """
    assert (
        isinstance(num_tasks, int)
        and not isinstance(num_tasks, bool)
        and num_tasks > 0
    )
    assert 0.0 < obstacle_size < 1.0
    assert endpoint_clearance >= 0.0
    assert obstacle_gap >= 0.0
    assert (
        isinstance(feasibility_grid_size, int)
        and not isinstance(feasibility_grid_size, bool)
        and feasibility_grid_size >= 2 # at least cover the start and goal grid cells
    )
    #
    # Ensure all path across the grid not bridge any obstacle.
    # In other words, an obstacle must cover at least one grid cell.
    #
    grid_spacing = 1.0 / (feasibility_grid_size - 1)
    assert grid_spacing <= obstacle_size
    assert dtype in (torch.float32, torch.float64)
    assert seed is None or (
        isinstance(seed, int)
        and not isinstance(seed, bool)
        and seed >= 0
    )

    half_size = obstacle_size / 2.0
    assert obstacle_size + obstacle_gap < 1.0
    assert half_size + endpoint_clearance < 1.0 - half_size

    variable_count = isinstance(num_obstacles, tuple)
    if variable_count:
        assert len(num_obstacles) == 2
        min_obstacles, max_obstacles = num_obstacles
        assert isinstance(min_obstacles, int) and not isinstance(min_obstacles, bool)
        assert isinstance(max_obstacles, int) and not isinstance(max_obstacles, bool)
        assert 0 < min_obstacles <= max_obstacles
    else:
        assert (
            isinstance(num_obstacles, int)
            and not isinstance(num_obstacles, bool)
            and num_obstacles > 0
        )
        min_obstacles = max_obstacles = num_obstacles

    device = torch.get_default_device() if device is None else torch.device(device)
    effective_seed = (
        torch.randint(0, 2**31 - 1, (), device="cpu").item()
        if seed is None
        else seed
    )

    #
    # Preassign a balanced obstacle-count schedule before sampling layouts.
    # Each count in [min_obstacles, max_obstacles] receives either floor(K / L)
    # or ceil(K / L) tasks, where L is the number of available count levels.
    # Rejected layouts are resampled for the same requested count, preventing
    # harder high-obstacle layouts from being underrepresented.
    #
    if variable_count:
        num_obstacle_selections = max_obstacles - min_obstacles + 1
        base_count, remainder = divmod(num_tasks, num_obstacle_selections)
        per_selection_repeats = torch.full((num_obstacle_selections,), base_count, dtype=torch.long, device="cpu")
        count_generator = torch.Generator(device="cpu")
        count_generator.manual_seed(effective_seed)
        extra_levels = torch.randperm(num_obstacle_selections, generator=count_generator, device="cpu")[:remainder]
        per_selection_repeats[extra_levels] += 1

        obstacle_count_selections = torch.arange(min_obstacles, max_obstacles + 1, dtype=torch.long, device="cpu")
        # Expand per-count quotas into one requested obstacle count for each task, then shuffle the task order 
        desired_obstacle_counts = torch.repeat_interleave(input=obstacle_count_selections, repeats=per_selection_repeats)
        task_order = torch.randperm(num_tasks, generator=count_generator, device="cpu")
        desired_obstacle_counts = desired_obstacle_counts[task_order]
    else:
        desired_obstacle_counts = torch.full((num_tasks,), max_obstacles, dtype=torch.long, device="cpu")

    engine = torch.quasirandom.SobolEngine(
        dimension=2 * max_obstacles,
        scramble=True,
        seed=effective_seed,
    )
    max_sampling_attempts = max(10_000, 1_000 * num_tasks)
    accepted_centers: list[Tensor] = []
    accepted_counts: list[int] = []
    sampling_attempts = 0

    while len(accepted_centers) < num_tasks and sampling_attempts < max_sampling_attempts:
        obstacle_count = int(desired_obstacle_counts[len(accepted_centers)].item())
        sample = engine.draw(1, dtype=dtype)[0]
        sampling_attempts += 1
        raw_centers = sample.reshape(max_obstacles, 2)
        obstacle_centers = half_size + (1.0 - obstacle_size) * raw_centers # map to [half_size, 1 - half_size]^2
        active_centers = obstacle_centers[:obstacle_count]

        if not _is_valid_obstacle_layout(
            active_centers,
            obstacle_size,
            endpoint_clearance,
            obstacle_gap,
            feasibility_grid_size,
        ):
            continue

        accepted_centers.append(obstacle_centers)
        accepted_counts.append(obstacle_count)

    if len(accepted_centers) < num_tasks:
        next_obstacle_count = int(desired_obstacle_counts[len(accepted_centers)].item())
        raise RuntimeError(
            f"Generated only {len(accepted_centers)} of {num_tasks} valid layouts after "
            f"{max_sampling_attempts} sampling attempts; reduce the obstacle "
            f"count or relax the layout constraints. The next requested layout "
            f"contains {next_obstacle_count} obstacles."
        )

    obstacle_centers = torch.stack(accepted_centers)
    obstacle_counts = torch.tensor(accepted_counts, dtype=torch.long, device="cpu")
    obstacle_mask = (
        torch.arange(max_obstacles, device="cpu").unsqueeze(0) < obstacle_counts.unsqueeze(1)
    ) # (1, M) < (K, 1) -> (K, M)

    return (obstacle_centers.to(device=device), obstacle_mask.to(device=device),)
