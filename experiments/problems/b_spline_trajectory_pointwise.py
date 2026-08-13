from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from ._b_spline_trajectory_tasks import (
    build_b_spline_trajectory_cache_file,
    generate_b_spline_trajectory_task_descriptors,
)


__all__ = [
    "BSplineTrajectory_PointwiseReference",
]


class BSplineTrajectory_PointwiseReference:
    """
    Pointwise fidelity reference for a reproducible B-spline trajectory benchmark instantiation.

    Tasks are generated internally as square-obstacle layouts using a scrambled Sobol sequence. 
    A normalized solution in [0, 1]^D contains the y coordinates of the
    D internal control points. The fixed endpoints (0, 0) and (1, 1) are
    prepended and appended, while all control-point x coordinates are uniformly
    spaced on [0, 1]. A clamped B-spline trajectory interpolates both endpoints.

    Let D be the number of optimized internal control points, while R is the
    number of parameter positions used to evaluate the continuous curve (i.e., resolution).
    The objective is the sampled trajectory length plus a fixed penalty for
    each active square obstacle intersected by at least one sampled trajectory
    segment. Each obstacle is counted at most once per trajectory.
    Each trajectory sample is computed as the weighted sum of a sliding
    window of `spline_degree + 1` adjacent control points using a
    precomputed local B-spline weight vector.

    NOTE: This class follows the direct pointwise evaluation order and is
    intended as a readable fidelity reference. Its individual- and
    sample-wise execution is not parallel-friendly and is not intended for
    routine use. It should be timed only when an explicit pointwise-reference
    comparison is required.

    Inspired by:
    @inproceedings{DBLP:conf/ieeecai/LinLXT24,
      author       = {Wu Lin and
                      Qiuzhen Lin and
                      Xiaoming Xue and
                      Kay Chen Tan},
      title        = {Sequential Transfer via Clustering-Based Similarity Measurement for
                      Faster Trajectory Optimization},
      booktitle    = {{IEEE} Conference on Artificial Intelligence, {CAI}},
      pages        = {1296--1301},
      publisher    = {{IEEE}},
      year         = {2024},
      url          = {https://doi.org/10.1109/CAI59869.2024.00229},
      doi          = {10.1109/CAI59869.2024.00229},
    }
    @article{DBLP:journals/tec/LiZTZ22,
      author       = {Jian{-}Yu Li and
                      Zhi{-}Hui Zhan and
                      Kay Chen Tan and
                      Jun Zhang},
      title        = {A Meta-Knowledge Transfer-Based Differential Evolution for Multitask
                      Optimization},
      journal      = {{IEEE} Trans. Evol. Comput.},
      volume       = {26},
      number       = {4},
      pages        = {719--734},
      year         = {2022},
      url          = {https://doi.org/10.1109/TEVC.2021.3131236},
      doi          = {10.1109/TEVC.2021.3131236},
    }
    """
    def __init__(self,
        num_source_tasks: int,
        dim: int = 60,
        num_evaluation_points: int = 60, # resolution of the trajectory curve evaluation
        num_obstacles: int | tuple[int, int] = 20,
        obstacle_size: float = 0.05,
        endpoint_clearance: float = 0.05,
        obstacle_gap: float = 0.005,
        feasibility_grid_size: int = 256,
        collision_weight: float = 20.0,
        spline_degree: int = 3,
        seed: Optional[int] = None,
        save_path: Optional[str | Path] = None,
        force_rebuild: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[str | torch.device] = None,
    ) -> None:
        assert (
            isinstance(num_source_tasks, int)
            and not isinstance(num_source_tasks, bool)
            and num_source_tasks >= 0
        )
        assert isinstance(force_rebuild, bool)
        if save_path is not None and seed is None:
            raise ValueError("`seed` must be explicitly provided when problem caching is enabled.")

        requested_device = torch.get_default_device() if device is None else torch.device(device)
        metadata = {
            "num_source_tasks": num_source_tasks,
            "dim": dim,
            "num_evaluation_points": num_evaluation_points,
            "num_obstacles": (
                tuple(num_obstacles)
                if isinstance(num_obstacles, tuple)
                else num_obstacles
            ),
            "obstacle_size": obstacle_size,
            "endpoint_clearance": endpoint_clearance,
            "obstacle_gap": obstacle_gap,
            "feasibility_grid_size": feasibility_grid_size,
            "spline_degree": spline_degree,
            "seed": seed,
            "dtype": str(dtype),
        }
        cache_file = None
        cached_data = None
        if save_path is not None:
            cache_file = build_b_spline_trajectory_cache_file(
                save_path=save_path,
                class_name=self.__class__.__name__,
                num_source_tasks=num_source_tasks,
                dim=dim,
                num_evaluation_points=num_evaluation_points,
                seed=seed,
            )
            if cache_file.exists() and not force_rebuild:
                candidate_cache = torch.load(cache_file, map_location="cpu")
                candidate_metadata = candidate_cache.get("metadata")
                if (
                    isinstance(candidate_metadata, dict)
                    and candidate_metadata == metadata
                    and all(
                        type(candidate_metadata[key]) is type(value)
                        for key, value in metadata.items()
                    )
                    and (
                        not isinstance(num_obstacles, tuple)
                        or all(
                            type(candidate_count) is type(obstacle_count)
                            for candidate_count, obstacle_count in zip(
                                candidate_metadata["num_obstacles"],
                                num_obstacles,
                            )
                        )
                    )
                ):
                    cached_data = candidate_cache

        if cached_data is not None:
            obstacle_centers = cached_data["obstacle_centers"].to(device=requested_device, dtype=dtype)
            obstacle_mask = cached_data["obstacle_mask"].to(device=requested_device)
            basis_window_starts = tuple(cached_data["basis_window_starts"])
            basis_window_weights = cached_data["basis_window_weights"].to(device=requested_device, dtype=dtype)
        else:
            obstacle_centers, obstacle_mask = generate_b_spline_trajectory_task_descriptors(
                num_tasks=num_source_tasks + 1,
                num_obstacles=num_obstacles,
                obstacle_size=obstacle_size,
                endpoint_clearance=endpoint_clearance,
                obstacle_gap=obstacle_gap,
                feasibility_grid_size=feasibility_grid_size,
                seed=seed,
                dtype=dtype,
                device=requested_device,
            )
        self.device = obstacle_centers.device
        self.obstacle_centers = obstacle_centers
        self.obstacle_mask = obstacle_mask # (K, M)

        assert obstacle_centers.ndim == 3
        assert obstacle_centers.shape == (num_source_tasks + 1, obstacle_mask.shape[1], 2,)
        assert obstacle_mask.ndim == 2
        assert obstacle_mask.shape == obstacle_centers.shape[:2]
        assert obstacle_mask.dtype == torch.bool
        assert obstacle_centers.dtype == dtype
        assert obstacle_centers.device == self.device
        assert obstacle_mask.device == self.device

        assert isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
        assert (
            isinstance(num_evaluation_points, int)
            and not isinstance(num_evaluation_points, bool)
            and num_evaluation_points >= 2
        )
        assert (
            isinstance(spline_degree, int)
            and not isinstance(spline_degree, bool)
            and spline_degree >= 1
        )
        assert dim + 2 >= spline_degree + 1
        assert collision_weight >= 0.0

        num_control_points = dim + 2 # include the fixed endpoints (0, 0) and (1, 1)
        self.control_x = torch.linspace(0.0, 1.0, num_control_points, dtype=self.obstacle_centers.dtype, device=self.device)
        if cached_data is None:
            basis_window_starts, basis_window_weights = self._build_b_spline_basis_windows(
                dim=dim,
                num_evaluation_points=num_evaluation_points,
                spline_degree=spline_degree,
                dtype=dtype,
                device=self.device,
            )
            if cache_file is not None:
                torch.save({
                    "metadata": metadata,
                    "obstacle_centers": obstacle_centers.detach().cpu(),
                    "obstacle_mask": obstacle_mask.detach().cpu(),
                    "basis_window_starts": basis_window_starts,
                    "basis_window_weights": basis_window_weights.detach().cpu(),
                }, cache_file,)
                print(f"{self.__class__.__name__} is built and saved into path `{cache_file}`.")
        self.basis_window_starts = basis_window_starts
        self.basis_window_weights = basis_window_weights
        assert len(self.basis_window_starts) == num_evaluation_points
        assert self.basis_window_weights.shape == (num_evaluation_points, spline_degree + 1)
        assert self.basis_window_weights.dtype == obstacle_centers.dtype
        assert self.basis_window_weights.device == self.device

        self.num_source_tasks = num_source_tasks
        self.dim = dim
        self.num_control_points = num_control_points
        self.num_evaluation_points = num_evaluation_points
        self.obstacle_size = obstacle_size
        self.half_obstacle_size = obstacle_size / 2.0
        self.collision_weight = collision_weight
        self.spline_degree = spline_degree

    @staticmethod
    def _build_b_spline_basis_windows(
        dim: int,
        num_evaluation_points: int, # resolution of the trajectory curve evaluation
        spline_degree: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> tuple[tuple[int, ...], Tensor]:
        """
        Build the fixed local B-spline basis representation shared by all tasks.

        For trajectory sample q_r, a_r is the start of its local control-point
        window and w_r contains spline_degree + 1 local B-spline weights, of which
        at most spline_degree + 1 are nonzero. 
        Given control points denoting p, the sample is evaluated as

            q_r = sum_j w_{r,j} p_{a_r+j}.

        The construction stores only the local sliding-window
        representation required by the pointwise reference.

        Returns
        -------
        basis_window_starts:
            Tuple of length num_evaluation_points containing a_r for each
            trajectory sample point.

        basis_window_weights:
            Tensor of shape [num_evaluation_points, spline_degree + 1]
            containing w_r for each trajectory sample.
        """
        assert isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
        assert (
            isinstance(num_evaluation_points, int)
            and not isinstance(num_evaluation_points, bool)
            and num_evaluation_points >= 2 # ensure at least the start and goal evaluation points
        )
        assert (
            isinstance(spline_degree, int)
            and not isinstance(spline_degree, bool)
            and spline_degree >= 1
        )
        assert dim + 2 >= spline_degree + 1 # control points must be enough to support at least one B-spline interpolation segment
        assert dtype in (torch.float32, torch.float64)
        device = torch.device(device)

        D, R = dim, num_evaluation_points
        num_control_points = D + 2 # include the fixed endpoints (0, 0) and (1, 1)
        num_interior_knots = num_control_points - spline_degree - 1
        knot_denominator = num_control_points - spline_degree
        interior_knots = tuple(knot_index / knot_denominator for knot_index in range(1, num_interior_knots + 1))
        knot_vector = (
            (0.0,) * (spline_degree + 1)
            + interior_knots
            + (1.0,) * (spline_degree + 1)
        )
        assert len(knot_vector) == num_control_points + spline_degree + 1

        #
        # The spline parameters t_r are R uniformly spaced evaluation positions on [0, 1].
        # They produce R trajectory samples and therefore R - 1 adjacent segments for cost evaluation.
        #
        spline_parameters = torch.linspace(0.0, 1.0, R, dtype=dtype, device=device)

        #
        # Start with degree-0 interval indicators. Each recurrence blends
        # two adjacent lower-degree bases, increasing both smoothness and
        # support by one knot interval. After the requested degree, each
        # row contains the local control-point weights of one trajectory sample.
        #
        basis = torch.zeros((R, len(knot_vector) - 1), dtype=dtype, device=device)
        for basis_index in range(basis.shape[1]): # degree-0
            left_knot = knot_vector[basis_index]
            right_knot = knot_vector[basis_index + 1]
            if left_knot == right_knot:
                continue
            basis[:, basis_index] = ((left_knot <= spline_parameters) & (spline_parameters < right_knot))

        for current_degree in range(1, spline_degree + 1): # degree-1 to degree-d
            next_basis = torch.zeros((R, basis.shape[1] - 1), dtype=dtype, device=device)
            for basis_index in range(next_basis.shape[1]):
                left_denominator = knot_vector[basis_index + current_degree] - knot_vector[basis_index]
                if left_denominator > 0.0:
                    next_basis[:, basis_index] += (
                        (spline_parameters - knot_vector[basis_index])
                        / left_denominator
                        * basis[:, basis_index]
                    )

                right_denominator = knot_vector[basis_index + current_degree + 1] - knot_vector[basis_index + 1]
                if right_denominator > 0.0:
                    next_basis[:, basis_index] += (
                        (knot_vector[basis_index + current_degree + 1] - spline_parameters)
                        / right_denominator
                        * basis[:, basis_index + 1]
                    )
            basis = next_basis

        assert basis.shape == (R, num_control_points)
        basis[0].zero_()
        basis[0, 0] = 1.0 # ensure the first sample point is exactly aligned to the first control point (start point)
        basis[-1].zero_()
        basis[-1, -1] = 1.0 # ensure the last sample point is exactly aligned to the last control point (goal point)

        nonnegative_tolerance = 10.0 * torch.finfo(dtype).eps
        assert torch.all(basis >= -nonnegative_tolerance)
        assert torch.allclose(basis.sum(dim=1), torch.ones(R, dtype=dtype, device=device))

        num_control_point_windows = num_control_points - spline_degree # `spline_degree + 1` elements per window 
        basis_window_starts = tuple(
            min(
                (sample_index * num_control_point_windows // (R - 1)),
                num_control_point_windows - 1,
            )
            for sample_index in range(R)
        )
        assert all(0 <= window_start <= num_control_points - spline_degree - 1 for window_start in basis_window_starts)

        window_start_tensor = torch.tensor(basis_window_starts, dtype=torch.long, device=device)
        window_offsets = torch.arange(spline_degree + 1, dtype=torch.long, device=device)
        window_indices = window_start_tensor.unsqueeze(1) + window_offsets.unsqueeze(0) # (R,1) + (1,d+1) -> (R,d+1)
        basis_window_weights = basis.gather(dim=1, index=window_indices) # (R, d+1)
        assert basis_window_weights.shape == (R, spline_degree + 1)
        assert torch.all(basis_window_weights >= -nonnegative_tolerance)
        assert torch.allclose(basis_window_weights.sum(dim=1), torch.ones(R, dtype=dtype, device=device))

        first_window_weights = torch.zeros(spline_degree + 1, dtype=dtype, device=device)
        first_window_weights[0] = 1.0
        last_window_weights = torch.zeros(spline_degree + 1, dtype=dtype, device=device)
        last_window_weights[-1] = 1.0
        assert basis_window_starts[0] == 0
        assert torch.allclose(basis_window_weights[0], first_window_weights)
        assert basis_window_starts[-1] == (num_control_points - spline_degree - 1)
        assert torch.allclose(basis_window_weights[-1], last_window_weights)
        return basis_window_starts, basis_window_weights

    def _evaluate_single_task(self, task_pop: Tensor, task_idx: int) -> Tensor:
        assert 0 <= task_idx <= self.num_source_tasks
        assert isinstance(task_pop, Tensor)
        assert task_pop.ndim == 2
        assert task_pop.shape[1] == self.dim
        assert task_pop.device == self.device
        assert task_pop.dtype == self.obstacle_centers.dtype

        N = task_pop.shape[0]
        task_fitness = task_pop.new_empty((N,))
        task_obstacles = self.obstacle_centers[task_idx]  # (M, 2)
        task_obstacle_mask = self.obstacle_mask[task_idx] # (M,)
        half_size = self.half_obstacle_size
        obstacle_min = task_obstacles - half_size
        obstacle_max = task_obstacles + half_size

        control_points = task_pop.new_empty((self.num_control_points, 2)) # (D + 2, 2)
        control_points[:, 0] = self.control_x
        control_points[0, 1] = 0.0
        control_points[-1, 1] = 1.0
        start_trajectory_point = control_points[0] # actually a trajectory start point q_0 = p_0
        remaining_window_starts = self.basis_window_starts[1:]
        remaining_window_weights = self.basis_window_weights[1:] # (R - 1, spline_degree + 1)
        direction_tolerance = torch.finfo(task_pop.dtype).eps

        for individual_idx in range(N):
            control_points[1:-1, 1] = task_pop[individual_idx]
            previous_trajectory_point = start_trajectory_point
            distance_cost = task_pop.new_zeros(())
            collided_obstacles = torch.zeros_like(task_obstacle_mask)

            for window_start, local_weights in zip(remaining_window_starts, remaining_window_weights):
                local_control_points = control_points[window_start:window_start + self.spline_degree + 1]
                current_trajectory_point = local_weights @ local_control_points # (spline_degree+1,)@(spline_degree+1,2)->(2,)
                segment_direction = current_trajectory_point - previous_trajectory_point
                parallel_axes = torch.abs(segment_direction) <= direction_tolerance # (2,)
                safe_direction = torch.where(
                    parallel_axes,
                    torch.ones_like(segment_direction),
                    segment_direction,
                )
                t_a = (obstacle_min - previous_trajectory_point) / safe_direction # (M, 2)
                t_b = (obstacle_max - previous_trajectory_point) / safe_direction # (M, 2)
                t_near = torch.minimum(t_a, t_b)
                t_far = torch.maximum(t_a, t_b)
                t_near = torch.where(
                    parallel_axes,
                    torch.full_like(t_near, -torch.inf),
                    t_near,
                )
                t_far = torch.where(
                    parallel_axes,
                    torch.full_like(t_far, torch.inf),
                    t_far,
                )
                parallel_inside = (
                    (~parallel_axes) | (
                        (obstacle_min <= previous_trajectory_point) & (previous_trajectory_point <= obstacle_max)
                    )
                ) # (M, 2)
                entry_t = torch.maximum(t_near.amax(dim=1), torch.zeros_like(t_near[:, 0])) # (M,)
                exit_t  = torch.minimum(t_far.amin(dim=1), torch.ones_like(t_far[:, 0]))    # (M,)
                segment_intersects_obstacles = (parallel_inside.all(dim=1) & (entry_t <= exit_t)) # (M,)
                collided_obstacles |= (segment_intersects_obstacles & task_obstacle_mask) # (M,)

                segment_length = torch.linalg.vector_norm(segment_direction)
                distance_cost += segment_length

                previous_trajectory_point = current_trajectory_point

            collision_count = collided_obstacles.sum().type_as(task_pop)
            task_fitness[individual_idx] = distance_cost + self.collision_weight * collision_count

        return task_fitness

    def evaluate(self,
        pop: Tensor,
        *,
        eval_task_indices: int,
    ) -> Tensor:
        assert (
            isinstance(eval_task_indices, int)
            and not isinstance(eval_task_indices, bool)
        ), "`eval_task_indices` must be an integer."
        assert 0 <= eval_task_indices <= self.num_source_tasks, "Unsupported value of `eval_task_indices`."

        return self._evaluate_single_task(task_pop=pop, task_idx=eval_task_indices)
