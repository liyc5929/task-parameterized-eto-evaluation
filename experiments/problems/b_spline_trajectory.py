from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from ._b_spline_trajectory_tasks import (
    build_b_spline_trajectory_cache_file,
    generate_b_spline_trajectory_task_descriptors,
)


__all__ = [
    "BSplineTrajectory",
]


class BSplineTrajectory:
    """
    Parallel realization of the reproducible B-spline trajectory benchmark.

    Tasks are generated internally as square-obstacle layouts using a scrambled
    Sobol sequence. A normalized solution in [0, 1]^D contains the y coordinates
    of the D internal control points. Fixed endpoints (0, 0) and (1, 1) are
    prepended and appended, while all control-point x coordinates are uniformly
    spaced on [0, 1]. The objective is the sampled trajectory length plus a
    fixed penalty for each active square obstacle intersected by at least one
    sampled trajectory segment. Each obstacle is counted at most once per trajectory.

    For a population of N candidates, the two-dimensional control points are
    organized as a tensor of shape [N, D + 2, 2]. The fixed B-spline basis matrix
    maps all candidates to R trajectory samples jointly,

        Q = B @ P,

    yielding a trajectory tensor of shape [N, R, 2]. Adjacent segment lengths
    and all (R - 1)-by-M segment-obstacle relations are evaluated by tensor
    slicing, broadcasting, and reduction without individual- or sample-wise
    Python loops.  This class is the parallel realization intended for routine
    use and runtime timing.

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
            basis_matrix = cached_data["basis_matrix"].to(device=requested_device, dtype=dtype)
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
            basis_matrix = self._build_b_spline_basis_matrix(
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
                    "basis_matrix": basis_matrix.detach().cpu(),
                }, cache_file)
                print(f"{self.__class__.__name__} is built and saved into path `{cache_file}`.")
        self.basis_matrix = basis_matrix
        assert self.basis_matrix.shape == (num_evaluation_points, num_control_points)
        assert self.basis_matrix.dtype == obstacle_centers.dtype
        assert self.basis_matrix.device == self.device

        self.num_source_tasks = num_source_tasks
        self.dim = dim
        self.num_control_points = num_control_points
        self.num_evaluation_points = num_evaluation_points
        self.obstacle_size = obstacle_size
        self.half_obstacle_size = obstacle_size / 2.0
        self.collision_weight = collision_weight
        self.spline_degree = spline_degree

    @staticmethod
    def _build_b_spline_basis_matrix(
        dim: int,
        num_evaluation_points: int, # resolution of the trajectory curve evaluation
        spline_degree: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        """
        Build the fixed B-spline basis matrix shared by all tasks.
        
        For R uniformly spaced spline parameters t_r in [0, 1], entry
        B[r, i] is the degree-`spline_degree` B-spline weight of
        control point p_i at spline parameter t_r. Therefore,
        
            q_r = sum_i B[r, i] p_i,
        
        and all trajectory samples satisfy Q = B @ P.
        
        The construction starts from degree-0 interval indicators over
        adjacent knots. Each recurrence blends two neighboring lower-degree
        bases to obtain the next degree, until the requested spline degree
        is reached. Repeated endpoint knots anchor the first and last
        trajectory samples to the first and last control points.
        
        Returns
        -------
        basis_matrix:
            Tensor of shape [num_evaluation_points, dim + 2].
            Each row contains the zero-padded local control-point weights
            of one trajectory sample.
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

        basis_matrix = basis
        assert basis_matrix.shape == (R, num_control_points)
        basis_matrix[0].zero_()
        basis_matrix[0, 0] = 1.0 # ensure the first sample point is exactly aligned to the first control point (start point)
        basis_matrix[-1].zero_()
        basis_matrix[-1, -1] = 1.0 # ensure the last sample point is exactly aligned to the last control point (goal point)

        nonnegative_tolerance = 10.0 * torch.finfo(dtype).eps
        assert torch.all(basis_matrix >= -nonnegative_tolerance)
        assert torch.allclose(basis_matrix.sum(dim=1), torch.ones(R, dtype=dtype, device=device))
        return basis_matrix

    def _evaluate_single_task(self, task_pop: Tensor, task_idx: int) -> Tensor:
        assert 0 <= task_idx <= self.num_source_tasks
        assert isinstance(task_pop, Tensor)
        assert task_pop.ndim == 2
        assert task_pop.shape[1] == self.dim
        assert task_pop.device == self.device
        assert task_pop.dtype == self.obstacle_centers.dtype

        N = task_pop.shape[0]
        task_obstacles = self.obstacle_centers[task_idx]  # (M, 2)
        task_obstacle_mask = self.obstacle_mask[task_idx] # (M,)
        half_size = self.half_obstacle_size

        control_points = task_pop.new_empty((N, self.num_control_points, 2)) # (N, D + 2, 2)
        control_points[:, :, 0] = self.control_x
        control_points[:, 1:-1, 1] = task_pop
        control_points[:, 0, 1] = 0.0
        control_points[:, -1, 1] = 1.0

        trajectory_points = torch.matmul(self.basis_matrix, control_points) # (R, D + 2) @ (N, D + 2, 2) -> (N, R, 2)

        trajectory_segments = trajectory_points[:, 1:, :] - trajectory_points[:, :-1, :] # (N, R - 1, 2)
        segment_lengths = torch.linalg.vector_norm(trajectory_segments, dim=2) # (N, R - 1)
        distance_cost = segment_lengths.sum(dim=1) # (N,)

        segment_starts = trajectory_points[:, :-1, None, :] # (N, R - 1, 1, 2)
        segment_directions = trajectory_segments[:, :, None, :] # (N, R - 1, 1, 2)
        obstacle_min = task_obstacles[None, None, :, :] - half_size # (1, 1, M, 2)
        obstacle_max = task_obstacles[None, None, :, :] + half_size # (1, 1, M, 2)
        parallel_axes = torch.abs(segment_directions) <= torch.finfo(segment_directions.dtype).eps # (N, R - 1, 1, 2)
        safe_directions = torch.where(
            parallel_axes,
            torch.ones_like(segment_directions),
            segment_directions,
        )
        t_a = (obstacle_min - segment_starts) / safe_directions # (N, R - 1, M, 2)
        t_b = (obstacle_max - segment_starts) / safe_directions # (N, R - 1, M, 2)
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
                (obstacle_min <= segment_starts) & (segment_starts <= obstacle_max)
            )
        ) # (N, R - 1, M, 2)
        entry_t = torch.maximum(t_near.amax(dim=3), torch.zeros_like(t_near[:, :, :, 0])) # (N, R - 1, M)
        exit_t = torch.minimum(t_far.amin(dim=3), torch.ones_like(t_far[:, :, :, 0])) # (N, R - 1, M)
        segment_intersects_obstacles = (parallel_inside.all(dim=3) & (entry_t <= exit_t)) # (N, R - 1, M)
        collided_obstacles = torch.any(segment_intersects_obstacles & task_obstacle_mask[None, None, :], dim=1) # (N, M)
        collision_count = collided_obstacles.sum(dim=1).type_as(task_pop) # (N,)

        return distance_cost + self.collision_weight * collision_count

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
