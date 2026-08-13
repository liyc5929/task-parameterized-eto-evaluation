import math
from collections.abc import Sequence
from typing import List, Optional

import torch
from torch import Tensor

from ._kinematic_arm_tasks import generate_kinematic_arm_task_descriptors


__all__ = [
    "KinematicArm",
]


class KinematicArm:
    """
    Task-tensorized planar kinematic-arm benchmark from Mouret and Maguire
    GECCO 2020.

    Each task is described by ``[L, alpha_max]``, where ``L`` is the
    normalized total arm length and ``alpha_max`` controls the joint-angle
    range. For a normalized solution ``x in [0, 1]^D``, the relative joint
    angles are decoded as

        theta_i = (x_i - 0.5) * alpha_max * 2 * pi / D.

    This implementation uses the cumulative direction-angle form

        phi_i = sum_{j=1}^i theta_j,
        p = sum_{i=1}^D (L / D) * [cos(phi_i), sin(phi_i)],

    which is mathematically equivalent to the homogeneous-transform reference
    implemented in ``_reference/kinematic_arm_matrix.py``. Fitness is the
    negative distance from the end-effector to the target, so larger values are
    better.

    References:
    @inproceedings{DBLP:conf/gecco/MouretM20,
      author       = {Jean{-}Baptiste Mouret and
                      Glenn Maguire},
      title        = {Quality diversity for multi-task optimization},
      booktitle    = {{GECCO} '20: Genetic and Evolutionary Computation Conference},
      pages        = {121--129},
      publisher    = {{ACM}},
      year         = {2020},
      url          = {https://doi.org/10.1145/3377930.3390203},
      doi          = {10.1145/3377930.3390203},
    }
    """

    def __init__(self,
        num_tasks: int,
        dim: int = 10,
        task_seed: Optional[int] = None,
        target_position: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()
        assert num_tasks > 0
        assert dim > 0
        assert dtype in (torch.float32, torch.float64)
        device = torch.get_default_device() if device is None else torch.device(device)

        task_descriptors = generate_kinematic_arm_task_descriptors(
            num_tasks=num_tasks,
            seed=task_seed,
            dtype=dtype,
            device=device,
        )
        if target_position is None:
            target_position = torch.tensor((1.0, 1.0), dtype=dtype, device=device)
        else:
            assert target_position.shape == (2,)
            target_position = target_position.to(device=task_descriptors.device, dtype=dtype)

        self.task_descriptors = task_descriptors
        self.task_total_lengths = task_descriptors[:, 0]
        self.task_max_angles = task_descriptors[:, 1]
        self.target_position = target_position

        self.num_tasks = num_tasks
        self.dim = dim
        self.device = task_descriptors.device

    @classmethod
    def list_problems(cls) -> list[str]:
        return ["KinematicArm",]

    def _evaluate_tasks(self, task_pops: Tensor, task_indices: Tensor) -> Tensor:
        assert task_pops.ndim == 3
        assert task_pops.shape[2] == self.dim
        assert task_pops.device == self.task_descriptors.device
        assert task_pops.dtype == self.task_descriptors.dtype
        assert task_indices.ndim == 1
        assert task_indices.shape[0] == task_pops.shape[0]
        assert task_indices.device == self.task_descriptors.device
        assert task_indices.dtype == torch.long
        D = self.dim
        target_x = self.target_position[0]
        target_y = self.target_position[1]

        task_max_angles = self.task_max_angles[task_indices, None, None]
        task_link_lengths = self.task_total_lengths[task_indices, None, None] / D

        # Relative joint angles and cumulative link directions
        joint_angles = (task_pops - 0.5) * task_max_angles * (2.0 * math.pi / D)
        cumulative_angles = torch.cumsum(joint_angles, dim=-1)

        end_x = torch.sum(task_link_lengths * torch.cos(cumulative_angles), dim=-1)
        end_y = torch.sum(task_link_lengths * torch.sin(cumulative_angles), dim=-1)

        task_fits = -torch.sqrt((end_x - target_x).square() + (end_y - target_y).square())
        return task_fits

    def evaluate(self,
        task_pops: Tensor | Sequence[Tensor],
        *,
        indiv_task_indices: Optional[Tensor] = None,
        eval_task_indices: Optional[Sequence[int]] = None,
        valid_indiv_mask: Optional[Tensor] = None, # unused
    ) -> List[Tensor]:
        """
        task_pops:
            Tensor(K, N, D) | Sequence[K * Tensor(N, D)] when evaluating all tasks;
            Tensor(K_eval, N, D) | Sequence[K_eval * Tensor(N, D)]
            when eval_task_indices is provided. Sequence inputs must be stackable
            into a dense Tensor because this implementation is task-tensorized.

        indiv_task_indices:
            Individual-to-task assignments used by unified-population MTO.

        eval_task_indices:
            Global indices of the tasks represented by the local task populations.

        valid_indiv_mask:
            Kept for the standard Problem interface; unused here.

        output:
            List[K * Tensor(N,)] when evaluating all tasks;
            List[K_eval * Tensor(N,)] when evaluating selected tasks.
        """
        K = self.num_tasks

        if eval_task_indices is not None:
            if not isinstance(eval_task_indices, Sequence):
                raise TypeError("`eval_task_indices` must be a Python sequence of integers.")
            assert all(0 <= task_idx < K for task_idx in eval_task_indices)
            local_task_pops = task_pops if isinstance(task_pops, Tensor) else torch.stack(list(task_pops), dim=0)
            assert local_task_pops.shape[0] == len(eval_task_indices)
            task_indices = torch.as_tensor(eval_task_indices, dtype=torch.long, device=self.task_descriptors.device)

            if indiv_task_indices is None:
                local_task_fits = self._evaluate_tasks(task_pops=local_task_pops, task_indices=task_indices)
                return list(local_task_fits.unbind(dim=0))

            if isinstance(indiv_task_indices, Tensor) and indiv_task_indices.ndim == 1:
                assert indiv_task_indices.shape[0] == local_task_pops.shape[1]
                assert indiv_task_indices.device == local_task_pops.device
                N = local_task_pops.shape[1]
                indiv_positions = torch.arange(N, device=local_task_pops.device)
                valid_assignment_mask = (indiv_task_indices[None, :] == task_indices[:, None]) # (K_eval, N)
                local_assignment_positions = valid_assignment_mask.to(torch.long).argmax(dim=0) # (N,)
                has_assignment = valid_assignment_mask.any(dim=0) # (N,)

                pair_pops = local_task_pops[local_assignment_positions, indiv_positions] # (N, D)
                pair_task_indices = task_indices[local_assignment_positions] # (N,)
                pair_fits = self._evaluate_tasks(task_pops=pair_pops[:, None, :], task_indices=pair_task_indices).squeeze(1) # (N,)
                pair_fits.masked_fill_(~has_assignment, -torch.inf)

                local_task_fits = local_task_pops.new_full((task_indices.shape[0], N), -torch.inf)
                local_task_fits[local_assignment_positions, indiv_positions] = pair_fits
                return list(local_task_fits.unbind(dim=0))

            raise TypeError("Unsupported type for `indiv_task_indices`.")

        local_task_pops = task_pops if isinstance(task_pops, Tensor) else torch.stack(list(task_pops), dim=0)
        assert local_task_pops.shape[0] == K
        task_indices = torch.arange(K, device=self.task_descriptors.device)

        if indiv_task_indices is None:
            local_task_fits = self._evaluate_tasks(task_pops=local_task_pops, task_indices=task_indices)
            return list(local_task_fits.unbind(dim=0))

        if isinstance(indiv_task_indices, Tensor) and indiv_task_indices.ndim == 1:
            assert indiv_task_indices.shape[0] == local_task_pops.shape[1]
            assert indiv_task_indices.device == local_task_pops.device
            N = local_task_pops.shape[1]
            indiv_positions = torch.arange(N, device=local_task_pops.device)
            assigned_task_indices = indiv_task_indices.to(dtype=torch.long)

            pair_pops = local_task_pops[assigned_task_indices, indiv_positions] # (N, D)
            pair_fits = self._evaluate_tasks(task_pops=pair_pops[:, None, :], task_indices=assigned_task_indices).squeeze(1) # (N,)

            local_task_fits = local_task_pops.new_full((K, N), -torch.inf)
            local_task_fits[assigned_task_indices, indiv_positions] = pair_fits
            return list(local_task_fits.unbind(dim=0))

        raise TypeError("Unsupported type for `indiv_task_indices`.")
