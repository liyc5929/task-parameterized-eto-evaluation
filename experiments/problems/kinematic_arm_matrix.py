import math
from collections.abc import Sequence
from typing import List, Optional

import torch
from torch import Tensor

from ._kinematic_arm_tasks import generate_kinematic_arm_task_descriptors


class KinematicArm_MatrixReference:
    """
    Homogeneous-transform reference for the planar kinematic arm benchmark.

    This class implements the planar-arm problem described by Mouret and
    Maguire using homogeneous transformations. Homogeneous coordinates
    represent a point as [x, y, z, 1]^T, allowing rotations and translations
    to be composed by 4*4 matrix multiplication.

    Fitness is the negative distance from the end-effector to the target,
    so larger values are better. 

    NOTE: This implementation is intended only for mathematical correctness
    validation and is not recommended for normal execution or timing.

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
        dtype: torch.dtype = torch.float32,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()
        assert num_tasks > 0
        assert dim > 0
        device = torch.get_default_device() if device is None else torch.device(device)

        task_descriptors = generate_kinematic_arm_task_descriptors(
            num_tasks=num_tasks,
            seed=task_seed,
            dtype=dtype,
            device=device,
        )
        target_position = torch.tensor((1.0, 1.0), dtype=dtype, device=device)

        self.task_descriptors = task_descriptors
        self.task_total_lengths = task_descriptors[:, 0]
        self.task_max_angles = task_descriptors[:, 1]
        self.target_position = target_position

        self.num_tasks = num_tasks
        self.dim = dim
        self.device = task_descriptors.device

    def _evaluate_single_task(self, task_pop: Tensor, task_idx: int) -> Tensor:
        assert 0 <= task_idx < self.num_tasks
        assert task_pop.ndim == 2
        assert task_pop.shape[1] == self.dim
        assert task_pop.device == self.task_descriptors.device
        assert task_pop.dtype == self.task_descriptors.dtype

        N, D = task_pop.shape
        task_fit = task_pop.new_empty((N,))
        local_origin_homogeneous = task_pop.new_tensor((0.0, 0.0, 0.0, 1.0))

        task_total_length = self.task_total_lengths[task_idx]
        task_max_angle = self.task_max_angles[task_idx]
        link_length = task_total_length / D

        for n in range(N):
            transform = torch.eye(4, device=task_pop.device, dtype=task_pop.dtype) # denotes M_0 in paper
            joint_angles = (task_pop[n] - 0.5) * task_max_angle * (2.0 * math.pi) / D # Eq. (8) in paper
            augmented_angles = torch.empty((D + 1,), device=task_pop.device, dtype=task_pop.dtype)
            augmented_angles[:D] = joint_angles
            augmented_angles[D] = 0.0 # extend the final link without adding another joint rotation

            augmented_link_lengths = torch.empty((D + 1,), device=task_pop.device, dtype=task_pop.dtype)
            augmented_link_lengths[0] = 0.0 # apply the first joint rotation before any link translation
            augmented_link_lengths[1:] = link_length

            for i in range(D + 1):
                theta = augmented_angles[i]
                current_link_length = augmented_link_lengths[i]
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)

                joint_transform = torch.eye(4, device=task_pop.device, dtype=task_pop.dtype)
                joint_transform[0, 0] = cos_theta
                joint_transform[0, 1] = -sin_theta
                joint_transform[0, 3] = current_link_length
                joint_transform[1, 0] = sin_theta
                joint_transform[1, 1] = cos_theta

                transform = transform @ joint_transform # Eq. (11) in paper

            end_position = transform @ local_origin_homogeneous # Eq. (12) in paper
            task_fit[n] = -torch.linalg.norm(end_position[:2] - self.target_position) # Eq. (13) in paper

        return task_fit

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
            when eval_task_indices is provided.

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

        if eval_task_indices is not None:
            if not isinstance(eval_task_indices, Sequence):
                raise TypeError("`eval_task_indices` must be a Python sequence of integers.")
            assert len(task_pops) == len(eval_task_indices)

            if indiv_task_indices is None:
                return [
                    self._evaluate_single_task(task_pop=task_pops[local_task_idx], task_idx=global_task_idx)
                    for local_task_idx, global_task_idx in enumerate(eval_task_indices)
                ]

            if isinstance(indiv_task_indices, Tensor) and indiv_task_indices.ndim == 1:
                task_fitness = list()
                for local_task_idx, global_task_idx in enumerate(eval_task_indices):
                    m = (indiv_task_indices == global_task_idx)
                    pop = task_pops[local_task_idx]
                    fit = torch.full((pop.size(0),), -torch.inf, device=pop.device, dtype=pop.dtype)
                    fit[m] = self._evaluate_single_task(task_pop=pop[m], task_idx=global_task_idx)
                    task_fitness.append(fit)
                return task_fitness

            raise TypeError("Unsupported type for `indiv_task_indices`.")

        K = self.num_tasks
        assert len(task_pops) == K
        if indiv_task_indices is None:
            return [
                self._evaluate_single_task(task_pop=task_pops[k], task_idx=k)
                for k in range(K)
            ]

        if isinstance(indiv_task_indices, Tensor) and indiv_task_indices.ndim == 1:
            task_fitness = list()
            for k in range(K):
                m = (indiv_task_indices == k)
                pop = task_pops[k]
                fit = torch.full((pop.size(0),), -torch.inf, device=pop.device, dtype=pop.dtype)
                fit[m] = self._evaluate_single_task(task_pop=pop[m], task_idx=k)
                task_fitness.append(fit)
            return task_fitness

        raise TypeError("Unsupported type for `indiv_task_indices`.")
