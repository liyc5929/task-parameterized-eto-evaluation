import torch
from typing import Optional, Protocol

from evox.core import Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp


__all__ = [
    "MA_MTO",
]


class Problem(Protocol):
    def evaluate(self, task_pops: torch.Tensor):
        ...


class MA_MTO:
    """
    Mean Alignment Multi-Task Optimization (MA-MTO) algorithm.

    Inspired by:
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

    This simplified realization uses GA evolution and mean-aligned elite injection;
    it is inspired by, but is not an exact implementation of MKTDE.
    """
    def __init__(self,
        problem: Problem,
        num_tasks: int,
        task_pop_size: int,
        task_dim: int,
        task_lb: torch.Tensor,
        task_ub: torch.Tensor,
        minimization: bool = True,
        transfer_pop_size: int = 10,
        transfer_interval: int = 10,
        ga_mu_c: float = 2.0,
        ga_mu_m: float = 5.0,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()
        assert num_tasks > 0
        assert task_pop_size >= 1
        assert task_dim >= 1
        assert 0 <= transfer_pop_size <= task_pop_size
        assert (num_tasks > 1) or (transfer_pop_size == 0)
        assert transfer_interval >= 1
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        reference_dtype = task_lb.dtype
        assert reference_dtype in (torch.float32, torch.float64)
        K, N, S, D = num_tasks, task_pop_size, transfer_pop_size, task_dim

        task_lb = task_lb.to(device=device, dtype=reference_dtype)
        task_ub = task_ub.to(device=device, dtype=reference_dtype)
        if task_lb.ndim == 0:
            task_lb = task_lb.expand(D)
        if task_ub.ndim == 0:
            task_ub = task_ub.expand(D)
        assert task_lb.shape == (D,)
        assert task_ub.shape == (D,)
        assert torch.all(task_lb < task_ub)

        task_pops = Mutable((
            torch.rand((K, N, D), dtype=reference_dtype, device=device)
            * (task_ub - task_lb)[None, None, :]
            + task_lb[None, None, :]
        ))
        task_fits = Mutable(torch.full((K, N), torch.inf, dtype=reference_dtype, device=device))
        task_offspring_pops = Mutable(torch.zeros_like(task_pops)) # (K, N, D)
        task_offspring_fits = Mutable(torch.full((K, N), torch.inf, dtype=reference_dtype, device=device))
        task_candidate_pops = Mutable(torch.zeros((K, 2 * N, D), dtype=reference_dtype, device=device))
        task_candidate_fits = Mutable(torch.full((K, 2 * N), torch.inf, dtype=reference_dtype, device=device))
        task_transfer_pops = Mutable(torch.empty((K, S, D), dtype=reference_dtype, device=device))

        self.problem = problem
        self.minimization = minimization

        self.task_pops = task_pops
        self.task_fits = task_fits
        self.task_offspring_pops = task_offspring_pops
        self.task_offspring_fits = task_offspring_fits
        self.task_candidate_pops = task_candidate_pops
        self.task_candidate_fits = task_candidate_fits
        self.task_transfer_pops = task_transfer_pops

        self.num_tasks = K
        self.task_pop_size = N
        self.task_dim = D
        self.task_lb = task_lb
        self.task_ub = task_ub
        self.transfer_pop_size = transfer_pop_size
        self.transfer_interval = transfer_interval
        self.ga_mu_c = ga_mu_c
        self.ga_mu_m = ga_mu_m
        self.device  = device

        self.generation_idx = 0

    @property
    def best_fitnesses(self) -> torch.Tensor:
        return self.task_fits.min(dim=1).values * (1.0 if self.minimization else -1.0)

    def init_step(self):
        K, N, S, D = self.num_tasks, self.task_pop_size, self.transfer_pop_size, self.task_dim

        # Get fitnesses of initial populations
        task_fits = self.problem.evaluate(self.task_pops)
        task_fits = (
            task_fits if isinstance(task_fits, torch.Tensor) else torch.stack(task_fits)
        ) * (1.0 if self.minimization else -1.0)
        assert task_fits.shape == (K, N)
        self.task_fits.copy_(task_fits)

        # Get initial transfer populations
        selected_indices = torch.topk(input=self.task_fits, k=S, dim=1, largest=False).indices
        task_transfer_pops = torch.gather(
            input=self.task_pops,
            dim=1,
            index=selected_indices[:, :, None].expand(-1, -1, D),
        )
        self.task_transfer_pops.copy_(task_transfer_pops)

        # Reset generation index
        self.generation_idx = 0

    def step(self):
        K, N, S, D = self.num_tasks, self.task_pop_size, self.transfer_pop_size, self.task_dim

        # Transfer
        if (self.transfer_pop_size > 0) and ((self.generation_idx + 1) % self.transfer_interval == 0):
            task_injected_pops = torch.empty_like(self.task_transfer_pops)
            source_task_indices = torch.randint(low=0, high=K - 1, size=(K,), device=self.device)
            target_task_indices = torch.arange(K, device=self.device)
            source_task_indices = source_task_indices + (source_task_indices >= target_task_indices)

            for target_task_idx in range(K):
                source_task_idx = source_task_indices[target_task_idx]
                source_pop_mean = torch.mean(self.task_pops[source_task_idx], dim=0)
                target_pop_mean = torch.mean(self.task_pops[target_task_idx], dim=0)
                task_injected_pops[target_task_idx] = clamp(
                    self.task_transfer_pops[source_task_idx] - source_pop_mean[None, :] + target_pop_mean[None, :],
                    self.task_lb,
                    self.task_ub,
                )

            # Evaluate selected individuals (solutions) to be injected
            task_injected_fits = self.problem.evaluate(task_injected_pops)
            task_injected_fits = (
                task_injected_fits if isinstance(task_injected_fits, torch.Tensor) else torch.stack(task_injected_fits)
            ) * (1.0 if self.minimization else -1.0)
            assert task_injected_fits.shape == (K, S)
            # Update population for each task by injecting individuals
            replace_scores = torch.rand((K, N), device=self.device)
            replace_indices = torch.topk(replace_scores, k=S, dim=1, largest=False, sorted=True).indices
            self.task_pops.scatter_(
                dim=1,
                index=replace_indices[:, :, None].expand(-1, -1, D),
                src=task_injected_pops,
            )
            self.task_fits.scatter_(dim=1, index=replace_indices, src=task_injected_fits)

        # Offspring generation
        for k in range(K):
            task_pop = self.task_pops[k]
            perm = torch.randperm(N, device=self.device)
            offspring_pop = simulated_binary(task_pop[perm], dis_c=self.ga_mu_c)
            offspring_pop = polynomial_mutation(offspring_pop, lb=self.task_lb, ub=self.task_ub, dis_m=self.ga_mu_m)
            offspring_pop = clamp(offspring_pop, self.task_lb, self.task_ub)
            self.task_offspring_pops[k].copy_(offspring_pop)

        # Offspring evaluation and candidate updating
        task_offspring_fits = self.problem.evaluate(self.task_offspring_pops)
        task_offspring_fits = (
            task_offspring_fits if isinstance(task_offspring_fits, torch.Tensor) else torch.stack(task_offspring_fits)
        ) * (1.0 if self.minimization else -1.0)
        assert task_offspring_fits.shape == (K, N)
        self.task_offspring_fits.copy_(task_offspring_fits)

        self.task_candidate_pops[:, :N].copy_(self.task_pops)
        self.task_candidate_pops[:, N:].copy_(self.task_offspring_pops)
        self.task_candidate_fits[:, :N].copy_(self.task_fits)
        self.task_candidate_fits[:, N:].copy_(self.task_offspring_fits)

        # Selection
        selected_candidate_indices = torch.topk(self.task_candidate_fits, k=N, dim=1, largest=False, sorted=True).indices
        selected_task_pops = torch.gather(
            input=self.task_candidate_pops,
            dim=1,
            index=selected_candidate_indices[:, :, None].expand(-1, -1, D),
        )
        selected_task_fits = torch.gather(
            input=self.task_candidate_fits,
            dim=1,
            index=selected_candidate_indices,
        )
        self.task_pops.copy_(selected_task_pops)
        self.task_fits.copy_(selected_task_fits)

        # Get next transfer populations
        selected_indices = torch.topk(input=self.task_fits, k=S, dim=1, largest=False).indices
        task_transfer_pops = torch.gather(
            input=self.task_pops,
            dim=1,
            index=selected_indices[:, :, None].expand(-1, -1, D),
        )
        self.task_transfer_pops.copy_(task_transfer_pops)

        # Update generation index
        self.generation_idx += 1
