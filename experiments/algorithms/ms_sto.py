import torch
from typing import Optional, Protocol

from evox.core import Mutable
from evox.operators.sampling import latin_hypercube_sampling_standard
from evox.utils import clamp

from ._sto_knowledge_base import KnowledgeBase


__all__ = [
    "MS_STO",
]
 

class Problem(Protocol):
    def evaluate(self, target_pop: torch.Tensor, *, eval_task_indices: int):
        ...


class MS_STO:
    """
    Algorithm: M1 (Mean Similarity Transfer)

    Source:
    @article{DBLP:journals/tec/XueYFZST24,
      author       = {Xiaoming Xue and
                      Cuie Yang and
                      Liang Feng and
                      Kai Zhang and
                      Linqi Song and
                      Kay Chen Tan},
      title        = {Solution Transfer in Evolutionary Optimization: An Empirical Study
                      on Sequential Transfer},
      journal      = {{IEEE} Trans. Evol. Comput.},
      volume       = {28},
      number       = {6},
      pages        = {1776--1793},
      year         = {2024},
      url          = {https://doi.org/10.1109/TEVC.2023.3339506},
      doi          = {10.1109/TEVC.2023.3339506},
    }

    Inspired by:
    @article{DBLP:journals/tec/DingYJC19,
      author       = {Jinliang Ding and
                      Cuie Yang and
                      Yaochu Jin and
                      Tianyou Chai},
      title        = {Generalized Multitasking for Evolutionary Optimization of Expensive
                      Problems},
      journal      = {{IEEE} Trans. Evol. Comput.},
      volume       = {23},
      number       = {1},
      pages        = {44--58},
      year         = {2019},
      url          = {https://doi.org/10.1109/TEVC.2017.2785351},
      doi          = {10.1109/TEVC.2017.2785351},
    }
    """
    def __init__(self, 
        problem: Problem,
        num_source_tasks: int,
        target_pop_size: int,
        max_generation: int,
        encoded_dim: int,
        encoded_lb: torch.Tensor,
        encoded_ub: torch.Tensor,
        mu_c: float = 15.0, 
        mu_m: float = 15.0,
        knowledge_base: KnowledgeBase | None = None,
        transfer_pop_size: int = 1, 
        transfer_interval: int = 1,
        device: Optional[str | torch.device] = None, 
    ) -> None:
        super().__init__()
        assert transfer_pop_size <= target_pop_size
        assert transfer_interval >= 1
        assert max_generation >= 1
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if knowledge_base is None:
            raise ValueError("`knowledge_base` must be provided.")
        knowledge_shape = knowledge_base.manifest["dimensions"]
        if (knowledge_shape["num_sources"] != num_source_tasks
            or knowledge_shape["num_generations"] != max_generation
            or knowledge_shape["encoded_dimension"] != encoded_dim
        ):
            raise ValueError(f"The knowledge base is incompatible with the {self.__class__.__name__} configuration.")
        assert knowledge_base["population"].dtype == encoded_lb.dtype

        knowledge_base_pops = knowledge_base["population"] # CPU-backed memmap: (G, K, N_src, D_max)
        N_src, N_tar, S, D = knowledge_base_pops.size(-2), target_pop_size, transfer_pop_size, encoded_dim
        if S > N_src:
            raise ValueError("`transfer_pop_size` must not exceed the source population size.")

        final_generation_source_pops = knowledge_base["population"][-1].to(device)
        if torch.device(device).type == "cpu":
            final_generation_source_pops = final_generation_source_pops.clone()
        current_source_pop_buffer = (
            torch.empty_like(final_generation_source_pops)
            if transfer_pop_size > 0 and transfer_interval < max_generation 
            else None
        )

        encoded_lb = encoded_lb.to(device=device)
        encoded_ub = encoded_ub.to(device=device)

        target_pop = Mutable(latin_hypercube_sampling_standard(n=N_tar, d=D, device=device, smooth=True)) # (N_tar, D_tar)
        target_fit = Mutable(torch.full((N_tar,), torch.inf, device=device))
        candidate_target_pop = Mutable(torch.empty((2*N_tar, D), device=device))
        candidate_target_fit = Mutable(torch.full((2*N_tar,), torch.inf, device=device))

        def _crossover(x: torch.Tensor, dis_c: float, pro_c: float = 1.0) -> torch.Tensor:
            N, D = x.size()
            parent1_dec = x[: N // 2, :]
            parent2_dec = x[N // 2 : N // 2 * 2, :]
            mu = torch.rand(N // 2, D, device=x.device)
            # Beta calculation for SBX
            beta = torch.zeros(mu.size(), device=x.device)
            beta = torch.where(mu <= 0.5, torch.pow(2 * mu, 1 / (dis_c + 1)), beta)
            beta = torch.where(mu > 0.5, torch.pow(2 - 2 * mu, -1 / (dis_c + 1)), beta)
            # Apply crossover probability to mutate
            beta = torch.where(torch.rand(mu.size(), device=x.device) > pro_c, 1, beta)
            offspring1_dec = (parent1_dec + parent2_dec) / 2 + beta * (parent1_dec - parent2_dec) / 2
            offspring2_dec = (parent1_dec + parent2_dec) / 2 - beta * (parent1_dec - parent2_dec) / 2
            offspring_dec = torch.cat([offspring1_dec, offspring2_dec])
            return offspring_dec

        def _mutation(x: torch.Tensor, dis_m: float, pro_m: float = 1.0) -> torch.Tensor:
            N, D = x.size()
            parent_dec = x
            mu = torch.rand((N, D), device=device)
            offspring_dec = torch.where(mu <= 0.5,
                parent_dec + parent_dec * (torch.pow(2 * mu, 1 / (1 + dis_m)) - 1),
                parent_dec + (1 - parent_dec) * (1 - torch.pow(2 * (1 - mu), 1 / (1 + dis_m)))
            ) 
            offspring_dec = torch.where(torch.rand((N, D), device=device) < pro_m / D, offspring_dec, parent_dec)
            return offspring_dec

        def _ga_reproduction(pop: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
            N = pop.size(-2)
            half_len = N // 2
            main_len = half_len * 2
            perm = torch.randperm(N, device=self.device)
            crossed_offspring_pop = clamp(_crossover(pop[perm], dis_c=mu_c), lb=lb, ub=ub)
            mutated_offspring_pop = clamp(_mutation(crossed_offspring_pop, dis_m=mu_m), lb=lb, ub=ub)
            # Variable swap
            offspring1_dec = mutated_offspring_pop[:half_len, :]
            offspring2_dec = mutated_offspring_pop[half_len:main_len, :]
            swap_mask = torch.rand_like(offspring1_dec) >= 0.5 
            temp = offspring2_dec.clone()
            offspring2_dec = torch.where(swap_mask, offspring1_dec, offspring2_dec)
            offspring1_dec = torch.where(swap_mask, temp, offspring1_dec)
            mutated_offspring_pop[:half_len, :] = offspring1_dec
            mutated_offspring_pop[half_len:main_len, :] = offspring2_dec
            return mutated_offspring_pop

        self.problem = problem

        self.target_pop = target_pop
        self.target_fit = target_fit
        self.candidate_target_pop = candidate_target_pop
        self.candidate_target_fit = candidate_target_fit
        self.reproduction_func = _ga_reproduction

        self.num_source_tasks = num_source_tasks
        self.target_pop_size = target_pop_size
        self.max_generation = max_generation
        self.encoded_dim = encoded_dim
        self.encoded_lb = encoded_lb
        self.encoded_ub = encoded_ub
        self.mu_c = mu_c
        self.mu_m = mu_m
        self.transfer_pop_size = transfer_pop_size
        self.transfer_interval = transfer_interval
        self.device = device

        self.knowledge_base = knowledge_base
        self.knowledge_base_pops = knowledge_base_pops
        self.source_pop_size = N_src
        self.final_generation_source_pops = final_generation_source_pops
        self.current_source_pop_buffer = current_source_pop_buffer 

        self.generation_idx = 0
        self.loaded_generation_idx: int | None = None

    def init_step(self):
        K = self.num_source_tasks

        # Evaluate population for target task
        self.target_fit.copy_(self.problem.evaluate(self.target_pop, eval_task_indices=K))

        # Reset generation index
        self.generation_idx = 0
        self.loaded_generation_idx = None

    def _is_transfer_generation(self) -> bool:
        return self.transfer_pop_size > 0 and ((self.generation_idx + 1) % self.transfer_interval == 0)

    def load_step_data(self) -> None:
        """Load external knowledge required by the next optimization step onto the execution device."""
        if self.generation_idx >= self.max_generation:
            raise RuntimeError("Cannot load step data after all generations have completed.")
        if self.loaded_generation_idx == self.generation_idx:
            return
    
        if self._is_transfer_generation() and (self.generation_idx != (self.max_generation - 1)):
            assert self.current_source_pop_buffer is not None
            self.current_source_pop_buffer.copy_(self.knowledge_base_pops[self.generation_idx])
        self.loaded_generation_idx = self.generation_idx

    def step(self):
        assert self.generation_idx < self.max_generation
        K, N_src, N_tar, S = self.num_source_tasks, self.source_pop_size, self.target_pop_size, self.transfer_pop_size

        if self.loaded_generation_idx != self.generation_idx:
            raise RuntimeError(
                "Step data has not been loaded for the current generation. " 
                "Call `workflow.load_step_data()` before `workflow.step()`."
            )

        # Target task offspring generation
        offspring_pop = self.reproduction_func(pop=self.target_pop, lb=self.encoded_lb, ub=self.encoded_ub)

        # Transfer from source tasks
        if self._is_transfer_generation():
            # Calculate similarity values between sources and target
            if self.generation_idx == (self.max_generation - 1):
                loaded_source_pops_g = self.final_generation_source_pops
            else:
                loaded_source_pops_g = self.current_source_pop_buffer
                assert loaded_source_pops_g is not None
            target_pop_mean_g = torch.mean(self.target_pop, dim=0) # (D,)
            similarity_values_g = torch.empty((K,), dtype=target_pop_mean_g.dtype, device=self.device)

            for source_task_idx in range(K):
                source_pop_means_g = torch.mean(loaded_source_pops_g[source_task_idx], dim=0) # (D,)
                dist_g = torch.norm(source_pop_means_g - target_pop_mean_g)
                similarity_values_g[source_task_idx] = (
                    1 / torch.clamp(dist_g, min=torch.finfo(dist_g.dtype).eps)
                )
            # Select candidate transfer individuals 
            transfer_task_idx = torch.argmax(similarity_values_g, dim=0)
            transfer_pop_indices = torch.randperm(N_src, device=self.device)[:S]
            candidate_transfer_indivs = self.final_generation_source_pops[transfer_task_idx, transfer_pop_indices] # (S, D)
            # Update offspring population by candidate transfer individuals
            replace_pop_indices = torch.randperm(N_tar, device=self.device)[:S]
            offspring_pop[replace_pop_indices] = candidate_transfer_indivs

        # Target task offspring evaluation
        offspring_fit = self.problem.evaluate(offspring_pop, eval_task_indices=K)
        self.candidate_target_pop[:N_tar] = self.target_pop
        self.candidate_target_pop[N_tar:] = offspring_pop
        self.candidate_target_fit[:N_tar] = self.target_fit
        self.candidate_target_fit[N_tar:] = offspring_fit

        # Selection
        selected_indices = torch.topk(self.candidate_target_fit, k=N_tar, largest=False).indices
        self.target_pop.copy_(self.candidate_target_pop[selected_indices])
        self.target_fit.copy_(self.candidate_target_fit[selected_indices])

        # Update generation index
        self.generation_idx += 1
