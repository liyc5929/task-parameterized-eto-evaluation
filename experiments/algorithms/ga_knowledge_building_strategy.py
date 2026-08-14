import torch
from typing import Sequence, Optional
from tqdm import tqdm

from evox.core import Problem
from evox.operators.sampling import latin_hypercube_sampling
from evox.utils import clamp
from tensordict import TensorDictBase

from ._sto_knowledge_base import KnowledgeType


__all__ = [
    "GAKnowledgeBuildingStrategy",
]


class GAKnowledgeBuildingStrategy:
    """
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

    `STRATEGY_VERSION` is incremented when implementation changes alter the
    generated knowledge under otherwise identical settings.
    """
    STRATEGY_VERSION = 1

    def __init__(self, 
        num_source_tasks: int,
        source_pop_sizes: Sequence[int],
        source_dims: Sequence[int],
        source_lbs: Sequence[torch.Tensor],
        source_ubs: Sequence[torch.Tensor],
        encoded_lb: torch.Tensor,
        encoded_ub: torch.Tensor,
        max_generation: int = 1,
        mu_c: float = 15.0,
        mu_m: float = 15.0,
        show_progress: bool = False,
        device: Optional[str | torch.device] = None,
    ) -> None:
        assert num_source_tasks == len(source_pop_sizes) == len(source_dims) == len(source_lbs) == len(source_ubs)
        assert max_generation >= 1
        G, K, N_max, D_max = max_generation, num_source_tasks, max(source_pop_sizes), max(source_dims)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        source_lbs = [lb.to(device) for lb in source_lbs]
        source_ubs = [ub.to(device) for ub in source_ubs]
        encoded_lb = encoded_lb.to(device)
        encoded_ub = encoded_ub.to(device)

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
    
        def _ea_reproduction(pop: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
            N = pop.size(-2)
            half_len = N // 2
            main_len = half_len * 2
            perm = torch.randperm(N, device=device)
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

        def _encode_population(pop: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
            return (pop - lb) / (ub - lb) * (encoded_ub - encoded_lb) + encoded_lb 

        self.reproduction_func   = _ea_reproduction
        self.domain_encode_func  = _encode_population

        self.num_source_tasks = num_source_tasks
        self.source_pop_sizes = source_pop_sizes
        self.source_dims = source_dims
        self.source_lbs = source_lbs
        self.source_ubs = source_ubs
        self.encoded_lb = encoded_lb
        self.encoded_ub = encoded_ub
        self.max_generation = max_generation
        self.mu_c = mu_c
        self.mu_m = mu_m
        self.show_progress = show_progress
        self.device = device
        dtype_name = str(encoded_lb.dtype).removeprefix("torch.")
        self.knowledge_spec = {
            "knowledge_type": KnowledgeType.SOLUTION_POPULATION.value,
            "builder": {
                "name": self.__class__.__name__,
                "version": self.STRATEGY_VERSION,
                "mu_c": mu_c,
                "mu_m": mu_m,
            },
            "dimensions": {
                "num_sources": K,
                "num_generations": G,
                "population_size": N_max,
                "encoded_dimension": D_max,
            },
            "schema": {
                "batch_size": [G, K],
                "fields": {
                    "population": {"shape": [N_max, D_max], "dtype": dtype_name},
                    "fitness": {"shape": [N_max,], "dtype": dtype_name},
                },
            },
            "encoded_domain": {
                "lower_bound": encoded_lb.detach().cpu().item(),
                "upper_bound": encoded_ub.detach().cpu().item(),
            },
        }

    def __call__(self, problem: Problem, knowledge_tensors: TensorDictBase) -> None:
        G, K = self.max_generation, self.num_source_tasks
        knowledge_base_pops = knowledge_tensors["population"]
        knowledge_base_fits = knowledge_tensors["fitness"]
        N_max, D_max = knowledge_base_pops.shape[-2:]

        iterable = zip(self.source_pop_sizes, self.source_dims, self.source_lbs, self.source_ubs)
        for k, (N_k, D_k, lb_k, ub_k) in tqdm(enumerate(iterable),
            total=K, desc="Building STO knowledge database", disable=not self.show_progress,
        ):
            if N_k < N_max:
                knowledge_base_pops[:, k, N_k:].zero_()
                knowledge_base_fits[:, k, N_k:].fill_(torch.inf)
            if D_k < D_max:
                knowledge_base_pops[:, k, :N_k, D_k:].zero_()
            # In generation 0
            pop_k: torch.Tensor = latin_hypercube_sampling(n=N_k, lb=lb_k.expand(D_k), ub=ub_k.expand(D_k), smooth=True)
            fit_k: torch.Tensor = problem.evaluate(pop_k, eval_task_indices=k)
            knowledge_base_pops[0, k, :N_k, :D_k].copy_(pop_k)
            knowledge_base_fits[0, k, :N_k].copy_(fit_k)
    
            # In generation 1 to G-1
            for g in range(1, G):
                # Offspring generation
                offspring_pop_k = self.reproduction_func(pop=pop_k, lb=lb_k, ub=ub_k)
                # Offspring evaluation
                offspring_fit_k = problem.evaluate(offspring_pop_k, eval_task_indices=k)
                # Selection
                candidate_pop_k = torch.cat([pop_k, offspring_pop_k], dim=0)
                candidate_fit_k = torch.cat([fit_k, offspring_fit_k], dim=0)
                selected_indices = torch.topk(candidate_fit_k, k=N_k, largest=False).indices
                pop_k.copy_(candidate_pop_k[selected_indices])
                fit_k.copy_(candidate_fit_k[selected_indices])
                knowledge_base_pops[g, k, :N_k, :D_k].copy_(self.domain_encode_func(pop=pop_k, lb=lb_k, ub=ub_k))
                knowledge_base_fits[g, k, :N_k].copy_(fit_k)
