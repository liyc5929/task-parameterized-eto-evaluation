from typing import Optional

import torch
from torch import Tensor


def generate_kinematic_arm_task_descriptors(
    num_tasks: int,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[str | torch.device] = None,
) -> Tensor:
    """
    Generate normalized task descriptors [L, alpha_max] in [0, 1)^2, where L is the
    normalized total arm length and alpha_max is the normalized joint-angle range.

    Tasks are sampled by a scrambled Sobol sequence for low-discrepancy coverage of the 2-D task space.
    """
    assert num_tasks > 0
    assert dtype in (torch.float32, torch.float64)
    assert seed is None or (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0)
    device = torch.get_default_device() if device is None else torch.device(device)

    effective_seed = (
        torch.randint(0, 2**31 - 1, (), device="cpu").item()
        if seed is None
        else seed
    )

    engine = torch.quasirandom.SobolEngine(
        dimension=2,
        scramble=True,
        seed=effective_seed,
    )
    return engine.draw(num_tasks, dtype=dtype).to(device=device)
