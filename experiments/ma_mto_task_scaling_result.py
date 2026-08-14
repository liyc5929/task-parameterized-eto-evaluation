import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from problems import KinematicArm, KinematicArm_MatrixReference
from algorithms import MA_MTO


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tasks", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_headers", action="store_true", default=False)
    args = parser.parse_args()

    experiment_seed = args.seed
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    K = args.num_tasks
    N = 16
    D = 60
    G = 100

    problem = KinematicArm(
        num_tasks=K,
        dim=D,
        task_seed=experiment_seed,
        device=device,
    )
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    encoded_lb = torch.tensor(0.0, device=device)
    encoded_ub = torch.tensor(1.0, device=device)
    algorithm = MA_MTO(
        problem           = problem,
        num_tasks         = K, 
        task_pop_size     = N, 
        task_dim          = D,
        task_lb           = encoded_lb, 
        task_ub           = encoded_ub, 
        minimization      = False,
        transfer_pop_size = 8,
        transfer_interval = 5,
        ga_mu_c           = 2.0,
        ga_mu_m           = 5.0,
        device            = device,
    )

    def get_cpu_name():
        import re
        import subprocess
        info = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode()
        return re.search(r"model name\s*:\s*(.*)", info).group(1)

    if not args.no_headers:
        print(
            "Seed,Problem,Algorithm,Device,Num Tasks,Population Size,Dimension,Num Generations,Device Info,",
            end="",
        )
        print(",".join([f"Task {k}" for k in range(K)]))
    device_info = torch.cuda.get_device_name(device) if device.type == "cuda" else get_cpu_name()
    print(
        f"{experiment_seed},{problem.__class__.__name__},{algorithm.__class__.__name__},"
        f"{device.type},{K},{N},{D},{G},{device_info}", 
        end="",
    )

    algorithm.init_step()
    for g in range(G):
        algorithm.step()

    for k in range(K):
        print(f",{algorithm.best_fitnesses[k].item():.6f}", end="")
    print()
