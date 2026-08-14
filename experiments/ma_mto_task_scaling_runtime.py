import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import time

from problems import KinematicArm, KinematicArm_MatrixReference
from algorithms import MA_MTO


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--problem_realization", type=str, choices=["matrix", "parallel"], required=True)
    parser.add_argument("--no_headers", action="store_true", default=False)
    args = parser.parse_args()

    experiment_seed = 0
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    K = args.num_tasks
    N = 16
    D = 60
    G = 2

    if args.problem_realization == "matrix":
        problem = KinematicArm_MatrixReference(
            num_tasks=K,
            dim=D,
            task_seed=experiment_seed,
            device=device,
        )
    elif args.problem_realization == "parallel":
        problem = KinematicArm(
            num_tasks=K,
            dim=D,
            task_seed=experiment_seed,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported problem realization: {args.problem_realization}.")

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
        transfer_pop_size = 8,
        transfer_interval = 1,
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
        print("Problem,Problem Realization,Algorithm,Device,Num Tasks,Population Size,Dimension,Seed,Device Info,Runtime")
    device_info = torch.cuda.get_device_name(device) if device.type == "cuda" else get_cpu_name()
    print(
        f"{problem.__class__.__name__},{args.problem_realization},{algorithm.__class__.__name__},"
        f"{device.type},{K},{N},{D},{experiment_seed},{device_info}", 
        end="",
    )

    algorithm.init_step()
    for g in range(G):
        if g != 0:
            if torch.cuda.is_available(): torch.cuda.synchronize(device)
            start_time = time.perf_counter()

        algorithm.step()

        if g != 0:
            if torch.cuda.is_available(): torch.cuda.synchronize(device)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f",{run_time}")
