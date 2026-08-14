import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import time

from problems import KinematicArm, KinematicArm_MatrixReference


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=5000)
    parser.add_argument("--pop_size", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=60)
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
    N = args.pop_size
    D = args.dimension

    def get_cpu_name():
        import re
        import subprocess
        info = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode()
        return re.search(r"model name\s*:\s*(.*)", info).group(1)

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

    # Reset RNG before sampling task populations to keep inputs identical across realizations
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    task_pops = torch.rand((K, N, D), device=device, dtype=problem.task_descriptors.dtype)

    if torch.cuda.is_available(): torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    task_fits = problem.evaluate(task_pops)
    if torch.cuda.is_available(): torch.cuda.synchronize(device)
    end_time = time.perf_counter()
    run_time = end_time - start_time

    task_fits = torch.stack(task_fits, dim=0)
    assert task_fits.shape == (K, N)

    if not args.no_headers:
        print("Problem,Problem Realization,Device,Num Tasks,Population Size,Dimension,Seed,Device Info,Runtime")
    device_info = torch.cuda.get_device_name(device) if device.type == "cuda" else get_cpu_name()
    print(
        f"{problem.__class__.__name__},{args.problem_realization},"
        f"{device.type},{K},{N},{D},{experiment_seed},{device_info}," 
        f"{run_time}",
    )
