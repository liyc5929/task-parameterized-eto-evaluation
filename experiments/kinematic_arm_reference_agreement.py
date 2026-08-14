import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from problems import KinematicArm, KinematicArm_MatrixReference


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tasks", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--problem_realization", type=str, choices=["matrix", "parallel"], required=True)
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

    if not args.no_headers:
        fitness_headers = ",".join([f"Fitness {n}" for n in range(N)])
        print(
            f"Seed,Device,Device Info,Problem Realization,Problem,Num Tasks,Population Size,Dimension,"
            f"Task Index,{fitness_headers}"
        )

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
    task_fits = problem.evaluate(task_pops)
    task_fits = torch.stack(task_fits, dim=0)
    assert task_fits.shape == (K, N)

    device_info = torch.cuda.get_device_name(device) if device.type == "cuda" else get_cpu_name()
    task_fits_cpu = task_fits.detach().cpu()
    for k in range(K):
        fitness_values = ",".join(f"{task_fits_cpu[k, n].item():.10e}" for n in range(N))
        print(
            f"{args.seed},{device.type},{device_info},{args.problem_realization},{problem.__class__.__name__},"
            f"{K},{N},{D},{k},{fitness_values}"
        )
