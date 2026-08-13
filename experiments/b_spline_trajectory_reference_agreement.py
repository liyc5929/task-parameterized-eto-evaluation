import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from problems import BSplineTrajectory, BSplineTrajectory_PointwiseReference


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--problem_resolution", type=int, default=6000)
    parser.add_argument("--problem_realization", type=str, choices=["pointwise", "parallel"], required=True)
    parser.add_argument("--no_headers", action="store_true", default=False)
    args = parser.parse_args()

    experiment_seed = args.seed
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    K = 1
    N = 16
    D = 60
    R = args.problem_resolution

    if not args.no_headers:
        fitness_headers = ",".join([f"Fitness {n}" for n in range(N)])
        print(
            f"Seed,Device,Device Info,Problem Realization,Problem,Num Tasks,Population Size,Dimension,Resolution,"
            f"{fitness_headers}"
        )

    def get_cpu_name():
        import re
        import subprocess
        info = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode()
        return re.search(r"model name\s*:\s*(.*)", info).group(1)

    if args.problem_realization == "pointwise":
        problem = BSplineTrajectory_PointwiseReference(
            num_source_tasks=K,
            dim=D,
            num_evaluation_points=R,
            seed=experiment_seed,
            device=device,
        )
    elif args.problem_realization == "parallel":
        problem = BSplineTrajectory(
            num_source_tasks=K,
            dim=D,
            num_evaluation_points=R,
            seed=experiment_seed,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported problem realization: {args.problem_realization}.")

    # Reset RNG before sampling task populations to keep inputs identical across realizations
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    pop = torch.rand((N, D), device=device, dtype=problem.obstacle_centers.dtype)
    fit = problem.evaluate(pop, eval_task_indices=0)
    assert fit.shape == (N,)

    device_info = torch.cuda.get_device_name(device) if device.type == "cuda" else get_cpu_name()
    fit_cpu = fit.detach().cpu()
    fitness_values = ",".join(f"{fit_cpu[n].item():.10e}" for n in range(N))
    print(
        f"{args.seed},{device.type},{device_info},{args.problem_realization},{problem.__class__.__name__},"
        f"{K},{N},{D},{R},{fitness_values}"
    )
