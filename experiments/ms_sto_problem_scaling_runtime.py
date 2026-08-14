import torch
import time

from algorithms import MS_STO, build_sto_knowledge_database, GAKnowledgeBuildingStrategy
from problems import BSplineTrajectory, BSplineTrajectory_PointwiseReference


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_source_tasks", type=int, default=5000)
    parser.add_argument("--pop_size", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=60)
    parser.add_argument("--problem_resolution", type=int, default=6000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--problem_realization", type=str, choices=["pointwise", "parallel"])
    parser.add_argument("--no_headers", action="store_true", default=False)
    args = parser.parse_args()

    PROBLEM_BUILD_SEED = 0
    KNOWLEDGE_BUILD_SEED = 0
    experiment_seed = 0
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    K = args.num_source_tasks
    N = args.pop_size
    D = args.dimension
    R = args.problem_resolution
    G = 2
    G_max = 100
    data_root = "./data/ms_sto_b_spline_trajectory/"

    if args.problem_realization == "pointwise":
        problem = BSplineTrajectory_PointwiseReference(
            num_source_tasks=K,
            dim=D,
            num_evaluation_points=R,
            seed=PROBLEM_BUILD_SEED,
            save_path=data_root,
            force_rebuild=False,
            device=device,
        )
    elif args.problem_realization == "parallel":
        problem = BSplineTrajectory(
            num_source_tasks=K,
            dim=D,
            num_evaluation_points=R,
            seed=PROBLEM_BUILD_SEED,
            save_path=data_root,
            force_rebuild=False,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported problem realization: {args.problem_realization}.")

    encoded_lb = torch.tensor(0.0, device=device)
    encoded_ub = torch.tensor(1.0, device=device)
    building_strategy = GAKnowledgeBuildingStrategy(
        num_source_tasks = K,
        source_pop_sizes = K * (N,),
        source_dims      = K * (problem.dim,),
        source_lbs       = K * (encoded_lb,),
        source_ubs       = K * (encoded_ub,),
        encoded_lb       = encoded_lb,
        encoded_ub       = encoded_ub,
        max_generation   = G_max,
        mu_c             = 15.0,
        mu_m             = 15.0,
        show_progress    = False,
        device           = device,
    )
    knowledge_base, _, __ = build_sto_knowledge_database(
        problem          = BSplineTrajectory(
            num_source_tasks=K, dim=D, num_evaluation_points=R, seed=PROBLEM_BUILD_SEED,
            save_path=data_root, force_rebuild=False, device=device,
        ),
        num_source_tasks = K,
        strategy         = building_strategy,
        build_seed       = KNOWLEDGE_BUILD_SEED,
        data_root        = data_root,
        problem_key      = f"BSplineTrajectory-problem_seed{PROBLEM_BUILD_SEED}-resolution{R}",
        force_rebuild    = False,
    )
    algorithm = MS_STO(
        problem           = problem,
        num_source_tasks  = K, 
        target_pop_size   = N, 
        max_generation    = G_max,
        encoded_dim       = problem.dim,
        encoded_lb        = encoded_lb, 
        encoded_ub        = encoded_ub, 
        mu_c              = 15.0, 
        mu_m              = 15.0,
        knowledge_base    = knowledge_base,
        transfer_pop_size = 8,
        transfer_interval = 1,
        device            = device,
    )

    def get_cpu_name():
        import re
        import subprocess
        info = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode()
        return re.search(r"model name\s*:\s*(.*)", info).group(1)

    if not args.no_headers:
        print(
            "Algorithm,Device,Problem,Num Source Tasks,Population Size,"
            "Dimension,Resolution,Seed,Device Info,Runtime"
        )
    print(
        f"{algorithm.__class__.__name__},{device.type},{problem.__class__.__name__},{K},{N},"
        f"{problem.dim},{problem.num_evaluation_points},{experiment_seed},"
        f"{torch.cuda.get_device_name() if device.type == 'cuda' else get_cpu_name()}", 
        end="",
    )

    algorithm.init_step()
    for g in range(G):
        algorithm.load_step_data()

        if g != 0:
            if torch.cuda.is_available(): torch.cuda.synchronize(device)
            start_time = time.perf_counter()

        algorithm.step()

        if g != 0:
            if torch.cuda.is_available(): torch.cuda.synchronize(device)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f",{run_time}")
