import csv
from pathlib import Path

import torch

from algorithms import MS_STO, build_sto_knowledge_database, GAKnowledgeBuildingStrategy
from problems import BSplineTrajectory


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_source_tasks", type=int, default=5000)
    parser.add_argument("--pop_size", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=60)
    parser.add_argument("--problem_resolution", type=int, default=6000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_headers", action="store_true", default=False)
    parser.add_argument("--draw_best_result", action="store_true", default=False)
    parser.add_argument("--draw_path", type=str, default=".")
    args = parser.parse_args()

    PROBLEM_BUILD_SEED = 0
    KNOWLEDGE_BUILD_SEED = 0
    experiment_seed = args.seed
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    K = args.num_source_tasks
    N = args.pop_size
    D = args.dimension
    R = args.problem_resolution
    G = 100
    G_max = 100
    data_root = "./data/ms_sto_b_spline_trajectory/"

    problem = BSplineTrajectory(
        num_source_tasks=K,
        dim=D,
        num_evaluation_points=R,
        seed=PROBLEM_BUILD_SEED,
        save_path=data_root,
        force_rebuild=False,
        device=device,
    )
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
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
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
            "Seed,Algorithm,Device,Problem,Num Source Tasks,Population Size,"
            "Dimension,Resolution,max generation,Device Info,Target Best Fitness"
        )

    algorithm.init_step()
    for g in range(G):
        algorithm.load_step_data()
        algorithm.step()

    final_target_best_fitness = algorithm.target_fit.min(dim=0).values
    best_individual_idx = torch.argmin(algorithm.target_fit)
    best_internal_control_y = algorithm.target_pop[best_individual_idx]
    control_points = algorithm.target_pop.new_empty((problem.num_control_points, 2))
    control_points[:, 0] = problem.control_x
    control_points[1:-1, 1] = best_internal_control_y
    control_points[0, 1] = 0.0
    control_points[-1, 1] = 1.0
    trajectory_points = problem.basis_matrix @ control_points

    target_task_idx = K
    target_obstacles = problem.obstacle_centers[target_task_idx]
    target_obstacle_mask = problem.obstacle_mask[target_task_idx]
    trajectory_segments = trajectory_points[1:] - trajectory_points[:-1]
    segment_starts = trajectory_points[:-1, None, :]
    segment_directions = trajectory_segments[:, None, :]
    obstacle_min = target_obstacles[None, :, :] - problem.half_obstacle_size
    obstacle_max = target_obstacles[None, :, :] + problem.half_obstacle_size
    parallel_axes = torch.abs(segment_directions) <= torch.finfo(segment_directions.dtype).eps
    safe_directions = torch.where(
        parallel_axes,
        torch.ones_like(segment_directions),
        segment_directions,
    )
    t_a = (obstacle_min - segment_starts) / safe_directions
    t_b = (obstacle_max - segment_starts) / safe_directions
    t_near = torch.minimum(t_a, t_b)
    t_far = torch.maximum(t_a, t_b)
    t_near = torch.where(
        parallel_axes,
        torch.full_like(t_near, -torch.inf),
        t_near,
    )
    t_far = torch.where(
        parallel_axes,
        torch.full_like(t_far, torch.inf),
        t_far,
    )
    parallel_inside = (
        (~parallel_axes) | (
            (segment_starts >= obstacle_min) & (segment_starts <= obstacle_max)
        )
    )
    entry_t = torch.maximum(
        t_near.amax(dim=2),
        torch.zeros_like(t_near[:, :, 0]),
    )
    exit_t = torch.minimum(
        t_far.amin(dim=2),
        torch.ones_like(t_far[:, :, 0]),
    )
    segment_intersects_obstacles = (parallel_inside.all(dim=2) & (entry_t <= exit_t))
    collided_obstacles = torch.any(
        segment_intersects_obstacles & target_obstacle_mask[None, :],
        dim=0,
    )
    final_collision_count_tensor = collided_obstacles.sum()
    final_collision_count = int(final_collision_count_tensor.detach().cpu())
    assert 0 <= final_collision_count <= target_obstacle_mask.shape[0]

    print(
        f"{experiment_seed},{algorithm.__class__.__name__},{device.type},{problem.__class__.__name__},{K},{N},"
        f"{problem.dim},{problem.num_evaluation_points},{G},"
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else get_cpu_name()},"
        f"{final_target_best_fitness.item()}"
    )
    print(
        f"Final collision count for seed {experiment_seed}: "
        f"{final_collision_count}."
    )
    if final_collision_count > 0:
        print(f"Warning: the final best trajectory still intersects {final_collision_count} active obstacle(s).")


    if args.draw_best_result:
        import os
        import shutil
        import subprocess
        import tempfile

        draw_path = Path(args.draw_path).expanduser()
        if draw_path.suffix.lower() != ".pdf":
            draw_path = draw_path.with_suffix(".pdf")
        draw_path.parent.mkdir(parents=True, exist_ok=True)
        draw_data_path = draw_path.with_name(f"{draw_path.stem}_data.csv")

        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager
        from matplotlib.lines import Line2D
        from matplotlib.path import Path as MplPath
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

        def _font_available(font_name: str) -> bool:
            try:
                font_manager.findfont(font_manager.FontProperties(family=[font_name]), fallback_to_default=False)
            except ValueError:
                return False
            return True

        times_install_attempted = False
        times_install_succeeded = False
        if not _font_available("Times New Roman"):
            apt_get = shutil.which("apt-get")
            sudo = shutil.which("sudo")
            command_prefix = []
            if hasattr(os, "geteuid") and os.geteuid() != 0:
                command_prefix = [sudo, "-n"] if sudo is not None else None

            if apt_get is not None and command_prefix is not None:
                times_install_attempted = True
                install_environment = os.environ.copy()
                install_environment["DEBIAN_FRONTEND"] = "noninteractive"
                try:
                    subprocess.run(
                        command_prefix + [apt_get, "update"],
                        env=install_environment,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=120,
                        check=False,
                    )
                    install_result = subprocess.run(
                        command_prefix + [
                            apt_get,
                            "install",
                            "-y",
                            "ttf-mscorefonts-installer",
                            "fontconfig",
                        ],
                        env=install_environment,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=120,
                        check=False,
                    )
                    times_install_succeeded = install_result.returncode == 0
                except (OSError, subprocess.SubprocessError):
                    times_install_succeeded = False

                fc_cache = shutil.which("fc-cache")
                if fc_cache is not None:
                    try:
                        subprocess.run(
                            [fc_cache, "-f"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=60,
                            check=False,
                        )
                    except (OSError, subprocess.SubprocessError):
                        pass
                try:
                    font_manager.fontManager = font_manager._load_fontmanager(try_read_cache=False)
                    font_manager.findfont = font_manager.fontManager.findfont
                except (AttributeError, OSError):
                    pass

        font_candidates = (
            "Times New Roman",
            "STIX Two Text",
            "Liberation Serif",
            "DejaVu Serif",
        )
        selected_font = next(
            (font_name for font_name in font_candidates if _font_available(font_name)),
            "DejaVu Serif",
        )
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = [selected_font]
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42

        target_obstacles = target_obstacles[target_obstacle_mask]

        target_obstacles_cpu = target_obstacles.detach().cpu()
        control_points_cpu = control_points.detach().cpu()
        trajectory_points_cpu = trajectory_points.detach().cpu()
        final_best_fitness_value = final_target_best_fitness.detach().cpu().item()

        segment_lengths_cpu = torch.linalg.vector_norm(
            trajectory_points_cpu[1:] - trajectory_points_cpu[:-1],
            dim=1,
        )
        cumulative_lengths = torch.cat((
            trajectory_points_cpu.new_zeros((1,)),
            torch.cumsum(segment_lengths_cpu, dim=0),
        ))
        total_length = float(cumulative_lengths[-1])
        assert R >= 4, "At least four trajectory points are required to draw the arrow."
        arrow_start_index = int(torch.searchsorted(cumulative_lengths, 0.92 * total_length))
        arrow_end_index = int(torch.searchsorted(cumulative_lengths, 0.99 * total_length))
        arrow_start_index = min(max(arrow_start_index, 0), R - 4)
        arrow_end_index = min(max(arrow_end_index, arrow_start_index + 2), R - 2)
        assert (0 <= arrow_start_index and arrow_start_index + 2 <= arrow_end_index < R - 1)

        arrow_vertices = trajectory_points_cpu[arrow_start_index:arrow_end_index + 1].numpy()
        arrow_codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(arrow_vertices) - 1)
        arrow_path = MplPath(arrow_vertices, arrow_codes)

        plot_metadata = {
            "seed": experiment_seed,
            "target_task_idx": target_task_idx,
            "final_best_fitness": final_best_fitness_value,
            "final_collision_count": final_collision_count,
            "arrow_start_index": arrow_start_index,
            "arrow_end_index": arrow_end_index,
            "num_source_tasks": K,
            "population_size": N,
            "dimension": D,
            "resolution": R,
            "generation": algorithm.generation_idx,
        }
        plot_fieldnames = [
            "seed",
            "target_task_idx",
            "final_best_fitness",
            "final_collision_count",
            "arrow_start_index",
            "arrow_end_index",
            "num_source_tasks",
            "population_size",
            "dimension",
            "resolution",
            "generation",
            "record_type",
            "index",
            "role",
            "x",
            "y",
            "obstacle_size",
        ]
        with draw_data_path.open("w", newline="", encoding="utf-8") as plot_data_file:
            writer = csv.DictWriter(plot_data_file, fieldnames=plot_fieldnames)
            writer.writeheader()
            for obstacle_idx, obstacle_center in enumerate(target_obstacles_cpu):
                writer.writerow({
                    **plot_metadata,
                    "record_type": "obstacle",
                    "index": obstacle_idx,
                    "role": "",
                    "x": float(obstacle_center[0]),
                    "y": float(obstacle_center[1]),
                    "obstacle_size": problem.obstacle_size,
                })
            for control_point_idx, control_point in enumerate(control_points_cpu):
                role = (
                    "start"
                    if control_point_idx == 0
                    else "goal"
                    if control_point_idx == problem.num_control_points - 1
                    else "internal"
                )
                writer.writerow({
                    **plot_metadata,
                    "record_type": "control_point",
                    "index": control_point_idx,
                    "role": role,
                    "x": float(control_point[0]),
                    "y": float(control_point[1]),
                    "obstacle_size": "",
                })
            for trajectory_point_idx, trajectory_point in enumerate(trajectory_points_cpu):
                role = (
                    "start"
                    if trajectory_point_idx == 0
                    else "goal"
                    if trajectory_point_idx == R - 1
                    else "internal"
                )
                writer.writerow({
                    **plot_metadata,
                    "record_type": "trajectory_point",
                    "index": trajectory_point_idx,
                    "role": role,
                    "x": float(trajectory_point[0]),
                    "y": float(trajectory_point[1]),
                    "obstacle_size": "",
                })

        axes_face = "#F6F3EF"
        obstacle_face = "#4A4542"
        obstacle_edge = "#2F2B29"
        trajectory_color = "#B55A6A"
        start_face = "#D9B44A"
        start_edge = "#8B6A1E"
        goal_face = "#C46E3B"
        goal_edge = "#7A4525"
        spine_color = "#383430"
        grid_color = "#FFFFFF"

        fig, ax = plt.subplots(figsize=(5.2, 5.1))
        fig.subplots_adjust(
            left=0.12,
            right=0.97,
            bottom=0.11,
            top=0.83,
        )
        ax.set_facecolor(axes_face)
        for obstacle_center in target_obstacles_cpu:
            obstacle_patch = Rectangle(
                (
                    float(obstacle_center[0]) - problem.half_obstacle_size,
                    float(obstacle_center[1]) - problem.half_obstacle_size,
                ),
                problem.obstacle_size,
                problem.obstacle_size,
                facecolor=obstacle_face,
                edgecolor=obstacle_edge,
                linewidth=0.65,
                alpha=1.0,
                zorder=2,
            )
            ax.add_patch(obstacle_patch)

        ax.plot(
            trajectory_points_cpu[:, 0],
            trajectory_points_cpu[:, 1],
            color=trajectory_color,
            linewidth=1.6,
            linestyle=(0, (1.5, 1.0)),
            zorder=3,
        )
        direction_arrow = FancyArrowPatch(
            path=arrow_path,
            arrowstyle="-|>,head_length=0.4,head_width=0.15",
            mutation_scale=13,
            linewidth=1.8,
            color=trajectory_color,
            zorder=5,
        )
        ax.add_patch(direction_arrow)
        ax.scatter(
            trajectory_points_cpu[0, 0],
            trajectory_points_cpu[0, 1],
            s=95,
            color=start_face,
            edgecolor=start_edge,
            linewidth=0.8,
            marker="^",
            zorder=6,
        )
        ax.scatter(
            trajectory_points_cpu[-1, 0],
            trajectory_points_cpu[-1, 1],
            s=155,
            color=goal_face,
            edgecolor=goal_edge,
            linewidth=0.8,
            marker="*",
            zorder=6,
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel("$y$", fontsize=12)
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_axisbelow(True)
        ax.grid(
            True,
            color=grid_color,
            linewidth=1.0,
            alpha=0.7,
        )
        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            color=spine_color,
            labelcolor=spine_color,
            labelsize=12,
            width=0.7,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(spine_color)
            spine.set_linewidth(0.8)

        legend_handles = [
            Line2D(
                [],
                [],
                linestyle="None",
                marker="^",
                markersize=9.5,
                markerfacecolor=start_face,
                markeredgecolor=start_edge,
                markeredgewidth=0.8,
            ),
            Line2D(
                [],
                [],
                linestyle="None",
                marker="*",
                markersize=12.5,
                markerfacecolor=goal_face,
                markeredgecolor=goal_edge,
                markeredgewidth=0.8,
            ),
            Patch(
                facecolor=obstacle_face,
                edgecolor=obstacle_edge,
                linewidth=0.65,
            ),
            Line2D(
                [],
                [],
                color=trajectory_color,
                linewidth=1.6,
                linestyle=(0, (1.8, 1.4)),
            ),
        ]
        legend_labels = [
            "Start Position",
            "Goal Position",
            "Obstacles",
            "Optimized Trajectory",
        ]
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=2,
            fontsize=12,
            frameon=False,
            columnspacing=2.0,
            handletextpad=0.45,
            handlelength=1.6,
            borderaxespad=0.0,
        )
        fig.savefig(draw_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        if times_install_attempted:
            print(
                "Times New Roman installation "
                f"{'succeeded' if times_install_succeeded else 'did not succeed'}."
            )
        print(f"Trajectory plot font: {selected_font}.")
        print(f"Trajectory figure is saved into `{draw_path}`.")
        print(f"Trajectory plot data is saved into `{draw_data_path}`.")
