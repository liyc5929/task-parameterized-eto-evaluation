<div align="center">
  
---

<h3>
Towards Efficient Evaluation of Evolutionary Transfer Optimization: Case Studies on Task-Parameterized Applications
</h3>

---

</div>

This repository provides implementations and experiment scripts for studying problem-side evaluation scaling in task-parameterized evolutionary transfer optimization (ETO). It includes two representative applications: *kinematic-arm multi-task optimization* and *B-spline trajectory sequential transfer optimization*, together with their reference evaluation forms and reformulations for parallel execution. For end-to-end experiments, mean alignment multi-task optimization (MA-MTO) and mean similarity sequential transfer optimization (MS-STO) are used for the kinematic-arm and trajectory cases, respectively. The experiments cover numerical agreement, evaluation runtime, and end-to-end optimization results and runtime.

- **Accepted manuscript** available on *arXiv*:  
  [arXiv:2609.05040](https://arxiv.org/abs/2609.05040)
  

## Environment Setup

All project dependencies are specified in [`pyproject.toml`](./pyproject.toml), with the exact dependency resolution used for this release recorded in [`uv.lock`](./uv.lock). We recommend using [uv](https://github.com/astral-sh/uv) to create and synchronize the environment. The commands below assume a Linux environment:
```bash
uv venv
source .venv/bin/activate
uv sync --frozen --extra vis
```
The `vis` extra includes the dependencies required for result visualization. If `uv` is not available, the complete dependency specification can be obtained directly from `pyproject.toml` and installed using another Python environment manager.

> **Note:** In some systems, the environment may be installed successfully but experiment execution may fail because the installed PyTorch build is incompatible with the local GPU driver. If a CUDA- or driver-related compatibility error occurs, try using the following PyTorch version in `pyproject.toml`:
>
> ```toml
> dependencies = [
>   "torch==2.6.0",
>   # Other dependencies are unchanged.
> ]
> ```
>
> Then recreate or resynchronize the environment.


## Experiment Reproduction

The scripts in [`scripts`](./scripts) reproduce the numerical agreement, evaluation runtime, and end-to-end optimization experiments for both application cases.

### Numerical Agreement

```bash
bash ./scripts/run_kinematic_arm_reference_agreement.sh
bash ./scripts/run_b_spline_trajectory_reference_agreement.sh
```

### Evaluation Runtime

```bash
bash ./scripts/run_kinematic_arm_runtime_evaluation.sh
bash ./scripts/run_b_spline_trajectory_runtime_evaluation.sh
```

### End-to-End Optimization Result

```bash
bash ./scripts/run_ma_mto_task_scaling_result.sh
bash ./scripts/run_ms_sto_problem_scaling_result.sh
```

### End-to-End Optimization Runtime
```bash
bash ./scripts/run_ma_mto_task_scaling_runtime.sh
bash ./scripts/run_ms_sto_problem_scaling_runtime.sh
```

## Citation
If you find this work useful, please consider citing:

```bibtex
@misc{li2026taskparameterizedeto,
  title         = {Towards efficient evaluation of evolutionary transfer optimization: {Case} studies on task-parameterized applications},
  author        = {Yanchen Li and Xiaoming Xue and Kay Chen Tan},
  year          = {2026},
  eprint        = {2609.05040},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.05040},
}
