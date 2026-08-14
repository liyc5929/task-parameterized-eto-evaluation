# task-parameterized-eto-evaluation

```bash
uv venv
source .venv/bin/activate
uv sync --extra vis
```

Numerical Agreement
```bash
bash ./scripts/run_kinematic_arm_reference_agreement.sh
bash ./scripts/run_b_spline_trajectory_reference_agreement.sh
```

Evaluation Runtime
```bash
bash ./scripts/run_kinematic_arm_runtime_evaluation.sh
bash ./scripts/run_b_spline_trajectory_runtime_evaluation.sh
```

Optimization Result
```bash
bash ./scripts/run_ma_mto_task_scaling_result.sh

bash ./scripts/run_ms_sto_problem_scaling_result.sh
bash ./scripts/draw_ms_sto_existing_trajectory.sh
```

Optimization Runtime
```bash
bash ./scripts/run_ma_mto_task_scaling_runtime.sh
bash ./scripts/run_ms_sto_problem_scaling_runtime.sh
```
