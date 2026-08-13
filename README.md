# task-parameterized-eto-evaluation

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

Draw results
```bash
python ./experiments/draw_kinematic_arm_results.py
python ./experiments/draw_trajectory_optimization_results.py
```
