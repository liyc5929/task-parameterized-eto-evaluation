#!/bin/bash
PID=$$
TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="./results"
RESULT_FILE="$RESULT_DIR/ms_sto_problem_scaling_result.$TIME.$PID.csv"
ARTIFACT_DIR="$RESULT_DIR/ms_sto_problem_scaling_result.$TIME.$PID"
LOG_FILE="$ARTIFACT_DIR/run.log"
DRAW_FILE="$ARTIFACT_DIR/trajectory.seed1.pdf"
mkdir -p "$RESULT_DIR" "$ARTIFACT_DIR"
TMUX_SESSION=$(tmux display-message -p '#S')

# CPU configuration
# GPU0-3 -> NUMA 0 -> CPU0-31,64-95
# GPU4-7 -> NUMA 1 -> CPU32-63,96-127
if (( TMUX_SESSION >= 0 && TMUX_SESSION <= 3 )); then
    CPU_LIST="0-31"
elif (( TMUX_SESSION >= 4 && TMUX_SESSION <= 7 )); then
    CPU_LIST="32-63"
fi
CPU_THREADS=32
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export BLIS_NUM_THREADS="$CPU_THREADS"

echo "Procedure $PID has started at $(date '+%Y-%m-%d %H:%M:%S') on device $TMUX_SESSION and CPU core $CPU_LIST with $CPU_THREADS threads."
echo "Result CSV: $RESULT_FILE"
echo "Artifact directory: $ARTIFACT_DIR"
echo "Log file: $LOG_FILE"
echo "Figure: $DRAW_FILE"

first_output=1
for seed in {0..9}; do
    if (( first_output == 1 )); then
        header_arg=""
        first_output=0
    else
        header_arg="--no_headers"
    fi

    draw_args=()
    if (( seed == 1 )); then
        draw_args=(
            --draw_best_result
            --draw_path "$DRAW_FILE"
        )
    fi

    CUDA_VISIBLE_DEVICES="$TMUX_SESSION" taskset -c "$CPU_LIST" \
        python ./experiments/ms_sto_problem_scaling_result.py --seed "$seed" --num_source_tasks 5000 --pop_size 16 --dimension 60 --problem_resolution 6000 --device cuda $header_arg "${draw_args[@]}" 2>&1 \
        | tee -a "$LOG_FILE" \
        | grep --line-buffered -E '^(Seed|[0-9]+),' \
        >> "$RESULT_FILE"
done
