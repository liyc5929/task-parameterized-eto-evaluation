#!/bin/bash
PID=$$
TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="./results"
RESULT_FILE="$RESULT_DIR/ms_sto_problem_scaling_runtime.$TIME.$PID.csv"
mkdir -p $RESULT_DIR
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
echo "See result file: $RESULT_FILE."
exec > "$RESULT_FILE" 2>&1

first_output=1
for resolution in 60 200 600 2000 6000; do
    for run_id in {1..3}; do
        if (( first_output == 1 )); then
            header_arg=""
            first_output=0
        else
            header_arg="--no_headers"
        fi

        CUDA_VISIBLE_DEVICES="$TMUX_SESSION" taskset -c "$CPU_LIST" \
            python ./experiments/ms_sto_problem_scaling_runtime.py --num_source_tasks 5000 --pop_size 16 --dimension 60 --problem_resolution "$resolution" --problem_realization parallel --device cuda $header_arg
    done

    for run_id in {1..3}; do
        CUDA_VISIBLE_DEVICES="$TMUX_SESSION" taskset -c "$CPU_LIST" \
            python ./experiments/ms_sto_problem_scaling_runtime.py --num_source_tasks 5000 --pop_size 16 --dimension 60 --problem_resolution "$resolution" --problem_realization pointwise --device cuda --no_headers
    done
    echo
done
