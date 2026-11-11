#!/usr/bin/env bash
# run_parallel_sdg.sh
# Launches parallel headless Isaac Sim workers to generate demonstration data,
# then consolidates them into a unified Hugging Face LeRobot dataset.

set -e

# Configuration (defaults can be overridden via environment variables or args)
NUM_WORKERS="${1:-4}"
EPISODES_PER_WORKER="${2:-25}"
OUTPUT_BASE="${3:-./lerobot_dataset}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Number of GPUs available (default 1)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi

echo "========================================================"
echo "  Launching $NUM_WORKERS Parallel Isaac Sim Workers"
echo "  Episodes per Worker: $EPISODES_PER_WORKER"
echo "  Total Target:        $((NUM_WORKERS * EPISODES_PER_WORKER)) episodes"
echo "  Detected GPUs:       $NUM_GPUS"
echo "  Base Output Dir:     $OUTPUT_BASE"
echo "========================================================"

mkdir -p "$OUTPUT_BASE"

# Array to store background PIDs
PIDS=()

for ((i=0; i<NUM_WORKERS; i++)); do
    # Assign GPU in round-robin fashion across available GPUs
    GPU_ID=$((i % NUM_GPUS))
    WORKER_DIR="${OUTPUT_BASE}/worker_${i}"

    CUDA_VISIBLE_DEVICES=$GPU_ID "${PROJECT_DIR}/run.sh" scripts/generate_lerobot_dataset.py \
        --headless \
        --worker-id $i \
        --num-episodes $EPISODES_PER_WORKER \
        --output-dir "$WORKER_DIR" \
        --record-interval 2 > "${OUTPUT_BASE}/worker_${i}.log" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "  [Worker $i] Launched on GPU $GPU_ID (PID: $PID) -> Log: ${OUTPUT_BASE}/worker_${i}.log"
done

echo ""
echo "All $NUM_WORKERS workers running in background. Waiting for completion..."
echo "Tip: You can monitor progress with: tail -f ${OUTPUT_BASE}/worker_*.log"

# Wait for all background worker processes to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "========================================================"
echo "  All parallel workers completed! Consolidating dataset..."
echo "========================================================"

# Run the consolidation script to produce the unified LeRobot v2.0 dataset
"${PROJECT_DIR}/run.sh" scripts/consolidate_dataset.py --output-base "$OUTPUT_BASE"

echo ""
echo "Demonstration dataset ready at: $OUTPUT_BASE"
echo "========================================================"
