#!/usr/bin/env bash
# Find the maximum batch size before GPU OOM using binary search.
# This script is intended for GPU environments - will fail gracefully on CPU.

set -uo pipefail

SCRIPT="profile_ddp_gpu.py"
TRACE_DIR="traces"
LOG_DIR="run_logs"

# Check if we're in GPU mode
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
    echo "No CUDA GPUs detected. This exercise requires a GPU environment."
    echo "Skipping max batch size search."
    exit 0
fi

mkdir -p "${TRACE_DIR}" "${LOG_DIR}"

# Binary search parameters
MIN_BATCH=64
MAX_BATCH=2048
CURRENT_BATCH=$MIN_BATCH
LAST_SUCCESS=$MIN_BATCH
LAST_FAILURE=-1

echo "=========================================="
echo "Finding Maximum Batch Size Before OOM"
echo "=========================================="
echo "Starting binary search between ${MIN_BATCH} and ${MAX_BATCH}"
echo ""

test_batch_size() {
    local batch=$1
    local trace_name="max_batch_test_${batch}"
    local log_file="${LOG_DIR}/${trace_name}_stdout.log"

    echo -n "Testing batch_size=${batch}... "

    if torchrun --standalone --nproc_per_node=2 "${SCRIPT}" \
        --trace-name "${trace_name}" \
        --trace-dir "${TRACE_DIR}" \
        --batch-size "${batch}" \
        --steps 5 \
        > "${log_file}" 2>&1; then
        echo "SUCCESS"
        return 0
    else
        # Look for OOM
        if grep -q "out of memory\|OutOfMemoryError\|CUDA out of memory" "${log_file}"; then
            echo "OOM"
            return 1
        else
            echo "FAILED (not OOM - see ${log_file})"
            return 2
        fi
    fi
}

# Binary search loop
while (( MAX_BATCH - MIN_BATCH > 32 )); do
    CURRENT_BATCH=$(( (MIN_BATCH + MAX_BATCH) / 2 ))
    if test_batch_size "${CURRENT_BATCH}"; then
        LAST_SUCCESS=$CURRENT_BATCH
        MIN_BATCH=$CURRENT_BATCH
    else
        LAST_FAILURE=$CURRENT_BATCH
        MAX_BATCH=$CURRENT_BATCH
    fi
    echo "  -> Range narrowed to [${MIN_BATCH}, ${MAX_BATCH}]"
    echo ""
done

echo "=========================================="
echo "Search Complete"
echo "=========================================="
echo "Maximum successful batch size: ${LAST_SUCCESS}"
if (( LAST_FAILURE > 0 )); then
    echo "First failing batch size: ${LAST_FAILURE}"
fi
echo ""

# Run final profiling with max batch
FINAL_TRACE="max_batch_final"
echo "Running final profiling with batch_size=${LAST_SUCCESS}..."
torchrun --standalone --nproc_per_node=2 "${SCRIPT}" \
    --trace-name "${FINAL_TRACE}" \
    --trace-dir "${TRACE_DIR}" \
    --batch-size "${LAST_SUCCESS}" \
    --steps 5 \
    2>&1 | tee "${LOG_DIR}/${FINAL_TRACE}_stdout.log"

echo ""
echo "=========================================="
echo "Results"
echo "=========================================="
echo "Maximum batch size: ${LAST_SUCCESS}"
echo "Trace files:"
echo "  - ${TRACE_DIR}/${FINAL_TRACE}_rank0.json"
echo "  - ${TRACE_DIR}/${FINAL_TRACE}_rank1.json"
echo "Log file:"
echo "  - ${LOG_DIR}/${FINAL_TRACE}_stdout.log"
echo ""
echo "Next: Compare this trace against baseline using analyze_batch_scaling.py"
echo "=========================================="
