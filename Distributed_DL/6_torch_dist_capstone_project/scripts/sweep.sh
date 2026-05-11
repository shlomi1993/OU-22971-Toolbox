#!/usr/bin/env bash
set -euo pipefail

# Sweep local batch size values and analyze each run.
# Run from the project root: bash scripts/sweep.sh

SCRIPT_PATH="train.py"
ANALYZE_PATH="analyze.py"
OUTPUT_DIR="output"
NUM_STEPS=10
DATASET_SIZE=2048

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Expected ${SCRIPT_PATH} in the current working directory: $(pwd)" >&2
    exit 1
fi

BATCH_SIZES=(4 8 16 32 64 128)

export KMP_DUPLICATE_LIB_OK=TRUE

for BS in "${BATCH_SIZES[@]}"; do
    RUN_NAME="sweep_bs${BS}"
    echo ""
    echo "============================================================"
    echo "Sweeping local_batch_size=${BS}  run: ${RUN_NAME}"
    echo "============================================================"

    torchrun \
        --standalone \
        --nproc_per_node=4 "${SCRIPT_PATH}" \
        --local-batch-size "${BS}" \
        --num-steps "${NUM_STEPS}" \
        --dataset-size "${DATASET_SIZE}" \
        --profile \
        --run-name "${RUN_NAME}"

    echo ""
    echo "Analysis for local_batch_size=${BS}"
    python "${ANALYZE_PATH}" --run-dir "${OUTPUT_DIR}/${RUN_NAME}"
    echo ""
done

echo "============================================================"
echo "Sweep complete. Results in ${OUTPUT_DIR}/sweep_bs*"
echo "============================================================"
