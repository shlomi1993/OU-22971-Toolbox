#!/usr/bin/env bash
set -euo pipefail

# Sweep local batch size values and analyze each run.
# Run from the project root: bash scripts/sweep.sh

SCRIPT_PATH="train.py"
ANALYZE_PATH="analyze.py"
SUMMARY_PATH="summarize_sweep.py"
OUTPUT_DIR="output"
NUM_STEPS=10
DATASET_SIZE=2048

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Expected ${SCRIPT_PATH} in the current working directory: $(pwd)" >&2
    exit 1
fi

BATCH_SIZES=(4 8 16 32 64 128)

export KMP_DUPLICATE_LIB_OK=TRUE

NPROC=4
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    _cores=$(nproc 2>/dev/null || echo "${NPROC}")
    export OMP_NUM_THREADS=$(( _cores / NPROC > 0 ? _cores / NPROC : 1 ))
fi

for BS in "${BATCH_SIZES[@]}"; do
    RUN_NAME="sweep_bs${BS}"
    echo ""
    echo "============================================================"
    echo "Sweeping local_batch_size=${BS}  run: ${RUN_NAME}"
    echo "============================================================"

    if ! torchrun \
        --standalone \
        --nproc_per_node=4 "${SCRIPT_PATH}" \
        --local-batch-size "${BS}" \
        --num-steps "${NUM_STEPS}" \
        --dataset-size "${DATASET_SIZE}" \
        --profile \
        --run-name "${RUN_NAME}"; then
        echo "WARNING: run failed for local_batch_size=${BS} (likely out of memory), skipping and continuing." >&2
        rm -rf "${OUTPUT_DIR:?}/${RUN_NAME}"
        continue
    fi

    echo ""
    echo "Analysis for local_batch_size=${BS}"
    python "${ANALYZE_PATH}" --run-dir "${OUTPUT_DIR}/${RUN_NAME}"
    echo ""
done

echo "============================================================"
echo "Writing manual sweep summary and diagnosis"
echo "============================================================"
python "${SUMMARY_PATH}" --output-dir "${OUTPUT_DIR}" --pattern "sweep_bs*"

echo "============================================================"
echo "Sweep complete. Results in ${OUTPUT_DIR}/sweep_bs*"
echo "Summary: ${OUTPUT_DIR}/manual_sweep_summary.csv"
echo "Diagnosis: ${OUTPUT_DIR}/diagnosis_summary.md"
echo "============================================================"
