#!/usr/bin/env bash
# demo.sh - Execute the full demo flow: baseline run, analysis, tuning decision, follow-up run, comparison.
#
# Matches the required demo pattern from the design doc (steps 2-7):
#   1. Baseline profiled run
#   2. Trace analysis
#   3. Tuning decision (pick a better batch size)
#   4. Follow-up run with updated config
#   5. Comparison of old vs new
#
# Usage:
#   ./demo.sh [--no-wait] [--nproc N] [--baseline-bs N] [--followup-bs N] [--num-steps N]
#
# Options:
#   --no-wait         Skip pauses between steps (run continuously)
#   --nproc N         Number of processes (default: 4)
#   --baseline-bs N   Baseline local batch size (default: 8)
#   --followup-bs N   Follow-up local batch size (default: 32)
#   --num-steps N     Training steps per run (default: 10)

set -euo pipefail

TOTAL_START=$SECONDS

# ANSI color codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
GRAY='\033[0;90m'
RED='\033[0;31m'
NC='\033[0m'

error_handler() {
    local line_num=$1
    echo ""
    echo -e "${RED}ERROR: Demo failed at line $line_num${NC}"
    exit 1
}
trap 'error_handler $LINENO' ERR

# Defaults
NO_WAIT=false
NPROC=4
BASELINE_BS=8
FOLLOWUP_BS=32
NUM_STEPS=10
DATASET_SIZE=2048

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-wait)      NO_WAIT=true; shift ;;
        --nproc)        NPROC="$2"; shift 2 ;;
        --baseline-bs)  BASELINE_BS="$2"; shift 2 ;;
        --followup-bs)  FOLLOWUP_BS="$2"; shift 2 ;;
        --num-steps)    NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
cd "${PROJECT_DIR}"

OUTPUT_DIR="output"
BASELINE_RUN="demo_baseline"
FOLLOWUP_RUN="demo_followup"

export KMP_DUPLICATE_LIB_OK=TRUE

# Helpers
format_duration() {
    local secs=$1
    printf '%dm %02ds' $((secs / 60)) $((secs % 60))
}

log_and_run() {
    echo -e "${GREEN}$*${NC}"
    "$@"
}

wait_for_user() {
    if [ "$NO_WAIT" = false ]; then
        local next_step="$1"
        echo ""
        echo -e "${CYAN}Press Enter to continue → ${next_step}${NC}"
        read -r
    fi
}

# Start
echo ""
echo -e "${CYAN}═════════════════════════${NC}"
echo -e "${CYAN}Distributed SimCLR - Demo${NC}"
echo -e "${CYAN}═════════════════════════${NC}"
echo "nproc         : ${NPROC}"
echo "baseline bs   : ${BASELINE_BS}"
echo "follow-up bs  : ${FOLLOWUP_BS}"
echo "steps/run     : ${NUM_STEPS}"
echo "dataset size  : ${DATASET_SIZE}"
echo ""

# Step 1: Baseline profiled run
echo -e "${CYAN}Step 1: Baseline profiled run (bs=${BASELINE_BS})${NC}"
STEP_START=$SECONDS

log_and_run torchrun --standalone --nproc_per_node="${NPROC}" \
    train.py \
    --local-batch-size "${BASELINE_BS}" \
    --num-steps "${NUM_STEPS}" \
    --dataset-size "${DATASET_SIZE}" \
    --profile \
    --run-name "${BASELINE_RUN}"

echo -e "${GRAY}  Step 1 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
wait_for_user "Trace analysis"

# Step 2: Baseline trace analysis
echo ""
echo -e "${CYAN}Step 2: Analyze baseline traces${NC}"
STEP_START=$SECONDS

log_and_run python analyze.py --run-dir "${OUTPUT_DIR}/${BASELINE_RUN}"

echo -e "${GRAY}  Step 2 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
wait_for_user "Follow-up run"

# Step 3: Follow-up profiled run
echo ""
echo -e "${CYAN}Step 3: Follow-up run with tuned batch size (bs=${FOLLOWUP_BS})${NC}"
STEP_START=$SECONDS

log_and_run torchrun --standalone --nproc_per_node="${NPROC}" \
    train.py \
    --local-batch-size "${FOLLOWUP_BS}" \
    --num-steps "${NUM_STEPS}" \
    --dataset-size "${DATASET_SIZE}" \
    --profile \
    --run-name "${FOLLOWUP_RUN}"

echo -e "${GRAY}  Step 3 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
wait_for_user "Follow-up analysis"

# Step 4: Follow-up trace analysis
echo ""
echo -e "${CYAN}Step 4: Analyze follow-up traces${NC}"
STEP_START=$SECONDS

log_and_run python analyze.py --run-dir "${OUTPUT_DIR}/${FOLLOWUP_RUN}"

echo -e "${GRAY}  Step 4 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
wait_for_user "Comparison"

# Step 5: Side-by-side comparison
echo ""
echo -e "${CYAN}Step 5: Comparison${NC}"
echo ""

# Extract images/s from both runs
BASELINE_IPS=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/${BASELINE_RUN}/run_config.json'))['images_per_sec'])")
FOLLOWUP_IPS=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/${FOLLOWUP_RUN}/run_config.json'))['images_per_sec'])")

echo -e "${YELLOW}Baseline  (bs=${BASELINE_BS}): ${BASELINE_IPS} images/s${NC}"
echo -e "${YELLOW}Follow-up (bs=${FOLLOWUP_BS}): ${FOLLOWUP_IPS} images/s${NC}"

# Compute speedup
python -c "
bl, fu = ${BASELINE_IPS}, ${FOLLOWUP_IPS}
if bl > 0:
    speedup = fu / bl
    label = 'faster' if speedup > 1 else 'slower'
    print(f'  Speedup: {speedup:.2f}x ({label})')
"

# Final summary
echo ""
echo -e "${CYAN}════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Demo complete in $(format_duration $((SECONDS - TOTAL_START)))${NC}"
echo -e "${CYAN}════════════════════════════════════════════════${NC}"
echo ""
echo "Output artifacts:"
echo "  ${OUTPUT_DIR}/${BASELINE_RUN}/  (baseline)"
echo "  ${OUTPUT_DIR}/${FOLLOWUP_RUN}/  (follow-up)"
echo ""
echo "Trace files can be loaded in chrome://tracing or Perfetto UI."
