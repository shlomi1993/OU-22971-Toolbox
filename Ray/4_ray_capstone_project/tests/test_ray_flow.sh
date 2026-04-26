#!/usr/bin/env bash
# test_ray_flow.sh - Full flow test: downloads data, prepares assets, runs all 3 demo modes.
#
# Usage:
#   ./test_ray_flow.sh [--keep-artifacts] [--max-ticks N]
#
# Options:
#   --keep-artifacts    Keep generated artifacts after test completion
#   --max-ticks N       Limit number of ticks to run (default: no limit)
#
# This file executes:
#   1. bash scripts/download_data.sh
#   2. prepare --ref-parquet data/green_tripdata_2023-01.parquet --replay-parquet data/green_tripdata_2023-02.parquet --output-dir output/prepared --n-zones 20
#   3. run --prepared-dir output/prepared --output-dir output/run --mode blocking --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0
#   4. run --prepared-dir output/prepared --output-dir output/run --mode async --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --tick-timeout-s 2.0 --completion-fraction 0.75 --max-inflight-zones 4
#   5. run --prepared-dir output/prepared --output-dir output/run --mode stress --slow-zone-fraction 0.6 --slow-zone-sleep-s 3.0 --tick-timeout-s 2.0
#   6. Verify all output artifacts exist (run_config.json, metrics.csv, latency_log.json, tick_summary.json, actor_counters.json, comparison.json)

set -euo pipefail

# ANSI color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Error handler
error_handler() {
    local line_num=$1
    echo ""
    echo -e "${RED}ERROR: Test failed at line $line_num${NC}"
    exit 1
}

trap 'error_handler $LINENO' ERR

# ANSI color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Suppress Ray warnings

trap 'error_handler $LINENO' ERR

# Suppress Ray warnings
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=0

log_and_run() {
    echo -e "${GREEN}$*${NC}"
    if [ "$KEEP_ARTIFACTS" = true ]; then
        echo "$*" | sed "s|$PROJECT_DIR/||g" >> "$LOG_FILE"
    fi
    "$@"
}

format_duration() {
    local secs=$1
    printf '%dm %02ds' $((secs / 60)) $((secs % 60))
}

TOTAL_START=$SECONDS

KEEP_ARTIFACTS=false
MAX_TICKS_FLAG=""  # Default: no limit (run all ticks)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-artifacts)
            KEEP_ARTIFACTS=true
            shift
            ;;
        --max-ticks)
            MAX_TICKS_FLAG="--max-ticks $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_DIR="$PROJECT_DIR/data"
PREPARED_DIR="$PROJECT_DIR/output/prepared"
OUTPUT_DIR="$PROJECT_DIR/output/run"

# Set up logging if keeping artifacts
if [ "$KEEP_ARTIFACTS" = true ]; then
    mkdir -p "$PROJECT_DIR/output"
    LOG_FILE="$PROJECT_DIR/output/test_ray_flow.txt"
    echo "Date: $(date)" > "$LOG_FILE"
    echo "" >> "$LOG_FILE"
fi

REF_FILE="$DATA_DIR/green_tripdata_2023-01.parquet"
REPLAY_FILE="$DATA_DIR/green_tripdata_2023-02.parquet"

echo ""
if [ "$KEEP_ARTIFACTS" = true ]; then
    echo "Log file: $LOG_FILE"
    echo "Started at: $(date)"
    echo ""
fi
echo -e "${CYAN}Ray Capstone - Full Flow Test${NC}"
echo -e "${CYAN}=============================${NC}"

# --- Download data ---
echo ""
echo -e "${CYAN}Step 1: Download TLC data${NC}"
STEP_START=$SECONDS
log_and_run bash "$PROJECT_DIR/scripts/download_data.sh"
echo -e "${GRAY}Step 1 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"

# --- Prepare assets ---
echo ""
echo -e "${CYAN}Step 2: Prepare replay assets${NC}"
STEP_START=$SECONDS
log_and_run prepare \
    --ref-parquet "$REF_FILE" \
    --replay-parquet "$REPLAY_FILE" \
    --output-dir "$PREPARED_DIR" \
    --n-zones 20
echo "Prepared assets written to $PREPARED_DIR"
echo -e "${GRAY}Step 2 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"

# --- Run 1: Blocking baseline ---
echo ""
echo -e "${CYAN}Step 3: Run blocking baseline${NC}"
STEP_START=$SECONDS
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode blocking \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    $MAX_TICKS_FLAG
echo "Blocking run complete. Artifacts in $OUTPUT_DIR/blocking/"
echo -e "${GRAY}Step 3 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"

# --- Run 2: Async controller ---
echo ""
echo -e "${CYAN}Step 4: Run async controller${NC}"
STEP_START=$SECONDS
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode async \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 \
    --completion-fraction 0.75 \
    --max-inflight-zones 4 \
    $MAX_TICKS_FLAG
echo "Async run complete. Artifacts in $OUTPUT_DIR/async/"
echo -e "${GRAY}Step 4 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"

# --- Run 3: Stress test ---
echo ""
echo -e "${CYAN}Step 5: Run skew stress test${NC}"
STEP_START=$SECONDS
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode stress \
    --slow-zone-fraction 0.6 \
    --slow-zone-sleep-s 3.0 \
    --tick-timeout-s 2.0 \
    $MAX_TICKS_FLAG
echo "Stress run complete. Artifacts in $OUTPUT_DIR/stress/"
echo -e "${GRAY}Step 5 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"

# --- Verify artifacts ---
echo ""
echo -e "${CYAN}Step 6: Verify output artifacts${NC}"
STEP_START=$SECONDS
for mode in blocking async; do
    for f in run_config.json metrics.csv latency_log.json tick_summary.json actor_counters.json; do
        if [ ! -f "$OUTPUT_DIR/$mode/$f" ]; then
            echo -e "${RED}FAIL: Missing $OUTPUT_DIR/$mode/$f${NC}"
            exit 1
        else
            echo "Found $OUTPUT_DIR/$mode/$f"
        fi
    done
done
if [ ! -f "$OUTPUT_DIR/stress/comparison.json" ]; then
    echo -e "${RED}FAIL: Missing stress comparison.json${NC}"
    exit 1
fi

echo -e "${GRAY}Step 6 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"

# --- Verdict ---
echo ""
echo -e "${GREEN}Full flow tests passed!${NC}"
echo -e "${GRAY}Total elapsed: $(format_duration $((SECONDS - TOTAL_START)))${NC}"

# --- Cleanup ---
echo ""
if [ "$KEEP_ARTIFACTS" = true ]; then
    echo "Keeping artifacts:"
    find "$PROJECT_DIR/output" -type f | sort
    echo -e "\n${GREEN}Log saved to: $LOG_FILE${NC}"
else
    echo "Cleaning up generated artifacts"
    rm -rf "$PROJECT_DIR/output"
fi