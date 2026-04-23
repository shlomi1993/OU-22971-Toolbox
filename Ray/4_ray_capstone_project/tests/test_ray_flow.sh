#!/usr/bin/env bash
# test_flow.sh — System test: downloads data, prepares assets, runs all 3 demo modes.
#
# This file executes:
#   1. bash scripts/bash/download_data.sh
#   2. prepare --ref-parquet data/green_tripdata_2023-01.parquet --replay-parquet data/green_tripdata_2023-02.parquet --output-dir prepared --n-zones 20
#   3. run --prepared-dir prepared --output-dir output --mode blocking --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0
#   4. run --prepared-dir prepared --output-dir output --mode async --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --tick-timeout-s 2.0 --completion-fraction 0.75 --max-inflight-zones 4
#   5. run --prepared-dir prepared --output-dir output --mode stress --slow-zone-fraction 0.6 --slow-zone-sleep-s 3.0 --tick-timeout-s 2.0
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
NC='\033[0m' # No Color

# Suppress Ray warnings

trap 'error_handler $LINENO' ERR

# Suppress Ray warnings
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=0

log_and_run() {
    echo -e "${GREEN}$*${NC}"
    "$@"
}

KEEP_ARTIFACTS=false
for arg in "$@"; do
    case "$arg" in
        --keep-artifacts|-k) KEEP_ARTIFACTS=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_DIR="$PROJECT_DIR/data"
PREPARED_DIR="$PROJECT_DIR/prepared"
OUTPUT_DIR="$PROJECT_DIR/output"

REF_FILE="$DATA_DIR/green_tripdata_2023-01.parquet"
REPLAY_FILE="$DATA_DIR/green_tripdata_2023-02.parquet"

echo ""
echo -e "${CYAN}Ray Capstone - Full Flow Test${NC}"
echo -e "${CYAN}=============================${NC}"

# --- Download data ---
echo ""
echo -e "${CYAN}Step 1: Download TLC data${NC}"
log_and_run bash "$PROJECT_DIR/scripts/bash/download_data.sh"

# --- Prepare assets ---
echo ""
echo -e "${CYAN}Step 2: Prepare replay assets${NC}"
log_and_run prepare \
    --ref-parquet "$REF_FILE" \
    --replay-parquet "$REPLAY_FILE" \
    --output-dir "$PREPARED_DIR" \
    --n-zones 20 \
echo "Prepared assets written to $PREPARED_DIR"

# --- Run 1: Blocking baseline ---
echo ""
echo -e "${CYAN}Step 3: Run blocking baseline${NC}"
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode blocking \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --max-ticks 50
echo "Blocking run complete. Artifacts in $OUTPUT_DIR/blocking/"

# --- Run 2: Async controller ---
echo ""
echo -e "${CYAN}Step 4: Run async controller${NC}"
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode async \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 \
    --completion-fraction 0.75 \
    --max-inflight-zones 4 \
    --max-ticks 50
echo "Async run complete. Artifacts in $OUTPUT_DIR/async/"

# --- Run 3: Stress test ---
echo ""
echo -e "${CYAN}Step 5: Run skew stress test${NC}"
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode stress \
    --slow-zone-fraction 0.6 \
    --slow-zone-sleep-s 3.0 \
    --tick-timeout-s 2.0 \
    --max-ticks 50
echo "Stress run complete. Artifacts in $OUTPUT_DIR/stress/"

# --- Verify artifacts ---
echo ""
echo -e "${CYAN}Step 6: Verify output artifacts${NC}"
for mode in blocking async; do
    for f in run_config.json metrics.csv latency_log.json tick_summary.json actor_counters.json; do
        if [ ! -f "$OUTPUT_DIR/$mode/$f" ]; then
            echo -e "${RED}FAIL: Missing $OUTPUT_DIR/$mode/$f${NC}"
            exit 1
        fi
    done
done
if [ ! -f "$OUTPUT_DIR/stress/comparison.json" ]; then
    echo -e "${RED}FAIL: Missing stress comparison.json${NC}"
    exit 1
fi

# --- Verdict ---
echo ""
echo -e "${GREEN}Full flow tests passed!${NC}"

# --- Cleanup ---
echo ""
if [ "$KEEP_ARTIFACTS" = false ]; then
    echo "Cleaning up generated artifacts"
    rm -rf "$PREPARED_DIR" "$OUTPUT_DIR"
else
    echo "Keeping artifacts:"
    find "$OUTPUT_DIR" -type f | sort
fi