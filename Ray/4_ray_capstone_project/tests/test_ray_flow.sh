#!/usr/bin/env bash
# test_flow.sh — System test: downloads data, prepares assets, runs all 3 demo modes.
#
# This file executes:
#   1. bash scripts/download_data.sh
#   2. prepare --ref-parquet data/green_tripdata_2023-01.parquet --replay-parquet data/green_tripdata_2023-02.parquet --output-dir prepared --n-zones 20 --seed 42
#   3. run --prepared-dir prepared --output-dir output --mode blocking --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --seed 42
#   4. run --prepared-dir prepared --output-dir output --mode async --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --tick-timeout-s 2.0 --completion-fraction 0.75 --max-inflight-zones 4 --seed 42
#   5. run --prepared-dir prepared --output-dir output --mode stress --slow-zone-fraction 0.6 --slow-zone-sleep-s 3.0 --tick-timeout-s 2.0 --seed 42
#   6. Verify all output artifacts exist (run_config.json, metrics.csv, latency_log.json, tick_summary.json, actor_counters.json, comparison.json)

set -euo pipefail

# Suppress Ray warnings
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=0

# ANSI color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_and_run() {
    echo -e "${GREEN}>>> $*${NC}"
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

echo "============================================"
echo "  Ray Capstone - System Test"
echo "============================================"

# --- Download data ---
echo ""
echo "Step 1: Download TLC data"
log_and_run bash "$PROJECT_DIR/scripts/download_data.sh"

# --- Prepare assets ---
echo ""
echo "Step 2: Prepare replay assets"
log_and_run prepare \
    --ref-parquet "$REF_FILE" \
    --replay-parquet "$REPLAY_FILE" \
    --output-dir "$PREPARED_DIR" \
    --n-zones 20 \
    --seed 42
echo "Prepared assets written to $PREPARED_DIR"

# --- Run 1: Blocking baseline ---
echo ""
echo "Step 3: Run blocking baseline"
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode blocking \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --seed 42
echo "Blocking run complete. Artifacts in $OUTPUT_DIR/blocking/"

# --- Run 2: Async controller ---
echo ""
echo "Step 4: Run async controller"
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode async \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 \
    --completion-fraction 0.75 \
    --max-inflight-zones 4 \
    --seed 42
echo "Async run complete. Artifacts in $OUTPUT_DIR/async/"

# --- Run 3: Stress test ---
echo ""
echo "Step 5: Run stress test (harsher skew)"
log_and_run run \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode stress \
    --slow-zone-fraction 0.6 \
    --slow-zone-sleep-s 3.0 \
    --tick-timeout-s 2.0 \
    --seed 42
echo "Stress run complete. Artifacts in $OUTPUT_DIR/stress/"

# --- Verify artifacts ---
echo ""
echo "Step 6: Verify output artifacts"
for mode in blocking async; do
    for f in run_config.json metrics.csv latency_log.json tick_summary.json actor_counters.json; do
        if [ ! -f "$OUTPUT_DIR/$mode/$f" ]; then
            echo "FAIL: Missing $OUTPUT_DIR/$mode/$f"
            exit 1
        fi
    done
done
if [ ! -f "$OUTPUT_DIR/stress/comparison.json" ]; then
    echo "FAIL: Missing stress comparison.json"
    exit 1
fi

echo ""
echo "============================================"
echo "  All system tests PASSED"
echo "============================================"
echo ""
echo "Output artifacts:"
find "$OUTPUT_DIR" -type f | sort

# --- Cleanup ---
if [ "$KEEP_ARTIFACTS" = false ]; then
    echo ""
    echo "Cleaning up generated artifacts"
    rm -rf "$PREPARED_DIR" "$OUTPUT_DIR"
    echo "Removed $PREPARED_DIR and $OUTPUT_DIR"
else
    echo ""
    echo "Keeping artifacts (--keep-artifacts)"
fi
