#!/usr/bin/env bash
# test_flow.sh — System test: downloads data, prepares assets, runs all 3 demo modes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_DIR="$PROJECT_DIR/TLC_data"
PREPARED_DIR="$PROJECT_DIR/prepared"
OUTPUT_DIR="$PROJECT_DIR/output"

# TLC Green Taxi URLs (2023-01 reference, 2023-02 replay)
REF_URL="https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet"
REPLAY_URL="https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet"
REF_FILE="$DATA_DIR/green_tripdata_2023-01.parquet"
REPLAY_FILE="$DATA_DIR/green_tripdata_2023-02.parquet"

echo "============================================"
echo "  Ray Capstone - System Test"
echo "============================================"

# --- Download data ---
echo ""
echo ">>> Step 1: Download TLC data"
mkdir -p "$DATA_DIR"
if [ ! -f "$REF_FILE" ]; then
    echo "Downloading reference month..."
    curl -L -o "$REF_FILE" "$REF_URL"
else
    echo "Reference file already exists, skipping."
fi
if [ ! -f "$REPLAY_FILE" ]; then
    echo "Downloading replay month..."
    curl -L -o "$REPLAY_FILE" "$REPLAY_URL"
else
    echo "Replay file already exists, skipping."
fi

# --- Prepare assets ---
echo ""
echo ">>> Step 2: Prepare replay assets"
python prepare.py \
    --reference-parquet "$REF_FILE" \
    --replay-parquet "$REPLAY_FILE" \
    --output-dir "$PREPARED_DIR" \
    --n-zones 20 \
    --seed 42
echo "Prepared assets written to $PREPARED_DIR"

# --- Run 1: Blocking baseline ---
echo ""
echo ">>> Step 3: Run blocking baseline"
python run.py \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode blocking \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --seed 42
echo "Blocking run complete. Artifacts in $OUTPUT_DIR/blocking/"

# --- Run 2: Async controller ---
echo ""
echo ">>> Step 4: Run async controller"
python run.py \
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
echo ">>> Step 5: Run stress test (harsher skew)"
python run.py \
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
echo ">>> Step 6: Verify output artifacts"
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
