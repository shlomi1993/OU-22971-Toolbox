#!/usr/bin/env bash
# demo.sh - Interactive demo that executes the full flow: downloads data, prepares assets, runs all 3 modes.
#
# Usage:
#   ./demo.sh [--keep-artifacts] [--max-ticks N] [--no-docker] [--no-wait]
#
# Options:
#   --keep-artifacts    Keep generated artifacts after demo completion
#   --max-ticks N       Limit number of ticks to run (default: 20, use "0" or "unlimited" for full month)
#   --no-docker         Run replay on local Ray instead of Docker cluster (default: Docker)
#   --no-wait           Skip pauses between steps (run continuously)
#
# Default Docker execution flow:
#   1. Download data
#   2. Start Docker cluster
#   3. Prepare assets (on host, then restart cluster to sync)
#   4. Run blocking baseline (on cluster)
#   5. Run async controller (on cluster)
#   6. Run skew stress test (on cluster)
#   7. Stop Docker cluster (unless --keep-artifacts is set)
#
# Alternative local execution flow:
#   1. Download data
#   2. Prepare assets
#   3. Run blocking baseline
#   4. Run async controller
#   5. Run skew stress test


set -euo pipefail

TOTAL_START=$SECONDS

# ANSI color codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Ray Dashboard URL for Docker cluster mode
RAY_DASHBOARD_URL="http://localhost:8265"

# Error handler
error_handler() {
    local line_num=$1
    echo ""
    echo -e "${RED}ERROR: Demo failed at line $line_num${NC}"
    exit 1
}
trap 'error_handler $LINENO' ERR

# Suppress Ray warnings
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=0

# A function to log and run shell commands
log_and_run() {
    echo -e "${GREEN}$*${NC}"
    if [ "$KEEP_ARTIFACTS" = true ]; then
        echo "$*" | sed "s|$PROJECT_DIR/||g" >> "$LOG_FILE"
    fi
    "$@"
}

# Format duration in seconds to mm:ss
format_duration() {
    local secs=$1
    printf '%dm %02ds' $((secs / 60)) $((secs % 60))
}

# Wait for user input to proceed
wait_for_user() {
    next_step="$1"
    echo ""
    echo -e "${CYAN}Press Enter to continue to the next step: ${next_step}${NC}"
    read -r
}

# Parse command-line arguments
KEEP_ARTIFACTS=false
USE_DOCKER=true
NO_WAIT=false
MAX_TICKS="20"  # Default: 20 ticks (use "0" or "unlimited" to run all ticks)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-artifacts) KEEP_ARTIFACTS=true; shift ;;
        --max-ticks) MAX_TICKS="$2"; shift 2 ;;
        --no-docker) USE_DOCKER=false; shift ;;
        --no-wait) NO_WAIT=true; shift ;;
        *) shift ;;
    esac
done

# Build MAX_TICKS_FLAG based on MAX_TICKS value
if [ "$MAX_TICKS" = "0" ] || [ "$MAX_TICKS" = "unlimited" ]; then
    MAX_TICKS_FLAG=""
else
    MAX_TICKS_FLAG="--max-ticks $MAX_TICKS"
fi

# Set up directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
cd "$PROJECT_DIR"

# Define data and output directories
DATA_DIR="$PROJECT_DIR/data"
PREPARED_DIR="$PROJECT_DIR/output/prepared"
OUTPUT_DIR="$PROJECT_DIR/output/run"

# Set up logging if keeping artifacts
if [ "$KEEP_ARTIFACTS" = true ]; then
    mkdir -p "$PROJECT_DIR/output"
    LOG_FILE="$PROJECT_DIR/output/demo.txt"
    echo "Date: $(date)" > "$LOG_FILE"
    echo "" >> "$LOG_FILE"
fi

# Define reference and replay files
REF_FILE="$DATA_DIR/green_tripdata_2023-01.parquet"
REPLAY_FILE="$DATA_DIR/green_tripdata_2023-02.parquet"

# --- Start of demo ---
echo -e "${CYAN}Ray Capstone - Demo${NC}"
echo -e "${CYAN}==================${NC}"
echo "Started: $(date)"
if [ "$USE_DOCKER" = true ]; then
    echo -e "Mode: Docker cluster (Ray Dashboard: $RAY_DASHBOARD_URL)"
else
    echo -e "Mode: Local Ray"
fi
if [ "$KEEP_ARTIFACTS" = true ]; then
    echo "Log file: $LOG_FILE"
fi
echo ""

# --- Download data ---
echo ""
echo -e "${CYAN}Step 1: Download TLC data${NC}"
STEP_START=$SECONDS
log_and_run bash "$PROJECT_DIR/scripts/download_data.sh"
echo -e "${GRAY}Step 1 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
if [ "$NO_WAIT" = false ]; then
    if [ "$USE_DOCKER" = true ]; then
        wait_for_user "Start Docker cluster"
    else
        wait_for_user "Prepare assets"
    fi
fi

# --- Start Docker cluster if requested ---
if [ "$USE_DOCKER" = true ]; then
    echo ""
    echo -e "${CYAN}Step 2: Start Docker cluster${NC}"
    STEP_START=$SECONDS
    log_and_run docker-compose up -d
    echo "Waiting for Ray cluster to be ready..."
    sleep 10
    log_and_run docker-compose ps
    echo -e "${GRAY}Step 2 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
    if [ "$NO_WAIT" = false ]; then
        wait_for_user "Prepare replay assets"
    fi
    STEP_OFFSET=1
else
    STEP_OFFSET=0
fi

# --- Prepare assets ---
echo ""
echo -e "${CYAN}Step $((2 + STEP_OFFSET)): Prepare replay assets${NC}"
STEP_START=$SECONDS

# Clean output directory to ensure fresh prepare
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning existing output directory"
    rm -rf "$OUTPUT_DIR"
fi

# Use fewer zones for Docker to reduce memory usage
if [ "$USE_DOCKER" = true ]; then
    echo -e "${YELLOW}Note:${NC} Using 8 zones for Docker mode to balance demo quality and memory usage"
    N_ZONES=8
else
    N_ZONES=20
fi
log_and_run prepare \
    --ref-parquet "$REF_FILE" \
    --replay-parquet "$REPLAY_FILE" \
    --output-dir "$PREPARED_DIR" \
    --n-zones $N_ZONES

# Verify all required files were created
REQUIRED_FILES=("replay.parquet" "baseline.parquet" "active_zones.json" "prep_meta.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$PREPARED_DIR/$file" ]; then
        echo -e "${RED}ERROR: Required file not created: $PREPARED_DIR/$file${NC}"
        exit 1
    fi
done

echo "Prepared assets written to $PREPARED_DIR"
# Restart Docker containers to ensure volume mounts pick up newly created files
if [ "$USE_DOCKER" = true ]; then
    echo "Restarting Docker cluster to sync volume mounts..."
    log_and_run docker-compose restart
    sleep 5  # Wait for cluster to be ready
fi
echo -e "${GRAY}Step $((2 + STEP_OFFSET)) completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
if [ "$NO_WAIT" = false ]; then
    wait_for_user "Run blocking baseline"
fi

# --- Run 1: Blocking baseline ---
echo ""
echo -e "${CYAN}Step $((3 + STEP_OFFSET)): Run blocking baseline${NC}"
STEP_START=$SECONDS
BLOCKING_ARGS="\
    --mode blocking \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --seed 42"
if [ "$USE_DOCKER" = true ]; then
    log_and_run ray job submit \
        --address "$RAY_DASHBOARD_URL" \
        --working-dir . \
        -- python main.py run \
            --prepared-dir /workspace/output/prepared \
            --output-dir /workspace/output/run \
            --ray-address auto \
            $BLOCKING_ARGS \
            $MAX_TICKS_FLAG
else
    log_and_run run \
        --prepared-dir "$PREPARED_DIR" \
        --output-dir "$OUTPUT_DIR" \
        $BLOCKING_ARGS \
        $MAX_TICKS_FLAG
fi
echo "Blocking run complete. Artifacts in $OUTPUT_DIR/blocking/"
echo -e "${GRAY}Step $((3 + STEP_OFFSET)) completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
if [ "$NO_WAIT" = false ]; then
    wait_for_user "Run async controller"
fi

# --- Run 2: Async controller ---
echo ""
echo -e "${CYAN}Step $((4 + STEP_OFFSET)): Run async controller${NC}"
STEP_START=$SECONDS
ASYNC_ARGS="\
    --mode async \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 \
    --completion-fraction 0.75 \
    --max-inflight-zones 4 \
    --seed 42"
if [ "$USE_DOCKER" = true ]; then
    log_and_run ray job submit \
        --address "$RAY_DASHBOARD_URL" \
        --working-dir . \
        -- python main.py run \
            --prepared-dir /workspace/output/prepared \
            --output-dir /workspace/output/run \
            --ray-address auto \
            $ASYNC_ARGS \
            $MAX_TICKS_FLAG
else
    log_and_run run \
        --prepared-dir "$PREPARED_DIR" \
        --output-dir "$OUTPUT_DIR" \
        $ASYNC_ARGS \
        $MAX_TICKS_FLAG
fi
echo "Async run complete. Artifacts in $OUTPUT_DIR/async/"
echo -e "${GRAY}Step $((4 + STEP_OFFSET)) completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
if [ "$NO_WAIT" = false ]; then
    wait_for_user "Run skew stress test"
fi

# --- Run 3: Stress test ---
echo ""
echo -e "${CYAN}Step $((5 + STEP_OFFSET)): Run skew stress test${NC}"
STEP_START=$SECONDS
STRESS_ARGS="\
    --mode stress \
    --slow-zone-fraction 0.6 \
    --slow-zone-sleep-s 3.0 \
    --tick-timeout-s 2.0 \
    --seed 42"
if [ "$USE_DOCKER" = true ]; then
    log_and_run ray job submit \
        --address "$RAY_DASHBOARD_URL" \
        --working-dir . \
        -- python main.py run \
            --prepared-dir /workspace/output/prepared \
            --output-dir /workspace/output/run \
            --ray-address auto \
            $STRESS_ARGS \
            $MAX_TICKS_FLAG
else
    log_and_run run \
        --prepared-dir "$PREPARED_DIR" \
        --output-dir "$OUTPUT_DIR" \
        $STRESS_ARGS \
        $MAX_TICKS_FLAG
fi
echo "Stress run complete. Artifacts in $OUTPUT_DIR/stress/"
echo -e "${GRAY}Step $((5 + STEP_OFFSET)) completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
if [ "$NO_WAIT" = false ]; then
    if [ "$USE_DOCKER" = true ] && [ "$KEEP_ARTIFACTS" = false ]; then
        wait_for_user "Stop Docker cluster"
    else
        wait_for_user "Complete demo"
    fi
fi

# --- Stop Docker cluster if used ---
if [ "$USE_DOCKER" = true ]; then
    echo ""
    if [ "$KEEP_ARTIFACTS" = true ]; then
        echo -e "${CYAN}Docker cluster still running (not stopped due to --keep-artifacts)${NC}"
        echo "To stop manually: docker-compose down"
    else
        echo -e "${CYAN}Stopping Docker cluster${NC}"
        log_and_run docker-compose down
    fi
fi

# --- Summary ---
echo ""
echo -e "${GREEN}Demo complete!${NC}"
echo -e "${GRAY}Total elapsed: $(format_duration $((SECONDS - TOTAL_START)))${NC}"

# --- Cleanup ---
echo ""
if [ "$KEEP_ARTIFACTS" = true ]; then
    echo -e "Artifacts written to: ${GREEN}$PROJECT_DIR/output${NC}"
else
    echo "Cleaning up generated artifacts"
    rm -rf "$PROJECT_DIR/output"
fi
