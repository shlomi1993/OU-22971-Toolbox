#!/usr/bin/env bash
# download_data.sh - Download TLC Green Taxi parquet files into the data/ directory.
# Skips files that already exist. Safe to run multiple times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$DATA_DIR"

REF_URL="https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet"
REPLAY_URL="https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet"
REF_FILE="$DATA_DIR/green_tripdata_2023-01.parquet"
REPLAY_FILE="$DATA_DIR/green_tripdata_2023-02.parquet"

if [ ! -f "$REF_FILE" ]; then
    echo "Downloading reference month (2023-01)..."
    curl -L -o "$REF_FILE" "$REF_URL"
else
    echo "Reference file already exists, skipping."
fi

if [ ! -f "$REPLAY_FILE" ]; then
    echo "Downloading replay month (2023-02)..."
    curl -L -o "$REPLAY_FILE" "$REPLAY_URL"
else
    echo "Replay file already exists, skipping."
fi

echo "Data directory: $DATA_DIR"
