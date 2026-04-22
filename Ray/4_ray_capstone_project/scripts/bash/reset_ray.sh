#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "Stopping Ray..."
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found" >&2
    exit 1
fi
if ! conda env list | grep -q 22971-ray-capstone; then
    echo "ERROR: conda env 22971-ray-capstone not found" >&2
    exit 1
fi
conda run -n 22971-ray-capstone ray stop --force 2>/dev/null || true

echo "Removing prepared assets..."
rm -rf "$PROJECT_DIR/prepared"

echo "Removing output artifacts..."
rm -rf "$PROJECT_DIR/output"

echo "Reset complete."