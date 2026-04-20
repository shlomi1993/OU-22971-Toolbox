#!/usr/bin/env bash
# reset.sh — Stop Ray and remove all generated artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "Stopping Ray..."
ray stop --force 2>/dev/null || true

echo "Removing prepared assets..."
rm -rf "$PROJECT_DIR/prepared"

echo "Removing output artifacts..."
rm -rf "$PROJECT_DIR/output"

echo "Removing pytest cache..."
rm -rf "$PROJECT_DIR/.pytest_cache"
rm -rf "$PROJECT_DIR/__pycache__"
rm -rf "$PROJECT_DIR/src/__pycache__"
rm -rf "$PROJECT_DIR/tests/__pycache__"

echo "Reset complete. TLC_data/ kept (re-download is slow)."
