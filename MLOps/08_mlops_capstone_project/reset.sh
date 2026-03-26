#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PORT="${PORT:-5001}"

# Kill server
pids=""
pids="$(lsof -ti tcp:"$PORT" || true)"
if [[ -n "$pids" ]]; then
    echo "Stopping process(es) on port $PORT: $pids"
    echo "$pids" | xargs kill
    sleep 1
else
    echo "No process found on port $PORT"
fi

# Clean artifacts
echo "Removing MLflow and Metaflow artifacts..."
rm -rf "$WORKSPACE_DIR/mlflow_tracking"
rm -rf "$PROJECT_DIR/mlflow_tracking" "$PROJECT_DIR/.metaflow" "$PROJECT_DIR/mlruns"
rm -f "$PROJECT_DIR/mlflow.db"