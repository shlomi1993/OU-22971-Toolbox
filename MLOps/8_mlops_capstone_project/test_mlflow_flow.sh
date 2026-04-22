#!/bin/bash
# Test script for the complete MLOps capstone flow
# Runs three scenarios: Bootstrap, Retrain/Promote, and Additional batch

set -e  # Exit on error

# Default: no sleep between runs
SLEEP_MINUTES=${1:-0}

echo "============================================"
echo "MLOps Capstone Flow - Full Test Suite"
echo "============================================"
echo "Sleep between runs: ${SLEEP_MINUTES} minutes"
echo ""

# Function to sleep between runs
sleep_between_runs() {
    if [ "$SLEEP_MINUTES" -gt 0 ]; then
        echo ""
        echo "Sleeping for ${SLEEP_MINUTES} minutes..."
        sleep $((SLEEP_MINUTES * 60))
        echo ""
    fi
}

# Run 1: Bootstrap (Baseline)
echo "============================================"
echo "RUN 1: Bootstrap - Create initial champion"
echo "============================================"
echo "Data: January 2020 (batch only, no reference)"
echo ""
conda run -n 22971-mlflow python capstone_flow.py run --batch-path TLC_data/green_tripdata_2020-01.parquet

sleep_between_runs

# Run 2: Retrain & Promote (COVID Impact)
echo "============================================"
echo "RUN 2: Retrain & Promote - COVID batch"
echo "============================================"
echo "Reference: January 2020"
echo "Batch: April 2020 (COVID-impacted)"
echo ""
conda run -n 22971-mlflow python capstone_flow.py run --ref-path TLC_data/green_tripdata_2020-01.parquet --batch-path TLC_data/green_tripdata_2020-04.parquet

sleep_between_runs

# Run 3a: Failure Simulation (Inject RuntimeError)
echo "============================================"
echo "RUN 3a: Failure Simulation - Inject error"
echo "============================================"
echo "Reference: April 2020"
echo "Batch: August 2020"
echo "Injecting RuntimeError into retrain step..."
echo ""

# Inject failure into retrain step
sed -i.bak '/^    @step$/,/^    def retrain/ {
    /^    def retrain/a\
        """Train a new XGBoost model on the combined reference + batch data."""\
        raise RuntimeError("Simulated failure for demo")
}' capstone_flow.py

# Temporarily disable exit on error for the expected failure
set +e
conda run -n 22971-mlflow python capstone_flow.py run --ref-path TLC_data/green_tripdata_2020-04.parquet --batch-path TLC_data/green_tripdata_2020-08.parquet
FAILURE_EXIT_CODE=$?
set -e

echo ""
echo "Expected failure occurred (exit code: ${FAILURE_EXIT_CODE})"
echo ""

sleep_between_runs

# Run 3b: Resume from Failure
echo "============================================"
echo "RUN 3b: Resume - Fix and continue from checkpoint"
echo "============================================"
echo "Removing injected error..."
echo "Resuming from retrain step..."
echo ""

# Remove the injected failure
mv capstone_flow.py.bak capstone_flow.py

# Resume from the failed step
conda run -n 22971-mlflow python capstone_flow.py resume retrain

echo ""
echo "============================================"
echo "All pipeline tests completed successfully!"
echo "============================================"

sleep_between_runs

# Run 4: Redeployment Demo
echo ""
echo "============================================"
echo "RUN 4: Redeployment Demo - Model Serving"
echo "============================================"
echo ""

# Check if port 5002 is already in use
if lsof -Pi :5002 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "WARNING: Port 5002 is already in use. Killing existing process..."
    lsof -ti:5002 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Check current champion version
echo "Current champion version:"
conda run -n 22971-mlflow python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
client = mlflow.MlflowClient()
for v in client.search_model_versions(\"name='green_taxi_tip_model'\"):
    if 'champion' in v.aliases:
        print(f'  Version {v.version} (@champion)')
        CURRENT_VERSION={v.version}
"

echo ""
echo "Starting model server with @champion..."
echo "   Server will run on http://127.0.0.1:5002"
echo ""

# Start model server in background
export MLFLOW_TRACKING_URI=http://localhost:5001
conda run -n 22971-mlflow mlflow models serve -m "models:/green_taxi_tip_model@champion" -p 5002 --env-manager local > /tmp/mlflow_server_1.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready with timeout
echo "Waiting for server to start..."
MAX_WAIT=60
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://127.0.0.1:5002/health > /dev/null 2>&1; then
        echo "Server ready after $ELAPSED seconds"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Server failed to start within $MAX_WAIT seconds"
    echo "Check logs at: /tmp/mlflow_server_1.log"
    tail -20 /tmp/mlflow_server_1.log
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Test prediction with current model
echo ""
echo "Testing prediction with current champion..."
PREDICTION_1=$(curl -s http://127.0.0.1:5002/invocations -H "Content-Type: application/json" --data-binary "@payload.json" 2>&1)

if [ $? -ne 0 ]; then
    echo "ERROR: Prediction failed"
    echo "Response: $PREDICTION_1"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "   Prediction: $PREDICTION_1"

sleep_between_runs

# Run a new batch to trigger promotion
echo ""
echo "Running new batch to trigger promotion..."
echo "   Reference: August 2020"
echo "   Batch: January 2020"
echo ""
conda run -n 22971-mlflow python capstone_flow.py run --ref-path TLC_data/green_tripdata_2020-08.parquet --batch-path TLC_data/green_tripdata_2020-01.parquet

# Check new champion version
echo ""
echo "New champion version:"
conda run -n 22971-mlflow python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
client = mlflow.MlflowClient()
for v in client.search_model_versions(\"name='green_taxi_tip_model'\"):
    if 'champion' in v.aliases:
        print(f'  Version {v.version} (@champion)')
"

# Stop the old server
echo ""
echo "Stopping old server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
sleep 3

# Restart server with new champion
echo ""
echo "Restarting server to pick up new @champion..."
conda run -n 22971-mlflow mlflow models serve -m "models:/green_taxi_tip_model@champion" -p 5002 --env-manager local > /tmp/mlflow_server_2.log 2>&1 &
NEW_SERVER_PID=$!

# Wait for server to be ready with timeout
echo "Waiting for server to restart..."
MAX_WAIT=60
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://127.0.0.1:5002/health > /dev/null 2>&1; then
        echo "Server ready after $ELAPSED seconds"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Server failed to restart within $MAX_WAIT seconds"
    echo "Check logs at: /tmp/mlflow_server_2.log"
    tail -20 /tmp/mlflow_server_2.log
    kill $NEW_SERVER_PID 2>/dev/null || true
    exit 1
fi

# Test prediction with new model
echo ""
echo "Testing prediction with NEW champion..."
PREDICTION_2=$(curl -s http://127.0.0.1:5002/invocations -H "Content-Type: application/json" --data-binary "@payload.json" 2>&1)

if [ $? -ne 0 ]; then
    echo "ERROR: Prediction failed"
    echo "Response: $PREDICTION_2"
    kill $NEW_SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "   Prediction: $PREDICTION_2"

# Stop the server
echo ""
echo "Stopping server (PID: $NEW_SERVER_PID)..."
kill $NEW_SERVER_PID 2>/dev/null || true
sleep 2

# Clean up any remaining processes on port 5002
if lsof -Pi :5002 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Cleaning up remaining processes on port 5002..."
    lsof -ti:5002 | xargs kill -9 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "Redeployment demo completed!"
echo "============================================"
echo ""
echo "Comparison:"
echo "   Before promotion: $PREDICTION_1"
echo "   After promotion:  $PREDICTION_2"
echo ""
echo "Key insight: Using @champion alias, the same serve command"
echo "   automatically picks up the newly promoted model version."
echo ""
echo "View all results at: http://localhost:5001"
echo ""
echo "To check Model Registry versions:"
echo "  conda run -n 22971-mlflow python -c \"import mlflow; mlflow.set_tracking_uri('http://localhost:5001'); client = mlflow.MlflowClient(); [print(f'v{v.version}: aliases={v.aliases}') for v in client.search_model_versions(\\\"name='green_taxi_tip_model'\\\")]\""
echo ""
