#!/usr/bin/env bash
set -euo pipefail

TOTAL_START=$SECONDS

# ANSI color codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
GRAY='\033[0;90m'
RED='\033[0;31m'
NC='\033[0m'

error_handler() {
    local line_num=$1
    echo ""
    echo -e "${RED}ERROR: Demo failed at line $line_num${NC}"
    exit 1
}
trap 'error_handler $LINENO' ERR

# Defaults
NO_WAIT=false
NPROC=4
BASELINE_BS=8
FOLLOWUP_BS=32
NUM_STEPS=10
DATASET_SIZE=2048

print_usage() {
    cat <<'USAGE'
Usage:
  ./demo.sh [OPTIONS]

Run the full Distributed SimCLR demo flow: baseline profiled run, trace analysis, follow-up profiled run, follow-up
analysis, and throughput comparison.

Options:
  -h, --help          Show this help message and exit
  --no-wait          Skip pauses between demo steps
  --nproc N          Number of torchrun processes (default: 4)
  --baseline-bs N    Baseline local batch size (default: 8)
  --followup-bs N    Follow-up local batch size (default: 32)
  --num-steps N      Training steps per run (default: 10)

Environment:
  PERFETTO_TRACE_PORT  Local HTTP port used to serve trace JSONs (default: 9001)
  PERFETTO_TRACE_BIND  Address the trace server binds to (default: 0.0.0.0)
  PERFETTO_TRACE_HOST  Hostname printed in browser URLs (default: 127.0.0.1)
  PERFETTO_UI_URL      Perfetto UI base URL (default: https://ui.perfetto.dev)

Examples:
  ./demo.sh
  ./demo.sh --no-wait --nproc 2 --num-steps 5
  PERFETTO_TRACE_PORT=9010 ./demo.sh
USAGE
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)     print_usage; exit 0 ;;
        --no-wait)     NO_WAIT=true; shift ;;
        --nproc)       NPROC="$2"; shift 2 ;;
        --baseline-bs) BASELINE_BS="$2"; shift 2 ;;
        --followup-bs) FOLLOWUP_BS="$2"; shift 2 ;;
        --num-steps)   NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; echo "Run ./demo.sh --help for usage." >&2; exit 1 ;;
    esac
done

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
cd "${PROJECT_DIR}"

OUTPUT_DIR="output"
BASELINE_RUN="demo_baseline"
FOLLOWUP_RUN="demo_followup"
CONDA_ENV="22971-td"
PERFETTO_TRACE_BIND="${PERFETTO_TRACE_BIND:-0.0.0.0}"
PERFETTO_TRACE_HOST="${PERFETTO_TRACE_HOST:-127.0.0.1}"
PERFETTO_TRACE_PORT="${PERFETTO_TRACE_PORT:-9001}"
PERFETTO_UI_URL="${PERFETTO_UI_URL:-https://ui.perfetto.dev}"
PERFETTO_SERVER_PID=""

activate_conda_env_if_needed() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "${CONDA_ENV}" ]; then
        return
    fi

    local conda_base=""
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
    elif [ -n "${CONDA_EXE:-}" ]; then
        conda_base="$(${CONDA_EXE} info --base 2>/dev/null || true)"
    elif [ -d "/opt/conda" ]; then
        conda_base="/opt/conda"
    fi

    if [ -z "${conda_base}" ] || [ ! -f "${conda_base}/etc/profile.d/conda.sh" ]; then
        return
    fi

    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
        echo -e "${GRAY}Activating conda env: ${CONDA_ENV}${NC}"
        conda activate "${CONDA_ENV}"
    fi
}

activate_conda_env_if_needed

if ! command -v torchrun >/dev/null 2>&1; then
    echo -e "${RED}ERROR: torchrun not found. Activate conda env ${CONDA_ENV} or install PyTorch with distributed support.${NC}" >&2
    exit 1
fi

export KMP_DUPLICATE_LIB_OK=TRUE

# Helpers
format_duration() {
    local secs=$1
    printf '%dm %02ds' $((secs / 60)) $((secs % 60))
}

cleanup_perfetto_server() {
    if [ -n "${PERFETTO_SERVER_PID}" ] && kill -0 "${PERFETTO_SERVER_PID}" >/dev/null 2>&1; then
        kill "${PERFETTO_SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup_perfetto_server EXIT

choose_perfetto_port() {
    python - "${PERFETTO_TRACE_BIND}" "${PERFETTO_TRACE_PORT}" <<'PY_PORT'
import socket
import sys

host = sys.argv[1]
preferred = int(sys.argv[2])

for port in [preferred, 0]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            print(sock.getsockname()[1])
            break
    except OSError:
        if port == 0:
            raise
PY_PORT
}

start_perfetto_trace_server() {
    if [ -n "${PERFETTO_SERVER_PID}" ] && kill -0 "${PERFETTO_SERVER_PID}" >/dev/null 2>&1; then
        return
    fi

    PERFETTO_TRACE_PORT="$(choose_perfetto_port)"
    python - "${PROJECT_DIR}" "${PERFETTO_TRACE_BIND}" "${PERFETTO_TRACE_PORT}" <<'PY_SERVER' >/dev/null 2>&1 &
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import sys

directory, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])

class TraceHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args):
        pass

handler = partial(TraceHandler, directory=directory)
ThreadingHTTPServer((host, port), handler).serve_forever()
PY_SERVER
    PERFETTO_SERVER_PID=$!
    sleep 0.2

    if ! kill -0 "${PERFETTO_SERVER_PID}" >/dev/null 2>&1; then
        echo -e "${RED}ERROR: failed to start local Perfetto trace server on ${PERFETTO_TRACE_BIND}:${PERFETTO_TRACE_PORT}${NC}" >&2
        exit 1
    fi
}

primary_trace_for_run() {
    local run_dir="${OUTPUT_DIR}/$1"
    local rank_one_trace="${run_dir}/traces/rank1.json"
    local rank_zero_trace="${run_dir}/traces/rank0.json"

    if [ -f "${rank_one_trace}" ]; then
        echo "${rank_one_trace}"
    elif [ -f "${rank_zero_trace}" ]; then
        echo "${rank_zero_trace}"
    else
        echo ""
    fi
}

trace_url_for_trace() {
    local trace_path="$1"
    python - "${PERFETTO_TRACE_HOST}" "${PERFETTO_TRACE_PORT}" "${trace_path}" <<'PY_TRACE_URL'
from urllib.parse import quote
import sys

host, port, trace_path = sys.argv[1:4]
print(f"http://{host}:{port}/{quote(trace_path)}")
PY_TRACE_URL
}

perfetto_link_for_trace() {
    local trace_path="$1"
    python - "${PERFETTO_UI_URL}" "$(trace_url_for_trace "${trace_path}")" <<'PY_LINK'
from urllib.parse import quote
import sys

ui_url, trace_url = sys.argv[1:3]
print(f"{ui_url}/#!/viewer?url={quote(trace_url, safe=':/?=&')}")
PY_LINK
}

log_perfetto_links() {
    local label="$1"
    local run_name="$2"
    local run_dir="${OUTPUT_DIR}/${run_name}"
    local primary_trace
    primary_trace="$(primary_trace_for_run "${run_name}")"

    if [ -z "${primary_trace}" ]; then
        echo -e "${YELLOW}Perfetto link unavailable: no trace JSON found under ${run_dir}/traces${NC}"
        return
    fi

    start_perfetto_trace_server

    echo ""
    echo -e "${YELLOW}${label} Perfetto trace:${NC} $(perfetto_link_for_trace "${primary_trace}")"
}

log_and_run() {
    echo -e "${GREEN}$*${NC}"
    "$@"
}

wait_for_user() {
    if [ "$NO_WAIT" = false ]; then
        local next_step="$1"
        echo ""
        echo -e "${CYAN}Press Enter to continue to the next step: ${next_step}${NC}"
        read -r
    fi
}

# Start
echo ""
echo -e "${CYAN}═════════════════════════${NC}"
echo -e "${CYAN}Distributed SimCLR - Demo${NC}"
echo -e "${CYAN}═════════════════════════${NC}"
echo "nproc         : ${NPROC}"
echo "baseline bs   : ${BASELINE_BS}"
echo "follow-up bs  : ${FOLLOWUP_BS}"
echo "steps/run     : ${NUM_STEPS}"
echo "dataset size  : ${DATASET_SIZE}"
echo "perfetto ui   : ${PERFETTO_UI_URL}"
echo "trace server : ${PERFETTO_TRACE_BIND}:${PERFETTO_TRACE_PORT} -> browser ${PERFETTO_TRACE_HOST}:${PERFETTO_TRACE_PORT}"
echo ""

# Step 1: Baseline profiled run
echo -e "${CYAN}Step 1: Baseline profiled run (bs=${BASELINE_BS})${NC}"
STEP_START=$SECONDS

log_and_run torchrun --standalone --nproc_per_node="${NPROC}" \
    train.py \
    --local-batch-size "${BASELINE_BS}" \
    --num-steps "${NUM_STEPS}" \
    --dataset-size "${DATASET_SIZE}" \
    --profile \
    --run-name "${BASELINE_RUN}"

echo -e "${GRAY}Step 1 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
log_perfetto_links "Baseline" "${BASELINE_RUN}"
wait_for_user "Trace analysis"

# Step 2: Baseline trace analysis
echo ""
echo -e "${CYAN}Step 2: Analyze baseline traces${NC}"
STEP_START=$SECONDS

log_and_run python analyze.py --run-dir "${OUTPUT_DIR}/${BASELINE_RUN}"

echo -e "${GRAY}Step 2 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
wait_for_user "Follow-up run"

# Step 3: Follow-up profiled run
echo ""
echo -e "${CYAN}Step 3: Follow-up run with tuned batch size (bs=${FOLLOWUP_BS})${NC}"
STEP_START=$SECONDS

log_and_run torchrun --standalone --nproc_per_node="${NPROC}" \
    train.py \
    --local-batch-size "${FOLLOWUP_BS}" \
    --num-steps "${NUM_STEPS}" \
    --dataset-size "${DATASET_SIZE}" \
    --profile \
    --run-name "${FOLLOWUP_RUN}"

echo -e "${GRAY}Step 3 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
log_perfetto_links "Follow-up" "${FOLLOWUP_RUN}"
wait_for_user "Follow-up analysis"

# Step 4: Follow-up trace analysis
echo ""
echo -e "${CYAN}Step 4: Analyze follow-up traces${NC}"
STEP_START=$SECONDS

log_and_run python analyze.py --run-dir "${OUTPUT_DIR}/${FOLLOWUP_RUN}"

echo -e "${GRAY}Step 4 completed in $(format_duration $((SECONDS - STEP_START)))${NC}"
wait_for_user "Comparison"

# Step 5: Side-by-side comparison
echo ""
echo -e "${CYAN}Step 5: Comparison${NC}"
echo ""

# Extract images/s from both runs
BASELINE_IPS=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/${BASELINE_RUN}/run_config.json'))['images_per_sec'])")
FOLLOWUP_IPS=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/${FOLLOWUP_RUN}/run_config.json'))['images_per_sec'])")

echo -e "${YELLOW}Baseline  (bs=${BASELINE_BS}): ${BASELINE_IPS} images/s${NC}"
echo -e "${YELLOW}Follow-up (bs=${FOLLOWUP_BS}): ${FOLLOWUP_IPS} images/s${NC}"

# Compute speedup
python -c "
bl, fu = ${BASELINE_IPS}, ${FOLLOWUP_IPS}
if bl > 0:
    speedup = fu / bl
    label = 'faster' if speedup > 1 else 'slower'
    print(f'  Speedup: {speedup:.2f}x ({label})')
"

# Final summary
echo ""
echo -e "${CYAN}════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Demo complete in $(format_duration $((SECONDS - TOTAL_START)))${NC}"
echo -e "${CYAN}════════════════════════════════════════════════${NC}"
echo ""
echo "Output artifacts:"
echo "  ${OUTPUT_DIR}/${BASELINE_RUN}/"
echo "  ${OUTPUT_DIR}/${FOLLOWUP_RUN}/"
echo ""
BASELINE_TRACE="$(primary_trace_for_run "${BASELINE_RUN}")"
FOLLOWUP_TRACE="$(primary_trace_for_run "${FOLLOWUP_RUN}")"
if [ -n "${BASELINE_TRACE}" ] && [ -n "${FOLLOWUP_TRACE}" ]; then
    start_perfetto_trace_server
    echo "Perfetto trace links:"
    echo "  Baseline : $(perfetto_link_for_trace "${BASELINE_TRACE}")"
    echo "  Follow-up: $(perfetto_link_for_trace "${FOLLOWUP_TRACE}")"
fi

if [ -n "${PERFETTO_SERVER_PID}" ] && kill -0 "${PERFETTO_SERVER_PID}" >/dev/null 2>&1; then
    echo ""
    if [ "${NO_WAIT}" = false ]; then
        echo -e "${CYAN}Perfetto links stay live while this script is open. Press Enter to stop the local trace server and exit.${NC}"
        read -r
    else
        trap - EXIT
        echo -e "${YELLOW}Local Perfetto trace server left running as PID ${PERFETTO_SERVER_PID}.${NC}"
        echo "Stop it with: kill ${PERFETTO_SERVER_PID}"
    fi
fi
