"""
Shared utilities for integration tests.
"""

import csv
import json
import subprocess

from dataclasses import dataclass
from pathlib import Path

from src.common import CONFIG_FILENAME, METRICS_FILENAME


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NPROC = 4
DEFAULT_TIMEOUT = 180
DEFAULT_NUM_STEPS = 5
DEFAULT_DATASET_SIZE = 128
DEFAULT_LOCAL_BATCH_SIZE = 4

EVEN_RANK_SPANS = {
    "prepare_views", "stage0_forward", "send_boundary", "recv_boundary_grad", "stage0_backward", "grad_sync_stage0",
    "optimizer_step",
}

ODD_RANK_SPANS = {
    "recv_boundary", "stage1_forward", "gather_embeddings", "loss_calculation", "send_boundary_grad",
    "grad_sync_stage1", "optimizer_step",
}

COMPUTE_SPANS = {
    "prepare_views", "stage0_forward", "stage1_forward", "loss_calculation", "stage0_backward",
}

COMM_SPANS = {
    "send_boundary", "recv_boundary", "gather_embeddings", "send_boundary_grad", "recv_boundary_grad",
    "grad_sync_stage0", "grad_sync_stage1",
}

REQUIRED_CONFIG_KEYS = {"local_batch_size", "global_batch_size", "images_per_sec"}
REQUIRED_CSV_COLUMNS = {"step", "rank", "loss", "step_time_s"}


@dataclass
class RunResult:
    """
    Captured output and artifacts from a subprocess run.
    """
    run_dir: Path
    stdout: str
    stderr: str
    returncode: int


def run_torchrun(output_dir: Path, run_name: str, nproc: int = DEFAULT_NPROC,
                 local_batch_size: int = DEFAULT_LOCAL_BATCH_SIZE, num_steps: int = DEFAULT_NUM_STEPS,
                 dataset_size: int = DEFAULT_DATASET_SIZE, profile: bool = True, overlap: bool = False,
                 timeout: int = DEFAULT_TIMEOUT) -> RunResult:
    """
    Launch a torchrun training job and return captured result.
    """
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={nproc}",
        "train.py",
        "--local-batch-size", str(local_batch_size),
        "--num-steps", str(num_steps),
        "--dataset-size", str(dataset_size),
        "--output-dir", str(output_dir),
        "--run-name", run_name,
    ]
    if profile:
        cmd.append("--profile")
    if overlap:
        cmd.append("--overlap")

    proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return RunResult(
        run_dir=output_dir / run_name,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
    )


def run_analyze_cli(run_dir: Path, timeout: int = 30) -> RunResult:
    """
    Run analyze.py against a run directory.
    """
    cmd = ["python", "analyze.py", "--run-dir", str(run_dir)]
    proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return RunResult(run_dir=run_dir, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)


def run_controller_cli(output_dir: Path, batch_sizes: list[int], split_layers: list[str], nproc: int = DEFAULT_NPROC,
                       num_steps: int = DEFAULT_NUM_STEPS, dataset_size: int = DEFAULT_DATASET_SIZE,
                       timeout: int = DEFAULT_TIMEOUT * 3) -> RunResult:
    """
    Run controller.py with the given sweep configuration.
    """
    cmd = [
        "python", "controller.py",
        "--batch-sizes", *[str(b) for b in batch_sizes],
        "--split-layers", *split_layers,
        "--num-steps", str(num_steps),
        "--dataset-size", str(dataset_size),
        "--output-dir", str(output_dir),
        "--nproc", str(nproc),
    ]
    proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return RunResult(run_dir=output_dir, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)


def load_run_config(run_dir: Path) -> dict:
    """
    Load run_config.json from a run directory.
    """
    with open(run_dir / CONFIG_FILENAME) as f:
        return json.load(f)


def load_metrics_rows(run_dir: Path) -> list[dict]:
    """
    Load all rows from metrics.csv.
    """
    with open(run_dir / METRICS_FILENAME) as f:
        return list(csv.DictReader(f))


def load_losses_for_rank(run_dir: Path, rank: int = 1) -> list[float]:
    """
    Extract loss values for a rank, sorted by step.
    """
    rows = load_metrics_rows(run_dir)
    return [float(r["loss"]) for r in sorted(rows, key=lambda r: int(r["step"])) if r["loss"] and int(r["rank"]) == rank]


def load_trace_span_names(run_dir: Path, rank: int) -> set[str]:
    """
    Extract unique profiler span names from a rank trace.
    """
    events = load_trace_events(run_dir, rank)
    return {e["name"] for e in events}


def load_trace_events(run_dir: Path, rank: int) -> list[dict]:
    """
    Load duration events from a rank trace JSON.
    """
    with open(run_dir / "traces" / f"rank{rank}.json") as f:
        data = json.load(f)
    return [e for e in data["traceEvents"] if e.get("ph") == "X"]
