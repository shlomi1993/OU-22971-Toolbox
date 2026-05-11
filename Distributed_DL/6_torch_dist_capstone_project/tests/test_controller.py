"""
Integration tests for the load-balancing controller.
"""

import json

from tests.helpers import RunResult


EXPECTED_SUB_RUNS = ["ctrl_bs4_layer2", "ctrl_bs8_layer2"]


def test_decision_log_selects_best_by_throughput(controller_output: RunResult) -> None:
    """
    Best run in the decision log has the highest images/s across all sweep runs.
    """
    log_path = controller_output.run_dir / "controller_log.json"
    assert log_path.exists(), "controller_log.json not created"
    with open(log_path) as f:
        log = json.load(f)
    assert "best" in log, "decision log missing 'best' key"
    best_ips = log["best"]["images_per_sec"]
    assert best_ips > 0, f"best images_per_sec must be positive, got {best_ips}"
    assert all(r["images_per_sec"] <= best_ips for r in log["runs"]), "best does not have highest images/s"


def test_decision_log_records_comm_and_imbalance(controller_output: RunResult) -> None:
    """
    Each run in the decision log includes communication percentage and stage imbalance.
    """
    with open(controller_output.run_dir / "controller_log.json") as f:
        log = json.load(f)
    for run in log["runs"]:
        assert "comm_pct" in run, f"run {run['run_name']} missing comm_pct"
        assert "stage_imbalance" in run, f"run {run['run_name']} missing stage_imbalance"
        assert run["comm_pct"] >= 0, f"comm_pct must be non-negative: {run['comm_pct']}"
        assert run["stage_imbalance"] > 0, f"stage_imbalance must be positive: {run['stage_imbalance']}"


def test_each_sub_run_produces_training_artifacts(controller_output: RunResult) -> None:
    """
    Every controller sub-run produces run_config.json, metrics.csv, and rank traces.
    """
    for sub in EXPECTED_SUB_RUNS:
        run_dir = controller_output.run_dir / sub
        assert (run_dir / "run_config.json").exists(), f"{sub}/run_config.json missing"
        assert (run_dir / "metrics.csv").exists(), f"{sub}/metrics.csv missing"
        trace_files = list((run_dir / "traces").glob("rank*.json"))
        assert len(trace_files) > 0, f"{sub}/traces has no rank trace files"


def test_sweep_covers_requested_batch_sizes(controller_output: RunResult) -> None:
    """
    Decision log contains entries for all requested batch sizes in the sweep.
    """
    with open(controller_output.run_dir / "controller_log.json") as f:
        log = json.load(f)
    batch_sizes_seen = {run["local_batch_size"] for run in log["runs"]}
    assert batch_sizes_seen == {4, 8}, f"expected batch sizes {{4, 8}}, got {batch_sizes_seen}"
