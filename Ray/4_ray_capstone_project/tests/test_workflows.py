"""
Script-level integration and end-to-end workflow tests.

Covers:
- Prepare script with various configurations
- Run script (blocking/async modes) with parameterizations
- Stress mode comparison tests
- Full end-to-end workflows (prepare → run → verify)
"""

import json
import os
import pytest

from pathlib import Path
from typing import Dict

from tests.helpers import run_script, run_prepare_script


os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


BLOCKING_ASYNC_ARTIFACTS = [
    "run_config.json",
    "metrics.csv",
    "tick_summary.json",
    "latency_log.json",
    "actor_counters.json",
]


# ============================================================================
# Prepare script tests
# ============================================================================


@pytest.mark.parametrize("n_zones,seed", [
    (3, None),
    pytest.param(5, None, marks=pytest.mark.full),
    pytest.param(3, 42, marks=pytest.mark.full),
    pytest.param(5, 42, marks=pytest.mark.full),
])
def test_prepare_script(synthetic_parquets: Dict[str, Path], tmp_path: Path, n_zones: int, seed: int | None) -> None:
    """
    Test prepare script with various zone counts and seeds (reduced in quick mode).
    """
    out = tmp_path / "prepared"
    args = [
        "main.py",
        "prepare",
        "--ref-parquet", str(synthetic_parquets["ref"]),
        "--replay-parquet", str(synthetic_parquets["replay"]),
        "--output-dir", str(out),
        "--n-zones", str(n_zones),
    ]
    if seed is not None:
        args.extend(["--seed", str(seed)])

    result = run_script(args)
    assert result.returncode == 0, f"main.py prepare --n-zones {n_zones} --seed {seed} failed"
    for fname in ["baseline.parquet", "replay.parquet", "active_zones.json", "prep_meta.json"]:
        assert (out / fname).exists(), f"main.py prepare must produce {fname}"
    with open(out / "active_zones.json") as f:
        zones = json.load(f)
    assert len(zones) == n_zones, f"Expected {n_zones} zones, got {len(zones)}"
    with open(out / "prep_meta.json") as f:
        meta = json.load(f)
    if seed is not None:
        assert meta["seed"] == seed, f"prep_meta.json seed mismatch: expected {seed}, got {meta['seed']}"
    assert meta["n_zones"] == n_zones, f"prep_meta.json n_zones mismatch: expected {n_zones}, got {meta['n_zones']}"


# ============================================================================
# Run script tests
# ============================================================================


@pytest.mark.parametrize("mode,slow_frac,slow_sleep,timeout_s,max_inflight,fallback", [
    ("blocking", "0.25", "0.1", "2.0", "4", "always_previous"),
    pytest.param("blocking", "0.0", "0.0", "2.0", "4", "always_previous", marks=pytest.mark.full),
    pytest.param("blocking", "0.4", "0.2", "2.0", "4", "always_previous", marks=pytest.mark.full),
    ("async", "0.25", "0.1", "2.0", "4", "always_previous"),
    pytest.param("async", "0.0", "0.0", "2.0", "4", "always_previous", marks=pytest.mark.full),
    pytest.param("async", "0.4", "0.2", "1.0", "2", "always_previous", marks=pytest.mark.full),
    pytest.param("async", "0.25", "0.1", "3.0", "5", "always_previous", marks=pytest.mark.full),
    pytest.param("async", "0.4", "0.3", "1.5", "3", "always_previous", marks=pytest.mark.full),
])
def test_run_script(synthetic_parquets: Dict[str, Path], tmp_path: Path, max_ticks: int, 
                    mode: str, slow_frac: str, slow_sleep: str,
                    timeout_s: str, max_inflight: str, fallback: str) -> None:
    """
    Test run script with various modes and parameters (reduced in quick mode).
    """
    # Prepare data first
    prepared_dir = tmp_path / "prepared"
    result = run_prepare_script(synthetic_parquets["ref"], synthetic_parquets["replay"], prepared_dir, n_zones=5)
    assert result.returncode == 0, "Prepare step failed"

    # Run the script
    out = tmp_path / "output"
    result = run_script([
        "main.py",
        "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", mode,
        "--n-zones", "5",
        "--slow-zone-fraction", slow_frac,
        "--slow-zone-sleep-s", slow_sleep,
        "--tick-timeout-s", timeout_s,
        "--max-inflight-zones", max_inflight,
        "--fallback-policy", fallback,
        "--seed", "42",
        "--max-ticks", str(max_ticks),
    ])
    assert result.returncode == 0, f"main.py run --mode {mode} failed"
    artifact_dir = out / mode
    for fname in BLOCKING_ASYNC_ARTIFACTS:
        assert (artifact_dir / fname).exists(), f"main.py run --mode {mode} must produce {fname}"


def test_run_stress_script(synthetic_parquets: Dict[str, Path], tmp_path: Path, max_ticks: int) -> None:
    """
    Test stress mode script (uses max_ticks from mode).
    """
    # Prepare data first
    prepared_dir = tmp_path / "prepared"
    result = run_prepare_script(synthetic_parquets["ref"], synthetic_parquets["replay"], prepared_dir, n_zones=5)
    assert result.returncode == 0, "Prepare step failed"

    # Run stress mode
    out = tmp_path / "output"
    result = run_script([
        "main.py",
        "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", "stress",
        "--n-zones", "5",
        "--slow-zone-fraction", "0.6",
        "--slow-zone-sleep-s", "0.3",
        "--tick-timeout-s", "2.0",
        "--seed", "42",
        "--max-ticks", str(max_ticks),
    ], timeout=300)
    assert result.returncode == 0, f"main.py run --mode stress failed"
    stress_dir = out / "stress"
    assert (stress_dir / "comparison.json").exists(), "Stress mode must produce comparison.json"
    with open(stress_dir / "comparison.json") as f:
        comparison = json.load(f)
    assert "blocking" in comparison, "comparison.json must contain 'blocking' key"
    assert "async" in comparison, "comparison.json must contain 'async' key"
    for key in ["blocking", "async"]:
        assert "mean_tick_latency" in comparison[key], f"comparison.json['{key}'] must have mean_tick_latency"
    for sub in ["blocking", "async"]:
        sub_dir = stress_dir / sub
        for fname in BLOCKING_ASYNC_ARTIFACTS:
            assert (sub_dir / fname).exists(), f"Stress sub-run {sub} must produce {fname}"


# ============================================================================
# End-to-end workflow tests
# ============================================================================


def test_blocking_workflow(synthetic_parquets: Dict[str, Path], tmp_path: Path, max_ticks: int) -> None:
    """
    Test full blocking workflow: prepare → run → verify.
    """
    # Prepare with 20 zones
    prepared_dir = tmp_path / "prepared"
    result = run_prepare_script(synthetic_parquets["ref"], synthetic_parquets["replay"], prepared_dir, n_zones=20)
    assert result.returncode == 0, "Prepare step failed"

    # Run blocking mode
    out = tmp_path / "output"
    result = run_script([
        "main.py",
        "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", "blocking",
        "--n-zones", "20",
        "--slow-zone-fraction", "0.25",
        "--slow-zone-sleep-s", "1.0",
        "--max-ticks", str(max_ticks),
    ])
    assert result.returncode == 0, f"Blocking run failed"

    mode_dir = out / "blocking"
    for fname in BLOCKING_ASYNC_ARTIFACTS:
        assert (mode_dir / fname).exists(), f"Missing blocking/{fname}"


def test_async_workflow(synthetic_parquets: Dict[str, Path], tmp_path: Path, max_ticks: int) -> None:
    """
    Test full async workflow: prepare → run → verify.
    """
    # Prepare with 20 zones
    prepared_dir = tmp_path / "prepared"
    result = run_prepare_script(synthetic_parquets["ref"], synthetic_parquets["replay"], prepared_dir, n_zones=20)
    assert result.returncode == 0, "Prepare step failed"

    # Run async mode
    out = tmp_path / "output"
    result = run_script([
        "main.py",
        "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", "async",
        "--n-zones", "20",
        "--slow-zone-fraction", "0.25",
        "--slow-zone-sleep-s", "1.0",
        "--tick-timeout-s", "2.0",
        "--completion-fraction", "0.75",
        "--max-inflight-zones", "4",
        "--max-ticks", str(max_ticks),
    ])
    assert result.returncode == 0, f"Async run failed"

    mode_dir = out / "async"
    for fname in BLOCKING_ASYNC_ARTIFACTS:
        assert (mode_dir / fname).exists(), f"Missing async/{fname}"


def test_stress_workflow(synthetic_parquets: Dict[str, Path], tmp_path: Path, max_ticks: int) -> None:
    """
    Test full stress workflow: prepare → run → verify comparison.
    """
    # Prepare with 20 zones
    prepared_dir = tmp_path / "prepared"
    result = run_prepare_script(synthetic_parquets["ref"], synthetic_parquets["replay"], prepared_dir, n_zones=20)
    assert result.returncode == 0, "Prepare step failed"

    # Run stress mode
    out = tmp_path / "output"
    result = run_script([
        "main.py",
        "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", "stress",
        "--n-zones", "20",
        "--slow-zone-fraction", "0.6",
        "--slow-zone-sleep-s", "3.0",
        "--tick-timeout-s", "2.0",
        "--max-ticks", str(max_ticks),
    ], timeout=300)
    assert result.returncode == 0, f"Stress run failed"

    stress_dir = out / "stress"
    assert (stress_dir / "comparison.json").exists(), "Missing stress/comparison.json"

    for sub in ["blocking", "async"]:
        sub_dir = stress_dir / sub
        for fname in BLOCKING_ASYNC_ARTIFACTS:
            assert (sub_dir / fname).exists(), f"Missing stress/{sub}/{fname}"
