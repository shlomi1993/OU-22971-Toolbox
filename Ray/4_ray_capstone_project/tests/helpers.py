"""
Helper functions for Ray capstone project tests.

This module contains utility functions for generating synthetic test data and running subprocess commands.
"""

import requests
import subprocess
import sys
import time
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List


RNG_SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DOCKER_URL = "http://localhost:8265"


def make_trips(year: int, month: int, n_zones: int = 30, base_count: int = 50) -> pd.DataFrame:
    """
    Generate synthetic trip data for testing.

    Args:
        year (int): Year for the trip timestamps
        month (int): Month for the trip timestamps
        n_zones (int): Number of zones to generate trips for. Default is 30.
        base_count (int): Base number of trips per zone. Default is 50.

    Returns:
        pd.DataFrame: DataFrame with columns: lpep_pickup_datetime, lpep_dropoff_datetime, PULocationID
    """
    rng = np.random.RandomState(RNG_SEED)
    rows = []
    for i in range(n_zones):
        zone_id = i + 1
        count = base_count + (n_zones - i) * 10
        for _ in range(count):
            day = 1 + rng.randint(0, 27)
            hour = rng.randint(0, 23)
            minute = rng.randint(0, 59)
            dt = pd.Timestamp(year, month, day, hour, minute)
            rows.append({
                "lpep_pickup_datetime": dt,
                "lpep_dropoff_datetime": dt + pd.Timedelta(minutes=15),
                "PULocationID": zone_id,
            })
    return pd.DataFrame(rows)


def make_zone_data(zone_id: int = 10, n_ticks: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic zone replay and baseline data for testing.

    Args:
        zone_id (int): Zone identifier. Default is 10.
        n_ticks (int): Number of ticks to generate. Default is 5.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of (replay_df, baseline_df)
    """
    base = pd.Timestamp("2023-02-01")
    ticks = [base + pd.Timedelta(minutes=15 * i) for i in range(n_ticks)]
    demands = [10.0, 12.0, 8.0, 15.0, 20.0][:n_ticks]
    replay = pd.DataFrame({"zone_id": zone_id, "tick_start": ticks, "demand": demands})
    baseline = pd.DataFrame({
        "zone_id": [zone_id],
        "hour_of_day": [0],
        "day_of_week": [2],
        "mean_demand": [10.0],
        "std_demand": [2.0],
    })
    return replay, baseline


def wait_for_ray_dashboard(url: str = DEFAULT_DOCKER_URL, timeout: int = 120, expected_nodes: int = 3) -> None:
    """
    Poll the Ray dashboard until it responds AND all expected nodes are alive.

    Args:
        url (str): Ray dashboard URL. Default is http://localhost:8265.
        timeout (int): Maximum seconds to wait. Default is 120.
        expected_nodes (int): Number of alive nodes to wait for. Default is 3.

    Raises:
        RuntimeError: If the cluster is not fully ready within the timeout.
    """
    deadline = time.monotonic() + timeout

    # Phase 1: wait for dashboard to respond
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/api/version", timeout=5)
            if r.status_code == 200:
                print(f"Ray dashboard is responding at {url}", flush=True)
                break
        except requests.ConnectionError:
            pass
        time.sleep(3)
    else:
        raise RuntimeError(f"Ray dashboard at {url} not ready within {timeout}s")

    # Phase 2: wait for all nodes to register
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/nodes?view=summary", timeout=5)
            if r.status_code == 200:
                data = r.json()
                nodes = data.get("data", {}).get("summary", [])
                alive = [n for n in nodes if n.get("raylet", {}).get("state") == "ALIVE"]
                if len(alive) >= expected_nodes:
                    print(f"Ray cluster ready: {len(alive)}/{expected_nodes} nodes alive", flush=True)
                    break
                print(f"Waiting for nodes: {len(alive)}/{expected_nodes} alive ...", flush=True)
        except (requests.ConnectionError, ValueError):
            pass
        time.sleep(5)
    else:
        raise RuntimeError(f"Ray cluster did not reach {expected_nodes} alive nodes within {timeout}s")

    # Phase 3: verify job submission actually works by submitting a trivial job
    while time.monotonic() < deadline:
        try:
            submit_resp = requests.post(
                f"{url}/api/jobs/",
                json={"entrypoint": "python -c \"print('ready')\"", "runtime_env": {}},
                timeout=10,
            )
            if submit_resp.status_code == 200:
                job_id = submit_resp.json().get("job_id", "")
                print(f"Probe job submitted: {job_id}", flush=True)
                # Wait for the probe job to finish (SUCCEEDED or FAILED both mean the agent works)
                while time.monotonic() < deadline:
                    status_resp = requests.get(f"{url}/api/jobs/{job_id}", timeout=5)
                    if status_resp.status_code == 200:
                        status = status_resp.json().get("status", "")
                        if status in ("SUCCEEDED", "FAILED"):
                            print(f"Probe job finished with status: {status}", flush=True)
                            if status == "SUCCEEDED":
                                return
                            break  # FAILED - retry outer loop
                    time.sleep(3)
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(5)
    raise RuntimeError(f"Ray job agent not ready within {timeout}s")


def run_command(cmd: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """
    Run a command as a subprocess, log its output live, and return the completed process.

    Args:
        cmd (List[str]): Command and arguments to run.
        timeout (int): Maximum execution time in seconds. Default is 120.

    Returns:
        subprocess.CompletedProcess: CompletedProcess instance with returncode
    """
    print(f"\nRunning:\n\033[34m{' '.join(cmd)}\033[0m\n", flush=True)
    return subprocess.run(cmd, cwd=str(PROJECT_DIR), timeout=timeout)


def run_python_script(args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """
    Run a Python script as a subprocess, log its output live, and return the completed process.

    Args:
        args (List[str]): Command-line arguments (script name and parameters)
        timeout (int): Maximum execution time in seconds. Default is 120.

    Returns:
        subprocess.CompletedProcess: CompletedProcess instance with returncode
    """
    cmd = [sys.executable] + args
    return run_command(cmd, timeout=timeout)


def run_prepare_script(ref_parquet: Path, replay_parquet: Path, output_dir: Path, n_zones: int) -> subprocess.CompletedProcess:
    """
    Run the prepare script to generate baseline and replay data.

    Args:
        ref_parquet (Path): Path to reference parquet file
        replay_parquet (Path): Path to replay parquet file
        output_dir (Path): Output directory for prepared data
        n_zones (int): Number of zones to prepare

    Returns:
        subprocess.CompletedProcess: CompletedProcess instance with returncode
    """
    return run_python_script([
        "main.py",
        "prepare",
        "--ref-parquet", str(ref_parquet),
        "--replay-parquet", str(replay_parquet),
        "--output-dir", str(output_dir),
        "--n-zones", str(n_zones),
    ])
