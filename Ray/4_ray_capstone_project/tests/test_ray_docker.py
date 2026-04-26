"""
Tests for Docker command examples documented in README.md.

Covers:
- Docker Compose cluster lifecycle examples - up, ps, and down
- Ray job submit examples - quick mode runs async only, full mode runs all three modes
"""

import os
import pytest
import shutil
import subprocess

from pathlib import Path
from typing import Dict

from src.common import ReplayMode as RunMode
from tests.helpers import PROJECT_DIR, DEFAULT_DOCKER_URL, run_command, run_prepare_script, wait_for_ray_dashboard


os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


OUTPUT_DIR = PROJECT_DIR / "output"
PREPARED_DIR = OUTPUT_DIR / "prepared_docker_test"
RUN_DIR = OUTPUT_DIR / "run_docker_test"
BLOCKING_ASYNC_ARTIFACTS = ["run_config.json", "metrics.csv", "tick_summary.json", "latency_log.json", "actor_counters.json"]


@pytest.fixture(scope="module")
def compose_cmd() -> list[str]:
    """
    Determine the appropriate Docker Compose command based on the environment.

    Returns:
        list[str]: Docker Compose command
    """
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    probe = subprocess.run(["docker", "compose", "version"], cwd=str(PROJECT_DIR), timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        return ["docker", "compose"]
    raise RuntimeError("Docker Compose CLI not found.")


@pytest.fixture(scope="module")
def docker_cluster(compose_cmd: list[str]):
    """
    Set up a Ray cluster using Docker Compose for the duration of the tests, and tear it down afterwards.

    Args:
        compose_cmd (list[str]): Docker Compose command determined by compose_cmd fixture
    """
    if not run_command(["docker", "info"], timeout=30).returncode == 0:
        raise RuntimeError("Docker daemon is not running. Please start it before running the tests.")
    up = run_command([*compose_cmd, "up", "-d"], timeout=1200)
    assert up.returncode == 0, "README docker-compose up -d example failed"
    ps = subprocess.run([*compose_cmd, "ps"], cwd=str(PROJECT_DIR), timeout=120, text=True, capture_output=True)
    assert ps.returncode == 0, "README docker-compose ps example failed"
    text = (ps.stdout or "") + (ps.stderr or "")
    assert "ray-capstone-head" in text, "docker-compose ps must list ray-capstone-head"
    assert "ray-capstone-worker-1" in text, "docker-compose ps must list ray-capstone-worker-1"
    assert "ray-capstone-worker-2" in text, "docker-compose ps must list ray-capstone-worker-2"
    wait_for_ray_dashboard(url=DEFAULT_DOCKER_URL, timeout=180)

    yield

    down = run_command([*compose_cmd, "down", "-v"], timeout=300)
    assert down.returncode == 0, "README docker-compose down -v example failed"


@pytest.fixture(scope="module")
def docker_prepared_assets(synthetic_parquets: Dict[str, Path]):
    """
    Prepare baseline and replay data using the prepare script for use in Docker README examples.

    Args:
        synthetic_parquets (Dict[str, Path]): Dictionary with paths to synthetic reference and replay parquet files
    """

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    result = run_prepare_script(synthetic_parquets["ref"], synthetic_parquets["replay"], PREPARED_DIR, n_zones=5)
    assert result.returncode == 0, "Prepare step for Docker README tests failed"
    for fname in ["baseline.parquet", "replay.parquet", "active_zones.json", "prep_meta.json"]:
        assert (PREPARED_DIR / fname).exists(), f"Prepared assets must include {fname}"


def submit_job(mode: RunMode, max_ticks: int, extra_args: list[str]) -> subprocess.CompletedProcess:
    prepared_path = "/workspace/" + str(PREPARED_DIR.relative_to(PROJECT_DIR))
    run_path = "/workspace/" + str(RUN_DIR.relative_to(PROJECT_DIR))
    cmd = [
        "ray", "job", "submit",
        "--address", DEFAULT_DOCKER_URL,
        "--working-dir", ".",
        "--",
        "python", "main.py", "run",
        "--prepared-dir", prepared_path,
        "--output-dir", run_path,
        "--mode", mode.value,
        *extra_args,
        "--ray-address", "auto",
        "--max-ticks", str(max_ticks),
    ]
    return run_command(cmd, timeout=1800)


@pytest.mark.usefixtures("docker_cluster", "docker_prepared_assets")
def test_readme_ray_job_submit_async(max_ticks: int) -> None:
    extra_args = [
        "--tick-timeout-s", "2.0",
        "--completion-fraction", "0.75",
        "--max-inflight-zones", "4"
    ]
    result = submit_job(RunMode.ASYNC, max_ticks, extra_args)
    assert result.returncode == 0, f"README ray job submit example failed for mode={RunMode.ASYNC}"
    assert all((RUN_DIR / RunMode.ASYNC / fname).exists() for fname in BLOCKING_ASYNC_ARTIFACTS), f"README {RunMode.ASYNC} example must produce all expected artifacts"


@pytest.mark.full
@pytest.mark.usefixtures("docker_cluster", "docker_prepared_assets")
def test_readme_ray_job_submit_blocking(max_ticks: int) -> None:
    result = submit_job(RunMode.BLOCKING, max_ticks, [])
    assert result.returncode == 0, f"README ray job submit example failed for mode={RunMode.BLOCKING}"
    assert all((RUN_DIR / RunMode.BLOCKING / fname).exists() for fname in BLOCKING_ASYNC_ARTIFACTS), \
        f"README {RunMode.BLOCKING} example must produce all expected artifacts"


@pytest.mark.full
@pytest.mark.usefixtures("docker_cluster", "docker_prepared_assets")
def test_readme_ray_job_submit_stress(max_ticks: int) -> None:
    extra_args = [
        "--slow-zone-fraction", "0.6",
        "--slow-zone-sleep-s", "3.0",
        "--tick-timeout-s", "2.0"
    ]
    result = submit_job(RunMode.STRESS, max_ticks, extra_args)
    assert result.returncode == 0, f"README ray job submit example failed for mode={RunMode.STRESS}"
    assert (RUN_DIR / "stress" / "comparison.json").exists(), "README stress example must produce comparison.json"
    for sub in ["blocking", "async"]:
        assert all((RUN_DIR / "stress" / sub / fname).exists() for fname in BLOCKING_ASYNC_ARTIFACTS), \
            f"README stress/{sub} example must produce all expected artifacts"
