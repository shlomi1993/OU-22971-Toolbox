"""
Pytest fixtures for distributed training integration tests.
"""

import pytest

from tests.helpers import DEFAULT_NPROC, RunResult, run_controller_cli, run_torchrun


@pytest.fixture(scope="session")
def sync_run(tmp_path_factory: pytest.TempPathFactory) -> RunResult:
    """
    Profiled sync training run, shared across the test session.
    """
    output_dir = tmp_path_factory.mktemp("sync")
    result = run_torchrun(output_dir, "sync")
    assert result.returncode == 0, f"Sync training failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
    return result


@pytest.fixture(scope="session")
def overlap_run(tmp_path_factory: pytest.TempPathFactory) -> RunResult:
    """
    Profiled overlap training run, shared across the test session.
    """
    output_dir = tmp_path_factory.mktemp("overlap")
    result = run_torchrun(output_dir, "overlap", overlap=True)
    assert result.returncode == 0, f"Overlap training failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
    return result


@pytest.fixture(scope="session")
def controller_output(tmp_path_factory: pytest.TempPathFactory) -> RunResult:
    """
    Controller sweep over two batch sizes, shared across the test session.
    """
    output_dir = tmp_path_factory.mktemp("controller")
    result = run_controller_cli(output_dir, batch_sizes=[4, 8], split_layers=["layer2"], nproc=DEFAULT_NPROC)
    assert result.returncode == 0, f"Controller failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
    return result
