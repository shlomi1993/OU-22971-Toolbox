"""
Pytest configuration and fixtures for Ray capstone project tests.
"""

import os
import pytest
import ray
import shutil
import warnings

from pathlib import Path
from typing import Dict, Generator

from tests.helpers import make_trips


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Add custom command-line options.
    """
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Run full test suite with all parameterizations and max tick counts"
    )


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest and validate command-line options.
    """
    # Prevent using -m full as only --full flag should be used
    markexpr = config.getoption("-m", default="")
    if markexpr and "full" in markexpr:
        raise pytest.UsageError("Do not use '-m full' to select full tests. Use --full flag instead.")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Remove tests marked as full when not in full run mode.
    """
    if config.getoption("--full"):
        return  # Don't filter anything in full run mode

    # Remove full-mode tests from collection instead of skipping them
    items[:] = [item for item in items if "full" not in item.keywords]


def pytest_runtest_logfinish(nodeid: str) -> None:
    """
    Print a separator after each test to improve visual distinction.
    """
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    print(f"\n{'─' * terminal_width}")


@pytest.fixture(scope="session")
def max_ticks(request: pytest.FixtureRequest) -> int:
    """
    Return max_ticks value based on run mode.
    """
    return 50 if request.config.getoption("--full") else 1


@pytest.fixture(scope="session")
def ray_ctx() -> Generator[None, None, None]:
    """
    Initialize and teardown Ray once for entire test session.
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="ray")
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(num_cpus=2, logging_level="ERROR", log_to_driver=False)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def synthetic_parquets(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """
    Generate synthetic parquet files for testing (ref and replay months).
    """
    base = tmp_path_factory.mktemp("parquet")
    ref = make_trips(2023, 1, n_zones=10, base_count=30)
    replay = make_trips(2023, 2, n_zones=10, base_count=30)
    ref_path = base / "ref.parquet"
    replay_path = base / "replay.parquet"
    ref.to_parquet(ref_path, index=False)
    replay.to_parquet(replay_path, index=False)
    return {"ref": ref_path, "replay": replay_path}
