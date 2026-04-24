"""
Runtime execution module for distributed replay with Ray.

Orchestrates blocking, async, and stress test modes using Ray remote actors and tasks.
Each mode creates ZoneActors, launches scoring tasks with simulated skew, and writes detailed metrics and decision logs.

Execution modes:
- blocking: Wait for all zones before advancing each tick (baseline)
- async: Bounded concurrency with timeout and partial-readiness fallback
- stress: Harsh skew (60% slow zones, 3s delay) to stress-test async controller
"""

import json
import logging
import numpy as np
import ray

from pathlib import Path
from typing import List

from src.tlc import ReplayConfig, ReplayMode, TickMetrics, write_json
from src.replay import BlockingReplay, AsyncReplay


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_blocking(prepared_dir: Path, output_dir: Path, config: ReplayConfig) -> List[TickMetrics]:
    """
    Blocking baseline: for each tick, collect snapshots, launch scoring tasks, wait for ALL results, write accepted
    decisions into actors.

    Args:
        prepared_dir (Path): Directory with prepared assets from prepare.py
        output_dir (Path): Root output directory (artifacts go into output_dir/blocking/)
        config (ReplayConfig): Runtime configuration

    Returns:
        List[TickMetrics]: Per-tick metrics for the blocking run
    """
    replay = BlockingReplay(prepared_dir, output_dir, config)
    return replay.run()


def run_async(prepared_dir: Path, output_dir: Path, config: ReplayConfig) -> List[TickMetrics]:
    """
    Async controller: scoring tasks report to actors, driver polls readiness,
    finalizes ticks under partial-readiness policy.

    Args:
        prepared_dir (Path): Directory with prepared assets from prepare.py
        output_dir (Path): Root output directory (artifacts go into output_dir/async/)
        config (ReplayConfig): Runtime configuration

    Returns:
        List[TickMetrics]: Per-tick metrics for the async run
    """
    replay = AsyncReplay(prepared_dir, output_dir, config)
    return replay.run()


def run_stress(prepared_dir: Path, output_dir: Path, config: ReplayConfig) -> List[TickMetrics]:
    """
    Stress test: reuse blocking and async paths with harsher skew (60% slow zones, 3s delay).

    Args:
        prepared_dir (Path): Directory with prepared assets from prepare.py
        output_dir (Path): Root output directory (artifacts go into output_dir/stress/)
        config (ReplayConfig): Base runtime configuration (skew fields are overridden)

    Returns:
        List[TickMetrics]: Per-tick metrics for the async stress run
    """
    stress_config = ReplayConfig(
        n_zones=config.n_zones,
        tick_minutes=config.tick_minutes,
        max_inflight_zones=config.max_inflight_zones,
        tick_timeout_s=config.tick_timeout_s,
        completion_fraction=config.completion_fraction,
        slow_zone_fraction=0.6,  # 60% of zones are slow
        slow_zone_sleep_s=3.0,  # 3 seconds sleep
        fallback_policy=config.fallback_policy,
        seed=config.seed,
        max_ticks=config.max_ticks,
    )

    logger.info("")
    logger.info("Stress Mode")
    blocking_metrics = run_blocking(prepared_dir, output_dir / "stress", stress_config)
    async_metrics = run_async(prepared_dir, output_dir / "stress", stress_config)

    # Write comparison summary
    comparison = {
        "blocking": {
            "mean_tick_latency": float(np.mean([m.total_tick_latency_s for m in blocking_metrics])),
            "max_tick_latency": float(np.max([m.total_tick_latency_s for m in blocking_metrics])),
            "total_fallbacks": sum(m.n_zones_fallback for m in blocking_metrics),
        },
        "async": {
            "mean_tick_latency": float(np.mean([m.total_tick_latency_s for m in async_metrics])),
            "max_tick_latency": float(np.max([m.total_tick_latency_s for m in async_metrics])),
            "total_fallbacks": sum(m.n_zones_fallback for m in async_metrics),
        },
    }
    write_json(comparison, output_dir / "stress" / "comparison.json")
    logger.info(f"Stress comparison: {json.dumps(comparison, indent=2)}")

    return async_metrics


def run_replay(ray_address: str, prepared_dir: Path, output_dir: Path, mode: ReplayMode, config: ReplayConfig) -> None:
    """
    Run the replay in the specified mode with the given configuration.

    Args:
        ray_address (str): Ray cluster address. None for local
        prepared_dir (Path): Directory with prepared assets from prepare.py
        output_dir (Path): Root output directory for artifacts
        mode (ReplayMode): Execution mode (blocking, async, or stress)
        config (ReplayConfig): Runtime configuration for the replay
    """
    with ray.init(address=ray_address):
        if mode == ReplayMode.BLOCKING:
            run_blocking(prepared_dir, output_dir, config)
        elif mode == ReplayMode.ASYNC:
            run_async(prepared_dir, output_dir, config)
        else:
            run_stress(prepared_dir, output_dir, config)
