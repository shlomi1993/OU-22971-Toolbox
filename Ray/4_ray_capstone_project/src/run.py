"""
Replay execution script for Ray-based TLC replay system.

Orchestrates blocking, async, and stress test modes using Ray remote actors and tasks.
Each mode creates ZoneActors, launches scoring tasks with simulated skew, and writes detailed metrics and decision logs.

Execution modes:
- blocking: Wait for all zones before advancing each tick (baseline)
- async: Bounded concurrency with timeout and partial-readiness fallback
- stress: Harsh skew (many slow zones with delay) to stress-test async controller
"""

import argparse
import json
import numpy as np
import ray

from pathlib import Path
from typing import List

from src.artifacts import write_json
from src.common import (
    DEFAULT_COMPLETION_FRACTION,
    DEFAULT_MAX_INFLIGHT_ZONES,
    DEFAULT_N_ZONES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_SLOW_ZONE_FRACTION,
    DEFAULT_SLOW_ZONE_SLEEP_S,
    DEFAULT_TICK_TIMEOUT_S,
    FALLBACK_POLICY_PREVIOUS,
    TICK_MINUTES,
    ReplayMode,
    ReplayConfig,
    TickMetrics,
)
from src.logger import logger
from src.replay.blocking import BlockingReplay
from src.replay.asynchronous import AsyncReplay


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


def build_run_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for run command.

    Returns:
        argparse.ArgumentParser: Parser for run command arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run TLC replay in blocking/async/stress mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False  # Disable help when used as parent parser
    )
    parser.add_argument("--prepared-dir", type=Path, required=True, help="Directory with prepared assets from prepare command")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Root output directory for run artifacts")
    parser.add_argument("--ray-address", default=None, help="Ray cluster address, None for local mode")
    parser.add_argument("--mode", choices=[m.value for m in ReplayMode], required=True, help="Execution mode: blocking waits for all zones, async uses bounded concurrency with timeout, stress tests harsh skew")
    parser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES, help="Number of active zones to use")
    parser.add_argument("--tick-minutes", type=int, default=TICK_MINUTES, help="Number of minutes per tick, that is the time window for each recommendation batch")
    parser.add_argument("--max-inflight-zones", type=int, default=DEFAULT_MAX_INFLIGHT_ZONES, help="Max concurrent scoring tasks in async mode")
    parser.add_argument("--tick-timeout-s", type=float, default=DEFAULT_TICK_TIMEOUT_S, help="Tick timeout in seconds for async mode")
    parser.add_argument("--completion-fraction", type=float, default=DEFAULT_COMPLETION_FRACTION, help="Minimum fraction of zones required for finalization")
    parser.add_argument("--slow-zone-fraction", type=float, default=DEFAULT_SLOW_ZONE_FRACTION, help="Fraction of zones to simulate as slow")
    parser.add_argument("--slow-zone-sleep-s", type=float, default=DEFAULT_SLOW_ZONE_SLEEP_S, help="Artificial delay in seconds for slow zones")
    parser.add_argument("--fallback-policy", default=FALLBACK_POLICY_PREVIOUS, help="Fallback policy for late zones")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--max-ticks", type=int, default=None, help="Limit the max number of ticks to run (for testing)")
    return parser


def main():
    standalone_parser = argparse.ArgumentParser(parents=[build_run_parser()])
    args = standalone_parser.parse_args()
    mode = ReplayMode(args.mode)
    config = ReplayConfig.from_args(args)
    run_replay(args.ray_address, args.prepared_dir, args.output_dir, mode, config)


if __name__ == "__main__":
    main()
