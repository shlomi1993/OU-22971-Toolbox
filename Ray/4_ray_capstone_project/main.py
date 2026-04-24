"""
Ray capstone main entry: TLC-backed per-zone recommendations under skew.

CLI entry point for the Ray capstone project.
Workflow: prepare TLC replay assets -> initialize per-zone actors -> compare blocking and async execution.

Provides three subcommands:
- `prepare`: Validate TLC parquet data, select active zones, build baseline and replay tables.
- `run`: Execute blocking, async, or stress mode with Ray distributed actors.
- `reset`: Stop Ray and remove generated artifacts.

Example usage:
    python main.py prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir prepared/
    python main.py run --prepared-dir prepared/ --output-dir output/ --mode async
    python main.py reset
"""

import argparse

from pathlib import Path

from src.prepare import prepare_assets
from src.reset import reset_ray
from src.run import run_replay
from src.replay.core import (
    DEFAULT_COMPLETION_FRACTION,
    DEFAULT_MAX_INFLIGHT_ZONES,
    DEFAULT_N_ZONES,
    DEFAULT_SEED,
    DEFAULT_SLOW_ZONE_FRACTION,
    DEFAULT_SLOW_ZONE_SLEEP_S,
    DEFAULT_TICK_TIMEOUT_S,
    FALLBACK_POLICY_PREVIOUS,
    TICK_MINUTES,
    ReplayConfig,
    ReplayMode,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Build main parser with prepare, run, and reset subcommands.

    Returns:
        argparse.ArgumentParser: Main parser with subcommands for prepare, run, and reset.
    """
    parser = argparse.ArgumentParser(
        description="TLC-backed per-zone recommendations under skew",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add prepare subcommand
    prepare_subparser = subparsers.add_parser(
        name="prepare",
        help="Prepare replay assets from TLC parquet files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    prepare_subparser.add_argument("--ref-parquet", type=Path, required=True, help="Path to reference month parquet file (e.g., green_tripdata_2023-01.parquet)")
    prepare_subparser.add_argument("--replay-parquet", type=Path, required=True, help="Path to replay month parquet file (e.g., green_tripdata_2023-02.parquet)")
    prepare_subparser.add_argument("--output-dir", type=Path, required=True, help="Directory to write prepared assets (baseline.parquet, replay.parquet, active_zones.json)")
    prepare_subparser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES, help="Number of active zones to select")
    prepare_subparser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for zone selection reproducibility")
    prepare_subparser.set_defaults(handler=handle_prepare)

    # Add run subcommand
    run_subparser = subparsers.add_parser(
        name="run",
        help="Run replay in blocking/async/stress mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    run_subparser.add_argument("--prepared-dir", type=Path, required=True, help="Directory with prepared assets from prepare command")
    run_subparser.add_argument("--output-dir", type=Path, required=True, help="Root output directory for run artifacts")
    run_subparser.add_argument("--ray-address", default=None, help="Ray cluster address, None for local mode")
    run_subparser.add_argument("--mode", choices=[m.value for m in ReplayMode], required=True, help="Execution mode: blocking waits for all zones, async uses bounded concurrency with timeout, stress tests harsh skew")
    run_subparser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES, help="Number of active zones to use")
    run_subparser.add_argument("--tick-minutes", type=int, default=TICK_MINUTES, help="Number of minutes per tick, that is the time window for each recommendation batch")
    run_subparser.add_argument("--max-inflight-zones", type=int, default=DEFAULT_MAX_INFLIGHT_ZONES, help="Max concurrent scoring tasks in async mode")
    run_subparser.add_argument("--tick-timeout-s", type=float, default=DEFAULT_TICK_TIMEOUT_S, help="Tick timeout in seconds for async mode")
    run_subparser.add_argument("--completion-fraction", type=float, default=DEFAULT_COMPLETION_FRACTION, help="Minimum fraction of zones required for finalization")
    run_subparser.add_argument("--slow-zone-fraction", type=float, default=DEFAULT_SLOW_ZONE_FRACTION, help="Fraction of zones to simulate as slow")
    run_subparser.add_argument("--slow-zone-sleep-s", type=float, default=DEFAULT_SLOW_ZONE_SLEEP_S, help="Artificial delay in seconds for slow zones")
    run_subparser.add_argument("--fallback-policy", default=FALLBACK_POLICY_PREVIOUS, help="Fallback policy for late zones")
    run_subparser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    run_subparser.add_argument("--max-ticks", type=int, default=None, help="Limit the max number of ticks to run (for testing), default is all ticks in replay table")
    run_subparser.set_defaults(handler=handle_run)

    # Add reset subcommand
    reset_subparser = subparsers.add_parser(
        name="reset",
        help="Stop Ray and remove generated artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    reset_subparser.set_defaults(handler=handle_reset)

    return parser


def handle_prepare(args: argparse.Namespace) -> None:
    """
    Handle prepare subcommand: prepare TLC replay assets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    prepare_assets(args.ref_parquet, args.replay_parquet, args.output_dir, args.n_zones, args.seed)


def handle_run(args: argparse.Namespace) -> None:
    """
    Handle run subcommand: run replay with specified mode and configuration.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    mode = ReplayMode(args.mode)
    config = ReplayConfig(
        n_zones=args.n_zones,
        max_inflight_zones=args.max_inflight_zones,
        tick_timeout_s=args.tick_timeout_s,
        completion_fraction=args.completion_fraction,
        slow_zone_fraction=args.slow_zone_fraction,
        slow_zone_sleep_s=args.slow_zone_sleep_s,
        fallback_policy=args.fallback_policy,
        seed=args.seed,
        max_ticks=args.max_ticks,
    )
    run_replay(args.ray_address, args.prepared_dir, args.output_dir, mode, config)


def handle_reset(args: argparse.Namespace) -> None:
    """
    Handle reset subcommand: stop Ray and remove generated artifacts.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    reset_ray(project_dir=Path(__file__).parent)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
