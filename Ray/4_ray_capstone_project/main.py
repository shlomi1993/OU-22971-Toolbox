# Ray capstone main entry: TLC-backed per-zone recommendations under skew.
# Runs: prepare TLC replay assets -> initialize per-zone actors -> compare blocking and async execution.

"""
CLI entry point for the Ray capstone project.

Provides two subcommands:
- `prepare`: Validate TLC parquet data, select active zones, build baseline and replay tables.
- `run`: Execute blocking, async, or stress mode with Ray distributed actors.

Example usage:
    python main.py prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir prepared/
    python main.py run --prepared-dir prepared/ --output-dir output/ --mode async
"""

import argparse
from pathlib import Path

from src.prepare import prepare_assets
from src.run import run_replay
from src.tlc import (
    DEFAULT_N_ZONES,
    DEFAULT_SEED,
    DEFAULT_MAX_INFLIGHT_ZONES,
    DEFAULT_TICK_TIMEOUT_S,
    DEFAULT_COMPLETION_FRACTION,
    DEFAULT_SLOW_ZONE_FRACTION,
    DEFAULT_SLOW_ZONE_SLEEP_S,
    FALLBACK_POLICY_PREVIOUS,
    RunMode,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Build main parser with prepare and run subcommands.

    Returns:
        argparse.ArgumentParser: Main parser with subcommands for prepare and run.
    """
    parser = argparse.ArgumentParser(
        description="TLC-backed per-zone recommendations under skew",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add prepare subcommand
    prepare_subparser = subparsers.add_parser("prepare", help="Prepare replay assets from TLC parquet files"
                                              )
    prepare_subparser.add_argument("--ref-parquet", type=Path, required=True, help="Path to reference month parquet file (e.g., green_tripdata_2023-01.parquet)")
    prepare_subparser.add_argument("--replay-parquet", type=Path, required=True, help="Path to replay month parquet file (e.g., green_tripdata_2023-02.parquet)")
    prepare_subparser.add_argument("--output-dir", type=Path, required=True, help="Directory to write prepared assets (baseline.parquet, replay.parquet, active_zones.json)")
    prepare_subparser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES, help=f"Number of active zones to select (default: {DEFAULT_N_ZONES})")
    prepare_subparser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for zone selection reproducibility (default: {DEFAULT_SEED})")
    prepare_subparser.set_defaults(handler=handle_prepare)

    # Add run subcommand
    run_subparser = subparsers.add_parser("run", help="Run replay in blocking/async/stress mode")
    run_subparser.add_argument("--prepared-dir", type=Path, required=True, help="Directory with prepared assets from prepare command")
    run_subparser.add_argument("--output-dir", type=Path, required=True, help="Root output directory for run artifacts")
    run_subparser.add_argument("--mode", choices=[m.value for m in RunMode], required=True, help="Execution mode: blocking (wait for all), async (bounded concurrency + timeout), or stress (harsh skew test)")
    run_subparser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES, help=f"Number of active zones to use (default: {DEFAULT_N_ZONES})")
    run_subparser.add_argument("--max-inflight-zones", type=int, default=DEFAULT_MAX_INFLIGHT_ZONES, help=f"Max concurrent scoring tasks in async mode (default: {DEFAULT_MAX_INFLIGHT_ZONES})")
    run_subparser.add_argument("--tick-timeout-s", type=float, default=DEFAULT_TICK_TIMEOUT_S, help=f"Tick timeout in seconds for async mode (default: {DEFAULT_TICK_TIMEOUT_S})")
    run_subparser.add_argument("--completion-fraction", type=float, default=DEFAULT_COMPLETION_FRACTION, help=f"Minimum fraction of zones required for finalization (default: {DEFAULT_COMPLETION_FRACTION})")
    run_subparser.add_argument("--slow-zone-fraction", type=float, default=DEFAULT_SLOW_ZONE_FRACTION, help=f"Fraction of zones to simulate as slow (default: {DEFAULT_SLOW_ZONE_FRACTION})")
    run_subparser.add_argument("--slow-zone-sleep-s", type=float, default=DEFAULT_SLOW_ZONE_SLEEP_S, help=f"Artificial delay in seconds for slow zones (default: {DEFAULT_SLOW_ZONE_SLEEP_S})")
    run_subparser.add_argument("--fallback-policy", default=FALLBACK_POLICY_PREVIOUS, help=f"Fallback policy for late zones (default: {FALLBACK_POLICY_PREVIOUS})")
    run_subparser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    run_subparser.add_argument("--ray-address", default=None, help="Ray cluster address (default: None for local mode)")
    run_subparser.set_defaults(handler=handle_run)

    return parser


def handle_prepare(args: argparse.Namespace) -> None:
    """
    Handle prepare subcommand: prepare TLC replay assets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    prepare_assets(
        ref_parquet=args.ref_parquet,
        replay_parquet=args.replay_parquet,
        output_dir=args.output_dir,
        n_zones=args.n_zones,
        seed=args.seed
    )


def handle_run(args: argparse.Namespace) -> None:
    """Handle run subcommand: run replay with specified mode and configuration.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    run_replay(
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        n_zones=args.n_zones,
        max_inflight_zones=args.max_inflight_zones,
        tick_timeout_s=args.tick_timeout_s,
        completion_fraction=args.completion_fraction,
        slow_zone_fraction=args.slow_zone_fraction,
        slow_zone_sleep_s=args.slow_zone_sleep_s,
        fallback_policy=args.fallback_policy,
        seed=args.seed,
        ray_address=args.ray_address,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
