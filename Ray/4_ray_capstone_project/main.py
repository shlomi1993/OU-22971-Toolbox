"""
Ray capstone main entry: TLC-backed per-zone recommendations under skew.

CLI entry point for the Ray capstone project.
Workflow: prepare TLC replay assets -> initialize per-zone actors -> compare blocking and async execution.

Provides three subcommands:
- `prepare`: Validate TLC parquet data, select active zones, and build baseline and replay tables.
- `run`: Execute blocking, async, or stress mode with Ray distributed actors.
- `reset`: Stop Ray and delete generated artifacts.

Example usage:
    python main.py prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir prepared/
    python main.py run --prepared-dir prepared/ --mode async --output-dir output/
    python main.py reset

Each subcommand can also be executed as a standalone script (e.g., python src/prepare.py).
"""

import argparse

from src.prepare import prepare_assets, build_prepare_parser
from src.reset import reset_ray, build_reset_parser
from src.run import run_replay, build_run_parser
from src.core import ReplayConfig, ReplayMode


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
        parents=[build_prepare_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare_subparser.set_defaults(handler=handle_prepare)

    # Add run subcommand
    run_subparser = subparsers.add_parser(
        name="run",
        help="Run replay in blocking/async/stress mode",
        parents=[build_run_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_subparser.set_defaults(handler=handle_run)

    # Add reset subcommand
    reset_subparser = subparsers.add_parser(
        name="reset",
        help="Stop Ray and remove generated artifacts",
        parents=[build_reset_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    config = ReplayConfig.from_args(args)
    run_replay(args.ray_address, args.prepared_dir, args.output_dir, mode, config)


def handle_reset(args: argparse.Namespace) -> None:
    """
    Handle reset subcommand: stop Ray and remove generated artifacts.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    reset_ray(args.prepared_dir, args.output_dir)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
