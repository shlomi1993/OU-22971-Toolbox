"""
The main entry point for the Ray-based distributed TLC zone recommendation system under skew.

Provides three subcommands:
- `prepare`: Read TLC parquet files and prepare assets for replay experiments.
- `run`: Execute blocking, async, or stress mode with Ray distributed actors.
- `reset`: Stop Ray and delete generated artifacts.

Example usage:
    python main.py prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir output/prepared/
    python main.py run --prepared-dir output/prepared/ --mode async --output-dir output/run/
    python main.py reset

Each subcommand can also be run independently as a standalone script:
    python src/prepare.py --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir output/prepared/
    python src/run.py --prepared-dir output/prepared/ --mode async --output-dir output/run/
    python scripts/reset.py

Or via a dedicated command available after installing the package:
    prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir output/prepared/
    run --prepared-dir output/prepared/ --mode async --output-dir output/run/
    reset
"""

import argparse

from scripts.reset import reset, build_reset_parser
from src.prepare import prepare_assets, build_prepare_parser
from src.run import run_replay, build_run_parser
from src.common import ReplayConfig, ReplayMode


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
    prepare_help = "Prepare replay assets from TLC parquet files"
    prepare_subparser = subparsers.add_parser(
        name="prepare",
        help=prepare_help,
        description=prepare_help,
        parents=[build_prepare_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare_subparser.set_defaults(handler=handle_prepare)

    # Add run subcommand
    run_help = "Run distributed replay in blocking, async, or stress mode"
    run_subparser = subparsers.add_parser(
        name="run",
        help=run_help,
        description=run_help,
        parents=[build_run_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_subparser.set_defaults(handler=handle_run)

    # Add reset subcommand
    reset_help = "Stop Ray and delete generated artifacts (output/ directory)"
    reset_subparser = subparsers.add_parser(
        name="reset",
        help=reset_help,
        description=reset_help,
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
    Handle reset subcommand: stop Ray and clean up artifacts.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    reset(prepared_dir=args.prepared_dir, output_dir=args.output_dir)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
