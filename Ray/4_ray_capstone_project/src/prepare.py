"""
Data preparation module for TLC replay experiments.

Validates adjacent-month parquet files, selects the n busiest zones, aggregates pickups into 15-minute ticks, builds
per-zone baseline statistics (mean/std by hour and day of week), and writes prepared assets to disk.

Output artifacts:
- baseline.parquet: Per-zone baseline statistics
- replay.parquet: Aggregated replay demand by zone and tick
- active_zones.json: List of selected zone IDs
- prep_meta.json: Preparation metadata
"""

import argparse
import logging
import pandas as pd

from pathlib import Path

from src.core import (
    DEFAULT_N_ZONES,
    DEFAULT_SEED,
    TICK_MINUTES,
    aggregate_ticks,
    build_baseline_table,
    build_replay_table,
    cross_check_replay,
    load_parquet,
    identify_busiest_zones,
    validate_adjacent_months,
    write_json,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def write_prepared_assets(output_dir: Path, baseline: pd.DataFrame, replay_table: pd.DataFrame, active_zones: list,
                          ref_label: str, replay_label: str, ref_df: pd.DataFrame, replay_df: pd.DataFrame, seed: int) -> None:
    """
    Write all prepared assets to disk.

    Args:
        output_dir (Path): Directory to write artifacts to.
        baseline (pd.DataFrame): Baseline table with per-zone statistics.
        replay_table (pd.DataFrame): Replay table with aggregated demand.
        active_zones (List[int]): List of active zone IDs.
        ref_label (str): Reference month label.
        replay_label (str): Replay month label.
        ref_df (pd.DataFrame): Original reference dataframe for row count.
        replay_df (pd.DataFrame): Original replay dataframe for row count.
        seed (int): Random seed used for zone selection.
    """
    baseline.to_parquet(output_dir / "baseline.parquet", index=False)
    replay_table.to_parquet(output_dir / "replay.parquet", index=False)
    prep_meta = {
        "ref_label": ref_label,
        "replay_label": replay_label,
        "n_zones": len(active_zones),
        "tick_minutes": TICK_MINUTES,
        "seed": seed,
        "n_ref_rows": len(ref_df),
        "n_replay_rows": len(replay_df),
        "n_replay_ticks": replay_table["tick_start"].nunique(),
    }
    write_json(prep_meta, output_dir / "prep_meta.json")
    write_json(active_zones, output_dir / "active_zones.json")


def prepare_assets(ref_parquet: Path, replay_parquet: Path, output_dir: Path, n_zones: int, seed: int = DEFAULT_SEED) -> None:
    """
    Prepare assets for TLC replay experiments through two main steps.

    Step A - Load the monthly datasets:
        - Read the reference-month and replay-month parquet files into local Python dataframes

    Step B - Build the prepared assets:
        - Identify the busiest active pickup zones from the reference month
        - Aggregate reference demand into 15-minute ticks
        - Build the zone/time baseline table from the reference month
        - Aggregate replay-month pickups into one row per (zone_id, tick)
        - Write prepared assets for the runtime

    Args:
        ref_parquet (Path): Path to the reference month parquet.
        replay_parquet (Path): Path to the replay month parquet.
        output_dir (Path): Directory to write prepared assets.
        n_zones (int): Number of active zones to select.
        seed (int, optional): Random seed for reproducibility. Default is DEFAULT_SEED.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step A - load monthly datasets
    logger.info(f"Loading reference: {ref_parquet}")
    ref_df = load_parquet(ref_parquet)
    logger.info(f"Loading replay: {replay_parquet}")
    replay_df = load_parquet(replay_parquet)
    ref_label, replay_label = validate_adjacent_months(ref_df, replay_df)

    # Step B - build prepared assets
    active_zones = identify_busiest_zones(ref_df, n_zones, seed)
    ref_agg = aggregate_ticks(ref_df)
    baseline = build_baseline_table(ref_agg)
    replay_agg = aggregate_ticks(replay_df)
    replay_table = build_replay_table(replay_agg, active_zones)
    cross_check_replay(replay_df, replay_table, active_zones)  # Ensure replay table matches raw replay data for active zones
    write_prepared_assets(output_dir, baseline, replay_table, active_zones, ref_label, replay_label, ref_df, replay_df, seed)

    logger.info(f"Prepared assets written to {output_dir}")


def build_prepare_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for prepare command.

    Returns:
        argparse.ArgumentParser: Parser for prepare command arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare TLC replay assets from adjacent-month parquet files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False  # Disable help when used as parent parser
    )
    parser.add_argument("--ref-parquet", type=Path, required=True, help="Path to reference month parquet file")
    parser.add_argument("--replay-parquet", type=Path, required=True, help="Path to replay month parquet file")
    parser.add_argument("--output-dir", type=Path, default=Path("prepared"), help="Directory to write prepared assets")
    parser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES, help="Number of active zones to select")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for zone selection reproducibility")
    return parser


if __name__ == "__main__":
    standalone_parser = argparse.ArgumentParser(parents=[build_prepare_parser()])
    args = standalone_parser.parse_args()
    prepare_assets(args.ref_parquet, args.replay_parquet, args.output_dir, args.n_zones, args.seed)
