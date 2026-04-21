import argparse
import logging

from pathlib import Path

from src.tlc import (
    DEFAULT_N_ZONES,
    DEFAULT_SEED,
    TICK_MINUTES,
    aggregate_ticks,
    build_baseline_table,
    build_replay_table,
    cross_check_replay,
    load_parquet,
    select_active_zones,
    validate_adjacent_months,
    write_json,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def prepare_assets(ref_parquet: Path, replay_parquet: Path, output_dir: Path, n_zones: int = DEFAULT_N_ZONES,
                   seed: int = DEFAULT_SEED) -> None:
    """
    Read reference and replay parquets, validate adjacent months, select active zones, build baseline and replay tables,
    run cross-check, and write prepared assets.

    Args:
        ref_parquet (Path): Path to the reference month parquet.
        replay_parquet (Path): Path to the replay month parquet.
        output_dir (Path): Directory to write prepared assets.
        n_zones (int): Number of active zones to select.
        seed (int): Random seed for reproducibility. Default is 42.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step A - load
    logger.info(f"Loading reference: {ref_parquet}")
    ref_df = load_parquet(ref_parquet)
    logger.info(f"Loading replay: {replay_parquet}")
    replay_df = load_parquet(replay_parquet)

    # Validate adjacent months
    ref_label, replay_label = validate_adjacent_months(ref_df, replay_df)

    # Step B - build prepared assets
    active_zones = select_active_zones(ref_df, n_zones)

    # Aggregate into ticks
    ref_agg = aggregate_ticks(ref_df)
    ref_agg = ref_agg[ref_agg["zone_id"].isin(active_zones)]  # Filter to active zones only for simplicity
    replay_agg = aggregate_ticks(replay_df)

    # Build baseline from reference
    baseline = build_baseline_table(ref_agg)

    # Build replay table filtered to active zones
    replay_table = build_replay_table(replay_agg, active_zones)

    # Cross-check
    cross_check_replay(replay_df, replay_table, active_zones)

    # Write prepared assets
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

    logger.info(f"Prepared assets written to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TLC replay assets")
    parser.add_argument("--ref-parquet", type=Path, required=True)
    parser.add_argument("--replay-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_assets(args.ref_parquet, args.replay_parquet, args.output_dir, args.n_zones, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
