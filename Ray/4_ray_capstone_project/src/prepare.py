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

import logging
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Tuple

from src.tlc import TICK_MINUTES, DEFAULT_SEED, load_parquet, write_json


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


CROSS_CHECK_N_TICKS = 4  # Number of ticks to sample for cross-check validation


def validate_adjacent_months(ref_df: pd.DataFrame, replay_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Validate that reference and replay datasets are adjacent months from the same year.

    Args:
        ref_df (pd.DataFrame): Reference month data.
        replay_df (pd.DataFrame): Replay month data.

    Returns:
        Tuple[str, str]: Reference and replay labels in "YYYY-MM" format. Example: ("2021-01", "2021-02")
    """
    ref_dates = pd.to_datetime(ref_df["lpep_pickup_datetime"], errors="coerce")
    replay_dates = pd.to_datetime(replay_df["lpep_pickup_datetime"], errors="coerce")

    ref_year = ref_dates.dt.year.mode().iloc[0]
    replay_year = replay_dates.dt.year.mode().iloc[0]
    if ref_year != replay_year:
        raise ValueError(f"Reference year {ref_year} != replay year {replay_year}")

    ref_month = ref_dates.dt.month.mode().iloc[0]
    replay_month = replay_dates.dt.month.mode().iloc[0]
    expected_next = ref_month + 1 if ref_month < 12 else 1
    if replay_month != expected_next:
        raise ValueError(f"Replay month {replay_month} is not adjacent to reference month {ref_month} (expected {expected_next})")

    ref_label = f"{ref_year}-{ref_month:02d}"
    replay_label = f"{replay_year}-{replay_month:02d}"
    logger.info(f"Validated adjacent months: reference={ref_label}, replay={replay_label}")
    return ref_label, replay_label


def identify_busiest_active_zones(ref_df: pd.DataFrame, n_zones: int, seed: int = DEFAULT_SEED) -> List[int]:
    """
    Select the n busiest pickup zones from the reference month deterministically.

    Args:
        ref_df (pd.DataFrame): Reference month data.
        n_zones (int): Number of active zones to select.
        seed (int, optional): Random seed for deterministic tie-breaking when zones have equal counts. Defaults to DEFAULT_SEED.

    Returns:
        List[int]: Sorted list of selected active zone IDs. Example: [138, 161, 238, ...]
    """
    counts = ref_df.groupby("PULocationID").size().sort_values(ascending=False)

    # Deterministic tie-breaking when multiple zones have same count at boundary
    rng = np.random.RandomState(seed)
    top = counts.head(n_zones * 2)

    if len(top) > n_zones:
        cutoff_value = top.iloc[n_zones - 1]
        candidates = top[top >= cutoff_value].index.tolist()  # Get all zones at or above the cutoff
        if len(candidates) > n_zones:
            selected = rng.choice(candidates, n_zones, replace=False)  # Use seeded random selection if too many candidates
        else:
            selected = candidates[:n_zones]
    else:
        selected = top.index.tolist()
    selected.sort()

    logger.info(f"Selected {len(selected)} active zones: {selected}")
    return selected


def aggregate_ticks(df: pd.DataFrame, tick_minutes: int = TICK_MINUTES) -> pd.DataFrame:
    """
    Aggregate pickups into fixed-width time ticks.

    Args:
        df (pd.DataFrame): Input data with lpep_pickup_datetime and PULocation columns.
        tick_minutes (int, optional): Width of each tick in minutes. Defaults to TICK_MINUTES.

    Returns:
        pd.DataFrame: Aggregated data.
    """
    df = df.copy()
    df["pickup_dt"] = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_dt"])
    df["tick_start"] = df["pickup_dt"].dt.floor(f"{tick_minutes}min")
    agg = df.groupby(["PULocationID", "tick_start"]).size().reset_index(name="demand")
    agg.rename(columns={"PULocationID": "zone_id"}, inplace=True)
    return agg


def build_baseline_table(ref_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-zone reference baseline by hour_of_day and day_of_week.

    Args:
        ref_agg (pd.DataFrame): Aggregated reference data with zone_id, tick_start, and demand.

    Returns:
        pd.DataFrame: Baseline data, one row per (zone_id, hour_of_day, day_of_week) with mean_demand and std_demand.
    """
    ref_agg = ref_agg.copy()
    ref_agg["hour_of_day"] = ref_agg["tick_start"].dt.hour
    ref_agg["day_of_week"] = ref_agg["tick_start"].dt.dayofweek
    baseline = (
        ref_agg.groupby(["zone_id", "hour_of_day", "day_of_week"])["demand"]
        .agg(["mean", "std"])
        .reset_index()
    )
    baseline.rename(columns={"mean": "mean_demand", "std": "std_demand"}, inplace=True)
    baseline["std_demand"] = baseline["std_demand"].fillna(0.0)
    return baseline


def build_replay_table(replay_agg: pd.DataFrame, active_zones: List[int]) -> pd.DataFrame:
    """
    Build the replay table: one row per (zone_id, tick_start), filtered to active zones.

    Args:
         replay_agg (pd.DataFrame): Aggregated replay data with zone_id, tick_start, and demand.
         active_zones (List[int]): List of active zone IDs to include in the replay table.

    Returns:
        pd.DataFrame: Replay table with zone_id, tick_start, and demand, filtered to active zones.
    """
    replay = replay_agg[replay_agg["zone_id"].isin(active_zones)].copy()
    replay.sort_values(["tick_start", "zone_id"], inplace=True)
    replay.reset_index(drop=True, inplace=True)
    return replay


def cross_check_replay(raw_df: pd.DataFrame, replay_table: pd.DataFrame, active_zones: List[int],
                       tick_minutes: int = TICK_MINUTES) -> bool:
    """
    Pandas cross-check: confirm prepared replay counts match a direct grouped calculation on a sample window.

    Args:
        raw_df (pd.DataFrame): Original raw replay data with lpep_pickup_datetime and PULocationID.
        replay_table (pd.DataFrame): Prepared replay table with zone_id, tick_start, and demand.
        active_zones (List[int]): List of active zone IDs that should be included in the check.
        tick_minutes (int, optional): The tick width in minutes used for aggregation, to ensure consistent tick boundaries in the
            direct calculation. Defaults to TICK_MINUTES.

    Returns:
        bool: True if the cross-check passes, False otherwise.
    """
    sample_ticks = sorted(replay_table["tick_start"].unique())[:CROSS_CHECK_N_TICKS]
    if len(sample_ticks) == 0:
        return True

    # Direct calculation from raw data
    raw = raw_df.copy()
    raw["pickup_dt"] = pd.to_datetime(raw["lpep_pickup_datetime"], errors="coerce")
    raw = raw.dropna(subset=["pickup_dt"])
    raw["tick_start"] = raw["pickup_dt"].dt.floor(f"{tick_minutes}min")
    raw = raw[raw["PULocationID"].isin(active_zones)]
    raw = raw[raw["tick_start"].isin(sample_ticks)]
    direct = raw.groupby(["PULocationID", "tick_start"]).size().reset_index(name="demand")
    direct.rename(columns={"PULocationID": "zone_id"}, inplace=True)

    # Compare with prepared table
    prepared = replay_table[replay_table["tick_start"].isin(sample_ticks)]
    merged = pd.merge(direct, prepared, on=["zone_id", "tick_start"], how="outer", suffixes=("_direct", "_prepared"))
    merged = merged.fillna(0)

    match = (merged["demand_direct"] == merged["demand_prepared"]).all()
    if match:
        logger.info("Cross-check PASSED: prepared replay counts match direct calculation")
    else:
        logger.warning("Cross-check FAILED: mismatch between prepared and direct counts")
    return bool(match)


def write_prepared_assets(output_dir: Path, baseline: pd.DataFrame, replay_table: pd.DataFrame, active_zones: List[int],
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
    Read reference and replay parquets, validate adjacent months, select active zones, build baseline and replay tables,
    run cross-check, and write prepared assets.

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
    active_zones = identify_busiest_active_zones(ref_df, n_zones, seed)
    ref_agg = aggregate_ticks(ref_df)
    baseline = build_baseline_table(ref_agg)
    replay_agg = aggregate_ticks(replay_df)
    replay_table = build_replay_table(replay_agg, active_zones)
    cross_check_replay(replay_df, replay_table, active_zones)  # Ensure replay table matches raw replay data for active zones
    write_prepared_assets(output_dir, baseline, replay_table, active_zones, ref_label, replay_label, ref_df, replay_df, seed)

    logger.info(f"Prepared assets written to {output_dir}")
