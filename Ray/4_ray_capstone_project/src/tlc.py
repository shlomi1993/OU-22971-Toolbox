"""
Shared utilities and data structures for TLC replay experiments.

Provides:
- Constants: Tick duration, default parameters, fallback policies
- Enums: RunMode
- Dataclasses: RunConfig, TickMetrics, PreparedData
- Data functions: Parquet loading, validation, aggregation, baseline building
- Artifact writers: JSON, CSV, latency logs, tick summaries
- Helper functions: Zone selection, slow zone sampling, prepared asset loading
"""

import json
import logging
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


CROSS_CHECK_N_TICKS = 4  # Number of ticks to sample for cross-check validation
DEFAULT_COMPLETION_FRACTION = 0.75  # Minimum fraction of zones required for finalization
DEFAULT_MAX_INFLIGHT_ZONES = 4  # Max concurrent scoring tasks in async mode
DEFAULT_N_ZONES = 20  # Number of active zones to select for the experiment
DEFAULT_SEED = 42  # Default random seed for reproducibility
DEFAULT_SLOW_ZONE_FRACTION = 0.25  # Fraction of zones to simulate as slow in async mode
DEFAULT_SLOW_ZONE_SLEEP_S = 1.0  # Artificial delay in seconds for slow zones in async mode
DEFAULT_TICK_TIMEOUT_S = 2.0  # Tick timeout in seconds for async mode
DEMAND_WINDOW_SIZE = 6  # Number of recent demand values to track for scoring
FALLBACK_POLICY_PREVIOUS = "always_previous"  # Fallback policy: always use previous tick's demand for late zones
REQUIRED_PARQUET_COLS = ["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID"]  # Required columns in input files
TICK_MINUTES = 15  # Duration of each tick in minutes


@dataclass
class RoundedDataclass:
    """
    Base dataclass with utility to round floats and stringify keys for JSON serialization.
    """

    @staticmethod
    def _round_floats(obj: Any, n_digits: int = 4) -> Any:
        """
        Recursively round floats in nested dicts/lists and stringify dict keys.

        Args:
            obj (Any): The object to round (can be a float, dict, list, or other).
            n_digits (int): Number of decimal places to round to.

        Returns:
            Any: The object with floats rounded and dict keys stringified.
        """
        if isinstance(obj, float):
            return round(obj, n_digits)
        if isinstance(obj, dict):
            return {str(k): RoundedDataclass._round_floats(v, n_digits) for k, v in obj.items()}
        if isinstance(obj, list):
            return [RoundedDataclass._round_floats(item, n_digits) for item in obj]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to dict with rounded floats and string keys.

        Returns:
            Dict[str, Any]: The dataclass as a dict with rounded floats.
        """
        return self._round_floats(asdict(self))

class RunMode(str, Enum):
    BLOCKING = "blocking"
    ASYNC = "async"
    STRESS = "stress"


@dataclass
class RunConfig(RoundedDataclass):
    """
    Runtime configuration for the replay loop.
    """
    n_zones: int = DEFAULT_N_ZONES
    tick_minutes: int = TICK_MINUTES
    max_inflight_zones: int = DEFAULT_MAX_INFLIGHT_ZONES
    tick_timeout_s: float = DEFAULT_TICK_TIMEOUT_S
    completion_fraction: float = DEFAULT_COMPLETION_FRACTION
    slow_zone_fraction: float = DEFAULT_SLOW_ZONE_FRACTION
    slow_zone_sleep_s: float = DEFAULT_SLOW_ZONE_SLEEP_S
    fallback_policy: str = FALLBACK_POLICY_PREVIOUS
    seed: int = DEFAULT_SEED
    max_ticks: int = None  # None = no limit


@dataclass
class TickMetrics(RoundedDataclass):
    """
    Metrics for a single tick.
    """
    tick_id: int
    mode: str
    n_zones_completed: int = 0
    n_zones_fallback: int = 0
    n_late_reports: int = 0
    n_duplicate_reports: int = 0
    mean_zone_latency_s: float = 0.0
    max_zone_latency_s: float = 0.0
    max_mean_ratio: float = 0.0
    total_tick_latency_s: float = 0.0
    per_zone_latency: Dict[int, float] = field(default_factory=dict)


@dataclass
class PreparedData:
    """
    Prepared assets loaded from disk.
    """
    replay: pd.DataFrame
    baseline: pd.DataFrame
    active_zones: List[int]


def load_parquet(path: Path) -> pd.DataFrame:
    """
    Load a parquet file and validate required columns.

    Args:
        path (Path): Path to the parquet file.

    Returns:
        pd.DataFrame: Parquet file contents.
    """
    df = pd.read_parquet(path)
    missing = set(REQUIRED_PARQUET_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def validate_adjacent_months(ref_df: pd.DataFrame, replay_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Validate that reference and replay datasets are adjacent months from the same year.

    Args:
        ref_df (pd.DataFrame): Reference month data.
        replay_df (pd.DataFrame): Replay month data.

    Returns:
        Tuple[str, str]: (ref_label, replay_label) in "YYYY-MM" format. Example: ("2021-01", "2021-02")
    """
    ref_dates = pd.to_datetime(ref_df["lpep_pickup_datetime"], errors="coerce")
    replay_dates = pd.to_datetime(replay_df["lpep_pickup_datetime"], errors="coerce")

    ref_year = ref_dates.dt.year.mode().iloc[0]
    ref_month = ref_dates.dt.month.mode().iloc[0]
    replay_year = replay_dates.dt.year.mode().iloc[0]
    replay_month = replay_dates.dt.month.mode().iloc[0]

    if ref_year != replay_year:
        raise ValueError(f"Reference year {ref_year} != replay year {replay_year}")

    expected_next = ref_month + 1 if ref_month < 12 else 1
    if replay_month != expected_next:
        raise ValueError(f"Replay month {replay_month} is not adjacent to reference month {ref_month} (expected {expected_next})")

    ref_label = f"{ref_year}-{ref_month:02d}"
    replay_label = f"{replay_year}-{replay_month:02d}"
    logger.info(f"Validated adjacent months: reference={ref_label}, replay={replay_label}")
    return ref_label, replay_label


def identify_busiest_zones(ref_df: pd.DataFrame, n_zones: int, seed: int = DEFAULT_SEED) -> List[int]:
    """
    Select the n busiest pickup zones from the reference month deterministically.

    Args:
        ref_df (pd.DataFrame): Reference month data.
        n_zones (int): Number of active zones to select.
        seed (int): Random seed for deterministic tie-breaking when zones have equal counts.

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
        tick_minutes (int): Width of each tick in minutes.

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
        tick_minutes (int): The tick width in minutes used for aggregation, to ensure consistent tick boundaries in the
            direct calculation.

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


def write_json(data: Any, path: Path) -> None:
    """
    Write a JSON-serializable object to disk.

    Args:
        data (Any): JSON-serializable object to write.
        path (Path): Path to the output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Wrote {path}")


def write_metrics_csv(tick_metrics: List[TickMetrics], path: Path) -> None:
    """
    Write tick metrics to a CSV file.

    Args:
        tick_metrics (List[TickMetrics]): List of tick metrics to write.
        path (Path): Path to the output CSV file.
    """
    rows = [m.to_dict() for m in tick_metrics]
    for row in rows:
        row.pop("per_zone_latency", None)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Wrote {path}")


def write_tick_summary(tick_metrics: List[TickMetrics], decisions: Dict[int, Dict[int, str]], path: Path) -> None:
    """
    Write a tick-level summary JSON with decisions and metrics.

    Args:
        tick_metrics (List[TickMetrics]): List of tick metrics.
        decisions (Dict[int, Dict[int, str]]): Decisions for each tick and zone.
        path (Path): Path to the output JSON file.
    """
    summary = []
    for m in tick_metrics:
        tick_decisions = decisions.get(m.tick_id, {})
        summary.append({
            **m.to_dict(),
            "decisions": {str(z): d for z, d in tick_decisions.items()},
        })
    write_json(summary, path)


def write_latency_log(tick_metrics: List[TickMetrics], path: Path) -> None:
    """
    Write per-zone latency log as JSON.

    Args:
        tick_metrics (List[TickMetrics]): List of tick metrics.
        path (Path): Path to the output JSON file.
    """
    log_entries = []
    for m in tick_metrics:
        for zone_id, lat in m.per_zone_latency.items():
            log_entries.append({"tick_id": m.tick_id, "zone_id": zone_id, "latency_s": round(lat, 4)})
    write_json(log_entries, path)


def load_prepared(prepared_dir: Path) -> PreparedData:
    """
    Load prepared assets from disk.

    Args:
        prepared_dir (Path): Directory containing baseline.parquet, replay.parquet, active_zones.json.

    Returns:
        PreparedData: Dataclass containing replay table, baseline table, and active zones list.
    """
    replay = pd.read_parquet(prepared_dir / "replay.parquet")
    baseline = pd.read_parquet(prepared_dir / "baseline.parquet")
    with open(prepared_dir / "active_zones.json") as f:
        active_zones = json.load(f)
    return PreparedData(replay, baseline, active_zones)


def select_slow_zones(active_zones: List[int], config: RunConfig) -> set:
    """
    Deterministically select slow zones based on config.

    Args:
        active_zones (List[int]): List of active zone IDs.
        config (RunConfig): Runtime configuration (uses seed and slow_zone_fraction).

    Returns:
        set: Zone IDs that will receive artificial delay.
    """
    rng = np.random.RandomState(config.seed)
    n_slow = max(1, int(len(active_zones) * config.slow_zone_fraction))
    slow = set(rng.choice(active_zones, size=n_slow, replace=False))
    logger.info(f"Slow zones ({len(slow)}): {sorted(slow)}")
    return slow


def get_tick_ids(replay: pd.DataFrame) -> List[int]:
    """
    Return list of tick indices.

    Args:
        replay (pd.DataFrame): Replay table with tick_start column.

    Returns:
        List[int]: Sequential tick indices [0, 1, ..., n_ticks - 1].
    """
    n_ticks = replay["tick_start"].nunique()
    return list(range(n_ticks))
