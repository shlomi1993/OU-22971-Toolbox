
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


TICK_MINUTES = 15
DEFAULT_N_ZONES = 20
DEFAULT_SEED = 42
FALLBACK_POLICY_PREVIOUS = "always_previous"

REQUIRED_PARQUET_COLS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
]


class Decision(str, Enum):
    """
    Scoring outcome for a zone at a given tick.
    """
    NEED = "NEED"
    OK = "OK"


class RunMode(str, Enum):
    BLOCKING = "blocking"
    ASYNC = "async"
    STRESS = "stress"


@dataclass
class RoundedDataclass:
    """
    Mixin base for dataclasses that need JSON-friendly serialization.
    Provides to_dict() with recursive float rounding and string keys.
    """

    @staticmethod
    def _round_floats(obj: Any, n_digits: int = 4) -> Any:
        """
        Recursively round floats in nested dicts/lists and stringify dict keys.

        Args:
            obj (Any): The object to round (can be a float, dict, list, or other).
            n_digits (int): Number of decimal places to round to.

        Returns:
            Any: The rounded object.
        """
        if isinstance(obj, float):
            return round(obj, n_digits)
        if isinstance(obj, dict):
            return {str(k): RoundedDataclass._round_floats(v, n_digits) for k, v in obj.items()}
        if isinstance(obj, list):
            return [RoundedDataclass._round_floats(item, n_digits) for item in obj]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return self._round_floats(asdict(self))


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
class RunConfig(RoundedDataclass):
    """
    Runtime configuration for the replay loop.
    """
    n_zones: int = DEFAULT_N_ZONES
    tick_minutes: int = TICK_MINUTES
    max_inflight_zones: int = 4
    tick_timeout_s: float = 2.0
    completion_fraction: float = 0.75
    slow_zone_fraction: float = 0.25
    slow_zone_sleep_s: float = 1.0
    fallback_policy: str = FALLBACK_POLICY_PREVIOUS
    seed: int = DEFAULT_SEED


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
    Returns (ref_month_label, replay_month_label).
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
        raise ValueError(
            f"Replay month {replay_month} is not adjacent to reference month {ref_month} "
            f"(expected {expected_next})"
        )

    ref_label = f"{ref_year}-{ref_month:02d}"
    replay_label = f"{replay_year}-{replay_month:02d}"
    logger.info(f"Validated adjacent months: reference={ref_label}, replay={replay_label}")
    return ref_label, replay_label


def select_active_zones(ref_df: pd.DataFrame, n_zones: int, seed: int) -> List[int]:
    """
    Select the n busiest pickup zones from the reference month deterministically.
    """
    counts = ref_df.groupby("PULocationID").size().sort_values(ascending=False)
    rng = np.random.RandomState(seed)

    # Take top zones; if ties exist at the boundary, break with seed
    top = counts.head(n_zones * 2)
    if len(top) <= n_zones:
        selected = top.index.tolist()[:n_zones]
    else:
        selected = top.head(n_zones).index.tolist()

    selected.sort()
    logger.info(f"Selected {len(selected)} active zones: {selected}")
    return selected


def aggregate_ticks(df: pd.DataFrame, tick_minutes: int = TICK_MINUTES) -> pd.DataFrame:
    """
    Aggregate pickups into fixed-width time ticks.
    Returns DataFrame with columns: zone_id, tick_start, demand.
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
    Returns DataFrame with columns: zone_id, hour_of_day, day_of_week, mean_demand, std_demand.
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
    Sorted by tick_start for ordered replay.
    """
    replay = replay_agg[replay_agg["zone_id"].isin(active_zones)].copy()
    replay.sort_values(["tick_start", "zone_id"], inplace=True)
    replay.reset_index(drop=True, inplace=True)
    return replay


def cross_check_replay(raw_df: pd.DataFrame, replay_table: pd.DataFrame,
                       active_zones: List[int], tick_minutes: int = TICK_MINUTES) -> bool:
    """
    Pandas cross-check: confirm prepared replay counts match a direct grouped calculation
    on a sample window (first 4 ticks).
    """
    sample_ticks = sorted(replay_table["tick_start"].unique())[:4]
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
    """Write a JSON-serializable object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Wrote {path}")


def write_metrics_csv(tick_metrics: List[TickMetrics], path: Path) -> None:
    """Write tick metrics to a CSV file."""
    rows = [m.to_dict() for m in tick_metrics]
    for row in rows:
        row.pop("per_zone_latency", None)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Wrote {path}")


def write_tick_summary(tick_metrics: List[TickMetrics], decisions: Dict[int, Dict[int, str]],
                       path: Path) -> None:
    """
    Write a tick-level summary JSON with decisions and metrics.
    decisions: {tick_id: {zone_id: decision}}
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
    """Write per-zone latency log as JSON."""
    log_entries = []
    for m in tick_metrics:
        for zone_id, lat in m.per_zone_latency.items():
            log_entries.append({
                "tick_id": m.tick_id,
                "zone_id": zone_id,
                "latency_s": round(lat, 4),
            })
    write_json(log_entries, path)


def load_prepared(prepared_dir: Path):
    """
    Load prepared assets from disk.

    Args:
        prepared_dir (Path): Directory containing baseline.parquet, replay.parquet, active_zones.json.

    Returns:
        Tuple of (replay DataFrame, baseline DataFrame, active_zones list).
    """
    replay = pd.read_parquet(prepared_dir / "replay.parquet")
    baseline = pd.read_parquet(prepared_dir / "baseline.parquet")
    with open(prepared_dir / "active_zones.json") as f:
        active_zones = json.load(f)
    return replay, baseline, active_zones


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
