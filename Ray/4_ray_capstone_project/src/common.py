"""
Core constants, enums, and dataclasses for the Ray-based TLC replay system.
"""

import argparse

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


# Tick constants
TICK_MINUTES = 15  # Duration of each tick in minutes
DEMAND_WINDOW_SIZE = 6  # Number of recent demand values to track for scoring

# Default directories
DEFAULT_PREPARED_DIR = "prepared"  # Default directory for prepared assets
DEFAULT_OUTPUT_DIR = "output"  # Default output directory for run artifacts

# Default configuration parameters
DEFAULT_N_ZONES = 20  # Number of active zones to select for the experiment
DEFAULT_SEED = 42  # Default random seed for reproducibility
DEFAULT_COMPLETION_FRACTION = 0.75  # Minimum fraction of zones required for finalization
DEFAULT_MAX_INFLIGHT_ZONES = 4  # Max concurrent scoring tasks in async mode
DEFAULT_TICK_TIMEOUT_S = 2.0  # Tick timeout in seconds for async mode
DEFAULT_SLOW_ZONE_FRACTION = 0.25  # Fraction of zones to simulate as slow in async mode
DEFAULT_SLOW_ZONE_SLEEP_S = 1.0  # Artificial delay in seconds for slow zones in async mode

# Fallback policies
FALLBACK_POLICY_PREVIOUS = "always_previous"  # Fallback policy: always use previous tick's demand for late zones

# Validation constants
CROSS_CHECK_N_TICKS = 4  # Number of ticks to sample for cross-check validation
REQUIRED_PARQUET_COLS = ["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID"]  # Required columns in input files


class ReplayMode(str, Enum):
    BLOCKING = "blocking"
    ASYNC = "async"
    STRESS = "stress"


class DemandVerdict(str, Enum):
    """
    Decision verdict for a zone at a given tick, based on recent demand vs baseline.
    """
    NEED = "NEED"
    OK = "OK"


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


@dataclass
class ReplayConfig(RoundedDataclass):
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
    max_ticks: int = None  # None or 0 mean no limit

    @classmethod
    def from_args(cls: type["ReplayConfig"], args: argparse.Namespace) -> "ReplayConfig":
        """
        Create ReplayConfig from parsed command-line arguments.

        Args:
            args (argparse.Namespace): argparse.Namespace with parsed arguments

        Returns:
            ReplayConfig: ReplayConfig instance populated from args
        """
        return cls(
            n_zones=args.n_zones,
            tick_minutes=args.tick_minutes,
            max_inflight_zones=args.max_inflight_zones,
            tick_timeout_s=args.tick_timeout_s,
            completion_fraction=args.completion_fraction,
            slow_zone_fraction=args.slow_zone_fraction,
            slow_zone_sleep_s=args.slow_zone_sleep_s,
            fallback_policy=args.fallback_policy,
            seed=args.seed,
            max_ticks=args.max_ticks,
        )


@dataclass
class ZoneSnapshot(RoundedDataclass):
    """
    Snapshot of demand and baseline for a zone at a given tick, used as input for scoring.
    """
    zone_id: int
    tick_id: int
    recent_demand: List[float] = field(default_factory=list)
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    is_slow_zone: bool = False
    slow_sleep_s: float = 0.0


@dataclass
class ScoringResult(RoundedDataclass):
    """
    Result of scoring for a zone at a given tick.
    """
    zone_id: int
    tick_id: int
    decision: DemandVerdict
    task_latency_s: float = 0.0


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
