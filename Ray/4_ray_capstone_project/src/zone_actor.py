import pandas as pd
import ray

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.tlc_lib import (
    DEFAULT_N_ZONES,
    DEFAULT_SEED,
    FALLBACK_POLICY_PREVIOUS,
    FIRST_TICK_FALLBACK,
    TICK_MINUTES,
)


class Decision(str, Enum):
    """
    Scoring outcome for a zone at a given tick.
    """
    NEED = "NEED"
    OK = "OK"


class WriteStatus(str, Enum):
    """
    Status of a decision write attempt to a ZoneActor, used for observability.
    """
    ACCEPTED = "accepted"
    WRITTEN = "written"
    DUPLICATE = "duplicate"
    LATE = "late"


@dataclass
class RunConfig:
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ZoneDecision:
    """
    Result from a scoring task.
    """
    zone_id: int
    tick_id: int
    decision: Decision
    task_latency_s: float = 0.0


@dataclass
class ZoneSnapshot:
    """
    Minimal snapshot passed to a scoring task.
    """
    zone_id: int
    tick_id: int
    recent_demand: List[float] = field(default_factory=list)
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    is_slow_zone: bool = False
    slow_sleep_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_fallback(policy: str, last_decision: Optional[str]) -> str:
    """
    Apply the specified fallback policy to determine a decision when no scoring result is available.

    Args:
        policy (str): The name of the fallback policy to apply.
        last_decision (Optional[str]): The last accepted decision for this zone, if any.

    Returns:
        str: The fallback decision to apply.
    """
    if policy != FALLBACK_POLICY_PREVIOUS:
        return FIRST_TICK_FALLBACK
    if last_decision is not None:
        return last_decision
    return FIRST_TICK_FALLBACK


@dataclass
class ZoneCounters:
    """
    Observability counters for a single zone actor.
    """
    zone_id: int
    n_duplicates: int = 0
    n_late: int = 0
    n_fallbacks: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


Baseline = Dict[Tuple[int, int], Tuple[float, float]]


@ray.remote
class ZoneActor:
    """
    Per-zone Ray actor owning mutable replay state and accepted decisions. Writes are idempotent by (zone_id, tick_id).
    In blocking mode the driver scores and writes decisions directly; in async mode scoring tasks report to the actor
    and the driver finalizes ticks under a partial-readiness policy.
    """

    def __init__(self, zone_id: int, replay_partition: pd.DataFrame, baseline_partition: pd.DataFrame,
                 config: RunConfig) -> None:
        """
        Initialize the ZoneActor with its zone_id, replay data partition, baseline partition, and runtime config.

        Args:
            zone_id (int): The ID of the zone this actor is responsible for.
            replay_partition (pd.DataFrame): The partition of the replay data for this zone.
            baseline_partition (pd.DataFrame): The partition of the baseline data for this zone.
            config (RunConfig): Runtime configuration parameters.
        """
        self.zone_id = zone_id
        self.config = config

        # Replay data: {tick_start -> demand}
        self.replay = dict(zip(replay_partition["tick_start"], replay_partition["demand"]))
        self.tick_order = sorted(self.replay.keys())

        # Baseline: {(hour_of_day, day_of_week) -> (mean_demand, std_demand)}
        self.baseline: Baseline = {}
        for _, row in baseline_partition.iterrows():
            key = (int(row["hour_of_day"]), int(row["day_of_week"]))
            self.baseline[key] = (float(row["mean_demand"]), float(row["std_demand"]))

        # Mutable state
        self.recent_demand: List[float] = []
        self.active_tick_id: Optional[int] = None
        self.reported_decision: Optional[ZoneDecision] = None
        self.accepted_decisions: Dict[int, str] = {}  # tick_id -> decision
        self.last_accepted_decision: Optional[str] = None

        # Observability counters
        self.n_duplicates = 0
        self.n_late = 0
        self.n_fallbacks = 0

    def activate_tick(self, tick_id: int) -> None:
        """
        Mark a new tick as active

        Args:
            tick_id (int): The ID of the tick to activate.
        """
        self.active_tick_id = tick_id
        self.reported_decision = None

    def get_snapshot(self, tick_id: int) -> ZoneSnapshot:
        """
        Return the minimal snapshot needed by the scoring task for this zone and tick.

        Args:
            tick_id (int): The current tick index.

        Returns:
            ZoneSnapshot: Snapshot object with zone_id, tick_id, recent_demand, baseline_mean, baseline_std.
        """
        tick_start: Optional[pd.Timestamp]
        if tick_id < len(self.tick_order):
            tick_start = self.tick_order[tick_id]
            current_demand = float(self.replay.get(tick_start, 0.0))
        else:
            tick_start = None
            current_demand = 0.0

        # Baseline lookup
        if tick_start is not None:
            hour, dow = tick_start.hour, tick_start.dayofweek
        else:
            hour, dow = 12, 0
        baseline_mean, baseline_std = self.baseline.get((hour, dow), (0.0, 0.0))

        # Update recent demand for snapshot
        demand_window = list(self.recent_demand[-5:]) + [current_demand]

        return ZoneSnapshot(self.zone_id, tick_id, demand_window, baseline_mean, baseline_std)

    def report_decision(self, tick_id: int, decision: str, latency: float) -> WriteStatus:
        """
        Async mode: scoring task reports its decision to this actor.

        Args:
            tick_id (int): The tick this decision belongs to.
            decision (str): "NEED" or "OK".
            latency (float): Task execution time in seconds.

        Returns:
            WriteStatus: ACCEPTED, DUPLICATE, or LATE.
        """
        # Late: tick already closed or not the active tick
        if tick_id in self.accepted_decisions or tick_id != self.active_tick_id:
            self.n_late += 1
            return WriteStatus.LATE

        # Duplicate: already reported for this tick
        if self.reported_decision is not None and self.reported_decision.tick_id == tick_id:
            self.n_duplicates += 1
            return WriteStatus.DUPLICATE

        # Accept: store the reported decision (not yet finalized)
        self.reported_decision = ZoneDecision(self.zone_id, tick_id, decision, latency)
        return WriteStatus.ACCEPTED

    def has_decision_for_tick(self, tick_id: int) -> bool:
        """
        Async mode: check if this actor has a reported decision for the given tick.

        Args:
            tick_id (int): The tick to check.

        Returns:
            bool: True if a decision exists (reported or already accepted).
        """
        return tick_id in self.accepted_decisions or (self.reported_decision and self.reported_decision.tick_id == tick_id)

    def write_decision(self, tick_id: int, decision: str, used_fallback: bool = False) -> str:
        """
        Blocking mode: controller writes an accepted decision. Idempotent by (zone_id, tick_id).

        Args:
            tick_id (int): The tick to write the decision for.
            decision (str): "NEED" or "OK".
            used_fallback (bool, optional): Whether this decision came from the fallback policy. Defaults to False.

        Returns:
            WriteStatus: WRITTEN or DUPLICATE.
        """
        # Idempotent write: if already accepted for this tick, count as duplicate
        if tick_id in self.accepted_decisions:
            return WriteStatus.DUPLICATE

        # Accept the decision
        self.accepted_decisions[tick_id] = decision
        self.last_accepted_decision = decision
        if used_fallback:
            self.n_fallbacks += 1

        # Update recent demand (advance cursor)
        if tick_id < len(self.tick_order):
            tick_start = self.tick_order[tick_id]
            demand = float(self.replay.get(tick_start, 0.0))
            self.recent_demand.append(demand)

        # Write accepted decision
        return WriteStatus.WRITTEN

    def finalize_tick(self, tick_id: int, fallback_policy: str) -> str:
        """
        Async mode: finalize the tick using the reported decision or the fallback policy.
        Idempotent by (zone_id, tick_id).

        Args:
            tick_id (int): The tick to finalize.
            fallback_policy (str): Policy name to apply if no decision was reported.

        Returns:
            WriteStatus: WRITTEN or DUPLICATE.
        """
        if tick_id in self.accepted_decisions:
            return WriteStatus.DUPLICATE

        # Determine decision to finalize: reported decision or fallback
        if self.reported_decision is not None and self.reported_decision.tick_id == tick_id:
            decision = self.reported_decision.decision
            used_fallback = False
        else:
            decision = apply_fallback(fallback_policy, self.last_accepted_decision)
            used_fallback = True

        return self.write_decision(tick_id, decision, used_fallback)

    def get_accepted_decisions(self) -> Dict[int, str]:
        """
        Return the full history of accepted decisions.

        Returns:
            Dict[int, str]: Mapping of tick_id to accepted decision string.
        """
        return dict(self.accepted_decisions)

    def get_counters(self) -> ZoneCounters:
        """
        Return observability counters. Each actor tracks its own counters for simplicity.

        Returns:
            ZoneCounters: Counters object with zone_id, n_duplicates, n_late, n_fallbacks.
        """
        return ZoneCounters(self.zone_id, self.n_duplicates, self.n_late, self.n_fallbacks)
