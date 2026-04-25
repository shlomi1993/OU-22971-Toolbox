"""
Ray actor implementation for per-zone state management.

ZoneActor owns mutable replay state and accepted decisions for a single zone.
Supports both blocking and async modes:
- In blocking mode, the driver writes decisions directly.
- In async mode, scoring tasks report to the actor and the driver finalizes ticks with partial-readiness policy.

Key features:
- Idempotent writes keyed by (zone_id, tick_id)
- Fallback policy for incomplete ticks (always_previous)
- Observability counters for late/duplicate reports and fallback usage
"""

import hashlib
import numpy as np
import pandas as pd
import ray

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque

from src.core import DEMAND_WINDOW_SIZE, FALLBACK_POLICY_PREVIOUS, RoundedDataclass, ReplayConfig


Baseline = Dict[Tuple[int, int], Tuple[float, float]]


class Recommendation(str, Enum):
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
class ZoneRecommendation(RoundedDataclass):
    """
    Result from a scoring task.
    """
    zone_id: int
    tick_id: int
    decision: Recommendation
    task_latency_s: float = 0.0


@dataclass
class ZoneSnapshot(RoundedDataclass):
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

    def compute_decision(self) -> Recommendation:
        """
        Simple threshold scoring rule.

        Returns:
            Recommendation: "NEED" if recent demand exceeds baseline mean + 1 std, else "OK".
        """
        if not self.recent_demand:
            return Recommendation.OK

        recent_avg = np.mean(self.recent_demand)
        threshold = self.baseline_mean + max(self.baseline_std, 1.0)
        return Recommendation.NEED if recent_avg > threshold else Recommendation.OK


@dataclass
class ZoneCounters(RoundedDataclass):
    """
    Observability counters for a single zone actor.
    """
    zone_id: int
    n_duplicates: int = 0
    n_late: int = 0
    n_fallbacks: int = 0
    n_delayed_withheld: int = 0
    n_delayed_released: int = 0


@ray.remote
class ZoneActor:
    """
    Per-zone Ray actor owning mutable replay state and accepted decisions. Writes are idempotent by (zone_id, tick_id).
    In blocking mode the driver scores and writes decisions directly; in async mode scoring tasks report to the actor
    and the driver finalizes ticks under a partial-readiness policy.
    """

    def __init__(self, zone_id: int, replay_part: pd.DataFrame, baseline_part: pd.DataFrame, config: ReplayConfig) -> None:
        """
        Initialize the ZoneActor with its zone_id, replay data partition, baseline partition, and runtime config.

        Args:
            zone_id (int): The ID of the zone this actor is responsible for.
            replay_part (pd.DataFrame): The partition of the replay data for this zone.
            baseline_part (pd.DataFrame): The partition of the baseline data for this zone.
            config (ReplayConfig): Runtime configuration parameters.
        """
        self.zone_id = zone_id
        self.config = config

        # Replay data: {tick_start -> demand}
        self.replay = dict(zip(replay_part["tick_start"], replay_part["demand"]))
        self.tick_order = sorted(self.replay.keys())

        # Baseline: {(hour_of_day, day_of_week) -> (mean_demand, std_demand)}
        self.baseline: Baseline = {}
        for _, row in baseline_part.iterrows():
            key = (int(row["hour_of_day"]), int(row["day_of_week"]))
            self.baseline[key] = (float(row["mean_demand"]), float(row["std_demand"]))

        # Mutable state
        self.recent_demand: Deque[float] = deque(maxlen=DEMAND_WINDOW_SIZE)
        self.active_tick_id: Optional[int] = None
        self.reported_decision: Optional[ZoneRecommendation] = None
        self.accepted_decisions: Dict[int, str] = {}  # tick_id -> decision
        self.last_accepted_decision: Optional[str] = None

        # Observability counters
        self.n_duplicates = 0
        self.n_late = 0
        self.n_fallbacks = 0

        # Stretch A - Delayed arrival tracking
        self.pending_releases: Dict[int, List[Tuple[int, float]]] = {}  # release_tick -> [(orig_tick, demand)]
        self.withheld_ticks: set = set()
        self.visible_demand: Deque[float] = deque(maxlen=DEMAND_WINDOW_SIZE)
        self.n_delayed_withheld = 0
        self.n_delayed_released = 0
        self.delay_log: List[Dict] = []

        # Stretch B - Straggler tracking
        self.tick_latencies: List[float] = []
        self.straggler_ticks: int = 0

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

        # Update recent demand for snapshot using deque
        demand_window = list(self.recent_demand) + [current_demand]

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
        self.reported_decision = ZoneRecommendation(self.zone_id, tick_id, decision, latency)
        return WriteStatus.ACCEPTED

    def has_decision_for_tick(self, tick_id: int) -> bool:
        """
        Async mode: check if this actor has a reported decision for the given tick.

        Args:
            tick_id (int): The tick to check.

        Returns:
            bool: True if a decision exists (reported or already accepted).
        """
        return tick_id in self.accepted_decisions or (self.reported_decision is not None and self.reported_decision.tick_id == tick_id)

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

    @staticmethod
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
            return Recommendation.OK
        if last_decision is not None:
            return last_decision
        return Recommendation.OK

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
            decision = self.apply_fallback(fallback_policy, self.last_accepted_decision)
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
            ZoneCounters: Counters object with zone_id, n_duplicates, n_late, n_fallbacks,
                n_delayed_withheld, n_delayed_released.
        """
        return ZoneCounters(self.zone_id, self.n_duplicates, self.n_late, self.n_fallbacks, self.n_delayed_withheld,
                            self.n_delayed_released)

    @staticmethod
    def _is_demand_delayed(zone_id: int, tick_id: int, delayed_fraction: float, seed: int) -> bool:
        """
        Deterministically decide whether this tick's demand should be withheld.

        Uses a hash of (zone_id, tick_id, seed) for reproducibility.

        Args:
            zone_id (int): The zone ID.
            tick_id (int): The tick ID.
            delayed_fraction (float): Probability of withholding demand (0.0 – 1.0).
            seed (int): Random seed for deterministic behavior.

        Returns:
            bool: True if demand should be withheld for this tick.
        """
        hash = hashlib.sha256(f"{zone_id}:{tick_id}:{seed}".encode()).hexdigest()
        return (int(hash, 16) % 1000) < int(delayed_fraction * 1000)

    def activate_tick_delayed(self, tick_id: int, delay_ticks: int, delayed_fraction: float, seed: int) -> Dict:
        """
        Activate a tick under delayed-arrival semantics.

        1. Release any pending demand whose release tick has arrived.
        2. Determine whether this tick's demand is withheld.
        3. If withheld, buffer it for release at tick_id + delay_ticks.

        Args:
            tick_id (int): The tick to activate.
            delay_ticks (int): Number of ticks to withhold demand before release.
            delayed_fraction (float): Fraction of ticks whose demand is delayed.
            seed (int): Random seed for deterministic delay decisions.

        Returns:
            Dict: Info about released and withheld demand for logging.
        """
        self.active_tick_id = tick_id
        self.reported_decision = None

        info: Dict = {"released": [], "withheld": False, "release_tick": None}

        # Release pending demand that is due at this tick
        if tick_id in self.pending_releases:
            for orig_tick, demand in self.pending_releases.pop(tick_id):
                self.visible_demand.append(demand)
                self.n_delayed_released += 1
                entry = {"event": "release", "zone_id": self.zone_id,
                         "orig_tick": orig_tick, "release_tick": tick_id, "demand": demand}
                self.delay_log.append(entry)
                info["released"].append(entry)

        # Decide whether current tick's demand is withheld
        if tick_id < len(self.tick_order):
            tick_start = self.tick_order[tick_id]
            current_demand = float(self.replay.get(tick_start, 0.0))
        else:
            current_demand = 0.0
        if self._is_demand_delayed(self.zone_id, tick_id, delayed_fraction, seed):
            release_tick = tick_id + delay_ticks
            self.pending_releases.setdefault(release_tick, []).append((tick_id, current_demand))
            self.withheld_ticks.add(tick_id)
            self.n_delayed_withheld += 1
            entry = {"event": "withhold", "zone_id": self.zone_id,
                     "tick_id": tick_id, "release_tick": release_tick, "demand": current_demand}
            self.delay_log.append(entry)
            info["withheld"] = True
            info["release_tick"] = release_tick

        return info

    def get_snapshot_delayed(self, tick_id: int) -> ZoneSnapshot:
        """
        Return a snapshot that uses only *visible* demand.

        If the current tick's demand is withheld, the snapshot will not include it as the scoring logic sees an
        incomplete picture.

        Args:
            tick_id (int): The current tick index.

        Returns:
            ZoneSnapshot: Snapshot with visible demand only.
        """
        # Baseline lookup (same as regular get_snapshot)
        if tick_id < len(self.tick_order):
            tick_start = self.tick_order[tick_id]
            hour, dow = tick_start.hour, tick_start.dayofweek
        else:
            hour, dow = 12, 0
        baseline_mean, baseline_std = self.baseline.get((hour, dow), (0.0, 0.0))

        # Build demand window from visible demand only
        if tick_id in self.withheld_ticks:
            demand_window = list(self.visible_demand)  # current demand not included
        else:
            if tick_id < len(self.tick_order):
                tick_start = self.tick_order[tick_id]
                current_demand = float(self.replay.get(tick_start, 0.0))
            else:
                current_demand = 0.0
            demand_window = list(self.visible_demand) + [current_demand]

        return ZoneSnapshot(self.zone_id, tick_id, demand_window, baseline_mean, baseline_std)

    def finalize_tick_delayed(self, tick_id: int, fallback_policy: str) -> str:
        """
        Finalize a tick under delayed-arrival semantics.

        Args:
            tick_id (int): The tick to finalize.
            fallback_policy (str): Policy name for fallback.

        Returns:
            WriteStatus: WRITTEN or DUPLICATE.
        """
        if tick_id in self.accepted_decisions:
            return WriteStatus.DUPLICATE

        # Determine decision
        if self.reported_decision is not None and self.reported_decision.tick_id == tick_id:
            decision = self.reported_decision.decision
            used_fallback = False
        else:
            decision = self.apply_fallback(fallback_policy, self.last_accepted_decision)
            used_fallback = True

        # Accept decision
        self.accepted_decisions[tick_id] = decision
        self.last_accepted_decision = decision
        if used_fallback:
            self.n_fallbacks += 1

        # Update demand: always update recent_demand; only update visible if not withheld
        if tick_id < len(self.tick_order):
            tick_start = self.tick_order[tick_id]
            demand = float(self.replay.get(tick_start, 0.0))
            self.recent_demand.append(demand)
            if tick_id not in self.withheld_ticks:
                self.visible_demand.append(demand)

        return WriteStatus.WRITTEN

    def get_delay_log(self) -> List[Dict]:
        """
        Return the full delayed-arrival event log.

        Returns:
            List[Dict]: List of withhold / release events.
        """
        return list(self.delay_log)

    def record_tick_latency(self, latency: float, slow_threshold: float) -> None:
        """
        Record the observed task latency for the most recent tick.

        If the latency exceeds slow_threshold, increment the straggler counter.

        Args:
            latency (float): Observed scoring latency in seconds.
            slow_threshold (float): Latency threshold above which the tick counts as slow.
        """
        self.tick_latencies.append(latency)
        if latency > slow_threshold:
            self.straggler_ticks += 1

    def get_straggler_ticks(self) -> int:
        """
        Return the cumulative count of slow ticks for this zone.

        Returns:
            int: Number of ticks where observed latency exceeded the slow threshold.
        """
        return self.straggler_ticks
