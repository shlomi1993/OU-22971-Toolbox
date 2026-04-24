"""
Abstract base class for TLC zone recommendation replay execution.

Defines the template method pattern for running distributed replays using Ray actors and tasks.
Subclasses should implement mode-specific behavior.
"""

import logging
import time
import numpy as np
import pandas as pd
import ray

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from ray.actor import ActorHandle

from src.tlc import (
    PreparedData,
    RunConfig,
    TickMetrics,
    load_prepared,
    write_json,
    write_latency_log,
    write_metrics_csv,
    write_tick_summary,
)
from src.zone_actor import ZoneActor, ZoneSnapshot


logger = logging.getLogger(__name__)


@dataclass
class InitializedRuntime:
    """
    Result of initialization containing actors, skew configuration, and tick parameters.
    """
    actors: Dict[int, ActorHandle]
    slow_zones: set[int]
    tick_ids: List[int]
    max_ticks: int


class Replay(ABC):
    """
    Abstract base class defining the template method for replay execution.

    Template method pattern: the run() method orchestrates the common flow while delegating mode-specific behavior to
    abstract methods implemented by subclasses.
    """

    def __init__(self, prepared_dir: Path, output_dir: Path, config: RunConfig) -> None:
        """
        Initialize the replay system.

        Args:
            prepared_dir (Path): Directory containing prepared assets from prepare.py
            output_dir (Path): Root output directory for artifacts
            config (RunConfig): Runtime configuration for the replay
        """
        self.prepared_dir = prepared_dir
        self.output_dir = output_dir
        self.config = config
        self.runtime = None  # Will hold InitializedRuntime after initialization
        self.all_metrics: List[TickMetrics] = []
        self.all_decisions: Dict[int, Dict[int, str]] = {}

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """
        Get the display name for this execution mode.

        Returns:
            str: Mode name (e.g., "Blocking", "Async")
        """
        raise NotImplementedError()

    def _create_actors(self, prepared: PreparedData) -> Dict[int, ActorHandle]:
        """
        Create one ZoneActor per active zone.

        Args:
            prepared (PreparedData): Prepared data containing replay, baseline, and active zones

        Returns:
            Dict[int, ActorHandle]: Mapping of zone_id to Ray actor handle
        """
        actors = {}
        for zone_id in prepared.active_zones:
            zone_replay = prepared.replay[prepared.replay["zone_id"] == zone_id].reset_index(drop=True)
            zone_baseline = prepared.baseline[prepared.baseline["zone_id"] == zone_id].reset_index(drop=True)
            actors[zone_id] = ZoneActor.remote(zone_id, zone_replay, zone_baseline, self.config)
        logger.info(f"Created {len(actors)} ZoneActors")
        return actors

    def _select_slow_zones(self, active_zones: List[int]) -> set[int]:
        """
        Deterministically select slow zones based on config.

        Args:
            active_zones (List[int]): List of active zone IDs

        Returns:
            set[int]: Zone IDs that will receive artificial delay
        """
        rng = np.random.RandomState(self.config.seed)
        n_slow = max(1, int(len(active_zones) * self.config.slow_zone_fraction))
        slow = set(rng.choice(active_zones, size=n_slow, replace=False))
        logger.info(f"Slow zones ({len(slow)}): {sorted(slow)}")
        return slow

    @staticmethod
    def _get_tick_ids(replay_df: pd.DataFrame) -> List[int]:
        """
        Return list of tick indices from replay DataFrame.

        Args:
            replay_df (pd.DataFrame): Replay DataFrame with tick_start column

        Returns:
            List[int]: Sequential tick indices [0, 1, ..., n_ticks - 1]
        """
        n_ticks = replay_df["tick_start"].nunique()
        return list(range(n_ticks))

    def _apply_tick_limit(self, tick_ids: List[int]) -> tuple[List[int], int]:
        """
        Apply max_ticks limit from config if set.

        Args:
            tick_ids (List[int]): Full list of tick IDs from replay data

        Returns:
            Tuple[List[int], int]: (Possibly truncated list of tick IDs, max_ticks value)
        """
        if self.config.max_ticks and self.config.max_ticks > 0:
            tick_ids = tick_ids[:self.config.max_ticks]
        return tick_ids, len(tick_ids)

    def _initialize_runtime(self) -> InitializedRuntime:
        """
        Step C - Initialize the runtime.

        - Create one ZoneActor per active zone
        - Give each actor ownership of its own prepared replay partition
        - Initialize any global run configuration and output locations

        Returns:
            InitializedRuntime: Initialized actors and execution parameters
        """
        prepared = load_prepared(self.prepared_dir)
        actors = self._create_actors(prepared)
        slow_zones = self._select_slow_zones(prepared.active_zones)
        tick_ids = self._get_tick_ids(prepared.replay)
        tick_ids, max_ticks = self._apply_tick_limit(tick_ids)
        return InitializedRuntime(actors, slow_zones, tick_ids, max_ticks)

    def _advance_replay_tick(self, tick_id: int) -> Dict[int, ZoneSnapshot]:
        """
        Step D - Advance one replay tick.

        At the start of each tick:
        - Tell each actor that this tick is now active
        - Ask each actor for the snapshot needed for the next recommendation
        - Keep the snapshot minimal and derived from actor-owned state

        Args:
            tick_id (int): Current tick ID

        Returns:
            Dict[int, ZoneSnapshot]: Mapping of zone_id to snapshot for this tick
        """
        ray.get([actor.activate_tick.remote(tick_id) for actor in self.runtime.actors.values()])
        snapshot_refs = {zone_id: actor.get_snapshot.remote(tick_id) for zone_id, actor in self.runtime.actors.items()}
        return {zone_id: ray.get(ref) for zone_id, ref in snapshot_refs.items()}

    @abstractmethod
    def _run_scoring(self, tick_id: int, snapshots: Dict[int, ZoneSnapshot]) -> None:
        """
        Step E - Run per-zone scoring.

        Blocking mode:
        - Submit each zone snapshot to a scoring task
        - Collect all task returns in the controller for the current tick

        Async mode:
        - Submit each zone snapshot to a scoring task
        - Have each scoring task report its result to that zone's actor for the current tick

        Args:
            tick_id (int): Current tick ID
            snapshots (Dict[int, ZoneSnapshot]): Mapping of zone_id to snapshot
        """
        raise NotImplementedError()

    @abstractmethod
    def _finalize_tick(self, tick_id: int) -> Dict[int, bool]:
        """
        Step F - Finalize the tick under partial readiness.

        Blocking mode:
        - Wait until all task results for the current tick have been returned to the controller
        - Close the tick only after the controller has a complete result set

        Async mode:
        - Check which actors have already received a report for the current tick by asking
          actors for their current tick status
        - Decide whether the policy says to keep waiting or to close the tick

        Args:
            tick_id (int): Current tick ID

        Returns:
            Dict[int, bool]: Mapping of zone_id to readiness status (True if zone has a decision, False otherwise)
        """
        raise NotImplementedError()

    @abstractmethod
    def _close_tick(self, tick_id: int, readiness: Dict[int, bool]) -> None:
        """
        Step G - Close the tick in each actor.

        Blocking mode:
        - Have the controller write the accepted decision for each zone into its actor
        - Ensure duplicate accepted writes for the same zone and tick are safe to replay

        Async mode:
        - Ask each actor to finalize the current tick using either its reported decision
          or the fallback policy
        - Ensure duplicate reports for the same zone and tick are safe to replay
        - Late results that arrive after finalization should be logged and ignored by the actor

        In both modes:
        - Update actor state needed for the next tick

        Args:
            tick_id (int): Current tick ID
            readiness (Dict[int, bool]): Mapping of zone_id to readiness status from _finalize_tick
        """
        raise NotImplementedError()

    @abstractmethod
    def _collect_tick_metrics(self, tick_id: int, readiness: Dict[int, bool], tick_elapsed: float) -> TickMetrics:
        """
        Collect metrics for the completed tick.

        Args:
            tick_id (int): Current tick ID
            readiness (Dict[int, bool]): Mapping of zone_id to readiness status from _finalize_tick
            tick_elapsed (float): Total time elapsed for this tick in seconds

        Returns:
            TickMetrics: Metrics object for this tick, including latency, readiness, and any other relevant data
        """
        raise NotImplementedError()

    def _write_artifacts(self) -> None:
        """
        Step H - Finalize artifacts.

        - Aggregate latency, tick-level metrics, and actor-accepted decisions
        - Compare blocking and asynchronous runs on the same replay window and configuration
        """
        mode_dir = self.output_dir / self.mode_name.lower()
        mode_dir.mkdir(parents=True, exist_ok=True)

        write_json(self.config.to_dict(), mode_dir / "run_config.json")
        write_metrics_csv(self.all_metrics, mode_dir / "metrics.csv")
        write_latency_log(self.all_metrics, mode_dir / "latency_log.json")
        write_tick_summary(self.all_metrics, self.all_decisions, mode_dir / "tick_summary.json")

        # Collect actor counters
        counter_refs = {zone_id: actor.get_counters.remote() for zone_id, actor in self.runtime.actors.items()}
        counters = {zone_id: ray.get(ref) for zone_id, ref in counter_refs.items()}
        write_json([c.to_dict() for c in counters.values()], mode_dir / "actor_counters.json")

        logger.info(f"Artifacts written to {mode_dir}")

    def run(self) -> List[TickMetrics]:
        """
        Template method orchestrating the complete replay flow.

        Flow steps:
        1. Initialize the runtime (step C)
        2. For each tick:
           a. Advance the tick and get snapshots (step D)
           b. Run per-zone scoring (step E)
           c. Finalize the tick under partial readiness (step F)
           d. Close the tick in actors (step G)
           e. Collect tick metrics
        3. Finalize artifacts (step H)

        Returns:
            List[TickMetrics]: Per-tick metrics for the replay run
        """
        # Step C - Initialize the runtime
        self.runtime = self._initialize_runtime()

        logger.info(f"\n{self.mode_name} Replay")

        # Steps D-G - Run each tick
        for tick_id in self.runtime.tick_ids:
            tick_start = time.time()
            logger.info(f"[{self.mode_name.lower()}] tick {tick_id}/{self.runtime.max_ticks - 1}")

            # Step D - Advance one replay tick
            snapshots = self._advance_replay_tick(tick_id)

            # Step E - Run per-zone scoring
            self._run_scoring(tick_id, snapshots)

            # Step F - Finalize the tick under partial readiness
            readiness = self._finalize_tick(tick_id)

            # Step G - Close the tick in each actor
            self._close_tick(tick_id, readiness)

            # Collect tick metrics
            tick_elapsed = time.time() - tick_start
            metrics = self._collect_tick_metrics(tick_id, readiness, tick_elapsed)
            self.all_metrics.append(metrics)

        # Step H - Finalize artifacts
        self._write_artifacts()

        return self.all_metrics
