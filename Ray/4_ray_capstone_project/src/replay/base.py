"""
Abstract base class for TLC zone recommendation replay execution.

Defines the template method pattern for running distributed replays using Ray actors and tasks. Subclasses should
implement the abstract methods to define specific execution modes.
"""

import time
import numpy as np
import pandas as pd
import ray

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
from ray.actor import ActorHandle

from src.artifacts import write_json, write_latency_log, write_metrics_csv, write_tick_summary
from src.common import ReplayConfig, TickMetrics, ZoneSnapshot
from src.data_preparation import PreparedData, load_prepared
from src.logger import logger
from src.zone_actor import ZoneActor


class Replay(ABC):
    """
    Abstract base class defining the template method for replay execution.

    Template method pattern: the run() method orchestrates the common flow while delegating mode-specific behavior to
    abstract methods implemented by subclasses.
    """

    def __init__(self, prepared_dir: Path, output_dir: Path, config: ReplayConfig) -> None:
        """
        Initialize the replay system.

        Args:
            prepared_dir (Path): Directory containing prepared assets from prepare.py
            output_dir (Path): Root output directory for artifacts
            config (ReplayConfig): Runtime configuration for the replay
        """
        self.prepared_dir = prepared_dir
        self.output_dir = output_dir
        self.config = config
        self.all_metrics = []
        self.all_decisions = {}
        self.actors = None
        self.slow_zones = None
        self.tick_ids = None
        self.max_ticks = None

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """
        Get the display name for this execution mode.

        Returns:
            str: Replay mode display name
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
        slow_zones = set(rng.choice(active_zones, size=n_slow, replace=False))
        logger.info(f"Slow zones ({len(slow_zones)}): {sorted(slow_zones)}")
        return slow_zones

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

    def _initialize_runtime(self) -> None:
        """
        Step C - Initialize the runtime:
            - Create one ZoneActor per active zone
            - Give each actor ownership of its own prepared replay partition
            - Initialize any global run configuration and output locations
        """
        self.all_metrics = []
        self.all_decisions = {}

        # Load prepared data and initialize runtime
        prepared = load_prepared(self.prepared_dir)
        self.actors = self._create_actors(prepared)
        self.slow_zones = self._select_slow_zones(prepared.active_zones)
        self.tick_ids = self._get_tick_ids(prepared.replay)
        self.tick_ids, self.max_ticks = self._apply_tick_limit(self.tick_ids)

    def _advance_replay_tick(self, tick_id: int) -> Dict[int, ZoneSnapshot]:
        """
        Step D - Advance one replay tick:
            - Tell each actor that this tick is now active
            - Ask each actor for the snapshot needed for the next recommendation
            - Keep the snapshot minimal and derived from actor-owned state

        Args:
            tick_id (int): Current tick ID

        Returns:
            Dict[int, ZoneSnapshot]: Mapping of zone ID to snapshot for this tick
        """
        ray.get([actor.activate_tick.remote(tick_id) for actor in self.actors.values()])
        snapshot_refs = {zone_id: actor.get_snapshot.remote(tick_id) for zone_id, actor in self.actors.items()}
        return {zone_id: ray.get(ref) for zone_id, ref in snapshot_refs.items()}

    @abstractmethod
    def _run_scoring(self, tick_id: int, snapshots: Dict[int, ZoneSnapshot]) -> None:
        """
        Step E - Run per-zone scoring:
            - Submit zone snapshots to scoring tasks
            - Handle result collection

        Args:
            tick_id (int): Current tick ID
            snapshots (Dict[int, ZoneSnapshot]): Mapping of zone ID to snapshot
        """
        raise NotImplementedError()

    @abstractmethod
    def _finalize_tick(self, tick_id: int) -> Dict[int, bool]:
        """
        Step F - Finalize the tick under partial readiness:
            - Determine which zones are ready
            - Decide when to proceed to tick closure

        Args:
            tick_id (int): Current tick ID

        Returns:
            Dict[int, bool]: Mapping of zone ID to readiness status (True if zone has a decision, False otherwise)
        """
        raise NotImplementedError()

    @abstractmethod
    def _close_tick(self, tick_id: int, readiness: Dict[int, bool]) -> None:
        """
        Step G - Close the tick in each actor:
            - Finalize decisions for the tick and update actor state.
            - Ensures idempotent writes.

        Args:
            tick_id (int): Current tick ID
            readiness (Dict[int, bool]): Mapping of zone ID to readiness status from _finalize_tick
        """
        raise NotImplementedError()

    @abstractmethod
    def _collect_tick_metrics(self, tick_id: int, readiness: Dict[int, bool], tick_elapsed: float) -> TickMetrics:
        """

        Args:
            tick_id (int): Current tick ID
            readiness (Dict[int, bool]): Mapping of zone ID to readiness status from _finalize_tick
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
        counter_refs = {zone_id: actor.get_counters.remote() for zone_id, actor in self.actors.items()}
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
        self._initialize_runtime()

        logger.info(f"\n{self.mode_name} Replay")

        # Steps D-G - Run each tick
        for tick_id in self.tick_ids:
            tick_start = time.time()
            logger.info(f"[{self.mode_name.lower()}] tick {tick_id}/{self.max_ticks - 1}")

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
