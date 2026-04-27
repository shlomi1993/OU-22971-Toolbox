"""
Asynchronous replay implementation.

In async mode, scoring tasks report decisions to ZoneActors, and the driver polls actor readiness before finalizing each
tick. This allows for partial tick completion and more flexible policies for handling late zones, but requires more
complex coordination and state management.
"""

import time
import numpy as np
import ray

from typing import Dict
from src.common import TickMetrics, ZoneSnapshot
from src.logger import g_logger
from src.replay.base import Replay
from src.scoring_task import score_zone


class AsyncReplay(Replay):
    """
    Asynchronous replay execution.

    For each tick:
    - Collect snapshots from all zones
    - Launch scoring tasks with bounded concurrency
    - Tasks report decisions to their ZoneActors
    - Poll actor readiness and finalize tick under partial-readiness policy
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize async replay.

        Args:
            prepared_dir (Path): Directory containing prepared assets from prepare.py
            output_dir (Path): Root output directory for artifacts
            config (ReplayConfig): Runtime configuration for the replay
        """
        super().__init__(*args, **kwargs)
        self.prev_late_count = 0
        self.prev_dup_count = 0

    def _initialize_runtime(self) -> None:
        """
        Initialize runtime and reset async-specific counters.
        """
        super()._initialize_runtime()
        self.prev_late_count = 0
        self.prev_dup_count = 0

    @property
    def mode_name(self) -> str:
        """
        Get the display name for async mode.

        Returns:
            str: "Async"
        """
        return "Async"

    def _run_scoring(self, tick_id: int, snapshots: Dict[int, ZoneSnapshot]) -> None:
        """
        Step E - Run per-zone scoring (async mode):
        - Submit each zone snapshot to a scoring task
        - Have each scoring task report its result to that zone's actor for the current tick

        Args:
            tick_id (int): Current tick ID
            snapshots (Dict[int, ZoneSnapshot]): Mapping of zone_id to snapshot
        """
        pending = {}
        zone_queue = list(snapshots.keys())
        tick_start = time.time()

        while zone_queue or pending:
            # Launch up to max_inflight zones
            while zone_queue and len(pending) < self.config.max_inflight_zones:
                zone_id = zone_queue.pop(0)
                sleep_s = self.config.slow_zone_sleep_s if zone_id in self.slow_zones else 0.0
                ref = score_zone.remote(snapshots[zone_id], slow_sleep_s=sleep_s, actor_handle=self.actors[zone_id])
                pending[ref] = zone_id

            if not pending:
                break

            # Wait for at least one task to complete
            ready, not_ready = ray.wait(list(pending.keys()), num_returns=1,
                                       timeout=self.config.tick_timeout_s)
            for ref in ready:
                ray.get(ref)
                del pending[ref]

            # Check if we've hit the timeout with remaining pending tasks
            elapsed = time.time() - tick_start
            if elapsed >= self.config.tick_timeout_s and pending:
                g_logger.warning(f"[async] tick {tick_id}: timeout after {elapsed:.2f}s, {len(pending)} zones pending")
                break

    def _finalize_tick(self, tick_id: int) -> Dict[int, bool]:
        """
        Step F - Finalize the tick under partial readiness (async mode):
        - Check which actors have already received a report for the current tick by asking actors for their current tick status
        - Decide whether the policy says to keep waiting or to close the tick

        Args:
            tick_id (int): Current tick ID

        Returns:
            Dict[int, bool]: Mapping of zone_id to readiness status (True if zone has a decision, False otherwise)
        """
        readiness = {}
        for zone_id, actor in self.actors.items():
            readiness[zone_id] = ray.get(actor.has_decision_for_tick.remote(tick_id))

        n_ready = sum(readiness.values())
        g_logger.info(f"[async] tick {tick_id}: {n_ready}/{len(readiness)} zones ready")

        return readiness

    def _close_tick(self, tick_id: int, readiness: Dict[int, bool]) -> None:
        """
        Step G - Close the tick in each actor (async mode):
        - Ask each actor to finalize the current tick using either its reported decision or the fallback policy
        - Ensure duplicate reports for the same zone and tick are safe to replay
        - Late results that arrive after finalization should be logged and ignored by the actor
        - Update actor state needed for the next tick

        Args:
            tick_id (int): Current tick ID
            readiness (Dict[int, bool]): Mapping of zone_id to readiness status from _finalize_tick
        """
        # Finalize each actor for this tick
        n_fallback = 0
        for zone_id, actor in self.actors.items():
            ray.get(actor.finalize_tick.remote(tick_id, self.config.fallback_policy))
            if not readiness[zone_id]:
                n_fallback += 1

        # Collect accepted decisions from actors
        tick_decisions = {}
        for zone_id, actor in self.actors.items():
            decisions = ray.get(actor.get_accepted_decisions.remote())
            if tick_id in decisions:
                tick_decisions[zone_id] = decisions[tick_id]

        self.all_decisions[tick_id] = tick_decisions

        if n_fallback > 0:
            g_logger.info(f"[async] tick {tick_id}: {n_fallback} zones used fallback policy")

    def _collect_tick_metrics(self, tick_id: int, readiness: Dict[int, bool],
                             tick_elapsed: float) -> TickMetrics:
        """
        Collect metrics for the completed tick in async mode.

        Also logs:
        - number of late task reports ignored
        - number of duplicate task reports ignored

        Args:
            tick_id (int): Current tick ID
            readiness (Dict[int, bool]): Mapping of zone_id to readiness status
            tick_elapsed (float): Total time elapsed for this tick in seconds

        Returns:
            TickMetrics: TickMetrics object for this tick
        """
        n_ready = sum(readiness.values())
        n_fallback = sum(1 for ready in readiness.values() if not ready)

        # Collect counters periodically to reduce overhead
        # Collect on first, last, and every 10th tick
        if tick_id == 0 or tick_id == self.max_ticks - 1 or tick_id % 10 == 0:
            counter_refs = {zone_id: actor.get_counters.remote()
                          for zone_id, actor in self.actors.items()}
            counters = {zone_id: ray.get(ref) for zone_id, ref in counter_refs.items()}
            current_late = sum(c.n_late for c in counters.values())
            current_dup = sum(c.n_duplicates for c in counters.values())
            n_late_delta = current_late - self.prev_late_count
            n_dup_delta = current_dup - self.prev_dup_count
            self.prev_late_count = current_late
            self.prev_dup_count = current_dup
        else:
            # Use zeros for intermediate ticks - counters will be accurate on collection ticks
            n_late_delta = 0
            n_dup_delta = 0

        # Collect per-zone latency from actors for zones that reported in
        latency_refs = {zone_id: actor.get_reported_latency.remote(tick_id) for zone_id, actor in self.actors.items()}
        latencies = {zone_id: ray.get(ref) for zone_id, ref in latency_refs.items()}
        reported_latencies = [v for v in latencies.values() if v > 0.0]
        mean_lat = float(np.mean(reported_latencies)) if reported_latencies else 0.0
        max_lat = float(np.max(reported_latencies)) if reported_latencies else 0.0

        return TickMetrics(
            tick_id=tick_id,
            mode=self.mode_name.lower(),
            n_zones_completed=n_ready,
            n_zones_fallback=n_fallback,
            n_late_reports=n_late_delta,
            n_duplicate_reports=n_dup_delta,
            mean_zone_latency_s=mean_lat,
            max_zone_latency_s=max_lat,
            total_tick_latency_s=tick_elapsed,
            per_zone_latency=latencies,
        )
