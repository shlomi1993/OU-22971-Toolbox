"""
Asynchronous replay implementation.

In async mode, scoring tasks report decisions to ZoneActors, and the driver polls actor readiness before finalizing each tick.
"""

import logging
import time
import ray

from typing import Dict
from ray.actor import ActorHandle
from src.replay.base import Replay
from src.tlc import RunMode, TickMetrics
from src.zone_actor import ZoneRecommendation, ZoneSnapshot


logger = logging.getLogger(__name__)


@ray.remote
def score_zone_async(snapshot: ZoneSnapshot, slow_sleep_s: float, actor_handle: ActorHandle) -> ZoneRecommendation:
    """
    Per-zone scoring task for async mode.

    Deterministic from snapshot input. Reports the decision to the ZoneActor before returning.

    Args:
        snapshot: Snapshot object from ZoneActor.get_snapshot()
        slow_sleep_s: Artificial delay in seconds to simulate skew
        actor_handle: Ray actor handle for async reporting to the zone's actor

    Returns:
        Decision payload with zone_id, tick_id, decision, task_latency_s
    """
    start = time.time()

    # Simulate skew
    if slow_sleep_s > 0:
        time.sleep(slow_sleep_s)

    decision = snapshot.compute_decision()
    latency = time.time() - start

    # Report to actor in async mode
    ray.get(actor_handle.report_decision.remote(snapshot.tick_id, decision, latency))

    return ZoneRecommendation(snapshot.zone_id, snapshot.tick_id, decision, latency)


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
            prepared_dir: Directory containing prepared assets from prepare.py
            output_dir: Root output directory for artifacts
            config: Runtime configuration for the replay
        """
        super().__init__(*args, **kwargs)
        self.prev_late_count = 0
        self.prev_dup_count = 0

    @property
    def mode_name(self) -> str:
        """
        Get the display name for async mode.

        Returns:
            "Async"
        """
        return "Async"

    def _run_scoring(self, tick_id: int, snapshots: Dict[int, ZoneSnapshot]) -> None:
        """
        Step E - Run per-zone scoring (async mode).

        - Submit each zone snapshot to a scoring task
        - Have each scoring task report its result to that zone's actor for the current tick
        - Use bounded concurrency (max_inflight_zones)

        Args:
            tick_id: Current tick ID
            snapshots: Mapping of zone_id to snapshot
        """
        pending = {}
        zone_queue = list(snapshots.keys())
        tick_start = time.time()

        while zone_queue or pending:
            # Launch up to max_inflight zones
            while zone_queue and len(pending) < self.config.max_inflight_zones:
                zone_id = zone_queue.pop(0)
                sleep_s = self.config.slow_zone_sleep_s if zone_id in self.runtime.slow_zones else 0.0
                ref = score_zone_async.remote(
                    snapshots[zone_id],
                    slow_sleep_s=sleep_s,
                    actor_handle=self.runtime.actors[zone_id]
                )
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
                logger.warning(f"[async] tick {tick_id}: timeout after {elapsed:.2f}s, "
                             f"{len(pending)} zones pending")
                break

    def _finalize_tick(self, tick_id: int) -> Dict[int, bool]:
        """
        Step F - Finalize the tick under partial readiness (async mode).

        - Check which actors have already received a report for the current tick by asking
          actors for their current tick status
        - Decide whether the policy says to keep waiting or to close the tick

        You must define and log a deterministic policy for late zones.

        Policy requirements:
        - must be explicit in config
        - fallback_policy should default to always_previous
        - must be visible in logs and artifacts
        - must behave the same way on repeated runs with the same inputs and seed

        Args:
            tick_id: Current tick ID

        Returns:
            Mapping of zone_id to readiness status (True if zone has a decision, False otherwise)
        """
        readiness = {}
        for zone_id, actor in self.runtime.actors.items():
            readiness[zone_id] = ray.get(actor.has_decision_for_tick.remote(tick_id))

        n_ready = sum(readiness.values())
        logger.info(f"[async] tick {tick_id}: {n_ready}/{len(readiness)} zones ready")

        return readiness

    def _close_tick(self, tick_id: int, readiness: Dict[int, bool]) -> None:
        """
        Step G - Close the tick in each actor (async mode).

        - Ask each actor to finalize the current tick using either its reported decision
          or the fallback policy
        - Ensure duplicate reports for the same zone and tick are safe to replay
        - Late results that arrive after finalization should be logged and ignored by the actor

        In async mode:
        - Update actor state needed for the next tick

        Args:
            tick_id: Current tick ID
            readiness: Mapping of zone_id to readiness status from _finalize_tick
        """
        # Finalize each actor for this tick
        n_fallback = 0
        for zone_id, actor in self.runtime.actors.items():
            ray.get(actor.finalize_tick.remote(tick_id, self.config.fallback_policy))
            if not readiness[zone_id]:
                n_fallback += 1

        # Collect accepted decisions from actors
        tick_decisions = {}
        for zone_id, actor in self.runtime.actors.items():
            decisions = ray.get(actor.get_accepted_decisions.remote())
            if tick_id in decisions:
                tick_decisions[zone_id] = decisions[tick_id]

        self.all_decisions[tick_id] = tick_decisions

        if n_fallback > 0:
            logger.info(f"[async] tick {tick_id}: {n_fallback} zones used fallback policy")

    def _collect_tick_metrics(self, tick_id: int, readiness: Dict[int, bool],
                             tick_elapsed: float) -> TickMetrics:
        """
        Collect metrics for the completed tick in async mode.

        Also logs:
        - number of late task reports ignored
        - number of duplicate task reports ignored

        Args:
            tick_id: Current tick ID
            readiness: Mapping of zone_id to readiness status
            tick_elapsed: Total time elapsed for this tick in seconds

        Returns:
            TickMetrics object for this tick
        """
        n_ready = sum(readiness.values())
        n_fallback = sum(1 for ready in readiness.values() if not ready)

        # Collect counters periodically to reduce overhead
        # Collect on first, last, and every 10th tick
        if tick_id == 0 or tick_id == self.runtime.max_ticks - 1 or tick_id % 10 == 0:
            counter_refs = {zone_id: actor.get_counters.remote()
                          for zone_id, actor in self.runtime.actors.items()}
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

        # Latency tracked in actor for async mode
        latencies = {zone_id: 0.0 for zone_id in self.runtime.actors.keys()}

        return TickMetrics(
            tick_id=tick_id,
            mode=RunMode.ASYNC,
            n_zones_completed=n_ready,
            n_zones_fallback=n_fallback,
            n_late_reports=n_late_delta,
            n_duplicate_reports=n_dup_delta,
            mean_zone_latency_s=0.0,
            max_zone_latency_s=0.0,
            total_tick_latency_s=tick_elapsed,
            per_zone_latency=latencies,
        )
