"""
Stretch B - Adaptive load-balancing replay with zone sub-actors.

Extends async replay to detect repeat straggler zones and spawn dedicated ZoneSubActor helpers for them.
"""

import time
import numpy as np
import ray

from typing import Dict, List

from src.core import TickMetrics, write_json
from src.replay.asynchronous import AsyncReplay, score_zone_async
from src.zone_actor import ZoneRecommendation, ZoneSnapshot
from src.zone_subactor import ZoneSubActor


class SubActorReplay(AsyncReplay):
    """
    Async replay with adaptive sub-actor creation for hot zones (Stretch B).
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.zone_slow_counts: Dict[int, int] = {}  # Per-zone slow-tick counts (observed by the driver)
        self.subactors: Dict[int, ray.actor.ActorHandle] = {}  # Active sub-actors: zone_id -> ActorHandle
        self.repeat_straggler_zones: set = set()  # Set of repeat-straggler zone IDs (always slow)
        self.subactor_creation_log: List[Dict] = []  # Log of sub-actor creation events
        self.current_tick_latencies: Dict[int, float] = {}  # Per-tick latency observations for metrics

    @property
    def mode_name(self) -> str:
        return "SubActor"

    def _initialize_runtime(self) -> None:
        """
        Step C - Runtime initialization override - pick repeat stragglers
        """
        super()._initialize_runtime()
        self.zone_slow_counts = {z: 0 for z in self.actors}
        self.subactors = {}
        self.subactor_creation_log = []

        # Select repeat straggler zones (deterministic, always slow)
        rng = np.random.RandomState(self.config.seed + 7)
        n_repeat = max(1, int(len(self.actors) * self.config.repeat_straggler_fraction))
        all_zones = sorted(self.actors.keys())
        self.repeat_straggler_zones = set(rng.choice(all_zones, size=n_repeat, replace=False))
        n_stragglers, sorted_ids = len(self.repeat_straggler_zones), sorted(self.repeat_straggler_zones)
        self.logger.info(f"[subactor] repeat straggler zones ({n_stragglers}): {sorted_ids}")

    def _sleep_for_zone(self, zone_id: int) -> float:
        """
        Return the artificial sleep for a zone in this mode.

        Repeat stragglers get 2x the configured slow sleep. Regular slow zones get normal slow sleep. Zones with
        sub-actors get ZERO sleep.
        """
        if zone_id in self.subactors:
            return 0.0  # sub-actor handles this zone
        if zone_id in self.repeat_straggler_zones:
            return self.config.slow_zone_sleep_s * 2.0
        if zone_id in self.slow_zones:
            return self.config.slow_zone_sleep_s
        return 0.0

    def _maybe_create_subactor(self, zone_id: int, tick_id: int) -> None:
        """
        If zone has exceeded the straggler trigger count, create a sub-actor.
        """
        if zone_id in self.subactors:
            return
        if self.zone_slow_counts.get(zone_id, 0) >= self.config.straggler_trigger_count:
            subactor = ZoneSubActor.remote(zone_id, self.actors[zone_id])
            self.subactors[zone_id] = subactor
            event = {
                "zone_id": zone_id,
                "created_at_tick": tick_id,
                "slow_count": self.zone_slow_counts[zone_id],
            }
            self.subactor_creation_log.append(event)
            self.logger.info(
                f"[subactor] created sub-actor for zone {zone_id} at tick {tick_id} "
                f"(slow {self.zone_slow_counts[zone_id]} times)"
            )

    def _run_scoring(self, tick_id: int, snapshots: Dict[int, ZoneSnapshot]) -> None:
        """
        Step E override - route hot zones through sub-actors

        Score zones.  Zones with sub-actors are scored through their
        dedicated sub-actor (no artificial delay).  Other zones use the
        standard async scoring path.
        """
        pending = {}
        zone_queue = list(snapshots.keys())
        tick_start = time.time()
        self.current_tick_latencies = {}

        while zone_queue or pending:
            while zone_queue and len(pending) < self.config.max_inflight_zones:
                zone_id = zone_queue.pop(0)

                if zone_id in self.subactors:
                    # Route through sub-actor (no skew delay)
                    ref = self.subactors[zone_id].score_and_report.remote(snapshots[zone_id])
                else:
                    sleep_s = self._sleep_for_zone(zone_id)
                    ref = score_zone_async.remote(snapshots[zone_id], sleep_s, self.actors[zone_id])
                pending[ref] = zone_id

            if not pending:
                break

            ready, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=self.config.tick_timeout_s)
            for ref in ready:
                result: ZoneRecommendation = ray.get(ref)
                z = pending.pop(ref)
                self.current_tick_latencies[z] = result.task_latency_s

            elapsed = time.time() - tick_start
            if elapsed >= self.config.tick_timeout_s and pending:
                self.logger.warning(f"[subactor] tick {tick_id}: timeout after {elapsed:.2f}s, " f"{len(pending)} zones pending")
                break

    def _close_tick(self, tick_id: int, readiness: Dict[int, bool]) -> None:
        """
        After each tick - update straggler counts & maybe create sub-actors

        Close the tick, then update straggler detection state and
        potentially create sub-actors for zones that crossed the threshold.
        """
        # Regular async close
        super()._close_tick(tick_id, readiness)

        # Update slow-tick counts from observed latencies
        slow_threshold = self.config.slow_zone_sleep_s * 0.5
        for zone_id, lat in self.current_tick_latencies.items():
            if lat > slow_threshold:
                self.zone_slow_counts[zone_id] = self.zone_slow_counts.get(zone_id, 0) + 1
            # Also record on the actor for visibility
            ray.get(self.actors[zone_id].record_tick_latency.remote(lat, slow_threshold))

        # Create sub-actors for zones that now exceed the trigger count
        for zone_id in list(self.actors.keys()):
            self._maybe_create_subactor(zone_id, tick_id)

    def _collect_tick_metrics(self, tick_id: int, readiness: Dict[int, bool], tick_elapsed: float) -> TickMetrics:
        metrics = super()._collect_tick_metrics(tick_id, readiness, tick_elapsed)
        # Override per-zone latency with actual observations
        if self.current_tick_latencies:
            metrics.per_zone_latency = dict(self.current_tick_latencies)
            lat_vals = list(self.current_tick_latencies.values())
            metrics.mean_zone_latency_s = float(np.mean(lat_vals)) if lat_vals else 0.0
            metrics.max_zone_latency_s = float(np.max(lat_vals)) if lat_vals else 0.0
            if metrics.mean_zone_latency_s > 0:
                metrics.max_mean_ratio = metrics.max_zone_latency_s / metrics.mean_zone_latency_s
        return metrics

    def _write_artifacts(self) -> None:
        """
        Artifact override - include sub-actor log
        """
        super()._write_artifacts()

        mode_dir = self.output_dir / self.mode_name.lower()
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Collect sub-actor stats
        subactor_stats = []
        for sa in self.subactors.values():
            stats = ray.get(sa.get_stats.remote())
            subactor_stats.append(stats)

        summary = {
            "repeat_straggler_zones": sorted(self.repeat_straggler_zones),
            "subactors_created": len(self.subactors),
            "creation_events": self.subactor_creation_log,
            "subactor_stats": subactor_stats,
            "zone_slow_counts": {str(z): c for z, c in sorted(self.zone_slow_counts.items())},
        }
        write_json(summary, mode_dir / "subactor_log.json")
        self.logger.info(f"[subactor] wrote sub-actor log ({len(self.subactors)} sub-actors created)")
