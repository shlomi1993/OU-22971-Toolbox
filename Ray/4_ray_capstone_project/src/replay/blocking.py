"""
Blocking replay implementation.

In blocking mode, scoring tasks return their decisions to the controller, and the controller waits for all zones before
advancing each tick. This is the simplest execution mode with straightforward coordination and state management, but is
sensitive to skew since the tick cannot advance until all zones have reported in.
"""

import time
import numpy as np
import ray

from typing import Dict
from src.replay.base import Replay
from src.tlc import TickMetrics
from src.zone_actor import ZoneRecommendation, ZoneSnapshot


@ray.remote
def score_zone_blocking(snapshot: ZoneSnapshot, slow_sleep_s: float = 0.0) -> ZoneRecommendation:
    """
    Per-zone scoring task for blocking mode.

    Deterministic from snapshot input. Returns the decision payload to the controller.

    Args:
        snapshot (ZoneSnapshot): Snapshot object from ZoneActor.get_snapshot()
        slow_sleep_s (float): Artificial delay in seconds to simulate skew

    Returns:
        ZoneRecommendation: Decision payload with zone_id, tick_id, decision, task_latency_s
    """
    start = time.time()

    # Simulate skew
    if slow_sleep_s > 0:
        time.sleep(slow_sleep_s)

    decision = snapshot.compute_decision()
    latency = time.time() - start

    return ZoneRecommendation(snapshot.zone_id, snapshot.tick_id, decision, latency)


class BlockingReplay(Replay):
    """
    Blocking baseline replay execution.

    For each tick:
    - Collect snapshots from all zones
    - Launch scoring tasks for all zones
    - Wait for ALL results to complete
    - Write accepted decisions into actors
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize blocking replay.

        Args:
            prepared_dir (Path): Directory containing prepared assets from prepare.py
            output_dir (Path): Root output directory for artifacts
            config (RunConfig): Runtime configuration for the replay
        """
        super().__init__(*args, **kwargs)
        self.current_tick_results = {}

    @property
    def mode_name(self) -> str:
        """
        Get the display name for blocking mode.

        Returns:
            str: "Blocking"
        """
        return "Blocking"

    def _run_scoring(self, tick_id: int, snapshots: Dict[int, ZoneSnapshot]) -> None:
        """
        Step E - Run per-zone scoring (blocking mode).

        - Submit each zone snapshot to a scoring task
        - Collect all task returns in the controller for the current tick

        Args:
            tick_id (int): Current tick ID
            snapshots (Dict[int, ZoneSnapshot]): Mapping of zone_id to snapshot
        """
        task_refs = {}
        for zone_id, snap in snapshots.items():
            sleep_s = self.config.slow_zone_sleep_s if zone_id in self.slow_zones else 0.0
            task_refs[zone_id] = score_zone_blocking.remote(snap, slow_sleep_s=sleep_s)

        # Wait for all results
        self.current_tick_results = {zone_id: ray.get(ref) for zone_id, ref in task_refs.items()}

    def _finalize_tick(self, tick_id: int) -> Dict[int, bool]:
        """
        Step F - Finalize the tick under partial readiness (blocking mode).

        - Wait until all task results for the current tick have been returned to the controller
        - Close the tick only after the controller has a complete result set

        All zones are ready by definition in blocking mode since we wait for all tasks.

        Args:
            tick_id (int): Current tick ID

        Returns:
            Dict[int, bool]: Mapping of zone_id to True (all zones are ready in blocking mode)
        """
        # In blocking mode, all zones are ready by definition
        return {zone_id: True for zone_id in self.actors.keys()}

    def _close_tick(self, tick_id: int, readiness: Dict[int, bool]) -> None:
        """
        Step G - Close the tick in each actor (blocking mode).

        - Have the controller write the accepted decision for each zone into its actor
        - Ensure duplicate accepted writes for the same zone and tick are safe to replay

        In blocking mode:
        - Update actor state needed for the next tick

        Args:
            tick_id (int): Current tick ID
            readiness (Dict[int, bool]): Mapping of zone_id to readiness status (all True in blocking mode)
        """
        tick_decisions = {}
        for zone_id, res in self.current_tick_results.items():
            # Write decision to actor
            ray.get(self.actors[zone_id].write_decision.remote(tick_id, res.decision))
            tick_decisions[zone_id] = res.decision

        self.all_decisions[tick_id] = tick_decisions

    def _collect_tick_metrics(self, tick_id: int, readiness: Dict[int, bool],
                             tick_elapsed: float) -> TickMetrics:
        """
        Collect metrics for the completed tick in blocking mode.

        Args:
            tick_id (int): Current tick ID
            readiness (Dict[int, bool]): Mapping of zone_id to readiness status (all True in blocking mode)
            tick_elapsed (float): Total time elapsed for this tick in seconds

        Returns:
            TickMetrics: TickMetrics object for this tick
        """
        latencies = {zone_id: res.task_latency_s for zone_id, res in self.current_tick_results.items()}
        lat_values = list(latencies.values())

        return TickMetrics(
            tick_id=tick_id,
            mode=self.mode_name.lower(),
            n_zones_completed=len(self.current_tick_results),
            n_zones_fallback=0,  # No fallbacks in blocking mode
            mean_zone_latency_s=float(np.mean(lat_values)) if lat_values else 0.0,
            max_zone_latency_s=float(np.max(lat_values)) if lat_values else 0.0,
            max_mean_ratio=(float(np.max(lat_values)) / float(np.mean(lat_values)))
                          if lat_values and np.mean(lat_values) > 0 else 0.0,
            total_tick_latency_s=tick_elapsed,
            per_zone_latency=latencies,
        )
