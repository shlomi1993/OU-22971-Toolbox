"""
Distributed per-zone scoring task logic implementation
"""

import time
import ray

from typing import Optional
from ray.actor import ActorHandle
from src.zone_actor import ZoneRecommendation, ZoneSnapshot


@ray.remote
def score_zone(snapshot: ZoneSnapshot, slow_sleep_s: float = 0.0, actor_handle: Optional[ActorHandle] = None) -> ZoneRecommendation:
    """
    Per-zone scoring task.


    Args:
        snapshot (ZoneSnapshot): Snapshot object from ZoneActor.get_snapshot()
        slow_sleep_s (float): Artificial delay in seconds to simulate skew
        actor_handle (Optional[ActorHandle]): Ray actor handle for reporting (if provided)

    Returns:
        ZoneRecommendation: Decision payload with zone_id, tick_id, decision, task_latency_s
    """
    start = time.time()

    # Simulate skew
    if slow_sleep_s > 0:
        time.sleep(slow_sleep_s)

    decision = snapshot.compute_decision()
    latency = time.time() - start

    # Report to actor if provided
    if actor_handle is not None:
        ray.get(actor_handle.report_decision.remote(snapshot.tick_id, decision, latency))

    return ZoneRecommendation(snapshot.zone_id, snapshot.tick_id, decision, latency)
