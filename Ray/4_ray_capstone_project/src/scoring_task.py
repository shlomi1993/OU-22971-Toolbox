"""
Distributed per-zone scoring task logic implementation.

Ray remote task that scores a ZoneSnapshot against its baseline and returns a ScoringResult.
"""

import time
import numpy as np
import ray

from ray.actor import ActorHandle
from src.core import DemandVerdict, ScoringResult, ZoneSnapshot


@ray.remote
def score_zone(snapshot: ZoneSnapshot, slow_sleep_s: float = 0.0, actor_handle: ActorHandle = None) -> ScoringResult:
    """
    Per-zone scoring task.

    Args:
        snapshot (ZoneSnapshot): Input snapshot for the zone and tick to score.
        slow_sleep_s (float): Artificial delay in seconds to simulate skew.
        actor_handle (ActorHandle, optional): Actor handle to report the decision back to. If None, no report is made.

    Returns:
        ScoringResult: Decision payload with zone_id, tick_id, decision, task_latency_s
    """
    start = time.time()

    # Simulate skew
    if slow_sleep_s > 0:
        time.sleep(slow_sleep_s)

    # Scoring logic
    if snapshot.recent_demand:
        recent_avg = np.mean(snapshot.recent_demand)
        threshold = snapshot.baseline_mean + max(snapshot.baseline_std, 1.0)
        decision = DemandVerdict.NEED if recent_avg > threshold else DemandVerdict.OK
    else:
        decision = DemandVerdict.OK

    # Compute latency
    latency = time.time() - start

    # Report to actor if provided
    if actor_handle is not None:
        ray.get(actor_handle.report_decision.remote(snapshot.tick_id, decision, latency))

    return ScoringResult(snapshot.zone_id, snapshot.tick_id, decision, latency)
