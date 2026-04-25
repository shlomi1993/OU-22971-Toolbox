"""
Stretch B - Zone sub-actor for adaptive load balancing.

A lightweight helper actor created when a zone is identified as a repeat
straggler.  The sub-actor takes over scoring for the zone so that it runs
on a dedicated actor with no artificial skew delay, reducing latency for
chronically slow zones while preserving the same decision semantics.

Work that moves to the sub-actor:
- Snapshot scoring (compute_decision)
- Reporting the decision to the parent ZoneActor

Work that stays on the parent ZoneActor:
- State ownership (recent demand, accepted decisions)
- Tick activation and finalization
- Idempotent write enforcement
"""

import time
import ray

from typing import Dict

from ray.actor import ActorHandle

from src.zone_actor import ZoneRecommendation, ZoneSnapshot


@ray.remote
class ZoneSubActor:
    """
    Dedicated scoring helper for a hot zone.

    Created by the driver when a zone exceeds the straggler trigger count.
    Runs scoring without artificial sleep, effectively bypassing the skew that caused the zone to be a repeat straggler.
    """

    def __init__(self, zone_id: int, parent_handle: ActorHandle) -> None:
        """
        Args:
            zone_id (int): The zone this sub-actor is helping.
            parent_handle (ActorHandle): Handle to the parent ZoneActor for
                reporting decisions.
        """
        self.zone_id = zone_id
        self.parent = parent_handle
        self.n_scored = 0

    def score_and_report(self, snapshot: ZoneSnapshot) -> ZoneRecommendation:
        """
        Score a snapshot and report the decision to the parent ZoneActor.

        No artificial delay is applied - the sub-actor is a dedicated resource
        for this zone, so it does not share the skew that affected the zone in
        the normal scoring pipeline.

        Args:
            snapshot (ZoneSnapshot): Snapshot from the parent actor.

        Returns:
            ZoneRecommendation: Decision payload.
        """
        start = time.time()
        decision = snapshot.compute_decision()
        latency = time.time() - start

        # Report to parent actor (same path as async scoring tasks)
        ray.get(self.parent.report_decision.remote(snapshot.tick_id, decision, latency))

        self.n_scored += 1
        return ZoneRecommendation(self.zone_id, snapshot.tick_id, decision, latency)

    def get_stats(self) -> Dict:
        """
        Return sub-actor statistics.

        Returns:
            Dict: zone_id and number of ticks scored by this sub-actor.
        """
        return {"zone_id": self.zone_id, "n_scored": self.n_scored}
