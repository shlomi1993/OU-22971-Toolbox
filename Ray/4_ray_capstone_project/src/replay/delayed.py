"""
Stretch A - Delayed-arrival replay implementation.

Extends async replay so a fraction of zone demand is withheld for several ticks before release. The scoring logic sees
incomplete data until the withheld demand appears, which can flip later recommendations from OK to NEED.
"""

import ray

from typing import Any, Dict, List

from src.core import write_json
from src.replay.asynchronous import AsyncReplay
from src.zone_actor import ZoneSnapshot


class DelayedReplay(AsyncReplay):
    """
    Async replay with delayed demand arrivals (Stretch A).
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tick_release_log: List[Dict] = []  # aggregate release/withhold info per tick

    @property
    def mode_name(self) -> str:
        return "Delayed"

    def _advance_replay_tick(self, tick_id: int) -> Dict[int, ZoneSnapshot]:
        """
        Step D override - use delay-aware activation and snapshots

        Activate tick with delayed-arrival semantics and collect delayed
        snapshots from each actor.

        Returns:
            Dict[int, ZoneSnapshot]: Per-zone snapshots using visible demand only.
        """
        # 1. Activate with delayed logic (releases pending, may withhold current)
        info_refs = {
            zone_id: actor.activate_tick_delayed.remote(
                tick_id,
                self.config.delay_ticks,
                self.config.delayed_fraction,
                self.config.seed,
            )
            for zone_id, actor in self.actors.items()
        }
        tick_info = {zone_id: ray.get(ref) for zone_id, ref in info_refs.items()}

        # Log released and withheld events
        for zone_id, info in tick_info.items():
            if info.get("released"):
                self.logger.info(f"[delayed] tick {tick_id}: zone {zone_id} released {len(info['released'])} delayed demand entries")
            if info.get("withheld"):
                self.logger.info(f"[delayed] tick {tick_id}: zone {zone_id} demand withheld (release at tick {info['release_tick']})")
        self.tick_release_log.append({"tick_id": tick_id, "zones": tick_info})

        # 2. Get delayed-aware snapshots (visible demand only)
        snap_refs = {zone_id: actor.get_snapshot_delayed.remote(tick_id) for zone_id, actor in self.actors.items()}
        return {zone_id: ray.get(ref) for zone_id, ref in snap_refs.items()}

    def _close_tick(self, tick_id: int, readiness: Dict[int, bool]) -> None:
        """
        Step G override - use delay-aware finalization

        Close tick using delay-aware finalization so that visible_demand is updated correctly.
        Demand is only appended when not withheld.
        """
        for zone_id, actor in self.actors.items():
            ray.get(actor.finalize_tick_delayed.remote(tick_id, self.config.fallback_policy))

        # Collect accepted decisions from actors
        tick_decisions = {}
        for zone_id, actor in self.actors.items():
            decisions = ray.get(actor.get_accepted_decisions.remote())
            if tick_id in decisions:
                tick_decisions[zone_id] = decisions[tick_id]
        self.all_decisions[tick_id] = tick_decisions

        n_fallback = sum(1 for ready in readiness.values() if not ready)
        if n_fallback > 0:
            self.logger.info(f"[delayed] tick {tick_id}: {n_fallback} zones used fallback policy")

    def _write_artifacts(self) -> None:
        """
        Artifact override - include delay log
        """
        super()._write_artifacts()

        mode_dir = self.output_dir / self.mode_name.lower()
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Collect per-actor delay logs
        log_refs = {zone_id: actor.get_delay_log.remote() for zone_id, actor in self.actors.items()}
        all_delay_events = []
        for ref in log_refs.values():
            all_delay_events.extend(ray.get(ref))

        write_json(all_delay_events, mode_dir / "delay_log.json")
        self.logger.info(f"[delayed] wrote {len(all_delay_events)} delay events to delay_log.json")
