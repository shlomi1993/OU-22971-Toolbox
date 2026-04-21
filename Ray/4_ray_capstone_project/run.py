import argparse
import json
import logging
import time
import numpy as np
import pandas as pd
import ray

from pathlib import Path
from typing import Dict, List
from ray.actor import ActorHandle

from src.tlc import (
    RunConfig,
    RunMode,
    TickMetrics,
    get_tick_ids,
    load_prepared,
    select_slow_zones,
    write_json,
    write_latency_log,
    write_metrics_csv,
    write_tick_summary,
)
from src.zone_actor import ZoneActor, ZoneDecision, ZoneSnapshot


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@ray.remote
def score_zone(snapshot: ZoneSnapshot, slow_sleep_s: float = 0.0, actor_handle: ActorHandle = None,
               mode: RunMode = RunMode.BLOCKING) -> ZoneDecision:
    """
    Per-zone scoring task. Deterministic from snapshot input.
    In blocking mode, returns the decision payload to the controller.
    In async mode, reports the decision to the ZoneActor before returning.

    Args:
        snapshot (ZoneSnapshot): Snapshot object from ZoneActor.get_snapshot().
        slow_sleep_s (float): Artificial delay in seconds to simulate skew.
        actor_handle (ActorHandle, optional): Ray actor handle for async reporting. None in blocking mode. Defaults to None.
        mode (RunMode, optional): BLOCKING or ASYNC. Defaults to RunMode.BLOCKING.

    Returns:
        ZoneDecision: Decision payload with zone_id, tick_id, decision, task_latency_s.
    """
    start = time.time()

    # Simulate skew
    if slow_sleep_s > 0:
        time.sleep(slow_sleep_s)

    decision = snapshot.compute_decision()
    latency = time.time() - start

    if mode == RunMode.ASYNC and actor_handle is not None:
        ray.get(actor_handle.report_decision.remote(snapshot.tick_id, decision, latency))

    return ZoneDecision(snapshot.zone_id, snapshot.tick_id, decision, latency)


def create_actors(replay: pd.DataFrame, baseline: pd.DataFrame, active_zones: List[int], config: RunConfig) -> Dict[int, ActorHandle]:
    """
    Create one ZoneActor per active zone.

    Args:
        replay (pd.DataFrame): Full replay table with zone_id, tick_start, demand.
        baseline (pd.DataFrame): Full baseline table with zone_id, hour_of_day, day_of_week, mean_demand, std_demand.
        active_zones (List[int]): List of active zone IDs.
        config (RunConfig): Runtime configuration.

    Returns:
        Dict[int, ActorHandle]: Mapping of zone_id to Ray actor handle.
    """
    actors = {}
    for zone_id in active_zones:
        zone_replay = replay[replay["zone_id"] == zone_id].reset_index(drop=True)
        zone_baseline = baseline[baseline["zone_id"] == zone_id].reset_index(drop=True)
        actors[zone_id] = ZoneActor.remote(zone_id, zone_replay, zone_baseline, config)
    logger.info(f"Created {len(actors)} ZoneActors")
    return actors


def write_artifacts(output_dir: Path, config: RunConfig, tick_metrics: List[TickMetrics],
                    all_decisions: Dict[int, Dict[int, str]], actors: Dict[int, ActorHandle]) -> None:
    """
    Collect actor state and write all output artifacts.

    Args:
        output_dir (Path): Directory to write artifacts into.
        config (RunConfig): Runtime configuration (written as run_config.json).
        tick_metrics (List[TickMetrics]): Per-tick metrics collected during replay.
        all_decisions (Dict[int, Dict[int, str]]): Mapping of tick_id to {zone_id: decision}.
        actors (Dict[int, ActorHandle]): Mapping of zone_id to ZoneActor handles (for counter collection).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(config.to_dict(), output_dir / "run_config.json")
    write_metrics_csv(tick_metrics, output_dir / "metrics.csv")
    write_latency_log(tick_metrics, output_dir / "latency_log.json")
    write_tick_summary(tick_metrics, all_decisions, output_dir / "tick_summary.json")

    # Collect actor counters
    counter_refs = {zone_id: actor.get_counters.remote() for zone_id, actor in actors.items()}
    counters = {zone_id: ray.get(ref) for zone_id, ref in counter_refs.items()}
    write_json([c.to_dict() for c in counters.values()], output_dir / "actor_counters.json")

    logger.info(f"Artifacts written to {output_dir}")


def run_blocking(prepared_dir: Path, output_dir: Path, config: RunConfig) -> List[TickMetrics]:
    """
    Blocking baseline: for each tick, collect snapshots, launch scoring tasks, wait for ALL results, write accepted
    decisions into actors.

    Args:
        prepared_dir (Path): Directory with prepared assets from prepare.py.
        output_dir (Path): Root output directory (artifacts go into output_dir/blocking/).
        config (RunConfig): Runtime configuration.

    Returns:
        List[TickMetrics]: Per-tick metrics for the blocking run.
    """
    replay, baseline, active_zones = load_prepared(prepared_dir)
    actors = create_actors(replay, baseline, active_zones, config)
    slow_zones = select_slow_zones(active_zones, config)
    tick_ids = get_tick_ids(replay)

    # Limit ticks for manageable demo
    max_ticks = min(len(tick_ids), 50)
    tick_ids = tick_ids[:max_ticks]

    all_metrics = []
    all_decisions = {}

    for tick_id in tick_ids:
        tick_start = time.time()
        logger.info(f"[blocking] tick {tick_id}/{max_ticks - 1}")

        # Step D: activate tick and collect snapshots
        ray.get([actor.activate_tick.remote(tick_id) for actor in actors.values()])
        snapshot_refs = {zone_id: actor.get_snapshot.remote(tick_id) for zone_id, actor in actors.items()}
        snapshots = {zone_id: ray.get(ref) for zone_id, ref in snapshot_refs.items()}

        # Step E: launch scoring tasks and wait for all
        task_refs = {}
        for zone_id, snap in snapshots.items():
            sleep_s = config.slow_zone_sleep_s if zone_id in slow_zones else 0.0
            task_refs[zone_id] = score_zone.remote(snap, slow_sleep_s=sleep_s, mode=RunMode.BLOCKING)

        results = {zone_id: ray.get(ref) for zone_id, ref in task_refs.items()}

        # Step F+G: write accepted decisions (all complete in blocking mode)
        tick_decisions = {}
        latencies = {}
        for zone_id, res in results.items():
            ray.get(actors[zone_id].write_decision.remote(tick_id, res.decision))
            tick_decisions[zone_id] = res.decision
            latencies[zone_id] = res.task_latency_s

        all_decisions[tick_id] = tick_decisions
        tick_elapsed = time.time() - tick_start
        lat_values = list(latencies.values())

        metrics = TickMetrics(
            tick_id=tick_id,
            mode=RunMode.BLOCKING,
            n_zones_completed=len(results),
            n_zones_fallback=0,
            mean_zone_latency_s=float(np.mean(lat_values)) if lat_values else 0.0,
            max_zone_latency_s=float(np.max(lat_values)) if lat_values else 0.0,
            max_mean_ratio=(float(np.max(lat_values)) / float(np.mean(lat_values))) if lat_values and np.mean(lat_values) > 0 else 0.0,
            total_tick_latency_s=tick_elapsed,
            per_zone_latency=latencies,
        )
        all_metrics.append(metrics)

    write_artifacts(output_dir / "blocking", config, all_metrics, all_decisions, actors)
    return all_metrics


def run_async(prepared_dir: Path, output_dir: Path, config: RunConfig) -> List[TickMetrics]:
    """
    Async controller: scoring tasks report to actors, driver polls readiness,
    finalizes ticks under partial-readiness policy.

    Args:
        prepared_dir (Path): Directory with prepared assets from prepare.py.
        output_dir (Path): Root output directory (artifacts go into output_dir/async/).
        config (RunConfig): Runtime configuration.

    Returns:
        List[TickMetrics]: Per-tick metrics for the async run.
    """
    replay, baseline, active_zones = load_prepared(prepared_dir)
    actors = create_actors(replay, baseline, active_zones, config)
    slow_zones = select_slow_zones(active_zones, config)
    tick_ids = get_tick_ids(replay)

    max_ticks = min(len(tick_ids), 50)
    tick_ids = tick_ids[:max_ticks]

    all_metrics = []
    all_decisions = {}

    for tick_id in tick_ids:
        tick_start = time.time()
        logger.info(f"[async] tick {tick_id}/{max_ticks - 1}")

        # Step D: activate tick and collect snapshots
        ray.get([actor.activate_tick.remote(tick_id) for actor in actors.values()])
        snapshot_refs = {zone_id: actor.get_snapshot.remote(tick_id) for zone_id, actor in actors.items()}
        snapshots = {zone_id: ray.get(ref) for zone_id, ref in snapshot_refs.items()}

        # Step E: launch scoring tasks with bounded concurrency
        pending = {}  # ray_ref -> zone_id
        zone_queue = list(snapshots.keys())
        launched = 0

        while zone_queue or pending:
            # Launch up to max_inflight
            while zone_queue and len(pending) < config.max_inflight_zones:
                zone_id = zone_queue.pop(0)
                sleep_s = config.slow_zone_sleep_s if zone_id in slow_zones else 0.0
                ref = score_zone.remote(snapshots[zone_id], slow_sleep_s=sleep_s, actor_handle=actors[zone_id],
                                        mode=RunMode.ASYNC)
                pending[ref] = zone_id
                launched += 1

            if not pending:
                break

            # Wait for at least one task to finish
            ready, not_ready = ray.wait(list(pending.keys()), num_returns=1, timeout=config.tick_timeout_s)

            for ref in ready:
                ray.get(ref)  # Retrieve to catch exceptions
                del pending[ref]

            # Check if we've hit the timeout with remaining pending
            elapsed = time.time() - tick_start
            if elapsed >= config.tick_timeout_s and pending:
                logger.warning(f"[async] tick {tick_id}: timeout after {elapsed:.2f}s, {len(pending)} zones pending")
                break

        # Step F: check partial readiness via polling
        readiness = {}
        for zone_id, actor in actors.items():
            readiness[zone_id] = ray.get(actor.has_decision_for_tick.remote(tick_id))

        n_ready = sum(readiness.values())

        # Step G: finalize tick
        n_fallback = 0
        for zone_id, actor in actors.items():
            ray.get(actor.finalize_tick.remote(tick_id, config.fallback_policy))
            if not readiness[zone_id]:
                n_fallback += 1

        # Collect accepted decisions from actors
        tick_decisions = {}
        latencies = {}
        for zone_id, actor in actors.items():
            decisions = ray.get(actor.get_accepted_decisions.remote())
            if tick_id in decisions:
                tick_decisions[zone_id] = decisions[tick_id]
            latencies[zone_id] = 0.0  # Latency tracked in actor for async

        all_decisions[tick_id] = tick_decisions
        tick_elapsed = time.time() - tick_start

        # Collect counters for late/dup metrics
        counter_refs = {zone_id: actor.get_counters.remote() for zone_id, actor in actors.items()}
        counters = {zone_id: ray.get(ref) for zone_id, ref in counter_refs.items()}
        n_late = sum(c.n_late for c in counters.values())
        n_dup = sum(c.n_duplicates for c in counters.values())

        metrics = TickMetrics(
            tick_id=tick_id,
            mode=RunMode.ASYNC,
            n_zones_completed=n_ready,
            n_zones_fallback=n_fallback,
            n_late_reports=n_late,
            n_duplicate_reports=n_dup,
            mean_zone_latency_s=0.0,
            max_zone_latency_s=0.0,
            total_tick_latency_s=tick_elapsed,
            per_zone_latency=latencies,
        )
        all_metrics.append(metrics)

    write_artifacts(output_dir / "async", config, all_metrics, all_decisions, actors)
    return all_metrics


def run_stress(prepared_dir: Path, output_dir: Path, config: RunConfig) -> List[TickMetrics]:
    """
    Stress test: reuse blocking and async paths with harsher skew (60% slow zones, 3s delay).

    Args:
        prepared_dir (Path): Directory with prepared assets from prepare.py.
        output_dir (Path): Root output directory (artifacts go into output_dir/stress/).
        config (RunConfig): Base runtime configuration (skew fields are overridden).

    Returns:
        List[TickMetrics]: Per-tick metrics for the async stress run.
    """
    stress_config = RunConfig(
        n_zones=config.n_zones,
        tick_minutes=config.tick_minutes,
        max_inflight_zones=config.max_inflight_zones,
        tick_timeout_s=config.tick_timeout_s,
        completion_fraction=config.completion_fraction,
        slow_zone_fraction=0.6,  # 60% of zones are slow
        slow_zone_sleep_s=3.0,  # 3 seconds sleep
        fallback_policy=config.fallback_policy,
        seed=config.seed,
    )

    logger.info("=== STRESS: blocking baseline ===")
    blocking_metrics = run_blocking(prepared_dir, output_dir / "stress", stress_config)

    logger.info("=== STRESS: async controller ===")
    async_metrics = run_async(prepared_dir, output_dir / "stress", stress_config)

    # Write comparison summary
    comparison = {
        "blocking": {
            "mean_tick_latency": float(np.mean([m.total_tick_latency_s for m in blocking_metrics])),
            "max_tick_latency": float(np.max([m.total_tick_latency_s for m in blocking_metrics])),
            "total_fallbacks": sum(m.n_zones_fallback for m in blocking_metrics),
        },
        "async": {
            "mean_tick_latency": float(np.mean([m.total_tick_latency_s for m in async_metrics])),
            "max_tick_latency": float(np.max([m.total_tick_latency_s for m in async_metrics])),
            "total_fallbacks": sum(m.n_zones_fallback for m in async_metrics),
        },
    }
    write_json(comparison, output_dir / "stress" / "comparison.json")
    logger.info(f"Stress comparison: {json.dumps(comparison, indent=2)}")

    return async_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TLC-backed per-zone recommendation replay")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=[m.value for m in RunMode], required=True)
    parser.add_argument("--n-zones", type=int, default=20)
    parser.add_argument("--max-inflight-zones", type=int, default=4)
    parser.add_argument("--tick-timeout-s", type=float, default=2.0)
    parser.add_argument("--completion-fraction", type=float, default=0.75)
    parser.add_argument("--slow-zone-fraction", type=float, default=0.25)
    parser.add_argument("--slow-zone-sleep-s", type=float, default=1.0)
    parser.add_argument("--fallback-policy", default="always_previous")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ray-address", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    ray.init(address=args.ray_address if args.ray_address else None)

    config = RunConfig(
        n_zones=args.n_zones,
        max_inflight_zones=args.max_inflight_zones,
        tick_timeout_s=args.tick_timeout_s,
        completion_fraction=args.completion_fraction,
        slow_zone_fraction=args.slow_zone_fraction,
        slow_zone_sleep_s=args.slow_zone_sleep_s,
        fallback_policy=args.fallback_policy,
        seed=args.seed,
    )

    mode = RunMode(args.mode)

    if mode == RunMode.BLOCKING:
        run_blocking(args.prepared_dir, args.output_dir, config)
    elif mode == RunMode.ASYNC:
        run_async(args.prepared_dir, args.output_dir, config)
    else:
        run_stress(args.prepared_dir, args.output_dir, config)

    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
