"""
Tests for Stretch A (delayed arrivals) and Stretch B (adaptive sub-actors).
"""

import pytest
import ray

from pathlib import Path
from typing import Dict

from src.core import FALLBACK_POLICY_PREVIOUS, ReplayConfig
from src.zone_actor import WriteStatus, ZoneActor
from src.zone_subactor import ZoneSubActor
from tests.helpers import make_zone_data, run_script, run_prepare_script



# ============================================================================
# Stretch A — Delayed arrival unit tests
# ============================================================================


@pytest.mark.usefixtures("ray_ctx")
def test_delayed_withhold_deterministic() -> None:
    """
    Delay decision is deterministic, respects fraction=0 (never) and fraction=1 (always).
    """
    results = [ZoneActor._is_demand_delayed(10, 0, 0.5, 42) for _ in range(5)]
    assert len(set(results)) == 1, "Delayed decision must be deterministic across calls"

    for tick in range(10):
        assert not ZoneActor._is_demand_delayed(1, tick, 0.0, 42), \
            f"delayed_fraction=0 must never withhold (tick {tick})"
        assert ZoneActor._is_demand_delayed(1, tick, 1.0, 42), \
            f"delayed_fraction=1.0 must always withhold (tick {tick})"


@pytest.mark.usefixtures("ray_ctx")
def test_activate_tick_delayed_withholds_and_releases() -> None:
    """
    Demand withheld at tick T is released at tick T + delay_ticks,
    and finalize_tick_delayed is idempotent.
    """
    replay, baseline = make_zone_data(zone_id=10, n_ticks=5)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(10, replay, baseline, config)

    # Tick 0: withheld, release scheduled at tick 2
    info = ray.get(actor.activate_tick_delayed.remote(0, 2, 1.0, 42))
    assert info["withheld"] is True, "Tick 0 demand must be withheld"
    assert info["release_tick"] == 2, "Demand should release at tick 2"
    first = ray.get(actor.finalize_tick_delayed.remote(0, FALLBACK_POLICY_PREVIOUS))
    second = ray.get(actor.finalize_tick_delayed.remote(0, FALLBACK_POLICY_PREVIOUS))
    assert first == WriteStatus.WRITTEN, "First finalize must be WRITTEN"
    assert second == WriteStatus.DUPLICATE, "Repeated finalize must be DUPLICATE"

    # Tick 1: nothing released yet
    info1 = ray.get(actor.activate_tick_delayed.remote(1, 2, 1.0, 42))
    assert info1["withheld"] is True, "Tick 1 demand must be withheld"
    assert len(info1["released"]) == 0, "Nothing should be released at tick 1"
    ray.get(actor.finalize_tick_delayed.remote(1, FALLBACK_POLICY_PREVIOUS))

    # Tick 2: tick 0's demand released
    info2 = ray.get(actor.activate_tick_delayed.remote(2, 2, 1.0, 42))
    assert len(info2["released"]) >= 1, "Tick 0 demand should be released at tick 2"
    released_orig_ticks = [e["orig_tick"] for e in info2["released"]]
    assert 0 in released_orig_ticks, "Released entry must reference original tick 0"


@pytest.mark.usefixtures("ray_ctx")
def test_delayed_snapshot_and_release_visibility() -> None:
    """
    Withheld demand is excluded from the delayed snapshot; once released it appears in later snapshots.
    """
    replay, baseline = make_zone_data(zone_id=20, n_ticks=5)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(20, replay, baseline, config)

    # Withhold tick 0 demand
    ray.get(actor.activate_tick_delayed.remote(0, 1, 1.0, 42))
    snap_delayed = ray.get(actor.get_snapshot_delayed.remote(0))
    snap_regular = ray.get(actor.get_snapshot.remote(0))
    assert len(snap_delayed.recent_demand) < len(snap_regular.recent_demand), \
        "Delayed snapshot must exclude the withheld demand"

    ray.get(actor.finalize_tick_delayed.remote(0, FALLBACK_POLICY_PREVIOUS))

    # Tick 1: tick 0's demand released
    info1 = ray.get(actor.activate_tick_delayed.remote(1, 1, 1.0, 42))
    assert len(info1["released"]) >= 1, "Tick 0 demand should be released at tick 1"
    snap = ray.get(actor.get_snapshot_delayed.remote(1))
    assert len(snap.recent_demand) > 0, "Released demand must appear in the snapshot"


@pytest.mark.usefixtures("ray_ctx")
def test_delay_counters_and_log() -> None:
    """
    Counters track withheld/released counts and the delay log contains both event types.
    """
    replay, baseline = make_zone_data(zone_id=50, n_ticks=5)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(50, replay, baseline, config)

    # 3 ticks with fraction=1.0 -> 3 withholds; delay=1 -> releases at ticks 1, 2, 3
    for t in range(3):
        ray.get(actor.activate_tick_delayed.remote(t, 1, 1.0, 42))
        ray.get(actor.finalize_tick_delayed.remote(t, FALLBACK_POLICY_PREVIOUS))

    counters = ray.get(actor.get_counters.remote())
    assert counters.n_delayed_withheld == 3, f"Expected 3 withholds, got {counters.n_delayed_withheld}"
    assert counters.n_delayed_released == 2, f"Expected 2 releases, got {counters.n_delayed_released}"

    log = ray.get(actor.get_delay_log.remote())
    events = {e["event"] for e in log}
    assert "withhold" in events, "Log must contain withhold events"
    assert "release" in events, "Log must contain release events"


# ============================================================================
# Stretch B — Sub-actor unit tests
# ============================================================================


@pytest.mark.usefixtures("ray_ctx")
def test_subactor_score_report_and_stats() -> None:
    """
    ZoneSubActor produces the same decision as direct scoring, reports to parent, and tracks stats.
    """
    replay, baseline = make_zone_data(zone_id=100, n_ticks=5)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(100, replay, baseline, config)
    subactor = ZoneSubActor.remote(100, actor)

    for t in range(3):
        ray.get(actor.activate_tick.remote(t))
        snap = ray.get(actor.get_snapshot.remote(t))
        expected = snap.compute_decision()
        result = ray.get(subactor.score_and_report.remote(snap))
        assert result.decision == expected, \
            f"Tick {t}: sub-actor decision {result.decision} != direct {expected}"
        assert result.zone_id == 100, f"Expected zone_id 100, got {result.zone_id}"
        assert result.tick_id == t, f"Expected tick_id {t}, got {result.tick_id}"
        has = ray.get(actor.has_decision_for_tick.remote(t))
        assert has is True, f"Sub-actor must report decision to parent for tick {t}"
        ray.get(actor.finalize_tick.remote(t, FALLBACK_POLICY_PREVIOUS))

    stats = ray.get(subactor.get_stats.remote())
    assert stats["n_scored"] == 3, f"Expected 3 scored, got {stats['n_scored']}"


@pytest.mark.usefixtures("ray_ctx")
def test_straggler_tracking() -> None:
    """
    ZoneActor counts slow ticks via record_tick_latency.
    """
    replay, baseline = make_zone_data(zone_id=120, n_ticks=5)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(120, replay, baseline, config)

    for lat in [1.0, 0.8, 0.3, 1.2]:
        ray.get(actor.record_tick_latency.remote(lat, 0.5))

    count = ray.get(actor.get_straggler_ticks.remote())
    assert count == 3, f"Expected 3 straggler ticks, got {count}"


@pytest.mark.usefixtures("ray_ctx")
def test_subactor_preserves_decision_semantics() -> None:
    """
    Multi-tick decisions via sub-actor match the regular scoring path exactly.
    """
    replay, baseline = make_zone_data(zone_id=130, n_ticks=5)
    config = ReplayConfig(n_zones=1)

    # Regular path
    actor_regular = ZoneActor.remote(130, replay, baseline, config)
    regular_decisions = {}
    for t in range(3):
        ray.get(actor_regular.activate_tick.remote(t))
        snap = ray.get(actor_regular.get_snapshot.remote(t))
        decision = snap.compute_decision()
        ray.get(actor_regular.write_decision.remote(t, decision))
        regular_decisions[t] = decision

    # Sub-actor path
    actor_sub = ZoneActor.remote(130, replay, baseline, config)
    subactor = ZoneSubActor.remote(130, actor_sub)
    sub_decisions = {}
    for t in range(3):
        ray.get(actor_sub.activate_tick.remote(t))
        snap = ray.get(actor_sub.get_snapshot.remote(t))
        result = ray.get(subactor.score_and_report.remote(snap))
        ray.get(actor_sub.finalize_tick.remote(t, FALLBACK_POLICY_PREVIOUS))
        sub_decisions[t] = result.decision

    for t in range(3):
        assert sub_decisions[t] == regular_decisions[t], \
            f"Tick {t}: sub-actor decision {sub_decisions[t]} != regular {regular_decisions[t]}"


# ============================================================================
# End-to-end script tests
# ============================================================================


STRETCH_ARTIFACTS = [
    "run_config.json",
    "metrics.csv",
    "tick_summary.json",
    "latency_log.json",
    "actor_counters.json",
]


def test_delayed_mode_script(synthetic_parquets: Dict[str, Path], tmp_path: Path, max_ticks: int) -> None:
    """
    End-to-end: prepare -> run delayed -> verify artifacts.
    """
    prepared_dir = tmp_path / "prepared"
    result = run_prepare_script(
        synthetic_parquets["ref"], synthetic_parquets["replay"], prepared_dir, n_zones=5
    )
    assert result.returncode == 0, "Prepare step failed"

    out = tmp_path / "output"
    result = run_script([
        "main.py", "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", "delayed",
        "--n-zones", "5",
        "--slow-zone-fraction", "0.25",
        "--slow-zone-sleep-s", "0.1",
        "--tick-timeout-s", "2.0",
        "--delayed-fraction", "0.4",
        "--delay-ticks", "2",
        "--seed", "42",
        "--max-ticks", str(max_ticks),
    ])
    assert result.returncode == 0, "delayed mode run failed"

    mode_dir = out / "delayed"
    for fname in STRETCH_ARTIFACTS:
        assert (mode_dir / fname).exists(), f"delayed mode must produce {fname}"
    assert (mode_dir / "delay_log.json").exists(), "delayed mode must produce delay_log.json"


def test_subactor_mode_script(synthetic_parquets: Dict[str, Path], tmp_path: Path, max_ticks: int) -> None:
    """
    End-to-end: prepare -> run subactor -> verify artifacts.
    """
    prepared_dir = tmp_path / "prepared"
    result = run_prepare_script(
        synthetic_parquets["ref"], synthetic_parquets["replay"], prepared_dir, n_zones=5
    )
    assert result.returncode == 0, "Prepare step failed"

    out = tmp_path / "output"
    result = run_script([
        "main.py", "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", "subactor",
        "--n-zones", "5",
        "--slow-zone-fraction", "0.25",
        "--slow-zone-sleep-s", "0.5",
        "--tick-timeout-s", "3.0",
        "--repeat-straggler-fraction", "0.2",
        "--straggler-trigger-count", "2",
        "--seed", "42",
        "--max-ticks", str(max(max_ticks, 5)),
    ])
    assert result.returncode == 0, "subactor mode run failed"

    mode_dir = out / "subactor"
    for fname in STRETCH_ARTIFACTS:
        assert (mode_dir / fname).exists(), f"subactor mode must produce {fname}"
    assert (mode_dir / "subactor_log.json").exists(), "subactor mode must produce subactor_log.json"
