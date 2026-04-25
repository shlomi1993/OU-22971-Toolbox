"""
Tests for ZoneActor state management and fault-tolerance.

Covers:
- Write decision idempotency
- Report decision duplicate detection
- Late report handling
- Finalize tick behavior
- Fallback policy application
- Multi-tick decision accumulation
- Blocking mode (all zones complete)
- Async mode (partial readiness with fallback)
"""

import pytest
import ray

from src.core import FALLBACK_POLICY_PREVIOUS, ReplayConfig
from src.zone_actor import Recommendation, WriteStatus, ZoneActor
from tests.helpers import make_zone_data, RNG_SEED


# ============================================================================
# ZoneActor fault-tolerance invariants
# ============================================================================


@pytest.mark.usefixtures("ray_ctx")
def test_write_decision_idempotent() -> None:
    replay, baseline = make_zone_data(zone_id=10)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(10, replay, baseline, config)
    ray.get(actor.activate_tick.remote(0))

    first = ray.get(actor.write_decision.remote(0, "NEED"))
    second = ray.get(actor.write_decision.remote(0, "NEED"))

    assert first == WriteStatus.WRITTEN, "First write should be WRITTEN"
    assert second == WriteStatus.DUPLICATE, "Duplicate write for same (zone, tick) must return DUPLICATE"
    conflicting = ray.get(actor.write_decision.remote(0, "OK"))
    assert conflicting == WriteStatus.DUPLICATE, "Write with different decision for same tick must still be DUPLICATE"
    decisions = ray.get(actor.get_accepted_decisions.remote())
    assert decisions[0] == "NEED", "Conflicting write must not overwrite the original accepted decision"


@pytest.mark.usefixtures("ray_ctx")
def test_report_decision_duplicate_detected() -> None:
    replay, baseline = make_zone_data(zone_id=20)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(20, replay, baseline, config)
    ray.get(actor.activate_tick.remote(0))

    first = ray.get(actor.report_decision.remote(0, "NEED", 0.1))
    second = ray.get(actor.report_decision.remote(0, "NEED", 0.1))

    assert first == WriteStatus.ACCEPTED, "First report should be ACCEPTED"
    assert second == WriteStatus.DUPLICATE, "Second report for same tick must be DUPLICATE"
    counters = ray.get(actor.get_counters.remote())
    assert counters.n_duplicates == 1, f"Expected 1 duplicate, got {counters.n_duplicates}"


@pytest.mark.usefixtures("ray_ctx")
def test_report_late_for_closed_or_inactive_tick() -> None:
    replay, baseline = make_zone_data(zone_id=30)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(30, replay, baseline, config)

    ray.get(actor.activate_tick.remote(0))
    ray.get(actor.report_decision.remote(0, "OK", 0.05))
    ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))

    ray.get(actor.activate_tick.remote(1))

    status_closed = ray.get(actor.report_decision.remote(0, "NEED", 0.1))
    assert status_closed == WriteStatus.LATE, "Report for already-closed tick must be LATE"

    status_inactive = ray.get(actor.report_decision.remote(5, "NEED", 0.1))
    assert status_inactive == WriteStatus.LATE, "Report for non-active tick must be LATE"

    counters = ray.get(actor.get_counters.remote())
    assert counters.n_late == 2, f"Expected 2 late reports, got {counters.n_late}"


@pytest.mark.usefixtures("ray_ctx")
def test_finalize_prefers_reported_decision() -> None:
    replay, baseline = make_zone_data(zone_id=40)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(40, replay, baseline, config)

    ray.get(actor.activate_tick.remote(0))
    ray.get(actor.report_decision.remote(0, "NEED", 0.05))
    ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))

    decisions = ray.get(actor.get_accepted_decisions.remote())
    assert decisions[0] == "NEED", "Finalize must accept the reported decision when one exists"


@pytest.mark.usefixtures("ray_ctx")
def test_finalize_uses_fallback_when_no_report() -> None:
    replay, baseline = make_zone_data(zone_id=50)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(50, replay, baseline, config)

    ray.get(actor.activate_tick.remote(0))
    ray.get(actor.report_decision.remote(0, "NEED", 0.05))
    ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))

    ray.get(actor.activate_tick.remote(1))
    result = ray.get(actor.finalize_tick.remote(1, FALLBACK_POLICY_PREVIOUS))
    assert result == WriteStatus.WRITTEN, "Finalize with fallback should return WRITTEN"

    decisions = ray.get(actor.get_accepted_decisions.remote())
    assert decisions[1] == "NEED", "Fallback under always_previous must repeat last accepted decision"
    counters = ray.get(actor.get_counters.remote())
    assert counters.n_fallbacks == 1, f"Expected 1 fallback, got {counters.n_fallbacks}"


@pytest.mark.usefixtures("ray_ctx")
def test_finalize_tick_idempotent() -> None:
    replay, baseline = make_zone_data(zone_id=60)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(60, replay, baseline, config)

    ray.get(actor.activate_tick.remote(0))
    ray.get(actor.report_decision.remote(0, "OK", 0.05))

    first = ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))
    second = ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))

    assert first == WriteStatus.WRITTEN, "First finalize should be WRITTEN"
    assert second == WriteStatus.DUPLICATE, "Repeated finalize for same tick must be DUPLICATE"


@pytest.mark.usefixtures("ray_ctx")
def test_late_report_after_finalize_does_not_overwrite() -> None:
    replay, baseline = make_zone_data(zone_id=70)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(70, replay, baseline, config)

    ray.get(actor.activate_tick.remote(0))
    ray.get(actor.report_decision.remote(0, "OK", 0.05))
    ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))

    ray.get(actor.activate_tick.remote(1))
    status = ray.get(actor.report_decision.remote(0, "NEED", 0.1))
    assert status == WriteStatus.LATE, "Report after finalization must be LATE"

    decisions = ray.get(actor.get_accepted_decisions.remote())
    assert decisions[0] == "OK", "Late report must not overwrite the finalized decision"


@pytest.mark.usefixtures("ray_ctx")
def test_first_tick_finalize_without_report_defaults_ok() -> None:
    replay, baseline = make_zone_data(zone_id=80)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(80, replay, baseline, config)

    ray.get(actor.activate_tick.remote(0))
    ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))

    decisions = ray.get(actor.get_accepted_decisions.remote())
    assert decisions[0] == Recommendation.OK, "First tick with no report and no history must default to OK"
    counters = ray.get(actor.get_counters.remote())
    assert counters.n_fallbacks == 1, "First-use fallback must be counted"


@pytest.mark.usefixtures("ray_ctx")
def test_multi_tick_decisions_accumulate() -> None:
    replay, baseline = make_zone_data(zone_id=90, n_ticks=5)
    config = ReplayConfig(n_zones=1)
    actor = ZoneActor.remote(90, replay, baseline, config)

    for tick_id in range(3):
        ray.get(actor.activate_tick.remote(tick_id))
        snap = ray.get(actor.get_snapshot.remote(tick_id))
        decision = snap.compute_decision()
        ray.get(actor.write_decision.remote(tick_id, decision))

    decisions = ray.get(actor.get_accepted_decisions.remote())
    assert len(decisions) == 3, f"Expected 3 accumulated decisions, got {len(decisions)}"
    for tick_id in range(3):
        assert tick_id in decisions, f"Missing decision for tick {tick_id}"
        assert decisions[tick_id] in ("NEED", "OK"), f"Invalid decision for tick {tick_id}: {decisions[tick_id]}"


# ============================================================================
# Blocking mode - all zones complete, no fallback
# ============================================================================


@pytest.mark.usefixtures("ray_ctx")
def test_blocking_all_zones_decided_no_fallback() -> None:
    config = ReplayConfig(n_zones=3, seed=RNG_SEED)
    zone_ids = [100, 200, 300]
    actors = {}
    for zid in zone_ids:
        replay, baseline = make_zone_data(zone_id=zid)
        actors[zid] = ZoneActor.remote(zid, replay, baseline, config)

    tick_id = 0
    ray.get([a.activate_tick.remote(tick_id) for a in actors.values()])
    snapshots = {zid: ray.get(a.get_snapshot.remote(tick_id)) for zid, a in actors.items()}

    for zid, snap in snapshots.items():
        decision = snap.compute_decision()
        ray.get(actors[zid].write_decision.remote(tick_id, decision))

    for zid in zone_ids:
        accepted = ray.get(actors[zid].get_accepted_decisions.remote())
        assert tick_id in accepted, f"Zone {zid} missing decision for tick {tick_id}"
        assert accepted[tick_id] in ("NEED", "OK"), f"Zone {zid} has invalid decision: {accepted[tick_id]}"
        counters = ray.get(actors[zid].get_counters.remote())
        assert counters.n_fallbacks == 0, f"Zone {zid} should have 0 fallbacks in blocking mode"


# ============================================================================
# Async mode - partial readiness triggers fallback
# ============================================================================


@pytest.mark.usefixtures("ray_ctx")
def test_async_partial_readiness_triggers_fallback() -> None:
    config = ReplayConfig(n_zones=3, seed=RNG_SEED)
    zone_ids = [110, 220, 330]
    actors = {}
    for zid in zone_ids:
        replay, baseline = make_zone_data(zone_id=zid)
        actors[zid] = ZoneActor.remote(zid, replay, baseline, config)

    tick_id = 0
    ray.get([a.activate_tick.remote(tick_id) for a in actors.values()])

    ray.get(actors[110].report_decision.remote(tick_id, "NEED", 0.05))

    readiness = {zid: ray.get(a.has_decision_for_tick.remote(tick_id)) for zid, a in actors.items()}
    assert readiness[110] is True, "Zone 110 should have a reported decision"
    assert readiness[220] is False, "Zone 220 has_decision_for_tick must return False, not None"
    assert readiness[330] is False, "Zone 330 has_decision_for_tick must return False, not None"
    assert sum(readiness.values()) == 1, "sum(readiness) must work - driver uses this to count ready zones"

    for actor in actors.values():
        ray.get(actor.finalize_tick.remote(tick_id, FALLBACK_POLICY_PREVIOUS))

    d110 = ray.get(actors[110].get_accepted_decisions.remote())
    assert d110[tick_id] == "NEED", "Zone 110 must use its reported decision"

    for zid in [220, 330]:
        counters = ray.get(actors[zid].get_counters.remote())
        assert counters.n_fallbacks >= 1, f"Zone {zid} must have used fallback"
