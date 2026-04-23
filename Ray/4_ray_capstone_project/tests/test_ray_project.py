"""
Unit and integration tests for the Ray capstone project.

Tests cover:
- Data validation: Adjacent month validation, zone selection, tick aggregation
- Scoring logic: Snapshot computation, decision rules, fallback policies
- Actor behavior: ZoneActor state management, idempotent writes, counters
- Replay mechanics: Blocking vs async execution patterns

Run with: pytest tests/test_ray_project.py -v
"""

import json
import os
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
import pytest
import ray

from pathlib import Path
from typing import Dict, Generator, List, Tuple

from src.tlc import (
    FALLBACK_POLICY_PREVIOUS,
    TICK_MINUTES,
    RunConfig,
    TickMetrics,
    aggregate_ticks,
    build_baseline_table,
    build_replay_table,
    cross_check_replay,
    identify_busiest_zones,
    select_slow_zones,
    validate_adjacent_months,
    write_json,
    write_latency_log,
    write_metrics_csv,
    write_tick_summary,
)
from src.zone_actor import Recommendation, WriteStatus, ZoneActor, ZoneSnapshot

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # Ensure Ray doesn't override our environment variables during testing

RNG_SEED = 42


@pytest.fixture(scope="function")
def ray_ctx() -> Generator[None, None, None]:
    warnings.filterwarnings("ignore", category=FutureWarning, module="ray")
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(num_cpus=2, logging_level="ERROR", log_to_driver=False)
    yield
    ray.shutdown()


def make_trips(year: int, month: int, n_zones: int = 30, base_count: int = 50) -> pd.DataFrame:
    rng = np.random.RandomState(RNG_SEED)
    rows = []
    for i in range(n_zones):
        zone_id = i + 1
        count = base_count + (n_zones - i) * 10
        for _ in range(count):
            day = 1 + rng.randint(0, 27)
            hour = rng.randint(0, 23)
            minute = rng.randint(0, 59)
            dt = pd.Timestamp(year, month, day, hour, minute)
            rows.append({
                "lpep_pickup_datetime": dt,
                "lpep_dropoff_datetime": dt + pd.Timedelta(minutes=15),
                "PULocationID": zone_id,
            })
    return pd.DataFrame(rows)


def make_zone_data(zone_id: int = 10, n_ticks: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.Timestamp("2023-02-01")
    ticks = [base + pd.Timedelta(minutes=15 * i) for i in range(n_ticks)]
    demands = [10.0, 12.0, 8.0, 15.0, 20.0][:n_ticks]
    replay = pd.DataFrame({"zone_id": zone_id, "tick_start": ticks, "demand": demands})
    baseline = pd.DataFrame({
        "zone_id": [zone_id],
        "hour_of_day": [0],
        "day_of_week": [2],
        "mean_demand": [10.0],
        "std_demand": [2.0],
    })
    return replay, baseline


# ── Step 2: Preparation and validation ──────────────────────────────────────


def test_validate_adjacent_months_accepts_consecutive() -> None:
    ref = make_trips(2023, 1, n_zones=5, base_count=20)
    replay = make_trips(2023, 2, n_zones=5, base_count=20)
    ref_label, replay_label = validate_adjacent_months(ref, replay)
    assert ref_label == "2023-01", f"Expected ref_label='2023-01', got '{ref_label}'"
    assert replay_label == "2023-02", f"Expected replay_label='2023-02', got '{replay_label}'"


def test_validate_adjacent_months_rejects_non_adjacent() -> None:
    ref = make_trips(2023, 1, n_zones=5, base_count=20)
    replay = make_trips(2023, 3, n_zones=5, base_count=20)
    with pytest.raises(ValueError, match="not adjacent"):
        validate_adjacent_months(ref, replay)


def test_validate_adjacent_months_rejects_different_years() -> None:
    ref = make_trips(2023, 1, n_zones=5, base_count=20)
    replay = make_trips(2024, 2, n_zones=5, base_count=20)
    with pytest.raises(ValueError, match="year"):
        validate_adjacent_months(ref, replay)


def test_active_zone_selection_deterministic_and_sorted() -> None:
    ref = make_trips(2023, 1, n_zones=30, base_count=50)
    zones_a = identify_busiest_zones(ref, 10)
    zones_b = identify_busiest_zones(ref, 10)
    assert zones_a == zones_b, "Active zone selection must be deterministic under fixed input"
    assert zones_a == sorted(zones_a), "Active zones must be returned sorted"
    assert len(zones_a) == 10, f"Expected 10 zones, got {len(zones_a)}"


def test_aggregate_ticks_aligns_to_15min_boundaries() -> None:
    ref = make_trips(2023, 1, n_zones=5, base_count=50)
    agg = aggregate_ticks(ref)
    residuals = agg["tick_start"].dt.minute % TICK_MINUTES
    assert (residuals == 0).all(), "All tick_start values must be aligned to 15-minute boundaries"
    assert set(agg.columns) >= {"zone_id", "tick_start", "demand"}, "Aggregated table missing required columns"


def test_replay_table_filters_to_active_zones() -> None:
    replay_raw = make_trips(2023, 2, n_zones=30, base_count=50)
    agg = aggregate_ticks(replay_raw)
    active = [1, 2, 3, 5, 10]
    replay_table = build_replay_table(agg, active)
    found_zones = set(replay_table["zone_id"].unique())
    assert found_zones.issubset(set(active)), f"Replay table has non-active zones: {found_zones - set(active)}"
    assert len(replay_table) > 0, "Replay table should not be empty for valid active zones"


def test_cross_check_replay_validates_consistency() -> None:
    raw_df = make_trips(2023, 2, n_zones=10, base_count=50)
    agg = aggregate_ticks(raw_df)
    active = list(range(1, 11))
    replay_table = build_replay_table(agg, active)
    assert cross_check_replay(raw_df, replay_table, active), "Cross-check must pass on consistently prepared data"


def test_baseline_table_columns_and_values() -> None:
    ref = make_trips(2023, 1, n_zones=5, base_count=50)
    agg = aggregate_ticks(ref)
    baseline = build_baseline_table(agg)
    required_cols = {"zone_id", "hour_of_day", "day_of_week", "mean_demand", "std_demand"}
    assert required_cols.issubset(set(baseline.columns)), f"Baseline missing columns: {required_cols - set(baseline.columns)}"
    assert (baseline["mean_demand"] >= 0).all(), "Baseline mean_demand must be non-negative"
    assert (baseline["std_demand"] >= 0).all(), "Baseline std_demand must be non-negative"
    assert len(baseline) > 0, "Baseline table should not be empty"


# ── Scoring and decision logic ──────────────────────────────────────────────


def test_snapshot_decision_threshold() -> None:
    snap_need = ZoneSnapshot(zone_id=1, tick_id=0, recent_demand=[15.0, 16.0], baseline_mean=10.0, baseline_std=2.0)
    snap_ok = ZoneSnapshot(zone_id=1, tick_id=0, recent_demand=[8.0, 9.0], baseline_mean=10.0, baseline_std=2.0)
    snap_empty = ZoneSnapshot(zone_id=1, tick_id=0, recent_demand=[], baseline_mean=10.0, baseline_std=2.0)
    assert snap_need.compute_decision() == Recommendation.NEED, "Demand above baseline+std must produce NEED"
    assert snap_ok.compute_decision() == Recommendation.OK, "Demand below baseline+std must produce OK"
    assert snap_empty.compute_decision() == Recommendation.OK, "Empty recent demand must produce OK"


def test_fallback_always_previous_policy() -> None:
    assert ZoneActor.apply_fallback(FALLBACK_POLICY_PREVIOUS, "NEED") == "NEED", "Fallback should return previous decision when available"
    assert ZoneActor.apply_fallback(FALLBACK_POLICY_PREVIOUS, None) == Recommendation.OK, "Fallback should default to OK when no previous decision exists (first-use edge case)"
    assert ZoneActor.apply_fallback("unknown_policy", "NEED") == Recommendation.OK, "Unknown fallback policy should default to OK"


# ── ZoneActor fault-tolerance invariants ────────────────────────────────────


@pytest.mark.usefixtures("ray_ctx")
def test_write_decision_idempotent() -> None:
    replay, baseline = make_zone_data(zone_id=10)
    config = RunConfig(n_zones=1)
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
    config = RunConfig(n_zones=1)
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
    config = RunConfig(n_zones=1)
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
    config = RunConfig(n_zones=1)
    actor = ZoneActor.remote(40, replay, baseline, config)

    ray.get(actor.activate_tick.remote(0))
    ray.get(actor.report_decision.remote(0, "NEED", 0.05))
    ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))

    decisions = ray.get(actor.get_accepted_decisions.remote())
    assert decisions[0] == "NEED", "Finalize must accept the reported decision when one exists"


@pytest.mark.usefixtures("ray_ctx")
def test_finalize_uses_fallback_when_no_report() -> None:
    replay, baseline = make_zone_data(zone_id=50)
    config = RunConfig(n_zones=1)
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
    config = RunConfig(n_zones=1)
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
    config = RunConfig(n_zones=1)
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
    config = RunConfig(n_zones=1)
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
    config = RunConfig(n_zones=1)
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


# ── Step 3: Blocking mode — all zones complete, no fallback ─────────────────


@pytest.mark.usefixtures("ray_ctx")
def test_blocking_all_zones_decided_no_fallback() -> None:
    config = RunConfig(n_zones=3, seed=RNG_SEED)
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


# ── Step 4: Async mode — partial readiness triggers fallback ────────────────


@pytest.mark.usefixtures("ray_ctx")
def test_async_partial_readiness_triggers_fallback() -> None:
    config = RunConfig(n_zones=3, seed=RNG_SEED)
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
    assert sum(readiness.values()) == 1, "sum(readiness) must work — driver uses this to count ready zones"

    for actor in actors.values():
        ray.get(actor.finalize_tick.remote(tick_id, FALLBACK_POLICY_PREVIOUS))

    d110 = ray.get(actors[110].get_accepted_decisions.remote())
    assert d110[tick_id] == "NEED", "Zone 110 must use its reported decision"

    for zid in [220, 330]:
        counters = ray.get(actors[zid].get_counters.remote())
        assert counters.n_fallbacks >= 1, f"Zone {zid} must have used fallback"


# ── Skew model ──────────────────────────────────────────────────────────────


def test_slow_zone_selection_deterministic() -> None:
    zones = list(range(1, 21))
    config = RunConfig(seed=RNG_SEED, slow_zone_fraction=0.25)
    slow_a = select_slow_zones(zones, config)
    slow_b = select_slow_zones(zones, config)
    assert slow_a == slow_b, "Slow zone selection must be deterministic with fixed seed"
    expected_count = max(1, int(len(zones) * config.slow_zone_fraction))
    assert len(slow_a) == expected_count, f"Expected {expected_count} slow zones, got {len(slow_a)}"


# ── Step 6: Artifact file verification ──────────────────────────────────────


def test_artifact_files_written(tmp_path: Path) -> None:
    config = RunConfig(n_zones=2)
    metrics = [
        TickMetrics(tick_id=0, mode="blocking", n_zones_completed=2, total_tick_latency_s=0.5, per_zone_latency={1: 0.1, 2: 0.2})
    ]
    decisions = {0: {1: "OK", 2: "NEED"}}

    out = tmp_path / "artifacts"
    out.mkdir()
    write_json(config.to_dict(), out / "run_config.json")
    write_metrics_csv(metrics, out / "metrics.csv")
    write_tick_summary(metrics, decisions, out / "tick_summary.json")
    write_latency_log(metrics, out / "latency_log.json")

    for fname in ["run_config.json", "metrics.csv", "tick_summary.json", "latency_log.json"]:
        assert (out / fname).exists(), f"Artifact '{fname}' must be written to disk"
        assert (out / fname).stat().st_size > 0, f"Artifact '{fname}' must not be empty"

    with open(out / "run_config.json") as f:
        run_config = json.load(f)
    assert "n_zones" in run_config, "run_config.json must contain 'n_zones'"
    assert "fallback_policy" in run_config, "run_config.json must contain 'fallback_policy'"

    with open(out / "tick_summary.json") as f:
        summary = json.load(f)
    assert len(summary) == 1, "tick_summary.json should have one entry per tick"
    assert "decisions" in summary[0], "tick_summary entry must contain 'decisions'"


# ── Script-level subprocess tests ───────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent

BLOCKING_ASYNC_ARTIFACTS = [
    "run_config.json", "metrics.csv", "tick_summary.json", "latency_log.json", "actor_counters.json",
]


@pytest.fixture(scope="module")
def synthetic_parquets(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    base = tmp_path_factory.mktemp("parquet")
    ref = make_trips(2023, 1, n_zones=10, base_count=30)
    replay = make_trips(2023, 2, n_zones=10, base_count=30)
    ref_path = base / "ref.parquet"
    replay_path = base / "replay.parquet"
    ref.to_parquet(ref_path, index=False)
    replay.to_parquet(replay_path, index=False)
    return {"ref": ref_path, "replay": replay_path}


def _run_script(args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable] + args, capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=timeout)


@pytest.fixture(scope="module")
def prepared_dir(synthetic_parquets: Dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("prepared")
    result = _run_script([
        "main.py",
        "prepare",
        "--ref-parquet", str(synthetic_parquets["ref"]),
        "--replay-parquet", str(synthetic_parquets["replay"]),
        "--output-dir", str(out),
        "--n-zones", "5",
    ])
    assert result.returncode == 0, f"main.py prepare fixture failed: {result.stderr}"
    return out


@pytest.mark.parametrize("n_zones,seed", [(3, None), (5, None), (3, 42), (5, 42)])
def test_prepare_script(synthetic_parquets: Dict[str, Path], tmp_path: Path, n_zones: int, seed: int | None) -> None:
    out = tmp_path / "prepared"
    args = [
        "main.py",
        "prepare",
        "--ref-parquet", str(synthetic_parquets["ref"]),
        "--replay-parquet", str(synthetic_parquets["replay"]),
        "--output-dir", str(out),
        "--n-zones", str(n_zones),
    ]
    if seed is not None:
        args.extend(["--seed", str(seed)])

    result = _run_script(args)
    assert result.returncode == 0, f"main.py prepare --n-zones {n_zones} --seed {seed} failed: {result.stderr}"
    for fname in ["baseline.parquet", "replay.parquet", "active_zones.json", "prep_meta.json"]:
        assert (out / fname).exists(), f"main.py prepare must produce {fname}"
    with open(out / "active_zones.json") as f:
        zones = json.load(f)
    assert len(zones) == n_zones, f"Expected {n_zones} zones, got {len(zones)}"
    with open(out / "prep_meta.json") as f:
        meta = json.load(f)
    if seed is not None:
        assert meta["seed"] == seed, f"prep_meta.json seed mismatch: expected {seed}, got {meta['seed']}"
    assert meta["n_zones"] == n_zones, f"prep_meta.json n_zones mismatch: expected {n_zones}, got {meta['n_zones']}"


@pytest.mark.parametrize("mode,slow_frac,slow_sleep,timeout_s,max_inflight,fallback", [
    ("blocking", "0.25", "0.1", "2.0", "4", "always_previous"),
    ("blocking", "0.0", "0.0", "2.0", "4", "always_previous"),
    ("blocking", "0.4", "0.2", "2.0", "4", "always_previous"),
    ("async", "0.25", "0.1", "2.0", "4", "always_previous"),
    ("async", "0.0", "0.0", "2.0", "4", "always_previous"),
    ("async", "0.4", "0.2", "1.0", "2", "always_previous"),
    ("async", "0.25", "0.1", "3.0", "5", "always_previous"),
    ("async", "0.4", "0.3", "1.5", "3", "always_previous"),
])
def test_run_script(prepared_dir: Path, tmp_path: Path, mode: str, slow_frac: str, slow_sleep: str, timeout_s: str,
                    max_inflight: str, fallback: str) -> None:
    out = tmp_path / "output"
    result = _run_script([
        "main.py",
        "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", mode,
        "--n-zones", "5",
        "--slow-zone-fraction", slow_frac,
        "--slow-zone-sleep-s", slow_sleep,
        "--tick-timeout-s", timeout_s,
        "--max-inflight-zones", max_inflight,
        "--fallback-policy", fallback,
        "--seed", "42",
        "--max-ticks", "50",
    ])
    assert result.returncode == 0, f"main.py run --mode {mode} failed: {result.stderr}"
    artifact_dir = out / mode
    for fname in BLOCKING_ASYNC_ARTIFACTS:
        assert (artifact_dir / fname).exists(), f"main.py run --mode {mode} must produce {fname}"


def test_run_stress_script(prepared_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "output"
    result = _run_script([
        "main.py",
        "run",
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(out),
        "--mode", "stress",
        "--n-zones", "5",
        "--slow-zone-fraction", "0.6",
        "--slow-zone-sleep-s", "0.3",
        "--tick-timeout-s", "2.0",
        "--seed", "42",
        "--max-ticks", "50",
    ], timeout=300)
    assert result.returncode == 0, f"main.py run --mode stress failed: {result.stderr}"
    stress_dir = out / "stress"
    assert (stress_dir / "comparison.json").exists(), "Stress mode must produce comparison.json"
    with open(stress_dir / "comparison.json") as f:
        comparison = json.load(f)
    assert "blocking" in comparison, "comparison.json must contain 'blocking' key"
    assert "async" in comparison, "comparison.json must contain 'async' key"
    for key in ["blocking", "async"]:
        assert "mean_tick_latency" in comparison[key], f"comparison.json['{key}'] must have mean_tick_latency"
    for sub in ["blocking", "async"]:
        sub_dir = stress_dir / sub
        for fname in BLOCKING_ASYNC_ARTIFACTS:
            assert (sub_dir / fname).exists(), f"Stress sub-run {sub} must produce {fname}"
