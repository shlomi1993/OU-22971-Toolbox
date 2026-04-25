"""
Tests for data validation, scoring logic, and artifact verification.

Covers:
- Adjacent month validation
- Zone selection and tick aggregation
- Replay table construction
- Baseline table generation
- Snapshot decision thresholds
- Fallback policy logic
- Artifact file writing
"""

import json
import pytest

from pathlib import Path

from src.core import (
    FALLBACK_POLICY_PREVIOUS,
    TICK_MINUTES,
    ReplayConfig,
    TickMetrics,
    aggregate_ticks,
    build_baseline_table,
    build_replay_table,
    cross_check_replay,
    identify_busiest_zones,
    validate_adjacent_months,
    write_json,
    write_latency_log,
    write_metrics_csv,
    write_tick_summary,
)
from src.core import DemandVerdict
from src.zone_actor import ZoneActor
from tests.helpers import make_trips


# ============================================================================
# Data validation tests
# ============================================================================


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


# ============================================================================
# Scoring and decision logic tests
# ============================================================================


def test_fallback_always_previous_policy() -> None:
    assert ZoneActor.apply_fallback(FALLBACK_POLICY_PREVIOUS, "NEED") == "NEED", "Fallback should return previous decision when available"
    assert ZoneActor.apply_fallback(FALLBACK_POLICY_PREVIOUS, None) == DemandVerdict.OK, "Fallback should default to OK when no previous decision exists (first-use edge case)"
    assert ZoneActor.apply_fallback("unknown_policy", "NEED") == DemandVerdict.OK, "Unknown fallback policy should default to OK"


# ============================================================================
# Artifact file verification tests
# ============================================================================


def test_artifact_files_written(tmp_path: Path) -> None:
    config = ReplayConfig(n_zones=2)
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
