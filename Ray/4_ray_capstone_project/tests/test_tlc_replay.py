# test_tlc_replay.py — Unit tests for tlc_lib, prepare, and run modules.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray

from src.tlc_lib import (
    FALLBACK_POLICY_PREVIOUS,
    FIRST_TICK_FALLBACK,
    TickMetrics,
    aggregate_ticks,
    build_baseline_table,
    build_replay_table,
    compute_decision,
    cross_check_replay,
    load_parquet,
    select_active_zones,
    validate_adjacent_months,
    write_json,
    write_latency_log,
    write_metrics_csv,
    write_tick_summary,
)
from src.zone_actor import RunConfig, WriteStatus, ZoneSnapshot, apply_fallback


@pytest.fixture(scope="session")
def ray_session():
    """Initialize Ray once for all tests."""
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


def _make_taxi_df(year: int, month: int, n_rows: int = 500, n_zones: int = 30,
                  seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Green Taxi dataframe for testing."""
    rng = np.random.RandomState(seed)
    days = pd.Period(f"{year}-{month:02d}", freq="M").days_in_month
    pickup_times = pd.to_datetime(
        [f"{year}-{month:02d}-{rng.randint(1, days + 1):02d} "
         f"{rng.randint(0, 24):02d}:{rng.randint(0, 60):02d}:00"
         for _ in range(n_rows)]
    )
    dropoff_times = pickup_times + pd.to_timedelta(rng.randint(5, 60, size=n_rows), unit="min")
    zones = rng.choice(range(1, n_zones + 1), size=n_rows, p=None)
    return pd.DataFrame({
        "lpep_pickup_datetime": pickup_times,
        "lpep_dropoff_datetime": dropoff_times,
        "PULocationID": zones,
    })


@pytest.fixture
def ref_df():
    return _make_taxi_df(2023, 1, n_rows=1000, seed=42)


@pytest.fixture
def replay_df():
    return _make_taxi_df(2023, 2, n_rows=800, seed=43)


@pytest.fixture
def tmp_parquets(ref_df, replay_df, tmp_path):
    ref_path = tmp_path / "ref.parquet"
    replay_path = tmp_path / "replay.parquet"
    ref_df.to_parquet(ref_path)
    replay_df.to_parquet(replay_path)
    return ref_path, replay_path


@pytest.fixture
def prepared_dir(ref_df, replay_df, tmp_path):
    """Run prepare logic and return the output directory."""
    from prepare import prepare_assets
    ref_path = tmp_path / "ref.parquet"
    replay_path = tmp_path / "replay.parquet"
    ref_df.to_parquet(ref_path)
    replay_df.to_parquet(replay_path)
    out = tmp_path / "prepared"
    prepare_assets(ref_path, replay_path, out, n_zones=10, seed=42)
    return out


class TestLoadParquet:
    def test_load_valid(self, tmp_parquets):
        ref_path, _ = tmp_parquets
        df = load_parquet(ref_path)
        assert "PULocationID" in df.columns

    def test_load_missing_columns(self, tmp_path):
        bad = pd.DataFrame({"foo": [1, 2]})
        path = tmp_path / "bad.parquet"
        bad.to_parquet(path)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_parquet(path)


class TestValidateAdjacentMonths:
    def test_adjacent_passes(self, ref_df, replay_df):
        ref_label, replay_label = validate_adjacent_months(ref_df, replay_df)
        assert ref_label == "2023-01"
        assert replay_label == "2023-02"

    def test_same_month_fails(self, ref_df):
        with pytest.raises(ValueError, match="not adjacent"):
            validate_adjacent_months(ref_df, ref_df)

    def test_different_year_fails(self, ref_df):
        other = _make_taxi_df(2024, 2)
        with pytest.raises(ValueError, match="year"):
            validate_adjacent_months(ref_df, other)


class TestSelectActiveZones:
    def test_deterministic(self, ref_df):
        z1 = select_active_zones(ref_df, 10, seed=42)
        z2 = select_active_zones(ref_df, 10, seed=42)
        assert z1 == z2

    def test_sorted(self, ref_df):
        zones = select_active_zones(ref_df, 10, seed=42)
        assert zones == sorted(zones)

    def test_respects_n_zones(self, ref_df):
        zones = select_active_zones(ref_df, 5, seed=42)
        assert len(zones) <= 5


class TestAggregateTicks:
    def test_output_columns(self, ref_df):
        agg = aggregate_ticks(ref_df)
        assert set(agg.columns) == {"zone_id", "tick_start", "demand"}

    def test_demand_positive(self, ref_df):
        agg = aggregate_ticks(ref_df)
        assert (agg["demand"] > 0).all()

    def test_total_demand_matches_rows(self, ref_df):
        agg = aggregate_ticks(ref_df)
        # Total demand should equal total rows (minus any NaT)
        valid = pd.to_datetime(ref_df["lpep_pickup_datetime"], errors="coerce").notna().sum()
        assert agg["demand"].sum() == valid


class TestBuildBaselineTable:
    def test_output_columns(self, ref_df):
        agg = aggregate_ticks(ref_df)
        baseline = build_baseline_table(agg)
        expected = {"zone_id", "hour_of_day", "day_of_week", "mean_demand", "std_demand"}
        assert set(baseline.columns) == expected

    def test_no_negative_std(self, ref_df):
        agg = aggregate_ticks(ref_df)
        baseline = build_baseline_table(agg)
        assert (baseline["std_demand"] >= 0).all()


class TestBuildReplayTable:
    def test_filters_to_active_zones(self, replay_df):
        agg = aggregate_ticks(replay_df)
        active = [1, 2, 3]
        replay_table = build_replay_table(agg, active)
        assert set(replay_table["zone_id"].unique()).issubset(set(active))

    def test_sorted_by_tick_start(self, replay_df):
        agg = aggregate_ticks(replay_df)
        active = [1, 2, 3]
        replay_table = build_replay_table(agg, active)
        assert replay_table["tick_start"].is_monotonic_increasing


class TestCrossCheckReplay:
    def test_passes_on_correct_data(self, replay_df):
        agg = aggregate_ticks(replay_df)
        active = list(agg["zone_id"].unique()[:5])
        replay_table = build_replay_table(agg, active)
        assert cross_check_replay(replay_df, replay_table, active) is True


class TestComputeDecision:
    def test_need_when_high_demand(self):
        snap = ZoneSnapshot(zone_id=1, tick_id=0, recent_demand=[10, 12, 15],
                            baseline_mean=5.0, baseline_std=2.0)
        assert compute_decision(snap) == "NEED"

    def test_ok_when_normal_demand(self):
        snap = ZoneSnapshot(zone_id=1, tick_id=0, recent_demand=[3, 4, 5],
                            baseline_mean=5.0, baseline_std=2.0)
        assert compute_decision(snap) == "OK"

    def test_empty_demand_returns_fallback(self):
        snap = ZoneSnapshot(zone_id=1, tick_id=0, recent_demand=[],
                            baseline_mean=5.0, baseline_std=2.0)
        assert compute_decision(snap) == FIRST_TICK_FALLBACK

    def test_deterministic(self):
        snap = ZoneSnapshot(zone_id=1, tick_id=0, recent_demand=[10, 12],
                            baseline_mean=5.0, baseline_std=2.0)
        d1 = compute_decision(snap)
        d2 = compute_decision(snap)
        assert d1 == d2


class TestApplyFallback:
    def test_previous_policy_with_history(self):
        assert apply_fallback(FALLBACK_POLICY_PREVIOUS, "NEED") == "NEED"

    def test_previous_policy_without_history(self):
        assert apply_fallback(FALLBACK_POLICY_PREVIOUS, None) == FIRST_TICK_FALLBACK

    def test_unknown_policy_returns_ok(self):
        assert apply_fallback("unknown", "NEED") == FIRST_TICK_FALLBACK


class TestRunConfig:
    def test_round_trip(self):
        config = RunConfig(n_zones=10, slow_zone_fraction=0.5)
        d = config.to_dict()
        config2 = RunConfig.from_dict(d)
        assert config2.n_zones == 10
        assert config2.slow_zone_fraction == 0.5


class TestTickMetrics:
    def test_to_dict(self):
        m = TickMetrics(tick_id=0, mode="blocking", n_zones_completed=5,
                        total_tick_latency_s=1.234)
        d = m.to_dict()
        assert d["tick_id"] == 0
        assert d["mode"] == "blocking"


class TestArtifactWriters:
    def test_write_json(self, tmp_path):
        data = {"key": "value"}
        path = tmp_path / "out.json"
        write_json(data, path)
        assert json.loads(path.read_text()) == data

    def test_write_metrics_csv(self, tmp_path):
        metrics = [TickMetrics(tick_id=0, mode="blocking", total_tick_latency_s=1.0)]
        path = tmp_path / "metrics.csv"
        write_metrics_csv(metrics, path)
        df = pd.read_csv(path)
        assert len(df) == 1

    def test_write_tick_summary(self, tmp_path):
        metrics = [TickMetrics(tick_id=0, mode="blocking")]
        decisions = {0: {1: "OK", 2: "NEED"}}
        path = tmp_path / "summary.json"
        write_tick_summary(metrics, decisions, path)
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert "decisions" in data[0]

    def test_write_latency_log(self, tmp_path):
        metrics = [TickMetrics(tick_id=0, mode="blocking", per_zone_latency={1: 0.5, 2: 0.3})]
        path = tmp_path / "latency.json"
        write_latency_log(metrics, path)
        data = json.loads(path.read_text())
        assert len(data) == 2


class TestPrepareAssets:
    def test_produces_expected_files(self, prepared_dir):
        expected = ["baseline.parquet", "replay.parquet", "active_zones.json", "prep_meta.json"]
        for name in expected:
            assert (prepared_dir / name).exists(), f"Missing {name}"

    def test_active_zones_count(self, prepared_dir):
        with open(prepared_dir / "active_zones.json") as f:
            zones = json.load(f)
        assert len(zones) == 10


class TestZoneActor:
    def test_write_decision_idempotent(self, ray_session, prepared_dir):
        from run import ZoneActor
        replay = pd.read_parquet(prepared_dir / "replay.parquet")
        baseline = pd.read_parquet(prepared_dir / "baseline.parquet")
        with open(prepared_dir / "active_zones.json") as f:
            zones = json.load(f)
        z = zones[0]
        config = RunConfig(n_zones=10)
        actor = ZoneActor.remote(
            z,
            replay[replay["zone_id"] == z].reset_index(drop=True),
            baseline[baseline["zone_id"] == z].reset_index(drop=True),
            config,
        )
        ray.get(actor.activate_tick.remote(0))
        r1 = ray.get(actor.write_decision.remote(0, "OK"))
        r2 = ray.get(actor.write_decision.remote(0, "NEED"))
        assert r1 == WriteStatus.WRITTEN
        assert r2 == WriteStatus.DUPLICATE

    def test_report_decision_late(self, ray_session, prepared_dir):
        from run import ZoneActor
        replay = pd.read_parquet(prepared_dir / "replay.parquet")
        baseline = pd.read_parquet(prepared_dir / "baseline.parquet")
        with open(prepared_dir / "active_zones.json") as f:
            zones = json.load(f)
        z = zones[0]
        config = RunConfig(n_zones=10)
        actor = ZoneActor.remote(
            z,
            replay[replay["zone_id"] == z].reset_index(drop=True),
            baseline[baseline["zone_id"] == z].reset_index(drop=True),
            config,
        )
        # Close tick 0
        ray.get(actor.activate_tick.remote(0))
        ray.get(actor.write_decision.remote(0, "OK"))
        # Report for already-closed tick
        r = ray.get(actor.report_decision.remote(0, "NEED", 0.1))
        assert r == WriteStatus.LATE

    def test_finalize_tick_with_fallback(self, ray_session, prepared_dir):
        from run import ZoneActor
        replay = pd.read_parquet(prepared_dir / "replay.parquet")
        baseline = pd.read_parquet(prepared_dir / "baseline.parquet")
        with open(prepared_dir / "active_zones.json") as f:
            zones = json.load(f)
        z = zones[0]
        config = RunConfig(n_zones=10)
        actor = ZoneActor.remote(
            z,
            replay[replay["zone_id"] == z].reset_index(drop=True),
            baseline[baseline["zone_id"] == z].reset_index(drop=True),
            config,
        )
        # Activate tick 0 but don't report any decision
        ray.get(actor.activate_tick.remote(0))
        ray.get(actor.finalize_tick.remote(0, FALLBACK_POLICY_PREVIOUS))
        decisions = ray.get(actor.get_accepted_decisions.remote())
        assert decisions[0] == FIRST_TICK_FALLBACK  # No previous, so OK

    def test_get_snapshot(self, ray_session, prepared_dir):
        from run import ZoneActor
        replay = pd.read_parquet(prepared_dir / "replay.parquet")
        baseline = pd.read_parquet(prepared_dir / "baseline.parquet")
        with open(prepared_dir / "active_zones.json") as f:
            zones = json.load(f)
        z = zones[0]
        config = RunConfig(n_zones=10)
        actor = ZoneActor.remote(
            z,
            replay[replay["zone_id"] == z].reset_index(drop=True),
            baseline[baseline["zone_id"] == z].reset_index(drop=True),
            config,
        )
        ray.get(actor.activate_tick.remote(0))
        snap = ray.get(actor.get_snapshot.remote(0))
        assert snap["zone_id"] == z
        assert snap["tick_id"] == 0
        assert "recent_demand" in snap


class TestScoreZone:
    def test_returns_decision(self, ray_session):
        from run import score_zone
        snap = ZoneSnapshot(zone_id=1, tick_id=0, recent_demand=[10, 12],
                            baseline_mean=5.0, baseline_std=2.0)
        result = ray.get(score_zone.remote(snap, slow_sleep_s=0.0, mode="blocking"))
        assert result.decision in ("NEED", "OK")
        assert result.zone_id == 1
        assert result.task_latency_s >= 0.0


class TestBlockingRun:
    def test_produces_artifacts(self, ray_session, prepared_dir, tmp_path):
        from run import run_blocking
        config = RunConfig(n_zones=10, slow_zone_fraction=0.2, slow_zone_sleep_s=0.1)
        out = tmp_path / "output"
        run_blocking(prepared_dir, out, config)
        assert (out / "blocking" / "run_config.json").exists()
        assert (out / "blocking" / "metrics.csv").exists()
        assert (out / "blocking" / "tick_summary.json").exists()


class TestAsyncRun:
    def test_produces_artifacts(self, ray_session, prepared_dir, tmp_path):
        from run import run_async
        config = RunConfig(n_zones=10, slow_zone_fraction=0.2,
                           slow_zone_sleep_s=0.1, tick_timeout_s=5.0)
        out = tmp_path / "output"
        run_async(prepared_dir, out, config)
        assert (out / "async" / "run_config.json").exists()
        assert (out / "async" / "metrics.csv").exists()
