"""
Integration tests for the distributed training system.
"""

import math

from tests.helpers import (
    DEFAULT_NPROC,
    EVEN_RANK_SPANS,
    ODD_RANK_SPANS,
    REQUIRED_CONFIG_KEYS,
    REQUIRED_CSV_COLUMNS,
    RunResult,
    load_losses_for_rank,
    load_metrics_rows,
    load_run_config,
    load_trace_span_names,
)
from src.common import CONFIG_FILENAME, METRICS_FILENAME


def test_profiled_run_creates_per_rank_traces(sync_run: RunResult) -> None:
    """
    Profiled training produces one trace JSON per rank.
    """
    traces_dir = sync_run.run_dir / "traces"
    assert all((traces_dir / f"rank{r}.json").exists() for r in range(DEFAULT_NPROC)), "rank trace missing"


def test_config_reports_batch_sizes_and_throughput(sync_run: RunResult) -> None:
    """
    Run config file contains local_batch_size, global_batch_size, and images_per_sec.
    """
    config = load_run_config(sync_run.run_dir)
    missing = REQUIRED_CONFIG_KEYS - set(config.keys())
    assert not missing, f"{CONFIG_FILENAME} missing keys: {missing}"
    assert config["images_per_sec"] > 0, f"images_per_sec must be positive, got {config['images_per_sec']}"


def test_global_batch_equals_local_times_num_pairs(sync_run: RunResult) -> None:
    """
    Global batch size equals local batch size times world size divided by 2.
    """
    config = load_run_config(sync_run.run_dir)
    num_pairs = DEFAULT_NPROC // 2
    expected = config["local_batch_size"] * num_pairs
    actual = config["global_batch_size"]
    assert actual == expected, f"global_batch_size {actual} != {config['local_batch_size']} * {num_pairs}"


def test_metrics_csv_covers_all_ranks_and_steps(sync_run: RunResult) -> None:
    """
    metrics.csv has required columns and rows for every rank.
    """
    rows = load_metrics_rows(sync_run.run_dir)
    columns = set(rows[0].keys())
    missing = REQUIRED_CSV_COLUMNS - columns
    assert not missing, f"{METRICS_FILENAME} missing columns: {missing}"
    ranks_seen = {int(r["rank"]) for r in rows}
    assert ranks_seen == set(range(DEFAULT_NPROC)), f"expected all {DEFAULT_NPROC} ranks, got {ranks_seen}"


def test_losses_remain_finite_across_steps(sync_run: RunResult) -> None:
    """
    Loss values on odd ranks are finite and positive.
    """
    losses = load_losses_for_rank(sync_run.run_dir, rank=1)
    assert len(losses) > 0, "no loss values recorded for rank 1"
    assert all(math.isfinite(l) and l > 0 for l in losses), f"non-finite or non-positive loss found in {losses}"


def test_even_rank_traces_contain_required_spans(sync_run: RunResult) -> None:
    """
    Even-rank traces include all stage-0, send/recv, and sync spans from the design doc.
    """
    actual = load_trace_span_names(sync_run.run_dir, rank=0)
    missing = EVEN_RANK_SPANS - actual
    assert not missing, f"even rank missing spans: {missing}"


def test_odd_rank_traces_contain_required_spans(sync_run: RunResult) -> None:
    """
    Odd-rank traces include all stage-1, loss, gather, and sync spans from the design doc.
    """
    actual = load_trace_span_names(sync_run.run_dir, rank=1)
    missing = ODD_RANK_SPANS - actual
    assert not missing, f"odd rank missing spans: {missing}"


def test_comm_structure_printed_at_startup(sync_run: RunResult) -> None:
    """
    Training logs the communication group structure at startup.
    """
    output = sync_run.stdout + sync_run.stderr
    expected = ["world_group", "pair_group", "stage0_group", "stage1_group"]
    missing = [l for l in expected if l not in output]
    assert not missing, f"communication structure missing {missing} in startup output"


def test_overlap_mode_produces_valid_results(overlap_run: RunResult) -> None:
    """
    Stretch A: overlap mode produces per-rank traces and finite losses.
    """
    traces_dir = overlap_run.run_dir / "traces"
    assert all((traces_dir / f"rank{r}.json").exists() for r in range(DEFAULT_NPROC)), "overlap rank trace missing"
    losses = load_losses_for_rank(overlap_run.run_dir, rank=1)
    assert len(losses) > 0, "overlap mode produced no loss values"
    assert all(math.isfinite(l) and l > 0 for l in losses), "overlap losses contain non-finite values"
