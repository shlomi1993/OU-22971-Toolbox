"""
Integration tests for trace analysis.
"""

from analyze import TraceAnalyzer
from tests.helpers import DEFAULT_NPROC, RunResult, load_trace_span_names, run_analyze_cli


def test_analyzer_loads_traces_for_all_ranks(sync_run: RunResult) -> None:
    """
    TraceAnalyzer loads and summarizes traces from every rank.
    """
    analyzer = TraceAnalyzer(sync_run.run_dir)
    analyzer.load()
    b_summaries = len(analyzer.summaries)
    assert b_summaries == DEFAULT_NPROC, f"expected {DEFAULT_NPROC} rank summaries, got {b_summaries}"


def test_time_breakdown_percentages_sum_near_100(sync_run: RunResult) -> None:
    """
    Compute + communication + optimizer percentages sum to approximately 100.
    """
    analyzer = TraceAnalyzer(sync_run.run_dir)
    analyzer.load()
    for rank, summary in analyzer.summaries.items():
        breakdown = analyzer.compute_breakdown(summary)
        total = breakdown.compute_pct + breakdown.comm_pct + breakdown.optimizer_pct
        assert 95 <= total <= 105, f"rank {rank}: breakdown sums to {total}%, expected ~100%"


def test_stage_imbalance_is_computed(sync_run: RunResult) -> None:
    """
    Stage imbalance ratio between even and odd ranks is positive and finite.
    """
    analyzer = TraceAnalyzer(sync_run.run_dir)
    analyzer.load()
    even = [r for r in analyzer.summaries if r % 2 == 0]
    odd = [r for r in analyzer.summaries if r % 2 != 0]
    stage0_ms = analyzer.calc_mean_span_ms(even, {"stage0_forward", "stage0_backward"})
    stage1_ms = analyzer.calc_mean_span_ms(odd, {"stage1_forward", "loss_calculation"})
    assert stage0_ms > 0, "stage0 compute time is zero"
    assert stage1_ms > 0, "stage1 compute time is zero"
    ratio = stage1_ms / stage0_ms
    assert ratio > 0, f"imbalance ratio must be positive, got {ratio}"


def test_odd_ranks_carry_contrastive_loss_overhead(sync_run: RunResult) -> None:
    """
    gather_embeddings and loss_calculation appear only on odd ranks, not even ranks.
    """
    contrastive_spans = {"gather_embeddings", "loss_calculation"}
    odd_spans = load_trace_span_names(sync_run.run_dir, rank=1)
    even_spans = load_trace_span_names(sync_run.run_dir, rank=0)
    assert contrastive_spans <= odd_spans, f"odd rank missing contrastive spans: {contrastive_spans - odd_spans}"
    overlap = contrastive_spans & even_spans
    assert not overlap, f"even rank should not have contrastive spans: {overlap}"


def test_analyze_cli_reports_breakdown_and_imbalance(sync_run: RunResult) -> None:
    """
    analyze.py CLI outputs time breakdown and stage imbalance sections.
    """
    result = run_analyze_cli(sync_run.run_dir)
    assert result.returncode == 0, f"analyze.py exited with {result.returncode}"
    output = result.stdout + result.stderr
    assert "Time breakdown" in output, "analyze output missing 'Time breakdown'"
    assert "Stage imbalance" in output, "analyze output missing 'Stage imbalance'"
