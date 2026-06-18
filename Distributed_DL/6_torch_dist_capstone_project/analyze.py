"""
Post-hoc analysis of profiler traces and metrics from a training run.
"""

import argparse
import csv
import json
import sys

from dataclasses import dataclass
from pathlib import Path

from prettytable import PrettyTable

from src.common import CONFIG_FILENAME, METRICS_FILENAME, TRACES_DIR
from src.logger import g_logger


TraceSummary = dict[str, dict[str, float]]
RankSummaries = dict[int, TraceSummary]


COMPUTE_SPANS = {
    "prepare_views",
    "stage0_forward",
    "stage1_forward",
    "loss_calculation",
    "stage0_backward",
    "stage1_backward"
}

COMM_SPANS = {
    "send_boundary",
    "recv_boundary",
    "gather_embeddings",
    "send_boundary_grad",
    "recv_boundary_grad",
    "grad_sync_stage0",
    "grad_sync_stage1"
}

ALL_SPANS = COMPUTE_SPANS | COMM_SPANS | {"optimizer_step"}

# Config keys to display in the run summary
CONFIG_DISPLAY_KEYS = [
    "local_batch_size",
    "global_batch_size",
    "split_layer",
    "num_steps",
    "images_per_sec"
]


@dataclass
class TimeBreakdown:
    """
    Compute vs communication vs optimizer percentage breakdown.
    """
    compute_pct: float  # Percentage of time spent in compute spans (e.g. stage0_forward, loss_calculation).
    comm_pct: float  # Percentage of time spent in communication spans (e.g. send_boundary, recv_boundary).
    optimizer_pct: float  # Percentage of time spent in optimizer step spans (optimizer_step).


class TraceAnalyzer:
    """
    Loads per-rank Chrome trace files, aggregates span durations, and reports compute vs communication breakdown and
    stage imbalance.
    """

    def __init__(self, run_dir: Path) -> None:
        """
        Initialize with the run directory containing the traces/ subdirectory and optional config and metrics files.

        Args:
            run_dir (Path): Path to the run output directory.
        """
        self.run_dir = run_dir
        self.trace_dir = run_dir / TRACES_DIR
        self.summaries: RankSummaries = {}  # Map each rank to its span summary

    @staticmethod
    def _load_trace(path: Path) -> list[dict]:
        """
        Read a Chrome trace JSON and keep only our annotated complete-event spans.

        Args:
            path (Path): Path to the trace JSON file.

        Returns:
            list[dict]: List of trace events. Each event is a dict with keys like "name", "dur" (duration in µs), and
                "ph" (event type, e.g. "X" for complete events).
        """
        with open(path) as f:
            data = json.load(f)
        return [e for e in data["traceEvents"] if e.get("name") in ALL_SPANS and e.get("ph") == "X"]

    @staticmethod
    def _summarize_spans(events: list[dict]) -> dict[str, dict]:
        """
        Aggregate span durations (µs) by name → count / total / mean.

        Args:
            events (list[dict]): List of trace events with keys like "name" and "dur" (duration in µs).

        Returns:
            dict[str, dict]: Map from span name to summary dict with keys: count, total_us, mean_us.
        """
        spans: dict[str, list[float]] = {}
        for e in events:
            spans.setdefault(e["name"], []).append(e["dur"])
        return {
            name: {"count": len(durations), "total_us": sum(durations), "mean_us": sum(durations) / len(durations)}
            for name, durations in sorted(spans.items())
        }

    def load(self) -> None:
        """
        Load all per-rank traces and build per-rank span summaries.
        """
        trace_files = sorted(self.trace_dir.glob("rank*.json"))
        for tf in trace_files:
            rank = int(tf.stem.replace("rank", ""))
            events = self._load_trace(tf)
            self.summaries[rank] = self._summarize_spans(events)
        g_logger.info(f"Loaded traces for {len(self.summaries)} ranks")

    def _log_config(self) -> None:
        """
        Log run configuration and mean loss if available.
        """
        config_path = self.run_dir / CONFIG_FILENAME
        csv_path = self.run_dir / METRICS_FILENAME

        table = PrettyTable()
        table.title = "Run configuration"
        table.field_names = ["key", "value"]
        table.align["key"] = "l"
        table.align["value"] = "l"
        table.header = False

        # Add run config rows
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            for key in CONFIG_DISPLAY_KEYS:
                if key in config:
                    table.add_row([key, config[key]])

        # Append mean loss
        if csv_path.exists():
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            losses = [float(r["loss"]) for r in rows if r["loss"]]
            if losses:
                table.add_row(["mean_loss", f"{sum(losses) / len(losses):.4f}"])

        if table.rows:
            g_logger.info("\n" + table.get_string())

    def _log_span_tables(self) -> None:
        """
        Log per-rank span summary tables.
        """
        for rank, summary in self.summaries.items():
            table = PrettyTable()
            table.title = f"Rank {rank} span summary"
            table.field_names = ["span", "count", "total_ms", "mean_ms"]
            for col in table.field_names:
                table.align[col] = "l"
            for name, stats in summary.items():
                table.add_row([name, stats["count"], f"{stats['total_us'] / 1000:.1f}", f"{stats['mean_us'] / 1000:.1f}"])
            g_logger.info("\n" + table.get_string())

    @staticmethod
    def compute_breakdown(summary: dict[str, dict]) -> TimeBreakdown:
        """
        Return compute vs communication vs optimizer percentage breakdown based on the span summary.
        """
        compute_us = sum(s["total_us"] for n, s in summary.items() if n in COMPUTE_SPANS)
        comm_us = sum(s["total_us"] for n, s in summary.items() if n in COMM_SPANS)
        opt_us = sum(s["total_us"] for n, s in summary.items() if n == "optimizer_step")
        total = compute_us + comm_us + opt_us
        if total == 0:
            return TimeBreakdown(compute_pct=0, comm_pct=0, optimizer_pct=0)
        return TimeBreakdown(
            compute_pct=round(100 * compute_us / total, 1),
            comm_pct=round(100 * comm_us / total, 1),
            optimizer_pct=round(100 * opt_us / total, 1),
        )

    def _log_time_breakdown(self) -> None:
        """
        Log compute vs communication vs optimizer time breakdown per rank.
        """
        table = PrettyTable()
        table.title = "Time breakdown"
        table.field_names = ["rank", "compute %", "comm %", "opt %"]
        for col in table.field_names:
            table.align[col] = "l"
        for rank, summary in self.summaries.items():
            bd = self.compute_breakdown(summary)
            table.add_row([rank, f"{bd.compute_pct:.1f}", f"{bd.comm_pct:.1f}", f"{bd.optimizer_pct:.1f}"])
        g_logger.info("\n" + table.get_string())

    def calc_mean_span_ms(self, ranks: list[int], span_names: set[str]) -> float:
        """
        Average mean duration (ms) of the given span names across the given ranks.
        """
        if not ranks:
            return 0.0
        total = 0.0
        for r in ranks:
            for name, stats in self.summaries[r].items():
                if name in span_names:
                    total += stats["mean_us"]
        return total / len(ranks) / 1000

    def _log_stage_imbalance(self) -> None:
        """
        Compare the stage-0 compute region (even ranks) against the stage-1-plus-loss region (odd ranks).

        Per the design doc, the balance metric compares ``stage0_forward`` against ``stage1_forward + loss_calculation``.
        Backward spans are deliberately excluded so the two sides are measured symmetrically (the odd-rank backward is
        reported separately in the per-rank span tables, not folded into one side of this ratio).
        """
        even_ranks, odd_ranks = [], []
        for rank in self.summaries.keys():
            if rank % 2 == 0:
                even_ranks.append(rank)
            else:
                odd_ranks.append(rank)

        stage0_ms = self.calc_mean_span_ms(even_ranks, {"stage0_forward"})
        stage1_ms = self.calc_mean_span_ms(odd_ranks, {"stage1_forward", "loss_calculation"})

        table = PrettyTable()
        table.title = "Stage imbalance"
        table.field_names = ["metric", "value"]
        table.align["metric"] = "l"
        table.align["value"] = "l"
        table.header = False
        table.add_row(["Stage 0 (even)", f"{stage0_ms:.1f} ms"])
        table.add_row(["Stage 1 (odd)", f"{stage1_ms:.1f} ms"])
        if stage0_ms > 0:
            table.add_row(["Ratio", f"{stage1_ms / stage0_ms:.2f}x"])
        g_logger.info("\n" + table.get_string())

    def report(self) -> None:
        """
        Print full analysis: config, per-rank spans, time breakdown, stage imbalance.
        """
        self._log_config()
        self._log_span_tables()
        self._log_time_breakdown()
        self._log_stage_imbalance()


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser.

    Returns:
        argparse.ArgumentParser: CLI parser for the analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Analyze profiler traces and metrics from a training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=str, help="Path to the run output directory.")
    return parser


def main() -> None:
    """
    Entry point: parse args, load traces, print analysis.
    """
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)

    # Check traces directory exists
    if not (run_dir / TRACES_DIR).exists():
        g_logger.error(f"No traces directory found at {run_dir / TRACES_DIR}")
        sys.exit(1)

    # Load traces and report
    analyzer = TraceAnalyzer(run_dir)
    analyzer.load()
    analyzer.report()


if __name__ == "__main__":
    main()
