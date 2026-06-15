"""
Create a durable manual sweep summary and diagnosis from completed profiled runs.

Typical usage after scripts/sweep.sh:
    python summarize_sweep.py --output-dir output --pattern "sweep_bs*"
"""

import argparse
import csv
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from analyze import TraceAnalyzer
from src.common import CONFIG_FILENAME, TRACES_DIR
from src.logger import g_logger


@dataclass
class SweepRow:
    """
    Trace-derived summary values for one completed sweep run.

    Each row is written to the manual sweep CSV and also feeds the Markdown diagnosis. Times are stored in milliseconds,
    percentages are stored as 0-100 values, and stage_imbalance is the odd-stage compute time divided by the even-stage
    compute time.
    """

    run_name: str
    local_batch_size: int
    global_batch_size: int
    images_per_sec: float
    split_layer: str
    comm_pct: float
    stage0_ms: float
    stage1_loss_ms: float
    stage_imbalance: float
    gather_ms: float
    loss_ms: float


def mean(values: list[float]) -> float:
    """
    Return the arithmetic mean of a list, or 0.0 for an empty list.

    Args:
        values (list[float]): Numeric values to average.

    Returns:
        float: Arithmetic mean, or 0.0 when values is empty.
    """
    return sum(values) / len(values) if values else 0.0


def summarize_run(run_dir: Path) -> Optional[SweepRow]:
    """
    Load one run directory and convert its config and traces into a SweepRow.

    Returns None when the directory is not a completed profiled run, which lets callers glob broadly over an output
    directory without failing on unrelated files or partial runs.

    Args:
        run_dir (Path): Directory containing run_config.json and traces/rank*.json.

    Returns:
        Optional[SweepRow]: Trace-derived summary row, or None for an incomplete/non-profiled run directory.
    """
    config_path = run_dir / CONFIG_FILENAME
    trace_dir = run_dir / TRACES_DIR
    if not config_path.exists() or not trace_dir.exists():
        return None

    with open(run_dir / CONFIG_FILENAME) as f:
        config = json.load(f)

    analyzer = TraceAnalyzer(run_dir)
    analyzer.load()

    even_ranks = [r for r in analyzer.summaries if r % 2 == 0]
    odd_ranks = [r for r in analyzer.summaries if r % 2 == 1]
    comm_pct = mean([analyzer.compute_breakdown(s).comm_pct for s in analyzer.summaries.values()])
    stage0_ms = analyzer.calc_mean_span_ms(even_ranks, {"stage0_forward", "stage0_backward"})
    stage1_loss_ms = analyzer.calc_mean_span_ms(odd_ranks, {"stage1_forward", "loss_calculation"})
    gather_ms = analyzer.calc_mean_span_ms(odd_ranks, {"gather_embeddings"})
    loss_ms = analyzer.calc_mean_span_ms(odd_ranks, {"loss_calculation"})
    imbalance = stage1_loss_ms / stage0_ms if stage0_ms > 0 else 0.0

    return SweepRow(
        run_name=run_dir.name,
        local_batch_size=int(config.get("local_batch_size", 0)),
        global_batch_size=int(config.get("global_batch_size", 0)),
        images_per_sec=float(config.get("images_per_sec", 0.0)),
        split_layer=str(config.get("split_layer", "")),
        comm_pct=round(comm_pct, 2),
        stage0_ms=round(stage0_ms, 2),
        stage1_loss_ms=round(stage1_loss_ms, 2),
        stage_imbalance=round(imbalance, 3),
        gather_ms=round(gather_ms, 2),
        loss_ms=round(loss_ms, 2),
    )


def write_csv(rows: list[SweepRow], path: Path) -> None:
    """
    Write the collected sweep rows to a machine-readable CSV summary.

    Args:
        rows (list[SweepRow]): Completed run summaries to persist.
        path (Path): Destination CSV path.
    """
    fieldnames = list(SweepRow.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def diagnose(rows: list[SweepRow]) -> str:
    """
    Build a short Markdown diagnosis explaining the throughput winner and trace evidence.

    The diagnosis intentionally treats images/s as the primary metric and uses communication percentage, stage
    imbalance, gather time, and loss time as supporting systems evidence, matching the design document.

    Args:
        rows (list[SweepRow]): Completed run summaries from one manual sweep.

    Returns:
        str: Markdown diagnosis document.
    """
    best = max(rows, key=lambda r: r.images_per_sec)
    ordered = sorted(rows, key=lambda r: r.local_batch_size)
    after_best = [r for r in ordered if r.local_batch_size > best.local_batch_size]

    lines = [
        "# Manual Batch-Size Sweep Diagnosis",
        "",
        "## Summary Table",
        "",
        "| run | local batch | global batch | images/s | comm % | stage1/stage0 | gather ms | loss ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        marker = " (best)" if row.run_name == best.run_name else ""
        lines.append(
            f"| {row.run_name}{marker} | {row.local_batch_size} | {row.global_batch_size} | "
            f"{row.images_per_sec:.2f} | {row.comm_pct:.2f} | {row.stage_imbalance:.3f} | "
            f"{row.gather_ms:.2f} | {row.loss_ms:.2f} |"
        )

    lines.extend([
        "",
        "## Tuning Decision",
        "",
        f"The best observed configuration is `{best.run_name}` with local batch size "
        f"{best.local_batch_size}, global batch size {best.global_batch_size}, and "
        f"{best.images_per_sec:.2f} images/s.",
        "",
        "The decision is based primarily on global throughput. The secondary evidence is the trace-derived "
        "communication percentage, the stage imbalance ratio, and the odd-rank `gather_embeddings` and "
        "`loss_calculation` spans, which capture the contrastive-loss overhead.",
    ])

    if after_best:
        worst_after = min(after_best, key=lambda r: r.images_per_sec)
        lines.extend([
            "",
            "## Why Larger Batches Stopped Helping",
            "",
            f"At least one larger batch was slower than the best run. For example, `{worst_after.run_name}` "
            f"reached {worst_after.images_per_sec:.2f} images/s with communication at {worst_after.comm_pct:.2f}% "
            f"and an odd/even compute ratio of {worst_after.stage_imbalance:.3f}x. This suggests the larger "
            "batch increased communication or odd-rank loss-side work more than it improved the local-work-to-sync ratio.",
        ])
    else:
        lines.extend([
            "",
            "## Larger-Batch Behavior",
            "",
            "Throughput did not decline after the best run in this sweep range. A wider sweep should continue until "
            "`images/s` flattens or degrades and then inspect whether communication percentage, waiting, or "
            "odd-rank loss-side work is responsible.",
        ])

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for sweep summarization.

    Returns:
        argparse.ArgumentParser: Configured CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Write manual_sweep_summary.csv and diagnosis_summary.md for completed profiled sweep runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Directory containing sweep run subdirectories.")
    parser.add_argument("--pattern", type=str, default="sweep_bs*", help="Glob pattern for sweep run directories.")
    parser.add_argument("--summary-prefix", type=str, default="manual_sweep", help="Prefix for generated summary artifacts.")
    return parser


def main() -> None:
    """
    Entry point: discover matching runs, write the CSV summary, and write the Markdown diagnosis.
    """
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    rows = []
    for run_dir in sorted(output_dir.glob(args.pattern)):
        if run_dir.is_dir():
            row = summarize_run(run_dir)
            if row is not None:
                rows.append(row)

    if not rows:
        raise SystemExit(f"No completed profiled runs matched {output_dir / args.pattern}")

    rows.sort(key=lambda r: (r.local_batch_size, r.run_name))
    csv_path = output_dir / f"{args.summary_prefix}_summary.csv"
    md_path = output_dir / "diagnosis_summary.md"
    write_csv(rows, csv_path)
    md_path.write_text(diagnose(rows))
    g_logger.info(f"Wrote {csv_path}")
    g_logger.info(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
