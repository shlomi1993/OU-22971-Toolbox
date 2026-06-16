"""
Load-balancing controller (Stretch B).

Automates batch-size and split-layer sweeps. For each configuration it launches a profiled training run, analyzes the
exported traces, and records images/s alongside the design-doc heuristics: the stage-0 (even) versus stage-1-plus-loss
(odd) compute times and the fraction of each step spent in activation transfer, embedding gather, other communication,
and waiting. The best configuration is chosen primarily by images/s, with communication-heaviness and stage imbalance
as secondary tie-breakers.
"""

import argparse
import json
import subprocess
import sys

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from analyze import TraceAnalyzer
from src.common import CONFIG_FILENAME, DEFAULT_DATASET_SIZE, DEFAULT_NUM_STEPS, DEFAULT_OUTPUT_DIR, DEFAULT_SEED, SPLIT_CHOICES, mean_or_zero
from src.logger import g_logger


DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64]
DEFAULT_SPLIT_LAYERS = ["layer1", "layer2", "layer3"]
DEFAULT_NPROC = 4
CONTROLLER_LOG = "controller_log.json"

# Prefer shard boundaries that leave the odd ranks somewhat lighter, since they also own the contrastive-loss path.
ODD_LIGHTER_TARGET = 0.8

# Trace span groups for the design-doc per-step breakdown.
ACTIVATION_TRANSFER_SPANS = {"send_boundary", "send_boundary_grad"}
WAITING_SPANS = {"recv_boundary", "recv_boundary_grad"}
GATHER_SPANS = {"gather_embeddings"}
OTHER_COMM_SPANS = {"grad_sync_stage0", "grad_sync_stage1"}


@dataclass
class RunResult:
    """
    Throughput and trace-derived metrics for a single sweep run.
    """
    run_name: str  # Run name (subdirectory under output_dir)
    local_batch_size: int  # Local batch size used in the run
    split_layer: str  # Split layer name
    images_per_sec: float  # Global throughput (primary selection metric)
    comm_pct: float  # Communication percentage of total step time
    activation_transfer_pct: float  # Boundary activation/gradient send percentage
    gather_pct: float  # Embedding all_gather percentage
    other_comm_pct: float  # Gradient-sync percentage
    waiting_pct: float  # Blocking-receive percentage (waiting proxy)
    stage0_ms: float  # Mean even-rank stage-0 compute time
    stage1_loss_ms: float  # Mean odd-rank stage-1 plus loss time
    stage_imbalance: float  # Stage imbalance ratio (odd/even compute time)


class Controller:
    """
    Sweeps batch sizes and split layers, launches training runs, analyzes traces, and picks the best configuration.
    """

    def __init__(self, batch_sizes: list[int] = None, split_layers: list[str] = None, num_steps: int = DEFAULT_NUM_STEPS,
                 dataset_size: int = DEFAULT_DATASET_SIZE, seed: int = DEFAULT_SEED, output_dir: str = DEFAULT_OUTPUT_DIR,
                 nproc: int = DEFAULT_NPROC) -> None:
        """
        Initialize the controller with sweep parameters and output configuration.

        Args:
            batch_sizes (list[int], optional): Local batch sizes to sweep. Defaults to DEFAULT_BATCH_SIZES.
            split_layers (list[str], optional): Split layer names to sweep. Defaults to DEFAULT_SPLIT_LAYERS.
            num_steps (int, optional): Training steps per run. Defaults to DEFAULT_NUM_STEPS.
            dataset_size (int, optional): Synthetic dataset size. Defaults to DEFAULT_DATASET_SIZE.
            seed (int, optional): Random seed for reproducibility. Defaults to DEFAULT_SEED.
            output_dir (str, optional): Base output directory for all runs. Defaults to DEFAULT_OUTPUT_DIR.
            nproc (int, optional): Number of processes (ranks) to launch. Defaults to DEFAULT_NPROC.
        """
        self.batch_sizes = batch_sizes or list(DEFAULT_BATCH_SIZES)
        self.split_layers = split_layers or list(DEFAULT_SPLIT_LAYERS)
        self.num_steps = num_steps
        self.dataset_size = dataset_size
        self.seed = seed
        self.output_dir = output_dir
        self.nproc = nproc
        self.results: list[RunResult] = []

    def _launch_training(self, batch_size: int, split_layer: str) -> str:
        """
        Launch a single profiled training run via torchrun and return the run name.

        Args:
            batch_size (int): Local batch size for this run.
            split_layer (str): Model split point name.

        Returns:
            str: Run name, or empty string if the run failed.
        """
        run_name = f"ctrl_bs{batch_size}_{split_layer}"
        cmd = [
            "torchrun", "--standalone", f"--nproc_per_node={self.nproc}",
            "train.py",
            "--local-batch-size", str(batch_size),
            "--split-layer", split_layer,
            "--num-steps", str(self.num_steps),
            "--dataset-size", str(self.dataset_size),
            "--seed", str(self.seed),
            "--profile",
            "--output-dir", self.output_dir,
            "--run-name", run_name,
        ]
        g_logger.info(f"Launching bs={batch_size}, split={split_layer}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            g_logger.error(f"Run failed (exit {result.returncode})")
            return ""
        return run_name

    @staticmethod
    def _span_pct(summary: dict[str, dict], span_names: set[str]) -> float:
        """
        Compute the percentage of annotated trace time spent in the requested spans.

        Args:
            summary (dict[str, dict]): Span summary for one rank.
            span_names (set[str]): Span names to include in the numerator.

        Returns:
            float: Percentage of annotated time spent in the requested spans.
        """
        total_us = sum(stats["total_us"] for stats in summary.values())
        span_us = sum(stats["total_us"] for name, stats in summary.items() if name in span_names)
        return 100 * span_us / total_us if total_us > 0 else 0.0

    @classmethod
    def _analyze_traces(cls, run_dir: Path) -> dict[str, float]:
        """
        Compute communication, waiting, and stage-balance metrics from rank traces.

        The returned keys match the trace-derived fields of RunResult, so the result can be spread directly into the
        dataclass constructor.

        Args:
            run_dir (Path): Directory containing trace files for a single run.

        Returns:
            dict[str, float]: Trace-derived percentages and stage timing metrics.
        """
        analyzer = TraceAnalyzer(run_dir)
        analyzer.load()

        summaries = list(analyzer.summaries.values())
        even_ranks = [rank for rank in analyzer.summaries if rank % 2 == 0]
        odd_ranks = [rank for rank in analyzer.summaries if rank % 2 == 1]
        stage0_ms = analyzer.calc_mean_span_ms(even_ranks, {"stage0_forward", "stage0_backward"})
        stage1_loss_ms = analyzer.calc_mean_span_ms(odd_ranks, {"stage1_forward", "loss_calculation"})

        return {
            "comm_pct": round(mean_or_zero([analyzer.compute_breakdown(s).comm_pct for s in summaries]), 2),
            "activation_transfer_pct": round(mean_or_zero([cls._span_pct(s, ACTIVATION_TRANSFER_SPANS) for s in summaries]), 2),
            "gather_pct": round(mean_or_zero([cls._span_pct(s, GATHER_SPANS) for s in summaries]), 2),
            "other_comm_pct": round(mean_or_zero([cls._span_pct(s, OTHER_COMM_SPANS) for s in summaries]), 2),
            "waiting_pct": round(mean_or_zero([cls._span_pct(s, WAITING_SPANS) for s in summaries]), 2),
            "stage0_ms": round(stage0_ms, 2),
            "stage1_loss_ms": round(stage1_loss_ms, 2),
            "stage_imbalance": round(stage1_loss_ms / stage0_ms if stage0_ms > 0 else 1.0, 4),
        }

    @staticmethod
    def _selection_key(result: RunResult) -> tuple[float, float, float]:
        """
        Rank runs primarily by images/s, then by the design-doc secondary heuristics.

        Ties are broken toward lower communication-plus-waiting and a stage imbalance near ODD_LIGHTER_TARGET, which
        keeps the odd ranks (which also own the contrastive loss) somewhat lighter than the even ranks.

        Args:
            result (RunResult): Completed run metrics.

        Returns:
            tuple[float, float, float]: Sort key where higher is better.
        """
        comm_heavy = result.comm_pct + result.waiting_pct
        imbalance_distance = abs(result.stage_imbalance - ODD_LIGHTER_TARGET)
        return (result.images_per_sec, -comm_heavy, -imbalance_distance)

    def _extract_result(self, run_name: str) -> Optional[RunResult]:
        """
        Extract metrics from a completed run's output artifacts.

        Args:
            run_name (str): Name of the run subdirectory.

        Returns:
            Optional[RunResult]: Extracted metrics, or None if the run config is missing.
        """
        run_dir = Path(self.output_dir) / run_name
        config_path = run_dir / CONFIG_FILENAME
        if not config_path.exists():
            g_logger.warning(f"No config file in {run_dir}")
            return None

        with open(config_path) as f:
            run_config: dict = json.load(f)

        return RunResult(
            run_name=run_name,
            local_batch_size=run_config.get("local_batch_size", 0),
            split_layer=run_config.get("split_layer", ""),
            images_per_sec=run_config.get("images_per_sec", 0.0),
            **self._analyze_traces(run_dir),
        )

    def _log_summary(self, best: RunResult) -> None:
        """
        Log a formatted summary table and the chosen best configuration.

        Args:
            best (RunResult): The best run result to highlight in the table.
        """
        header = (f"  {'batch_size':>10s} {'split_layer':>12s} {'images/s':>10s} {'comm%':>7s} "
                  f"{'wait%':>7s} {'gather%':>8s} {'imbalance':>10s}")
        separator = f"  {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 10}"

        lines = ["Sweep results", header, separator]
        for r in sorted(self.results, key=self._selection_key, reverse=True):
            marker = " <-- best" if r.run_name == best.run_name else ""
            lines.append(
                f"  {r.local_batch_size:>10d} {r.split_layer:>12s} {r.images_per_sec:>10.1f} "
                f"{r.comm_pct:>6.1f}% {r.waiting_pct:>6.1f}% {r.gather_pct:>7.1f}% "
                f"{r.stage_imbalance:>10.2f}x{marker}"
            )
        g_logger.info("\n".join(lines))

    def _save_decision_log(self, best: RunResult) -> None:
        """
        Write the controller decision log to JSON.

        Args:
            best (RunResult): The best run result to record.
        """
        log = {
            "controller_config": {
                "batch_sizes": self.batch_sizes,
                "split_layers": self.split_layers,
                "num_steps": self.num_steps,
                "dataset_size": self.dataset_size,
                "nproc": self.nproc,
            },
            "runs": [asdict(r) for r in self.results],
            "best": {
                **asdict(best),
                "selection_rule": "maximize images_per_sec; break ties with lower communication/waiting and stage imbalance near 0.8x",
            },
        }

        out_path = Path(self.output_dir) / CONTROLLER_LOG
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(log, f, indent=2)
        g_logger.info(f"Decision log saved to {out_path}")

    def run(self) -> None:
        """
        Execute the full sweep: launch runs, collect results, pick the best, and save the decision log.
        """
        g_logger.info(f"Sweep: {len(self.batch_sizes)} batch sizes x {len(self.split_layers)} split layers")

        for split_layer in self.split_layers:
            for batch_size in self.batch_sizes:
                run_name = self._launch_training(batch_size, split_layer)
                if not run_name:
                    continue

                result = self._extract_result(run_name)
                if result is None:
                    continue

                self.results.append(result)
                g_logger.info(
                    f"  => {result.images_per_sec:.1f} img/s, comm={result.comm_pct:.1f}%, "
                    f"wait={result.waiting_pct:.1f}%, gather={result.gather_pct:.1f}%, "
                    f"imbalance={result.stage_imbalance:.2f}x"
                )

        if not self.results:
            g_logger.error("No successful runs. Exiting.")
            sys.exit(1)

        best = max(self.results, key=self._selection_key)
        self._log_summary(best)
        self._save_decision_log(best)


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for the controller.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Load-balancing controller: sweep batch sizes and split layers, pick best config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES, help="Batch sizes to sweep.")
    parser.add_argument("--split-layers", type=str, nargs="+", default=DEFAULT_SPLIT_LAYERS, choices=SPLIT_CHOICES, help="Split layers to sweep.")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS, help="Training steps per run.")
    parser.add_argument("--dataset-size", type=int, default=DEFAULT_DATASET_SIZE, help="Synthetic dataset size.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base output directory.")
    parser.add_argument("--nproc", type=int, default=DEFAULT_NPROC, help="Number of processes (ranks).")
    return parser


def main() -> None:
    """
    Entry point: parse args, run the controller sweep.
    """
    args = build_parser().parse_args()
    controller = Controller(
        batch_sizes=args.batch_sizes,
        split_layers=args.split_layers,
        num_steps=args.num_steps,
        dataset_size=args.dataset_size,
        seed=args.seed,
        output_dir=args.output_dir,
        nproc=args.nproc,
    )
    controller.run()


if __name__ == "__main__":
    main()
