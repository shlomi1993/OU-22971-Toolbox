"""
Load-balancing controller (Stretch B).

Automates batch-size and split-layer sweeps. Runs training, analyzes traces, picks the next configuration based on
images/s, communication overhead, and stage imbalance, then reruns and compares.

Usage:
    python controller.py
    python controller.py --num-steps 10 --dataset-size 2048
    python controller.py --batch-sizes 4 8 16 32 --split-layers layer1 layer2
"""

import argparse
import csv
import json
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path

from src.common import CONFIG_FILENAME, DEFAULT_DATASET_SIZE, DEFAULT_NUM_STEPS, DEFAULT_OUTPUT_DIR, DEFAULT_SEED, SPLIT_CHOICES
from src.logger import g_logger
from analyze import TraceAnalyzer


DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64]
DEFAULT_SPLIT_LAYERS = ["layer1", "layer2", "layer3"]
DEFAULT_NPROC = 4
CONTROLLER_LOG = "controller_log.json"


@dataclass
class RunResult:
    """
    Metrics extracted from a single training run.
    """
    local_batch_size: int  # Local batch size used in the run
    split_layer: str  # Split layer name
    images_per_sec: float  # Images/s
    comm_pct: float  # Communication percentage of total step time
    stage_imbalance: float  # Stage imbalance ratio (odd/even compute time)
    run_name: str  # Run name (subdirectory under output_dir)
    mean_loss: float = 0.0  # Mean loss across profiled steps (odd ranks only)


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
            num_steps (int, optional): Training steps per run. Defaults to DEFAULT_N_STEPS.
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
    def _analyze_traces(run_dir: Path) -> tuple[float, float]:
        """
        Compute mean communication percentage and stage imbalance ratio from traces.

        Args:
            run_dir (Path): Directory containing trace files for a single run.

        Returns:
            tuple[float, float]: Mean communication percentage and imbalance ratio.
        """
        analyzer = TraceAnalyzer(run_dir)
        analyzer.load()

        # Mean comm percentage across all ranks
        comm_pcts = [analyzer.compute_breakdown(s).comm_pct for s in analyzer.summaries.values()]
        mean_comm = sum(comm_pcts) / len(comm_pcts) if comm_pcts else 0.0

        # Stage imbalance
        even_ranks = []
        odd_ranks = []
        for rank in analyzer.summaries.keys():
            if rank % 2 == 0:
                even_ranks.append(rank)
            else:
                odd_ranks.append(rank)

        # Stage imbalance ratio
        stage0_ms = analyzer.calc_mean_span_ms(even_ranks, {"stage0_forward", "stage0_backward"})
        stage1_ms = analyzer.calc_mean_span_ms(odd_ranks, {"stage1_forward", "loss_calculation"})
        imbalance = stage1_ms / stage0_ms if stage0_ms > 0 else 1.0

        return mean_comm, imbalance

    @staticmethod
    def _extract_mean_loss(run_dir: Path) -> float:
        """
        Extract mean loss from the metrics file (odd-rank rows only).

        Args:
            run_dir (Path): Directory containing the metrics file.

        Returns:
            float: Mean training loss across profiled steps, or 0.0 if the metrics file is missing.
        """
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            return 0.0

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        losses = [float(r["loss"]) for r in rows if r["loss"]]
        return sum(losses) / len(losses) if losses else 0.0

    def _extract_result(self, run_name: str) -> RunResult:
        """
        Extract metrics from a completed run's output artifacts.

        Args:
            run_name (str): Name of the run subdirectory.

        Returns:
            RunResult | None: Extracted metrics, or None if artifacts are missing.
        """
        run_dir = Path(self.output_dir) / run_name

        config_path = run_dir / CONFIG_FILENAME
        if not config_path.exists():
            g_logger.warning(f"No config file in {run_dir}")
            return None

        with open(config_path) as f:
            run_config: dict = json.load(f)

        comm_pct, imbalance = self._analyze_traces(run_dir)
        mean_loss = self._extract_mean_loss(run_dir)

        return RunResult(
            local_batch_size=run_config.get("local_batch_size", 0),
            split_layer=run_config.get("split_layer", ""),
            images_per_sec=run_config.get("images_per_sec", 0.0),
            comm_pct=comm_pct,
            stage_imbalance=imbalance,
            run_name=run_name,
            mean_loss=mean_loss,
        )

    def _log_summary(self, best: RunResult) -> None:
        """
        Log a formatted summary table and the chosen best configuration.

        Args:
            best (RunResult): The best run result to highlight in the table.
        """
        header = (f"  {'batch_size':>10s} {'split_layer':>12s} {'images/s':>10s} "
                  f"{'comm%':>7s} {'imbalance':>10s} {'loss':>8s}")
        separator = f"  {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 7} {'-' * 10} {'-' * 8}"

        lines = ["Sweep results", header, separator]
        for r in sorted(self.results, key=lambda x: -x.images_per_sec):
            marker = " <-- best" if r.run_name == best.run_name else ""
            lines.append(
                f"  {r.local_batch_size:>10d} {r.split_layer:>12s} {r.images_per_sec:>10.1f} "
                f"{r.comm_pct:>6.1f}% {r.stage_imbalance:>10.2f}x {r.mean_loss:>8.4f}{marker}"
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
            "runs": [
                {
                    "run_name": r.run_name,
                    "local_batch_size": r.local_batch_size,
                    "split_layer": r.split_layer,
                    "images_per_sec": r.images_per_sec,
                    "comm_pct": r.comm_pct,
                    "stage_imbalance": r.stage_imbalance,
                    "mean_loss": r.mean_loss,
                }
                for r in self.results
            ],
            "best": {
                "run_name": best.run_name,
                "local_batch_size": best.local_batch_size,
                "split_layer": best.split_layer,
                "images_per_sec": best.images_per_sec,
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
                g_logger.info(f"  => {result.images_per_sec:.1f} img/s, comm={result.comm_pct:.1f}%, imbalance={result.stage_imbalance:.2f}x")

        if not self.results:
            g_logger.error("No successful runs. Exiting.")
            sys.exit(1)

        best = max(self.results, key=lambda r: r.images_per_sec)  # Pick the best config by images/s
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
