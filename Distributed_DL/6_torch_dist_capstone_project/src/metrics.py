"""
Metrics collection and persistence for training runs.
"""

import csv
import json

from dataclasses import asdict

from src.common import TrainConfig
from src.logger import g_logger


def save_metrics(metrics: list[dict], config: TrainConfig, num_pairs: int, wall_time: float) -> None:
    """
    Write per-step metrics CSV and run configuration JSON from rank 0.

    Args:
        metrics (list[dict]): Per-step metric records - one dict per step per rank.
        config (TrainConfig): Training configuration.
        num_pairs (int): Number of model-replica pairs.
        wall_time (float): Total wall-clock training time in seconds.
    """
    # Ensure output directory exists
    out = config.output_path
    out.mkdir(parents=True, exist_ok=True)

    # Write metrics CSV
    csv_path = out / "metrics.csv"
    fieldnames = ["step", "loss", "step_time_s", "rank"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    # Write run config JSON
    global_batch = config.global_batch_size(num_pairs)
    num_profiled_steps = len(set(m["step"] for m in metrics))
    images_per_sec = (num_profiled_steps * global_batch) / wall_time if wall_time > 0 else 0.0

    # Build config dict with additional summary fields
    run_config = asdict(config)
    run_config["global_batch_size"] = global_batch
    run_config["images_per_sec"] = round(images_per_sec, 2)

    # Write config JSON
    json_path = out / "run_config.json"
    with open(json_path, "w") as f:
        json.dump(run_config, f, indent=2)

    g_logger.info(f"Metrics saved to {out}")
