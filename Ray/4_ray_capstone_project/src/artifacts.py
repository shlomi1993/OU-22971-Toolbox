"""
Artifact writers for persisting replay results to disk.

Provides JSON, CSV, and structured log writers for tick metrics, decisions, and latency data.
"""

import json
import logging
import pandas as pd

from pathlib import Path
from typing import Any, Dict, List

from src.common import TickMetrics


logger = logging.getLogger(__name__)


def write_json(data: Any, path: Path) -> None:
    """
    Write a JSON-serializable object to disk.

    Args:
        data (Any): JSON-serializable object to write.
        path (Path): Path to the output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Wrote {path}")


def write_metrics_csv(tick_metrics: List[TickMetrics], path: Path) -> None:
    """
    Write tick metrics to a CSV file.

    Args:
        tick_metrics (List[TickMetrics]): List of tick metrics to write.
        path (Path): Path to the output CSV file.
    """
    rows = [m.to_dict() for m in tick_metrics]
    for row in rows:
        row.pop("per_zone_latency", None)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Wrote {path}")


def write_tick_summary(tick_metrics: List[TickMetrics], decisions: Dict[int, Dict[int, str]], path: Path) -> None:
    """
    Write a tick-level summary JSON with decisions and metrics.

    Args:
        tick_metrics (List[TickMetrics]): List of tick metrics.
        decisions (Dict[int, Dict[int, str]]): Decisions for each tick and zone.
        path (Path): Path to the output JSON file.
    """
    summary = []
    for m in tick_metrics:
        tick_decisions = decisions.get(m.tick_id, {})
        summary.append({
            **m.to_dict(),
            "decisions": {str(z): d for z, d in tick_decisions.items()},
        })
    write_json(summary, path)


def write_latency_log(tick_metrics: List[TickMetrics], path: Path) -> None:
    """
    Write per-zone latency log as JSON.

    Args:
        tick_metrics (List[TickMetrics]): List of tick metrics.
        path (Path): Path to the output JSON file.
    """
    log_entries = []
    for m in tick_metrics:
        for zone_id, lat in m.per_zone_latency.items():
            log_entries.append({"tick_id": m.tick_id, "zone_id": zone_id, "latency_s": round(lat, 4)})
    write_json(log_entries, path)
