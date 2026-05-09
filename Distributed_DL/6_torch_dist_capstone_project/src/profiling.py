"""
Profiler context manager for conditional trace capture.
"""

from contextlib import contextmanager
from typing import Iterator

from torch.profiler import ProfilerActivity, profile

from src.common import TrainConfig
from src.logger import g_logger


@contextmanager
def profiler_context(config: TrainConfig, rank: int) -> Iterator[None]:
    """
    Context manager that wraps a block with the PyTorch profiler when config.profile is True.

    When profiling is enabled, captures CPU activity and exports a chrome trace JSON per rank.
    When disabled, yields immediately with no overhead.

    Args:
        config (TrainConfig): Training configuration (uses profile flag and output_path).
        rank (int): Global rank for the trace filename.
    """
    if not config.profile:
        yield
        return

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        yield

    trace_dir = config.output_path / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"rank{rank}.json"
    prof.export_chrome_trace(str(trace_path))
    if rank == 0:
        g_logger.info(f"Traces exported to {trace_dir}")
