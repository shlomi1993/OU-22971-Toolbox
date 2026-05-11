"""
Entry point for distributed SimCLR training with manual ResNet18 sharding.

Launched with torchrun:
    torchrun --nproc_per_node=4 train.py --local-batch-size 32 --num-steps 10
    torchrun --nproc_per_node=4 train.py --profile --run-name baseline
"""

import math
import time
import torch
import torch.distributed as dist

from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from src.augmentation import build_simclr_transform
from src.cli import parse_config
from src.common import ALIGNMENT_CHECK_MAX_STEPS, DEFAULT_PROFILER_WARMUP_STEPS, IMAGE_SIZE, NUM_CLASSES, TrainConfig
from src.groups import CommGroups
from src.logger import g_logger
from src.metrics import save_metrics
from src.model import split_resnet18
from src.profiling import profiler_context
from src.training_step import TrainingStep


StepMetrics = tuple[list[float], list[float]]


class TrainRunner:
    """
    Orchestrates the full distributed SimCLR training lifecycle.

    Encapsulates model construction, replica alignment, the training loop (with warmup and profiling), metric gathering,
    and persistence. Created once per run from a TrainConfig and CommGroups.
    """

    def __init__(self, config: TrainConfig, groups: CommGroups) -> None:
        """
        Args:
            config (TrainConfig): Training configuration.
            groups (CommGroups): Communication groups and rank metadata.
        """
        self.config = config
        self.groups = groups
        self._global_batch = config.global_batch_size(groups.num_pairs)

        # Set seed for reproducibility
        torch.manual_seed(config.seed + groups.rank)

        # Build model
        self.stage0, self.stage1 = split_resnet18(config.split_layer, config.projection_hidden, config.projection_dim)
        self._align_replicas()

        # Build dataset, optimizer, augmentation
        self._dataset = FakeData(config.dataset_size, IMAGE_SIZE, NUM_CLASSES, ToTensor(), random_offset=config.seed)
        params = list(self.stage0.parameters()) if groups.is_even else list(self.stage1.parameters())
        self._optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        self._transform = build_simclr_transform()

        # Build training stepper
        check = config.num_steps <= ALIGNMENT_CHECK_MAX_STEPS
        self._stepper = TrainingStep(self._dataset, config, groups, self.stage0, self.stage1, self._optimizer, self._transform, check)

    def _align_replicas(self) -> None:
        """
        One-time parameter broadcast so all replicas start with identical weights.
        """
        stage, group = ((self.stage0, self.groups.stage0_group) if self.groups.is_even else (self.stage1, self.groups.stage1_group))
        for p in stage.parameters():
            dist.broadcast(p.data, src=dist.get_global_rank(group, 0), group=group)

    def _run_steps(self, start_step: int, end_step: int) -> StepMetrics:
        """
        Run training steps and collect local per-step timing and loss.

        Args:
            start_step (int): First step index (inclusive).
            end_step (int): Last step index (exclusive).

        Returns:
            StepMetrics: (step_times, losses). Losses are NaN on even ranks.
        """
        times = []
        losses = []
        for step_idx in range(start_step, end_step):
            t0 = time.perf_counter()
            loss_value = self._stepper.step(step_idx)
            dt = time.perf_counter() - t0
            times.append(dt)
            losses.append(loss_value if loss_value is not None else float("nan"))
            if self.groups.rank == 1 and loss_value is not None:
                g_logger.info(f"step {step_idx:3d} | loss {loss_value:.4f} | time {dt:.3f}s")
        return times, losses

    def _gather_metrics(self, times: list[float], losses: list[float]) -> list[dict]:
        """
        Gather per-rank timing and loss vectors onto rank 0 and build a flat metrics table.

        Args:
            times (list[float]): Local step times.
            losses (list[float]): Local losses (NaN on even ranks).

        Returns:
            list[dict]: Full metrics table on rank 0, empty list on other ranks.
        """
        num_steps = len(times)
        time_tensor = torch.tensor(times, dtype=torch.float64)
        loss_tensor = torch.tensor(losses, dtype=torch.float64)

        # Gather onto rank 0
        if self.groups.rank == 0:
            all_times = [torch.zeros_like(time_tensor) for _ in range(self.groups.world_size)]
            all_losses = [torch.zeros_like(loss_tensor) for _ in range(self.groups.world_size)]
        else:
            all_times = None
            all_losses = None

        # Gather tensors from all ranks
        dist.gather(time_tensor, gather_list=all_times, dst=0)
        dist.gather(loss_tensor, gather_list=all_losses, dst=0)

        # Return empty on non-zero ranks
        if self.groups.rank != 0:
            return []

        # Build and return metrics table
        metrics: list[dict] = []
        for r in range(self.groups.world_size):
            for s in range(num_steps):
                loss_val = all_losses[r][s].item()
                metrics.append({
                    "step": s,
                    "rank": r,
                    "loss": None if math.isnan(loss_val) else round(loss_val, 6),
                    "step_time_s": round(all_times[r][s].item(), 6),
                })
        return metrics

    def run(self) -> None:
        """
        Execute the full training run: warmup, profiled training loop, metrics gathering, and persistence.
        """
        if self.groups.rank == 0:
            g_logger.info(f"Run training for {self.config.num_steps} steps...")

        # Warmup steps outside profiler
        warmup = min(DEFAULT_PROFILER_WARMUP_STEPS, self.config.num_steps) if self.config.profile else 0
        if warmup > 0:
            self._run_steps(0, warmup)

        # Profiled training loop
        t_start = time.perf_counter()
        with profiler_context(self.config, self.groups.rank):
            times, losses = self._run_steps(warmup, self.config.num_steps)
            if self.config.overlap:
                self._stepper.drain_overlap()
        t_total = time.perf_counter() - t_start

        # Gather and save metrics
        metrics = self._gather_metrics(times, losses)
        if self.groups.rank == 0:
            save_metrics(metrics, self.config, self.groups.num_pairs, t_total)

        # Summary
        if self.groups.rank == 0:
            profiled_steps = self.config.num_steps - warmup
            images_per_sec = (profiled_steps * self._global_batch) / t_total
            g_logger.info(f"Done. {self.config.num_steps} steps in {t_total:.2f}s | {images_per_sec:.1f} images/s")


def main() -> None:
    """
    Main training entry point. Parses CLI args, initializes distributed groups, and runs training.
    """
    config = parse_config()

    groups = CommGroups()
    groups.log_structure()

    if groups.rank == 0:
        g_logger.info(f"Configuration:\n{config.describe()}")

    runner = TrainRunner(config, groups)
    runner.run()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
