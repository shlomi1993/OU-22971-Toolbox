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
import torch.nn as nn

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


def align_replicas(stage0: nn.Sequential, stage1: nn.Sequential, groups: CommGroups) -> None:
    """
    One-time parameter broadcast so all replicas start with identical weights.

    Even ranks broadcast their stage params within the stage-0 group.
    Odd ranks broadcast their stage params within the stage-1 group.

    Args:
        stage0 (nn.Sequential): Stage 0 model.
        stage1 (nn.Sequential): Stage 1 model.
        groups (CommGroups): Communication groups.
    """
    stage, group = (stage0, groups.stage0_group) if groups.is_even else (stage1, groups.stage1_group)
    for p in stage.parameters():
        dist.broadcast(p.data, src=dist.get_global_rank(group, 0), group=group)


def run_training(start_step: int, end_step: int, groups: CommGroups, stepper: TrainingStep) -> StepMetrics:
    """
    Run training steps and collect local per-step timing and loss.

    Args:
        start_step (int): First step index (inclusive).
        end_step (int): Last step index (exclusive).
        groups (CommGroups): Communication groups.
        stepper (TrainingStep): Configured training step object.

    Returns:
        StepMetrics: (step_times, losses). Losses are NaN on even ranks.
    """
    times = []
    losses = []
    for step_idx in range(start_step, end_step):
        t0 = time.perf_counter()
        loss_value = stepper.step(step_idx)
        dt = time.perf_counter() - t0
        times.append(dt)
        losses.append(loss_value if loss_value is not None else float("nan"))
        if groups.rank == 1 and loss_value is not None:
            g_logger.info(f"step {step_idx:3d} | loss {loss_value:.4f} | time {dt:.3f}s")
    return times, losses


def gather_metrics(times: list[float], losses: list[float], groups: CommGroups, num_steps: int) -> list[dict]:
    """
    Gather per-rank timing and loss vectors onto rank 0 and build a flat metrics table.

    Args:
        times (list[float]): Local step times.
        losses (list[float]): Local losses (NaN on even ranks).
        groups (CommGroups): Communication groups.
        num_steps (int): Number of training steps.

    Returns:
        list[dict]: Full metrics table on rank 0, empty list on other ranks.
    """
    time_tensor = torch.tensor(times, dtype=torch.float64)
    loss_tensor = torch.tensor(losses, dtype=torch.float64)

    # Gather on rank 0
    if groups.rank == 0:
        all_times = [torch.zeros_like(time_tensor) for _ in range(groups.world_size)]
        all_losses = [torch.zeros_like(loss_tensor) for _ in range(groups.world_size)]
    else:
        all_times = None
        all_losses = None

    # Gather tensors
    dist.gather(time_tensor, gather_list=all_times, dst=0)
    dist.gather(loss_tensor, gather_list=all_losses, dst=0)

    # Return empty on non-zero ranks
    if groups.rank != 0:
        return []

    # Build and return metrics table
    metrics = []
    for r in range(groups.world_size):
        for s in range(num_steps):
            loss_val = all_losses[r][s].item()
            metrics.append({
                "step": s,
                "rank": r,
                "loss": None if math.isnan(loss_val) else round(loss_val, 6),
                "step_time_s": round(all_times[r][s].item(), 6),
            })
    return metrics


def main() -> None:
    """
    Main training entry point. Parses CLI args, sets up model and groups, runs training loop.
    """
    config = parse_config()

    # Initialize distributed groups
    groups = CommGroups.create()
    groups.log_structure()

    # Derive global batch size
    global_batch = config.global_batch_size(groups.num_pairs)
    if groups.rank == 0:
        g_logger.info(f"Configuration:\n{config.describe()}")

    # Set seed for reproducibility
    torch.manual_seed(config.seed + groups.rank)

    # Build model
    stage0, stage1 = split_resnet18(config.split_layer, config.projection_hidden, config.projection_dim)
    align_replicas(stage0, stage1, groups)  # Ensure all replicas start with the same weights after splitting

    # Build dataset
    dataset = FakeData(config.dataset_size, IMAGE_SIZE, NUM_CLASSES, ToTensor(), random_offset=config.seed)

    # Build optimizer
    params = list(stage0.parameters()) if groups.is_even else list(stage1.parameters())
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # Build augmentation
    transform = build_simclr_transform()

    # Build training stepper
    should_check_alignment = config.num_steps <= ALIGNMENT_CHECK_MAX_STEPS
    stepper = TrainingStep(dataset, config, groups, stage0, stage1, optimizer, transform, should_check_alignment)

    # Log start
    if groups.rank == 0:
        g_logger.info(f"Run training for {config.num_steps} steps...")

    # Warmup steps outside profiler to reduce trace noise
    warmup = min(DEFAULT_PROFILER_WARMUP_STEPS, config.num_steps) if config.profile else 0
    if warmup > 0:
        run_training(0, warmup, groups, stepper)

    # Training loop - the profiled portion
    t_start = time.perf_counter()
    with profiler_context(config, groups.rank):
        times, losses = run_training(warmup, config.num_steps, groups, stepper)
        if config.overlap:
            stepper.drain_overlap()
    t_total = time.perf_counter() - t_start

    # Gather metrics from all ranks - collective
    metrics = gather_metrics(times, losses, groups, len(times))

    # Save metrics and summary - rank 0 only
    if groups.rank == 0:
        save_metrics(metrics, config, groups.num_pairs, t_total)

    # Summary
    if groups.rank == 0:
        profiled_steps = config.num_steps - warmup
        images_per_sec = (profiled_steps * global_batch) / t_total
        g_logger.info(f"Done. {config.num_steps} steps in {t_total:.2f}s | {images_per_sec:.1f} images/s")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
