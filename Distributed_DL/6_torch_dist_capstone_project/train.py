"""
Entry point for distributed SimCLR training with manual ResNet18 sharding.

Launched with torchrun:
    torchrun --nproc_per_node=4 train.py --local-batch-size 32 --num-steps 10
    torchrun --nproc_per_node=4 train.py --profile --run-name baseline
"""

import time
import torch
import torch.distributed as dist
import torch.nn as nn

from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from src.augmentation import build_simclr_transform
from src.cli import parse_config
from src.common import IMAGE_SIZE, NUM_CLASSES, TrainConfig
from src.groups import CommGroups
from src.logger import g_logger
from src.model import split_resnet18
from src.training_step import TrainingStep


def _align_replicas(stage0: nn.Sequential, stage1: nn.Sequential, groups: CommGroups) -> None:
    """
    One-time parameter broadcast so all replicas start with identical weights.

    Even ranks broadcast stage0 params within stage0_group.
    Odd ranks broadcast stage1 params within stage1_group.

    Args:
        stage0 (nn.Sequential): Stage 0 model.
        stage1 (nn.Sequential): Stage 1 model.
        groups (CommGroups): Communication groups.
    """
    stage, group = (stage0, groups.stage0_group) if groups.is_even else (stage1, groups.stage1_group)
    for p in stage.parameters():
        dist.broadcast(p.data, src=dist.get_global_rank(group, 0), group=group)


def _run_training(config: TrainConfig, groups: CommGroups, stepper: TrainingStep) -> list[dict]:
    """
    Execute the training loop and collect per-step metrics.

    Args:
        config (TrainConfig): Training configuration.
        groups (CommGroups): Communication groups.
        stepper (TrainingStep): Configured training step object.

    Returns:
        list[dict]: Per-step metrics dicts with keys: step, loss, step_time_s, rank.
    """
    metrics = []
    for step_idx in range(config.num_steps):
        t0 = time.perf_counter()
        loss_value = stepper.step(step_idx)
        dt = time.perf_counter() - t0
        metrics.append({"step": step_idx, "loss": loss_value, "step_time_s": dt, "rank": groups.rank})
        if groups.rank == 1 and loss_value is not None:
            g_logger.info(f"step {step_idx:3d} | loss {loss_value:.4f} | time {dt:.3f}s")
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
        g_logger.info(config.describe())

    # Set seed for reproducibility
    torch.manual_seed(config.seed + groups.rank)

    # Build model
    stage0, stage1 = split_resnet18(config.split_layer, config.projection_hidden, config.projection_dim)
    _align_replicas(stage0, stage1, groups)  # Ensure all replicas start with the same weights after splitting

    # Build dataset
    dataset = FakeData(config.dataset_size, IMAGE_SIZE, NUM_CLASSES, ToTensor(), random_offset=config.seed)

    # Build optimizer
    params = list(stage0.parameters()) if groups.is_even else list(stage1.parameters())
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # Build augmentation
    transform = build_simclr_transform()

    # Build training stepper
    should_check_alignment = config.num_steps <= 20
    stepper = TrainingStep(dataset, config, groups, stage0, stage1, optimizer, transform, should_check_alignment)

    # Log start
    if groups.rank == 0:
        g_logger.info(f"Run training for {config.num_steps} steps...")

    # Training loop
    t_start = time.perf_counter()
    _run_training(config, groups, stepper)
    t_total = time.perf_counter() - t_start

    # Summary
    if groups.rank == 0:
        images_per_sec = (config.num_steps * global_batch) / t_total
        g_logger.info(f"Done. {config.num_steps} steps in {t_total:.2f}s | {images_per_sec:.1f} images/s")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
