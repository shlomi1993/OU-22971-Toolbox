"""
CLI argument parser for the SimCLR sharded training system.
"""

import argparse

from src.common import (
    DEFAULT_DATASET_SIZE,
    DEFAULT_LR,
    DEFAULT_LOCAL_BATCH_SIZE,
    DEFAULT_MOMENTUM,
    DEFAULT_NUM_STEPS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROJECTION_DIM,
    DEFAULT_PROJECTION_HIDDEN,
    DEFAULT_SEED,
    DEFAULT_SPLIT_LAYER,
    DEFAULT_TEMPERATURE,
    DEFAULT_WEIGHT_DECAY,
    SPLIT_CHOICES,
    TrainConfig,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for train.py.

    Returns:
        argparse.ArgumentParser: Configured parser with all training flags.
    """
    parser = argparse.ArgumentParser(
        description="Distributed SimCLR training with manual ResNet18 sharding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset-size", type=int, default=DEFAULT_DATASET_SIZE, help="Number of synthetic images in FakeData dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")

    # Training
    parser.add_argument("--local-batch-size", type=int, default=DEFAULT_LOCAL_BATCH_SIZE, help="Batch size per even rank (per model replica).")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS, help="Number of training steps to run.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay for SGD optimizer.")

    # SimCLR
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature scaling for contrastive loss.")
    parser.add_argument("--projection-dim", type=int, default=DEFAULT_PROJECTION_DIM, help="Output dimension of the projection head.")
    parser.add_argument("--projection-hidden", type=int, default=DEFAULT_PROJECTION_HIDDEN, help="Hidden dimension of the projection head.")

    # Model split
    parser.add_argument("--split-layer", type=str, default=DEFAULT_SPLIT_LAYER, choices=SPLIT_CHOICES, help="Layer after which to split ResNet18 into two stages.")

    # Profiling & output
    parser.add_argument("--profile", action="store_true", help="Enable profiler and export per-rank trace JSONs.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base output directory for traces and metrics.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional subdirectory name under output-dir.")

    # Stretch A
    parser.add_argument("--overlap", action="store_true", help="Enable async forward/backward overlap on even ranks (Stretch A).")

    return parser


def parse_config() -> TrainConfig:
    """
    Parse CLI arguments into a TrainConfig instance.

    Returns:
        TrainConfig: Populated configuration.
    """
    args = build_parser().parse_args()
    return TrainConfig(
        dataset_size=args.dataset_size,
        seed=args.seed,
        local_batch_size=args.local_batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        projection_hidden=args.projection_hidden,
        split_layer=args.split_layer,
        profile=args.profile,
        output_dir=args.output_dir,
        run_name=args.run_name,
        overlap=args.overlap,
    )
