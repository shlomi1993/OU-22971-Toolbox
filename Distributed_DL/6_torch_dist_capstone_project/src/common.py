"""
Common constants and configuration dataclass for the SimCLR sharded training system.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Dataset defaults
DEFAULT_DATASET_SIZE = 1024
IMAGE_CHANNELS = 3
IMAGE_CROP_SIZE = 224
IMAGE_SIZE = (IMAGE_CHANNELS, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
NUM_CLASSES = 1000
DEFAULT_SEED = 42

# Training defaults
DEFAULT_LOCAL_BATCH_SIZE = 32
DEFAULT_NUM_STEPS = 10
DEFAULT_LR = 0.03
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4

# SimCLR defaults
DEFAULT_TEMPERATURE = 0.5
DEFAULT_PROJECTION_DIM = 128
DEFAULT_PROJECTION_HIDDEN = 256

# Model split
SPLIT_CHOICES = ["layer1", "layer2", "layer3"]
DEFAULT_SPLIT_LAYER = "layer2"

# Output
DEFAULT_OUTPUT_DIR = "output"


@dataclass
class TrainConfig:
    """
    Runtime configuration for distributed SimCLR training.
    """

    dataset_size: int = DEFAULT_DATASET_SIZE
    seed: int = DEFAULT_SEED
    local_batch_size: int = DEFAULT_LOCAL_BATCH_SIZE
    num_steps: int = DEFAULT_NUM_STEPS
    lr: float = DEFAULT_LR
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    temperature: float = DEFAULT_TEMPERATURE
    projection_dim: int = DEFAULT_PROJECTION_DIM
    projection_hidden: int = DEFAULT_PROJECTION_HIDDEN
    split_layer: str = DEFAULT_SPLIT_LAYER
    profile: bool = False
    output_dir: str = DEFAULT_OUTPUT_DIR
    overlap: bool = False
    run_name: Optional[str] = None

    @property
    def output_path(self) -> Path:
        """
        Resolved output directory, including run_name subdirectory if set.
        """
        base = Path(self.output_dir)
        return base / self.run_name if self.run_name else base

    def global_batch_size(self, num_pairs: int) -> int:
        """
        Global batch size across all model-replica pairs.

        Args:
            num_pairs (int): Number of rank pairs (world_size // 2).

        Returns:
            int: Total images per training step.
        """
        return self.local_batch_size * num_pairs
