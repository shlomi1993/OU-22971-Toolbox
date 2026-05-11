"""
ResNet18 stage splitting and projection head for SimCLR sharded training.
"""

import torch
import torch.nn as nn

from torchvision.models import resnet18

from src.common import DEFAULT_PROJECTION_DIM, DEFAULT_PROJECTION_HIDDEN, DEFAULT_SPLIT_LAYER, SPLIT_CHOICES


# Output channels at each potential split point in ResNet18
BOUNDARY_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256}

# Spatial resolution at each split point (input 224x224)
BOUNDARY_SPATIAL = {"layer1": 56, "layer2": 28, "layer3": 14}

# ResNet18 feature dimension after avgpool + flatten
RESNET18_FEATURE_DIM = 512


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head used in SimCLR.

    Maps backbone features to a lower-dimensional embedding space where the contrastive loss is computed.
    """

    def __init__(self, in_dim: int = RESNET18_FEATURE_DIM, hidden_dim: int = DEFAULT_PROJECTION_HIDDEN,
                 out_dim: int = DEFAULT_PROJECTION_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_resnet18(split_layer: str = DEFAULT_SPLIT_LAYER, projection_hidden: int = DEFAULT_PROJECTION_HIDDEN,
                   projection_dim: int = DEFAULT_PROJECTION_DIM) -> tuple[nn.Sequential, nn.Sequential]:
    """
    Split a freshly initialized ResNet18 into two sequential stages.

    Stage 0 contains the stem (conv1, bn1, relu, maxpool) plus residual blocks up to and including split_layer.
    Stage 1 contains the remaining residual blocks, the adaptive average pool, a flatten operation, and a ProjectionHead.

    Args:
        split_layer (str): Layer after which to split. One of "layer1", "layer2", "layer3".
        projection_hidden (int): Hidden dimension of the projection head.
        projection_dim (int): Output dimension of the projection head.

    Returns:
        tuple[nn.Sequential, nn.Sequential]: a tuple of (stage0, stage1) where each is an nn.Sequential containing the
            respective layers.
    """
    if split_layer not in SPLIT_CHOICES:
        raise ValueError(f"split_layer must be one of {SPLIT_CHOICES}, got '{split_layer}'")

    model = resnet18(weights=None)

    # Ordered layer names in ResNet18 backbone (excluding fc)
    layer_names = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
    split_idx = layer_names.index(split_layer) + 1

    stage0_layers = [getattr(model, name) for name in layer_names[:split_idx]]
    stage1_layers = [getattr(model, name) for name in layer_names[split_idx:]]

    # Stage 1 ends with avgpool -> flatten -> projection head
    stage1_layers.extend([
        model.avgpool,
        nn.Flatten(start_dim=1),
        ProjectionHead(RESNET18_FEATURE_DIM, projection_hidden, projection_dim),
    ])

    return nn.Sequential(*stage0_layers), nn.Sequential(*stage1_layers)


def boundary_shape(split_layer: str, batch_size: int) -> tuple[int, ...]:
    """
    Expected tensor shape at the stage boundary for a given split point.

    Args:
        split_layer (str): One of "layer1", "layer2", "layer3".
        batch_size (int): Number of images (views) in the batch.

    Returns:
        tuple[int, ...]: Tensor shape at the boundary, in (batch_size, channels, height, width) format.
    """
    channels = BOUNDARY_CHANNELS[split_layer]
    spatial = BOUNDARY_SPATIAL[split_layer]
    return (batch_size, channels, spatial, spatial)
