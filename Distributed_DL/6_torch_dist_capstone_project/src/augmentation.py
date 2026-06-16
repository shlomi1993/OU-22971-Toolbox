"""
SimCLR augmentation pipeline and paired-view generation.
"""

import torch
import torchvision.transforms as T

from src.common import IMAGE_CROP_SIZE


def build_simclr_transform() -> T.Compose:
    """
    Build the fixed SimCLR augmentation pipeline.

    Returns:
        T.Compose: Composed transform applying RandomResizedCrop, RandomHorizontalFlip, ColorJitter, and RandomGrayscale.
    """
    return T.Compose([
        T.RandomResizedCrop(IMAGE_CROP_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
    ])


def create_paired_views(images: torch.Tensor, transform: T.Compose) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create two independently augmented views for each image in a batch.

    Each source image is transformed twice with the same stochastic pipeline, producing a positive pair per image.

    Args:
        images (torch.Tensor): Source image batch of shape (B, C, H, W).
        transform (T.Compose): Augmentation pipeline to apply.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two view batches, each (B, C, H, W).
    """
    view_1 = torch.stack([transform(img) for img in images])
    view_2 = torch.stack([transform(img) for img in images])
    return view_1, view_2
