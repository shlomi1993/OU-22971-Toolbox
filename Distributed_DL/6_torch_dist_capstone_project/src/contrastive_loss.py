"""
Local-approximate SimCLR contrastive loss for distributed training.
"""

import torch
import torch.nn.functional as F

from src.common import DEFAULT_TEMPERATURE


def simclr_loss(local_embeddings: torch.Tensor, all_embeddings: torch.Tensor, local_start_idx: int,
                temperature: float = DEFAULT_TEMPERATURE) -> torch.Tensor:
    """
    Compute the approximate SimCLR contrastive loss over local embeddings.

    Each even rank produces a batch of source images, which are augmented into two views. The views are interleaved so
    that source image i produces views at indices 2i and 2i+1. The positive pair for view 2i is view 2i+1 and vice versa.

    Only loss terms for local embeddings are computed (the ones that are still attached to the local autograd graph).
    Remote embeddings participate as fixed negatives - their gradients are not propagated through the loss.

    Flow:
        1. L2-normalize all embeddings.
        2. Compute cosine similarity of each local embedding against all embeddings, scaled by 1/temperature.
        3. For each local view, mask out the self-similarity entry.
        4. Identify the positive-pair index (the other augmented view of the same source image).
        5. Compute CE where the positive pair is the target class and all other views (excluding self) are negatives.
        6. Return the mean loss across all local views.

    Args:
        local_embeddings (torch.Tensor): Embeddings produced on this rank, shape (2 * local_batch_size, projection_dim).
            Must require grad.
        all_embeddings (torch.Tensor): Concatenation of local (live) and remote (detached) embeddings from all ranks in
            the stage-1 group, shape (2 * global_batch_size, projection_dim).
        local_start_idx (int): Start index of this rank's local embeddings within all_embeddings.
        temperature (float, optional): Temperature scaling. Default is 0.5.

    Returns:
        torch.Tensor: Scalar contrastive loss.
    """
    n_local = local_embeddings.shape[0]

    # Step 1: L2-normalize
    local_norm = F.normalize(local_embeddings, dim=1)
    all_norm = F.normalize(all_embeddings, dim=1)

    # Step 2: Cosine similarity matrix (n_local x n_total), scaled
    sim = local_norm @ all_norm.T / temperature

    # Step 3: Mask out self-similarity by setting it to a large negative value
    self_indices = torch.arange(n_local, device=sim.device) + local_start_idx
    sim[torch.arange(n_local, device=sim.device), self_indices] = float("-inf")

    # Step 4: Build positive-pair targets
    # - view_1 is interleaved at 2i and view_2 is interleaved at 2i+1
    # - XOR flips the last bit to find the partner
    positive_indices = (torch.arange(n_local, device=sim.device) + local_start_idx) ^ 1

    # Step 5: Cross-entropy loss with positive pair as target
    loss = F.cross_entropy(sim, positive_indices)

    return loss
