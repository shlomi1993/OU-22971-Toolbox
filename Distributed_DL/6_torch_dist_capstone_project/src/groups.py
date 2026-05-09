"""
Communication group setup for the two-stage sharded training system.
"""

import torch.distributed as dist

from dataclasses import dataclass

from src.common import Backend
from src.logger import g_logger


@dataclass
class CommGroups:
    """
    Holds all communication group handles and rank metadata.
    """
    rank: int  # Global rank of this process
    world_size: int  # Total number of ranks
    pair_group: dist.ProcessGroup  # This rank's (even, odd) pair group
    stage0_group: dist.ProcessGroup  # All even ranks (stage-0 owners)
    stage1_group: dist.ProcessGroup  # All odd ranks (stage-1 owners)
    pair_rank: int  # Global rank of the paired partner
    pair_id: int  # Index of this rank's pair (rank // 2)
    num_pairs: int  # Total number of rank pairs (world_size // 2)
    is_even: bool  # True if this rank owns stage 0

    @classmethod
    def create(cls, backend: Backend = Backend.GLOO) -> "CommGroups":
        """
        Initialize torch.distributed and create the required communication groups.

        Flow:
            1. Initialize the default world process group.
            2. Validate that world_size is even and >= 4.
            3. Create a pair group for each (even, odd) pair.
            4. Create a stage group for all even ranks.
            5. Create a stage group for all odd ranks.

        Args:
            backend (Backend, optional): Distributed backend. Default is Backend.GLOO.

        Returns:
            CommGroups: Populated communication groups and rank metadata.
        """
        dist.init_process_group(backend=backend.value)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size < 4 or world_size % 2 != 0:
            raise ValueError(f"world_size must be even and >= 4, got {world_size}")

        num_pairs = world_size // 2
        pair_id = rank // 2
        is_even = rank % 2 == 0
        pair_rank = rank + 1 if is_even else rank - 1

        # Create one pair group per (even, odd) pair; keep the one this rank belongs to
        my_pair_group = None
        for k in range(num_pairs):
            group = dist.new_group(ranks=[2 * k, 2 * k + 1])
            if k == pair_id:
                my_pair_group = group

        even_ranks = list(range(0, world_size, 2))
        odd_ranks = list(range(1, world_size, 2))
        stage0_group = dist.new_group(ranks=even_ranks)
        stage1_group = dist.new_group(ranks=odd_ranks)

        return cls(
            rank=rank,
            world_size=world_size,
            pair_group=my_pair_group,
            stage0_group=stage0_group,
            stage1_group=stage1_group,
            pair_rank=pair_rank,
            pair_id=pair_id,
            num_pairs=num_pairs,
            is_even=is_even,
        )

    def log_structure(self) -> None:
        """
        Log the communication structure once from rank 0.
        """
        if self.rank != 0:
            return

        even_ranks = list(range(0, self.world_size, 2))
        odd_ranks = list(range(1, self.world_size, 2))

        lines = [
            "",
            "=" * 50,
            "Communication Structure",
            "=" * 50,
            f"  world_group  : ranks {list(range(self.world_size))}",
        ]
        for k in range(self.num_pairs):
            lines.append(f"  pair_group({k}): ranks [{2 * k}, {2 * k + 1}]")
        lines.append(f"  stage0_group : ranks {even_ranks}")
        lines.append(f"  stage1_group : ranks {odd_ranks}")
        lines.append("=" * 50)
        g_logger.info("\n".join(lines))
