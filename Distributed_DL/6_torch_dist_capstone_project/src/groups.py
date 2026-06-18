"""
Communication group setup for the two-stage sharded training system.
"""

import torch.distributed as dist

from prettytable import PrettyTable

from src.common import Backend
from src.logger import g_logger


class CommGroups:
    """
    Holds all communication group handles and rank metadata.
    """

    def __init__(self, backend: Backend = Backend.GLOO) -> None:
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
        """
        dist.init_process_group(backend=backend.value)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        if self.world_size < 4 or self.world_size % 2 != 0:
            raise ValueError(f"world_size must be even and >= 4, got {self.world_size}")

        self.num_pairs = self.world_size // 2
        self.pair_id = self.rank // 2
        self.is_even = self.rank % 2 == 0
        self.pair_rank = self.rank + 1 if self.is_even else self.rank - 1

        # Create one pair group per (even, odd) pair and keep the one this rank belongs to
        self.pair_group = None
        for k in range(self.num_pairs):
            group = dist.new_group(ranks=[2 * k, 2 * k + 1])
            if k == self.pair_id:
                self.pair_group = group

        self.even_ranks = list(range(0, self.world_size, 2))
        self.odd_ranks = list(range(1, self.world_size, 2))
        self.stage0_group = dist.new_group(ranks=self.even_ranks)
        self.stage1_group = dist.new_group(ranks=self.odd_ranks)

    def log_structure(self) -> None:
        """
        Log the communication structure once from rank 0.
        """
        if self.rank != 0:
            return

        table = PrettyTable()
        table.title = "Communication Structure"
        table.field_names = ["group", "ranks"]
        table.align["group"] = "l"
        table.align["ranks"] = "l"
        table.header = False
        table.add_row(["world_group", list(range(self.world_size))])
        for k in range(self.num_pairs):
            table.add_row([f"pair_group({k})", [2 * k, 2 * k + 1]])
        table.add_row(["stage0_group", self.even_ranks])
        table.add_row(["stage1_group", self.odd_ranks])

        g_logger.info("\n" + table.get_string() + "\n")
