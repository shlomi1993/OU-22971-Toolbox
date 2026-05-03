"""
Print each rank's identity inside a simple distributed job.

Run:

    torchrun --standalone --nproc_per_node=4 1_collective_communication/1_hello_ranks.py

Expected result:

- four worker processes start
- every process prints the same ``world_size``
- every process prints a different ``rank``
- every process prints its own local tensor

What to notice:

- we launched one Python file
- ``torchrun`` launched four workers
- each worker ran the same code with different process-local state
- ``LOCAL_RANK`` and ``rank`` are related, but they are not always the same in multi-node jobs

Training loop connection:
One training script becomes one copy per rank; the code is shared, but each rank has its own local state and data.
"""

import os

import torch
import torch.distributed as dist

from pretty_print import print_block


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        tensor = torch.tensor([rank], dtype=torch.int64)
        print_block(
            f"rank {rank}",
            f"local_rank={local_rank}",
            f"world_size={world_size}",
            f"local_tensor={tensor.tolist()}",
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
