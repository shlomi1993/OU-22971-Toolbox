"""
Show how broadcast copies one rank's tensor to every other rank.

Run:

    torchrun --standalone --nproc_per_node=4 1_collective_communication/3_broadcast_demo.py

Expected result:

- rank ``0`` starts with ``[42]``
- the other ranks start with ``[-1]``
- after the broadcast, all ranks hold ``[42]``

What to notice:

- rank ``0`` is the source of truth for this operation
- the receiving ranks must already have a compatible destination tensor
- after the call returns, every rank holds the same value

Training loop connection:
Use broadcast for sharing state from one rank to the rest of the job.
"""

import torch
import torch.distributed as dist

from pretty_print import print_block


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        tensor = torch.tensor([42 if rank == 0 else -1], dtype=torch.int64)
        print_block(f"rank {rank}", f"before broadcast: {tensor.tolist()}")
        dist.broadcast(tensor, src=0)
        print_block(f"rank {rank}", f"after broadcast: {tensor.tolist()}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
