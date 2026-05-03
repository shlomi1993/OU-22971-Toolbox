"""
Demonstrate scattering one tensor shard from rank 0 to each rank.

Run:

    torchrun --standalone --nproc_per_node=4 1_collective_communication/6_scatter_demo.py

Expected result:

- rank ``0`` receives ``[10]``
- rank ``1`` receives ``[20]``
- rank ``2`` receives ``[30]``
- rank ``3`` receives ``[40]``

What to notice:

- rank ``0`` prepares one input tensor per worker
- each rank blocks in the same ``scatter`` call
- ``scatter`` is the opposite of ``gather``

Training loop connection:
Scatter is a simple model for handing out one per-rank shard of work from a central source.
"""

import torch
import torch.distributed as dist

from pretty_print import print_block


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        scatter_list = None if rank != 0 else [
            torch.tensor([10], dtype=torch.int64),
            torch.tensor([20], dtype=torch.int64),
            torch.tensor([30], dtype=torch.int64),
            torch.tensor([40], dtype=torch.int64),
        ]
        tensor = torch.zeros(1, dtype=torch.int64)
        dist.scatter(tensor, scatter_list=scatter_list, src=0)
        print_block(f"rank {rank}", f"after scatter: {tensor.tolist()}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
