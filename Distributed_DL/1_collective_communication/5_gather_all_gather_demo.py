"""
Contrast gather on rank 0 with all_gather on every rank.

Run:

    torchrun --standalone --nproc_per_node=4 1_collective_communication/5_gather_all_gather_demo.py

Expected result:

- for ``gather``, rank ``0`` receives ``[[0], [1], [2], [3]]``
- for ``all_gather``, every rank receives ``[[0], [1], [2], [3]]``

What to notice:

- ``gather`` is root-heavy: only the destination rank allocates the full output
- ``all_gather`` is symmetric: every rank allocates and receives the full output

Training loop connection:
Use ``gather`` for rank-0-only reporting and ``all_gather`` when every worker needs the global view before continuing.
"""

import torch
import torch.distributed as dist

from pretty_print import print_block


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.tensor([rank], dtype=torch.int64)
        gather_list = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)] if rank == 0 else None
        dist.gather(tensor, gather_list=gather_list, dst=0)
        if rank == 0:
            gathered = [item.tolist() for item in gather_list]
            print_block(f"rank {rank}", f"after gather: {gathered}")
        tensor_list = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        gathered = [item.tolist() for item in tensor_list]
        print_block(f"rank {rank}", f"after all_gather: {gathered}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
