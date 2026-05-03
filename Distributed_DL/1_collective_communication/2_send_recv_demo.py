"""
Demonstrate point-to-point send and receive between two ranks.

Run:

    torchrun --standalone --nproc_per_node=2 1_collective_communication/2_send_recv_demo.py

Expected result:

- rank ``0`` sends a tensor with value ``1.0``
- rank ``1`` receives that tensor

What to notice:

- one ``send`` matches one ``recv``
- both sides block by default until the transfer is complete
- if ranks disagree on the order or count of sends and receives, they can hang

Training loop connection:
Point-to-point communication exists, but ordinary training loops lean on collectives.
"""

import torch
import torch.distributed as dist

from pretty_print import print_block


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()

        tensor = torch.zeros(1)

        if rank == 0:
            tensor += 1
            dist.send(tensor=tensor, dst=1)
            print_block(
                f"rank {rank}",
                f"sent {tensor.tolist()}",
                "destination=rank 1",
            )
        elif rank == 1:
            dist.recv(tensor=tensor, src=0)
            print_block(
                f"rank {rank}",
                f"received {tensor.tolist()}",
                "source=rank 0",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
