"""
Illustrate how a slow rank turns barriers and broadcasts into waiting time.

Run:

    torchrun --standalone --nproc_per_node=4 1_collective_communication/7_barrier_demo.py

Expected result:

- in phase 1, rank ``0`` sleeps before entering ``barrier()``
- the other ranks reach ``barrier()`` first and wait there
- once rank ``0`` enters, everyone is released
- in phase 2, rank ``0`` sleeps before a ``broadcast``, and the other ranks wait inside ``broadcast`` instead
- the script then runs one final ``barrier()`` after the broadcast
- that wait should be small because the broadcast already aligned the ranks

What to notice:

- ``barrier`` does not move useful model data; it synchronizes control flow
- if the previous/next line is already a blocking collective such as ``broadcast``, an extra ``barrier`` is often redundant
- ``barrier`` is most useful around non-collective work such as checkpoint I/O, setup, teardown, or debugging
"""

import time

import torch
import torch.distributed as dist

from pretty_print import print_block


SLOW_RANK = 0
DELAY_SECONDS = 2.0


def sleep_on_slow_rank(rank: int, label: str) -> None:
    if rank == SLOW_RANK:
        print_block(f"rank {rank}", f"{label}: sleeping for {DELAY_SECONDS:.1f}s before the collective")
        time.sleep(DELAY_SECONDS)


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()

        sleep_on_slow_rank(rank, "phase 1")
        barrier_start = time.perf_counter()
        dist.barrier()
        barrier_elapsed = time.perf_counter() - barrier_start
        print_block(f"rank {rank}", f"phase 1 barrier wait: {barrier_elapsed:.2f}s")

        tensor = torch.tensor([99 if rank == 0 else -1], dtype=torch.int64)
        sleep_on_slow_rank(rank, "phase 2")
        broadcast_start = time.perf_counter()
        dist.broadcast(tensor, src=0)
        broadcast_elapsed = time.perf_counter() - broadcast_start
        barrier_start = time.perf_counter()
        dist.barrier()
        barrier_elapsed = time.perf_counter() - barrier_start
        print_block(
            f"rank {rank}",
            f"phase 2 broadcast wait: {broadcast_elapsed:.2f}s",
            f"tensor after broadcast: {tensor.tolist()}",
            f"phase 2 barrier wait: {barrier_elapsed:.2f}s",
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
