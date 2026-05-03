"""
Compare synchronous and asynchronous all_reduce under uneven local work.

Run:

    torchrun --standalone --nproc_per_node=4 1_collective_communication/8_async_all_reduce_demo.py

Expected result:

- the script runs one synchronous ``all_reduce``, then one ``all_reduce(async_op=True)``
- both phases also run a fake local function that takes longer on rank ``0``
- rank ``0`` prints a per-rank timing summary for both phases

What to notice:

- in sync mode, the local work starts only after ``all_reduce`` returns
- in async mode, ``all_reduce(..., async_op=True)`` returns a ``Work`` handle immediately, so each rank can do
    independent local work before ``wait()``
"""

import time

import torch
import torch.distributed as dist

from pretty_print import print_block


TENSOR_SIZE = 8_000_000
BASE_WORK_SECONDS = 0.15
EXTRA_SLOW_RANK = 0
EXTRA_DELAY_SECONDS = 0.10


def fake_local_work(rank: int) -> float:
    delay = BASE_WORK_SECONDS + (EXTRA_DELAY_SECONDS if rank == EXTRA_SLOW_RANK else 0.0)
    time.sleep(delay)
    return delay


def gather_metrics(values: list[float]) -> list[list[float]]:
    local = torch.tensor(values, dtype=torch.float64)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [row.tolist() for row in gathered]


def print_rank_zero_summary(title: str, rows: list[list[float]], cols: list[str]) -> None:
    if dist.get_rank() != 0:
        return

    lines = []
    for rank, row in enumerate(rows):
        metrics = ", ".join(f"{name}={value:.2f}s" for name, value in zip(cols, row))
        lines.append(f"rank {rank}: {metrics}")
    print_block(title, *lines)


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()

        tensor = torch.full((TENSOR_SIZE,), float(rank + 1), dtype=torch.float32)
        sync_start = time.perf_counter()
        dist.all_reduce(tensor)
        sync_collective = time.perf_counter() - sync_start

        local_delay = fake_local_work(rank)
        sync_total = time.perf_counter() - sync_start
        sync_rows = gather_metrics([sync_collective, local_delay, sync_total])
        print_rank_zero_summary(title="sync all_reduce", rows=sync_rows, cols=["collective", "local_work", "total"])

        dist.barrier()

        tensor = torch.full((TENSOR_SIZE,), float(rank + 1), dtype=torch.float32)
        async_start = time.perf_counter()
        work = dist.all_reduce(tensor, async_op=True)
        launch_time = time.perf_counter() - async_start

        local_delay = fake_local_work(rank)
        wait_start = time.perf_counter()
        work.wait()
        wait_time = time.perf_counter() - wait_start
        async_total = time.perf_counter() - async_start

        async_rows = gather_metrics([launch_time, local_delay, wait_time, async_total])
        print_rank_zero_summary(title="async all_reduce", rows=async_rows, cols=["launch", "local_work", "wait", "total"])

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
