"""
Compare reduce and all_reduce with a tiny integer tensor.

Run:

    torchrun --standalone --nproc_per_node=4 1_collective_communication/4_reduce_all_reduce_demo.py

Expected result:

- for ``reduce``, only rank ``0`` is guaranteed to finish with ``[10]``
- on the other ranks, the local tensor may contain backend-specific intermediate results, but those values are unspecified
- for ``all_reduce``, every rank finishes with ``[10]``

What to notice:

- ``reduce`` is root-heavy: one rank owns the final answer
- ``all_reduce`` is symmetric: every rank receives the same final answer

Training loop connection:
Use ``reduce`` when only rank ``0`` needs a final scalar; use ``all_reduce`` for synching gradients before the next optimizer step.
"""

import torch
import torch.distributed as dist

from pretty_print import print_block


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        tensor = torch.tensor([rank + 1], dtype=torch.int64)
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            print_block(f"rank {rank}", f"after reduce: {tensor.tolist()}")
        else:
            print_block(f"rank {rank}", "reduce complete", f"local tensor is not guaranteed: {tensor.tolist()}")
        tensor = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print_block(f"rank {rank}", f"after all_reduce: {tensor.tolist()}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
