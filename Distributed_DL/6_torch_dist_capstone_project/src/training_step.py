"""
Single distributed training step for the two-stage sharded SimCLR system.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as T

from torch.profiler import record_function
from typing import Optional

from src.augmentation import create_paired_views
from src.common import TrainConfig
from src.contrastive_loss import simclr_loss
from src.groups import CommGroups
from src.model import boundary_shape


class TrainingStep:
    """
    Encapsulates one distributed training step for the two-stage sharded SimCLR system.

    Implements the full forward-backward-sync cycle across even (stage-0) and odd (stage-1) ranks. Each even-odd pair
    forms one model replica. Even ranks own stage 0, odd ranks own stage 1 plus the contrastive loss.

    Session-level state (model stages, optimizer, groups, config) is captured at construction time.
    Per-step state (boundary activations, loss) is reset on each call to step().

    Design doc steps mapping:
        A: Prepare augmented views         (even only)
        B: Stage-0 forward + send boundary (even only)
        C: Receive boundary + stage-1 fwd  (odd only)
        D: Gather embeddings + loss        (odd only)
        E: Boundary gradient return        (odd sends, even receives)
        F: Stage-local gradient sync       (each stage group)
        G: Optimizer step                  (all ranks)
    """

    def __init__(self, dataset: torch.utils.data.Dataset, config: TrainConfig, groups: CommGroups,
                 stage0: nn.Sequential, stage1: nn.Sequential, optimizer: torch.optim.Optimizer, transform: T.Compose,
                 check_alignment: bool = False) -> None:
        """
        Args:
            dataset (torch.utils.data.Dataset): FakeData dataset.
            config (TrainConfig): Training configuration.
            groups (CommGroups): Communication groups and rank metadata.
            stage0 (nn.Sequential): Stage 0 of the sharded model (even ranks only use this).
            stage1 (nn.Sequential): Stage 1 of the sharded model (odd ranks only use this).
            optimizer (torch.optim.Optimizer): Optimizer for this rank's parameters.
            transform (T.Compose): SimCLR augmentation transform.
            check_alignment (bool, optional): If True, verify replica alignment after each step. Default is False.
        """
        # Session-level state
        self.dataset = dataset
        self.config = config
        self.groups = groups
        self.stage0 = stage0
        self.stage1 = stage1
        self.optimizer = optimizer
        self.transform = transform
        self.check_alignment = check_alignment

        # Precompute shapes and parameter lists
        self._n_views = 2 * config.local_batch_size
        self._bshape = boundary_shape(config.split_layer, self._n_views)
        self._stage0_params = list(stage0.parameters())
        self._stage1_params = list(stage1.parameters())

        # Per-step state, reset in step()
        self._boundary: Optional[torch.Tensor] = None
        self._loss: Optional[torch.Tensor] = None
        self.loss_value: Optional[float] = None

        # Overlap state (Stretch A: async forward/backward overlap on even ranks)
        self._overlap_state: dict[int, torch.Tensor] = {}
        self._overlap_pending: Optional[int] = None

    @staticmethod
    def _assert_finite(tensor: torch.Tensor, label: str) -> None:
        """
        Assert that a tensor contains only finite values.

        Args:
            tensor (torch.Tensor): Tensor to check.
            label (str): Name used in the error message.
        """
        assert torch.isfinite(tensor).all(), f"{label} contains non-finite values"

    @staticmethod
    def _interleave_views(view_1: torch.Tensor, view_2: torch.Tensor) -> torch.Tensor:
        """
        Interleave two view batches so that source image i maps to indices 2i and 2i+1.

        Args:
            view_1 (torch.Tensor): First augmented view batch, shape (B, C, H, W).
            view_2 (torch.Tensor): Second augmented view batch, shape (B, C, H, W).

        Returns:
            torch.Tensor: Interleaved tensor of shape (2B, C, H, W).
        """
        batch_size = view_1.shape[0]
        interleaved = torch.empty(2 * batch_size, *view_1.shape[1:], dtype=view_1.dtype, device=view_1.device)
        interleaved[0::2] = view_1
        interleaved[1::2] = view_2
        return interleaved

    @staticmethod
    def _check_replica_alignment(params: list[nn.Parameter], group: dist.ProcessGroup, label: str) -> None:
        """
        Verify that the first parameter is identical across replicas within a stage group.

        Args:
            params (list[nn.Parameter]): Parameter list for this stage.
            group (dist.ProcessGroup): Stage communication group.
            label (str): Label for the assertion message.
        """
        if len(params) == 0:
            return

        p = params[0].data.clone()
        ref = p.clone()
        dist.broadcast(ref, src=dist.get_global_rank(group, 0), group=group)
        assert torch.allclose(p, ref, atol=1e-6), f"Replica misalignment in {label}"

    def _even_forward(self, step_idx: int) -> None:
        """
        Even-rank forward: prepare views (Step A), run stage 0 and send boundary (Step B).

        Args:
            step_idx (int): Current training step index (used for dataset offset).
        """
        # Step A: Prepare local views
        with record_function("prepare_views"):
            n = len(self.dataset)
            start = (step_idx * self.config.local_batch_size) % n
            images = torch.stack([self.dataset[(start + i) % n][0] for i in range(self.config.local_batch_size)])
            view_1, view_2 = create_paired_views(images, self.transform)
            views = self._interleave_views(view_1, view_2)

        # Step B: Run stage 0
        with record_function("stage0_forward"):
            boundary = self.stage0(views)

        assert boundary.shape == self._bshape, f"Stage 0 output shape {boundary.shape} != expected {self._bshape}"
        self._assert_finite(boundary, "stage0 output")
        self._boundary = boundary

        # Send boundary to paired odd rank
        with record_function("send_boundary"):
            dist.send(boundary.detach().contiguous(), dst=self.groups.pair_rank, group=self.groups.pair_group)

    def _odd_forward(self) -> None:
        """
        Odd-rank forward: receive boundary (Step C), run stage 1, gather embeddings, compute loss (Step D).
        """
        # Step C: Receive boundary activation
        with record_function("recv_boundary"):
            boundary_recv = torch.empty(self._bshape)
            dist.recv(boundary_recv, src=self.groups.pair_rank, group=self.groups.pair_group)

        # Require grad on the received boundary for backprop (Step E)
        boundary_recv.requires_grad_(True)
        self._boundary = boundary_recv

        # Run stage 1
        with record_function("stage1_forward"):
            local_embeddings = self.stage1(boundary_recv)

        self._assert_finite(local_embeddings, "stage1 output")

        # Gather embeddings across stage1_group
        with record_function("gather_embeddings"):
            gathered = [torch.empty_like(local_embeddings) for _ in range(self.groups.num_pairs)]
            dist.all_gather(gathered, local_embeddings.detach().contiguous(), group=self.groups.stage1_group)
            stage1_rank = dist.get_group_rank(self.groups.stage1_group, self.groups.rank)
            gathered[stage1_rank] = local_embeddings
            all_embeddings = torch.cat(gathered, dim=0)

        # Step D: Compute contrastive loss
        with record_function("loss_calculation"):
            local_start_idx = stage1_rank * self._n_views
            loss = simclr_loss(local_embeddings, all_embeddings, local_start_idx, self.config.temperature)

        self._assert_finite(loss, "loss")

        self.loss_value = loss.item()
        self._loss = loss


    def _odd_backward(self) -> None:
        """
        Odd-rank backward: run loss.backward(), send boundary gradient back to even rank (Step E).
        """
        self._loss.backward()
        self._assert_finite(self._boundary.grad, "boundary gradient")
        with record_function("send_boundary_grad"):
            dist.send(self._boundary.grad.contiguous(), dst=self.groups.pair_rank, group=self.groups.pair_group)

    def _even_backward(self) -> None:
        """
        Even-rank backward: receive boundary gradient, run stage 0 backward (Step E continued).
        """
        with record_function("recv_boundary_grad"):
            boundary_grad = torch.empty(self._bshape)
            dist.recv(boundary_grad, src=self.groups.pair_rank, group=self.groups.pair_group)

        self._assert_finite(boundary_grad, "received boundary gradient")

        with record_function("stage0_backward"):
            self._boundary.backward(boundary_grad)

    def _sync_gradients(self, params: list[nn.Parameter], group: dist.ProcessGroup) -> None:
        """
        Average gradients across replicas within a stage group using all_reduce.

        Args:
            params (list[nn.Parameter]): Parameters whose gradients should be synced.
            group (dist.ProcessGroup): The stage group for the all_reduce.
        """
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
                p.grad.div_(self.groups.num_pairs)

    def _sync_and_step(self) -> None:
        """
        Gradient synchronization (Step F) and optimizer step (Step G) with optional alignment check.
        """
        if self.groups.is_even:
            with record_function("grad_sync_stage0"):
                self._sync_gradients(self._stage0_params, self.groups.stage0_group)
        else:
            with record_function("grad_sync_stage1"):
                self._sync_gradients(self._stage1_params, self.groups.stage1_group)

        with record_function("optimizer_step"):
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.check_alignment:
            if self.groups.is_even:
                self._check_replica_alignment(self._stage0_params, self.groups.stage0_group, "stage0")
            else:
                self._check_replica_alignment(self._stage1_params, self.groups.stage1_group, "stage1")

    def _step_overlap(self, step_idx: int) -> Optional[float]:
        """
        Overlap-mode step (Stretch A): even ranks pipeline forward and backward across consecutive steps.

        Even ranks do backward(t-1) then forward(t) in the same call, keeping at most one in-flight step.
        The first call (no pending backward) is a "prime" that only does forward. After the last step, drain_overlap()
        processes the final backward. Odd ranks run the standard sequential path unchanged.

        The overlap benefit: while the odd rank processes step t (stage1_forward + loss + backward + send_grad), the
        even rank is free to do backward(t-1) + sync + optim + forward(t) concurrently.

        Args:
            step_idx (int): Current training step index.

        Returns:
            Optional[float]: Loss value on odd ranks, None on even ranks.
        """
        if self.groups.is_even:
            # Backward for previous step if one is pending (recv grad first, then backward + sync + optim)
            if self._overlap_pending is not None:
                self._boundary = self._overlap_state.pop(self._overlap_pending)
                self._even_backward()
                self._sync_and_step()

            # Forward for current step and hand off boundary to paired odd rank
            self._even_forward(step_idx)
            self._overlap_state[step_idx] = self._boundary
            self._overlap_pending = step_idx
        else:
            # Odd rank: unchanged sequential processing
            self._odd_forward()
            self._odd_backward()
            self._sync_and_step()

        return self.loss_value

    def drain_overlap(self) -> Optional[float]:
        """
        Drain the overlap pipeline after the last training step.

        On even ranks, processes the final pending backward pass that was deferred by the overlap. On odd
        ranks and in non-overlap mode, returns None immediately.

        Returns:
            Optional[float]: Always None (even ranks have no loss).
        """
        if not self.config.overlap or not self.groups.is_even or self._overlap_pending is None:
            return None

        self._boundary = self._overlap_state.pop(self._overlap_pending)
        self._even_backward()
        self._sync_and_step()
        self._overlap_pending = None
        return self.loss_value

    def step(self, step_idx: int) -> Optional[float]:
        """
        Execute one full distributed training step.

        In synchronous mode, runs the standard forward-backward-sync cycle. In overlap mode (Stretch A), even ranks
        pipeline forward and backward across consecutive steps.

        Args:
            step_idx (int): Current training step index (used for dataset offset).

        Returns:
            Optional[float]: Loss value on odd ranks, None on even ranks.
        """
        self._boundary = None
        self._loss = None
        self.loss_value = None

        if self.config.overlap:
            return self._step_overlap(step_idx)

        if self.groups.is_even:
            self._even_forward(step_idx)
            self._even_backward()
        else:
            self._odd_forward()
            self._odd_backward()

        self._sync_and_step()
        return self.loss_value
