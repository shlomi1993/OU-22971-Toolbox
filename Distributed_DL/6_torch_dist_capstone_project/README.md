# Distributed DL Capstone - SimCLR Training with Manual ResNet Sharding

<img width="1672" height="941" alt="Distributed DL Capstone" src="https://github.com/user-attachments/assets/0f798911-3f65-4e18-b1a7-89b3bf739789" />

A distributed contrastive-learning training system that manually shards a ResNet18 across device pairs, trains with a SimCLR-like objective on synthetic ImageNet data, captures per-rank profiler traces, and supports batch-size and split-layer tuning through manual analysis or an automated controller sweep.

The implementation uses only raw `torch.distributed` primitives - no `DistributedDataParallel`, `DistributedSampler`, or pipeline helpers.

Training is orchestrated by the `TrainRunner` class (`train.py`), which encapsulates model construction, replica alignment, the training loop, metric gathering, and persistence. Communication groups are managed by the `CommGroups` class (`src/groups.py`).


## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Execution](#execution)
- [Demo](#demo)
- [Tests](#tests)
- [Bottleneck Categories](#bottleneck-categories)


## Architecture Overview

This section describes how the system **behaves at runtime** - the rank layout, communication groups, the training step, and the loss. The companion [`workflow.mmd`](workflow.mmd) diagram visualizes this end-to-end flow through setup → distributed training step → trace capture → analysis → manual tuning and rerun.

<img width="1672" height="941" alt="Project end-to-end runtime workflow" src="https://github.com/user-attachments/assets/1451f781-e1ca-4aa6-a4f5-0e403248eb4e" />

### Rank Layout

The system requires an even number of ranks (minimum 4). Ranks are paired into model replicas:

| Pair | Even rank (stage 0) | Odd rank (stage 1) |
|------|---------------------|---------------------|
| 0    | Rank 0              | Rank 1              |
| 1    | Rank 2              | Rank 3              |

- **Even ranks** own stage 0 (early ResNet18 layers up to the split point), prepare augmented views, and run stage-0 forward/backward.
- **Odd ranks** own stage 1 (remaining layers + projection head), compute the contrastive loss, and run stage-1 forward/backward.


### Communication Groups

| Group | Members | Purpose |
|-------|---------|---------|
| `pair_group(k)` | `(2k, 2k+1)` | Point-to-point boundary activation and gradient transfer |
| `stage0_group` | All even ranks | Stage-0 gradient synchronization (`all_reduce`) |
| `stage1_group` | All odd ranks | Embedding `all_gather` and stage-1 gradient synchronization |


### Training Step

Each training step follows this sequence:

1. **Prepare views** - even ranks create two augmented views per source image (positive pairs)
2. **Stage-0 forward** - even ranks run the front half of ResNet18
3. **Send boundary** - even ranks send boundary activations to their paired odd rank
4. **Receive boundary** - odd ranks receive the boundary tensor and mark it as requiring gradients
5. **Stage-1 forward** - odd ranks run the back half of ResNet18 + projection head
6. **Gather embeddings** - odd ranks `all_gather` embeddings across `stage1_group`
7. **Compute loss** - odd ranks compute the approximate SimCLR contrastive loss
8. **Backward** - odd ranks run `loss.backward()` and extract the boundary gradient
9. **Send boundary grad** - odd ranks send the boundary gradient back to the even rank
10. **Receive boundary grad** - even ranks receive and continue stage-0 backward
11. **Sync gradients** - `all_reduce` within `stage0_group` and `stage1_group` independently
12. **Optimizer step** - each rank steps its local optimizer


### Loss Calculation

The contrastive loss follows the SimCLR formulation. For each view in the local batch:

- Compute cosine similarity (scaled by temperature) against all gathered embeddings
- The correct class is the view's positive pair, all other views are negatives
- Apply cross-entropy by taking the positive-pair entry of `-log(softmax(similarities))`
- The view itself is excluded from the candidate set via masking

The final loss is the mean over all local view losses.


### Loss Gradient Approximation

Computing exact gradients through the `all_gather` would require differentiable communication, which adds complexity beyond the scope of this project. Instead, only local embeddings (produced on the current rank) remain attached to the autograd graph. Remote embeddings participate in the similarity and softmax computation as fixed (detached) values. This approximation is sufficient for the distributed-systems focus of this capstone.


## Project Structure

This section describes how the **code is organized on disk**. The companion [`architecture.mmd`](architecture.mmd) diagram shows these same modules and how they depend on each other.

```
train.py                        # Distributed training entry point (TrainRunner class, launched via torchrun)
analyze.py                      # Post-hoc trace analysis and metrics reporting
summarize_sweep.py              # Writes manual sweep summary table and diagnosis artifacts
controller.py                   # Load-balancing controller - automated sweep (Stretch B)
demo.sh                         # Interactive demo script for the required demo pattern
architecture.mmd                # Project architecture diagram
workflow.mmd                    # Project end-to-end workflow diagram
design_doc.md                   # Full project specification
pytest.ini                      # Pytest configuration
src/
├── __init__.py
├── common.py                   # Shared constants, enums, and TrainConfig dataclass
├── cli.py                      # CLI argument parser
├── logger.py                   # Colored logging singleton
├── model.py                    # ResNet18 stage splitting and projection head
├── augmentation.py             # SimCLR augmentation pipeline and paired-view creation
├── groups.py                   # CommGroups class - communication group setup (pair, stage0, stage1)
├── contrastive_loss.py         # SimCLR contrastive loss with XOR positive-pair targets
├── training_step.py            # Single training step orchestration with record_function spans
├── profiling.py                # Profiler context manager exporting Chrome traces
├── metrics.py                  # Metrics CSV and run config JSON persistence
scripts/
├── sweep.sh                    # Manual batch-size sweep script
tests/
├── conftest.py                 # Shared fixtures
├── helpers.py                  # Test utilities
├── test_train.py               # Training integration tests
├── test_analyze.py             # Analysis module tests
├── test_controller.py          # Controller tests
```


## Setup

**Prerequisites:**
- [Conda](https://docs.conda.io/en/latest/) installed (Miniconda is enough)
- Linux recommended (devcontainer or WSL); macOS works with the Gloo backend

**Create and activate the environment:**

The required environment file is `Distributed_DL/6_torch_dist_capstone_project/environment.yml`. Run the commands below from the `6_torch_dist_capstone_project` directory so the correct file is picked up. It creates the `22971-td` conda environment.

```bash
conda env create -f environment.yml
conda activate 22971-td
```


## Execution


### Baseline Profiled Run

Launch a baseline training run with profiling enabled:

```bash
torchrun --standalone --nproc_per_node=4 \
    train.py \
    --local-batch-size 8 \
    --num-steps 10 \
    --dataset-size 2048 \
    --profile \
    --run-name baseline
```

This produces per-rank Chrome trace files, a `metrics.csv`, and a `run_config.json` under `output/baseline/`.


### Trace Analysis

Analyze the traces from a completed run:

```bash
python analyze.py --run-dir output/baseline
```

Reports per-rank span tables, compute vs. communication vs. optimizer time breakdown, and stage imbalance ratio.


### Follow-up Run

After inspecting the baseline traces, pick a better batch size and rerun:

```bash
torchrun --standalone --nproc_per_node=4 \
    train.py \
    --local-batch-size 32 \
    --num-steps 10 \
    --dataset-size 2048 \
    --profile \
    --run-name followup
```

Compare `output/baseline/` and `output/followup/` traces and `images_per_sec` from the config files.


### Batch-Size Sweep

Run a manual sweep across multiple batch sizes:

```bash
bash scripts/sweep.sh
```

Sweeps batch sizes `4 8 16 32 64 128` with 4 processes, runs `analyze.py` after each, then writes `output/manual_sweep_summary.csv` and `output/diagnosis_summary.md`.


### Controller Sweep (Stretch B)

The automated controller sweeps batch sizes and split layers, then picks the best configuration by `images/s`:

```bash
python controller.py
python controller.py --batch-sizes 4 8 16 32 --split-layers layer1 layer2
python controller.py --num-steps 10 --dataset-size 2048
```

Results are logged to the console and saved as `output/controller_log.json`.


### Output Artifacts

Each profiled run produces:

| File | Description |
|------|-------------|
| `run_config.json` | Training configuration and `images_per_sec` throughput |
| `metrics.csv` | Per-step, per-rank loss and timing |
| `traces/rank{N}.json` | Chrome trace JSON for each rank (loadable in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev/)) |
| `manual_sweep_summary.csv` | Batch-size sweep table generated by `summarize_sweep.py` |
| `diagnosis_summary.md` | Short trace-backed diagnosis and tuning decision generated by `summarize_sweep.py` |


## Demo

The interactive demo script runs the full required demo pattern (baseline → analysis → follow-up → comparison):

```bash
./demo.sh
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--no-wait` | off | Skip interactive pauses between steps |
| `--nproc N` | 4 | Number of processes |
| `--baseline-bs N` | 8 | Baseline local batch size |
| `--followup-bs N` | 32 | Follow-up local batch size |
| `--num-steps N` | 10 | Training steps per run |


## Tests

Run the full test suite:

```bash
pytest
```

Tests cover training correctness, trace analysis, and controller logic.


## Bottleneck Categories

The trace analysis reveals these bottleneck categories (vocabulary from Units 2-3):

- **Compute** - `stage0_forward`, `stage1_forward`, `loss_calculation`, `stage0_backward`, `prepare_views`
- **Communication** - `send_boundary`, `recv_boundary`, `gather_embeddings`, `send_boundary_grad`, `recv_boundary_grad`, `grad_sync_stage0`, `grad_sync_stage1`
- **Optimizer** - `optimizer_step`

Stage imbalance manifests when one stage's compute time dominates the other, causing the faster stage to idle on boundary transfers. The split-layer choice (`--split-layer`) controls this balance. Communication overhead scales with batch size through larger boundary tensors and embedding gathers.
