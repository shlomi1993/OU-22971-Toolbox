# Ray Capstone - TLC-Backed Per-Zone Recommendations Under Skew

A replay-based recommendation system built on [Ray](https://www.ray.io/). The system processes NYC Green Taxi trip data in 15-minute windows (ticks), producing a per-zone demand recommendation (`NEED` or `OK`) at every tick. A blocking baseline and an asynchronous controller run the same replay side by side, exposing how skew, bounded concurrency, timeout-driven fallback, and idempotent actor writes affect latency and output correctness.

<img width="2816" height="1536" alt="NYC TLC Green Taxi zone illustration" src="https://github.com/user-attachments/assets/1cd8998a-972a-4a33-965c-3cf8778f40dd" />

## Table of contents

- [Ray Capstone - TLC-Backed Per-Zone Recommendations Under Skew](#ray-capstone---tlc-backed-per-zone-recommendations-under-skew)
  - [Table of contents](#table-of-contents)
  - [Video Walkthrough](#video-walkthrough)
  - [Architecture Overview](#architecture-overview)
    - [Workflow](#workflow)
    - [Decision Rule](#decision-rule)
    - [Partial-readiness Policy](#partial-readiness-policy)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Local Environment Setup](#2-local-environment-setup)
    - [3. Download Data](#3-download-data)
    - [4. Docker Cluster Setup](#4-docker-cluster-setup)
  - [Execution](#execution)
    - [Step 1 - Prepare Replay Assets](#step-1---prepare-replay-assets)
    - [Step 2 - Blocking Baseline](#step-2---blocking-baseline)
    - [Step 3 - Async Controller](#step-3---async-controller)
    - [Step 4 - Stress Test](#step-4---stress-test)
    - [Output Artifacts](#output-artifacts)
    - [Cleanup](#cleanup)
  - [Tests](#tests)
    - [Pytest](#pytest)
    - [Demo](#demo)
  - [Troubleshooting](#troubleshooting)
  - [Summary](#summary)


## Video Walkthrough

[Demo Video](#)


## Architecture Overview


### Workflow

<img width="1672" height="941" alt="System architecture diagram" src="https://github.com/user-attachments/assets/05894dea-62b0-48aa-9ac6-149884320ace" />


### Decision Rule

For each zone at each tick, the system compares recent average demand against the historical baseline for that zone's hour-of-day and day-of-week combination.

The zone receives a **NEED** recommendation if recent demand exceeds the baseline threshold: `baseline_mean + max(baseline_std, 1.0)`. Otherwise, the zone receives **OK**.

The baseline statistics are computed from the reference month, grouped by zone, hour-of-day, and day-of-week. This means each zone has different baseline values for each hour and weekday combination. The standard deviation floor of 1.0 ensures the threshold remains meaningful even for low-variance zones. Zones with no historical data default to **OK**.


### Partial-readiness Policy

The async controller finalizes each tick using:

- **Bounded concurrency** - `max_inflight_zones` limits simultaneous scoring tasks.
- **Timeout** - zones still pending after `tick_timeout_s` seconds are considered late.
- **Fallback** - late zones inherit their previous accepted decision (`always_previous`), and zones without history default to `OK`.

Fallback behavior is deterministic (same inputs + seed = same outcomes) and fully visible in artifacts: `actor_counters.json` tracks `n_fallbacks`, `n_late`, `n_duplicates` per zone; `metrics.csv` tracks `n_zones_fallback` per tick.


## Project Structure

```
main.py                     # The main entry point for the program.
src/
├── prepare.py              # Data preparation script.
├── run.py                  # Replay execution script.
├── common.py               # Shared constants, enums, and dataclasses.
├── data_preparation.py     # Data loading, validation, aggregation, and baseline computation.
├── scoring_task.py         # Ray remote scoring task for zone snapshots.
├── zone_actor.py           # Ray actor implementation for per-zone state management.
├── artifacts.py            # Artifact writers - JSON, CSV, summary, and latency log.
├── logger.py               # Centralized logging configuration with colored output.
└── replay/
    ├── base.py             # Abstract base class for TLC zone recommendation replay execution.
    ├── blocking.py         # Blocking replay implementation.
    └── asynchronous.py     # Asynchronous replay implementation.
scripts/
├── download_data.sh        # Download TLC parquet files in Linux/macOS environment.
├── download_data.ps1       # Download TLC parquet files in Windows environment.
└── reset.py                # Reset script to stop Ray and delete generated artifacts.
tests/
├── conftest.py             # Pytest configuration and fixtures.
├── helpers.py              # Test helper functions.
├── test_core_logic.py      # Tests for data validation, scoring logic, and artifact verification.
├── test_zone_actor.py      # Tests for ZoneActor state management and fault-tolerance.
├── test_workflows.py       # Tests for script-level integration and end-to-end workflow.
└── test_ray_docker.py      # Tests for Docker cluster setup, connectivity, and job submission.
demo.sh                     # Interactive demo script for validation and presentations.
Dockerfile                  # Docker image for Ray cluster nodes
docker-compose.yml          # Multi-node Ray cluster definition (1 head + 2 workers)
pyproject.toml              # Package config and console_scripts entry points
environment.yml             # Conda environment definition
pytest.ini                  # Pytest configuration
```


## Setup


### 1. Prerequisites

- [Conda](https://docs.conda.io/en/latest/) installed (Miniconda is enough)
- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Docker Compose](https://docs.docker.com/compose/install/) installed


### 2. Local Environment Setup

Create and activate the Conda virtual environment:

```bash
conda env create -f environment.yml
conda activate 22971-ray-capstone
```


### 3. Download Data

Download two adjacent monthly Green Taxi parquet files from [TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

**Linux/macOS:**
```bash
bash scripts/download_data.sh
```

**Windows PowerShell:**
```powershell
powershell -File scripts/download_data.ps1
```

This downloads `green_tripdata_2023-01.parquet` (reference) and `green_tripdata_2023-02.parquet` (replay) into `data/`.


### 4. Docker Cluster Setup

The project uses a multi-node Ray cluster to simulate distributed deployment. This demonstrates how the system behaves when actors and tasks are distributed across multiple machines, exposing real-world challenges like network latency, skew, and fault tolerance.

**Build and start the cluster:**

```bash
docker-compose up -d
```

This starts:
- `ray-capstone-head` - Ray head node with dashboard on port 8265
- `ray-capstone-worker-1` - First worker node
- `ray-capstone-worker-2` - Second worker node

**Verify the cluster is running:**

```bash
docker-compose ps
```

All three containers should show "Up" status. The Ray Dashboard is available at http://localhost:8265 and shows all connected nodes.

**View cluster logs (optional):**

```bash
# All nodes
docker-compose logs -f

# Specific node
docker-compose logs -f ray-head
```


## Execution

All execution steps use `ray job submit` to run on the distributed Docker cluster. This ensures the demo simulates real-world distributed deployment where actors and tasks are spread across multiple nodes.

**Preliminary:**
- Ensure the Docker cluster is running (see [Setup §3](#3-docker-cluster-setup))
- Conda environment activated: `conda activate 22971-ray-capstone`

**Commands:**
- `prepare` - Prepare assets for replay execution from raw TLC parquet files
- `run` - Execute replay on the distributed cluster in blocking, async, or stress mode
- `reset` - Stop Ray and delete generated artifacts

**Notes:**
- The `--ray-address auto` flag tells Ray to use the Docker cluster instead of starting a local instance
- Output artifacts are written to the mounted `output/` directory and accessible from the host
- The cluster uses shared memory (`shm_size: 2g`) for efficient data transfer between workers
- Examples below use `--max-ticks 50` for short runs. Omit to process the full month (~2600 ticks)


### Step 1 - Prepare Replay Assets

Preparation runs locally as a one-time data processing step:

```bash
prepare \  # Same as `python main.py prepare` or `python src/prepare.py`
    --ref-parquet data/green_tripdata_2023-01.parquet \
    --replay-parquet data/green_tripdata_2023-02.parquet \
    --output-dir output/prepared \
    --n-zones 20 \
    --seed 42  # For reproducibility
```

This validates the two adjacent-month parquet files, identifies the 20 busiest pickup zones from the reference month, aggregates ticks into 15-minute windows, builds per-zone baselines by `(zone_id, hour_of_day, day_of_week)`, and writes prepared assets to `output/prepared/`.

**Results:**

Prepared assets are written to `output/prepared/` (examples are available in [output_examples/prepared](output_examples/prepared)):
- [active_zones.json](output_examples/prepared/active_zones.json) - List of selected zone IDs.
- [baseline.parquet](output_examples/prepared/baseline.parquet) - Per-zone baseline statistics by `(zone_id, hour_of_day, day_of_week)`.
- [replay.parquet](output_examples/prepared/replay.parquet) - Replay table with `(zone_id, tick_start, demand)` for all active zones.
- [prep_meta.json](output_examples/prepared/prep_meta.json) - Preparation summary including months validated, row counts, tick count, and seed


### Step 2 - Blocking Baseline

**Docker cluster:**

```bash
ray job submit \
    --address http://localhost:8265 \
    --working-dir . \
    -- python main.py run \
        --prepared-dir /workspace/output/prepared \
        --output-dir /workspace/output/run \
        --ray-address auto \
        --mode blocking \
        --slow-zone-fraction 0.25 \
        --slow-zone-sleep-s 1.0 \
        --seed 42 \  # For reproducibility
        --max-ticks 50  # For short runs, omit to process the full month (~2600 ticks)
```

**Local execution:**

```bash
run \  # Same as `python main.py run` or `python src/run.py`
    --prepared-dir output/prepared \
    --output-dir output/run \
    --mode blocking \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --seed 42 \
    --max-ticks 50
```

Runs the replay in blocking mode with simulated skew (25% slow zones, 1s delay). Scoring tasks return decisions to the controller. The controller waits for **all** zones before closing each tick and writes accepted decisions into actors.

**Monitor execution:**
- Ray Dashboard: http://localhost:8265
- Job logs: `ray job logs <job-id>` (ID shown after submission)
- Container logs: `docker-compose logs -f`

**Results:**

Blocking run artifacts are written to `output/run/blocking/` (examples are available in [output_examples/run/blocking](output_examples/run/blocking)):
- [run_config.json](output_examples/run/blocking/run_config.json) - Runtime configuration used for the run.
- [metrics.csv](output_examples/run/blocking/metrics.csv) - Per-tick metrics showing latency, completions, fallbacks, and late/duplicate counts.
- [tick_summary.json](output_examples/run/blocking/tick_summary.json) - Per-tick decisions and metrics for each zone.
- [latency_log.json](output_examples/run/blocking/latency_log.json) - Per-zone per-tick latency entries.
- [actor_counters.json](output_examples/run/blocking/actor_counters.json) - Per-zone duplicate, late, and fallback counters.

**Takeaway:** Blocking is simple but skew-sensitive. A single slow zone dominates tick latency, visible in metrics where `max_zone_latency_s` far exceeds `mean_zone_latency_s`.


### Step 3 - Async Controller

**Docker cluster:**

```bash
ray job submit \
    --address http://localhost:8265 \
    --working-dir . \
    -- python main.py run \
        --prepared-dir /workspace/output/prepared \
        --output-dir /workspace/output/run \
        --ray-address auto \
        --mode async \
        --slow-zone-fraction 0.25 \
        --slow-zone-sleep-s 1.0 \
        --tick-timeout-s 2.0 \
        --completion-fraction 0.75 \
        --max-inflight-zones 4 \
        --seed 42 \
        --max-ticks 50
```

**Local execution:**

```bash
run \
    --prepared-dir output/prepared \
    --output-dir output/run \
    --mode async \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 \
    --completion-fraction 0.75 \
    --max-inflight-zones 4 \
    --seed 42 \
    --max-ticks 50
```

Runs the replay in async mode with simulated skew (25% slow zones, 1s delay), bounded concurrency (max 4 inflight zones), 2s timeout, and 75% completion threshold. Scoring tasks report decisions directly to actors. The driver polls actor readiness and closes ticks under the configured partial-readiness policy. Late zones receive a deterministic fallback.

**Results:**

Run artifacts are written to `output/run/async/` (examples are available in [output_examples/run/async](output_examples/run/async)):
- [run_config.json](output_examples/run/async/run_config.json) - Runtime configuration used for the run.
- [metrics.csv](output_examples/run/async/metrics.csv) - Per-tick metrics showing latency, completions, fallbacks, and late/duplicate counts.
- [tick_summary.json](output_examples/run/async/tick_summary.json) - Per-tick decisions and metrics for each zone.
- [latency_log.json](output_examples/run/async/latency_log.json) - Per-zone per-tick latency entries.
- [actor_counters.json](output_examples/run/async/actor_counters.json) - Per-zone duplicate, late, and fallback counters.

**Takeaway:** Async mode allows ticks to complete without waiting for slow zones, achieving lower total tick latency compared to blocking. The trade-off is increased complexity and reliance on fallback decisions for late zones.


### Step 4 - Stress Test

**Docker cluster:**

```bash
ray job submit \
    --address http://localhost:8265 \
    --working-dir . \
    -- python main.py run \
        --prepared-dir /workspace/output/prepared \
        --output-dir /workspace/output/run \
        --ray-address auto \
        --mode stress \
        --slow-zone-fraction 0.6 \
        --slow-zone-sleep-s 3.0 \
        --tick-timeout-s 2.0 \
        --seed 42 \
        --max-ticks 50
```

**Local execution:**

```bash
run \
    --prepared-dir output/prepared \
    --output-dir output/run \
    --mode stress \
    --slow-zone-fraction 0.6 \
    --slow-zone-sleep-s 3.0 \
    --tick-timeout-s 2.0 \
    --seed 42 \
    --max-ticks 50
```

Runs both blocking and async modes back-to-back with harsher skew (60% slow zones, 3s delay) and writes a side-by-side comparison to evaluate degradation under stress.

**Results:**

Run artifacts are written to `output/run/stress/` (examples are available in [output_examples/run/stress](output_examples/run/stress)):
- [blocking/](output_examples/run/stress/blocking) - Full blocking mode artifacts with harsh skew parameters.
- [async/](output_examples/run/stress/async) - Full async mode artifacts with harsh skew parameters.
- [comparison.json](output_examples/run/stress/comparison.json) - Side-by-side comparison of blocking vs async metrics.

**Takeaway:** Async degrades gracefully with controlled tick completion and more fallbacks. Blocking degrades sharply with mean and max tick latency increasing significantly under harsher skew.


### Output Artifacts

Each run mode writes into its own subdirectory under the specified output directory (`output/run/` by default):

| File | Content |
|---|---|
| `run_config.json` | Runtime configuration used for the run |
| `metrics.csv` | Per-tick metrics: latency, completions, fallbacks, late/duplicate counts |
| `latency_log.json` | Per-zone per-tick latency entries |
| `tick_summary.json` | Per-tick decisions and metrics |
| `actor_counters.json` | Per-zone duplicate/late/fallback counters |
| `comparison.json` | Blocking vs async comparison (stress mode only) |


### Cleanup

**Stop the Docker cluster:**

```bash
docker-compose down
```

**Remove volumes and rebuild from scratch (if needed):**

```bash
docker-compose down -v
```

**Verify Ray stopped and delete output artifacts (optional):**

```bash
python scripts/reset.py  # Or simply "reset" if venv is activated
```


## Tests


### Pytest

The project includes comprehensive test coverage for data validation, actor state management, and end-to-end workflows.

**Quick tests:**

```bash
pytest tests
```
- Runs unit-tests and short integration tests
- Expected duration: **~2 minutes**

**All tests:**

```bash
pytest tests --full
```
- Executes the full test suite, including full integration tests and end-to-end workflows
- Expected duration: **~15 minutes**

**Note:** Pytest is configured to use `-v` (verbose), `-s` (show print statements), and `--strict-markers` (enforce marker usage) by default.

### Demo

The repository includes a demo script that executes the complete workflow from data download through asset preparation and replay run across all modes, with interactive pauses.

```bash
./demo.sh --max-ticks 50
```

**Options:**
- `--keep-artifacts` - Preserve output files after the test completes.
- `--max-ticks N` - Limit the number of ticks to run (e.g., 5 for a fast test, 500 for a longer one), or **omit to process the full month (~2600 ticks)**.
- `--docker` - Run replay jobs on the Docker cluster instead of local Ray.
- `--no-wait` - Run continuously without interactive pauses.

**Docker Flow:** When running in Docker, the script:
1. Starts the Docker cluster via `docker-compose up -d`
2. Runs prepare locally
3. Submits blocking, async, and stress jobs to the cluster via `ray job submit`
4. Verifies the artifacts written to local `output/` via volume mount
5. Stops the cluster


**Important Note:** Docker mode requires a container restart after prepare (step 4) because macOS Docker Desktop doesn't immediately propagate new directories to running containers. This is handled automatically by the script.


## Troubleshooting

**Issue: `python` or `pip` commands use system Python instead of conda environment**

If after setup you see errors like `ModuleNotFoundError: No module named 'ray'` or `externally-managed-environment` when running commands, your shell may have `python` or `pip` aliased to system versions instead of the conda environment.

**Solution:**

```bash
# Check if python/pip are aliased
which python
which pip

# If they point to system Python, unalias them
unalias python
unalias pip

# Verify they now use the conda environment
which python  # Should show /path/to/miniconda3/envs/22971-ray-capstone/bin/python
which pip     # Should show /path/to/miniconda3/envs/22971-ray-capstone/bin/pip
```


## Summary

This project demonstrates distributed systems behavior on a **multi-node Ray cluster** deployed via Docker. All three execution modes - blocking, async, and stress - run on a three-node cluster with one head and two workers, where actors and tasks are distributed across containers, exposing real-world challenges like network latency, skew, and fault tolerance.

Three runs on the same replay data with identical scoring logic reveal the core tradeoff:

- **Blocking** is simple but skew-sensitive. Every tick waits for the slowest zone, so a small number of stragglers dominate latency. On a distributed cluster, this means the entire system waits even when most workers have finished.

- **Async** uses bounded concurrency, timeout, and `always_previous` fallback to let ticks close predictably regardless of individual zone delays. The driver polls distributed actors for readiness instead of blocking on slow workers, achieving graceful degradation under skew.

- **Stress** runs the same comparison under harsher conditions - 60% slow zones with 3 second delay - confirming that blocking degrades sharply while async maintains controlled behavior even when many workers are slow.

**All runs preserve distributed system invariants.** Every write is idempotent, keyed by `zone_id` and `tick_id`, so duplicates and retries never corrupt state. Tasks may complete out of order across workers, but the system never double-counts results. Late reports that arrive after a tick has already closed are logged but ignored. Fallback decisions use deterministic rules that produce identical outputs given identical inputs and seed values, with full tracking in the output artifacts.

The Docker cluster setup makes these distributed behaviors observable through the Ray Dashboard at http://localhost:8265, where you can watch actors migrate between workers, monitor task distribution, and see how the async controller handles uneven completion across nodes.
