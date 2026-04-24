# Ray Capstone — TLC-backed per-zone recommendations under skew

Replay-based recommendation system using NYC Green Taxi data. The system walks through historical taxi pickups in 15-minute windows and, for each active zone, decides whether demand looks elevated (`NEED`) or normal (`OK`). The project compares blocking and asynchronous execution strategies under simulated zone-level skew, focusing on actor-owned state, idempotent writes, bounded concurrency, and deterministic fallback behavior.

<img width="2816" height="1536" alt="Gemini_Generated_Image_grl1k1grl1k1grl1" src="https://github.com/user-attachments/assets/7efad955-12fa-4373-a9a3-2321a3442d67" />

---

## Table of contents

1. [Setup](#setup)
2. [Architecture overview](#architecture-overview)
3. [Code walkthrough](#code-walkthrough)
4. [Running the project](#running-the-project)
5. [Demo runs](#demo-runs)
6. [Decision rule](#decision-rule)
7. [Partial-readiness and fallback policy](#partial-readiness-and-fallback-policy)
8. [Failure model and invariants](#failure-model-and-invariants)
9. [Output artifacts](#output-artifacts)
10. [Project structure](#project-structure)
11. [Docker cluster](#docker-cluster)

---

## Setup

```bash
conda env create -f environment.yml
conda activate 22971-ray-capstone
```

**Note for Windows users**: This project includes both bash scripts (`.sh`) and PowerShell scripts (`.ps1`). Use the PowerShell versions. If you want to use the command wrappers (`prepare` and `run`), you'll need Git Bash or WSL bash in your PATH.

Download two adjacent monthly Green Taxi parquet files from [TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

**Linux/macOS:**
```bash
bash scripts/bash/download_data.sh
```

**Windows PowerShell:**
```powershell
powershell -File scripts/powershell/download_data.ps1
```

This downloads `green_tripdata_2023-01.parquet` (reference) and `green_tripdata_2023-02.parquet` (replay) into `data/`.

To reset all generated artifacts and stop Ray:

```bash
python main.py reset
```

Or use the convenience wrapper (installed in conda environment):
```bash
reset
```

---

## Architecture overview

The system follows this control loop:

**reference-month prep → actor initialization → tick replay → per-zone scoring → tick finalization under partial readiness → metrics and artifacts**

<img width="1495" height="1052" alt="ChatGPT Image Apr 23, 2026, 06_38_34 AM" src="https://github.com/user-attachments/assets/f00ab42f-2181-43b1-a542-f3879500205e" />

**Note**: Convenience wrappers `prepare` and `run` are available in the conda environment via `bin/` scripts.

### Key components

- **`ZoneActor`** — one Ray actor per active zone. Owns mutable state: recent demand history, active tick, reported/accepted decisions, observability counters. Only `ZoneActor` mutates durable zone state. Writes are idempotent by `(zone_id, tick_id)`.
- **`score_zone_blocking` / `score_zone_async`** — Ray remote tasks. Receive a `ZoneSnapshot`, compute a deterministic `NEED`/`OK` recommendation. In blocking mode returns a `ZoneRecommendation` to the controller; in async mode reports to the actor before returning.
- **Driver loop** — advances ticks, collects snapshots, launches scoring tasks, finalizes ticks. Two strategies: blocking baseline and asynchronous controller.
- **Skew model** — a configurable fraction of zones receive artificial sleep to simulate uneven completion times.

---

## Code walkthrough

### `main.py` — CLI entry point

Main entry point with three subcommands:

- **`prepare` subcommand**: Parses CLI arguments, calls `src.prepare.prepare_assets()`
- **`run` subcommand**: Parses CLI arguments, creates `ReplayConfig`, calls `src.run.run_replay()`
- **`reset` subcommand**: Calls `src.reset.reset_ray()` to stop Ray and remove generated artifacts

**Usage**:
```bash
python main.py prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet
python main.py run --prepared-dir prepared/ --mode async
```

Or use the convenience wrappers (installed in conda environment):
```bash
prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet
run --prepared-dir prepared/ --mode async
```

### `src/core.py` — shared constants, data functions, artifact writers

- **Constants**: `TICK_MINUTES=15`, `DEFAULT_N_ZONES=20`, `DEFAULT_SEED=42`
- **Enums**: `ReplayMode` (`blocking`/`async`/`stress`)
- **Dataclasses**: `ReplayConfig` (all runtime settings), `TickMetrics` (per-tick performance)
- **Data loading**: `load_parquet()` validates required columns; `validate_adjacent_months()` ensures the two files are consecutive months from the same year
- **Zone selection**: `identify_busiest_zones()` picks the top-N busiest zones from the reference month, deterministic under a fixed seed
- **Baseline**: `aggregate_ticks()` bins pickups into 15-minute windows; `build_baseline_table()` computes `(mean_demand, std_demand)` per `(zone_id, hour_of_day, day_of_week)`
- **Cross-check**: `cross_check_replay()` confirms prepared replay counts match a direct pandas grouped calculation on a sample window
- **Artifact writers**: `write_json()`, `write_metrics_csv()`, `write_tick_summary()`, `write_latency_log()`

### `src/zone_actor.py` — per-zone actor and decision types

- **`Recommendation`**: enum with `NEED`/`OK` values for scoring outcomes
- **`ZoneSnapshot`**: minimal snapshot passed to scoring tasks. Contains `zone_id`, `tick_id`, `recent_demand`, `baseline_mean`, `baseline_std`. The `compute_decision()` method implements the threshold rule.
- **`ZoneRecommendation`**: result from a scoring task with `zone_id`, `tick_id`, `decision`, `task_latency_s`
- **`ZoneActor`**: Ray actor owning mutable state per zone
  - `activate_tick()` / `get_snapshot()` — prepare per-tick state
  - `report_decision()` — async mode: scoring task reports back; idempotent
  - `write_decision()` — blocking mode: controller writes directly; idempotent
  - `finalize_tick()` — async mode: apply reported decision or fallback
  - `has_decision_for_tick()` — async mode: poll readiness
  - `get_accepted_decisions()` — returns full `{tick_id: decision}` history
  - `get_counters()` — returns `ZoneCounters` for observability

### `src/prepare.py` — data preparation

**`prepare_assets()` function:**

1. Loads reference and replay parquets
2. Validates adjacent months
3. Identifies busiest active zones (deterministic with seed)
4. Aggregates both months into 15-minute ticks
5. Builds baseline table from reference month
6. Builds replay table filtered to active zones
7. Runs pandas cross-check
8. Writes prepared assets: `baseline.parquet`, `replay.parquet`, `active_zones.json`, `prep_meta.json`

Called via `python main.py prepare` or the `prepare` wrapper command.

### `src/run.py` — runtime execution

**`run_replay()` function:**
Entry point that initializes Ray context, loads prepared assets, delegates to the appropriate driver (`run_blocking`, `run_async`, or `run_stress`), then writes artifacts and metrics.

Called via `python main.py run` or the `run` wrapper command.

### `src/replay/` — execution mode implementations

- **`base.py`**: Abstract `Replay` base class defining the template method pattern for replay execution. Subclasses implement mode-specific scoring and finalization.
- **`blocking.py`**: `BlockingReplay` subclass and `score_zone_blocking` remote task. Returns `ZoneRecommendation` to the controller.
- **`asynchronous.py`**: `AsyncReplay` subclass and `score_zone_async` remote task. Reports to the actor via `actor_handle.report_decision.remote()` before returning.

**Blocking driver (`BlockingReplay`):**
1. For each tick: activate tick on all actors, collect snapshots
2. Launch all `score_zone_blocking` tasks
3. `ray.get()` all results — waits for every zone
4. Write accepted decisions into actors via `write_decision()`
5. Tick latency is dominated by the slowest zone

**Async driver (`AsyncReplay`):**
1. For each tick: activate tick on all actors, collect snapshots
2. Launch `score_zone_async` tasks with bounded concurrency (`max_inflight_zones`)
3. Use `ray.wait()` with `tick_timeout_s` to collect completed tasks
4. Poll actor readiness via `has_decision_for_tick()`
5. Finalize all actors with `finalize_tick()` — actors with no report get fallback
6. Late zones don't block tick completion

**Stress driver (`run_stress`):**
- Runs both blocking and async with harsher skew: 60% slow zones, 3s sleep
- Writes `comparison.json` with side-by-side latency and fallback stats

---

## Command shortcuts

For convenience, you can install command wrappers that allow you to run `prepare` and `run` directly instead of `python main.py prepare/run`:

**Linux/macOS:**
```bash
# Activate your conda environment
conda activate 22971-ray-capstone

# Install command wrappers
bash scripts/bash/install.sh

# Now you can use shortened commands
prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir prepared/
run --prepared-dir prepared/ --output-dir output/ --mode async
```

**Windows PowerShell:**
```powershell
# Activate your conda environment
conda activate 22971-ray-capstone

# Install command wrappers (requires Git Bash or WSL bash in PATH)
powershell -File scripts/powershell/install.ps1

# Now you can use shortened commands
prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir prepared/
run --prepared-dir prepared/ --output-dir output/ --mode async
```

The examples below use the full `python main.py <subcommand>` syntax, but you can use the shortened commands after installation.

---

## Running the project

### 1. Prepare replay assets

```bash
python main.py prepare \
    --ref-parquet data/green_tripdata_2023-01.parquet \
    --replay-parquet data/green_tripdata_2023-02.parquet \
    --output-dir prepared \
    --n-zones 20 --seed 42
```

### 2. Run blocking baseline

```bash
python main.py run --prepared-dir prepared --output-dir output --mode blocking \
    --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --seed 42
```

### 3. Run async controller

```bash
python main.py run --prepared-dir prepared --output-dir output --mode async \
    --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 --completion-fraction 0.75 --max-inflight-zones 4 --seed 42
```

### 4. Run stress test

```bash
python main.py run --prepared-dir prepared --output-dir output --mode stress \
    --slow-zone-fraction 0.6 --slow-zone-sleep-s 3.0 --tick-timeout-s 2.0 --seed 42
```

### Full system test (all 3 runs)

**Linux/macOS:**
```bash
bash tests/test_ray_flow.sh
```

Add `--keep-artifacts` to retain output after the test completes.

### Unit tests

```bash
pytest tests/ -v
```

Test files: `test_core_logic.py`, `test_zone_actor.py`, `test_workflows.py`.

---

## Demo runs

The demo consists of three separate runs on the same replay data, followed by artifact inspection. Below is the structure used in the walkthrough video.

### Before recording

**Linux/macOS:**
```bash
conda activate 22971-ray-capstone
python main.py reset
bash scripts/bash/download_data.sh
```

**Windows PowerShell:**
```powershell
conda activate 22971-ray-capstone
python main.py reset
powershell -File scripts/powershell/download_data.ps1
```

Have two terminals: one for running commands, one for inspecting artifacts.

### Part 1 — Code walkthrough (~3 min)

Walk through the project files in order:

- **`main.py`**: main entry point with `prepare` and `run` subcommands; imports implementation from prepare and run modules
- **`src/core.py`**: constants, dataclasses (`ReplayConfig`, `TickMetrics`), data loading and validation, zone selection, baseline building, artifact writers
- **`src/zone_actor.py`**: `Recommendation` enum, `ZoneSnapshot.compute_decision()` threshold logic, `ZoneActor` with `activate_tick` / `get_snapshot` / `report_decision` / `write_decision` / `finalize_tick`, idempotent writes by `(zone_id, tick_id)`, duplicate/late counters
- **`src/prepare.py`**: loads 2 adjacent parquets → validates months → identifies busiest zones → aggregates ticks → builds baseline → writes prepared assets; includes pandas cross-check
- **`src/run.py`**: delegates to `BlockingReplay` / `AsyncReplay` classes, `run_stress` runs both with 60% slow zones / 3s sleep
- **`src/replay/`**: `base.py` (abstract `Replay` template), `blocking.py` (`score_zone_blocking` task, `BlockingReplay`), `asynchronous.py` (`score_zone_async` task, `AsyncReplay`)

### Part 2 — Blocking baseline run (~2 min)

```bash
python main.py prepare \
    --ref-parquet data/green_tripdata_2023-01.parquet \
    --replay-parquet data/green_tripdata_2023-02.parquet \
    --output-dir prepared --n-zones 20 --seed 42

python main.py run --prepared-dir prepared --output-dir output --mode blocking \
    --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 --seed 42
```

**What to show:**
- Logs: tick latency dominated by the slowest zones
- `output/blocking/metrics.csv`: point out `max_zone_latency_s` vs `mean_zone_latency_s` — high ratio shows skew impact
- `output/blocking/tick_summary.json`: per-zone decisions per tick
- **Key point**: every tick waits for all zones; skew directly hurts latency

### Part 3 — Async controller run (~2 min)

```bash
python main.py run --prepared-dir prepared --output-dir output --mode async \
    --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 --completion-fraction 0.75 --max-inflight-zones 4 --seed 42
```

**What to show:**
- Logs: tick finalization after timeout, fallback applied to late zones
- `output/async/metrics.csv`: lower `total_tick_latency_s` compared to blocking
- `output/async/actor_counters.json`: `n_late`, `n_duplicates`, `n_fallbacks` per zone
- **Key point**: ticks complete without waiting for stragglers; fallback keeps output semantics clean

### Part 4 — Stress test (~2 min)

```bash
python main.py run --prepared-dir prepared --output-dir output --mode stress \
    --slow-zone-fraction 0.6 --slow-zone-sleep-s 3.0 --tick-timeout-s 2.0 --seed 42
```

**What to show:**
- `output/stress/comparison.json`: blocking vs async side-by-side
- Blocking: much higher mean and max tick latency
- Async: controlled tick completion, more fallbacks but predictable behavior
- **Key point**: async degrades gracefully; blocking degrades sharply under harsher skew

### Part 5 — Summary (~1 min)

- Three runs on the same replay data with the same deterministic scoring logic
- Actor-owned state with idempotent writes keyed by `(zone_id, tick_id)`
- Blocking: simple but skew-sensitive — tick latency = slowest zone
- Async: bounded concurrency + timeout + `always_previous` fallback = controlled degradation
- All invariants hold: no double-counting, late reports logged and ignored, fallback usage visible in artifacts

---

## Decision rule

**NEED** if the zone's recent average demand exceeds `baseline_mean + max(baseline_std, 1.0)` for that zone/hour/day-of-week combination. Otherwise **OK**.

The baseline is built from the reference month, aggregated by `(zone_id, hour_of_day, day_of_week)`. The `max(std, 1.0)` floor prevents the threshold from being trivially tight for low-variance zones.

For the first tick of a zone with no prior history, the default decision is `OK`.

---

## Partial-readiness and fallback policy

The async controller finalizes each tick using:

- **Bounded concurrency**: `max_inflight_zones` limits how many scoring tasks run simultaneously
- **Timeout**: `tick_timeout_s` — after this duration, zones still pending are considered late
- **Fallback policy**: `always_previous` — late zones inherit their previous accepted decision; zones without any history default to `OK`

Policy behavior:
- Explicit in config (`--fallback-policy always_previous` is the default)
- Deterministic: same inputs and seed produce the same fallback decisions
- Visible in logs and artifacts: `actor_counters.json` tracks `n_fallbacks`, `n_late`, `n_duplicates` per zone; `metrics.csv` tracks `n_zones_fallback` per tick

---

## Failure model and invariants

The runtime assumes retries, duplicate delivery, and late arrivals can occur:

- **Idempotent writes**: every actor write is keyed by `(zone_id, tick_id)`. Duplicate writes return `DUPLICATE` status without mutating state.
- **Blocking mode**: the controller does not write the same accepted decision twice for the same `(zone_id, tick_id)`.
- **Async mode**: `report_decision()` rejects duplicate reports (returns `DUPLICATE`) and late reports for inactive/closed ticks (returns `LATE`). `finalize_tick()` is idempotent.
- **Late reports after finalization**: logged and ignored — they do not overwrite accepted outcomes.
- **No double-counting**: final metrics and artifacts derive from actor-accepted state, not raw task completions.
- **Observability**: all duplicate, late, and fallback events are counted in `ZoneCounters` and written to `actor_counters.json`.

---

## Output artifacts

Each run mode writes artifacts into its own subdirectory under `output/`:

| File | Content |
|---|---|
| `run_config.json` | Runtime configuration used for the run |
| `metrics.csv` | Per-tick metrics: latency, completions, fallbacks, late/duplicate counts |
| `latency_log.json` | Per-zone per-tick latency entries |
| `tick_summary.json` | Per-tick decisions and metrics |
| `actor_counters.json` | Per-zone duplicate/late/fallback counters |
| `comparison.json` | Blocking vs async comparison (stress mode only) |

---

## Project structure

| File | Purpose |
|---|---|
| [main.py](main.py) | **Main entry point**: CLI with `prepare` and `run` subcommands |
| [src/core.py](src/core.py) | Shared constants, dataclasses, data loading, scoring logic, artifact writers |
| [src/zone_actor.py](src/zone_actor.py) | `ZoneActor` Ray actor, `ZoneSnapshot`, `ZoneDecision`, fallback logic |
| [src/prepare.py](src/prepare.py) | Preparation module: load parquets, validate, build baseline and replay tables |
| [src/run.py](src/run.py) | Runtime module: delegates to `BlockingReplay`/`AsyncReplay`, stress mode comparison |
| [src/replay/base.py](src/replay/base.py) | Abstract `Replay` base class with template method pattern |
| [src/replay/blocking.py](src/replay/blocking.py) | `BlockingReplay` driver and `score_zone_blocking` remote task |
| [src/replay/asynchronous.py](src/replay/asynchronous.py) | `AsyncReplay` driver and `score_zone_async` remote task |
| [src/reset.py](src/reset.py) | Reset utilities: stop Ray and clean up artifacts |
| [bin/prepare](bin/prepare), [bin/run](bin/run), [bin/reset](bin/reset) | Command wrapper scripts (installed to conda environment) |
| [tests/test_core_logic.py](tests/test_core_logic.py) | Pytest unit tests for core module |
| [tests/test_zone_actor.py](tests/test_zone_actor.py) | Pytest unit tests for zone actor |
| [tests/test_workflows.py](tests/test_workflows.py) | Pytest workflow-level tests |
| [tests/conftest.py](tests/conftest.py) | Shared test fixtures |
| [tests/helpers.py](tests/helpers.py) | Test helper utilities |
| [tests/test_ray_flow.sh](tests/test_ray_flow.sh) | End-to-end system test for Linux/macOS (runs all 3 demo modes, verifies artifacts) |
| [scripts/bash/download_data.sh](scripts/bash/download_data.sh) | Download TLC parquet data into `data/` (Linux/macOS) |
| [scripts/powershell/download_data.ps1](scripts/powershell/download_data.ps1) | Download TLC parquet data into `data/` (Windows) |
| [scripts/bash/install.sh](scripts/bash/install.sh) | Install command wrappers into conda environment (Linux/macOS) |
| [scripts/powershell/install.ps1](scripts/powershell/install.ps1) | Install command wrappers into conda environment (Windows) |
| [scripts/bash/uninstall.sh](scripts/bash/uninstall.sh) | Remove command wrappers from conda environment (Linux/macOS) |
| [scripts/powershell/uninstall.ps1](scripts/powershell/uninstall.ps1) | Remove command wrappers from conda environment (Windows) |
| [pytest.ini](pytest.ini) | Pytest configuration |
| [environment.yml](environment.yml) | Conda environment specification |

---

## Docker cluster

Submit to a Ray cluster via `ray job submit`:

```bash
ray job submit \
    --address http://<head-node>:8265 \
    --working-dir . \
    -- python main.py run --prepared-dir prepared --output-dir output --mode blocking --ray-address auto
```

Replace `--mode blocking` with `async` or `stress` for the other run modes.

---

## Runtime configuration

| Parameter | Description | Default |
|---|---|---|
| `n_zones` | Number of active pickup zones | 20 |
| `tick_minutes` | Width of each replay tick | 15 |
| `max_inflight_zones` | Max concurrent scoring tasks (async) | 4 |
| `tick_timeout_s` | Timeout before finalizing late zones (async) | 2.0 |
| `completion_fraction` | Fraction of zones needed before considering finalization | 0.75 |
| `slow_zone_fraction` | Fraction of zones receiving artificial delay | 0.25 |
| `slow_zone_sleep_s` | Artificial delay in seconds for slow zones | 1.0 |
| `fallback_policy` | Policy for zones without a decision at tick finalization | `"always_previous"` |
| `seed` | Random seed for deterministic zone selection and skew | 42 |
