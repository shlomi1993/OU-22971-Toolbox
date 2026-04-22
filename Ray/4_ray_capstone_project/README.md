# Ray Capstone — TLC-backed per-zone recommendations under skew

Replay-based recommendation system using NYC Green Taxi data. The system walks through historical taxi pickups in 15-minute windows and, for each active zone, decides whether demand looks elevated (`NEED`) or normal (`OK`). The project compares blocking and asynchronous execution strategies under simulated zone-level skew, focusing on actor-owned state, idempotent writes, bounded concurrency, and deterministic fallback behavior.

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
bash scripts/download_data.sh
```

**Windows PowerShell:**
```powershell
powershell -File scripts/download_data.ps1
```

This downloads `green_tripdata_2023-01.parquet` (reference) and `green_tripdata_2023-02.parquet` (replay) into `data/`.

To reset all generated artifacts and stop Ray:

**Linux/macOS:**
```bash
bash scripts/reset_ray.sh
```

**Windows PowerShell:**
```powershell
powershell -File scripts/reset_ray.ps1
```

---

## Architecture overview

The system follows this control loop:

**reference-month prep → actor initialization → tick replay → per-zone scoring → tick finalization under partial readiness → metrics and artifacts**

```
                     ┌────────────────┐
                     │    main.py     │  (CLI entry point)
                     │                │
                     │  prepare | run │
                     └───┬────────┬───┘
                         │        │
         ┌───────────────┘        └───────────────────────┐
         ▼                                                ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  src/prepare.py │────▶│ prepared assets  │────▶│    src/run.py    │
│                 │     │ baseline.parquet │     │                  │
│ prepare_assets()│     │  replay.parquet  │     │   run_replay()   │
│                 │     │ active_zones.json│     │                  │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                            ┌─────────────────────────────┤
                            ▼                             ▼
                     ┌─────────────┐          ┌──────────────────┐
                     │  ZoneActor  │ ×N       │   score_zone()   │
                     │  (per zone) │◀─────────│   remote tasks   │
                     │  owns state │          │   (per zone)     │
                     └─────────────┘          └──────────────────┘
```

**Note**: Convenience wrappers `prepare` and `run` are available in the conda environment via `bin/` scripts.

### Key components

- **`ZoneActor`** — one Ray actor per active zone. Owns mutable state: recent demand history, active tick, reported/accepted decisions, observability counters. Only `ZoneActor` mutates durable zone state. Writes are idempotent by `(zone_id, tick_id)`.
- **`score_zone`** — Ray remote task. Receives a `ZoneSnapshot`, computes a deterministic `NEED`/`OK` decision. In blocking mode returns to the controller; in async mode reports to the actor.
- **Driver loop** — advances ticks, collects snapshots, launches scoring tasks, finalizes ticks. Two strategies: blocking baseline and asynchronous controller.
- **Skew model** — a configurable fraction of zones receive artificial sleep to simulate uneven completion times.

---

## Code walkthrough

### `main.py` — CLI entry point

Main entry point with two subcommands:

- **`prepare` subcommand**: Parses CLI arguments, calls `src.prepare.prepare_assets()`
- **`run` subcommand**: Parses CLI arguments, initializes Ray, creates `RunConfig`, calls `src.run.run_replay()`

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

### `src/tlc.py` — shared constants, data functions, artifact writers

- **Constants**: `TICK_MINUTES=15`, `DEFAULT_N_ZONES=20`, `DEFAULT_SEED=42`
- **Enums**: `Decision` (`NEED`/`OK`), `RunMode` (`blocking`/`async`/`stress`)
- **Dataclasses**: `RunConfig` (all runtime settings), `TickMetrics` (per-tick performance)
- **Data loading**: `load_parquet()` validates required columns; `validate_adjacent_months()` ensures the two files are consecutive months from the same year
- **Zone selection**: `select_active_zones()` picks the top-N busiest zones from the reference month, deterministic under a fixed seed
- **Baseline**: `aggregate_ticks()` bins pickups into 15-minute windows; `build_baseline_table()` computes `(mean_demand, std_demand)` per `(zone_id, hour_of_day, day_of_week)`
- **Cross-check**: `cross_check_replay()` confirms prepared replay counts match a direct pandas grouped calculation on a sample window
- **Artifact writers**: `write_json()`, `write_metrics_csv()`, `write_tick_summary()`, `write_latency_log()`

### `src/zone_actor.py` — per-zone actor and decision types

- **`ZoneSnapshot`**: minimal snapshot passed to scoring tasks. Contains `zone_id`, `tick_id`, `recent_demand`, `baseline_mean`, `baseline_std`. The `compute_decision()` method implements the threshold rule.
- **`src/prepare.py` — data preparation

**`prepare_assets()` function:**

1. Loads reference and replay parquets
2. Validates adjacent months
3. Selects active zones (deterministic with seed)
4. Aggregates both months into 15-minute ticks
5. Builds baseline table from reference month
6. Builds replay table filtered to active zones
7. Runs pandas cross-check
8. Writes prepared assets: `baseline.parquet`, `replay.parquet`, `active_zones.json`, `prep_meta.json`

Called via `python main.py prepare` or the `prepare` wrapper command.ack; idempotent
  - `get_accepted_decisions()` — returns full `{tick_id: decision}` history
  - `get_counters()` — returns `ZoneCounters` for observability

### `prepare.py` — data preparation (Steps A–B)

1. Loads reference and replay parquets
2. Validates adjacent months
3. Selects active zones (deterministic with seed)
4. Aggregates both months into 15-minute ticks
5. Builds baseline table from reference month
6. Busrc/run.py` — runtime execution

**`run_replay()` function:**
Entry point that initializes Ray context, loads prepared assets, creates actors, delegates to the appropriate driver (`run_blocking`, `run_async`, or `run_stress`), then writes artifacts and metrics.

Called via `python main.py run` or the `run` wrapper command. active zones
7. Runs pandas cross-check
8. Writes prepared assets: `baseline.parquet`, `replay.parquet`, `active_zones.json`, `prep_meta.json`

### `run.py` — runtime (Steps C–H)

**`score_zone` remote task:**
- Receives a `ZoneSnapshot` and optional `slow_sleep_s` for skew simulation
- Calls `snapshot.compute_decision()` — deterministic from input
- In blocking mode: returns `ZoneDecision` to the controller
- In async mode: calls `actor_handle.report_decision.remote()` before returning
- Retry-safe: task is stateless, reports are idempotent at the actor

**Blocking driver (`run_blocking`):**
1. For each tick: activate tick on all actors, collect snapshots
2. Launch all `score_zone` tasks
3. `ray.get()` all results — waits for every zone
4. Write accepted decisions into actors via `write_decision()`
5. Tick latency is dominated by the slowest zone

**Async driver (`run_async`):**
1. For each tick: activate tick on all actors, collect snapshots
2. Launch scoring tasks with bounded concurrency (`max_inflight_zones`)
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
bash scripts/install.sh

# Now you can use shortened commands
prepare --ref-parquet data/2023-01.parquet --replay-parquet data/2023-02.parquet --output-dir prepared/
run --prepared-dir prepared/ --output-dir output/ --mode async
```

**Windows PowerShell:**
```powershell
# Activate your conda environment
conda activate 22971-ray-capstone

# Install command wrappers (requires Git Bash or WSL bash in PATH)
powershell -File scripts/install.ps1

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

**Windows PowerShell:**
```powershell
powershell -File tests/test_ray_flow.ps1
```

Add `--keep-artifacts` (bash) or `-KeepArtifacts` (PowerShell) to retain output after the test completes.

### Unit tests

```bash
pytest tests/test_ray_capstone_project.py -v
```

---

## Demo runs

The demo consists of three separate runs on the same replay data, followed by artifact inspection. Below is the structure used in the walkthrough video.

### Before recording

**Linux/macOS:**
```bash
conda activate 22971-ray-capstone
bash scripts/reset_ray.sh
bash scripts/download_data.sh
```

**Windows PowerShell:**
```powershell
conda activate 22971-ray-capstone
powershell -File scripts/reset_ray.ps1
powershell -File scripts/download_data.ps1
```

Have two terminals: one for running commands, one for inspecting artifacts.

### Part 1 — Code walkthrough (~3 min)

Walk through the project files in order:

- **`main.py`**: main entry point with `prepare` and `run` subcommands; imports implementation from prepare and run modules
- **`src/tlc.py`**: constants, dataclasses (`RunConfig`, `TickMetrics`), data loading and validation, zone selection, baseline building, scoring rule, fallback policy, artifact writers
- **`src/zone_actor.py`**: `ZoneSnapshot.compute_decision()` threshold logic, `ZoneActor` with `activate_tick` / `get_snapshot` / `report_decision` / `write_decision` / `finalize_tick`, idempotent writes by `(zone_id, tick_id)`, duplicate/late counters
- **`prepare.py`**: loads 2 adjacent parquets → validates months → selects zones → aggregates ticks → builds baseline → writes prepared assets; includes pandas cross-check
- **`run.py`**: `score_zone` remote task (blocking returns to controller, async reports to actor), `run_blocking` (waits for all), `run_async` (bounded concurrency + timeout + polling + fallback), `run_stress` (both with 60% slow zones / 3s sleep)

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
| `main.py` | **Main entry point**: CLI with `prepare` and `run` subcommands |
| `src/tlc.py` | Shared constants, dataclasses, data loading, scoring logic, artifact writers |
| `src/zone_actor.py` | `ZoneActor` Ray actor, `ZoneSnapshot`, `ZoneDecision`, fallback logic |
| `src/prepare.py` | Preparation module: load parquets, validate, build baseline and replay tables |
| `src/run.py` | Runtime module: `score_zone` task, blocking/async/stress drivers, artifact writing |
| `bin/prepare`, `bin/run` | Command wrapper scripts (installed to conda environment) |
| `tests/test_ray_capstone_project.py` | Pytest unit tests for all modules |
| `tests/test_ray_flow.sh` | End-to-end system test for Linux/macOS (runs all 3 demo modes, verifies artifacts) |
| `tests/test_ray_flow.ps1` | End-to-end system test for Windows PowerShell |
| `scripts/download_data.sh` | Download TLC parquet data into `data/` (Linux/macOS) |
| `scripts/download_data.ps1` | Download TLC parquet data into `data/` (Windows) |
| `scripts/reset_ray.sh` | Stop Ray, remove `prepared/` and `output/` directories (Linux/macOS) |
| `scripts/reset_ray.ps1` | Stop Ray, remove `prepared/` and `output/` directories (Windows) |
| `scripts/install.sh` | Install command wrappers into conda environment (Linux/macOS) |
| `scripts/install.ps1` | Install command wrappers into conda environment (Windows) |
| `scripts/uninstall.sh` | Remove command wrappers from conda environment (Linux/macOS) |
| `scripts/uninstall.ps1` | Remove command wrappers from conda environment (Windows) |
| `environment.yml` | Conda environment specification |

---

## Docker cluster

Submit to a Ray cluster via `ray job submit`:

```bash
ray job submit --address http://<head-node>:8265 \
    --working-dir . \
    -- python main.py run --prepared-dir prepared --output-dir output --mode blocking \
       --ray-address auto
```

Replace `--mode blocking` with `async` or `stress` for the other run modes.

---

## Runtime configuration

| Parameter | Default | Description |
|---|---|---|
| `n_zones` | 20 | Number of active pickup zones |
| `tick_minutes` | 15 | Width of each replay tick |
| `max_inflight_zones` | 4 | Max concurrent scoring tasks (async) |
| `tick_timeout_s` | 2.0 | Timeout before finalizing late zones (async) |
| `completion_fraction` | 0.75 | Fraction of zones needed before considering finalization |
| `slow_zone_fraction` | 0.25 | Fraction of zones receiving artificial delay |
| `slow_zone_sleep_s` | 1.0 | Artificial delay in seconds for slow zones |
| `fallback_policy` | `always_previous` | Policy for zones without a decision at tick finalization |
| `seed` | 42 | Random seed for deterministic zone selection and skew |
