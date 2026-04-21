# Ray Capstone — TLC-backed per-zone recommendations under skew

Replay-based recommendation system using NYC Green Taxi data. Compares blocking and asynchronous execution strategies under simulated zone-level skew.

## Setup

```bash
conda env create -f environment.yml
conda activate 22971-ray-capstone
```

## Data

Download two adjacent monthly Green Taxi parquet files from [TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

```bash
bash scripts/download_data.sh
```

## Running locally

### 1. Prepare replay assets

```bash
python prepare.py \
    --ref-parquet data/green_tripdata_2023-01.parquet \
    --replay-parquet data/green_tripdata_2023-02.parquet \
    --output-dir prepared \
    --n-zones 20 --seed 42
```

### 2. Run blocking baseline

```bash
python run.py --prepared-dir prepared --output-dir output --mode blocking \
    --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0
```

### 3. Run async controller

```bash
python run.py --prepared-dir prepared --output-dir output --mode async \
    --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 --max-inflight-zones 4
```

### 4. Run stress test

```bash
python run.py --prepared-dir prepared --output-dir output --mode stress
```

### Full system test

```bash
bash tests/test_ray_flow.sh
```

### Unit tests

```bash
pytest tests/test_ray_capstone_project.py -v
```

### Reset

```bash
bash reset.sh
```

## Docker cluster (ray job submit)

```bash
ray job submit --address http://<head-node>:8265 \
    --working-dir . \
    -- python run.py --prepared-dir prepared --output-dir output --mode blocking \
       --ray-address auto
```

## Decision rule

**NEED** if the zone's recent average demand exceeds `baseline_mean + max(baseline_std, 1.0)` for that zone/hour/day-of-week combination. Otherwise **OK**.

Baseline is built from the reference month, aggregated by zone, hour of day, and day of week.

## Partial-readiness policy

The async controller finalizes each tick using:

- **Bounded concurrency**: `max_inflight_zones` limits how many scoring tasks run simultaneously
- **Timeout**: `tick_timeout_s` — after this duration, zones still pending are considered late
- **Fallback**: `always_previous` — late zones inherit their previous accepted decision; zones without history default to `OK`

All fallback usage is logged in `actor_counters.json` and `metrics.csv`.

## Project structure

| File | Purpose |
|---|---|
| `src/tlc.py` | Shared constants, dataclasses, data loading, scoring logic, artifact writers |
| `prepare.py` | Data preparation: load parquets, validate, build baseline and replay tables |
| `run.py` | Runtime: ZoneActor, score_zone task, blocking/async/stress drivers |
| `tests/test_ray_capstone_project.py` | Pytest unit tests for all modules |
| `tests/test_ray_flow.sh` | End-to-end system test script |
| `scripts/download_data.sh` | Download TLC parquet data into `data/` |

## Output artifacts

| File | Content |
|---|---|
| `run_config.json` | Runtime configuration used for the run |
| `metrics.csv` | Per-tick metrics: latency, completions, fallbacks |
| `latency_log.json` | Per-zone per-tick latency entries |
| `tick_summary.json` | Per-tick decisions and metrics |
| `actor_counters.json` | Per-zone duplicate/late/fallback counters |
| `comparison.json` | Stress test: blocking vs async comparison (stress mode only) |
