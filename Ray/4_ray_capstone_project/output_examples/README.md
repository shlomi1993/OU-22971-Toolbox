# Example Output Artifacts

This directory contains example output artifacts from a complete test run of the Ray Capstone project.

## Generation Commands

These artifacts were created by running the full test suite from the project root:

**1. Download TLC data:**
```bash
bash scripts/download_data.sh
```

**2. Prepare replay assets:**
```bash
prepare \
    --ref-parquet data/green_tripdata_2023-01.parquet \
    --replay-parquet data/green_tripdata_2023-02.parquet \
    --output-dir output_examples/prepared \
    --n-zones 20 \
    --seed 42
```

**3. Run blocking baseline:**
```bash
run \
    --prepared-dir output_examples/prepared \
    --output-dir output_examples/run \
    --mode blocking \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --max-ticks 50 \
    --seed 42
```

**4. Run async controller:**
```bash
run \
    --prepared-dir output_examples/prepared \
    --output-dir output_examples/run \
    --mode async \
    --slow-zone-fraction 0.25 \
    --slow-zone-sleep-s 1.0 \
    --tick-timeout-s 2.0 \
    --completion-fraction 0.75 \
    --max-inflight-zones 4 \
    --max-ticks 50 \
    --seed 42
```

**5. Run stress test:**
```bash
run \
    --prepared-dir output_examples/prepared \
    --output-dir output_examples/run \
    --mode stress \
    --slow-zone-fraction 0.6 \
    --slow-zone-sleep-s 3.0 \
    --tick-timeout-s 2.0 \
    --max-ticks 50 \
    --seed 42
```