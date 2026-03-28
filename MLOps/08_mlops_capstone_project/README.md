# MLOps Capstone Project - Green Taxi Tip Prediction

A pipeline for monitoring model performance, detecting degradation, automatically retraining when needed, and promoting better models to production.

Built with **Metaflow** for orchestration, **NannyML** for drift detection, and **MLflow** for experiment tracking and model registry.

**Pipeline Flow:**  
New batch → integrity checks → feature engineering → performance evaluation → retrain (if degraded) → promote (if improved) → champion model updated

## 🎥 Demo Video

Watch the complete demo walkthrough: [MLOps Capstone Demo](https://drive.google.com/file/d/1Oo6qLciW6hLclkFiXSVzqebLdbAvpDz6/view?usp=sharing)

## Setup

1. **Create / activate the conda environment:**

   ```bash
   conda env create -f environment.yml   # first time only
   conda activate 22971-mlflow
   ```

2. **Reset local artifacts (recommended before a fresh demo run):**

   ```bash
   ./reset.sh
   ```

   This script kills any running MLflow server and removes all tracking artifacts such as runs, models and database.

3. **Place data files** (parquet) under `TLC_data/`:
   - [`green_tripdata_2020-01.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-01.parquet) (used for reference and baseline batch)
   - [`green_tripdata_2020-04.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-04.parquet) (Run 2 — triggers retrain due to COVID-19 shift)
   - [`green_tripdata_2020-06.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-06.parquet) (Run 3 — for resume demo)

   Or download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

4. **Start the MLflow tracking server:**

   ```bash
   mlflow server \
       --workers 1 \
       --port 5001 \
       --backend-store-uri sqlite:///mlflow_tracking/mlflow.db \
       --default-artifact-root mlflow_tracking/mlruns
   ```

5. **Open the MLflow UI** at http://localhost:5001. Look for experiment **`08_capstone_green_taxi`**.

## Pipeline Implementation

### Flow Steps Overview

The pipeline consists of 9 Metaflow steps that execute sequentially with conditional branching:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                           start                                 │
│                             │                                   │
│                             v                                   │
│                         load_data                               │
│                             │                                   │
│                             v                                   │
│                      integrity_gate                             │
│                             │                                   │
│                ┌────────────┴────────────┐                      │
│                │                         │                      │
│          (accepted)                 (rejected)                  │
│                │                         │                      │
│                v                         │                      │
│       feature_engineering                │                      │
│                │                         │                      │
│                v                         │                      │
│         load_champion                    │                      │
│                │                         │                      │
│                v                         │                      │
│           model_gate                     │                      │
│                │                         │                      │
│      ┌─────────┴─────────┐               │                      │
│      │                   │               │                      │
│ (retrain needed)   (no retrain)          │                      │
│      │                   │               │                      │
│      v                   │               │                      │
│   retrain                │               │                      │
│      │                   │               │                      │
│      v                   │               │                      │
│ promotion_gate           │               │                      │
│      │                   │               │                      │
│      └───────────────────┴───────────────┘                      │
│                          │                                      │
│                          v                                      │
│                         end                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Decision Points:**
- **integrity_gate**: Branches to `end` if hard rules fail, otherwise continues to `feature_engineering`
- **model_gate**: Branches to `retrain` if RMSE degradation exceeds threshold, otherwise branches to `end`
- **Conditional steps**: `retrain` and `promotion_gate` only execute when retraining is triggered

1. **start** - Initialize MLflow tracking and model registry
2. **load_data** - Load reference and batch datasets from parquet files
3. **integrity_gate** - Run hard rules and NannyML checks (branches to `end` if batch rejected)
4. **feature_engineering** - Transform raw data into model-ready features
5. **load_champion** - Load existing champion or bootstrap new one
6. **model_gate** - Evaluate champion and decide if retraining needed (branches to `retrain` or `end`)
7. **retrain** - Train candidate model on combined data (conditional)
8. **promotion_gate** - Evaluate candidate and promote if better (conditional)
9. **end** - Log final outcome

### Integrity Gate

The integrity checks run on raw batch data before feature engineering to catch schema and data quality issues early.

**Hard Rules** - fail-fast:
- Required columns present:
  -  `lpep_pickup_datetime`
  -  `trip_distance`
  -  `fare_amount`
  -  `tip_amount`
  -  `passenger_count`
  -  `PULocationID`
  -  `DOLocationID`
  -  `payment_type`
  -  `trip_type`.
- Valid datetime ranges for pickup and dropoff times
- No impossible values: negative distances, negative durations where dropoff before pickup, values outside acceptable ranges
- Target variable availability for evaluation: tip_amount not more than 50% null

If hard rules fail, the batch is rejected with `action=reject_batch` logged to `decision.json` and the flow stops.

**NannyML Checks** - soft warnings:
- Missing value drift using NannyML `MissingValuesCalculator`
- Univariate distribution drift using NannyML `UnivariateDriftCalculator`
- Multivariate drift using NannyML `DataReconstructionDriftCalculator` with PCA-based reconstruction error
- Unseen categorical values in `PULocationID`, `DOLocationID` and `payment_type` features

Soft warnings are logged with `integrity_warn=true` tag but do not stop execution. Results are saved to `nannyml_details.json` artifact.

Implementation: See [capstone_lib.py](capstone_lib.py) functions `run_hard_integrity_checks()`, `run_soft_integrity_checks()`, and `run_integrity_checks()`.

### Feature Engineering

Transforms raw taxi trip data into 16 stable model-ready features:

**Temporal features:** `pickup_hour`, `pickup_weekday`, `pickup_month` derived from `lpep_pickup_datetime`  
**Duration:** `duration_min` calculated from pickup/dropoff timestamps  
**Original numerics:** `trip_distance`, `fare_amount`, `passenger_count`  
**Log transforms:** l`og_trip_distance`, `log_fare_amount`, `log_duration_min` for heavy-tailed distributions  
**Location features:** `PULocationID`, `DOLocationID` as raw IDs plus `PU_frequency`, `DO_frequency` using frequency encoding  
**Interaction features:** `distance_per_minute` as speed proxy, `fare_per_mile` as price efficiency

Applies consistent preprocessing: clips outliers, handles missing values, filters to credit card transactions with payment_type=1.

Feature schema is logged to MLflow as `feature_cols.json` for reproducibility.

Implementation: See [capstone_lib.py](capstone_lib.py) function `engineer_features()` and [capstone_flow.py](capstone_flow.py) `feature_engineering` step.

### Champion Bootstrap

On first run when no champion exists, the flow automatically:
1. Trains initial model on reference data using `GradientBoostingRegressor` with 200 estimators and `max_depth=6`
2. Logs model and metrics to MLflow in `bootstrap_train` run
3. Registers model version in Model Registry
4. Sets `@champion` alias to this version
5. Tags with `bootstrap=true` and `role=champion`

On subsequent runs, loads existing champion from `models:/green_taxi_tip_model@champion`.

Implementation: See [capstone_flow.py](capstone_flow.py) `load_champion` step.

### Evaluation Gate - Model Gate

The champion model is evaluated on the new batch after feature engineering:

1. Load champion from Model Registry via `models:/green_taxi_tip_model@champion`
2. Generate predictions on engineered batch features
3. Compute RMSE on batch for champion performance and on reference for baseline performance
4. Calculate performance degradation: `rmse_increase_pct = (batch_rmse - reference_rmse) / reference_rmse * 100`

**Retrain Decision:**
- If `rmse_increase_pct > 3%` **AND** integrity warnings present, set `retrain_recommended=true`
- If `rmse_increase_pct > 5%` with no integrity warnings, set `retrain_recommended=true`
- Otherwise, no retraining needed - flow ends

Logs champion evaluation metrics, dataset lineage via `mlflow.log_input()`, predictions artifact (`predictions.parquet`), and decision to MLflow.

Implementation: See [capstone_flow.py](capstone_flow.py) `model_gate` step.

### Retrain-Promotion Logic

**Retraining** - conditional step:
- Triggered only when `retrain_recommended=true` from model gate
- Trains new candidate model on expanded dataset with reference and batch combined
- Uses same architecture: `GradientBoostingRegressor` with median imputation
- Evaluates candidate on **both** batch for performance check and reference for stability check P3
- Logs candidate metrics, training dataset lineage, and predictions to MLflow

**Promotion Criteria** - all conditions must be met:
1. **P1 - Valid evaluation:** Candidate has evaluation metrics and dataset lineage logged - guaranteed by flow structure
2. **P2 - Performance improvement:** `candidate_rmse < champion_rmse * 0.99` for 1% minimum improvement threshold, configurable via `--min-improvement` parameter
3. **P3 - Stability check:** Candidate doesn't regress on reference by >5% to prevent overfitting to new batch
4. **P4 - Integrity sanity:** No hard integrity failures on the batch - guaranteed by flow structure

**Promotion Mechanics:**
- If all criteria met: Register candidate as new model version, update `@champion` alias to point to new version, tag old champion as `previous_champion`, log promotion decision
- If criteria not met: Still register candidate but tag as `validation_status=rejected` for audit trail, log rejection decision

Decision logged to `decision.json` with all P1-P4 criteria values and detailed reasoning.

Implementation: See [capstone_flow.py](capstone_flow.py) `retrain` and `promotion_gate` steps.

## Flow Execution

### Run 1 — Baseline

This is the initial run that bootstraps the champion model and compares January 2020 to itself by using the same data for both reference and batch. With identical data distributions, no drift or degradation is detected, and no retraining is triggered, and hence no promotion.

```bash
python capstone_flow.py run \
    --reference-path TLC_data/green_tripdata_2020-01.parquet \
    --batch-path TLC_data/green_tripdata_2020-01.parquet
```

In MLflow UI, verify:
1. `bootstrap_train` run creates the initial champion model and registers it in Model Registry
2. `model_gate` run shows champion evaluation metrics with `retrain_recommended=false` and `promotion_recommended=false`
3. **Inference demo:** `model_gate` run includes `predictions.parquet` artifact showing batch predictions on the new data
4. `decision.json` artifacts explain outcomes like `action=batch_accepted` and `action=no_retrain`
5. No `retrain` or `promotion_gate` runs since no action is needed

### Run 2 — Retrain & Promotion

This run uses the April 2020 batch, which exhibits COVID-19 related distribution shifts like fewer trips and different patterns. The model performance degrades beyond the acceptable threshold, triggering automatic retraining. A new candidate model is trained and compared to the champion. If better, it gets promoted to production via the `@champion` alias.

```bash
python capstone_flow.py run \
    --reference-path TLC_data/green_tripdata_2020-01.parquet \
    --batch-path TLC_data/green_tripdata_2020-04.parquet
```

In MLflow UI, verify:
- `model_gate` run shows champion evaluation metrics and `retrain_recommended=true`
- `retrain` run displays candidate vs champion metrics comparison such as `candidate_rmse` and `champion_rmse`
- **Inference demo:** Both `model_gate` and `retrain` runs include `predictions.parquet` artifacts with features, predictions, and actual values
- `promotion_gate` run includes decision tags and `decision.json` justifying the promotion
- Model Registry shows `green_taxi_tip_model` with a new model version registered and `@champion` alias updated

### Run 3 — Post Failure Resumption

This run demonstrates Metaflow's checkpointing and resume capability. By intentionally failing the flow mid-execution and then resuming, you'll see that completed steps are skipped and only the failed step and downstream steps are re-executed. This is critical for production pipelines with expensive computations.

1. Temporarily introduce an error in the `retrain` step by inserting the statement `raise RuntimeError("demo failure")` at the beginning of the retrain step function in the file [`capstone_flow.py`](capstone_flow.py).
2. Run the flow with a batch that causes retrain. It should fail at the `retrain` step.
3. Fix the error by removing the inserted line.
4. Resume:

   ```bash
   python capstone_flow.py resume retrain
   ```

In MLflow UI, verify:
- Flow resumes from the `retrain` step and does not restart from the beginning
- Previously completed steps like `integrity_gate`, `feature_engineering`, and `model_gate` are not re-executed
- MLflow shows both the failed run and the successful resumed run
- Final decisions and artifacts reflect the successful resumed execution

## Project Structure

| File | Purpose |
|---|---|
| [capstone_flow.py](capstone_flow.py) | Metaflow-based flow - the main pipeline |
| [capstone_lib.py](capstone_lib.py) | Shared utilities for data loading, feature engineering, hard and soft integrity checks, model building, champion/registry helpers and decision logging |
| [test_capstone_flow.py](test_capstone_flow.py) | Comprehensive test suite with 57 integration tests |
| [demo_walkthrough.txt](demo_walkthrough.txt) | Step-by-step video demo transcript |
| [environment.yml](environment.yml) | Conda environment specification |
| [design_doc.md](design_doc.md) | Full project specification |
| [reset.sh](reset.sh) | Cleanup script that kills MLflow server and removes tracking artifacts |

## MLflow UI — what to look for

- **Experiment:** `08_capstone_green_taxi`
- **Runs per flow execution:** `integrity_gate`, `feature_engineering`, `bootstrap_train` (first run only), `model_gate`, `retrain` (conditional), `promotion_gate` (conditional; only after retrain)
- **Key metrics:** `champion_rmse`, `baseline_rmse`, `rmse_increase_pct`, `candidate_rmse`
- **Key tags:** `pipeline_step`, `retrain_recommended`, `promotion_recommended`, `decision_action`, `integrity_warn`
- **Key artifacts:** `decision.json`, `hard_failures.json`, `nannyml_details.json`, `feature_cols.json`, `predictions.parquet`
- **Dataset lineage:** evaluation and training datasets logged via `mlflow.data`
- **Model Registry:** `green_taxi_tip_model` with `@champion` alias
