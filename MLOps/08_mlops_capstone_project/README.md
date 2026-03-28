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
3. `decision.json` artifacts explain outcomes like `action=batch_accepted` and `action=no_retrain`
4. No `retrain` or `promotion_gate` runs since no action is needed

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
