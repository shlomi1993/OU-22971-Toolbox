# MLOps Capstone Project — Green Taxi Tip Prediction

Manual monitoring → optional retraining → champion promotion pipeline using **Metaflow**, **NannyML**, and **MLflow**.

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

3. **Place data files** (parquet) under `../06_monitoring_data_drift/TLC_data/`:
   - `green_tripdata_2020-01.parquet` (reference)
   - `green_tripdata_2020-04.parquet` (batch A — triggers retrain)
   - `green_tripdata_2020-08.parquet` (batch B — no retrain needed)

   Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

4. **Start the MLflow tracking server:**

   ```bash
   mlflow server \
       --workers 1 \
       --port 5001 \
       --backend-store-uri sqlite:///mlflow_tracking/mlflow.db \
       --default-artifact-root mlflow_tracking/mlruns
   ```

5. **Open the MLflow UI** at http://localhost:5001. Look for experiment **`08_capstone_green_taxi`**.

---

## Running the flow

### Run 1 — Baseline (no retrain expected)

```bash
python capstone_flow.py run \
    --reference-path ../06_monitoring_data_drift/TLC_data/green_tripdata_2020-01.parquet \
    --batch-path ../06_monitoring_data_drift/TLC_data/green_tripdata_2020-08.parquet
```

In MLflow UI, verify:
- `integrity_gate` run → `decision.json` with `action=batch_accepted`
- `model_gate` run → `retrain_recommended=false`, `decision.json` with `action=no_retrain`
- No `retrain` / `promotion_gate` run for this execution path

### Run 2 — Retrain + promotion (performance degradation)

```bash
python capstone_flow.py run \
    --reference-path ../06_monitoring_data_drift/TLC_data/green_tripdata_2020-01.parquet \
    --batch-path ../06_monitoring_data_drift/TLC_data/green_tripdata_2020-04.parquet
```

In MLflow UI, verify:
- `model_gate` → `retrain_recommended=true`
- `retrain` run → `candidate_rmse`, `predictions.parquet` artifact
- `promotion_gate` → `promotion_recommended=true/false`, new model version in Model Registry (retrain path only)
- Check Model Registry → `green_taxi_tip_model` → `@champion` alias updated

### Run 3 — Failure + resume

1. Temporarily introduce an error in the `retrain` step (e.g. `raise RuntimeError("demo failure")`).
2. Run the flow — it will fail at `retrain`.
3. Fix the error.
4. Resume:

   ```bash
   python capstone_flow.py resume retrain
   ```

   Verify that previously completed steps are **not** re-executed.

---

## Project structure

| File | Purpose |
|---|---|
| `capstone_flow.py` | Metaflow flow — the main pipeline |
| `capstone_lib.py` | Shared utilities: data loading, feature engineering, integrity checks (hard + NannyML), model building, champion/registry helpers, decision logging |
| `flow_starter.py` | Original skeleton (reference only) |
| `demo_walkthrough.txt` | Step-by-step video demo transcript (3 required runs) |
| `environment.yml` | Conda environment specification |
| `design_doc.md` | Full project specification |

---

## MLflow UI — what to look for

- **Experiment:** `08_capstone_green_taxi`
- **Runs per flow execution:** `integrity_gate`, `feature_engineering`, `bootstrap_train` (first run only), `model_gate`, `retrain` (conditional), `promotion_gate` (conditional; only after retrain)
- **Key metrics:** `champion_rmse`, `baseline_rmse`, `rmse_increase_pct`, `candidate_rmse`
- **Key tags:** `pipeline_step`, `retrain_recommended`, `promotion_recommended`, `decision_action`, `integrity_warn`
- **Key artifacts:** `decision.json`, `hard_failures.json`, `nannyml_details.json`, `feature_cols.json`, `predictions.parquet`
- **Dataset lineage:** evaluation and training datasets logged via `mlflow.data`
- **Model Registry:** `green_taxi_tip_model` with `@champion` alias
