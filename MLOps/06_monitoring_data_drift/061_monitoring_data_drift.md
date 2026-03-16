# 06 — Model evaluation, monitoring, and temporal data drift (case study)

## Setup

1. Activate the conda env:

   ```bash
   conda activate 22971-mlflow
   ```
  
2. Start the tracking server:

   ```powershell
   mlflow server --workers 1 --port 5000 --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --default-artifact-root mlflow_tracking/mlruns
   ```
3. Open the UI at http://localhost:5001.

---
## Prerequisites

1. Review the **NYC Green Taxi Dataset EDA** notebook.
   
1. Get a high-level grasp of **concept drift**: skim https://arxiv.org/abs/2004.05785. 

---

## Drift

A common ML failure mode: 
1. Collect data over time.
2. Train a model on all data **up to time T**.
3. Continue collecting data
4. Use the model for **inference on new data**.
   - The new data differs from the old data used to train the model (due to seasonal \ geographical changes, customer tastes shift, etc.). 
   - The model **performance is significantly worse** than on the test set.

This is the crux of concept \ data \ distribution drift. 

---

## Monitoring with MLflow

To deal with this problem efficiently we will:
1. Document the data collection process.
2. Check new data slices for integrity and shift.
3. Track our model's performance on new datasets.
4. Retrain the model on fresh data when performance degradation is critical.  

---

## Case study: NYC green taxi 

**The workflow**:
1. `06_train_initial.py`: fit an initial model on historical green taxi data (early 2020).

2. `06_check_drift.py`:
   1. Simulate online data collection using later months.
   2. Intentionally introduce data issues (schema breaks, missing values, mislabling etc.)
   3. Run:
        - Data integrity checks
        - Data drift checks
   4. Evaluate the model on newer data to detect degradation. 
3. `06_retrain.py`: Retrain the model on new data when degradation is unacceptable.

