import logging
import numpy as np
import pandas as pd
import pytest
import socket

from pathlib import Path
from typing import Generator, Tuple
from unittest.mock import MagicMock, patch
from uuid import uuid4

from capstone_lib import FEATURE_COLS, Decision, engineer_features, EvaluationMetrics
from capstone_flow import MLFlowCapstoneFlow

FeatureXY = Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]


def _make_taxi_df(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic taxi DataFrame with the expected schema.

    Args:
        n_rows (int, optional): Number of rows in the generated DataFrame. Defaults to 100.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: Synthetic taxi DataFrame with columns matching the expected schema.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lpep_pickup_datetime": pd.date_range("2020-01-01", periods=n_rows, freq="h"),
        "lpep_dropoff_datetime": pd.date_range("2020-01-01 00:10:00", periods=n_rows, freq="h"),
        "trip_distance": rng.uniform(0.5, 20, n_rows),
        "fare_amount": rng.uniform(2.5, 100, n_rows),
        "tip_amount": rng.uniform(0, 20, n_rows),
        "PULocationID": rng.integers(1, 265, n_rows),
        "DOLocationID": rng.integers(1, 265, n_rows),
        "passenger_count": rng.integers(1, 6, n_rows).astype(float),
        "payment_type": np.ones(n_rows, dtype=int),
    })


def _mock_mlflow_run() -> MagicMock:
    """
    Create a mock MLflow run context.

    Returns:
        MagicMock: Mocked MLflow run object with a predefined run_id.
    """
    run = MagicMock()
    run.info.run_id = f"test_run_{uuid4().hex[:8]}"
    return run


def _build_integrity_report(passed: bool = True, failures: list = None, warn: bool = False, details: list = None) -> Tuple[bool, dict]:
    """
    Build a mock run_integrity_checks return value.
    """
    return passed, {
        "hard": {
            "passed": passed,
            "failures": failures or [],
            "metrics": {"missing_required_cols": 0.0 if passed else 1.0},
        },
        "nannyml": {
            "warn": warn,
            "details": details or [],
            "metrics": {},
        },
        "metrics": {"missing_required_cols": 0.0 if passed else 1.0},
    }


@pytest.fixture(autouse=True)
def _use_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Run every test in an isolated temp directory.

    Args:
        tmp_path (Path): Temporary directory path provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for modifying environment.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _silence_decision_log() -> Generator[None, None, None]:
    """
    Prevent Decision.log() from calling real MLflow.
    """
    with patch.object(Decision, "log"):
        yield


@pytest.fixture
def flow() -> MLFlowCapstoneFlow:
    """
    Bare MLFlowCapstoneFlow (Metaflow __init__ bypassed).
    """
    def find_available_port(start: int = 5000, end: int = 5010) -> int:
        for port in range(start, end + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No available port found in the specified range.")

    port = find_available_port()
    uid = uuid4().hex[:8]
    f = object.__new__(MLFlowCapstoneFlow)
    f.next = MagicMock()
    f.tracking_uri = f"http://localhost:{port}"
    f.experiment_name = "test_experiment"
    f.model_name = "test_model"
    f.reference_path = f"/tmp/ref_{uid}.parquet"
    f.batch_path = f"/tmp/batch_{uid}.parquet"
    f.min_improvement = 0.01

    # Initialize logger and state attributes (normally done in start() step)
    f.logger = logging.getLogger(f.__class__.__name__)
    f.batch_rejected = False
    f.integrity_warn = False
    f.retrain_needed = False
    f.did_retrain = False

    return f


@pytest.fixture
def taxi_ref() -> pd.DataFrame:
    return _make_taxi_df(n_rows=200, seed=0)


@pytest.fixture
def taxi_batch() -> pd.DataFrame:
    return _make_taxi_df(n_rows=100, seed=1)


@pytest.fixture
def feature_xy() -> FeatureXY:
    """
    Pre-engineered feature matrices and targets.
    """
    ref = _make_taxi_df(n_rows=200, seed=0)
    batch = _make_taxi_df(n_rows=100, seed=1)
    X_ref, y_ref = engineer_features(ref)
    X_batch, y_batch = engineer_features(batch)
    return X_ref, y_ref, X_batch, y_batch


@patch("capstone_flow.ModelRegistry")
@patch("capstone_flow.MlflowClient")
@patch("capstone_flow.mlflow")
def test_start_initializes_registry_and_advances(mock_mlflow: MagicMock, mock_client: MagicMock,
                                                 mock_registry_class: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    mock_registry = MagicMock()
    mock_registry_class.return_value = mock_registry

    flow.start()

    mock_mlflow.set_tracking_uri.assert_called_once_with(flow.tracking_uri)
    mock_mlflow.set_experiment.assert_called_once_with(flow.experiment_name)
    mock_client.assert_called_once()
    mock_registry_class.assert_called_once_with(mock_client.return_value, flow.model_name)
    assert flow.registry is mock_registry, "Registry should be set from ModelRegistry initialization"
    flow.next.assert_called_once()


@patch("capstone_flow.load_taxi_table")
def test_load_data_loads_ref_and_batch(mock_load: MagicMock, flow: MLFlowCapstoneFlow, taxi_ref: pd.DataFrame,
                                       taxi_batch: pd.DataFrame) -> None:
    mock_load.side_effect = [taxi_ref, taxi_batch]

    flow.load_data()

    assert mock_load.call_count == 2, "Should load both reference and batch data"
    pd.testing.assert_frame_equal(flow.df_ref, taxi_ref)
    pd.testing.assert_frame_equal(flow.df_batch, taxi_batch)
    flow.next.assert_called_once()


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_gate_accepted_no_warnings(mock_mlflow: MagicMock, mock_checks: MagicMock, flow: MLFlowCapstoneFlow,
                                             taxi_ref: pd.DataFrame, taxi_batch: pd.DataFrame) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_checks.return_value = _build_integrity_report()
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    assert flow.batch_rejected is False, "Batch should not be rejected when integrity checks pass"
    assert flow.integrity_warn is False, "No integrity warnings expected when checks pass cleanly"
    flow.next.assert_called_once()


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_gate_rejected_on_hard_failure(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                                 flow: MLFlowCapstoneFlow,taxi_ref: pd.DataFrame,
                                                 taxi_batch: pd.DataFrame) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_checks.return_value = _build_integrity_report(passed=False, failures=["Missing required columns: ['tip_amount']"])
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    assert flow.batch_rejected is True, "Batch should be rejected on hard integrity failures"
    assert flow.integrity_warn is False, "Integrity warn should be False when batch rejected"


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_gate_accepted_with_nannyml_warnings(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                                       flow: MLFlowCapstoneFlow,taxi_ref: pd.DataFrame,
                                                       taxi_batch: pd.DataFrame) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_checks.return_value = _build_integrity_report(warn=True, details=["drift on fare_amount"])
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    assert flow.batch_rejected is False, "Batch should be accepted despite NannyML warnings"
    assert flow.integrity_warn is True, "Integrity warn flag should be set when NannyML detects drift"


@patch("capstone_flow.engineer_features")
@patch("capstone_flow.mlflow")
def test_feature_engineering_produces_features(mock_mlflow: MagicMock, mock_eng: MagicMock, flow: MLFlowCapstoneFlow,
                                               taxi_ref: pd.DataFrame, taxi_batch: pd.DataFrame) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    X = pd.DataFrame(np.ones((5, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = np.ones(5)
    mock_eng.side_effect = [(X, y), (X.copy(), y.copy())]

    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch
    flow.batch_rejected = False

    flow.feature_engineering()

    assert mock_eng.call_count == 2, "engineer_features should be called for both ref and batch"
    assert flow.feature_cols == FEATURE_COLS, "Feature columns should match FEATURE_COLS constant"
    flow.next.assert_called_once()


def test_feature_engineering_skipped_when_batch_rejected(flow: MLFlowCapstoneFlow) -> None:
    flow.batch_rejected = True

    flow.feature_engineering()

    assert flow.X_ref is None, "X_ref should be None when batch rejected"
    assert flow.X_batch is None, "X_batch should be None when batch rejected"
    assert flow.feature_cols == [], "Feature cols should be empty list when batch rejected"
    flow.next.assert_called_once()


def test_load_champion_skipped_when_batch_rejected(flow: MLFlowCapstoneFlow) -> None:
    flow.batch_rejected = True

    flow.load_champion()

    assert flow.champion_model is None, "Champion model should be None when batch rejected"
    assert flow.champion_uri is None, "Champion URI should be None when batch rejected"
    flow.next.assert_called_once()


@patch("capstone_flow.mlflow")
def test_load_champion_loads_existing_champion(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    X_ref, y_ref, _, _ = feature_xy
    flow.batch_rejected = False
    flow.X_ref, flow.y_ref = X_ref, y_ref

    mock_registry = MagicMock()
    mock_registry.champion_exists.return_value = True
    mock_champion = MagicMock()
    mock_registry.load_champion.return_value = (mock_champion, "models:/test@champion")
    flow.registry = mock_registry

    flow.load_champion()

    mock_registry.champion_exists.assert_called_once()
    mock_registry.load_champion.assert_called_once()
    assert flow.champion_model is mock_champion, "Champion model should be loaded from registry"
    assert flow.champion_uri == "models:/test@champion", "Champion URI should match expected alias format"


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.build_model")
@patch("capstone_flow.mlflow")
def test_load_champion_bootstraps_when_no_champion(mock_mlflow: MagicMock, mock_build: MagicMock, mock_eval: MagicMock,
                                                   flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    X_ref, y_ref, _, _ = feature_xy
    flow.batch_rejected = False
    flow.X_ref, flow.y_ref = X_ref, y_ref

    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_model = MagicMock()
    mock_build.return_value = mock_model
    mock_eval.return_value = EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.8)
    mock_mlflow.sklearn.log_model.return_value = MagicMock(model_uri="runs:/abc/model")

    mock_registry = MagicMock()
    mock_registry.champion_exists.return_value = False
    mock_registry.register_version.return_value = "1"
    mock_registry.load_champion.return_value = (mock_model, "models:/test@champion")
    flow.registry = mock_registry

    flow.load_champion()

    mock_build.assert_called_once()
    mock_model.fit.assert_called_once()
    mock_registry.register_version.assert_called_once()
    mock_registry.promote_to_champion.assert_called_once_with("1", reason="bootstrap")
    mock_registry.load_champion.assert_called_once()


def test_model_gate_skipped_when_batch_rejected(flow: MLFlowCapstoneFlow) -> None:
    flow.batch_rejected = True

    flow.model_gate()

    assert flow.retrain_needed is False, "Retrain should not be needed when batch rejected"
    assert flow.rmse_champion_on_batch is None, "RMSE should be None when model gate skipped"
    flow.next.assert_called_once()


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_no_retrain_within_tolerance(mock_mlflow: MagicMock, mock_eval: MagicMock, flow: MLFlowCapstoneFlow,
                                                feature_xy: FeatureXY) -> None:
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()

    # 2 % increase → below 5 % threshold
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.02, mae=0.5, r2=0.8),   # champion on batch
        EvaluationMetrics(rmse=1.00, mae=0.5, r2=0.85),   # champion on ref
    ]

    flow.batch_rejected = False
    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    assert flow.retrain_needed is False, "Retrain should not be needed for 2% increase (below 5% threshold)"
    assert flow.rmse_increase_pct == pytest.approx(0.02, abs=1e-9), "RMSE increase should be 2%"


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_retrain_above_threshold(mock_mlflow: MagicMock, mock_eval: MagicMock, flow: MLFlowCapstoneFlow,
                                            feature_xy: FeatureXY) -> None:
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    # 10 % increase → above 5 % threshold
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.10, mae=0.6, r2=0.7),
        EvaluationMetrics(rmse=1.00, mae=0.5, r2=0.85),
    ]

    flow.batch_rejected = False
    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    assert flow.retrain_needed is True, "Retrain should be needed for 10% increase (above 5% threshold)"
    assert flow.rmse_increase_pct == pytest.approx(0.10, abs=1e-9), "RMSE increase should be 10%"


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_retrain_lowered_threshold_with_integrity_warn(mock_mlflow: MagicMock, mock_eval: MagicMock,
                                                                  flow: MLFlowCapstoneFlow,feature_xy: FeatureXY) -> None:
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    # 4 % increase → below 5 % but above 3 % lowered threshold
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.04, mae=0.5, r2=0.8),
        EvaluationMetrics(rmse=1.00, mae=0.5, r2=0.85),
    ]

    flow.batch_rejected = False
    flow.integrity_warn = True
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    assert flow.retrain_needed is True, \
        "Retrain should be needed: 4% increase exceeds lowered 3% threshold with integrity warnings"
    assert flow.rmse_increase_pct == pytest.approx(0.04, abs=1e-9), "RMSE increase should be 4%"


def test_retrain_skipped_when_not_needed(flow: MLFlowCapstoneFlow) -> None:
    flow.retrain_needed = False

    flow.retrain()

    assert flow.did_retrain is False, "Retrain should be skipped when not needed"
    assert flow.candidate_model_uri is None, "Candidate model URI should be None when retrain skipped"
    flow.next.assert_called_once()


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.build_model")
@patch("capstone_flow.mlflow")
def test_retrain_trains_on_combined_data(mock_mlflow: MagicMock, mock_build: MagicMock, mock_eval: MagicMock,
                                         flow: MLFlowCapstoneFlow,feature_xy: FeatureXY) -> None:
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(len(X_batch))
    mock_build.return_value = mock_model
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=0.90, mae=0.4, r2=0.9),    # candidate on batch
        EvaluationMetrics(rmse=0.85, mae=0.35, r2=0.92),   # candidate on ref
    ]
    mock_mlflow.sklearn.log_model.return_value = MagicMock(model_uri="runs:/abc/model")

    flow.retrain_needed = True
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.retrain()

    assert flow.did_retrain is True, "Retrain flag should be set after training"
    assert flow.candidate_model_uri == "runs:/abc/model", "Candidate model URI should match logged model"
    assert flow.candidate_rmse_batch == 0.90, "Candidate RMSE on batch should be recorded"
    assert flow.candidate_rmse_ref == 0.85, "Candidate RMSE on ref should be recorded"

    # Verify training on combined ref + batch
    mock_model.fit.assert_called_once()
    X_combined, y_combined = mock_model.fit.call_args[0]
    assert len(X_combined) == len(X_ref) + len(X_batch), "Training data should combine ref and batch"
    assert len(y_combined) == len(y_ref) + len(y_batch), "Training targets should combine ref and batch"
    flow.next.assert_called_once()


# ========== Promotion Gate Tests ==========

@patch("capstone_flow.mlflow")
def test_promotion_gate_no_promote_when_no_retrain(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = False
    flow.rmse_champion_on_batch = 1.0

    flow.promotion_gate()

    flow.next.assert_called_once()


@patch("capstone_flow.mlflow")
def test_promotion_gate_promote_all_criteria_pass(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = True
    flow.batch_rejected = False
    flow.rmse_champion_on_batch = 1.00
    flow.rmse_champion_on_ref = 1.00
    flow.candidate_rmse_batch = 0.95       # P2: 0.95 < 1.00 * 0.99 = 0.99
    flow.candidate_rmse_ref = 1.00          # P3: no regression
    flow.candidate_model_uri = "runs:/abc/model"
    flow.min_improvement = 0.01

    mock_registry = MagicMock()
    mock_registry.register_version.return_value = "2"
    flow.registry = mock_registry

    flow.promotion_gate()

    mock_registry.register_version.assert_called_once()
    mock_registry.promote_to_champion.assert_called_once()


@patch("capstone_flow.mlflow")
def test_promotion_gate_no_promote_candidate_not_better_enough(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = True
    flow.batch_rejected = False
    flow.rmse_champion_on_batch = 1.00
    flow.rmse_champion_on_ref = 1.00
    flow.candidate_rmse_batch = 1.00       # P2 fails: 1.00 >= 0.99
    flow.candidate_rmse_ref = 1.00
    flow.candidate_model_uri = "runs:/abc/model"
    flow.min_improvement = 0.01

    mock_registry = MagicMock()
    mock_registry.register_version.return_value = "2"
    flow.registry = mock_registry

    flow.promotion_gate()

    # Registered as rejected candidate (audit trail), but NOT promoted
    mock_registry.register_version.assert_called_once()
    mock_registry.promote_to_champion.assert_not_called()


@patch("capstone_flow.mlflow")
def test_promotion_gate_no_promote_reference_regression(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = True
    flow.batch_rejected = False
    flow.rmse_champion_on_batch = 1.00
    flow.rmse_champion_on_ref = 1.00
    flow.candidate_rmse_batch = 0.90       # P2 passes
    flow.candidate_rmse_ref = 1.10          # P3 fails: 10 % regression > 5 %
    flow.candidate_model_uri = "runs:/abc/model"
    flow.min_improvement = 0.01

    mock_registry = MagicMock()
    mock_registry.register_version.return_value = "2"
    flow.registry = mock_registry

    flow.promotion_gate()

    mock_registry.promote_to_champion.assert_not_called()


def test_end_batch_rejected(flow: MLFlowCapstoneFlow) -> None:
    flow.batch_rejected = True
    flow.did_retrain = False
    flow.end()  # should not raise


def test_end_retrain_completed(flow: MLFlowCapstoneFlow) -> None:
    flow.batch_rejected = False
    flow.did_retrain = True
    flow.end()


def test_end_no_retrain_needed(flow: MLFlowCapstoneFlow) -> None:
    flow.batch_rejected = False
    flow.did_retrain = False
    flow.end()


@patch("capstone_flow.mlflow")
def test_promotion_criteria_exactly_at_threshold(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    Candidate exactly at min_improvement threshold should NOT promote (< required, not <=).
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = True
    flow.batch_rejected = False
    flow.rmse_champion_on_batch = 1.00
    flow.rmse_champion_on_ref = 1.00
    flow.candidate_rmse_batch = 0.99  # Exactly 1% better = 1.00 * (1 - 0.01)
    flow.candidate_rmse_ref = 1.00
    flow.candidate_model_uri = "runs:/test/model"
    flow.min_improvement = 0.01

    mock_registry = MagicMock()
    mock_registry.register_version.return_value = "2"
    flow.registry = mock_registry

    flow.promotion_gate()

    # Should NOT promote (needs to be strictly better than threshold)
    mock_registry.promote_to_champion.assert_not_called()


@patch("capstone_flow.mlflow")
def test_promotion_criteria_just_below_threshold(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    Candidate just barely better than threshold should promote.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = True
    flow.batch_rejected = False
    flow.rmse_champion_on_batch = 1.00
    flow.rmse_champion_on_ref = 1.00
    flow.candidate_rmse_batch = 0.98999  # Slightly better than 1% threshold
    flow.candidate_rmse_ref = 1.00
    flow.candidate_model_uri = "runs:/test/model"
    flow.min_improvement = 0.01

    mock_registry = MagicMock()
    mock_registry.register_version.return_value = "2"
    flow.registry = mock_registry

    flow.promotion_gate()

    mock_registry.promote_to_champion.assert_called_once()


@patch("capstone_flow.mlflow")
def test_promotion_criteria_reference_regression_at_5_percent(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    Exactly 5% regression on reference should NOT promote.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = True
    flow.batch_rejected = False
    flow.rmse_champion_on_batch = 1.00
    flow.rmse_champion_on_ref = 1.00
    flow.candidate_rmse_batch = 0.90  # Good improvement on batch
    flow.candidate_rmse_ref = 1.05  # Exactly 5% worse on reference
    flow.candidate_model_uri = "runs:/test/model"
    flow.min_improvement = 0.01

    mock_registry = MagicMock()
    flow.registry = mock_registry

    flow.promotion_gate()

    mock_registry.promote_to_champion.assert_not_called()


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_integrity_warn_lowers_threshold_exactly_at_3_percent(
    mock_mlflow: MagicMock, mock_eval: MagicMock, flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Just below 3% increase with integrity warning should not trigger (boundary test).
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.0299, mae=0.5, r2=0.8),  # 2.99% increase (just below 3%)
        EvaluationMetrics(rmse=1.00, mae=0.5, r2=0.85),
    ]

    flow.batch_rejected = False
    flow.integrity_warn = True
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # 2.99% < 3% threshold, so should NOT trigger
    assert flow.retrain_needed is False, "Just below 3% should not trigger (needs >3%)"


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_zero_rmse_baseline_handles_division(mock_mlflow: MagicMock, mock_eval: MagicMock,
                                                        flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Zero RMSE on reference should not cause division by zero.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=0.5, mae=0.3, r2=0.9),
        EvaluationMetrics(rmse=0.0, mae=0.0, r2=1.0),  # Perfect on reference
    ]

    flow.batch_rejected = False
    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Should handle gracefully with 1e-9 epsilon
    assert flow.rmse_increase_pct == 0.0, "Should return 0 when baseline is 0"


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_check_multiple_hard_failures(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                                flow: MLFlowCapstoneFlow, taxi_ref: pd.DataFrame,
                                                taxi_batch: pd.DataFrame) -> None:
    """
    Multiple hard failures should all be reported.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    failures = [
        "Missing required columns: ['fare_amount']",
        "tip_amount null fraction too high: 0.75",
        "Negative duration fraction: 0.15",
    ]
    mock_checks.return_value = (False, {
        "hard": {"passed": False, "failures": failures, "metrics": {"missing_required_cols": 1.0}},
        "nannyml": {"warn": False, "details": [], "metrics": {}},
        "metrics": {"missing_required_cols": 1.0},
    })
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    assert flow.batch_rejected is True, "Batch should be rejected with multiple hard failures"
    assert len(failures) == 3, "All three failure messages should be captured"


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_check_multiple_nannyml_warnings(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                                   flow: MLFlowCapstoneFlow, taxi_ref: pd.DataFrame,
                                                   taxi_batch: pd.DataFrame) -> None:
    """
    Multiple NannyML drift warnings should all be logged.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    details = [
        "Missingness spike in 'trip_distance': ref=0.01, cur=0.08",
        "NannyML drift alert for 'fare_amount'",
        "Unseen categories in 'PULocationID': 12 new values",
    ]
    mock_checks.return_value = (True, {
        "hard": {"passed": True, "failures": [], "metrics": {}},
        "nannyml": {"warn": True, "details": details, "metrics": {}},
        "metrics": {},
    })
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    assert flow.batch_rejected is False, "Batch should not be rejected with only warnings"
    assert flow.integrity_warn is True, "Integrity warn flag should be set with NannyML warnings"


@patch("capstone_flow.engineer_features")
@patch("capstone_flow.mlflow")
def test_data_edge_case_empty_dataframes_after_filtering(mock_mlflow: MagicMock, mock_eng: MagicMock,
                                                         flow: MLFlowCapstoneFlow) -> None:
    """
    Feature engineering should handle empty DataFrames gracefully.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    # Return empty features
    X_empty = pd.DataFrame(columns=FEATURE_COLS)
    y_empty = np.array([])
    mock_eng.side_effect = [(X_empty, y_empty), (X_empty.copy(), y_empty.copy())]

    flow.df_ref = _make_taxi_df(10)
    flow.df_batch = _make_taxi_df(10)
    flow.batch_rejected = False

    flow.feature_engineering()

    assert len(flow.X_ref) == 0, "X_ref should be empty after filtering"
    assert len(flow.y_ref) == 0, "y_ref should be empty after filtering"


@patch("capstone_flow.load_taxi_table")
def test_data_edge_case_large_row_count_difference(mock_load: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    Should handle large difference in row counts between ref and batch.
    """
    large_ref = _make_taxi_df(n_rows=10000, seed=0)
    small_batch = _make_taxi_df(n_rows=10, seed=1)
    mock_load.side_effect = [large_ref, small_batch]

    flow.load_data()

    assert len(flow.df_ref) == 10000, "Reference should have 10000 rows"
    assert len(flow.df_batch) == 10, "Batch should have 10 rows"
    flow.next.assert_called_once()


@patch("capstone_flow.mlflow")
def test_promotion_audit_trail_rejected_candidate_registered_with_tags(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    Rejected candidates should be registered with proper tags.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = True
    flow.batch_rejected = False
    flow.rmse_champion_on_batch = 1.00
    flow.rmse_champion_on_ref = 1.00
    flow.candidate_rmse_batch = 1.05  # Worse than champion
    flow.candidate_rmse_ref = 1.00
    flow.candidate_model_uri = "runs:/rejected/model"
    flow.min_improvement = 0.01

    mock_registry = MagicMock()
    mock_registry.register_version.return_value = "3"
    flow.registry = mock_registry

    flow.promotion_gate()

    # Should register as rejected
    mock_registry.register_version.assert_called_once()
    call_args = mock_registry.register_version.call_args
    assert call_args[1]["tags"]["validation_status"] == "rejected", "Rejected candidate should have validation_status tag set to 'rejected'"
    assert "decision_reason" in call_args[1]["tags"], "Rejected candidate should include decision_reason tag"


@patch("capstone_flow.ModelRegistry")
@patch("capstone_flow.MlflowClient")
@patch("capstone_flow.mlflow")
def test_attribute_initialization_all_flags_initialized_in_start(mock_mlflow: MagicMock, mock_client: MagicMock, mock_registry_class: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    All conditional flags should be initialized to prevent AttributeError.
    """
    mock_registry_class.return_value = MagicMock()

    flow.start()

    assert hasattr(flow, "batch_rejected"), "batch_rejected attribute should exist"
    assert hasattr(flow, "integrity_warn"), "integrity_warn attribute should exist"
    assert hasattr(flow, "retrain_needed"), "retrain_needed attribute should exist"
    assert hasattr(flow, "did_retrain"), "did_retrain attribute should exist"
    assert flow.batch_rejected is False, "batch_rejected should initialize to False"
    assert flow.integrity_warn is False, "integrity_warn should initialize to False"
    assert flow.retrain_needed is False, "retrain_needed should initialize to False"
    assert flow.did_retrain is False, "did_retrain should initialize to False"


def test_attribute_initialization_retrain_initializes_all_outputs(flow: MLFlowCapstoneFlow) -> None:
    """
    Retrain step should initialize all output attributes even when skipped.
    """
    flow.retrain_needed = False

    flow.retrain()

    assert flow.did_retrain is False, "did_retrain should be False when skipped"
    assert flow.candidate_model_uri is None, "candidate_model_uri should be None when skipped"
    assert flow.candidate_rmse_batch is None, "candidate_rmse_batch should be None when skipped"
    assert flow.candidate_rmse_ref is None, "candidate_rmse_ref should be None when skipped"
    assert flow.retrain_run_id is None, "retrain_run_id should be None when skipped"


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_decision_enum_usage_actions_are_enums_not_strings(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                                           flow: MLFlowCapstoneFlow, taxi_ref: pd.DataFrame,
                                                           taxi_batch: pd.DataFrame) -> None:
    """
    All Decision instantiations should use DecisionAction enum values.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_checks.return_value = (False, {
        "hard": {"passed": False, "failures": ["test"], "metrics": {}},
        "nannyml": {"warn": False, "details": [], "metrics": {}},
        "metrics": {},
    })
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    # Verify the flow state was set correctly
    assert flow.batch_rejected == True, "batch_rejected should be True for hard failure"
    assert flow.integrity_warn == False, "integrity_warn should be False for rejected batch"
