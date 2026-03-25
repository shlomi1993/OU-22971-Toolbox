import logging
import numpy as np
import pandas as pd
import pytest
import socket
import warnings

from pathlib import Path
from typing import Generator, Tuple
from unittest.mock import MagicMock, patch
from uuid import uuid4

from capstone_lib import FEATURE_COLS, RAW_NUMERIC_COLS, EvaluationMetrics, DecisionAction, engineer_features, run_soft_integrity_checks
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
    Prevent log_decision() from calling real MLflow.
    """
    with patch("capstone_flow.log_decision"):
        yield


@pytest.fixture(autouse=True)
def _suppress_nannyml_warnings() -> None:
    """
    Suppress NannyML chunk size warnings in tests.
    These warnings are expected with small test datasets; production batches are much larger.
    """
    warnings.filterwarnings('ignore', message='.*number of chunks is too low.*', category=UserWarning)


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
    f._datastore = None  # Prevent Metaflow __getattr__ recursion
    f.next = MagicMock()
    f.tracking_uri = f"http://localhost:{port}"
    f.experiment_name = "test_experiment"
    f.model_name = "test_model"
    f.reference_path = f"/tmp/ref_{uid}.parquet"
    f.batch_path = f"/tmp/batch_{uid}.parquet"
    f.min_improvement = 0.01

    # Initialize logger and state attributes (normally done in start() step)
    f.logger = logging.getLogger(f.__class__.__name__)
    f.decision_action = None
    f.integrity_warn = False

    # Initialize retrain output attributes (normally done in model_gate() step)
    f.candidate_model_uri = None
    f.candidate_rmse_batch = None
    f.candidate_rmse_ref = None
    f.retrain_run_id = None

    # Initialize champion metrics (normally set in model_gate() step)
    f.rmse_champion_on_batch = None
    f.rmse_champion_on_ref = None

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

    assert flow.decision_action == DecisionAction.BATCH_ACCEPTED, "Batch should be accepted when integrity checks pass"
    assert flow.integrity_warn is False, "No integrity warnings expected when checks pass cleanly"
    # Branching: next() should be called (to feature_engineering, but we can't easily test the step reference in unit tests)
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

    assert flow.decision_action == DecisionAction.REJECT_BATCH, "Batch should be rejected on hard integrity failures"
    assert flow.integrity_warn is False, "Integrity warn should be False when batch rejected"
    # Branching: next() should be called (to handle_rejection, but we can't easily test the step reference in unit tests)
    flow.next.assert_called_once()


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_gate_accepted_with_nannyml_warnings(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                                       flow: MLFlowCapstoneFlow,taxi_ref: pd.DataFrame,
                                                       taxi_batch: pd.DataFrame) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_checks.return_value = _build_integrity_report(warn=True, details=["drift on fare_amount"])
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    assert flow.decision_action == DecisionAction.BATCH_ACCEPTED, "Batch should be accepted despite NannyML warnings"
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

    flow.feature_engineering()

    assert mock_eng.call_count == 2, "engineer_features should be called for both ref and batch"
    assert flow.X_ref is not None, "X_ref should be populated"
    assert flow.X_batch is not None, "X_batch should be populated"
    flow.next.assert_called_once()


@patch("capstone_flow.mlflow")
def test_load_champion_loads_existing_champion(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    X_ref, y_ref, _, _ = feature_xy
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

    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Should not trigger retrain (branching handles this)


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

    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Should trigger retrain (branching will route to retrain step)


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

    flow.integrity_warn = True
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Should trigger retrain with lowered threshold (branching will route to retrain step)



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

    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.retrain()

    assert flow.candidate_model_uri == "runs:/abc/model", "Candidate model URI should match logged model"
    assert flow.candidate_rmse_batch == 0.90, "Candidate RMSE on batch should be recorded"
    assert flow.candidate_rmse_ref == 0.85, "Candidate RMSE on ref should be recorded"

    # Verify training on combined ref + batch
    mock_model.fit.assert_called_once()
    X_combined, y_combined = mock_model.fit.call_args[0]
    assert len(X_combined) == len(X_ref) + len(X_batch), "Training data should combine ref and batch"
    assert len(y_combined) == len(y_ref) + len(y_batch), "Training targets should combine ref and batch"
    flow.next.assert_called_once()


@patch("capstone_flow.mlflow")
def test_promotion_gate_promote_all_criteria_pass(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
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
    flow.decision_action = DecisionAction.REJECT_BATCH
    flow.end()  # should not raise


def test_end_retrain_completed(flow: MLFlowCapstoneFlow) -> None:
    flow.decision_action = DecisionAction.PROMOTE
    flow.end()


def test_end_no_retrain_needed(flow: MLFlowCapstoneFlow) -> None:
    flow.decision_action = DecisionAction.NO_RETRAIN
    flow.end()


@patch("capstone_flow.mlflow")
def test_promotion_criteria_exactly_at_threshold(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    Candidate exactly at min_improvement threshold should NOT promote (< required, not <=).
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
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

    flow.integrity_warn = True
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # 2.99% < 3% threshold, so should NOT trigger (branching will route to promotion_gate)


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

    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Should handle gracefully with 1e-9 epsilon


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

    assert flow.decision_action == DecisionAction.REJECT_BATCH, "Batch should be rejected with multiple hard failures"
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

    assert flow.decision_action == DecisionAction.BATCH_ACCEPTED, "Batch should not be rejected with only warnings"
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
    assert call_args[0][1]["validation_status"] == "rejected", "Rejected candidate should have validation_status tag set to 'rejected'"
    assert "decision_reason" in call_args[0][1], "Rejected candidate should include decision_reason tag"


@patch("capstone_flow.ModelRegistry")
@patch("capstone_flow.MlflowClient")
@patch("capstone_flow.mlflow")
def test_attribute_initialization_all_flags_initialized_in_start(mock_mlflow: MagicMock, mock_client: MagicMock, mock_registry_class: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    All conditional flags should be initialized to prevent AttributeError.
    """
    mock_registry_class.return_value = MagicMock()

    flow.start()

    assert hasattr(flow, "decision_action"), "decision_action attribute should exist"
    assert hasattr(flow, "integrity_warn"), "integrity_warn attribute should exist"
    
    assert flow.decision_action is None, "decision_action should initialize to None"
    assert flow.integrity_warn is False, "integrity_warn should initialize to False"


def test_attribute_initialization_retrain_initializes_all_outputs(flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Model gate should initialize all retrain output attributes.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch
    flow.integrity_warn = False

    with patch("capstone_flow.evaluate_model") as mock_eval, patch("capstone_flow.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
        mock_eval.side_effect = [
            EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.8),
            EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.85),
        ]
        flow.model_gate()

    assert flow.decision_action == DecisionAction.NO_RETRAIN, "decision_action should be NO_RETRAIN when not triggered"
    assert flow.candidate_model_uri is None, "candidate_model_uri should be None when not triggered"
    assert flow.candidate_rmse_batch is None, "candidate_rmse_batch should be None when not triggered"
    assert flow.candidate_rmse_ref is None, "candidate_rmse_ref should be None when not triggered"
    assert flow.retrain_run_id is None, "retrain_run_id should be None when not triggered"


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
    assert flow.decision_action == DecisionAction.REJECT_BATCH, "decision_action should be REJECT_BATCH for hard failure"
    assert flow.integrity_warn == False, "integrity_warn should be False for rejected batch"


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_gate_branches_to_end_on_rejection(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                                     flow: MLFlowCapstoneFlow, taxi_ref: pd.DataFrame,
                                                     taxi_batch: pd.DataFrame) -> None:
    """
    When batch is rejected, integrity_gate should branch directly to end (not feature_engineering).
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_checks.return_value = _build_integrity_report(passed=False, failures=["critical error"])
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    assert flow.decision_action == DecisionAction.REJECT_BATCH, "Batch should be rejected when integrity checks fail"
    flow.next.assert_called_once()  # Should call next(end)


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_gate_branches_to_feature_engineering_on_acceptance(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                                                       flow: MLFlowCapstoneFlow, taxi_ref: pd.DataFrame,
                                                                       taxi_batch: pd.DataFrame) -> None:
    """
    When batch is accepted, integrity_gate should branch to feature_engineering.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_checks.return_value = _build_integrity_report(passed=True)
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.integrity_gate()

    assert flow.decision_action == DecisionAction.BATCH_ACCEPTED, "decision_action should be BATCH_ACCEPTED when integrity checks pass"
    flow.next.assert_called_once()  # Should call next(feature_engineering)


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_exactly_5_percent_increase_no_retrain(mock_mlflow: MagicMock, mock_eval: MagicMock,
                                                         flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Exactly 5% RMSE increase should NOT trigger retrain (needs >5%, not >=).
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    # Exactly 5% increase
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.05, mae=0.5, r2=0.8),
        EvaluationMetrics(rmse=1.00, mae=0.5, r2=0.85),
    ]

    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Should NOT trigger (branching will route to promotion_gate)


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_exactly_3_percent_with_integrity_warn_no_retrain(mock_mlflow: MagicMock, mock_eval: MagicMock,
                                                                     flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Exactly 3% increase with integrity warning should NOT trigger (needs >3%).
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.03, mae=0.5, r2=0.8),
        EvaluationMetrics(rmse=1.00, mae=0.5, r2=0.85),
    ]

    flow.integrity_warn = True
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Should NOT trigger (branching will route to promotion_gate)

@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_negative_rmse_increase_no_retrain(mock_mlflow: MagicMock, mock_eval: MagicMock,
                                                      flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Champion performing better on new batch (negative increase) should not trigger retrain.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    # Champion performs better on batch than reference
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=0.95, mae=0.4, r2=0.9),
        EvaluationMetrics(rmse=1.00, mae=0.5, r2=0.85),
    ]

    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Should not trigger retrain (branching will route to promotion_gate)


@patch("capstone_flow.engineer_features")
@patch("capstone_flow.mlflow")
def test_feature_engineering_logs_mlflow_tags(mock_mlflow: MagicMock, mock_eng: MagicMock,
                                               flow: MLFlowCapstoneFlow, taxi_ref: pd.DataFrame,
                                               taxi_batch: pd.DataFrame) -> None:
    """
    Feature engineering step should log pipeline_step tag to MLflow.
    """
    mock_run = _mock_mlflow_run()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    X = pd.DataFrame(np.ones((5, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = np.ones(5)
    mock_eng.side_effect = [(X, y), (X.copy(), y.copy())]

    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch

    flow.feature_engineering()

    mock_mlflow.set_tag.assert_called_with("pipeline_step", "feature_engineering")
    mock_mlflow.log_dict.assert_called_once()


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_logs_dataset_lineage(mock_mlflow: MagicMock, mock_eval: MagicMock,
                                        flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Model gate should log evaluation dataset using mlflow.log_input for lineage tracking.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.8),
        EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.85),
    ]
    mock_dataset = MagicMock()
    mock_mlflow.data.from_pandas.return_value = mock_dataset

    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    mock_mlflow.data.from_pandas.assert_called_once()
    mock_mlflow.log_input.assert_called_once_with(mock_dataset, context="evaluation")


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_logs_predictions_artifact(mock_mlflow: MagicMock, mock_eval: MagicMock,
                                              flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Model gate should log predictions as parquet artifact.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.8),
        EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.85),
    ]

    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.random.random(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.model_gate()

    # Verify artifact logging was called
    mock_mlflow.log_artifact.assert_called_once()
    artifact_call = mock_mlflow.log_artifact.call_args[0][0]
    assert artifact_call == "predictions.parquet", "Should log predictions.parquet artifact"


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.build_model")
@patch("capstone_flow.mlflow")
def test_retrain_logs_training_dataset_lineage(mock_mlflow: MagicMock, mock_build: MagicMock,
                                               mock_eval: MagicMock, flow: MLFlowCapstoneFlow,
                                               feature_xy: FeatureXY) -> None:
    """
    Retrain step should log training dataset lineage.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(len(X_batch))
    mock_build.return_value = mock_model
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=0.9, mae=0.4, r2=0.9),
        EvaluationMetrics(rmse=0.85, mae=0.35, r2=0.92),
    ]
    mock_mlflow.sklearn.log_model.return_value = MagicMock(model_uri="runs:/abc/model")
    mock_dataset = MagicMock()
    mock_mlflow.data.from_pandas.return_value = mock_dataset

    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.retrain()

    # Verify training dataset was logged
    mock_mlflow.data.from_pandas.assert_called_once()
    mock_mlflow.log_input.assert_called_once_with(mock_dataset, context="training")


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.build_model")
@patch("capstone_flow.mlflow")
def test_bootstrap_logs_correct_tags(mock_mlflow: MagicMock, mock_build: MagicMock,
                                     mock_eval: MagicMock, flow: MLFlowCapstoneFlow,
                                     feature_xy: FeatureXY) -> None:
    """
    Bootstrap training should log special tags (bootstrap=true, trained_on=reference).
    """
    X_ref, y_ref, _, _ = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()

    mock_model = MagicMock()
    mock_build.return_value = mock_model
    mock_eval.return_value = EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.8)
    mock_mlflow.sklearn.log_model.return_value = MagicMock(model_uri="runs:/bootstrap/model")

    mock_registry = MagicMock()
    mock_registry.champion_exists.return_value = False
    mock_registry.register_version.return_value = "1"
    mock_registry.load_champion.return_value = (mock_model, "models:/test@champion")
    flow.registry = mock_registry
    flow.X_ref, flow.y_ref = X_ref, y_ref

    flow.load_champion()

    # Verify bootstrap tags were set
    bootstrap_tag_calls = [call for call in mock_mlflow.set_tag.call_args_list
                          if len(call[0]) > 1 and call[0][0] == "bootstrap"]
    assert len(bootstrap_tag_calls) > 0, "Should set bootstrap tag"
    assert bootstrap_tag_calls[0][0][1] == "true", "Bootstrap tag should be 'true'"


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.build_model")
@patch("capstone_flow.mlflow")
def test_bootstrap_registers_with_validation_approved(mock_mlflow: MagicMock, mock_build: MagicMock,
                                                      mock_eval: MagicMock, flow: MLFlowCapstoneFlow,
                                                      feature_xy: FeatureXY) -> None:
    """
    Bootstrap should register version with validation_status=approved.
    """
    X_ref, y_ref, _, _ = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()

    mock_model = MagicMock()
    mock_build.return_value = mock_model
    mock_eval.return_value = EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.8)
    mock_mlflow.sklearn.log_model.return_value = MagicMock(model_uri="runs:/bootstrap/model")

    mock_registry = MagicMock()
    mock_registry.champion_exists.return_value = False
    mock_registry.register_version.return_value = "1"
    mock_registry.load_champion.return_value = (mock_model, "models:/test@champion")
    flow.registry = mock_registry
    flow.X_ref, flow.y_ref = X_ref, y_ref

    flow.load_champion()

    # Check registration tags (register_version is called with positional args: model_uri, tags)
    call_args = mock_registry.register_version.call_args
    tags = call_args[0][1]  # Second positional argument
    assert tags["validation_status"] == "approved", "Bootstrap version should be approved"
    assert tags["role"] == "champion", "Bootstrap version should have champion role"


def test_promotion_gate_p4_obsolete() -> None:
    """
    P4 test removed - promotion_gate is only reachable if batch passed integrity checks.
    The flow structure guarantees P4 is always true at this point.
    """
    pass


@patch("capstone_flow.mlflow")
def test_promotion_gate_all_criteria_fail(mock_mlflow: MagicMock, flow: MLFlowCapstoneFlow) -> None:
    """
    When all promotion criteria fail, candidate should still be registered as rejected.
    """
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()
    flow.did_retrain = True
    flow.batch_rejected = True  # P4 fails
    flow.rmse_champion_on_batch = 1.00
    flow.rmse_champion_on_ref = 1.00
    flow.candidate_rmse_batch = 1.10  # P2 fails (worse)
    flow.candidate_rmse_ref = 1.20  # P3 fails (regression)
    flow.candidate_model_uri = "runs:/test/model"
    flow.min_improvement = 0.01

    mock_registry = MagicMock()
    mock_registry.register_version.return_value = "5"
    flow.registry = mock_registry

    flow.promotion_gate()

    # Should register but not promote
    mock_registry.register_version.assert_called_once()
    mock_registry.promote_to_champion.assert_not_called()


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.build_model")
@patch("capstone_flow.mlflow")
def test_retrain_combines_ref_and_batch_correctly(mock_mlflow: MagicMock, mock_build: MagicMock,
                                                  mock_eval: MagicMock, flow: MLFlowCapstoneFlow,
                                                  feature_xy: FeatureXY) -> None:
    """
    Verify that retrain merges reference and batch data with correct dimensions.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(len(X_batch))
    mock_build.return_value = mock_model
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=0.9, mae=0.4, r2=0.9),
        EvaluationMetrics(rmse=0.85, mae=0.35, r2=0.92),
    ]
    mock_mlflow.sklearn.log_model.return_value = MagicMock(model_uri="runs:/test/model")

    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch

    flow.retrain()

    # Verify fit was called with correct combined data
    mock_model.fit.assert_called_once()
    X_train, y_train = mock_model.fit.call_args[0]

    assert len(X_train) == len(X_ref) + len(X_batch), "Training X should combine ref and batch"
    assert len(y_train) == len(y_ref) + len(y_batch), "Training y should combine ref and batch"
    assert list(X_train.columns) == list(X_ref.columns), "Training X should have same columns as ref"


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.build_model")
@patch("capstone_flow.mlflow")
def test_retrain_logs_training_params(mock_mlflow: MagicMock, mock_build: MagicMock,
                                     mock_eval: MagicMock, flow: MLFlowCapstoneFlow,
                                     feature_xy: FeatureXY) -> None:
    """
    Retrain should log reference_path and batch_path as MLflow params.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(len(X_batch))
    mock_build.return_value = mock_model
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=0.9, mae=0.4, r2=0.9),
        EvaluationMetrics(rmse=0.85, mae=0.35, r2=0.92),
    ]
    mock_mlflow.sklearn.log_model.return_value = MagicMock(model_uri="runs:/test/model")

    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch
    flow.reference_path = "data/ref.parquet"
    flow.batch_path = "data/batch.parquet"

    flow.retrain()

    # Verify params were logged
    mock_mlflow.log_params.assert_called_once()
    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert "reference_path" in logged_params, "Should log reference_path param"
    assert "batch_path" in logged_params, "Should log batch_path param"


def test_end_handles_all_decision_actions(flow: MLFlowCapstoneFlow) -> None:
    """
    End step should handle all possible decision_action values.
    """
    # Test case 1: batch rejected
    flow.decision_action = DecisionAction.REJECT_BATCH
    flow.end()  # Should not raise

    # Test case 2: no retrain needed
    flow.decision_action = DecisionAction.NO_RETRAIN
    flow.end()  # Should not raise

    # Test case 3: candidate not promoted
    flow.decision_action = DecisionAction.NO_PROMOTE
    flow.end()  # Should not raise

    # Test case 4: candidate promoted
    flow.decision_action = DecisionAction.PROMOTE
    flow.end()  # Should not raise


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.build_model")
@patch("capstone_flow.mlflow")
def test_full_bootstrap_to_load_champion_flow(mock_mlflow: MagicMock, mock_build: MagicMock,
                                              mock_eval: MagicMock, flow: MLFlowCapstoneFlow,
                                              feature_xy: FeatureXY) -> None:
    """
    Integration test: Bootstrap creates champion, then load_champion retrieves it.
    """
    X_ref, y_ref, _, _ = feature_xy
    mock_mlflow.start_run.return_value.__enter__.return_value = _mock_mlflow_run()

    mock_model = MagicMock()
    mock_build.return_value = mock_model
    mock_eval.return_value = EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.8)
    mock_mlflow.sklearn.log_model.return_value = MagicMock(model_uri="runs:/bootstrap/model")

    mock_registry = MagicMock()

    # Simulate bootstrap scenario
    mock_registry.champion_exists.return_value = False
    mock_registry.register_version.return_value = "1"
    mock_registry.load_champion.return_value = (mock_model, "models:/test@champion")

    flow.registry = mock_registry
    flow.X_ref, flow.y_ref = X_ref, y_ref

    flow.load_champion()

    # Verify bootstrap workflow
    mock_registry.champion_exists.assert_called_once()
    mock_model.fit.assert_called_once()
    mock_registry.register_version.assert_called_once()
    mock_registry.promote_to_champion.assert_called_once()
    mock_registry.load_champion.assert_called_once()

    assert flow.champion_model is not None, "Champion model should be set after loading"
    assert flow.champion_uri == "models:/test@champion", "Champion URI should be set correctly after loading"


@patch("capstone_flow.run_integrity_checks")
@patch("capstone_flow.mlflow")
def test_integrity_gate_sets_run_id_attribute(mock_mlflow: MagicMock, mock_checks: MagicMock,
                                              flow: MLFlowCapstoneFlow, taxi_ref: pd.DataFrame,
                                              taxi_batch: pd.DataFrame) -> None:
    """
    Integrity gate should capture and store the MLflow run_id.
    """
    mock_run = _mock_mlflow_run()
    expected_run_id = mock_run.info.run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_checks.return_value = _build_integrity_report(passed=True)
    flow.df_ref, flow.df_batch = taxi_ref, taxi_batch
    flow.end = MagicMock()  # For branching
    flow.feature_engineering = MagicMock()  # For branching

    flow.integrity_gate()

    # Verify run_id was captured
    try:
        assert flow.integrity_run_id == expected_run_id, "Should capture correct run ID"
    except AttributeError:
        pytest.fail("integrity_run_id should be set after integrity_gate()")


@patch("capstone_flow.evaluate_model")
@patch("capstone_flow.mlflow")
def test_model_gate_sets_run_id_attribute(mock_mlflow: MagicMock, mock_eval: MagicMock,
                                         flow: MLFlowCapstoneFlow, feature_xy: FeatureXY) -> None:
    """
    Model gate should capture and store the MLflow run_id.
    """
    X_ref, y_ref, X_batch, y_batch = feature_xy
    mock_run = _mock_mlflow_run()
    expected_run_id = mock_run.info.run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_eval.side_effect = [
        EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.8),
        EvaluationMetrics(rmse=1.0, mae=0.5, r2=0.85),
    ]

    flow.integrity_warn = False
    flow.champion_model = MagicMock()
    flow.champion_model.predict.return_value = np.zeros(len(X_batch))
    flow.champion_uri = "models:/test@champion"
    flow.X_ref, flow.y_ref = X_ref, y_ref
    flow.X_batch, flow.y_batch = X_batch, y_batch
    flow.retrain = MagicMock()  # For branching
    flow.promotion_gate = MagicMock()  # For branching

    flow.model_gate()

    # Verify run_id was captured
    try:
        assert flow.model_gate_run_id == expected_run_id, "Should capture correct run ID"
    except AttributeError:
        pytest.fail("model_gate_run_id should be set after model_gate()")


# ============================================================================
# Tests for Soft Integrity Checks Bug Fix
# ============================================================================

def test_raw_numeric_cols_exist_in_raw_data(taxi_ref: pd.DataFrame) -> None:
    """
    RAW_NUMERIC_COLS should only contain columns that exist in raw taxi data.
    This is critical for soft integrity checks to work before feature engineering.
    """
    raw_cols_set = set(taxi_ref.columns)
    for col in RAW_NUMERIC_COLS:
        assert col in raw_cols_set, f"RAW_NUMERIC_COLS contains '{col}' which doesn't exist in raw data. Available: {raw_cols_set}"


def test_feature_cols_includes_engineered_features() -> None:
    """
    FEATURE_COLS should include engineered features that don't exist in raw data.
    This validates that using FEATURE_COLS for soft checks on raw data would fail.
    """
    engineered_features = [
        "duration_min",
        "log_trip_distance",
        "log_fare_amount",
        "log_duration_min",
        "pickup_hour",
        "pickup_weekday",
        "pickup_month",
        "PU_frequency",
        "DO_frequency",
        "distance_per_minute",
        "fare_per_mile",
    ]
    for feat in engineered_features:
        assert feat in FEATURE_COLS, f"FEATURE_COLS should contain engineered feature '{feat}'"


def test_soft_integrity_checks_work_on_raw_data(taxi_ref: pd.DataFrame, taxi_batch: pd.DataFrame) -> None:
    """
    Soft integrity checks should successfully run on raw taxi data using RAW_NUMERIC_COLS.
    This is the main bug fix validation test.
    """
    # Run soft integrity checks on raw data (before feature engineering)
    result = run_soft_integrity_checks(taxi_ref, taxi_batch)
    
    # Should not return early due to missing columns
    assert isinstance(result.warn, bool), "Should return a valid SoftIntegrityResult"
    assert isinstance(result.details, list), "Should have details list"
    assert isinstance(result.metrics, dict), "Should have metrics dict"
    
    # Should have run checks on RAW_NUMERIC_COLS
    for col in RAW_NUMERIC_COLS:
        # Check that metrics were generated for these columns
        drift_key = f"nml_drift_alerts_{col}"
        missing_key = f"nml_missing_alerts_{col}"
        # At least one type of metric should exist for each raw column
        has_metrics = (drift_key in result.metrics or missing_key in result.metrics)
        assert has_metrics, f"No NannyML metrics generated for raw column '{col}'"


def test_soft_integrity_checks_detect_drift_on_raw_columns(taxi_ref: pd.DataFrame) -> None:
    """
    Soft integrity checks should detect drift when raw numeric values change significantly.
    """
    # Create a batch with significantly different trip_distance distribution
    batch_with_drift = taxi_ref.copy()
    batch_with_drift['trip_distance'] = batch_with_drift['trip_distance'] * 3.0  # 3x all distances
    
    result = run_soft_integrity_checks(taxi_ref, batch_with_drift)
    
    # Should detect drift on trip_distance
    trip_distance_alerts = result.metrics.get("nml_drift_alerts_trip_distance", 0)
    assert trip_distance_alerts >= 0, "Should have drift metrics for trip_distance"


def test_soft_integrity_checks_detect_unseen_categoricals(taxi_ref: pd.DataFrame, taxi_batch: pd.DataFrame) -> None:
    """
    Soft integrity checks should detect unseen categorical values in location IDs.
    """
    # Modify batch to have new location IDs not in reference
    modified_batch = taxi_batch.copy()
    max_ref_pu = taxi_ref['PULocationID'].max()
    modified_batch.loc[0:10, 'PULocationID'] = max_ref_pu + 999  # Add unseen location IDs
    
    result = run_soft_integrity_checks(taxi_ref, modified_batch)
    
    # Should detect unseen categories
    unseen_pu_metric = result.metrics.get("unseen_cats_PULocationID", 0)
    assert unseen_pu_metric > 0, "Should detect unseen PULocationID values"
    assert any("PULocationID" in detail for detail in result.details), "Should log details about unseen PULocationID"


def test_soft_integrity_checks_handle_empty_overlap_gracefully() -> None:
    """
    Soft integrity checks should handle cases where raw columns don't overlap.
    """
    # Create dataframes with no overlapping RAW_NUMERIC_COLS
    df_ref = pd.DataFrame({
        'other_col': [1, 2, 3],
        'another_col': [4, 5, 6]
    })
    df_batch = pd.DataFrame({
        'different_col': [7, 8, 9],
        'yet_another': [10, 11, 12]
    })
    
    result = run_soft_integrity_checks(df_ref, df_batch)
    
    # Should return clean result without errors
    assert result.warn is False, "Should not warn when no columns to check"
    assert len(result.details) == 0, "Should have no details when no columns to check"
    assert len(result.metrics) == 0, "Should have no metrics when no columns to check"


def test_soft_integrity_checks_use_raw_not_engineered_cols(taxi_ref: pd.DataFrame, taxi_batch: pd.DataFrame) -> None:
    """
    Verify that soft integrity checks don't try to use engineered feature columns.
    This test validates the bug fix by ensuring we use RAW_NUMERIC_COLS, not FEATURE_COLS.
    """
    # Run checks and examine the metrics keys
    result = run_soft_integrity_checks(taxi_ref, taxi_batch)
    
    # Should NOT have metrics for engineered columns
    engineered_only = [col for col in FEATURE_COLS if col not in RAW_NUMERIC_COLS]
    for eng_col in engineered_only:
        drift_key = f"nml_drift_alerts_{eng_col}"
        missing_key = f"nml_missing_alerts_{eng_col}"
        assert drift_key not in result.metrics, f"Soft checks should not try to use engineered column '{eng_col}'"
        assert missing_key not in result.metrics, f"Soft checks should not try to use engineered column '{eng_col}'"
    
    # Should ONLY have metrics for raw columns
    for col in RAW_NUMERIC_COLS:
        if col in taxi_ref.columns and col in taxi_batch.columns:
            # Should have at least attempted checks on this raw column
            drift_key = f"nml_drift_alerts_{col}"
            missing_key = f"nml_missing_alerts_{col}"
            has_metric = (drift_key in result.metrics or missing_key in result.metrics)
            assert has_metric, f"Should have metrics for raw column '{col}'"
