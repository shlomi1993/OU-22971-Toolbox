import logging
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import os
import pandas as pd

from metaflow import FlowSpec, Parameter, step
from mlflow.tracking import MlflowClient

from capstone_lib import (
    DEFAULT_EXPERIMENT,
    DEFAULT_URI,
    FEATURE_COLS,
    MIN_IMPROVEMENT_PCT,
    MODEL_NAME,
    ModelRegistry,
    Decision,
    DecisionAction,
    build_model,
    evaluate_model,
    load_taxi_table,
    engineer_features,
    run_integrity_checks,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class MLFlowCapstoneFlow(FlowSpec):
    """
    Manual monitoring loop: load → integrity → features → evaluate → retrain → promote.
    """

    reference_path = Parameter(
        "reference-path",
        help="Path to reference parquet (e.g. 2020-01)",
        required=True,
    )
    batch_path = Parameter(
        "batch-path",
        help="Path to new batch parquet (e.g. 2020-04)",
        required=True,
    )
    tracking_uri = Parameter(
        "tracking-uri",
        help="MLflow tracking URI",
        default=DEFAULT_URI,
    )
    experiment_name = Parameter(
        "experiment-name",
        help="MLflow experiment name",
        default=DEFAULT_EXPERIMENT,
    )
    model_name = Parameter(
        "model-name",
        help="MLflow registered model name",
        default=MODEL_NAME,
    )
    min_improvement = Parameter(
        "min-improvement",
        help="Minimum RMSE improvement fraction to promote (default 1%%)",
        default=MIN_IMPROVEMENT_PCT,
        type=float,
    )

    # Initialize MLflow
    @step
    def start(self):
        # Initialize conditional attributes
        self.batch_rejected = False
        self.integrity_warn = False
        self.retrain_needed = False
        self.did_retrain = False

        # Initialize MLflow tracking and registry
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.registry = ModelRegistry(MlflowClient(), self.model_name)

        logger.info(f"MLflow configured: {self.tracking_uri} / experiment={self.experiment_name}")
        self.next(self.load_data)

    # Step A — Load Data
    @step
    def load_data(self):
        self.df_ref = load_taxi_table(self.reference_path)
        self.df_batch = load_taxi_table(self.batch_path)
        logger.info(f"Loaded reference ({len(self.df_ref)} rows) and batch ({len(self.df_batch)} rows)")
        self.next(self.integrity_gate)

    # Step B — Integrity Gate
    @step
    def integrity_gate(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="integrity_gate") as run:
            self.integrity_run_id = run.info.run_id
            mlflow.set_tag("pipeline_step", "integrity_gate")
            mlflow.set_tag("batch_path", self.batch_path)
            mlflow.set_tag("reference_path", self.reference_path)

            # Combined hard + soft integrity checks
            ok, report = run_integrity_checks(self.df_ref, self.df_batch)

            # Log metrics and artifacts
            mlflow.log_metrics(report["metrics"])
            mlflow.log_dict({"hard_failures": report["hard"]["failures"]}, "hard_failures.json")
            mlflow.log_dict({"nannyml_details": report["nannyml"].get("details", [])}, "nannyml_details.json")

            if not ok:
                logger.warning("HARD INTEGRITY FAILURE — rejecting batch")
                decision = Decision(
                    action=DecisionAction.REJECT_BATCH,
                    reason="; ".join(report["hard"]["failures"]),
                    metrics=report["hard"]["metrics"],
                )
                decision.log()
                self.batch_rejected = True
                self.integrity_warn = False
                self.next(self.feature_engineering)
                return

            # Batch passed hard checks
            nannyml_warn = report["nannyml"].get("warn", False)
            mlflow.set_tag("integrity_warn", str(nannyml_warn).lower())

            self.integrity_warn = nannyml_warn
            self.batch_rejected = False

            decision = Decision(
                action=DecisionAction.BATCH_ACCEPTED,
                reason="Hard rules passed" + ("; NannyML warnings present" if nannyml_warn else ""),
                metrics=report["hard"]["metrics"],
                details={"nannyml_warn": nannyml_warn, "nannyml_details": report["nannyml"].get("details", [])},
            )
            decision.log()

        logger.info(f"Integrity gate passed (warn={self.integrity_warn})")
        self.next(self.feature_engineering)

    # Step C — Feature Engineering
    @step
    def feature_engineering(self):
        if self.batch_rejected:
            logger.info("Skipping feature engineering — batch was rejected")
            self.X_ref = None
            self.y_ref = None
            self.X_batch = None
            self.y_batch = None
            self.feature_cols = []
            self.next(self.load_champion)
            return

        self.X_ref, self.y_ref = engineer_features(self.df_ref)
        self.X_batch, self.y_batch = engineer_features(self.df_batch)
        self.feature_cols = FEATURE_COLS

        # Log feature spec to MLflow (design doc Step C)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name="feature_engineering"):
            mlflow.set_tag("pipeline_step", "feature_engineering")
            feature_spec = {c: str(self.X_ref[c].dtype) for c in self.feature_cols}
            mlflow.log_dict({"feature_cols": self.feature_cols, "dtypes": feature_spec}, "feature_cols.json")

        logger.info(f"Features engineered: ref={len(self.X_ref)}, batch={len(self.X_batch)}, cols={self.feature_cols}")
        self.next(self.load_champion)

    # Step D — Load Champion
    @step
    def load_champion(self):
        if self.batch_rejected:
            logger.info("Skipping champion load — batch was rejected")
            self.champion_model = None
            self.champion_uri = None
            self.next(self.model_gate)
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # Bootstrap champion if none exists (first run)
        if not self.registry.champion_exists():
            logger.info("No champion found — bootstrapping initial model on reference data")

            with mlflow.start_run(run_name="bootstrap_train") as run:
                mlflow.set_tag("pipeline_step", "train")
                mlflow.set_tag("bootstrap", "true")

                model = build_model()
                model.fit(self.X_ref, self.y_ref)

                metrics = evaluate_model(model, self.X_ref, self.y_ref)
                mlflow.log_metrics({f"train_{k}": v for k, v in metrics.as_dict().items()})
                mlflow.log_dict({"feature_cols": FEATURE_COLS}, "feature_cols.json")

                model_info = mlflow.sklearn.log_model(model, name="model", input_example=self.X_ref.head(5))
                mlflow.set_tag("model_uri", model_info.model_uri)

                version = self.registry.register_version(
                    model_info.model_uri,
                    tags={
                        "role": "champion",
                        "trained_on": "reference",
                        "validation_status": "approved",
                    },
                )
                self.registry.promote_to_champion(version, reason="bootstrap")
                logger.info(f"Bootstrap champion registered: version={version}")

        # Load champion
        self.champion_model, self.champion_uri = self.registry.load_champion()
        logger.info(f"Champion loaded: {self.champion_uri}")
        self.next(self.model_gate)

    # Step E — Model Gate
    @step
    def model_gate(self):
        if self.batch_rejected:
            logger.info("Skipping model gate — batch was rejected")
            self.retrain_needed = False
            self.rmse_champion_on_batch = None
            self.rmse_champion_on_ref = None
            self.rmse_increase_pct = None
            self.next(self.retrain)
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="model_gate") as run:
            self.model_gate_run_id = run.info.run_id
            mlflow.set_tag("pipeline_step", "model_gate")
            mlflow.set_tag("champion_uri", self.champion_uri)
            mlflow.set_tag("integrity_warn", str(self.integrity_warn).lower())

            # Evaluate champion on batch
            champ_metrics = evaluate_model(self.champion_model, self.X_batch, self.y_batch)
            mlflow.log_metrics({f"champion_{k}": v for k, v in champ_metrics.as_dict().items()})

            # Evaluate champion on reference (baseline)
            ref_metrics = evaluate_model(self.champion_model, self.X_ref, self.y_ref)
            mlflow.log_metrics({f"baseline_{k}": v for k, v in ref_metrics.as_dict().items()})

            # Log evaluation dataset lineage (P1 requirement)
            eval_dataset = mlflow.data.from_pandas(self.X_batch, name="batch_eval")
            mlflow.log_input(eval_dataset, context="evaluation")

            # Inference demo: log champion predictions on batch
            y_pred_champ = self.champion_model.predict(self.X_batch)
            pred_df = self.X_batch.copy()
            pred_df["prediction"] = y_pred_champ
            pred_df["actual"] = self.y_batch
            pred_path = "predictions.parquet"
            pred_df.to_parquet(pred_path, index=False)
            mlflow.log_artifact(pred_path)
            os.remove(pred_path)

            rmse_champ = champ_metrics.rmse
            rmse_base = ref_metrics.rmse
            rmse_increase_pct = (rmse_champ - rmse_base) / max(rmse_base, 1e-9) if rmse_base > 0 else 0.0
            mlflow.log_metric("rmse_increase_pct", rmse_increase_pct)

            self.rmse_champion_on_batch = rmse_champ
            self.rmse_champion_on_ref = rmse_base
            self.rmse_increase_pct = rmse_increase_pct

            # Decision: retrain if RMSE increased >5 %
            retrain_needed = rmse_increase_pct > 0.05
            self.retrain_needed = retrain_needed

            reason = (
                f"RMSE increase {rmse_increase_pct:.2%} > 5% threshold"
                if retrain_needed
                else f"RMSE increase {rmse_increase_pct:.2%} within tolerance"
            )
            # Integrity warnings lower the retrain threshold
            if self.integrity_warn and not retrain_needed:
                retrain_needed = rmse_increase_pct > 0.03
                self.retrain_needed = retrain_needed
                if retrain_needed:
                    reason += " (lowered threshold due to integrity warnings)"

            decision = Decision(
                action=DecisionAction.RETRAIN if self.retrain_needed else DecisionAction.NO_RETRAIN,
                retrain_recommended=self.retrain_needed,
                reason=reason,
                metrics={
                    "rmse_champion_on_batch": rmse_champ,
                    "rmse_champion_on_ref": rmse_base,
                    "rmse_increase_pct": rmse_increase_pct,
                },
            )
            decision.log()
            mlflow.set_tag("retrain_recommended", str(self.retrain_needed).lower())

        logger.info(f"Model gate: retrain_needed={self.retrain_needed} ({reason})")
        self.next(self.retrain)

    # Step F — Retrain if needed
    @step
    def retrain(self):
        self.did_retrain = False
        self.candidate_model_uri = None
        self.candidate_rmse_batch = None
        self.candidate_rmse_ref = None
        self.retrain_run_id = None

        if not self.retrain_needed:
            logger.info("Retrain step: skipped (not needed)")
            self.next(self.promotion_gate)
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # Train on merged reference + batch
        X_train = pd.concat([self.X_ref, self.X_batch], ignore_index=True)
        y_train = np.concatenate([self.y_ref, self.y_batch])

        with mlflow.start_run(run_name="retrain") as run:
            self.retrain_run_id = run.info.run_id
            mlflow.set_tag("pipeline_step", "retrain")
            mlflow.set_tag("trained_on_batches", f"{self.reference_path},{self.batch_path}")

            model = build_model()
            model.fit(X_train, y_train)

            # Log training dataset lineage
            train_dataset = mlflow.data.from_pandas(X_train, name="train_combined")
            mlflow.log_input(train_dataset, context="training")

            # Evaluate candidate on batch (same eval set as champion)
            candidate_metrics = evaluate_model(model, self.X_batch, self.y_batch)
            mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_metrics.as_dict().items()})

            # Evaluate candidate on reference (stability check — P3)
            ref_metrics = evaluate_model(model, self.X_ref, self.y_ref)
            mlflow.log_metrics({f"candidate_ref_{k}": v for k, v in ref_metrics.as_dict().items()})

            mlflow.log_dict({"feature_cols": FEATURE_COLS}, "feature_cols.json")
            mlflow.log_params({"reference_path": self.reference_path, "batch_path": self.batch_path})

            model_info = mlflow.sklearn.log_model(model, name="model", input_example=self.X_batch.head(5))
            mlflow.set_tag("model_uri", model_info.model_uri)

            # Inference demo: log batch predictions as artifact
            y_pred = model.predict(self.X_batch)
            pred_df = self.X_batch.copy()
            pred_df["prediction"] = y_pred
            pred_df["actual"] = self.y_batch
            pred_path = "predictions.parquet"
            pred_df.to_parquet(pred_path, index=False)
            mlflow.log_artifact(pred_path)
            os.remove(pred_path)

            self.candidate_model_uri = model_info.model_uri
            self.candidate_rmse_batch = candidate_metrics.rmse
            self.candidate_rmse_ref = ref_metrics.rmse
            self.did_retrain = True

        logger.info(f"Retrain done: candidate RMSE={self.candidate_rmse_batch:.4f} (batch), {self.candidate_rmse_ref:.4f} (ref)")
        self.next(self.promotion_gate)

    # Step G — Promotion Gate
    @step
    def promotion_gate(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="promotion_gate") as run:
            mlflow.set_tag("pipeline_step", "promotion_gate")

            if not self.did_retrain:
                decision = Decision(
                    action=DecisionAction.NO_PROMOTE,
                    reason="No retraining was performed",
                    metrics={
                        "rmse_champion_on_batch": self.rmse_champion_on_batch,
                    } if self.rmse_champion_on_batch is not None else {},
                )
                decision.log()
                logger.info("Promotion gate: skipped (no retrain)")
                self.next(self.end)
                return

            # Promotion criteria
            rmse_champ = self.rmse_champion_on_batch
            rmse_cand = self.candidate_rmse_batch
            rmse_cand_ref = self.candidate_rmse_ref

            mlflow.log_metrics({
                "champion_rmse_batch": rmse_champ,
                "candidate_rmse_batch": rmse_cand,
                "candidate_rmse_ref": rmse_cand_ref,
                "champion_rmse_ref": self.rmse_champion_on_ref,
                "min_improvement_pct": self.min_improvement,
            })

            # P1: Evaluation is valid
            p1 = rmse_cand is not None and rmse_champ is not None

            # P2: Candidate beats champion meaningfully
            threshold = rmse_champ * (1 - self.min_improvement)
            p2 = rmse_cand < threshold

            # P3: Stability — candidate doesn't regress on reference by >5%
            ref_regression = (
                (rmse_cand_ref - self.rmse_champion_on_ref) / max(self.rmse_champion_on_ref, 1e-9)
                if rmse_cand_ref is not None and self.rmse_champion_on_ref > 0
                else 0.0
            )
            p3 = ref_regression < 0.05

            # P4: No hard integrity failures
            p4 = not self.batch_rejected

            promote = p1 and p2 and p3 and p4
            reasons = []
            if not p1:
                reasons.append("missing evaluation metrics")
            if not p2:
                reasons.append(f"candidate RMSE {rmse_cand:.4f} >= threshold {threshold:.4f}")
            if not p3:
                reasons.append(f"reference regression {ref_regression:.2%} > 5%")
            if not p4:
                reasons.append("batch was rejected by integrity gate")
            if promote:
                reasons.append(f"candidate RMSE {rmse_cand:.4f} < {threshold:.4f} "
                               f"(>{self.min_improvement:.0%} better than champion {rmse_champ:.4f})")

            decision = Decision(
                action=DecisionAction.PROMOTE if promote else DecisionAction.NO_PROMOTE,
                retrain_recommended=True,
                promotion_recommended=promote,
                reason="; ".join(reasons),
                metrics={
                    "champion_rmse_batch": rmse_champ,
                    "candidate_rmse_batch": rmse_cand,
                    "candidate_rmse_ref": rmse_cand_ref,
                    "ref_regression_pct": ref_regression,
                    "min_improvement_pct": self.min_improvement,
                },
                details={"p1": p1, "p2": p2, "p3": p3, "p4": p4},
            )
            decision.log()

            if promote:
                version = self.registry.register_version(
                    self.candidate_model_uri,
                    tags={
                        "role": "candidate",
                        "trained_on_batches": f"{self.reference_path},{self.batch_path}",
                        "eval_batch_id": self.batch_path,
                        "validation_status": "approved",
                        "decision_reason": decision.reason,
                    },
                )
                self.registry.promote_to_champion(version, reason=decision.reason)
                mlflow.set_tag("promoted_version", version)
                logger.info(f"PROMOTED candidate to champion: version={version}")
            else:
                logger.info(f"Promotion declined: {decision.reason}")
                # Still register as rejected candidate for audit trail
                if self.candidate_model_uri:
                    version = self.registry.register_version(
                        self.candidate_model_uri,
                        tags={
                            "role": "candidate",
                            "validation_status": "rejected",
                            "decision_reason": decision.reason,
                        },
                    )

        self.next(self.end)

    # End
    @step
    def end(self):
        if self.batch_rejected:
            logger.info("Flow finished: batch was REJECTED by integrity gate.")
        elif self.did_retrain:
            logger.info("Flow finished: retrain + promotion gate completed.")
        else:
            logger.info("Flow finished: champion is still adequate, no retrain needed.")


if __name__ == "__main__":
    MLFlowCapstoneFlow()
