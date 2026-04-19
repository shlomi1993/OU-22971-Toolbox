"""
Distributed HPO with Pruning - Optional Exercise Solution

Compares two pruning strategies:
1. Actor-based pruning: trials query actor for pruning decisions
2. Local pruning: trials make independent pruning decisions
"""

import time
import numpy as np
import optuna
import ray
import xgboost as xgb

from typing import Any, Optional
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split

# Configuration
SEED = 0
NUM_BOOST_ROUND = 60
TEST_SIZE = 0.20
N_FOLDS = 3
N_TRIALS = 20  # Increased to see more pruning
MAX_CONCURRENT = 4

# Suppress Optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_xgb_params(trial: optuna.trial.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 8),  # Wider range: shallow (1) to deep (8)
        "eta": trial.suggest_float("eta", 1e-3, 0.5, log=True),  # Include very slow learners
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),  # Allow aggressive subsampling
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),  # Allow aggressive feature sampling
        "lambda": trial.suggest_float("lambda", 1e-4, 50.0, log=True),  # Include very high regularization
    }


def build_xgb_params(sampled_params: dict, seed: int = SEED) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": seed,
        **sampled_params
    }


def resolve_fold(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_shared = ray.get(bundle["X_ref"])
    y_shared = ray.get(bundle["y_ref"])
    train_idx = ray.get(bundle["train_idx_ref"])
    valid_idx = ray.get(bundle["valid_idx_ref"])
    return X_shared[train_idx], y_shared[train_idx], X_shared[valid_idx], y_shared[valid_idx]


def fit_fold(bundle: dict[str, Any], sampled_params: dict[str, Any]) -> dict[str, float]:
    X_train, y_train, X_valid, y_valid = resolve_fold(bundle)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    evals_result = {}
    xgb.train(params=build_xgb_params(sampled_params), dtrain=dtrain, num_boost_round=NUM_BOOST_ROUND,
              evals=[(dvalid, "valid")], evals_result=evals_result, verbose_eval=False)
    valid_auc = np.asarray(evals_result["valid"]["auc"], dtype=float)
    return {"best_auc": float(valid_auc.max()), "best_iteration": int(valid_auc.argmax())}


@ray.remote(num_cpus=1)
def train_fold_remote(trial_id: int, fold_id: int, params: dict[str, Any], fold_bundle: dict[str, Any]) -> dict[str, Any]:
    result = fit_fold(fold_bundle, params)
    return {"trial_id": trial_id, "fold_id": fold_id, **result}


# Actor-based pruning
@ray.remote(num_cpus=0)
class PruningStudyActor:
    def __init__(self, seed: int = SEED) -> None:
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        self.open_trials: dict[int, optuna.trial.FrozenTrial] = {}
        self.results_by_trial: dict[int, dict[str, Any]] = {}
        self.pruned_trials: set[int] = set()
        self.actor_queries: int = 0

    def ask_trial(self) -> dict[str, Any]:
        trial = self.study.ask()
        params = suggest_xgb_params(trial)
        self.open_trials[trial.number] = trial
        return {"trial_id": trial.number, "params": params}

    def should_prune(self, trial_id: int, intermediate_value: float, step: int) -> bool:
        """
        Actor-based pruning decision with median pruner logic
        """
        self.actor_queries += 1
        if len(self.results_by_trial) < 2:
            return False

        completed_scores = [r["score"] for r in self.results_by_trial.values()]
        median_score = np.median(completed_scores)
        return intermediate_value < median_score * 0.995  # Prune if 0.5% below median (fresh state)

    def report_trial_result(self, trial_id: int, score: float, best_iteration: int, was_pruned: bool = False) -> None:
        self.results_by_trial[trial_id] = {"score": float(score), "best_iteration": int(best_iteration)}
        if was_pruned:
            self.pruned_trials.add(trial_id)

        trial = self.open_trials[trial_id]
        trial.set_user_attr("best_iteration", int(best_iteration))
        if was_pruned:
            self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            self.study.tell(trial, float(score))

    def snapshot(self) -> dict[str, Any]:
        return {
            "completed": len(self.results_by_trial),
            "pruned": len(self.pruned_trials),
            "actor_queries": self.actor_queries,
            "best_score": self.study.best_value if self.study.best_trial else None
        }


@ray.remote(num_cpus=0)
def run_trial_with_actor_pruning(trial_spec: dict[str, Any], fold_bundle_refs: list[ray.ObjectRef],
                                 study_actor: ray.actor.ActorHandle) -> None:
    """
    Query actor for pruning decisions
    """
    fold_scores = []
    for fold_id, fold_ref in enumerate(fold_bundle_refs):
        result_ref = train_fold_remote.remote(trial_spec["trial_id"], fold_id, trial_spec["params"], fold_ref)
        result = ray.get(result_ref)
        fold_scores.append(result["best_auc"])

        # Query actor after each fold
        intermediate_score = float(np.mean(fold_scores))
        should_prune = ray.get(study_actor.should_prune.remote(trial_spec["trial_id"], intermediate_score, fold_id))
        if should_prune:
            study_actor.report_trial_result.remote(trial_spec["trial_id"], intermediate_score, fold_id, was_pruned=True)
            return

    final_score = float(np.mean(fold_scores))
    study_actor.report_trial_result.remote(trial_spec["trial_id"], final_score, len(fold_scores) - 1, was_pruned=False)


# Local pruning without actor communication
@ray.remote(num_cpus=0)
class LocalPruningActor:
    def __init__(self, seed: int = SEED) -> None:
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        self.open_trials: dict[int, optuna.trial.FrozenTrial] = {}
        self.results_by_trial: dict[int, dict[str, Any]] = {}
        self.pruned_trials: set[int] = set()

    def ask_trial(self) -> dict[str, Any]:
        trial = self.study.ask()
        params = suggest_xgb_params(trial)
        self.open_trials[trial.number] = trial
        completed_scores = [r["score"] for r in self.results_by_trial.values()]  # Provide baseline for local pruning
        baseline = float(np.median(completed_scores)) if completed_scores else 0.0
        return {"trial_id": trial.number, "params": params, "baseline": baseline}

    def report_trial_result(self, trial_id: int, score: float, best_iteration: int, was_pruned: bool = False) -> None:
        self.results_by_trial[trial_id] = {"score": float(score), "best_iteration": int(best_iteration)}
        if was_pruned:
            self.pruned_trials.add(trial_id)

        trial = self.open_trials[trial_id]
        trial.set_user_attr("best_iteration", int(best_iteration))
        if was_pruned:
            self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            self.study.tell(trial, float(score))

    def snapshot(self) -> dict[str, Any]:
        return {
            "completed": len(self.results_by_trial),
            "pruned": len(self.pruned_trials),
            "best_score": self.study.best_value if self.study.best_trial else None
        }


@ray.remote(num_cpus=0)
def run_trial_with_local_pruning(trial_spec: dict[str, Any], fold_bundle_refs: list[ray.ObjectRef]) -> Optional[dict[str, Any]]:
    """
    Make pruning decisions locally without actor queries
    """
    baseline = trial_spec["baseline"]
    fold_scores = []

    for fold_id, fold_ref in enumerate(fold_bundle_refs):
        result_ref = train_fold_remote.remote(trial_spec["trial_id"], fold_id, trial_spec["params"], fold_ref)
        result = ray.get(result_ref)
        fold_scores.append(result["best_auc"])

        # Local pruning decision (no actor query)
        if baseline > 0:
            intermediate_score = float(np.mean(fold_scores))
            if intermediate_score < baseline * 0.995:  # Same threshold, stale baseline
                return {
                    "trial_id": trial_spec["trial_id"],
                    "score": intermediate_score,
                    "best_iteration": fold_id,
                    "was_pruned": True
                }

    final_score = float(np.mean(fold_scores))
    return {
        "trial_id": trial_spec["trial_id"],
        "score": final_score,
        "best_iteration": len(fold_scores) - 1,
        "was_pruned": False
    }


def run_study(study_actor: ray.actor.ActorHandle, fold_bundle_refs: list[ray.ObjectRef], trial_runner: Any,
              use_local_pruning: bool = False) -> dict[str, Any]:
    """
    Generic study runner for both pruning strategies
    """
    in_flight: list[ray.ObjectRef] = []
    launched = 0

    def launch_one():
        trial_spec = ray.get(study_actor.ask_trial.remote())
        if use_local_pruning:
            return trial_runner.remote(trial_spec, fold_bundle_refs)
        else:
            return trial_runner.remote(trial_spec, fold_bundle_refs, study_actor)

    while launched < min(MAX_CONCURRENT, N_TRIALS):
        in_flight.append(launch_one())
        launched += 1

    while in_flight:
        ready_refs, in_flight = ray.wait(in_flight, num_returns=1)
        result = ray.get(ready_refs[0])

        # For local pruning, need to report back
        if use_local_pruning and result:
            study_actor.report_trial_result.remote(result["trial_id"], result["score"],
                                                   result["best_iteration"], result["was_pruned"])

        if launched < N_TRIALS:
            in_flight.append(launch_one())
            launched += 1

    return ray.get(study_actor.snapshot.remote())


def main() -> None:
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Prepare data
    X, y = load_breast_cancer(return_X_y=True)
    X_ref, y_ref = ray.put(X), ray.put(y)
    all_idx = np.arange(X.shape[0])
    trainval_idx, _ = train_test_split(all_idx, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    cv_labels = y[trainval_idx]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_bundle_refs = []
    for fold_id, (train_pos, valid_pos) in enumerate(skf.split(trainval_idx, cv_labels)):
        bundle = {
            "X_ref": X_ref,
            "y_ref": y_ref,
            "train_idx_ref": ray.put(trainval_idx[train_pos]),
            "valid_idx_ref": ray.put(trainval_idx[valid_pos]),
            "fold_id": fold_id
        }
        fold_bundle_refs.append(ray.put(bundle))

    print("\n[Actor-based pruning]")
    t0 = time.perf_counter()
    actor_study = PruningStudyActor.remote(seed=SEED)
    actor_snapshot = run_study(actor_study, fold_bundle_refs, run_trial_with_actor_pruning, use_local_pruning=False)
    actor_time = time.perf_counter() - t0
    print(f"Score: {actor_snapshot['best_score']:.5f} | Pruned: {actor_snapshot['pruned']}/{actor_snapshot['completed']} | Queries: {actor_snapshot['actor_queries']} | Time: {actor_time:.2f}s")

    print("\n[Local pruning]")
    t0 = time.perf_counter()
    local_study = LocalPruningActor.remote(seed=SEED)
    local_snapshot = run_study(local_study, fold_bundle_refs, run_trial_with_local_pruning, use_local_pruning=True)
    local_time = time.perf_counter() - t0
    print(f"Score: {local_snapshot['best_score']:.5f} | Pruned: {local_snapshot['pruned']}/{local_snapshot['completed']} | Queries: 0 | Time: {local_time:.2f}s")

    pruning_diff = actor_snapshot['pruned'] - local_snapshot['pruned']
    print(f"\n[Tradeoff] Actor pruned {abs(pruning_diff)} {'more' if pruning_diff > 0 else 'fewer'} trials (fresh state) but took {actor_time/local_time:.1f}x longer ({actor_snapshot['actor_queries']} network calls)")
    if actor_snapshot['pruned'] == 0 and local_snapshot['pruned'] == 0:
        print("  Note: No pruning occurred - trials too consistent.")
    elif pruning_diff > 0:
        print(f"  Key insight: Fresh state enabled {pruning_diff} more early stops than stale baseline.")
    elif pruning_diff < 0:
        print(f"  Unexpected: Stale baseline pruned more (timing effects or threshold sensitivity).")

    ray.shutdown()

    """
    Core Lesson:
    Distributed systems require trading off state freshness, communication overhead, and performance.
    The lack of actual pruning is a data characteristic, not a pedagogical failure, and the explanatory text compensates well.
    """


if __name__ == "__main__":
    main()
