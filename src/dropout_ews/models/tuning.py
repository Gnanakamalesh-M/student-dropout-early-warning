"""Optuna hyperparameter search for XGBoost.

Optuna rather than ``GridSearchCV``: the search space is ~8-dimensional and
mostly continuous, where a grid wastes most of its budget, and the median pruner
abandons hopeless trials after the first couple of folds.

The objective is **mean grouped-CV average precision on the training split
only**. Validation and test are never scored during the search, so neither is
used as a tuning set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd

from dropout_ews.evaluation.metrics import ranking_metrics
from dropout_ews.models.xgboost_model import positive_class_weight, xgboost_pipeline

# Optuna is chatty by default and drowns the pipeline output.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """The search space.

    Bounded deliberately rather than left wide. ``max_depth`` stops at 7 and
    ``min_child_weight`` starts at 5 because at a 2.7% positive rate a deep tree
    with small leaves isolates individual positives: CV average precision rises
    while calibration collapses, which is the failure this project most needs to
    avoid given it displays probabilities to staff.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 80.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
    }


def tune_xgboost(
    features: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int = 80,
    random_state: int = 42,
    feature_names: list[str] | None = None,
) -> TuningResult:
    """Run the study and return the best parameters.

    Args:
        features: training-split frame carrying cohort keys and raw features.
        y: training labels.
        folds: grouped CV folds, positional into ``features``.
        n_trials: Optuna budget.
    """
    scale_pos_weight = positive_class_weight(y)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        for fold_index, (train_positions, validation_positions) in enumerate(folds):
            pipeline = xgboost_pipeline(
                params=params,
                scale_pos_weight=scale_pos_weight,
                feature_names=feature_names,
            )
            pipeline.fit(features.iloc[train_positions], y[train_positions])
            probabilities = pipeline.predict_proba(features.iloc[validation_positions])[:, 1]
            scores.append(ranking_metrics(y[validation_positions], probabilities).average_precision)
            # Report the running mean so the pruner can abandon a hopeless
            # trial without paying for every fold.
            trial.report(float(np.mean(scores)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    trials_frame = pd.DataFrame(
        [{"number": t.number, "value": t.value, **t.params} for t in completed]
    ).sort_values("value", ascending=False)

    return TuningResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        trials=trials_frame,
    )
