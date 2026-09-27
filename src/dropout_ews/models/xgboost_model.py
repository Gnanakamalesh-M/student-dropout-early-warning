"""XGBoost pipeline — the primary model.

``scale_pos_weight`` handles the 2.7% positive rate by reweighting the loss
rather than resampling. Phase 6 tests that choice against SMOTE instead of
assuming it (see :mod:`dropout_ews.models.imbalance`).

``tree_method="hist"`` is used throughout: it is fast enough that an Optuna
study of 80 trials over 65k rows finishes in minutes, and on this data it scores
indistinguishably from the exact method.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from dropout_ews.models.base import make_pipeline

RANDOM_STATE = 42

# Starting point before tuning. Deliberately conservative: shallow trees and a
# strong minimum child weight, because at a 2.7% positive rate a deep tree
# isolates individual positives and destroys calibration.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20.0,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}


def positive_class_weight(y: np.ndarray) -> float:
    """``scale_pos_weight`` that balances the classes.

    The ratio of negatives to positives, which is what XGBoost expects. At a
    2.7% positive rate this is roughly 36.
    """
    y = np.asarray(y)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("cannot compute a positive class weight with no positives")
    return float((len(y) - positives) / positives)


def xgboost_pipeline(
    params: dict[str, Any] | None = None,
    scale_pos_weight: float | None = None,
    feature_names: list[str] | None = None,
    resampler: Any | None = None,
) -> Pipeline:
    """Build the XGBoost pipeline.

    Args:
        params: booster hyperparameters, merged over :data:`DEFAULT_PARAMS`.
        scale_pos_weight: class reweighting. ``None`` leaves XGBoost's default
            of 1, which is what the imbalance experiment needs when it is
            testing a resampler instead.
        feature_names: override the allowlist.
        resampler: an imbalanced-learn sampler inserted just before the
            classifier, so it is refitted per fold and never touches
            validation rows.
    """
    merged = {**DEFAULT_PARAMS, **(params or {})}
    classifier = XGBClassifier(
        **merged,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        # Reproducibility: XGBoost's default sampling is not deterministic
        # across thread counts without this.
        verbosity=0,
    )
    pipeline = make_pipeline(classifier, scale=False, feature_names=feature_names)
    if resampler is not None:
        # Insert immediately before the classifier: after imputation, so the
        # sampler never sees NaN, and inside the pipeline, so it is refitted
        # per fold.
        steps = list(pipeline.steps)
        steps.insert(-1, ("resample", resampler))
        pipeline = Pipeline(steps)
    return pipeline
