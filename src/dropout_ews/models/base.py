"""Model pipelines. Everything fitted lives inside them (ADR-0003 control #4).

The pipeline takes a DataFrame carrying **cohort keys plus raw builder
features**, not a bare feature matrix. That is deliberate:
:class:`~dropout_ews.preprocessing.cohort.CohortZScorer` needs
``(code_module, code_presentation, checkpoint_day)`` to compute cohort
statistics, and those statistics must be fitted per fold. During Phase 4
screening the z-scorer was fitted outside the folds, which was fine for feature
selection but would be preprocessing leakage in any reported metric.

Stage order::

    CohortZScorer      adds <feature>_cohort_z, fitted on the fold only
    FeatureSelector    narrows to the features.yaml allowlist, in a fixed order
    SimpleImputer      median fill plus missingness indicators
    StandardScaler     linear models only; trees do not need it
    classifier

Missingness indicators matter here rather than being boilerplate. Several
features are NaN for genuine structural reasons — ``days_since_last_activity``
for the 3,095 never-active rows, ``mean_score`` before any assessment is due —
and "this student has never engaged" is itself a signal. Median-filling without
an indicator would erase it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from dropout_ews.config.settings import load_feature_config
from dropout_ews.preprocessing.cohort import COHORT_KEYS, CohortZScorer


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select the allowlisted features, in a fixed order.

    Ordering is part of the contract: SHAP output, stored feature hashes, and
    the model artifact all assume a stable column order, and a silently
    reordered matrix would produce explanations attributed to the wrong
    features.

    Raises on a missing column rather than filling it, because a quietly absent
    feature trains a different model than the config describes.
    """

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names

    def _resolve(self) -> list[str]:
        if self.feature_names is not None:
            return list(self.feature_names)
        return load_feature_config().features.all_features()

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureSelector:
        self.feature_names_in_ = list(X.columns)
        self.selected_ = self._resolve()
        missing = [name for name in self.selected_ if name not in X.columns]
        if missing:
            raise ValueError(
                f"allowlisted features missing from the input frame: {missing}. "
                "Either the builder did not produce them or CohortZScorer did not run."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_"):
            raise ValueError("FeatureSelector must be fitted before transform")
        missing = [name for name in self.selected_ if name not in X.columns]
        if missing:
            raise ValueError(f"allowlisted features missing at transform time: {missing}")
        return X[self.selected_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.selected_)


def imputer_feature_names(selected: list[str], imputer: SimpleImputer) -> list[str]:
    """Column names after ``SimpleImputer(add_indicator=True)``.

    scikit-learn appends one indicator per column that had a missing value *at
    fit time*, so the output width is not knowable from the input alone. Getting
    this wrong misaligns every SHAP attribution, so the names are derived from
    the fitted ``indicator_`` rather than assumed.
    """
    names = list(selected)
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None and getattr(indicator, "features_", None) is not None:
        names += [f"{selected[i]}__missing" for i in indicator.features_]
    return names


def make_pipeline(
    classifier: BaseEstimator,
    scale: bool = False,
    feature_names: list[str] | None = None,
    min_cohort_size: int = 30,
) -> Pipeline:
    """Build the standard pipeline around a classifier.

    Args:
        classifier: the estimator to fit.
        scale: standardise features. Needed for regularised linear models,
            pointless for trees.
        feature_names: override the allowlist, used by ablation experiments.
        min_cohort_size: passed to :class:`CohortZScorer`.

    ``imblearn.pipeline.Pipeline`` is used rather than scikit-learn's so that
    Phase 6 can insert a resampler at the same point without restructuring.
    """
    steps: list[tuple[str, BaseEstimator]] = [
        ("cohort", CohortZScorer(min_cohort_size=min_cohort_size)),
        ("select", FeatureSelector(feature_names=feature_names)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def required_input_columns(feature_names: list[str] | None = None) -> list[str]:
    """Columns a frame must carry to be fed to a pipeline.

    The allowlist minus the cohort-derived features (which the pipeline adds
    itself), plus the cohort keys it needs to add them.
    """
    allowlist = feature_names or load_feature_config().features.all_features()
    derived = set(load_feature_config().features.relative)
    return sorted({*COHORT_KEYS, *(name for name in allowlist if name not in derived)})
