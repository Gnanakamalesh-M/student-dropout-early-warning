"""Cohort-relative z-scores as a fitted transformer.

Phase 3 found the checkpoint positive rate varies **4.2x across modules**
(1.25% for GGG to 5.30% for CCC), so "200 clicks in the last week" means
different things in different courses. Expressing engagement relative to the
student's own cohort is therefore necessary rather than decorative.

This is a fitted transformer, not part of :class:`FeatureBuilder`, for one
reason: cohort means and standard deviations are *statistics estimated from
data*, so fitting them on anything that includes validation or test rows is
preprocessing leakage (ADR-0003 control #3). Keeping them here means they are
fitted per fold inside a ``Pipeline`` and stored with the model, exactly like a
scaler.

Cohort is keyed on ``(code_module, code_presentation, checkpoint_day)``. The
checkpoint matters: engagement decays over a presentation for everyone, so a
z-score against the whole course would mostly measure how far through the
course a row is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COHORT_KEYS = ["code_module", "code_presentation", "checkpoint_day"]

# Features worth expressing relative to cohort. Deliberately a short list:
# z-scoring everything doubles the feature count for little gain and makes SHAP
# output harder to read.
DEFAULT_COLUMNS = (
    "clicks_7d",
    "clicks_28d",
    "active_days_28d",
    "mean_score",
    "submission_rate",
)


class CohortZScorer(BaseEstimator, TransformerMixin):
    """Add ``<column>_cohort_z`` columns, fitted on training rows only.

    Args:
        columns: feature columns to z-score. Missing columns are skipped so the
            transformer stays usable across feature-set revisions.
        min_cohort_size: cohorts smaller than this fall back to the global
            statistics. A standard deviation from three rows is noise, and
            dividing by it manufactures extreme values.

    Unseen cohorts at transform time (a new presentation in production) fall
    back to the global statistics rather than producing NaN, because a
    deployed model must still score a cohort it was not trained on.
    """

    def __init__(
        self,
        columns: tuple[str, ...] = DEFAULT_COLUMNS,
        min_cohort_size: int = 30,
    ) -> None:
        self.columns = columns
        self.min_cohort_size = min_cohort_size

    def fit(self, X: pd.DataFrame, y: object = None) -> CohortZScorer:
        missing_keys = [key for key in COHORT_KEYS if key not in X.columns]
        if missing_keys:
            raise ValueError(f"cohort keys missing from input: {missing_keys}")

        self.fitted_columns_ = [column for column in self.columns if column in X.columns]
        # Global fallback, computed once on the training rows.
        self.global_stats_ = {
            column: (float(X[column].mean(skipna=True)), float(X[column].std(skipna=True)))
            for column in self.fitted_columns_
        }
        grouped = X.groupby(COHORT_KEYS, dropna=False)
        stats = grouped[self.fitted_columns_].agg(["mean", "std", "size"])
        # Drop cohorts too small for a trustworthy standard deviation.
        sizes = grouped.size()
        keep = sizes[sizes >= self.min_cohort_size].index
        self.cohort_stats_ = stats.loc[stats.index.intersection(keep)]
        self.n_cohorts_ = len(self.cohort_stats_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fitted_columns_"):
            raise ValueError("CohortZScorer must be fitted before transform")

        out = X.copy()
        keys = pd.MultiIndex.from_frame(X[COHORT_KEYS])

        for column in self.fitted_columns_:
            global_mean, global_std = self.global_stats_[column]
            if column in self.cohort_stats_.columns.get_level_values(0):
                means = self.cohort_stats_[(column, "mean")].reindex(keys).to_numpy()
                stds = self.cohort_stats_[(column, "std")].reindex(keys).to_numpy()
            else:  # pragma: no cover - defensive
                means = np.full(len(X), np.nan)
                stds = np.full(len(X), np.nan)

            # Unknown or dropped cohort -> global statistics.
            means = np.where(np.isnan(means), global_mean, means)
            stds = np.where(np.isnan(stds), global_std, stds)
            # A zero or absent standard deviation means the cohort had no
            # variation; a z-score is undefined, so report 0 (at the mean)
            # rather than dividing by zero.
            stds = np.where((stds == 0) | np.isnan(stds), np.nan, stds)

            values = X[column].to_numpy(dtype="float64")
            with np.errstate(invalid="ignore", divide="ignore"):
                z = (values - means) / stds
            # NaN input stays NaN: `mean_score` is genuinely undefined before
            # any assessment is due, and imputing it here would hide that.
            z = np.where(np.isnan(values), np.nan, np.nan_to_num(z, nan=0.0))
            out[f"{column}_cohort_z"] = z
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray([f"{column}_cohort_z" for column in self.fitted_columns_])
