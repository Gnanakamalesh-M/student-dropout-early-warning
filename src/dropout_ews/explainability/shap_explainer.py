"""SHAP attributions for the primary model.

What is actually being explained
--------------------------------

The registered artifact is a ``CalibratedClassifierCV`` wrapping the pipeline, so
its output is an isotonic mapping of the booster's score. TreeSHAP cannot see
through that wrapper, so **attributions explain the pre-calibration score, in
log-odds**, not the calibrated probability.

That is defensible rather than a compromise, and the reason matters: isotonic
regression is **monotone**, so any feature that pushes the raw score up also
pushes the calibrated probability up. The *ranking* the explanation describes is
exactly the ranking the displayed probability reflects. What is lost is the
ability to say "this feature added 4 percentage points" — so the narrative layer
never makes that claim and reports direction and relative magnitude instead.

Why XGBoost computes the values, not the ``shap`` package
---------------------------------------------------------

``shap.TreeExplainer`` (0.49) fails on XGBoost 3.x: its model loader parses
``base_score`` as a float, and XGBoost now serialises it as an array string
(``'[2.7229803E-2]'``), raising ``ValueError``. Rather than pin XGBoost back and
retrain, this module uses XGBoost's own ``pred_contribs=True``, which is the same
exact TreeSHAP algorithm implemented inside XGBoost. Additivity was verified to
2.4e-6 against the model margin, and the final column XGBoost returns is the bias
term, which corresponds to SHAP's expected value.

The background sample is loaded from the model artifact rather than regenerated,
because a global summary computed against a different sample is not comparable
with a previously published one (see :mod:`dropout_ews.models.registry`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from dropout_ews.models.base import imputer_feature_names

# Contributions smaller than this are treated as no contribution. TreeSHAP
# returns tiny non-zero values for features that played no real part, and
# presenting one of those as a "reason" would be noise dressed as explanation.
NEGLIGIBLE_CONTRIBUTION = 1e-4


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's contribution to one prediction."""

    feature: str
    value: float
    shap_value: float
    """Signed contribution in log-odds of the pre-calibration score. Positive
    means the feature pushed the estimate toward higher risk."""

    @property
    def direction(self) -> str:
        return "increased" if self.shap_value > 0 else "decreased"

    @property
    def is_negligible(self) -> bool:
        return abs(self.shap_value) < NEGLIGIBLE_CONTRIBUTION


@dataclass(frozen=True)
class Explanation:
    """A single prediction's attributions, ordered by absolute magnitude."""

    contributions: list[FeatureContribution]
    base_value: float
    raw_score: float
    """Base value plus all contributions, in log-odds. Not the displayed
    probability; see the module docstring."""

    def top(self, k: int = 6, drop_negligible: bool = True) -> list[FeatureContribution]:
        selected = [c for c in self.contributions if not (drop_negligible and c.is_negligible)]
        return selected[:k]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": c.feature,
                    "value": c.value,
                    "shap_value": c.shap_value,
                    "direction": c.direction,
                }
                for c in self.contributions
            ]
        )


def unwrap_pipeline(model: object) -> Pipeline:
    """Get the fitted pipeline out of a calibrated wrapper.

    Handles both the ``FrozenEstimator`` path (scikit-learn >= 1.6) and the older
    ``cv="prefit"`` path, so the explainer works against either artifact.
    """
    if isinstance(model, Pipeline):
        return model
    if isinstance(model, CalibratedClassifierCV):
        inner = model.calibrated_classifiers_[0].estimator
        inner = getattr(inner, "estimator", inner)  # FrozenEstimator unwrap
        if isinstance(inner, Pipeline):
            return inner
        raise TypeError(f"expected a Pipeline inside the calibrator, found {type(inner)}")
    raise TypeError(f"cannot unwrap a model of type {type(model)}")


def transform_to_model_matrix(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Run a frame through every pipeline step except the classifier.

    The result is what the booster actually sees, including the missingness
    indicators ``SimpleImputer`` appends. Explaining the pre-pipeline frame
    instead would attribute to columns the model never received.
    """
    matrix: object = frame
    for name, step in pipeline.steps[:-1]:
        if name == "resample":
            # Resamplers act only during fit; a no-op at explain time.
            continue
        matrix = step.transform(matrix)

    selector = pipeline.named_steps["select"]
    names = imputer_feature_names(selector.selected_, pipeline.named_steps["impute"])
    array = np.asarray(matrix)
    if array.shape[1] != len(names):
        raise ValueError(
            f"transformed matrix has {array.shape[1]} columns but {len(names)} names "
            "were derived; the imputer indicator layout has changed"
        )
    return pd.DataFrame(array, columns=names, index=frame.index)


class RiskExplainer:
    """Exact TreeSHAP explanations for the primary model."""

    def __init__(self, model: object, background: pd.DataFrame | None = None) -> None:
        self.pipeline = unwrap_pipeline(model)
        self.classifier = self.pipeline.named_steps["classifier"]
        self.booster = self.classifier.get_booster()
        self._background = background

    @property
    def feature_names(self) -> list[str]:
        selector = self.pipeline.named_steps["select"]
        return imputer_feature_names(selector.selected_, self.pipeline.named_steps["impute"])

    def shap_values(self, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Return the SHAP matrix and the base value.

        The base value is XGBoost's bias term, which is constant across rows; it
        is returned as a scalar after checking that, so a future change in
        XGBoost's output layout fails loudly rather than silently averaging.
        """
        matrix = transform_to_model_matrix(self.pipeline, frame)
        dmatrix = xgboost.DMatrix(matrix.to_numpy(), feature_names=list(matrix.columns))
        contributions = self.booster.predict(dmatrix, pred_contribs=True)

        values = np.asarray(contributions[:, :-1])
        bias = np.asarray(contributions[:, -1])
        if bias.std() > 1e-5:
            raise ValueError(
                "XGBoost bias term is not constant across rows; the pred_contribs "
                "output layout may have changed"
            )
        return values, float(bias[0])

    def explain(self, frame: pd.DataFrame) -> list[Explanation]:
        """Explain every row of ``frame``."""
        matrix = transform_to_model_matrix(self.pipeline, frame)
        values, base = self.shap_values(frame)
        names = list(matrix.columns)

        explanations = []
        for position in range(len(matrix)):
            row = values[position]
            contributions = [
                FeatureContribution(
                    feature=names[index],
                    value=float(matrix.iloc[position, index]),
                    shap_value=float(row[index]),
                )
                for index in np.argsort(-np.abs(row))
            ]
            explanations.append(
                Explanation(
                    contributions=contributions,
                    base_value=base,
                    raw_score=base + float(row.sum()),
                )
            )
        return explanations

    def explain_one(self, frame: pd.DataFrame) -> Explanation:
        """Explain a single-row frame."""
        if len(frame) != 1:
            raise ValueError(f"expected exactly one row, got {len(frame)}")
        return self.explain(frame)[0]

    def global_importance(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, descending.

        Mean |SHAP| rather than XGBoost's built-in ``feature_importances_``:
        gain-based importance is computed on training data and is biased toward
        features with many split points, whereas this measures actual influence
        on the predictions being made.
        """
        source = frame if frame is not None else self._background
        if source is None:
            raise ValueError("no frame supplied and no background sample available")
        values, _ = self.shap_values(source)
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": np.abs(values).mean(axis=0),
                    "mean_shap": values.mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def attribution_stability(
    explainer: RiskExplainer,
    frame: pd.DataFrame,
    k: int = 3,
    n_neighbours: int = 5,
) -> pd.DataFrame:
    """How often near-identical students receive the same top factors.

    This exists because of a specific risk: at a 2.7% positive rate with 37
    correlated features, per-student attributions can be small and unstable. If
    two students with almost identical records receive different "main reasons",
    presenting those reasons to a counsellor as *the* explanation is misleading
    even when the probability itself is well calibrated.

    For each row the nearest neighbours in the model's own feature space are
    found and the overlap of top-*k* factor sets is measured. Per-row results are
    returned rather than an average, so the tail is visible: a good mean with a
    bad tail still means some students get an unreliable explanation.
    """
    matrix = transform_to_model_matrix(explainer.pipeline, frame)
    values, _ = explainer.shap_values(frame)
    names = np.asarray(list(matrix.columns))

    # Standardise before measuring distance, so a high-variance feature such as
    # clicks_all_time does not dominate the neighbour search.
    standardised = matrix.to_numpy(dtype="float64")
    spread = standardised.std(axis=0)
    spread[spread == 0] = 1.0
    standardised = (standardised - standardised.mean(axis=0)) / spread

    top_sets = [frozenset(names[np.argsort(-np.abs(row))[:k]]) for row in values]

    rows = []
    for position in range(len(matrix)):
        distances = np.linalg.norm(standardised - standardised[position], axis=1)
        distances[position] = np.inf
        neighbours = np.argsort(distances)[:n_neighbours]
        overlaps = [len(top_sets[position] & top_sets[n]) / k for n in neighbours]
        rows.append(
            {
                "row": position,
                "mean_top_k_overlap": float(np.mean(overlaps)),
                "min_top_k_overlap": float(np.min(overlaps)),
                "mean_neighbour_distance": float(distances[neighbours].mean()),
            }
        )
    return pd.DataFrame(rows)
