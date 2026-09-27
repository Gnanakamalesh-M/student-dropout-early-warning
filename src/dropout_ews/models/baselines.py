"""Baseline models: Logistic Regression and Random Forest.

Both use ``class_weight="balanced"`` rather than resampling. That is the Phase 5
position, and the reasoning is recorded rather than assumed: class weights
change the loss without inventing observations, so they cannot manufacture
student trajectories that never existed. SMOTE is evaluated properly in Phase 6
alongside threshold tuning, and ADR-0002's concern — that interpolating between
students with rolling-window temporal features produces physically impossible
histories — is tested there rather than asserted here.

Neither baseline is tuned. Tuning happens in Phase 6 for the primary model; a
tuned baseline compared against a tuned candidate is a fairer fight, but a
*deliberately untuned* baseline is the honest floor for "does the added
complexity buy anything".
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from dropout_ews.models.base import make_pipeline

RANDOM_STATE = 42


def logistic_regression_pipeline(feature_names: list[str] | None = None) -> Pipeline:
    """L2 logistic regression. The interpretable floor.

    ``lbfgs`` rather than ``saga``: on this feature set ``saga`` hit its
    iteration cap and emitted convergence warnings even at ``max_iter=3000``,
    which would make the baseline depend on where the optimiser happened to
    stop. ``lbfgs`` is a full-batch quasi-Newton method and converges cleanly
    here, which matters because an unconverged baseline is not a baseline.
    """
    return make_pipeline(
        LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        scale=True,
        feature_names=feature_names,
    )


def random_forest_pipeline(feature_names: list[str] | None = None) -> Pipeline:
    """Random forest with balanced subsampling.

    ``min_samples_leaf=20`` is a floor rather than a tuned value: at a 3%
    positive rate, leaves of one or two rows fit noise and destroy calibration.
    ``balanced_subsample`` reweights within each bootstrap sample, which suits
    a rare class better than a single global weighting.
    """
    return make_pipeline(
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale=False,
        feature_names=feature_names,
    )


def stratified_dummy_pipeline(feature_names: list[str] | None = None) -> Pipeline:
    """Predicts the base rate for everyone. The absolute floor.

    Included so the comparison table has a row whose PR-AUC is, by
    construction, the base rate. Any model failing to clear it is broken, and
    having that line present makes the scale of every other number legible.
    """
    return make_pipeline(
        DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        scale=False,
        feature_names=feature_names,
    )


BASELINES = {
    "dummy (base rate)": stratified_dummy_pipeline,
    "logistic regression": logistic_regression_pipeline,
    "random forest": random_forest_pipeline,
}
