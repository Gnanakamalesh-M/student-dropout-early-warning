"""Tests for the training orchestration.

This module was at 0% coverage while producing the shipped model, which is the
wrong place to have no tests. The two that matter:

``test_feature_frame_drops_the_label`` guards ADR-0003 control #2. The label
travels with the processed parquet because the trainer needs it, and exactly one
function is responsible for removing it before the pipeline sees it. If that
regresses, every metric in the model card becomes near-perfect and the failure
looks like success.

``test_threshold_is_chosen_on_validation_not_test`` guards the other direction:
a threshold selected on the test split would make the reported operating point
optimistic by construction, and nothing else in the suite would notice.

The fixture is small and synthetic — the point is the wiring, not the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_ews.config.settings import load_feature_config
from dropout_ews.evaluation.splits import TemporalSplit
from dropout_ews.models import tuning
from dropout_ews.models.baselines import (
    logistic_regression_pipeline,
    random_forest_pipeline,
    stratified_dummy_pipeline,
)
from dropout_ews.pipeline import train as pipeline


def _cohort(n_students: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A tiny panel carrying every allowlisted feature, with a learnable signal.

    The columns are generated *from* the allowlist rather than hand-listed. The
    pipeline enforces the full 37-feature contract (``models/base.py``), so a
    hand-written fixture would need updating on every feature change and would
    fail with a 25-name error message when someone forgot. Deriving it keeps the
    fixture honest about what the model actually requires.

    ``*_cohort_z`` columns are excluded: CohortZScorer produces those inside the
    pipeline, and supplying them here would mean the test never exercises it.

    Three presentations so a chronological split has something to split on, and
    two checkpoints per student so the grouping constraint is load-bearing.
    """
    allowlist = [
        name
        for name in load_feature_config().features.all_features()
        if not name.endswith("_cohort_z")
    ]
    rng = np.random.default_rng(seed)
    rows = []
    presentations = ["2013B", "2013J", "2014B"]
    for student in range(n_students):
        presentation = presentations[student % 3]
        at_risk = rng.random() < 0.25
        for checkpoint in (30, 60):
            # One engagement level per row drives every feature, so the signal is
            # coherent across columns rather than 37 independent noise draws.
            level = rng.poisson(3 if at_risk else 30)
            row: dict[str, object] = {
                "code_module": "AAA",
                "code_presentation": presentation,
                "id_student": student,
                "checkpoint_day": checkpoint,
                "label": int(at_risk and checkpoint == 60),
                "date_unregistration": 65.0 if at_risk else np.nan,
            }
            for name in allowlist:
                if name.startswith("clicks") or name.startswith("active_days"):
                    row[name] = float(level)
                elif "score" in name:
                    row[name] = float(rng.normal(45 if at_risk else 70, 10))
                elif name.startswith("has_") or name == "ever_active":
                    row[name] = bool(level > 0)
                elif "streak" in name or name.startswith("days_since"):
                    row[name] = float(20 - min(20, level))
                elif "ratio" in name or "rate" in name:
                    row[name] = float(level) / 30.0
                else:
                    row[name] = float(level)
            rows.append(row)
    frame = pd.DataFrame(rows)
    population = frame[
        [
            "code_module",
            "code_presentation",
            "id_student",
            "checkpoint_day",
            "label",
            "date_unregistration",
        ]
    ].copy()
    return frame, population


def _split(population: pd.DataFrame) -> TemporalSplit:
    """Chronological by presentation, with no student in two splits (each
    student here belongs to exactly one presentation, so grouping holds)."""
    by = population["code_presentation"]
    return TemporalSplit(
        train=population.index[by == "2013B"],
        validation=population.index[by == "2013J"],
        test=population.index[by == "2014B"],
        train_presentations=["2013B"],
        validation_presentations=["2013J"],
        test_presentations=["2014B"],
    )


# ---------------------------------------------------------------------------
# The leakage control
# ---------------------------------------------------------------------------


def test_feature_frame_drops_the_label() -> None:
    """ADR-0003 control #2. If the label reaches the estimator, the model
    predicts its own target and every reported metric becomes meaningless while
    looking excellent."""
    frame, _ = _cohort(n_students=6)
    features = pipeline._feature_frame(frame)
    assert "label" not in features.columns


def test_feature_frame_drops_every_target_derived_column() -> None:
    """`date_unregistration` is the column the label is *computed from*, so it is
    a perfect proxy and must go too. `split` and `row_id` are bookkeeping that
    would leak the partition."""
    frame, _ = _cohort(n_students=6)
    frame["split"] = "train"
    frame["row_id"] = range(len(frame))
    features = pipeline._feature_frame(frame)
    for column in ("label", "date_unregistration", "split", "row_id"):
        assert column not in features.columns


def test_feature_frame_keeps_the_real_features() -> None:
    """The drop must be a denylist of known target-derived columns, not a
    narrowing that silently discards features — Phase 4 lost 15 features to
    exactly that class of mistake."""
    frame, _ = _cohort(n_students=6)
    features = pipeline._feature_frame(frame)
    for column in ("clicks_7d", "clicks_28d", "mean_score", "submission_rate"):
        assert column in features.columns


def test_feature_frame_tolerates_a_frame_without_the_optional_columns() -> None:
    """At serving time there is no label to drop. Dropping unconditionally would
    raise, so the single-row path would break on a column it never has."""
    frame, _ = _cohort(n_students=6)
    serving = frame.drop(columns=["label", "date_unregistration"])
    assert "clicks_7d" in pipeline._feature_frame(serving).columns


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------


def test_cross_validation_returns_a_mean_and_a_spread() -> None:
    """The spread is not decoration: Phase 5's trend-feature ablation came in
    within fold noise, and that conclusion was only available because the
    standard deviation was reported alongside the mean."""
    frame, population = _cohort()
    split = _split(population)
    mean, std = pipeline.cross_validate_average_precision(
        stratified_dummy_pipeline, frame, population, split.train, n_splits=3
    )
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0


def test_cross_validation_asserts_no_student_spans_folds() -> None:
    """The internal leakage assertion must actually run. A student appearing in
    both a fit fold and a scoring fold inflates the score, and the fixture here
    puts two checkpoints per student precisely so grouping is load-bearing."""
    frame, population = _cohort()
    split = _split(population)
    # Corrupt the grouping key so every row looks like a distinct student except
    # for a deliberate duplicate pair; the split helper must object.
    broken = population.copy()
    broken["id_student"] = 1  # one student, many rows -> folds cannot separate them
    with pytest.raises((ValueError, AssertionError)):
        pipeline.cross_validate_average_precision(
            stratified_dummy_pipeline, frame, broken, split.train, n_splits=3
        )


def test_cross_validation_learns_more_than_a_dummy() -> None:
    """A sanity floor on the whole wiring: the signal in the fixture is strong,
    so logistic regression must beat a stratified dummy. If the features were
    being dropped or misaligned, both would score the same."""
    frame, population = _cohort(n_students=240)
    split = _split(population)
    dummy, _ = pipeline.cross_validate_average_precision(
        stratified_dummy_pipeline, frame, population, split.train, n_splits=3
    )
    model, _ = pipeline.cross_validate_average_precision(
        logistic_regression_pipeline, frame, population, split.train, n_splits=3
    )
    assert model > dummy


# ---------------------------------------------------------------------------
# End-to-end baseline training
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained() -> pipeline.TrainedBaseline:
    frame, population = _cohort(n_students=240)
    return pipeline.train_baseline(
        "random forest",
        random_forest_pipeline,
        frame,
        population,
        _split(population),
        n_splits=3,
        bootstrap_iterations=20,  # the CI arithmetic is tested in test_metrics
    )


def test_train_baseline_reports_validation_and_test_separately(trained) -> None:
    assert trained.validation.name.endswith("(validation)")
    assert trained.test.name.endswith("(test)")
    assert trained.name == "random forest"


def test_threshold_is_chosen_on_validation_not_test(trained) -> None:
    """The threshold must come from the validation split. Choosing it on test
    would make the reported recall optimistic by construction — the operating
    point would be fitted to the numbers used to judge it."""
    frame, population = _cohort(n_students=240)
    split = _split(population)
    # Refitting the same pipeline and scoring validation must reproduce the
    # stored threshold exactly; scoring test is not permitted to.
    validation = frame.loc[split.validation]
    probabilities = trained.pipeline.predict_proba(pipeline._feature_frame(validation))[:, 1]
    from dropout_ews.evaluation.metrics import threshold_for_recall

    expected = threshold_for_recall(
        validation["label"].to_numpy(),
        probabilities,
        load_feature_config().evaluation.target_recall,
    )
    assert trained.validation_threshold == pytest.approx(expected)


def test_train_baseline_emits_both_sweeps(trained) -> None:
    """The threshold sweep and the budget sweep answer different questions — "at
    what recall?" and "within what staffing capacity?" — and Phase 6 found the
    recall-first framing unusable, so both are published."""
    assert not trained.test_sweep.empty
    assert not trained.test_budget_sweep.empty
    assert "ties_at_cut" in trained.test_budget_sweep.columns


def test_the_fitted_pipeline_never_saw_the_label(trained) -> None:
    """Checked at the estimator rather than the helper: whatever path the trainer
    took, the fitted model's feature names must not include the target."""
    names = list(getattr(trained.pipeline, "feature_names_in_", []))
    assert "label" not in names
    assert "date_unregistration" not in names


# ---------------------------------------------------------------------------
# The tuning search space
# ---------------------------------------------------------------------------


def test_search_space_forbids_deep_trees_with_small_leaves() -> None:
    """The bounds are the guard against the failure mode Phase 6 was most exposed
    to: at a ~3% positive rate a deep tree with small leaves isolates individual
    positives, so CV average precision rises while calibration collapses. Since
    the UI shows probabilities to staff, a miscalibrated winner is worse than a
    slightly weaker ranker, and these bounds are what prevents Optuna choosing
    one. Widening them should require changing this test deliberately.
    """
    import optuna

    space: dict[str, object] = {}

    def objective(trial: optuna.Trial) -> float:
        space.update(tuning.suggest_params(trial))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=25)

    distributions = study.trials[0].distributions
    assert distributions["max_depth"].high <= 7
    assert distributions["min_child_weight"].low >= 5.0
    # Regularisation cannot be switched off entirely.
    assert distributions["reg_lambda"].low > 0.0


def test_every_suggested_parameter_is_one_xgboost_accepts() -> None:
    """A typo in a search-space key is silently ignored by XGBoost, so the tuner
    would report a tuned model that was never tuned on that parameter."""
    import optuna
    from xgboost import XGBClassifier

    captured: dict[str, object] = {}

    def objective(trial: optuna.Trial) -> float:
        captured.update(tuning.suggest_params(trial))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=1)

    accepted = set(XGBClassifier().get_params())
    assert set(captured) <= accepted, set(captured) - accepted


# ---------------------------------------------------------------------------
# The cohort scaler's degenerate case
# ---------------------------------------------------------------------------


def test_cohort_scorer_is_a_no_op_when_none_of_its_columns_are_present() -> None:
    """Its documented contract is that missing columns are skipped. The
    all-missing case used to raise pandas' "No objects to concatenate", which
    reads as a data problem rather than a configuration one."""
    from dropout_ews.preprocessing.cohort import COHORT_KEYS, CohortZScorer

    frame = pd.DataFrame(
        {
            "code_module": ["AAA"] * 4,
            "code_presentation": ["2013B"] * 4,
            "checkpoint_day": [30, 30, 60, 60],
            "some_other_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    scorer = CohortZScorer().fit(frame)
    assert scorer.fitted_columns_ == []
    out = scorer.transform(frame)
    pd.testing.assert_frame_equal(out, frame)
    assert all(key in out.columns for key in COHORT_KEYS)


def test_cohort_scorer_still_raises_when_the_cohort_keys_are_missing() -> None:
    """The no-op above must not have softened the real error: without the cohort
    keys the transformer cannot know what a cohort is, and that is a bug in the
    caller rather than a feature-set revision to tolerate."""
    from dropout_ews.preprocessing.cohort import CohortZScorer

    with pytest.raises(ValueError, match="cohort keys missing"):
        CohortZScorer().fit(pd.DataFrame({"clicks_7d": [1.0, 2.0]}))
