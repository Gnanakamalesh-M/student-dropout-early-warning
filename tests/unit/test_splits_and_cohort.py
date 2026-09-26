"""Tests for the split design and the cohort z-score transformer.

Both encode ADR-0003 controls. The split tests matter because group leakage is
invisible in results — it simply makes every score better — so the only defence
is asserting the property directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_ews.evaluation.splits import (
    assert_no_group_leakage_in_folds,
    grouped_cv_splits,
    make_temporal_split,
)
from dropout_ews.preprocessing.cohort import COHORT_KEYS, CohortZScorer

PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]


def _population(
    presentations: list[str] = PRESENTATIONS,
    students_per_presentation: int = 50,
    checkpoints: tuple[int, ...] = (30, 60, 90),
    seed: int = 0,
) -> pd.DataFrame:
    """Students are unique per presentation, as in a chronological cohort."""
    rng = np.random.default_rng(seed)
    rows = []
    student_id = 1
    for presentation in presentations:
        for _ in range(students_per_presentation):
            for checkpoint in checkpoints:
                rows.append(
                    {
                        "code_module": "AAA",
                        "code_presentation": presentation,
                        "id_student": student_id,
                        "checkpoint_day": checkpoint,
                        "label": int(rng.random() < 0.15),
                    }
                )
            student_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def test_split_is_chronological() -> None:
    """Training data must precede validation, which must precede test."""
    population = _population()
    split = make_temporal_split(population)
    assert split.train_presentations == ["2013B", "2013J"]
    assert split.validation_presentations == ["2014B"]
    assert split.test_presentations == ["2014J"]


def test_split_covers_every_row_exactly_once() -> None:
    """Splits plus dropped-overlap rows must partition the population, so no
    row silently disappears."""
    population = _population()
    split = make_temporal_split(population)
    combined = np.concatenate([split.train, split.validation, split.test, split.dropped_overlap])
    assert len(combined) == len(population)
    assert len(set(combined)) == len(population)


def test_no_student_appears_in_more_than_one_split() -> None:
    """The core group-leakage guarantee at the outer level."""
    population = _population()
    split = make_temporal_split(population)
    train = set(population.loc[split.train, "id_student"])
    validation = set(population.loc[split.validation, "id_student"])
    test = set(population.loc[split.test, "id_student"])
    assert not train & validation
    assert not train & test
    assert not validation & test


def _with_straddling_student() -> tuple[pd.DataFrame, int]:
    """Move one test-presentation row onto a student who is also in training.

    This is not hypothetical: 1,954 real students (7.9%) appear in more than one
    presentation, because they retake modules or study several.
    """
    population = _population()
    training_student = population.loc[
        population["code_presentation"] == "2013B", "id_student"
    ].iloc[0]
    mask = population["code_presentation"] == "2014J"
    population.loc[population.index[mask][0], "id_student"] = training_student
    return population, int(training_student)


def test_a_student_straddling_presentations_is_detected() -> None:
    """With resolution disabled the overlap must raise, never pass silently.

    This guards the guard: if detection were broken, the resolution below would
    have nothing to act on and the leak would return unnoticed.
    """
    population, _ = _with_straddling_student()
    with pytest.raises(ValueError, match="appear in both train and test"):
        make_temporal_split(population, resolve_overlap=False)


def test_straddling_student_is_dropped_from_the_later_split() -> None:
    """Resolution removes the row from test, not from train: contaminating the
    evaluation set is the error that inflates reported metrics."""
    population, student = _with_straddling_student()
    split = make_temporal_split(population)

    assert student in set(population.loc[split.train, "id_student"])
    assert student not in set(population.loc[split.test, "id_student"])
    assert len(split.dropped_overlap) == 1
    assert population.loc[split.dropped_overlap, "id_student"].tolist() == [student]


def test_dropped_overlap_is_reported_in_the_summary() -> None:
    """The exclusion must be auditable rather than invisible."""
    population, _ = _with_straddling_student()
    split = make_temporal_split(population)
    summary = split.summary(population).set_index("split")
    assert summary.loc["dropped_overlap", "rows"] == 1
    assert summary.loc["dropped_overlap", "students"] == 1


def test_no_rows_are_dropped_when_there_is_no_overlap() -> None:
    split = make_temporal_split(_population())
    assert len(split.dropped_overlap) == 0


def test_too_few_presentations_is_rejected() -> None:
    with pytest.raises(ValueError, match="need at least 3 presentations"):
        make_temporal_split(_population(presentations=["2013B", "2013J"]))


def test_split_summary_reports_counts_and_positive_rates() -> None:
    population = _population()
    split = make_temporal_split(population)
    summary = split.summary(population).set_index("split")
    assert summary.loc["train", "rows"] == 2 * 50 * 3
    assert summary.loc["test", "students"] == 50
    assert 0.0 <= summary.loc["test", "positive_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------


def test_cv_folds_never_split_a_student() -> None:
    """A student contributes three correlated rows here. Without grouping, a
    random fold would place some in train and some in validation, letting the
    model memorise the student."""
    population = _population()
    split = make_temporal_split(population)
    folds = grouped_cv_splits(population, split.train, n_splits=4)
    assert len(folds) == 4
    assert_no_group_leakage_in_folds(population, split.train, folds)


def test_leakage_checker_catches_a_deliberately_broken_fold() -> None:
    """Guard the guard: a checker that never fires is worthless."""
    population = _population()
    split = make_temporal_split(population)
    broken = [(np.array([0, 1, 2, 3]), np.array([0, 4, 5]))]  # position 0 in both
    with pytest.raises(ValueError, match="leaks"):
        assert_no_group_leakage_in_folds(population, split.train, broken)


def test_every_fold_contains_positives() -> None:
    """Stratification matters at a low positive rate: an unstratified fold can
    contain almost no positives, making fold scores unstable."""
    population = _population()
    split = make_temporal_split(population)
    labels = population.loc[split.train, "label"].to_numpy()
    for _, validation_positions in grouped_cv_splits(population, split.train, n_splits=4):
        assert labels[validation_positions].sum() > 0


def test_cv_rejects_a_single_class_split() -> None:
    population = _population()
    population["label"] = 0
    split = make_temporal_split(population)
    with pytest.raises(ValueError, match="single class"):
        grouped_cv_splits(population, split.train)


def test_cv_is_deterministic_for_a_fixed_seed() -> None:
    population = _population()
    split = make_temporal_split(population)
    first = grouped_cv_splits(population, split.train, random_state=7)
    second = grouped_cv_splits(population, split.train, random_state=7)
    for (a_train, a_test), (b_train, b_test) in zip(first, second, strict=True):
        assert np.array_equal(a_train, b_train)
        assert np.array_equal(a_test, b_test)


# ---------------------------------------------------------------------------
# Cohort z-scores
# ---------------------------------------------------------------------------


def _cohort_frame(seed: int = 0) -> pd.DataFrame:
    """Two modules with deliberately different engagement levels."""
    rng = np.random.default_rng(seed)
    rows = []
    for module, centre in (("AAA", 500.0), ("GGG", 50.0)):
        for _ in range(60):
            rows.append(
                {
                    "code_module": module,
                    "code_presentation": "2013J",
                    "checkpoint_day": 30,
                    "clicks_7d": rng.normal(centre, 20),
                    "clicks_28d": rng.normal(centre * 4, 50),
                    "active_days_28d": rng.normal(10, 2),
                    "mean_score": rng.normal(60, 10),
                    "submission_rate": rng.uniform(0, 1),
                }
            )
    return pd.DataFrame(rows)


def test_z_score_removes_the_between_module_level_difference() -> None:
    """The point of the transformer: a typical student in a low-engagement
    module should not look at-risk merely because the module is quieter."""
    frame = _cohort_frame()
    result = CohortZScorer().fit(frame).transform(frame)
    by_module = result.groupby("code_module")["clicks_7d_cohort_z"].mean()
    # Raw levels differ by an order of magnitude; z-scores centre near zero.
    assert abs(by_module["AAA"]) < 0.2
    assert abs(by_module["GGG"]) < 0.2
    raw = frame.groupby("code_module")["clicks_7d"].mean()
    assert raw["AAA"] > 5 * raw["GGG"]


def test_z_score_preserves_within_cohort_ordering() -> None:
    frame = _cohort_frame()
    result = CohortZScorer().fit(frame).transform(frame)
    subset = result[result["code_module"] == "AAA"]
    assert subset["clicks_7d"].corr(subset["clicks_7d_cohort_z"]) > 0.99


def test_statistics_come_only_from_the_fitted_rows() -> None:
    """Preprocessing leakage guard: fitting on train then transforming test
    must not shift the training statistics."""
    frame = _cohort_frame()
    train = frame.iloc[:60]
    test = frame.iloc[60:]

    scorer = CohortZScorer(min_cohort_size=10).fit(train)
    train_only = scorer.transform(train)
    after_test = scorer.transform(train)
    pd.testing.assert_frame_equal(train_only, after_test)

    # Test rows are a different module, unseen at fit time, so they fall back
    # to the global statistics rather than yielding NaN.
    transformed_test = scorer.transform(test)
    assert transformed_test["clicks_7d_cohort_z"].notna().all()


def test_unseen_cohort_falls_back_to_global_statistics() -> None:
    frame = _cohort_frame()
    scorer = CohortZScorer(min_cohort_size=10).fit(frame)
    unseen = frame.head(5).copy()
    unseen["code_presentation"] = "2099J"
    result = scorer.transform(unseen)
    assert result["clicks_7d_cohort_z"].notna().all()


def test_small_cohorts_are_excluded_from_fitted_statistics() -> None:
    """A standard deviation from a handful of rows is noise, and dividing by it
    manufactures extreme z-scores."""
    frame = _cohort_frame()
    tiny = frame.head(3).copy()
    tiny["code_module"] = "TINY"
    combined = pd.concat([frame, tiny], ignore_index=True)

    scorer = CohortZScorer(min_cohort_size=30).fit(combined)
    keys = {key[0] for key in scorer.cohort_stats_.index}
    assert "TINY" not in keys
    result = scorer.transform(combined)
    assert np.isfinite(result["clicks_7d_cohort_z"]).all()


def test_zero_variance_cohort_does_not_divide_by_zero() -> None:
    constant = pd.DataFrame(
        {
            "code_module": "AAA",
            "code_presentation": "2013J",
            "checkpoint_day": 30,
            "clicks_7d": [100.0] * 40,
            "clicks_28d": [400.0] * 40,
            "active_days_28d": [10.0] * 40,
            "mean_score": [60.0] * 40,
            "submission_rate": [1.0] * 40,
        }
    )
    result = CohortZScorer().fit(constant).transform(constant)
    assert (result["clicks_7d_cohort_z"] == 0).all()
    assert np.isfinite(result["clicks_7d_cohort_z"]).all()


def test_nan_input_stays_nan() -> None:
    """`mean_score` is genuinely undefined before any assessment is due;
    imputing a z-score there would hide a real state."""
    frame = _cohort_frame()
    frame.loc[frame.index[:10], "mean_score"] = np.nan
    result = CohortZScorer().fit(frame).transform(frame)
    assert result["mean_score_cohort_z"].isna().sum() == 10


def test_missing_cohort_key_fails_loudly() -> None:
    frame = _cohort_frame().drop(columns=["checkpoint_day"])
    with pytest.raises(ValueError, match="cohort keys missing"):
        CohortZScorer().fit(frame)


def test_transform_before_fit_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be fitted"):
        CohortZScorer().transform(_cohort_frame())


def test_absent_feature_columns_are_skipped() -> None:
    """Keeps the transformer usable across feature-set revisions."""
    frame = _cohort_frame().drop(columns=["mean_score", "submission_rate"])
    scorer = CohortZScorer().fit(frame)
    assert "mean_score" not in scorer.fitted_columns_
    result = scorer.transform(frame)
    assert "clicks_7d_cohort_z" in result.columns
    assert "mean_score_cohort_z" not in result.columns


def test_feature_names_out_matches_added_columns() -> None:
    frame = _cohort_frame()
    scorer = CohortZScorer().fit(frame)
    added = set(scorer.get_feature_names_out())
    result = scorer.transform(frame)
    assert added <= set(result.columns)
    assert added == {f"{column}_cohort_z" for column in scorer.fitted_columns_}


def test_cohort_keys_are_the_documented_three() -> None:
    """Checkpoint is part of the key on purpose: engagement decays over a
    presentation for everyone, so a whole-course z-score would mostly encode
    how far through the course a row sits."""
    assert COHORT_KEYS == ["code_module", "code_presentation", "checkpoint_day"]
