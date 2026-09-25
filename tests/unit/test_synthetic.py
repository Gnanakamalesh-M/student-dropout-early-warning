"""Tests for the synthetic cohort generator.

Two things need guarding here. First, **determinism**: a generator that is not
reproducible makes every result computed from it unverifiable. Second, the
**structural properties the cohort is supposed to have** — if the declining
students do not actually decline, the synthetic track is not exercising the
trend features it exists to exercise, and would give false confidence.

A small cohort is used throughout to keep the hazard calibration fast.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dropout_ews.data.synthetic import (
    PASS_MARK,
    SyntheticCohortGenerator,
    SyntheticConfig,
    build_synthetic_checkpoints,
)

SMALL = SyntheticConfig(n_students=400, n_semesters=6, seed=7)


@pytest.fixture(scope="module")
def cohort():
    return SyntheticCohortGenerator(SMALL).generate()


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
def test_invalid_dropout_rate_is_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="target_dropout_rate"):
        SyntheticConfig(target_dropout_rate=rate)


def test_trajectory_shares_cannot_exceed_one() -> None:
    with pytest.raises(ValueError, match="must not exceed 1"):
        SyntheticConfig(declining_share=0.7, improving_share=0.5)


def test_too_few_semesters_is_rejected() -> None:
    """Trend features need at least three points to have a slope."""
    with pytest.raises(ValueError, match="at least 3 semesters"):
        SyntheticConfig(n_semesters=2)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_the_cohort_exactly() -> None:
    first = SyntheticCohortGenerator(SMALL).generate()
    second = SyntheticCohortGenerator(SMALL).generate()
    pd.testing.assert_frame_equal(first.students, second.students)
    pd.testing.assert_frame_equal(first.semester_records, second.semester_records)
    pd.testing.assert_frame_equal(first.subject_attendance, second.subject_attendance)
    assert first.calibration == second.calibration


def test_different_seed_produces_a_different_cohort() -> None:
    other = SyntheticCohortGenerator(
        SyntheticConfig(n_students=400, n_semesters=6, seed=8)
    ).generate()
    baseline = SyntheticCohortGenerator(SMALL).generate()
    assert not baseline.students["dropout_semester"].equals(other.students["dropout_semester"])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_dropout_rate_is_calibrated_to_the_target() -> None:
    """The hazard intercept is bisected to hit the configured rate, so that
    changing another coefficient does not silently move the base rate."""
    config = SyntheticConfig(n_students=1_500, n_semesters=6, seed=11, target_dropout_rate=0.25)
    cohort = SyntheticCohortGenerator(config).generate()
    achieved = cohort.students["dropout_semester"].notna().mean()
    assert achieved == pytest.approx(0.25, abs=0.03)


def test_calibration_metadata_is_recorded() -> None:
    cohort = SyntheticCohortGenerator(SMALL).generate()
    for key in ("hazard_intercept", "target_dropout_rate", "achieved_dropout_rate", "seed"):
        assert key in cohort.calibration


# ---------------------------------------------------------------------------
# Structural properties the cohort must have
# ---------------------------------------------------------------------------


def test_declining_students_actually_decline(cohort) -> None:
    """The whole point of the drift term. Compared among students who reach the
    later semesters, so survivorship does not confound the comparison."""
    records = cohort.semester_records.merge(
        cohort.students[["student_id", "trajectory_shape"]], on="student_id"
    )
    last = int(records["semester"].max())
    survivors = records.loc[records["semester"] == last, "student_id"]
    panel = records[records["student_id"].isin(survivors)]

    means = panel.pivot_table(index="semester", columns="trajectory_shape", values="attendance_pct")
    declining_change = means["declining"].iloc[-1] - means["declining"].iloc[0]
    stable_change = means["stable"].iloc[-1] - means["stable"].iloc[0]

    assert declining_change < -4.0, "declining students should lose real attendance"
    assert declining_change < stable_change, "declining must fall faster than stable"


def test_dropout_risk_is_ordered_by_trajectory_shape(cohort) -> None:
    rates = (
        cohort.students.assign(dropped=cohort.students["dropout_semester"].notna())
        .groupby("trajectory_shape")["dropped"]
        .mean()
    )
    assert rates["declining"] > rates["stable"] > rates["improving"]


def test_backlogs_are_monotonically_non_decreasing_per_student(cohort) -> None:
    """Backlogs accumulate; they may never fall, since the generator has no
    re-sit mechanism. A drop would mean the accumulator was reset by mistake."""
    ordered = cohort.semester_records.sort_values(["student_id", "semester"])
    deltas = ordered.groupby("student_id")["backlogs_active"].diff().dropna()
    assert (deltas >= 0).all()


def test_backlogs_are_consistent_with_failed_subjects(cohort) -> None:
    ordered = cohort.semester_records.sort_values(["student_id", "semester"])
    cumulative = ordered.groupby("student_id")["subjects_failed"].cumsum()
    assert (cumulative.to_numpy() == ordered["backlogs_active"].to_numpy()).all()


def test_subject_marks_below_the_pass_mark_produce_failures(cohort) -> None:
    """Cross-checks the subject table against the aggregated semester table."""
    failures = (
        cohort.subject_attendance.assign(failed=cohort.subject_attendance["mark"] < PASS_MARK)
        .groupby(["student_id", "semester"])["failed"]
        .sum()
        .rename("expected")
    )
    merged = cohort.semester_records.merge(failures, on=["student_id", "semester"], how="left")
    assert (merged["subjects_failed"] == merged["expected"]).all()


# ---------------------------------------------------------------------------
# Value domains and missingness
# ---------------------------------------------------------------------------


def test_values_stay_in_their_natural_ranges(cohort) -> None:
    records = cohort.semester_records
    attendance = records["attendance_pct"].dropna()
    assert attendance.between(0, 100).all()
    assert records["gpa"].between(0, 10).all()
    assert (records["assignments_submitted"] <= records["assignments_set"]).all()
    assert (records["assignments_submitted"] >= 0).all()
    assert cohort.subject_attendance["mark"].between(0, 100).all()
    assert cohort.subject_attendance["attendance_pct"].between(0, 100).all()


def test_missingness_is_present_but_bounded(cohort) -> None:
    """Real institutional data has gaps; without them the imputation path is
    never exercised. But it must not swallow the dataset either."""
    records = cohort.semester_records
    attendance_missing = records["attendance_pct"].isna().mean()
    lms_missing = records["lms_sessions"].isna().mean()
    assert 0.0 < attendance_missing < 0.10
    assert 0.0 < lms_missing < 0.15


def test_every_row_is_labelled_synthetic(cohort) -> None:
    """Provenance is the load-bearing honesty mechanism of ADR-0002."""
    for frame in (cohort.students, cohort.semester_records, cohort.subject_attendance):
        assert (frame["data_source"] == "synthetic").all()


# ---------------------------------------------------------------------------
# Checkpoint construction
# ---------------------------------------------------------------------------


def test_no_semester_record_exists_after_a_student_leaves(cohort) -> None:
    records = cohort.semester_records.merge(
        cohort.students[["student_id", "dropout_semester"]], on="student_id"
    )
    left = records["dropout_semester"].notna()
    assert not (left & (records["semester"] > records["dropout_semester"])).any()


def test_checkpoints_label_the_leaving_semester_positive(cohort) -> None:
    checkpoints = build_synthetic_checkpoints(cohort, horizon_semesters=1)
    positives = checkpoints[checkpoints["label"] == 1]
    assert (positives["semester"] == positives["dropout_semester"]).all()


def test_checkpoints_drop_rows_whose_horizon_is_unobserved(cohort) -> None:
    """A row at the final semester has no next semester to observe, so it is
    dropped rather than assumed negative (docs/TASK_SPEC.md)."""
    last = int(cohort.semester_records["semester"].max())
    checkpoints = build_synthetic_checkpoints(cohort, horizon_semesters=1)
    assert checkpoints["semester"].max() == last - 1


def test_longer_horizon_drops_more_rows_and_finds_more_positives(cohort) -> None:
    one = build_synthetic_checkpoints(cohort, horizon_semesters=1)
    two = build_synthetic_checkpoints(cohort, horizon_semesters=2)
    assert two["semester"].max() < one["semester"].max()
    assert two["label"].mean() > one["label"].mean()


def test_horizon_must_be_at_least_one() -> None:
    cohort = SyntheticCohortGenerator(SMALL).generate()
    with pytest.raises(ValueError, match="at least 1"):
        build_synthetic_checkpoints(cohort, horizon_semesters=0)


def test_checkpoint_positive_rate_is_realistically_imbalanced(cohort) -> None:
    """The synthetic positive rate should sit in the same regime as OULAD's
    3.02%. If it were 30%, the pipeline would never exercise the imbalance
    handling that the real data demands."""
    checkpoints = build_synthetic_checkpoints(cohort)
    assert 0.005 < checkpoints["label"].mean() < 0.12


def test_trend_example_from_the_specification_exists(cohort) -> None:
    """The specification's motivating case is a student going roughly
    86% -> 78% -> 63%. At least one such trajectory must be generated, or the
    cohort does not represent the scenario the system targets."""
    wide = cohort.semester_records.pivot_table(
        index="student_id", columns="semester", values="attendance_pct"
    )
    first_three = wide[[1, 2, 3]].dropna()
    strictly_declining = (
        (first_three[1] > first_three[2])
        & (first_three[2] > first_three[3])
        & (first_three[1] - first_three[3] > 15.0)
    )
    assert strictly_declining.sum() > 0
