"""The as-of property test — the primary leakage guarantee (ADR-0003).

The property:

    build(full_history,               as_of=t)
      ==
    build(history truncated to <= t,  as_of=t)

If any feature reads data from after the checkpoint, the two differ. This is
stronger than inspecting the SQL by eye, because it holds the *whole* builder
to the guarantee, including the pandas derivations, the pivot fills, and any
future feature somebody adds without reading ADR-0003.

Both a Hypothesis search over generated histories and hand-built adversarial
fixtures are used. Hypothesis finds the cases nobody thought of; the fixtures
pin the ones already known to be dangerous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dropout_ews.features.builder import FeatureBuilder

pytestmark = pytest.mark.ml

MODULE, PRESENTATION = "AAA", "2013J"
PRESENTATION_LENGTH = 269


def _population(checkpoints: list[int], student: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code_module": MODULE,
            "code_presentation": PRESENTATION,
            "id_student": student,
            "checkpoint_day": checkpoints,
            "module_presentation_length": PRESENTATION_LENGTH,
        }
    )


def _engagement(days_clicks: list[tuple[int, int]], student: int = 1) -> pd.DataFrame:
    if not days_clicks:
        return pd.DataFrame(
            {
                "code_module": pd.Series(dtype="object"),
                "code_presentation": pd.Series(dtype="object"),
                "id_student": pd.Series(dtype="int64"),
                "course_day": pd.Series(dtype="int64"),
                "clicks": pd.Series(dtype="int64"),
                "distinct_resources": pd.Series(dtype="int64"),
            }
        )
    return pd.DataFrame(
        {
            "code_module": MODULE,
            "code_presentation": PRESENTATION,
            "id_student": student,
            "course_day": [day for day, _ in days_clicks],
            "clicks": [clicks for _, clicks in days_clicks],
            "distinct_resources": [1 for _ in days_clicks],
        }
    )


def _assessments(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    """rows: (id_assessment, due_day, weight)."""
    return pd.DataFrame(
        {
            "code_module": MODULE,
            "code_presentation": PRESENTATION,
            "id_assessment": [r[0] for r in rows],
            "assessment_type": "TMA",
            "date": [float(r[1]) for r in rows],
            "weight": [r[2] for r in rows],
        }
    )


def _submissions(rows: list[tuple[int, int, float]], student: int = 1) -> pd.DataFrame:
    """rows: (id_assessment, date_submitted, score)."""
    return pd.DataFrame(
        {
            "id_assessment": [r[0] for r in rows],
            "id_student": student,
            "date_submitted": [r[1] for r in rows],
            "is_banked": 0,
            "score": [r[2] for r in rows],
        }
    )


def _static(student: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code_module": [MODULE],
            "code_presentation": [PRESENTATION],
            "id_student": [student],
            "num_of_prev_attempts": [0],
            "studied_credits": [60],
            "date_registration": [-30.0],
        }
    )


def _build(population, engagement, assessments, submissions):
    return (
        FeatureBuilder()
        .build(
            population,
            engagement=engagement,
            assessments=assessments,
            student_assessment=submissions,
            static=_static(),
        )
        .matrix()
    )


def assert_as_of_invariant(
    checkpoint: int,
    engagement: pd.DataFrame,
    assessments: pd.DataFrame,
    submissions: pd.DataFrame,
) -> None:
    """Assert truncating the history at the checkpoint changes nothing."""
    population = _population([checkpoint])

    full = _build(population, engagement, assessments, submissions)
    trimmed = _build(
        population,
        engagement[engagement["course_day"] <= checkpoint],
        # Assessment *definitions* are course metadata known up front, so they
        # are not truncated. Submissions are events and must be.
        assessments,
        submissions[submissions["date_submitted"] <= checkpoint],
    )
    pd.testing.assert_frame_equal(full, trimmed, check_exact=True)


# ---------------------------------------------------------------------------
# Hypothesis: search for a history that breaks the property
# ---------------------------------------------------------------------------

_day_clicks = st.tuples(
    st.integers(min_value=-20, max_value=260), st.integers(min_value=1, max_value=500)
)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    activity=st.lists(_day_clicks, min_size=0, max_size=40),
    checkpoint=st.sampled_from([30, 60, 90, 120, 150, 180]),
)
def test_engagement_features_are_as_of_invariant(activity, checkpoint) -> None:
    """No engagement feature may change when post-checkpoint activity is
    deleted, for any generated history."""
    # Collapse duplicate days: the real student-day table is unique per day.
    collapsed: dict[int, int] = {}
    for day, clicks in activity:
        collapsed[day] = collapsed.get(day, 0) + clicks
    engagement = _engagement(sorted(collapsed.items()))
    assert_as_of_invariant(checkpoint, engagement, _assessments([(1, 50, 100.0)]), _submissions([]))


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    submissions=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=4),
            st.integers(min_value=0, max_value=260),
            st.floats(min_value=0, max_value=100),
        ),
        min_size=0,
        max_size=8,
        unique_by=lambda row: row[0],
    ),
    checkpoint=st.sampled_from([30, 60, 90, 120, 150, 180]),
)
def test_assessment_features_are_as_of_invariant(submissions, checkpoint) -> None:
    """No assessment feature may change when post-checkpoint submissions are
    deleted."""
    assessments = _assessments([(i, 20 * i, 25.0) for i in range(1, 5)])
    assert_as_of_invariant(
        checkpoint,
        _engagement([(10, 5), (40, 9)]),
        assessments,
        _submissions(list(submissions)),
    )


# ---------------------------------------------------------------------------
# Hand-built adversarial fixtures
# ---------------------------------------------------------------------------


def test_huge_burst_immediately_after_the_checkpoint_is_invisible() -> None:
    """The obvious leak: enormous activity one day past the boundary. If it
    leaked, every level feature would change by orders of magnitude."""
    engagement = _engagement([(25, 10), (31, 100_000), (200, 50_000)])
    assert_as_of_invariant(30, engagement, _assessments([(1, 10, 100.0)]), _submissions([]))


def test_activity_exactly_on_the_checkpoint_is_included() -> None:
    """The boundary is inclusive: day t counts. This is the complement of the
    leakage test — a builder that excluded day t would pass the invariant while
    silently discarding a day of real signal."""
    population = _population([30])
    with_day_30 = _build(population, _engagement([(30, 77)]), _assessments([]), _submissions([]))
    without = _build(population, _engagement([(29, 77)]), _assessments([]), _submissions([]))
    assert with_day_30["clicks_7d"].iloc[0] == 77
    assert with_day_30["days_since_last_activity"].iloc[0] == 0
    assert without["days_since_last_activity"].iloc[0] == 1


def test_late_submission_after_the_checkpoint_does_not_count() -> None:
    """An assessment due before t but submitted after t must read as missed at
    t, not as submitted."""
    assessments = _assessments([(1, 20, 100.0)])
    submissions = _submissions([(1, 45, 88.0)])
    population = _population([30])

    result = _build(population, _engagement([]), assessments, submissions).iloc[0]
    assert result["assessments_due"] == 1
    assert result["assessments_submitted"] == 0
    assert result["assessments_missed"] == 1
    assert pd.isna(result["mean_score"])
    assert_as_of_invariant(30, _engagement([]), assessments, submissions)


def test_assessment_not_yet_due_is_not_counted_as_missed() -> None:
    """Counting a not-yet-due assessment as missed would invent a backlog."""
    result = _build(
        _population([30]), _engagement([]), _assessments([(1, 100, 100.0)]), _submissions([])
    ).iloc[0]
    assert result["assessments_due"] == 0
    assert result["assessments_missed"] == 0
    assert pd.isna(result["submission_rate"])


def test_never_active_student_yields_nan_recency_not_zero() -> None:
    """3,095 real rows have no activity ever. A zero here would be
    indistinguishable from "active today"."""
    result = _build(_population([90]), _engagement([]), _assessments([]), _submissions([])).iloc[0]
    assert pd.isna(result["days_since_last_activity"])
    assert result["ever_active"] == 0
    assert result["clicks_28d"] == 0
    assert result["inactive_weeks_streak"] == 8  # the full panel width


def test_every_checkpoint_of_one_student_is_independently_as_of_correct() -> None:
    """Scoring six checkpoints at once must not let a later checkpoint's window
    contaminate an earlier one."""
    engagement = _engagement([(day, day) for day in range(0, 200, 3)])
    assessments = _assessments([(i, 25 * i, 20.0) for i in range(1, 6)])
    submissions = _submissions([(i, 25 * i + 2, 60.0) for i in range(1, 6)])
    checkpoints = [30, 60, 90, 120, 150, 180]

    combined = _build(_population(checkpoints), engagement, assessments, submissions)
    for index, checkpoint in enumerate(checkpoints):
        alone = _build(_population([checkpoint]), engagement, assessments, submissions)
        pd.testing.assert_frame_equal(
            combined.iloc[[index]].reset_index(drop=True),
            alone.reset_index(drop=True),
            check_exact=True,
        )


def test_batch_and_single_row_paths_agree() -> None:
    """Train/serve skew guard (ADR-0004): scoring a student alone, as the API
    does, must equal that student's row from a batch build."""
    engagement = pd.concat(
        [_engagement([(10, 5), (40, 20), (70, 3)], student=s) for s in (1, 2, 3)],
        ignore_index=True,
    )
    static = pd.concat([_static(student=s) for s in (1, 2, 3)], ignore_index=True)
    assessments = _assessments([(1, 30, 100.0)])
    submissions = pd.concat(
        [_submissions([(1, 32, 70.0)], student=s) for s in (1, 2, 3)], ignore_index=True
    )

    batch_population = pd.concat(
        [_population([90], student=s) for s in (1, 2, 3)], ignore_index=True
    )
    builder = FeatureBuilder()
    batch = builder.build(
        batch_population,
        engagement=engagement,
        assessments=assessments,
        student_assessment=submissions,
        static=static,
    ).matrix()

    for index, student in enumerate((1, 2, 3)):
        single = builder.build(
            _population([90], student=student),
            engagement=engagement,
            assessments=assessments,
            student_assessment=submissions,
            static=static,
        ).matrix()
        pd.testing.assert_frame_equal(
            batch.iloc[[index]].reset_index(drop=True),
            single.reset_index(drop=True),
            check_exact=True,
        )


def test_output_row_order_matches_input_row_order() -> None:
    """Features are concatenated positionally onto the population, so a
    reordering bug would silently mislabel every row."""
    students = [7, 3, 11, 2]
    population = pd.concat([_population([60], student=s) for s in students], ignore_index=True)
    engagement = pd.concat(
        [_engagement([(57, s * 100)], student=s) for s in students], ignore_index=True
    )
    static = pd.concat([_static(student=s) for s in students], ignore_index=True)
    result = FeatureBuilder().build(
        population,
        engagement=engagement,
        assessments=_assessments([]),
        student_assessment=_submissions([]),
        static=static,
    )
    assert result.frame["id_student"].tolist() == students
    assert result.frame["clicks_7d"].tolist() == [s * 100 for s in students]


def test_forbidden_columns_never_appear_in_the_matrix() -> None:
    """ADR-0003 control #2. The population may carry the label and the event
    time; the feature matrix may not."""
    population = _population([60])
    population["label"] = 1
    population["date_unregistration"] = 75.0

    result = FeatureBuilder().build(
        population,
        engagement=_engagement([(50, 5)]),
        assessments=_assessments([]),
        student_assessment=_submissions([]),
        static=_static(),
    )
    for forbidden in ("label", "date_unregistration", "final_result"):
        assert forbidden not in result.feature_names
        assert forbidden not in result.matrix().columns


def test_missing_required_population_column_fails_loudly() -> None:
    bad = _population([30]).drop(columns=["module_presentation_length"])
    with pytest.raises(ValueError, match="missing required columns"):
        FeatureBuilder().build(
            bad,
            engagement=_engagement([]),
            assessments=_assessments([]),
            student_assessment=_submissions([]),
            static=_static(),
        )


# ---------------------------------------------------------------------------
# Trend feature correctness — hand-computed expectations
# ---------------------------------------------------------------------------


def test_declining_trajectory_produces_a_negative_slope() -> None:
    """The specification's motivating case, in engagement terms: activity
    falling week over week must yield a negative slope and a decline streak."""
    # Weeks back from day 90: week 0 = days 84-90, week 1 = 77-83, etc.
    engagement = _engagement([(87, 10), (80, 20), (73, 30), (66, 40)])
    result = _build(_population([90]), engagement, _assessments([]), _submissions([])).iloc[0]
    assert result["clicks_slope_4w"] < 0
    assert result["declining_weeks_streak"] == 3
    assert result["clicks_7d"] == 10
    assert result["clicks_28d"] == 100


def test_improving_trajectory_produces_a_positive_slope() -> None:
    engagement = _engagement([(87, 40), (80, 30), (73, 20), (66, 10)])
    result = _build(_population([90]), engagement, _assessments([]), _submissions([])).iloc[0]
    assert result["clicks_slope_4w"] > 0
    assert result["declining_weeks_streak"] == 0


def test_slope_is_exactly_zero_for_flat_engagement() -> None:
    engagement = _engagement([(87, 25), (80, 25), (73, 25), (66, 25)])
    result = _build(_population([90]), engagement, _assessments([]), _submissions([])).iloc[0]
    assert result["clicks_slope_4w"] == pytest.approx(0.0)
    assert result["clicks_weekly_std"] > 0  # weeks 4-7 are zero, so still varies


def test_inactive_weeks_streak_counts_only_the_most_recent_run() -> None:
    """Activity two weeks ago then silence gives a streak of 2, not 0."""
    engagement = _engagement([(75, 50)])  # day 75 is 15 days back -> week 2
    result = _build(_population([90]), engagement, _assessments([]), _submissions([])).iloc[0]
    assert result["inactive_weeks_streak"] == 2
    assert result["clicks_7d"] == 0
    assert result["days_since_last_activity"] == 15


def test_baseline_ratio_detects_decline_against_the_students_own_baseline() -> None:
    """Phase 3 identified baseline-relative decline as the trend form worth
    testing, since it separates a genuine faller from a consistently low
    performer."""
    # Strong first four weeks (days 0-27), weak recent four weeks.
    early = [(day, 100) for day in range(0, 28, 7)]
    faller = _build(
        _population([90]), _engagement([*early, (87, 5)]), _assessments([]), _submissions([])
    ).iloc[0]
    # Same weak recent activity, but no strong baseline.
    steady = _build(
        _population([90]), _engagement([(87, 5)]), _assessments([]), _submissions([])
    ).iloc[0]

    assert faller["clicks_vs_baseline_ratio"] < steady["clicks_vs_baseline_ratio"]
    assert faller["clicks_28d"] == steady["clicks_28d"]  # level features identical


def test_late_submission_is_flagged_when_both_events_precede_the_checkpoint() -> None:
    result = _build(
        _population([90]),
        _engagement([]),
        _assessments([(1, 20, 50.0), (2, 40, 50.0)]),
        _submissions([(1, 35, 55.0), (2, 41, 62.0)]),
    ).iloc[0]
    assert result["assessments_due"] == 2
    assert result["assessments_submitted"] == 2
    assert result["assessments_late"] == 2
    assert result["mean_score"] == pytest.approx(58.5)
    assert result["submission_rate"] == pytest.approx(1.0)


def test_failed_assessment_is_counted() -> None:
    result = _build(
        _population([90]),
        _engagement([]),
        _assessments([(1, 20, 50.0), (2, 40, 50.0)]),
        _submissions([(1, 19, 30.0), (2, 39, 75.0)]),
    ).iloc[0]
    assert result["assessments_failed"] == 1
    assert result["min_score"] == pytest.approx(30.0)
    assert result["assessments_late"] == 0


def test_features_contain_no_infinities() -> None:
    """Ratio features use +1 smoothing; an infinity would break tree splits and
    every downstream metric."""
    engagement = _engagement([(89, 5000)])
    matrix = _build(_population([90]), engagement, _assessments([]), _submissions([]))
    numeric = matrix.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
