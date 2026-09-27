"""Tests for the EDA computations.

The important one is :func:`test_engagement_windows_respect_the_as_of_boundary`.
EDA findings drive the Phase 4 feature allowlist, so a leaking exploratory
statistic would suggest signal the model can never use — and would do so
persuasively, because the numbers would look good.

Fixtures are tiny and hand-built, with expected values computed by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_ews.eda import analysis as eda
from dropout_ews.evaluation.metrics import wilson_interval


@pytest.fixture
def engagement_parquet(tmp_path):
    """One student with activity on known days, written as Parquet."""

    def _write(rows: list[dict[str, object]]):
        path = tmp_path / "engagement.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return str(path)

    return _write


def _day(student: int, day: int, clicks: int, resources: int = 1) -> dict[str, object]:
    return {
        "code_module": "AAA",
        "code_presentation": "2013J",
        "id_student": student,
        "course_day": day,
        "clicks": clicks,
        "distinct_resources": resources,
    }


def _population(student: int, checkpoint: int, label: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": student,
                "checkpoint_day": checkpoint,
                "label": label,
                "date_unregistration": np.nan,
            }
        ]
    )


# ---------------------------------------------------------------------------
# The as-of boundary
# ---------------------------------------------------------------------------


def test_engagement_windows_respect_the_as_of_boundary(engagement_parquet) -> None:
    """Activity after the checkpoint must not contribute to any aggregate.

    Day 31 and day 100 clicks are large; if they leaked in, clicks_total would
    be 1160 rather than 60.
    """
    path = engagement_parquet(
        [
            _day(1, 20, 40),  # inside the 30d window, outside the 7d one
            _day(1, 30, 20),  # exactly at the checkpoint — included
            _day(1, 31, 100),  # after — excluded
            _day(1, 100, 1000),  # well after — excluded
        ]
    )
    result = eda.engagement_windows(_population(1, 30), path).iloc[0]
    assert result["clicks_total"] == 60
    assert result["clicks_7d"] == 20  # only day 30 falls in (23, 30]
    assert result["clicks_30d"] == 60  # days 20 and 30 both fall in (0, 30]
    assert result["last_active_day"] == 30


def test_trailing_windows_are_half_open_on_the_lower_bound(engagement_parquet) -> None:
    """A 7-day window at checkpoint 30 covers (23, 30], so day 23 is out."""
    path = engagement_parquet([_day(1, 23, 50), _day(1, 24, 7)])
    result = eda.engagement_windows(_population(1, 30), path).iloc[0]
    assert result["clicks_7d"] == 7
    assert result["clicks_30d"] == 57


def test_previous_window_does_not_overlap_the_recent_one(engagement_parquet) -> None:
    """clicks_prev_30d covers (t-60, t-30] and must be disjoint from
    clicks_30d, or the delta between them is meaningless."""
    path = engagement_parquet([_day(1, 50, 11), _day(1, 80, 22)])
    result = eda.engagement_windows(_population(1, 90), path).iloc[0]
    assert result["clicks_prev_30d"] == 11  # day 50 in (30, 60]
    assert result["clicks_30d"] == 22  # day 80 in (60, 90]
    assert result["clicks_delta_30d"] == 11


# ---------------------------------------------------------------------------
# The never-active population
# ---------------------------------------------------------------------------


def test_never_active_student_is_flagged_not_silently_zeroed(engagement_parquet) -> None:
    """3,095 real checkpoint rows have no activity ever. Days-since-last must
    be NaN rather than a sentinel, so downstream code cannot mistake "never
    engaged" for "engaged today"."""
    path = engagement_parquet([_day(2, 10, 5)])
    result = eda.engagement_windows(_population(1, 30), path).iloc[0]
    assert result["ever_active"] is np.False_ or result["ever_active"] == False  # noqa: E712
    assert pd.isna(result["days_since_last_activity"])
    assert result["clicks_total"] == 0


def test_days_since_last_activity_is_measured_from_the_checkpoint(
    engagement_parquet,
) -> None:
    path = engagement_parquet([_day(1, 18, 3)])
    result = eda.engagement_windows(_population(1, 30), path).iloc[0]
    assert result["days_since_last_activity"] == 12


def test_clicks_ratio_is_finite_when_the_previous_window_is_empty(
    engagement_parquet,
) -> None:
    """Smoothing must prevent a division by zero; an infinity would silently
    break the screening statistics."""
    path = engagement_parquet([_day(1, 80, 9)])
    result = eda.engagement_windows(_population(1, 90), path).iloc[0]
    assert result["clicks_prev_30d"] == 0
    assert np.isfinite(result["clicks_ratio_30d"])
    assert result["clicks_ratio_30d"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Event-aligned analysis
# ---------------------------------------------------------------------------


def test_aligned_engagement_excludes_activity_at_or_after_the_event(
    engagement_parquet,
) -> None:
    """Alignment is on the event, so post-event activity must be excluded or
    the curve would be contaminated by exactly what it is measuring."""
    path = engagement_parquet(
        [
            _day(1, 20, 5),  # 8 days before event -> week 1
            _day(1, 27, 7),  # 1 day before  -> week 0
            _day(1, 28, 999),  # on the event day -> excluded
            _day(1, 40, 999),  # after -> excluded
        ]
    )
    population = _population(1, 30)
    population["date_unregistration"] = 28.0
    result = eda.aligned_engagement_before_event(population, path, weeks_before=4)
    assert result.loc[0, "median_clicks"] == 7
    assert result.loc[1, "median_clicks"] == 5
    assert result["median_clicks"].max() < 999


# ---------------------------------------------------------------------------
# Screening statistics
# ---------------------------------------------------------------------------


def _screening_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    label = (rng.random(n) < 0.2).astype(int)
    return pd.DataFrame(
        {
            "checkpoint_day": rng.choice([30, 60], n),
            "label": label,
            # Informative: positives have fewer clicks.
            "clicks_7d": rng.poisson(np.where(label == 1, 2, 20)),
            "clicks_14d": rng.poisson(10, n),
            "clicks_30d": rng.poisson(10, n),
            "clicks_prev_30d": rng.poisson(10, n),
            "clicks_total": rng.poisson(30, n),
            "clicks_delta_30d": rng.normal(0, 5, n),
            "clicks_ratio_30d": rng.normal(1, 0.3, n),
            "active_days_30d": rng.poisson(5, n),
            "active_days_total": rng.poisson(15, n),
            "max_resources_30d": rng.poisson(3, n),
            "days_since_last_activity": rng.normal(5, 2, n),
            "ever_active": True,
        }
    )


def test_discriminative_power_ranks_the_informative_feature_first() -> None:
    result = eda.discriminative_power(_screening_frame())
    assert result.iloc[0]["feature"] == "clicks_7d"
    # Fewer clicks for positives means AUC below 0.5; the ranking uses the
    # absolute lift so the direction does not hide the signal.
    assert result.iloc[0]["auc"] < 0.5
    assert result.iloc[0]["abs_auc_lift"] > 0.2


def test_discriminative_power_reports_coverage_for_partial_features() -> None:
    frame = _screening_frame()
    frame.loc[frame.index[:100], "days_since_last_activity"] = np.nan
    result = eda.discriminative_power(frame, ["clicks_7d", "days_since_last_activity"])
    coverage = dict(zip(result["feature"], result["coverage"], strict=True))
    assert coverage["clicks_7d"] == 1.0
    assert coverage["days_since_last_activity"] == pytest.approx(0.75)


def test_discriminative_power_skips_a_constant_label_subset() -> None:
    frame = _screening_frame()
    frame["label"] = 0
    assert eda.discriminative_power(frame).empty


def test_per_checkpoint_breakdown_covers_every_checkpoint() -> None:
    result = eda.discriminative_power_by_checkpoint(_screening_frame(), "clicks_7d")
    assert sorted(result["checkpoint_day"]) == [30, 60]
    assert (result["auc"] < 0.5).all()


# ---------------------------------------------------------------------------
# Trivial baselines and intervals
# ---------------------------------------------------------------------------


def test_trivial_baselines_compute_precision_and_recall() -> None:
    frame = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "clicks_30d": [0, 5, 0, 5],
            "clicks_14d": [0, 5, 5, 5],
            "clicks_7d": [0, 0, 5, 5],
            "clicks_ratio_30d": [0.1, 1.0, 1.0, 1.0],
            "ever_active": [False, True, True, True],
        }
    )
    result = eda.trivial_rule_baselines(frame).set_index("rule")
    row = result.loc["no clicks in last 30d"]
    # Flags rows 0 and 2; one is a true positive out of two positives.
    assert row["flagged"] == 2
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("successes", "n"),
    [(0, 100), (3, 100), (50, 100), (100, 100)],
)
def test_wilson_interval_stays_within_zero_and_one(successes: int, n: int) -> None:
    """The reason for using Wilson rather than the normal approximation: at a
    ~3% rate the normal interval can extend below zero."""
    low, high = wilson_interval(successes, n)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = wilson_interval(30, 1000)
    assert low < 0.03 < high


def test_wilson_interval_handles_an_empty_group() -> None:
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_wilson_interval_narrows_with_sample_size() -> None:
    small = wilson_interval(3, 100)
    large = wilson_interval(300, 10_000)
    assert (large[1] - large[0]) < (small[1] - small[0])
