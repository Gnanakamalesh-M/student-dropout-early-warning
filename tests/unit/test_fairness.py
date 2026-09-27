"""Tests for the subgroup fairness audit.

The audit's job is to *refuse to overclaim*, in two directions, so those two
refusals are what these tests are mostly about:

1. A gap whose confidence intervals overlap must not be reported as a disparity.
   This is the failure mode of most fairness reporting: rank the groups, take the
   largest difference, write it up. At a 3% positive rate almost any ranking
   produces a plausible-looking gap.
2. A group too small to assess must be named as such, not silently dropped. An
   audit that omits what it could not measure reads as though it measured
   everything.

Fixtures are constructed so the expected counts are checkable by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_ews.evaluation import fairness


def _frame(groups: dict[str, tuple[int, int, float]]) -> pd.DataFrame:
    """Build an audit input from ``{group: (positives, negatives, p_flag)}``.

    ``p_flag`` is the share of *positives* that score above threshold, so recall
    is exact. Negatives are given a score below threshold unless stated, keeping
    the false-positive rate at zero so recall arithmetic is unambiguous.
    """
    rows = []
    for value, (positives, negatives, recall) in groups.items():
        caught = round(positives * recall)
        for i in range(positives):
            rows.append({"gender": value, "label": 1, "probability": 0.9 if i < caught else 0.1})
        for _ in range(negatives):
            rows.append({"gender": value, "label": 0, "probability": 0.1})
    return pd.DataFrame(rows)


THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# group_metrics arithmetic
# ---------------------------------------------------------------------------


def test_group_metrics_counts_are_exact() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = fairness.group_metrics("gender", "F", y_true, y_prob, THRESHOLD)

    assert m.rows == 10
    assert m.positives == 4
    assert m.flagged == 3
    assert m.recall == pytest.approx(0.5)  # 2 of 4
    assert m.false_negative_rate == pytest.approx(0.5)
    assert m.precision == pytest.approx(2 / 3)
    assert m.false_positive_rate == pytest.approx(1 / 6)  # 1 of 6 negatives


def test_recall_and_false_negative_rate_are_complementary() -> None:
    """They are reported separately because they read differently to a human, but
    they must never disagree: a 26% recall *is* a 74% miss rate."""
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.3).astype(int)
    y_prob = rng.random(500)
    m = fairness.group_metrics("gender", "F", y_true, y_prob, THRESHOLD)
    assert m.recall + m.false_negative_rate == pytest.approx(1.0)


def test_threshold_boundary_is_inclusive() -> None:
    """A score exactly at the cutoff is flagged. The operating point is derived
    positionally from the alert budget, so the k-th score *is* the threshold; an
    exclusive comparison would drop that student and undershoot the budget."""
    m = fairness.group_metrics(
        "gender", "F", np.array([1, 1]), np.array([0.5, 0.4999]), THRESHOLD
    )
    assert m.flagged == 1


def test_metrics_are_nan_not_zero_when_a_group_has_no_positives() -> None:
    """A recall of 0.0 asserts the model missed everyone; NaN says the question
    is unanswerable. Rendering the first as the second is how an audit invents a
    disparity out of an empty cell."""
    m = fairness.group_metrics(
        "gender", "F", np.array([0, 0, 0]), np.array([0.9, 0.1, 0.1]), THRESHOLD
    )
    assert np.isnan(m.recall)
    assert np.isnan(m.false_negative_rate)
    assert m.false_positive_rate == pytest.approx(1 / 3)  # still answerable


def test_precision_is_nan_when_nothing_is_flagged() -> None:
    m = fairness.group_metrics(
        "gender", "F", np.array([1, 0]), np.array([0.1, 0.1]), THRESHOLD
    )
    assert np.isnan(m.precision)
    assert m.recall == pytest.approx(0.0)  # answerable: there was a positive to catch


def test_calibration_gap_signs_the_direction_of_the_error() -> None:
    """Negative means the model understates risk for the group, which is the
    direction that denies support. The real audit found exactly this for
    `disability Y` (-0.0133), so the sign convention has to be unambiguous."""
    y_true = np.array([1, 1, 0, 0])  # observed rate 0.5
    understated = fairness.group_metrics(
        "d", "Y", y_true, np.array([0.2, 0.2, 0.2, 0.2]), THRESHOLD
    )
    overstated = fairness.group_metrics(
        "d", "N", y_true, np.array([0.8, 0.8, 0.8, 0.8]), THRESHOLD
    )
    assert understated.calibration_gap == pytest.approx(-0.3)
    assert overstated.calibration_gap == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Sufficiency of the sample
# ---------------------------------------------------------------------------


def test_group_below_the_positive_floor_is_inconclusive() -> None:
    """`age_band 55<=` in the real audit: 390 rows but 13 positives. The rows
    look sufficient and the recall interval spans most of [0, 1]."""
    m = fairness.group_metrics(
        "age_band",
        "55<=",
        np.array([1] * 13 + [0] * 377),
        np.concatenate([np.full(13, 0.9), np.full(377, 0.1)]),
        THRESHOLD,
    )
    assert m.positives < fairness.MIN_POSITIVES_FOR_RECALL
    assert m.conclusive is False
    assert m.recall == pytest.approx(1.0)  # a perfect point estimate...
    assert m.recall_ci[0] < 0.8  # ...on an interval that supports no claim


def test_group_below_the_row_floor_is_inconclusive() -> None:
    m = fairness.group_metrics(
        "gender",
        "X",
        np.array([1] * 40 + [0] * 60),
        np.concatenate([np.full(40, 0.9), np.full(60, 0.1)]),
        THRESHOLD,
    )
    assert m.positives >= fairness.MIN_POSITIVES_FOR_RECALL
    assert m.rows < fairness.MIN_ROWS_FOR_GROUP
    assert m.conclusive is False


def test_inconclusive_groups_are_listed_with_a_reason() -> None:
    frame = _frame({"F": (13, 377, 1.0), "M": (60, 940, 0.5)})
    audit = fairness.audit(frame, THRESHOLD, attributes=("gender",))
    listed = fairness.inconclusive_groups(audit)
    assert list(listed["value"]) == ["F"]
    assert listed.iloc[0]["reason"] == "too few positives"


def test_inconclusive_list_distinguishes_too_few_rows() -> None:
    frame = _frame({"F": (40, 60, 0.5), "M": (60, 940, 0.5)})
    listed = fairness.inconclusive_groups(
        fairness.audit(frame, THRESHOLD, attributes=("gender",))
    )
    assert listed.iloc[0]["reason"] == "too few rows"


# ---------------------------------------------------------------------------
# The overlap check — the central guard
# ---------------------------------------------------------------------------


def test_a_gap_within_noise_is_not_reported_as_a_finding() -> None:
    """Two groups with identical true rates and different draws. The point
    estimates differ; the intervals overlap; the audit must say so rather than
    rank them and write up the difference."""
    frame = _frame({"F": (100, 3000, 0.26), "M": (100, 3000, 0.30)})
    gaps = fairness.disparities(
        fairness.audit(frame, THRESHOLD, attributes=("gender",))
    )
    assert len(gaps) == 1
    row = gaps.iloc[0]
    assert row["gap"] == pytest.approx(0.04, abs=0.01)
    assert row["intervals_overlap"]
    assert not row["distinguishable_from_noise"]


def test_a_real_gap_survives_the_overlap_check() -> None:
    """The guard must not be inert. A model catching 80% in one group and 10% in
    another, with enough positives to separate the intervals, is a finding."""
    frame = _frame({"F": (300, 3000, 0.10), "M": (300, 3000, 0.80)})
    gaps = fairness.disparities(
        fairness.audit(frame, THRESHOLD, attributes=("gender",))
    )
    row = gaps.iloc[0]
    assert row["worst_group"] == "F"  # highest false-negative rate
    assert row["gap"] == pytest.approx(0.70, abs=0.01)
    assert not row["intervals_overlap"]
    assert row["distinguishable_from_noise"]


def test_the_guard_tracks_interval_width_not_gap_size() -> None:
    """What the guard is sensitive to is the interval width, which depends on
    both the effect and the sample. Measured: a 0.72 gap clears the check even at
    32 positives per group, while a 0.30 gap at 30 positives does not. So this is
    not a blanket small-sample veto — a large disparity in a small group is still
    reported, which is the behaviour you want from a safety check."""
    large_effect = fairness.disparities(
        fairness.audit(
            _frame({"F": (32, 400, 0.10), "M": (32, 400, 0.80)}),
            THRESHOLD,
            attributes=("gender",),
        )
    ).iloc[0]
    assert large_effect["gap"] == pytest.approx(0.72, abs=0.01)
    assert large_effect["distinguishable_from_noise"]

    moderate_effect = fairness.disparities(
        fairness.audit(
            _frame({"F": (30, 400, 0.40), "M": (30, 400, 0.70)}),
            THRESHOLD,
            attributes=("gender",),
        )
    ).iloc[0]
    assert moderate_effect["gap"] == pytest.approx(0.30, abs=0.01)
    assert moderate_effect["intervals_overlap"]


def test_worst_group_is_the_one_with_the_highest_miss_rate() -> None:
    """Direction matters: "worst" must mean worst-served. For a false-negative
    rate that is the maximum, for recall it would be the minimum, and getting it
    backwards would name the best-served group as the harmed one."""
    audit = fairness.audit(
        _frame({"F": (300, 3000, 0.10), "M": (300, 3000, 0.80)}),
        THRESHOLD,
        attributes=("gender",),
    )
    by_fnr = fairness.disparities(audit, "false_negative_rate").iloc[0]
    by_recall = fairness.disparities(audit, "recall").iloc[0]
    assert by_fnr["worst_group"] == "F"
    # `recall` ranks by maximum too, so its "worst" label is the better-served
    # group. The report uses false_negative_rate for this reason.
    assert by_recall["worst_group"] == "M"


def test_disparities_skips_an_attribute_with_only_one_assessable_group() -> None:
    """A gap needs two groups. With one assessable group there is nothing to
    compare, and the attribute must be absent rather than reported as a zero gap."""
    frame = _frame({"F": (100, 2000, 0.3), "M": (13, 200, 0.3)})
    gaps = fairness.disparities(
        fairness.audit(frame, THRESHOLD, attributes=("gender",))
    )
    assert gaps.empty
    # Columns must survive the empty case, or a caller filtering the result
    # crashes on exactly the outcome a small cohort most often produces.
    assert "distinguishable_from_noise" in gaps.columns
    assert gaps[gaps["distinguishable_from_noise"].astype(bool)].empty


def test_inconclusive_groups_are_excluded_from_gap_ranking_by_default() -> None:
    """The 13-positive group has a 100% miss rate by construction. Included, it
    would top the ranking and produce the audit's headline finding out of 13
    students."""
    frame = _frame({"F": (13, 377, 0.0), "M": (100, 2000, 0.3), "O": (100, 2000, 0.35)})
    default = fairness.disparities(
        fairness.audit(frame, THRESHOLD, attributes=("gender",))
    )
    assert default.iloc[0]["worst_group"] != "F"

    permissive = fairness.disparities(
        fairness.audit(frame, THRESHOLD, attributes=("gender",)), conclusive_only=False
    )
    assert permissive.iloc[0]["worst_group"] == "F"


# ---------------------------------------------------------------------------
# audit() over a frame
# ---------------------------------------------------------------------------


def test_missing_attribute_values_get_their_own_group() -> None:
    """5,766 real rows have no recorded deprivation band. Dropping them would
    hide whichever way that population differs; NaN becomes `missing`."""
    frame = _frame({"F": (60, 940, 0.5), "M": (60, 940, 0.5)})
    frame.loc[frame.index[:300], "gender"] = np.nan
    audit = fairness.audit(frame, THRESHOLD, attributes=("gender",))
    assert "missing" in set(audit["value"])
    assert audit["rows"].sum() == len(frame)


def test_audit_covers_every_row_exactly_once_per_attribute() -> None:
    frame = _frame({"F": (60, 940, 0.5), "M": (60, 940, 0.5), "O": (60, 940, 0.5)})
    frame["disability"] = ["Y", "N"] * (len(frame) // 2)
    audit = fairness.audit(frame, THRESHOLD, attributes=("gender", "disability"))
    for attribute, group in audit.groupby("attribute"):
        assert group["rows"].sum() == len(frame), attribute
        assert group["positives"].sum() == int(frame["label"].sum()), attribute


def test_audit_ignores_an_attribute_absent_from_the_frame() -> None:
    """The audit script joins from an isolated source; a column that failed to
    join must be skipped, not crash the run or be reported as an empty group."""
    audit = fairness.audit(
        _frame({"F": (60, 940, 0.5), "M": (60, 940, 0.5)}),
        THRESHOLD,
        attributes=("gender", "region"),
    )
    assert set(audit["attribute"]) == {"gender"}


def test_protected_attributes_are_not_in_the_model_feature_allowlist() -> None:
    """The guarantee this module depends on: it is the only consumer of these
    columns. If one ever reaches the allowlist, the audit is measuring a model
    that was trained on the attribute it is auditing."""
    from dropout_ews.config.settings import load_feature_config

    allowlist = set(load_feature_config().features.all_features())
    for attribute in fairness.PROTECTED_ATTRIBUTES:
        assert attribute not in allowlist
        assert not any(attribute in feature for feature in allowlist)
