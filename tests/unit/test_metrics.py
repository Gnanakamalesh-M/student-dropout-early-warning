"""Tests for the metrics harness.

Hand-computed expectations throughout. A metrics module that is subtly wrong is
worse than none: every downstream decision, threshold and model comparison
inherits the error, and nothing looks broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_ews.evaluation.metrics import (
    budget_sweep,
    cluster_bootstrap_intervals,
    metrics_at_threshold,
    per_checkpoint_metrics,
    ranking_metrics,
    threshold_for_recall,
    threshold_sweep,
    trivial_rule_scores,
)

# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------


def test_metrics_at_threshold_matches_hand_computation() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
    # At 0.5: flagged = 0.9, 0.8, 0.7, 0.6 -> 2 true, 2 false.
    result = metrics_at_threshold(y, p, 0.5)
    assert result.true_positives == 2
    assert result.false_positives == 2
    assert result.false_negatives == 1
    assert result.true_negatives == 5
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(2 / 3)
    assert result.flagged == 4
    assert result.flagged_share == pytest.approx(0.4)


def test_lift_is_precision_over_base_rate() -> None:
    """Lift is the number that tells a reader whether the model beats guessing."""
    y = np.array([1] + [0] * 9)  # base rate 0.1
    p = np.array([0.9] + [0.1] * 9)
    result = metrics_at_threshold(y, p, 0.5)
    assert result.precision == pytest.approx(1.0)
    assert result.lift_over_base_rate == pytest.approx(10.0)


def test_threshold_above_every_score_flags_nothing() -> None:
    y = np.array([1, 0, 0, 0])
    p = np.array([0.4, 0.3, 0.2, 0.1])
    result = metrics_at_threshold(y, p, 0.99)
    assert result.flagged == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


# ---------------------------------------------------------------------------
# Ranking metrics and the Brier guard
# ---------------------------------------------------------------------------


def test_perfect_ranking_scores_one() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    result = ranking_metrics(y, p)
    assert result.average_precision == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)


def test_random_ranking_average_precision_is_near_the_base_rate() -> None:
    """The reference point that makes every reported PR-AUC legible."""
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.03).astype(int)
    p = rng.random(20_000)
    result = ranking_metrics(y, p)
    assert result.average_precision == pytest.approx(result.base_rate, abs=0.005)


def test_brier_is_nan_for_non_probability_scores() -> None:
    """Brier measures calibration, which is meaningless for an arbitrary-scale
    ranker. The trivial rules include negated click counts, so reporting a Brier
    score for them would be a category error rather than a poor result."""
    y = np.array([1, 0, 1, 0])
    scores = np.array([-5569.0, -12.0, -3.0, -900.0])
    result = ranking_metrics(y, scores)
    assert np.isnan(result.brier_score)
    # Rank-based metrics stay valid at any scale.
    assert not np.isnan(result.average_precision)
    assert not np.isnan(result.roc_auc)


def test_brier_is_computed_for_probabilities() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    assert ranking_metrics(y, p).brier_score == pytest.approx(0.025, abs=1e-9)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def test_threshold_for_recall_achieves_at_least_the_target() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.2, 4000), 0, 1)
    for target in (0.5, 0.7, 0.8, 0.9):
        threshold = threshold_for_recall(y, p, target)
        assert metrics_at_threshold(y, p, threshold).recall >= target - 1e-9


def test_threshold_for_recall_picks_the_most_selective_qualifying_cut() -> None:
    """Among thresholds meeting the target, the highest gives the best
    precision, so the function must not return a needlessly permissive one."""
    rng = np.random.default_rng(2)
    y = (rng.random(4000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.2, 4000), 0, 1)
    chosen = threshold_for_recall(y, p, 0.8)
    at_chosen = metrics_at_threshold(y, p, chosen)
    higher = metrics_at_threshold(y, p, chosen + 0.05)
    assert at_chosen.recall >= 0.8
    assert higher.recall < 0.8  # any stricter cut misses the target


def test_unreachable_recall_target_falls_back_to_flagging_everything() -> None:
    """A degenerate but honest answer, rather than silently missing the target."""
    y = np.array([1, 1, 0, 0])
    p = np.array([0.1, 0.1, 0.1, 0.1])
    threshold = threshold_for_recall(y, p, 1.0)
    assert metrics_at_threshold(y, p, threshold).recall == pytest.approx(1.0)


@pytest.mark.parametrize("target", [0.0, -0.1, 1.5])
def test_invalid_recall_target_is_rejected(target: float) -> None:
    with pytest.raises(ValueError, match="target_recall"):
        threshold_for_recall(np.array([1, 0]), np.array([0.6, 0.4]), target)


def test_threshold_sweep_recall_rises_and_precision_falls() -> None:
    """The core tradeoff. If this ordering ever inverts, something is wrong."""
    rng = np.random.default_rng(3)
    y = (rng.random(6000) < 0.04).astype(int)
    p = np.clip(0.04 + 0.35 * y + rng.normal(0, 0.15, 6000), 0, 1)
    sweep = threshold_sweep(y, p)
    assert sweep["achieved_recall"].is_monotonic_increasing
    assert sweep["flagged"].is_monotonic_increasing
    assert sweep["precision"].iloc[0] > sweep["precision"].iloc[-1]


# ---------------------------------------------------------------------------
# Capacity view
# ---------------------------------------------------------------------------


def test_budget_sweep_respects_the_budget() -> None:
    rng = np.random.default_rng(4)
    y = (rng.random(10_000) < 0.03).astype(int)
    p = rng.random(10_000)
    sweep = budget_sweep(y, p)
    for _, row in sweep.iterrows():
        # Ties at the cut are included, so a small overshoot is expected.
        assert row["realised_share"] <= row["alert_budget"] + 0.01


def test_budget_sweep_recall_rises_with_budget() -> None:
    rng = np.random.default_rng(5)
    y = (rng.random(10_000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.2, 10_000), 0, 1)
    sweep = budget_sweep(y, p)
    assert sweep["recall"].is_monotonic_increasing
    assert sweep["precision"].iloc[0] >= sweep["precision"].iloc[-1]


def test_budget_sweep_on_a_perfect_ranker_catches_everything_it_can() -> None:
    """With 10 positives in 100 rows and a 10% budget, a perfect ranker should
    reach full recall at full precision."""
    y = np.array([1] * 10 + [0] * 90)
    p = np.concatenate([np.linspace(1.0, 0.9, 10), np.linspace(0.5, 0.0, 90)])
    row = budget_sweep(y, p, budgets=(0.10,)).iloc[0]
    assert row["recall"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)
    assert row["students_caught"] == 10


def test_budget_sweep_on_a_random_ranker_has_lift_near_one() -> None:
    rng = np.random.default_rng(6)
    y = (rng.random(40_000) < 0.03).astype(int)
    p = rng.random(40_000)
    row = budget_sweep(y, p, budgets=(0.20,)).iloc[0]
    assert row["lift"] == pytest.approx(1.0, abs=0.15)


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(7)
    n_students = 600
    groups = np.repeat(np.arange(n_students), 6)
    y = (rng.random(len(groups)) < 0.05).astype(int)
    p = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.2, len(groups)), 0, 1)

    point = ranking_metrics(y, p).average_precision
    low, high = cluster_bootstrap_intervals(y, p, groups, n_iterations=200)["average_precision"]
    assert low < point < high


def _interval_width(intervals: dict[str, tuple[float, float]], metric: str) -> float:
    low, high = intervals[metric]
    return high - low


def test_cluster_bootstrap_widens_when_labels_clump_within_a_student() -> None:
    """The mechanism, shown in the regime where it bites.

    When a whole cluster shares an outcome, drawing one student duplicates six
    identical observations, so there are genuinely 300 independent units rather
    than 1800. The clustered interval must then be clearly wider.

    Note this is *not* the regime of the real data — see the test below — but it
    is the case the clustered estimator exists to handle, so the machinery is
    verified here rather than assumed.
    """
    rng = np.random.default_rng(8)
    n_students, per_student = 300, 6
    groups = np.repeat(np.arange(n_students), per_student)
    # Outcome decided per student, then shared by all of that student's rows.
    student_label = (rng.random(n_students) < 0.08).astype(int)
    y = np.repeat(student_label, per_student)
    student_score = np.clip(0.05 + 0.5 * student_label + rng.normal(0, 0.12, n_students), 0, 1)
    p = np.repeat(student_score, per_student)

    clustered = cluster_bootstrap_intervals(y, p, groups, n_iterations=300)
    per_row = cluster_bootstrap_intervals(y, p, np.arange(len(groups)), n_iterations=300)
    assert _interval_width(clustered, "average_precision") > 1.5 * _interval_width(
        per_row, "average_precision"
    )


def test_cluster_bootstrap_design_effect_is_near_one_on_this_structure() -> None:
    """The regime this project is actually in, asserted so the documentation
    cannot drift from it.

    No row is emitted at or after a withdrawal, so at most one of a student's
    rows can be positive. Positives are spread across students rather than
    clumped, the design effect is close to 1, and the clustered interval comes
    out a little *narrower* than the naive one. Measured at 0.93x on the real
    test set.

    Clustering is still the default because the correct resampling unit is a
    property of the data, not of the width it produces.
    """
    rng = np.random.default_rng(11)
    n_students, per_student = 800, 6
    groups = np.repeat(np.arange(n_students), per_student)

    y = np.zeros(n_students * per_student, dtype=int)
    p = np.clip(rng.normal(0.05, 0.03, n_students * per_student), 0, 1)
    for student in range(n_students):
        if rng.random() < 0.18:  # this student withdraws at one checkpoint
            position = student * per_student + rng.integers(0, per_student)
            y[position] = 1
            p[position] = min(1.0, p[position] + 0.25)

    clustered = cluster_bootstrap_intervals(y, p, groups, n_iterations=400)
    per_row = cluster_bootstrap_intervals(y, p, np.arange(len(groups)), n_iterations=400)
    ratio = _interval_width(clustered, "average_precision") / _interval_width(
        per_row, "average_precision"
    )
    assert 0.75 < ratio < 1.25, f"expected a design effect near 1, got {ratio:.2f}"


def test_cluster_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(200), 4)
    y = (rng.random(len(groups)) < 0.06).astype(int)
    p = rng.random(len(groups))
    first = cluster_bootstrap_intervals(y, p, groups, n_iterations=100, random_state=3)
    second = cluster_bootstrap_intervals(y, p, groups, n_iterations=100, random_state=3)
    assert first == second


def test_cluster_bootstrap_returns_nan_when_too_few_usable_iterations() -> None:
    """With a single positive student, most resamples are single-class. Better a
    NaN than an interval computed from a handful of draws."""
    groups = np.repeat(np.arange(4), 2)
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    intervals = cluster_bootstrap_intervals(y, p, groups, n_iterations=10)
    assert np.isnan(intervals["average_precision"][0])


# ---------------------------------------------------------------------------
# Per-checkpoint breakdown
# ---------------------------------------------------------------------------


def test_per_checkpoint_covers_every_checkpoint_with_both_classes() -> None:
    rng = np.random.default_rng(10)
    n = 6000
    checkpoints = rng.choice([30, 60, 90], n)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.2, n), 0, 1)
    frame = per_checkpoint_metrics(y, p, checkpoints, 0.2)
    assert sorted(frame["checkpoint_day"]) == [30, 60, 90]
    assert frame["rows"].sum() == n


def test_per_checkpoint_skips_a_single_class_checkpoint() -> None:
    """A checkpoint with no positives has no defined AP; skipping beats
    emitting a misleading zero."""
    y = np.array([0, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.9, 0.3])
    checkpoints = np.array([30, 30, 60, 60])
    frame = per_checkpoint_metrics(y, p, checkpoints, 0.5)
    assert frame["checkpoint_day"].tolist() == [60]


# ---------------------------------------------------------------------------
# Trivial rules
# ---------------------------------------------------------------------------


def test_trivial_rule_scores_orient_higher_equals_riskier() -> None:
    """Every rule must be a score where larger means more at risk, or the
    comparison against the models is inverted and meaningless."""
    features = pd.DataFrame(
        {
            "clicks_7d": [0.0, 50.0],
            "clicks_14d": [0.0, 90.0],
            "clicks_28d": [0.0, 200.0],
            "days_since_last_activity": [30.0, 1.0],
            "checkpoint_day": [30, 30],
        }
    )
    for name, scores in trivial_rule_scores(features).items():
        assert scores[0] > scores[1], f"{name} is oriented the wrong way"


def test_trivial_rule_handles_never_active_students() -> None:
    """days_since_last_activity is NaN for never-active students; it must fall
    back to the checkpoint day rather than propagating NaN into a metric."""
    features = pd.DataFrame(
        {
            "clicks_7d": [0.0],
            "clicks_14d": [0.0],
            "clicks_28d": [0.0],
            "days_since_last_activity": [np.nan],
            "checkpoint_day": [90],
        }
    )
    scores = trivial_rule_scores(features)["rule: days since last activity"]
    assert scores[0] == pytest.approx(90.0)
