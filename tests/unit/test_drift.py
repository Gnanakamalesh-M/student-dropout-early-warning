"""Tests for drift detection.

The failure modes worth guarding are the ones where a monitor reports a number
that is not a measurement:

- PSI on a zero-inflated feature, where quantile edges collapse and the naive
  result is 0 or infinity depending on rounding. Nine features here are
  zero-inflated enough for that to matter, so it is marked degenerate instead.
- Drift pooled across checkpoints, which measures course progression. This is the
  big one: engagement decays for everyone over a presentation, so a pooled
  comparison manufactures drift that is not there.
- A feature that stopped being populated, which shows a *stable* PSI on the rows
  that remain. Coverage is tracked separately for exactly this.
- A checkpoint present in only one cohort, which would otherwise read as total
  drift at the start of every presentation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_ews.monitoring import drift

# ---------------------------------------------------------------------------
# PSI arithmetic
# ---------------------------------------------------------------------------


def test_psi_of_a_distribution_against_itself_is_zero() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5000)
    result = drift.population_stability_index(sample, sample)
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert not result.degenerate


def test_psi_grows_with_the_size_of_the_shift() -> None:
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 5000)
    small = drift.population_stability_index(reference, rng.normal(0.1, 1, 5000)).value
    large = drift.population_stability_index(reference, rng.normal(1.5, 1, 5000)).value
    assert 0 <= small < large


def test_psi_is_symmetric_enough_to_be_reported_either_way() -> None:
    """PSI is not formally symmetric, but the bins come from the reference, so a
    reader could reasonably expect swapping the arguments not to change the
    conclusion. It does change the number; this pins how much, so a future change
    to the binning cannot quietly make the direction matter."""
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 5000), rng.normal(0.5, 1, 5000)
    forward = drift.population_stability_index(a, b).value
    backward = drift.population_stability_index(b, a).value
    assert forward == pytest.approx(backward, rel=0.25)


def test_psi_counts_values_beyond_the_reference_range() -> None:
    """A cohort that shifted entirely outside the training range is the strongest
    drift signal available. Clipping into the end bins would understate it;
    dropping those rows would hide it. Open outer edges keep them counted."""
    reference = np.linspace(0, 10, 1000)
    far = np.linspace(100, 110, 1000)
    result = drift.population_stability_index(reference, far)
    assert result.value > drift.PSI_MAJOR
    assert result.current_n == 1000  # nothing discarded


def test_psi_bins_on_reference_quantiles_not_equal_width() -> None:
    """Clicks are heavy-tailed here. With equal-width bins a single 20,000-click
    student would put nearly every row in bin zero and PSI would stop responding
    to anything else."""
    rng = np.random.default_rng(3)
    reference = np.concatenate([rng.poisson(5, 999), [20_000]])
    current = np.concatenate([rng.poisson(20, 999), [20_000]])
    result = drift.population_stability_index(reference, current)
    assert result.value > drift.PSI_MINOR
    assert not result.degenerate


# ---------------------------------------------------------------------------
# Degenerate inputs, reported as such
# ---------------------------------------------------------------------------


def test_a_constant_reference_is_degenerate_not_zero_drift() -> None:
    """A PSI of 0.0 here would assert "no drift measured"; the truth is "drift not
    measurable". A monitor that reports the first will stay quiet forever on a
    feature that broke."""
    result = drift.population_stability_index(np.zeros(1000), np.ones(1000))
    assert result.degenerate
    assert np.isnan(result.value)


def test_a_heavily_zero_inflated_reference_is_degenerate() -> None:
    """`clicks_7d` is zero for a large share of rows, so its deciles collapse to a
    single repeated edge. Deduplicating leaves too few bins to measure with."""
    reference = np.concatenate([np.zeros(980), np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])])
    result = drift.population_stability_index(reference, reference)
    assert result.degenerate or result.n_bins < 10


def test_empty_samples_are_degenerate() -> None:
    assert drift.population_stability_index(np.array([]), np.ones(10)).degenerate
    assert drift.population_stability_index(np.ones(10), np.array([])).degenerate


def test_empty_bins_are_counted_so_a_floored_psi_is_visible() -> None:
    """PSI needs a floor for empty bins, and the floor drives the result when
    several bins need it. The count is reported so a reader can tell a measured
    PSI from an epsilon artefact."""
    reference = np.linspace(0, 100, 1000)
    current = np.linspace(0, 10, 1000)  # occupies only the bottom bins
    result = drift.population_stability_index(reference, current)
    assert result.empty_bins > 0


def test_non_finite_values_are_dropped_not_propagated() -> None:
    """One NaN would otherwise make every quantile edge NaN and the whole result
    silently NaN, which reads as "not applicable" rather than "bad input"."""
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 1000)
    current = np.concatenate([rng.normal(0, 1, 999), [np.nan, np.inf]])
    result = drift.population_stability_index(reference, current)
    assert np.isfinite(result.value)
    assert result.current_n == 999


# ---------------------------------------------------------------------------
# The checkpoint constraint
# ---------------------------------------------------------------------------


def _panel(checkpoints: dict[int, float], n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A panel where the feature mean is set per checkpoint, so the decay across a
    presentation is explicit and its effect on drift is measurable."""
    rng = np.random.default_rng(seed)
    frames = []
    for checkpoint, mean in checkpoints.items():
        frames.append(
            pd.DataFrame(
                {
                    "checkpoint_day": checkpoint,
                    "clicks_28d": rng.normal(mean, 5.0, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_pooling_checkpoints_manufactures_drift_that_is_not_there() -> None:
    """The reason `by_checkpoint` defaults to True, demonstrated.

    Both cohorts have *identical* per-checkpoint distributions. The only
    difference is how many rows sit at each checkpoint — which is what a
    mid-presentation cohort always looks like. Within checkpoint there is no
    drift; pooled, the composition shift alone produces a signal.
    """
    per_checkpoint = {30: 40.0, 90: 25.0, 150: 10.0}
    reference = _panel(per_checkpoint, n=600, seed=0)
    # Same distributions, different composition: weighted toward early checkpoints.
    current = pd.concat(
        [
            _panel({30: 40.0}, n=1000, seed=1),
            _panel({90: 25.0}, n=200, seed=2),
            _panel({150: 10.0}, n=50, seed=3),
        ],
        ignore_index=True,
    )

    within = drift.feature_drift(reference, current, ["clicks_28d"], by_checkpoint=True)
    pooled = drift.feature_drift(reference, current, ["clicks_28d"], by_checkpoint=False)

    assert (within["psi"] < drift.PSI_MINOR).all(), within.to_dict("records")
    assert pooled.iloc[0]["psi"] > within["psi"].max()


def test_within_checkpoint_drift_still_detects_a_real_shift() -> None:
    """The guard must not be inert: holding checkpoints fixed and shifting the
    feature must still fire."""
    reference = _panel({30: 40.0, 90: 25.0}, seed=0)
    current = _panel({30: 10.0, 90: 5.0}, seed=1)
    result = drift.feature_drift(reference, current, ["clicks_28d"])
    assert (result["psi"] > drift.PSI_MAJOR).all()
    assert set(result["band"]) == {"major"}


def test_a_checkpoint_missing_from_the_current_cohort_is_skipped() -> None:
    """Every presentation starts with only day-30 rows. Treating absent later
    checkpoints as total drift would fire every monitor every term."""
    reference = _panel({30: 40.0, 90: 25.0, 150: 10.0}, seed=0)
    current = _panel({30: 40.0}, seed=1)
    result = drift.feature_drift(reference, current, ["clicks_28d"])
    assert list(result["checkpoint_day"]) == [30]


def test_drift_is_reported_per_checkpoint_not_averaged() -> None:
    reference = _panel({30: 40.0, 90: 25.0}, seed=0)
    current = _panel({30: 5.0, 90: 25.0}, seed=1)  # day 30 shifted, day 90 stable
    result = drift.feature_drift(reference, current, ["clicks_28d"]).set_index("checkpoint_day")
    assert result.loc[30, "band"] == "major"
    assert result.loc[90, "band"] == "stable"


# ---------------------------------------------------------------------------
# Coverage: the failure PSI cannot see
# ---------------------------------------------------------------------------


def test_a_feature_that_stopped_being_populated_is_caught_by_coverage() -> None:
    """The realistic upstream break: a join changed and the feature is now null
    for most rows. PSI on the surviving rows is stable, because those rows really
    are unchanged — so PSI alone would say nothing is wrong."""
    reference = _panel({30: 40.0}, n=1000, seed=0)
    current = _panel({30: 40.0}, n=1000, seed=1)
    current.loc[current.index[:700], "clicks_28d"] = np.nan

    result = drift.feature_drift(reference, current, ["clicks_28d"])
    row = result.iloc[0]
    assert row["band"] == "stable"  # PSI sees nothing
    assert row["missing_rate_change"] == pytest.approx(0.7, abs=0.01)

    summary = drift.summarize(result)
    assert summary.coverage_drops == ["clicks_28d"]
    assert summary.requires_review  # ...but the monitor still escalates


def test_features_absent_from_either_frame_are_skipped() -> None:
    reference = _panel({30: 40.0})
    current = _panel({30: 40.0})
    result = drift.feature_drift(reference, current, ["clicks_28d", "not_a_feature"])
    assert list(result["feature"]) == ["clicks_28d"]


# ---------------------------------------------------------------------------
# Summarising into a review decision
# ---------------------------------------------------------------------------


def test_a_single_bad_checkpoint_escalates_the_feature() -> None:
    """Averaging across checkpoints would let a severe shift at day 30 — where an
    early-warning system is actually useful — be cancelled by stability at day
    180, where a warning arrives too late to act on."""
    reference = _panel({30: 40.0, 90: 25.0, 150: 10.0}, seed=0)
    current = _panel({30: 2.0, 90: 25.0, 150: 10.0}, seed=1)
    summary = drift.summarize(drift.feature_drift(reference, current, ["clicks_28d"]))
    assert summary.major == ["clicks_28d"]
    assert summary.requires_review


def test_minor_drift_alone_does_not_escalate() -> None:
    frame = pd.DataFrame(
        [
            {"feature": "a", "band": "minor", "missing_rate_change": 0.0},
            {"feature": "b", "band": "stable", "missing_rate_change": 0.0},
        ]
    )
    summary = drift.summarize(frame)
    assert summary.minor == ["a"]
    assert not summary.requires_review


def test_summary_escalates_to_review_not_to_retraining() -> None:
    """Deliberate: drift in an input distribution is not evidence the model got
    worse. The attribute is named `requires_review` and there is no
    `requires_retraining`, because nothing in this module can establish that
    retraining would help."""
    assert hasattr(drift.DriftSummary(features_checked=1), "requires_review")
    assert not hasattr(drift.DriftSummary(features_checked=1), "requires_retraining")


def test_summarize_handles_an_empty_frame() -> None:
    summary = drift.summarize(pd.DataFrame())
    assert summary.features_checked == 0
    assert not summary.requires_review


# ---------------------------------------------------------------------------
# Prediction drift — the label-free signal
# ---------------------------------------------------------------------------


def test_prediction_drift_reports_the_change_in_flagged_share() -> None:
    """The operationally meaningful number. The threshold was chosen for a
    staffing capacity, so a cohort that pushes the flagged share up has increased
    the workload the operating point promised, regardless of accuracy."""
    rng = np.random.default_rng(5)
    reference = rng.beta(2, 60, 5000)  # roughly a 3% base rate
    current = rng.beta(4, 60, 5000)  # riskier cohort
    threshold = float(np.quantile(reference, 0.95))

    result = drift.prediction_drift(reference, current, threshold)
    assert result.reference_alert_rate == pytest.approx(0.05, abs=0.01)
    assert result.current_alert_rate > result.reference_alert_rate
    assert result.alert_rate_change > 0


def test_prediction_drift_needs_no_labels() -> None:
    """Stated as a test because it is the reason this monitor matters: everything
    label-dependent lags the 30-day horizon, so this is the only signal that can
    fire on the cohort currently being scored."""
    import inspect

    parameters = inspect.signature(drift.prediction_drift).parameters
    assert not any("label" in name or name == "y_true" for name in parameters)


def test_identical_score_distributions_show_no_prediction_drift() -> None:
    rng = np.random.default_rng(6)
    scores = rng.beta(2, 60, 5000)
    result = drift.prediction_drift(scores, scores, 0.05)
    assert result.psi == pytest.approx(0.0, abs=1e-9)
    assert result.alert_rate_change == pytest.approx(0.0)
    assert result.band == "stable"


# ---------------------------------------------------------------------------
# Calibration drift — the one that needs labels
# ---------------------------------------------------------------------------


def test_calibration_drift_signs_understatement_as_negative() -> None:
    """Same convention as the fairness audit's `calibration_gap`, deliberately:
    negative means the model understates risk, which is the direction that
    withholds support. Two reports using opposite signs would be read against
    each other by mistake."""
    labels = np.array([1, 1, 0, 0])  # observed 0.5
    result = drift.calibration_drift(
        labels, np.full(4, 0.5), labels, np.full(4, 0.2), labels_complete=True
    )
    assert result.reference_gap == pytest.approx(0.0)
    assert result.current_gap == pytest.approx(-0.3)
    assert result.gap_change == pytest.approx(-0.3)


def test_incomplete_labels_are_flagged_on_the_result() -> None:
    """A calibration gap over unresolved rows counts every pending student as a
    negative, biasing the observed rate down and making the model look
    over-confident when it may not be. The caller cannot be trusted to remember
    this, so the result carries it."""
    labels = np.array([1, 0, 0, 0])
    result = drift.calibration_drift(
        labels, np.full(4, 0.25), labels, np.full(4, 0.25), labels_complete=False
    )
    assert result.labels_complete is False


def test_stable_calibration_shows_no_gap_change() -> None:
    rng = np.random.default_rng(7)
    labels = (rng.random(2000) < 0.03).astype(int)
    scores = np.full(2000, 0.03)
    result = drift.calibration_drift(labels, scores, labels, scores)
    assert result.gap_change == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Thresholds are configuration, not truth
# ---------------------------------------------------------------------------


def test_bands_are_configurable_because_the_defaults_are_convention() -> None:
    """0.10 and 0.25 come from credit-scoring practice and are not validated for
    this task. An institution setting a stricter band must be able to."""
    strict = drift.DriftThresholds(psi_minor=0.01, psi_major=0.02)
    assert strict.band(0.015) == "minor"
    assert strict.band(0.03) == "major"
    assert drift.DriftThresholds().band(0.03) == "stable"


def test_a_non_finite_psi_bands_as_unknown_not_stable() -> None:
    """NaN must never fall through to "stable". That is the difference between "we
    checked and it's fine" and "we could not check"."""
    assert drift.DriftThresholds().band(float("nan")) == "unknown"


# ---------------------------------------------------------------------------
# The small-sample noise floor
# ---------------------------------------------------------------------------


def test_psi_refuses_a_sample_too_small_to_measure() -> None:
    """At n=25 the measured median PSI between two samples of the *same*
    distribution was 1.31 — five times the conventional "major drift" band, from
    nothing at all. Reporting that number is worse than reporting none."""
    rng = np.random.default_rng(8)
    result = drift.population_stability_index(rng.normal(0, 1, 2000), rng.normal(0, 1, 25))
    assert result.degenerate
    assert np.isnan(result.value)
    assert "too small" in result.reason


def test_the_degenerate_reason_distinguishes_the_two_causes() -> None:
    """ "This feature is constant" and "this checkpoint has 30 rows" are different
    problems with different fixes, and a report that conflates them sends someone
    to look in the wrong place."""
    rng = np.random.default_rng(9)
    small = drift.population_stability_index(rng.normal(0, 1, 2000), rng.normal(0, 1, 30))
    constant = drift.population_stability_index(np.zeros(1000), np.ones(1000))
    assert "too small" in small.reason
    assert "constant" in constant.reason


def test_bin_count_adapts_to_the_smaller_sample() -> None:
    """Ten bins over 60 rows is six rows a bin, and the resulting PSI is mostly
    sampling noise. The bin count follows the data instead."""
    rng = np.random.default_rng(10)
    reference = rng.normal(0, 1, 5000)
    assert drift.population_stability_index(reference, rng.normal(0, 1, 60)).n_bins == 2
    assert drift.population_stability_index(reference, rng.normal(0, 1, 5000)).n_bins == 10


@pytest.mark.slow
def test_the_same_distribution_noise_floor_stays_below_the_minor_band() -> None:
    """The property the adaptive binning exists to provide, asserted directly
    rather than trusted from a docstring: two samples from the same distribution
    must not reach the "minor drift" band, at any sample size the monitor accepts.

    This is the test that would fail if someone restored a fixed bin count — which
    is easy to do while tidying, and would silently make the monitor fire every
    term on cohorts that had not changed.
    """
    rng = np.random.default_rng(11)
    for n in (50, 100, 200, 400, 1000):
        values = [
            drift.population_stability_index(rng.normal(0, 1, 2000), rng.normal(0, 1, n)).value
            for _ in range(120)
        ]
        p90 = float(np.quantile(values, 0.9))
        assert p90 < drift.PSI_MINOR, f"n={n}: 90th-percentile noise PSI {p90:.4f}"


@pytest.mark.slow
def test_adaptive_binning_still_detects_a_one_sd_shift_at_small_n() -> None:
    """The other half of the property: suppressing the noise floor must not cost
    sensitivity, or the guard has just switched the monitor off."""
    rng = np.random.default_rng(12)
    for n in (50, 100, 400):
        values = [
            drift.population_stability_index(rng.normal(0, 1, 2000), rng.normal(1.0, 1, n)).value
            for _ in range(60)
        ]
        assert float(np.median(values)) > drift.PSI_MAJOR, f"n={n}"


def test_pooling_checkpoints_can_also_mask_real_drift() -> None:
    """The failure mode I did not anticipate, and the dangerous one.

    Found by running the monitor on real cohorts:
    `days_since_last_submission` shifted +6.3 days at checkpoint 90 and -7.3 at
    checkpoint 150. Pooled, those cancel to PSI 0.19 ("minor"); within checkpoint
    every checkpoint was major. Pooling reported 1 major feature where the
    within-checkpoint comparison found 7.

    A monitor that averages a drifting cohort into looking fine is worse than one
    that cries wolf, so this direction is pinned as well.
    """
    reference = _panel({90: 25.0, 150: 32.0}, n=800, seed=0)
    # Opposite shifts of the same size: one checkpoint up, the other down.
    current = _panel({90: 31.0, 150: 25.0}, n=800, seed=1)

    within = drift.feature_drift(reference, current, ["clicks_28d"], by_checkpoint=True)
    pooled = drift.feature_drift(reference, current, ["clicks_28d"], by_checkpoint=False)

    assert set(within["band"]) == {"major"}, within.to_dict("records")
    # Pooled, the two shifts cancel and the feature reads far calmer than it is.
    assert pooled.iloc[0]["psi"] < within["psi"].min()
    assert drift.summarize(within).requires_review
    assert not drift.summarize(pooled).requires_review
