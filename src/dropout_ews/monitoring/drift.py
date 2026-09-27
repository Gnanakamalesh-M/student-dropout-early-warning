"""Drift detection for a deployed early-warning model.

Three things about this problem are specific to this system, and they shape every
function here.

**Drift must be measured within checkpoint.** Engagement decays across a
presentation for everyone, so a cohort scored at day 150 looks nothing like the
same cohort at day 30, and a pooled comparison is not interpretable in either
direction. It fails *both* ways, which I only established by running it:

- It can **manufacture** drift. Two cohorts with identical distributions at every
  checkpoint still differ pooled if their checkpoint composition differs — and a
  mid-presentation cohort always has a different composition. Demonstrated in
  ``test_pooling_checkpoints_manufactures_drift_that_is_not_there``.
- It can **mask** drift, which is worse, and is what actually happened on this
  data. ``days_since_last_submission`` shifted +6.3 days at checkpoint 90 and
  -7.3 days at checkpoint 150 — opposite directions. Pooled, the means are 23.6
  against 24.6 and PSI reads 0.19, a "minor" reading; within checkpoint all six
  checkpoints are major, PSI 0.71 to 1.15. Across the whole feature set, pooling
  reported 1 major feature where the within-checkpoint comparison found 7.

I expected the first effect and wrote the module around it. The real data showed
the second, and the second is the dangerous one: a monitor that averages a cohort
into looking fine is a monitor that will not fire when it should. Every function
here either takes a single checkpoint's rows or groups by checkpoint, and
:func:`feature_drift` pools only when asked explicitly. Phase 4 hit the same
structure when cohort z-scores had to be keyed on
``(module, presentation, checkpoint_day)``.

**Labels arrive 30 days late, so performance drift is always stale.** The target
is "withdraws within 30 days", so a row scored today cannot be scored *against*
for another month, and a presentation's final rows resolve only after it ends.
Anything that needs labels is therefore a lagging indicator by construction. That
is not a limitation of this module, it is the shape of the problem, and it is why
prediction drift and calibration drift are separated below: the first needs no
labels and can fire immediately, the second needs them and cannot.

**PSI thresholds are convention, not science, and PSI itself is biased upward on
small samples.** The 0.10 and 0.25 bands come from credit-scoring practice. They
are not validated for this task, this feature set, or this cohort size, and
nothing here establishes that a PSI of 0.26 means anything different from 0.24.

Worse, the bands are actively misleading at small n. Measured on this machine —
300 trials per row, two samples drawn from the *same* normal distribution, so the
true PSI is zero — with a fixed 10 bins against a 2,000-row reference:

    current n     median PSI    90th pct
           25         1.3115      2.5697
           50         0.1793      0.3677
          100         0.0910      0.1641
          200         0.0449      0.0777
          400         0.0256      0.0445
         5000         0.0059      0.0100

At n=50 the *median* noise reading already sits in the "minor drift" band and the
90th percentile clears "major". A monitor using the conventional thresholds on a
small checkpoint would therefore fire on nothing at all, every term. This is why
:func:`population_stability_index` adapts the bin count to the smaller sample
(roughly 40 rows per bin) and refuses outright below
:data:`MIN_SAMPLE_FOR_PSI`. Re-measured with adaptive bins, the same experiment
gives a 90th percentile of 0.059 at n=50 and 0.043 at n=400, while a genuine
one-SD shift still reads 0.56 at n=50 — so the false-positive floor drops by
roughly 6x at no measurable cost in sensitivity.

Treat a band as "look at this", never as a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# The conventional credit-scoring bands. See the module docstring: these are a
# reporting convenience, not a validated decision rule for this task.
PSI_MINOR = 0.10
PSI_MAJOR = 0.25

# PSI needs a floor for empty bins or the log term diverges. The value is
# arbitrary and it does affect the result, so it is named rather than inlined and
# is reported alongside any PSI computed with a bin that needed it.
EPSILON = 1e-6

# Rows per bin. Measured (see the module docstring): at roughly this density the
# same-distribution noise floor sits at a 90th percentile near 0.05, comfortably
# below the 0.10 "minor" band, while a one-SD shift still reads above 0.5.
ROWS_PER_BIN = 40

# Below this many rows in the smaller sample, PSI is not reported at all. At n=25
# the measured median noise reading on identical distributions was 1.31 — a number
# that would read as catastrophic drift and mean nothing.
MIN_SAMPLE_FOR_PSI = 50

CHECKPOINT_COLUMN = "checkpoint_day"


@dataclass(frozen=True)
class DriftThresholds:
    """Bands for reporting. Defaults are convention; override per institution."""

    psi_minor: float = PSI_MINOR
    psi_major: float = PSI_MAJOR

    def band(self, psi_value: float) -> str:
        if not np.isfinite(psi_value):
            return "unknown"
        if psi_value >= self.psi_major:
            return "major"
        if psi_value >= self.psi_minor:
            return "minor"
        return "stable"


@dataclass(frozen=True)
class PSIResult:
    """Population Stability Index between a reference and a current sample."""

    value: float
    n_bins: int
    reference_n: int
    current_n: int
    empty_bins: int
    """Bins where either sample had no rows and :data:`EPSILON` was substituted.
    A PSI resting on several floored bins is driven by the floor, not the data."""

    degenerate: bool
    """True when PSI is not reportable: a constant or near-constant reference, one
    so zero-inflated that quantile edges collapse, or a sample too small for the
    statistic to mean anything. All three return NaN rather than a number, because
    the alternative is a monitor that reports noise as a finding."""

    reason: str = ""
    """Why the result is degenerate, so a report can distinguish "this feature is
    constant" from "this checkpoint has 30 rows" — different problems with
    different fixes."""


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> PSIResult:
    """PSI between two samples, binned on reference quantiles.

    Quantile edges come from the *reference* so the bands mean "where the
    training data sat". Equal-width bins would be dominated by outliers: clicks
    are heavy-tailed here and one 20,000-click student would put almost every row
    in bin zero.

    Zero-inflation is handled explicitly rather than ignored. ``clicks_7d`` is
    zero for a large share of rows, so its deciles collapse to a single repeated
    edge; deduplicating the edges leaves fewer bins than requested, and where
    fewer than two survive the result is marked ``degenerate`` instead of
    returning a number that looks like a measurement.
    """
    reference = np.asarray(reference, dtype="float64")
    current = np.asarray(current, dtype="float64")
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return PSIResult(float("nan"), 0, len(reference), len(current), 0, True, "empty sample")

    smaller = min(len(reference), len(current))
    if smaller < MIN_SAMPLE_FOR_PSI:
        return PSIResult(
            float("nan"),
            0,
            len(reference),
            len(current),
            0,
            True,
            f"sample too small ({smaller} rows < {MIN_SAMPLE_FOR_PSI})",
        )

    # Adapt the bin count to the smaller sample. A fixed 10 bins against 50 rows
    # produces a median PSI of 0.18 on identical distributions; see the module
    # docstring for the measurement this number comes from.
    n_bins = max(2, min(n_bins, smaller // ROWS_PER_BIN))

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        # Fewer than two usable bins: the reference is constant or near-constant.
        # A PSI here would be 0 or infinity depending on rounding, and either
        # would read as a finding.
        return PSIResult(
            float("nan"),
            max(0, len(edges) - 1),
            len(reference),
            len(current),
            0,
            True,
            "reference constant or too zero-inflated to bin",
        )

    # Open the outer edges so current values outside the reference range are
    # counted rather than dropped. A cohort that shifted entirely beyond the
    # training range is the strongest drift signal there is, and clipping it into
    # the end bins would understate it, but discarding it would hide it.
    edges[0], edges[-1] = -np.inf, np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_share = reference_counts / reference_counts.sum()
    current_share = current_counts / current_counts.sum()

    empty = int(np.sum((reference_share == 0) | (current_share == 0)))
    reference_share = np.where(reference_share == 0, EPSILON, reference_share)
    current_share = np.where(current_share == 0, EPSILON, current_share)

    value = float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )
    return PSIResult(
        value=value,
        n_bins=len(edges) - 1,
        reference_n=len(reference),
        current_n=len(current),
        empty_bins=empty,
        degenerate=False,
        reason="",
    )


@dataclass(frozen=True)
class FeatureDrift:
    """Drift for one feature at one checkpoint."""

    feature: str
    checkpoint_day: int | None
    psi: float
    band: str
    ks_statistic: float
    ks_p_value: float
    reference_mean: float
    current_mean: float
    reference_n: int
    current_n: int
    empty_bins: int
    degenerate: bool
    degenerate_reason: str
    missing_rate_change: float
    """Change in the share of rows where the feature is null. A feature that
    silently stopped being populated shows a stable PSI on the rows that remain,
    so coverage is tracked separately — this is the failure an upstream pipeline
    change actually produces."""

    def to_row(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "checkpoint_day": self.checkpoint_day,
            "psi": round(self.psi, 5) if np.isfinite(self.psi) else None,
            "band": self.band,
            "ks_statistic": round(self.ks_statistic, 5) if np.isfinite(self.ks_statistic) else None,
            "ks_p_value": round(self.ks_p_value, 6) if np.isfinite(self.ks_p_value) else None,
            "reference_mean": round(self.reference_mean, 4),
            "current_mean": round(self.current_mean, 4),
            "reference_n": self.reference_n,
            "current_n": self.current_n,
            "empty_bins": self.empty_bins,
            "degenerate": self.degenerate,
            "degenerate_reason": self.degenerate_reason,
            "missing_rate_change": round(self.missing_rate_change, 4),
        }


def _feature_drift_one(
    feature: str,
    reference: pd.Series,
    current: pd.Series,
    checkpoint_day: int | None,
    thresholds: DriftThresholds,
    n_bins: int,
) -> FeatureDrift:
    reference_values = pd.to_numeric(reference, errors="coerce")
    current_values = pd.to_numeric(current, errors="coerce")

    psi_result = population_stability_index(
        reference_values.dropna().to_numpy(), current_values.dropna().to_numpy(), n_bins
    )

    reference_clean = reference_values.dropna().to_numpy()
    current_clean = current_values.dropna().to_numpy()
    if len(reference_clean) and len(current_clean):
        ks = stats.ks_2samp(reference_clean, current_clean)
        ks_statistic, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_statistic, ks_p = float("nan"), float("nan")

    return FeatureDrift(
        feature=feature,
        checkpoint_day=checkpoint_day,
        psi=psi_result.value,
        band="degenerate" if psi_result.degenerate else thresholds.band(psi_result.value),
        ks_statistic=ks_statistic,
        ks_p_value=ks_p,
        reference_mean=float(reference_values.mean()) if len(reference_clean) else float("nan"),
        current_mean=float(current_values.mean()) if len(current_clean) else float("nan"),
        reference_n=len(reference_clean),
        current_n=len(current_clean),
        empty_bins=psi_result.empty_bins,
        degenerate=psi_result.degenerate,
        degenerate_reason=psi_result.reason,
        missing_rate_change=float(current_values.isna().mean() - reference_values.isna().mean()),
    )


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    by_checkpoint: bool = True,
    thresholds: DriftThresholds | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-feature drift, by default computed within each checkpoint.

    Args:
        by_checkpoint: keep this True. Pooling compares cohorts at different
            points in a presentation and can both invent drift (composition
            shift) and hide it (opposite shifts at different checkpoints
            cancelling). On this data it hid it: 1 major feature pooled against 7
            within checkpoint. False exists only so the report can size the
            artefact on real data rather than assert it.

    Checkpoints present in only one of the two frames are skipped, not compared
    against nothing: a current cohort that has only reached day 60 has no day-150
    rows, and treating that as total drift on every day-150 feature would fire
    every monitor at the start of every presentation.
    """
    thresholds = thresholds or DriftThresholds()
    available = [f for f in features if f in reference.columns and f in current.columns]

    results: list[FeatureDrift] = []
    if not by_checkpoint:
        for feature in available:
            results.append(
                _feature_drift_one(
                    feature, reference[feature], current[feature], None, thresholds, n_bins
                )
            )
        return pd.DataFrame([r.to_row() for r in results])

    shared = sorted(
        set(reference[CHECKPOINT_COLUMN].unique()) & set(current[CHECKPOINT_COLUMN].unique())
    )
    for checkpoint in shared:
        reference_rows = reference[reference[CHECKPOINT_COLUMN] == checkpoint]
        current_rows = current[current[CHECKPOINT_COLUMN] == checkpoint]
        for feature in available:
            results.append(
                _feature_drift_one(
                    feature,
                    reference_rows[feature],
                    current_rows[feature],
                    int(checkpoint),
                    thresholds,
                    n_bins,
                )
            )
    return pd.DataFrame([r.to_row() for r in results])


@dataclass(frozen=True)
class PredictionDrift:
    """Shift in the score distribution. Needs no labels, so it fires immediately.

    This is the monitor that can act in real time. Everything label-dependent is
    at least 30 days behind, so if only one signal is watched, watch this one —
    while remembering it detects a changed *input* population, and says nothing
    about whether the model is still right.
    """

    checkpoint_day: int | None
    psi: float
    band: str
    reference_mean: float
    current_mean: float
    reference_alert_rate: float
    current_alert_rate: float
    alert_rate_change: float
    """The operationally meaningful number: at a fixed cutoff, how much more or
    less of the cohort is being flagged. A rise means staff are being asked to
    contact more students than the capacity the threshold was chosen for."""

    reference_n: int
    current_n: int
    degenerate: bool

    def to_row(self) -> dict[str, object]:
        return {
            "checkpoint_day": self.checkpoint_day,
            "psi": round(self.psi, 5) if np.isfinite(self.psi) else None,
            "band": self.band,
            "reference_mean": round(self.reference_mean, 5),
            "current_mean": round(self.current_mean, 5),
            "reference_alert_rate": round(self.reference_alert_rate, 4),
            "current_alert_rate": round(self.current_alert_rate, 4),
            "alert_rate_change": round(self.alert_rate_change, 4),
            "reference_n": self.reference_n,
            "current_n": self.current_n,
            "degenerate": self.degenerate,
        }


def prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    threshold: float,
    checkpoint_day: int | None = None,
    thresholds: DriftThresholds | None = None,
    n_bins: int = 10,
) -> PredictionDrift:
    """Score-distribution drift and the change in flagged share at a fixed cutoff.

    The alert-rate change is the number to act on. The threshold was chosen for a
    staffing capacity (a 5% alert budget), so a cohort that pushes the flagged
    share to 9% has quietly doubled the workload the operating point promised,
    whether or not the model is still accurate.
    """
    thresholds = thresholds or DriftThresholds()
    reference_scores = np.asarray(reference_scores, dtype="float64")
    current_scores = np.asarray(current_scores, dtype="float64")

    psi_result = population_stability_index(reference_scores, current_scores, n_bins)
    reference_rate = (
        float((reference_scores >= threshold).mean()) if len(reference_scores) else float("nan")
    )
    current_rate = (
        float((current_scores >= threshold).mean()) if len(current_scores) else float("nan")
    )

    return PredictionDrift(
        checkpoint_day=checkpoint_day,
        psi=psi_result.value,
        band="degenerate" if psi_result.degenerate else thresholds.band(psi_result.value),
        reference_mean=float(reference_scores.mean()) if len(reference_scores) else float("nan"),
        current_mean=float(current_scores.mean()) if len(current_scores) else float("nan"),
        reference_alert_rate=reference_rate,
        current_alert_rate=current_rate,
        alert_rate_change=current_rate - reference_rate,
        reference_n=len(reference_scores),
        current_n=len(current_scores),
        degenerate=psi_result.degenerate,
    )


@dataclass(frozen=True)
class CalibrationDrift:
    """Decay in the agreement between predicted and observed rates.

    This is the failure that matters most for this system, because the dashboard
    shows probabilities to staff. A model whose *ranking* still holds but whose
    probabilities have drifted will keep flagging roughly the right students while
    telling a counsellor that a 3% risk is a 12% risk — and the counsellor has no
    way to notice.

    Requires labels, so it lags the horizon by at least 30 days.
    """

    reference_observed: float
    current_observed: float
    reference_predicted: float
    current_predicted: float
    reference_gap: float
    current_gap: float
    gap_change: float
    labels_complete: bool
    """False when some current rows cannot have resolved yet. A calibration gap
    computed over unresolved rows counts every pending student as a negative,
    which biases the observed rate downward and makes the model look
    over-confident when it may not be."""

    def to_row(self) -> dict[str, object]:
        return {
            "reference_observed": round(self.reference_observed, 5),
            "current_observed": round(self.current_observed, 5),
            "reference_predicted": round(self.reference_predicted, 5),
            "current_predicted": round(self.current_predicted, 5),
            "reference_gap": round(self.reference_gap, 5),
            "current_gap": round(self.current_gap, 5),
            "gap_change": round(self.gap_change, 5),
            "labels_complete": self.labels_complete,
        }


def calibration_drift(
    reference_labels: np.ndarray,
    reference_scores: np.ndarray,
    current_labels: np.ndarray,
    current_scores: np.ndarray,
    labels_complete: bool = True,
) -> CalibrationDrift:
    """Change in mean predicted minus mean observed, reference against current.

    A negative gap means the model understates risk for that population, which is
    the direction that withholds support — the same convention as the fairness
    audit's ``calibration_gap``, deliberately, so the two reports cannot be read
    against each other by mistake.
    """
    reference_labels = np.asarray(reference_labels, dtype="float64")
    current_labels = np.asarray(current_labels, dtype="float64")
    reference_scores = np.asarray(reference_scores, dtype="float64")
    current_scores = np.asarray(current_scores, dtype="float64")

    reference_observed = float(reference_labels.mean()) if len(reference_labels) else float("nan")
    current_observed = float(current_labels.mean()) if len(current_labels) else float("nan")
    reference_predicted = float(reference_scores.mean()) if len(reference_scores) else float("nan")
    current_predicted = float(current_scores.mean()) if len(current_scores) else float("nan")

    reference_gap = reference_predicted - reference_observed
    current_gap = current_predicted - current_observed
    return CalibrationDrift(
        reference_observed=reference_observed,
        current_observed=current_observed,
        reference_predicted=reference_predicted,
        current_predicted=current_predicted,
        reference_gap=reference_gap,
        current_gap=current_gap,
        gap_change=current_gap - reference_gap,
        labels_complete=labels_complete,
    )


@dataclass
class DriftSummary:
    """What a monitor would surface, and what it deliberately does not conclude."""

    features_checked: int
    major: list[str] = field(default_factory=list)
    minor: list[str] = field(default_factory=list)
    degenerate: list[str] = field(default_factory=list)
    coverage_drops: list[str] = field(default_factory=list)

    @property
    def requires_review(self) -> bool:
        """Whether a human should look. Deliberately not "requires retraining".

        Nothing in this module can establish that retraining would help: drift in
        an input distribution is not evidence that the model got worse, and on
        this data a whole presentation's worth of drift coexisted with stable
        test performance. The monitor escalates to a person; the person decides.
        """
        return bool(self.major or self.coverage_drops)


def summarize(drift_frame: pd.DataFrame, coverage_tolerance: float = 0.05) -> DriftSummary:
    """Collapse a per-feature, per-checkpoint frame into a review decision.

    A feature is escalated if *any* checkpoint shows major drift, rather than on
    an average across checkpoints. Averaging would let a severe shift at day 30 —
    where an early-warning system is most useful — be cancelled out by stability
    at day 180, where a withdrawal warning arrives too late to act on.
    """
    if drift_frame.empty:
        return DriftSummary(features_checked=0)

    summary = DriftSummary(features_checked=int(drift_frame["feature"].nunique()))
    for feature, group in drift_frame.groupby("feature"):
        name = str(feature)
        bands = set(group["band"])
        if "major" in bands:
            summary.major.append(name)
        elif "minor" in bands:
            summary.minor.append(name)
        if "degenerate" in bands:
            summary.degenerate.append(name)
        if (group["missing_rate_change"].abs() > coverage_tolerance).any():
            summary.coverage_drops.append(name)

    for bucket in (summary.major, summary.minor, summary.degenerate, summary.coverage_drops):
        bucket.sort()
    return summary
