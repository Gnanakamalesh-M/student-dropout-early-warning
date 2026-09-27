"""Subgroup fairness audit.

The framing matters more than the arithmetic, so it is stated first.

**Alert-rate differences are expected and are not the finding.** A well-calibrated
model flags higher-base-rate groups more often, because those students really do
withdraw more often. Phase 3 recorded those base rates before any model existed
precisely so this phase could not mistake them for model behaviour: students
declaring a disability withdraw at 4.50% against 2.87%, so a fair model *will*
alert on them more. Demographic parity is the wrong test here, and chasing it
would mean withholding support from the group that needs it most.

**The question is whether errors differ.** Specifically the false-negative rate:
an at-risk student the model misses gets no offer of help. If that miss rate is
higher for one group, the system distributes its failures unequally, and that is a
finding regardless of what the alert rates look like.

**Where the sample is too small, this says so.** A subgroup with 26 positives
cannot support a conclusion about its recall, and a point estimate would invite
one. Every rate carries a Wilson interval, and groups below a minimum count are
flagged rather than quietly reported.

Protected attributes are read from the isolated ``student_demographics`` source
and are never features. This module is the only consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dropout_ews.evaluation.metrics import wilson_interval

# Below this many positives, recall is not estimable to any useful precision and
# the row is marked inconclusive rather than reported as a finding.
MIN_POSITIVES_FOR_RECALL = 30
MIN_ROWS_FOR_GROUP = 200

PROTECTED_ATTRIBUTES = ("gender", "age_band", "imd_band", "disability", "region")


@dataclass(frozen=True)
class GroupMetrics:
    """Error rates for one subgroup at one operating point."""

    attribute: str
    value: str
    rows: int
    positives: int
    base_rate: float
    flagged: int
    alert_rate: float
    recall: float
    recall_ci: tuple[float, float]
    false_negative_rate: float
    fnr_ci: tuple[float, float]
    precision: float
    precision_ci: tuple[float, float]
    false_positive_rate: float
    fpr_ci: tuple[float, float]
    mean_predicted: float
    calibration_gap: float
    """Mean predicted probability minus observed rate. A group whose risk is
    systematically over- or under-stated is miscalibrated for that group even if
    the model is calibrated overall."""

    conclusive: bool
    """False when the subgroup is too small to support a claim about its recall."""

    def to_row(self) -> dict[str, object]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "rows": self.rows,
            "positives": self.positives,
            "base_rate": round(self.base_rate, 4),
            "flagged": self.flagged,
            "alert_rate": round(self.alert_rate, 4),
            "recall": round(self.recall, 4),
            "recall_ci_low": round(self.recall_ci[0], 4),
            "recall_ci_high": round(self.recall_ci[1], 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "fnr_ci_low": round(self.fnr_ci[0], 4),
            "fnr_ci_high": round(self.fnr_ci[1], 4),
            "precision": round(self.precision, 4),
            "precision_ci_low": round(self.precision_ci[0], 4),
            "precision_ci_high": round(self.precision_ci[1], 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "fpr_ci_low": round(self.fpr_ci[0], 4),
            "fpr_ci_high": round(self.fpr_ci[1], 4),
            "mean_predicted": round(self.mean_predicted, 4),
            "calibration_gap": round(self.calibration_gap, 4),
            "conclusive": self.conclusive,
        }


def group_metrics(
    attribute: str,
    value: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> GroupMetrics:
    """Error rates and calibration for one subgroup."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype="float64")
    flagged_mask = y_prob >= threshold

    rows = len(y_true)
    positives = int(y_true.sum())
    negatives = rows - positives
    true_positive = int(y_true[flagged_mask].sum())
    flagged = int(flagged_mask.sum())
    false_positive = flagged - true_positive
    false_negative = positives - true_positive

    recall = true_positive / positives if positives else float("nan")
    fnr = false_negative / positives if positives else float("nan")
    precision = true_positive / flagged if flagged else float("nan")
    fpr = false_positive / negatives if negatives else float("nan")

    return GroupMetrics(
        attribute=attribute,
        value=value,
        rows=rows,
        positives=positives,
        base_rate=positives / rows if rows else float("nan"),
        flagged=flagged,
        alert_rate=flagged / rows if rows else float("nan"),
        recall=recall,
        recall_ci=wilson_interval(true_positive, positives),
        false_negative_rate=fnr,
        fnr_ci=wilson_interval(false_negative, positives),
        precision=precision,
        precision_ci=wilson_interval(true_positive, flagged),
        false_positive_rate=fpr,
        fpr_ci=wilson_interval(false_positive, negatives),
        mean_predicted=float(y_prob.mean()) if rows else float("nan"),
        calibration_gap=(
            float(y_prob.mean()) - positives / rows if rows else float("nan")
        ),
        conclusive=positives >= MIN_POSITIVES_FOR_RECALL and rows >= MIN_ROWS_FOR_GROUP,
    )


def audit(
    frame: pd.DataFrame,
    threshold: float,
    attributes: tuple[str, ...] = PROTECTED_ATTRIBUTES,
    label_column: str = "label",
    probability_column: str = "probability",
) -> pd.DataFrame:
    """Per-subgroup error rates at the operating point.

    Args:
        frame: rows carrying the label, the predicted probability, and each
            protected attribute.
        threshold: the operating cutoff the institution actually uses.

    Missing attribute values get their own ``missing`` group rather than being
    dropped: 5,766 rows have no recorded deprivation band, and silently excluding
    them would hide whichever way that population differs.
    """
    results: list[GroupMetrics] = []
    for attribute in attributes:
        if attribute not in frame.columns:
            continue
        values = frame[attribute].fillna("missing").astype(str)
        for value in sorted(values.unique()):
            subset = frame.loc[values == value]
            if subset.empty:
                continue
            results.append(
                group_metrics(
                    attribute,
                    value,
                    subset[label_column].to_numpy(),
                    subset[probability_column].to_numpy(),
                    threshold,
                )
            )
    return pd.DataFrame([item.to_row() for item in results])


@dataclass(frozen=True)
class DisparityFinding:
    """A gap between the best- and worst-served group for one attribute."""

    attribute: str
    metric: str
    worst_value: str
    worst_rate: float
    best_value: str
    best_rate: float
    gap: float
    intervals_overlap: bool
    """When the intervals overlap, the gap is not distinguishable from sampling
    noise and must not be reported as a disparity."""

    def to_row(self) -> dict[str, object]:
        return {
            "attribute": self.attribute,
            "metric": self.metric,
            "worst_group": self.worst_value,
            "worst_rate": round(self.worst_rate, 4),
            "best_group": self.best_value,
            "best_rate": round(self.best_rate, 4),
            "gap": round(self.gap, 4),
            "intervals_overlap": self.intervals_overlap,
            # Named distinctly from GroupMetrics.conclusive, which means "this
            # group has enough data to assess". This means "this gap is larger
            # than sampling noise". Reusing one word for both invites reading a
            # well-sampled group as evidence of a real gap.
            "distinguishable_from_noise": not self.intervals_overlap,
        }


_EMPTY_DISPARITY_COLUMNS = (
    "attribute",
    "metric",
    "worst_group",
    "worst_rate",
    "best_group",
    "best_rate",
    "gap",
    "intervals_overlap",
    "distinguishable_from_noise",
)


def disparities(
    audit_frame: pd.DataFrame,
    metric: str = "false_negative_rate",
    conclusive_only: bool = True,
) -> pd.DataFrame:
    """Largest gap per attribute, with an overlap check.

    ``false_negative_rate`` is the default because a missed at-risk student is the
    failure that denies someone support. A higher false-positive rate mostly costs
    staff time, which matters but is not the same kind of harm.

    Overlapping confidence intervals mean the gap is not distinguishable from
    noise. Reporting such a gap as a disparity is how a fairness audit
    manufactures findings, so it is flagged rather than asserted.
    """
    low_column = {
        "false_negative_rate": "fnr_ci_low",
        "recall": "recall_ci_low",
        "precision": "precision_ci_low",
        "false_positive_rate": "fpr_ci_low",
    }[metric]
    high_column = low_column.replace("_low", "_high")

    findings: list[DisparityFinding] = []
    # Only groups with enough data to assess; a gap computed against a group of
    # 13 positives is not a gap, it is noise with a label.
    source = audit_frame[audit_frame["conclusive"]] if conclusive_only else audit_frame

    for attribute, group in source.groupby("attribute"):
        usable = group.dropna(subset=[metric])
        if len(usable) < 2:
            continue
        worst = usable.loc[usable[metric].idxmax()]
        best = usable.loc[usable[metric].idxmin()]
        overlap = bool(
            worst[low_column] <= best[high_column] and best[low_column] <= worst[high_column]
        )
        findings.append(
            DisparityFinding(
                attribute=str(attribute),
                metric=metric,
                worst_value=str(worst["value"]),
                worst_rate=float(worst[metric]),
                best_value=str(best["value"]),
                best_rate=float(best[metric]),
                gap=float(worst[metric] - best[metric]),
                intervals_overlap=overlap,
            )
        )
    if not findings:
        # No attribute had two assessable groups. Return the empty frame with its
        # columns intact: a caller filtering on "distinguishable_from_noise" must
        # not crash on the case where nothing could be compared, which is the
        # most likely outcome on a small cohort.
        return pd.DataFrame(columns=list(_EMPTY_DISPARITY_COLUMNS))
    return pd.DataFrame([item.to_row() for item in findings]).sort_values(
        "gap", ascending=False
    )


def inconclusive_groups(audit_frame: pd.DataFrame) -> pd.DataFrame:
    """Groups too small to support a claim, with why.

    Reported explicitly. An audit that silently omits the groups it could not
    assess reads as though it assessed everything.
    """
    rows = audit_frame[~audit_frame["conclusive"]].copy()
    if rows.empty:
        return rows
    rows["reason"] = [
        "too few positives"
        if positives < MIN_POSITIVES_FOR_RECALL
        else "too few rows"
        for positives in rows["positives"]
    ]
    return rows[["attribute", "value", "rows", "positives", "reason"]]
