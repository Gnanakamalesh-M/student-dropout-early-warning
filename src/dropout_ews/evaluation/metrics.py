"""Evaluation metrics, bootstrap intervals, and operating-point selection.

Three choices here are deliberate and matter more than the metric list itself.

**PR-AUC is the headline, not ROC-AUC.** At a 3% positive rate, ROC-AUC is
dominated by the vast negative class and stays flattering even for a weak
model. Average precision tracks what an institution actually experiences: of
the students you flag, how many were about to leave.

**The bootstrap resamples students, not rows.** One student contributes up to
six checkpoint rows, so rows are not independent and a row-level bootstrap
assumes something untrue.

Measured on the real test set, the clustered interval turns out to be **0.93x
the width** of the naive row-level one — very slightly narrower, not wider.
That was not the expected direction, and the reason is structural rather than
accidental: because no row is emitted at or after a withdrawal, **at most one of
a student's six rows can be positive**. Positives are therefore spread across
students rather than clumped within them, and the design effect is close to 1.

The clustered version is still the default, because the correct unit of
resampling is a property of the data rather than of the result it produces. But
it is not doing the heavy lifting here, and claiming it corrected a badly
understated interval would be false.

**The operating point is chosen by recall target, never 0.5.** Missing a student
who is about to leave costs more than an unnecessary conversation, so the
threshold is set to hit a recall target on validation, and the resulting
precision *and absolute alert volume* are reported alongside. A threshold that
implies flagging a fifth of the cohort is not usable regardless of its F1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

GROUP_COLUMN = "id_student"


@dataclass
class ThresholdMetrics:
    """Metrics at one decision threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    flagged: int
    flagged_share: float
    """Share of the cohort flagged. The capacity question: an institution
    cannot run intensive outreach on 20% of its students."""
    lift_over_base_rate: float
    """Precision divided by the base rate. 1.0 means the model is no better
    than flagging at random."""


@dataclass
class RankingMetrics:
    """Threshold-free metrics, plus calibration."""

    average_precision: float
    roc_auc: float
    brier_score: float
    base_rate: float
    n_rows: int
    n_positives: int


@dataclass
class EvaluationResult:
    """Everything reported for one model on one dataset."""

    name: str
    ranking: RankingMetrics
    operating_point: ThresholdMetrics
    per_checkpoint: pd.DataFrame = field(default_factory=pd.DataFrame)
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {"model": self.name}
        row.update(asdict(self.ranking))
        row.update({f"op_{k}": v for k, v in asdict(self.operating_point).items()})
        for metric, (low, high) in self.intervals.items():
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        return row


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> RankingMetrics:
    """Threshold-free metrics.

    ``brier_score`` is **NaN unless ``y_score`` lies in [0, 1]**. Brier measures
    calibration, which only means something for a probability. The trivial-rule
    baselines are arbitrary-scale rankers (one is negated click counts), so
    reporting a Brier score for them would be a category error rather than a
    poor result. AP and ROC-AUC are rank-based and remain valid for any scale.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    is_probability = bool(y_score.min() >= 0.0 and y_score.max() <= 1.0)
    return RankingMetrics(
        average_precision=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        brier_score=float(brier_score_loss(y_true, y_score)) if is_probability else float("nan"),
        base_rate=float(y_true.mean()),
        n_rows=len(y_true),
        n_positives=int(y_true.sum()),
    )


def metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> ThresholdMetrics:
    """Confusion-matrix metrics at a given threshold."""
    y_true = np.asarray(y_true)
    predicted = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    flagged = int(tp + fp)
    precision = tp / flagged if flagged else 0.0
    positives = int(tp + fn)
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    base_rate = y_true.mean()

    return ThresholdMetrics(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
        flagged=flagged,
        flagged_share=float(flagged / len(y_true)) if len(y_true) else 0.0,
        lift_over_base_rate=float(precision / base_rate) if base_rate > 0 else float("nan"),
    )


def threshold_for_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float) -> float:
    """Lowest threshold achieving at least ``target_recall``.

    "Lowest" is the right choice: among thresholds meeting the recall target,
    the highest threshold gives the best precision, so we want the *largest*
    threshold whose recall still clears the bar. ``precision_recall_curve``
    returns recall in decreasing order, so the first index meeting the target
    from the high-recall end is the one to take.

    Falls back to the minimum score when the target is unreachable, which flags
    everything — a degenerate but honest answer rather than a silent miss.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")

    _, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # `thresholds` is one shorter than `recalls`/`precisions` by construction.
    eligible = np.nonzero(recalls[:-1] >= target_recall)[0]
    if eligible.size == 0:
        return float(np.min(y_score))
    return float(thresholds[eligible[-1]])


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_targets: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    """The tradeoff table published in the model card.

    Shows what each recall target costs in precision and in alert volume, so the
    operating point is a documented decision rather than a default.
    """
    rows = []
    for target in recall_targets:
        threshold = threshold_for_recall(y_true, y_score, target)
        metrics = metrics_at_threshold(y_true, y_score, threshold)
        rows.append(
            {
                "target_recall": target,
                "threshold": round(metrics.threshold, 5),
                "achieved_recall": round(metrics.recall, 4),
                "precision": round(metrics.precision, 4),
                "lift": round(metrics.lift_over_base_rate, 2),
                "flagged": metrics.flagged,
                "flagged_share": round(metrics.flagged_share, 4),
                "f1": round(metrics.f1, 4),
            }
        )
    return pd.DataFrame(rows)


def budget_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    budgets: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Recall achievable within a fixed alert budget — the capacity view.

    This is the inverse of :func:`threshold_sweep` and the more operationally
    honest direction. An institution does not choose a recall target; it has a
    fixed number of staff hours and can contact roughly N students per cycle.
    The real question is therefore "given we can follow up 5% of the cohort,
    how many of the students about to leave do we reach?"

    Asking it this way round also stops a recall target being quoted without
    its alert volume, which is how an unusable operating point gets reported as
    a success.

    Budgets are shares of the scored cohort. Ties at the cut are included, so
    the realised share can slightly exceed the budget.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    n = len(y_true)
    positives = int(y_true.sum())
    base_rate = positives / n if n else float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    rows = []
    for budget in budgets:
        k = max(1, round(budget * n))
        threshold = float(y_score[order[k - 1]])
        flagged_mask = y_score >= threshold
        flagged = int(flagged_mask.sum())
        caught = int(y_true[flagged_mask].sum())
        precision = caught / flagged if flagged else 0.0
        rows.append(
            {
                "alert_budget": budget,
                "flagged": flagged,
                "realised_share": round(flagged / n, 4),
                "threshold": round(threshold, 5),
                "recall": round(caught / positives, 4) if positives else float("nan"),
                "precision": round(precision, 4),
                "lift": round(precision / base_rate, 2) if base_rate else float("nan"),
                "students_caught": caught,
                "positives_total": positives,
            }
        )
    return pd.DataFrame(rows)


def per_checkpoint_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    checkpoint_day: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Metrics broken out per checkpoint.

    Mandatory (docs/TASK_SPEC.md). Aggregate metrics hide the failure mode that
    matters most: a model that only detects risk at day 180 has little
    intervention value even with a good overall score.
    """
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(y_score), "t": np.asarray(checkpoint_day)}
    )
    rows = []
    for checkpoint, group in frame.groupby("t"):
        y = group["y"].to_numpy()
        p = group["p"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        at = metrics_at_threshold(y, p, threshold)
        rows.append(
            {
                "checkpoint_day": int(checkpoint),
                "rows": len(y),
                "positives": int(y.sum()),
                "base_rate": round(float(y.mean()), 4),
                "average_precision": round(float(average_precision_score(y, p)), 4),
                "roc_auc": round(float(roc_auc_score(y, p)), 4),
                "precision": round(at.precision, 4),
                "recall": round(at.recall, 4),
                "lift": round(at.lift_over_base_rate, 2),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap intervals, resampling **students** not rows.

    Rows within a student are not independent, so resampling rows would assume
    something untrue about the data. Resampling whole students preserves
    whatever dependence exists and is the defensible unit regardless of which
    direction it moves the interval.

    How much it matters depends on within-student label correlation, and on this
    data it is small: because no row exists at or after a withdrawal, at most
    one of a student's rows can be positive, so positives are spread across
    students rather than clumped. The measured design effect on the real test
    set is about 0.93, i.e. the clustered interval is marginally narrower than
    the naive one. See the module docstring.

    Iterations that draw a single-class sample are skipped rather than scored,
    and fewer than 20 usable iterations yields NaN rather than an interval
    computed from a handful of draws.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    groups = np.asarray(groups)

    # Brier is skipped for non-probability scores; see ranking_metrics.
    is_probability = bool(y_score.min() >= 0.0 and y_score.max() <= 1.0)

    unique_groups = np.unique(groups)
    # Row positions per student, computed once.
    positions = {group: np.nonzero(groups == group)[0] for group in unique_groups}

    rng = np.random.default_rng(random_state)
    samples: dict[str, list[float]] = {"average_precision": [], "roc_auc": [], "brier_score": []}

    for _ in range(n_iterations):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([positions[group] for group in drawn])
        y_boot, p_boot = y_true[index], y_score[index]
        if len(np.unique(y_boot)) < 2:
            continue
        samples["average_precision"].append(float(average_precision_score(y_boot, p_boot)))
        samples["roc_auc"].append(float(roc_auc_score(y_boot, p_boot)))
        if is_probability:
            samples["brier_score"].append(float(brier_score_loss(y_boot, p_boot)))

    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, tuple[float, float]] = {}
    for metric, values in samples.items():
        if len(values) < 20:
            intervals[metric] = (float("nan"), float("nan"))
            continue
        array = np.asarray(values)
        intervals[metric] = (
            float(np.quantile(array, alpha)),
            float(np.quantile(array, 1.0 - alpha)),
        )
    return intervals


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    checkpoint_day: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    bootstrap_iterations: int = 1000,
    random_state: int = 42,
) -> EvaluationResult:
    """Full evaluation of one model on one dataset."""
    return EvaluationResult(
        name=name,
        ranking=ranking_metrics(y_true, y_score),
        operating_point=metrics_at_threshold(y_true, y_score, threshold),
        per_checkpoint=per_checkpoint_metrics(y_true, y_score, checkpoint_day, threshold),
        intervals=cluster_bootstrap_intervals(
            y_true,
            y_score,
            groups,
            n_iterations=bootstrap_iterations,
            random_state=random_state,
        ),
    )


def trivial_rule_scores(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Score functions for the Phase 3 trivial rules.

    Any model must beat these to justify its complexity. Expressed as scores
    rather than hard flags so they can be compared on the same PR-AUC footing:
    a rule is a degenerate ranker with two levels.
    """
    return {
        "rule: no clicks in 14d": (features["clicks_14d"] == 0).to_numpy(dtype="float64"),
        "rule: no clicks in 7d": (features["clicks_7d"] == 0).to_numpy(dtype="float64"),
        # Negated so that lower engagement means a higher score.
        "rule: inverse clicks_28d": -features["clicks_28d"].to_numpy(dtype="float64"),
        "rule: days since last activity": features["days_since_last_activity"]
        .fillna(features["checkpoint_day"])
        .to_numpy(dtype="float64"),
    }
