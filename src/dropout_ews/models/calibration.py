"""Probability calibration and risk-band derivation.

Calibration is not optional here. The dashboard shows a counsellor a percentage,
and a raw XGBoost output is a margin squashed through a sigmoid, not a
probability — an uncalibrated "87% risk" is a number that looks like a
probability and is not one. Isotonic regression fitted on the held-out
validation presentation fixes the mapping without assuming a functional form.

Isotonic rather than Platt scaling: Platt assumes a sigmoid relationship between
score and outcome, which is a strong assumption at a 3% base rate, and there are
993 validation positives — enough for the nonparametric fit isotonic needs.

**Risk bands are derived from the alert budget, not chosen.** Phase 5 showed a
recall target is not a usable control (80% recall means flagging 47% of the
cohort), so bands are cut so that the CRITICAL band matches what an institution
can actually staff. `thresholds.yaml` documented this procedure in Phase 1; this
module implements it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


@dataclass
class CalibrationReport:
    """Before/after calibration quality."""

    brier_before: float
    brier_after: float
    mean_predicted_before: float
    mean_predicted_after: float
    observed_rate: float
    reliability: pd.DataFrame

    @property
    def improved(self) -> bool:
        return self.brier_after <= self.brier_before

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "brier_before": round(self.brier_before, 6),
                    "brier_after": round(self.brier_after, 6),
                    "brier_improvement": round(self.brier_before - self.brier_after, 6),
                    "mean_predicted_before": round(self.mean_predicted_before, 5),
                    "mean_predicted_after": round(self.mean_predicted_after, 5),
                    "observed_rate": round(self.observed_rate, 5),
                }
            ]
        )


def calibrate(
    pipeline: BaseEstimator, X_validation: pd.DataFrame, y_validation: np.ndarray
) -> CalibratedClassifierCV:
    """Wrap an already-fitted pipeline in isotonic calibration.

    Only the isotonic mapping is fitted here; the pipeline keeps the parameters
    it learned on the training presentations. Passing an unfitted estimator
    would refit everything on validation and quietly discard the training data,
    which is the mistake this function exists to prevent.

    ``FrozenEstimator`` is used where available (scikit-learn >= 1.6), since
    ``cv="prefit"`` is deprecated there and removed in 1.8. The fallback keeps
    the older path working so the project still installs on 1.4/1.5.
    """
    try:  # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="isotonic")
    except ImportError:  # pragma: no cover - older scikit-learn
        calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv="prefit")
    calibrated.fit(X_validation, y_validation)
    return calibrated


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted rate per probability bin.

    Quantile bins rather than equal-width: at a 3% base rate almost every
    prediction lands in the lowest equal-width bin, so the table would say
    nothing about the high-risk region that actually drives decisions.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob, dtype="float64")})
    # `duplicates="drop"` because many predictions tie at low probabilities.
    frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        rows=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
    )
    grouped["gap"] = grouped["mean_predicted"] - grouped["observed_rate"]
    return grouped.reset_index().assign(bin=lambda frame: frame["bin"].astype(str)).round(5)


def assess_calibration(
    y_true: np.ndarray,
    prob_before: np.ndarray,
    prob_after: np.ndarray,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compare Brier score and reliability before and after calibration."""
    y_true = np.asarray(y_true)
    return CalibrationReport(
        brier_before=float(brier_score_loss(y_true, prob_before)),
        brier_after=float(brier_score_loss(y_true, prob_after)),
        mean_predicted_before=float(np.mean(prob_before)),
        mean_predicted_after=float(np.mean(prob_after)),
        observed_rate=float(y_true.mean()),
        reliability=reliability_table(y_true, prob_after, n_bins=n_bins),
    )


# ---------------------------------------------------------------------------
# Risk bands from the alert budget
# ---------------------------------------------------------------------------


def derive_bands_from_budget(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    critical_budget: float = 0.05,
    high_budget: float = 0.15,
    medium_budget: float = 0.35,
) -> pd.DataFrame:
    """Cut risk bands so each tier matches a support capacity.

    The budgets are nested shares of the cohort: ``critical_budget`` is the top
    slice, ``high_budget`` the top slice including CRITICAL, and so on. This is
    the procedure ``thresholds.yaml`` specified in Phase 1 — set the CRITICAL
    cutoff from how many intensive cases staff can run, not from where the
    probability distribution happens to bend.

    Returns one row per band with its cutoff and its realised precision, recall
    and lift, which is the table the model card publishes. A band whose
    precision is near the base rate is not informative and should be collapsed;
    reporting them makes that visible.
    """
    if not 0 < critical_budget < high_budget < medium_budget < 1:
        raise ValueError(
            "budgets must satisfy 0 < critical < high < medium < 1, got "
            f"{critical_budget}, {high_budget}, {medium_budget}"
        )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype="float64")
    n = len(y_prob)
    positives = int(y_true.sum())
    base_rate = positives / n if n else float("nan")

    # Cutoffs are upper quantiles: the top `budget` share of scores.
    cutoffs = {
        "critical": float(np.quantile(y_prob, 1.0 - critical_budget)),
        "high": float(np.quantile(y_prob, 1.0 - high_budget)),
        "medium": float(np.quantile(y_prob, 1.0 - medium_budget)),
        "low": 0.0,
    }

    rows = []
    ordered = ["low", "medium", "high", "critical"]
    for index, key in enumerate(ordered):
        lower = cutoffs[key]
        upper = cutoffs[ordered[index + 1]] if index + 1 < len(ordered) else np.inf
        # Bands are half-open [lower, upper), with CRITICAL unbounded above.
        in_band = (y_prob >= lower) & (y_prob < upper)
        count = int(in_band.sum())
        caught = int(y_true[in_band].sum())
        precision = caught / count if count else float("nan")
        # Cumulative view: everything at or above this band's cutoff.
        at_or_above = y_prob >= lower
        cumulative_caught = int(y_true[at_or_above].sum())
        rows.append(
            {
                "band": key,
                "min_probability": round(lower, 5),
                "students_in_band": count,
                "share_of_cohort": round(count / n, 4) if n else float("nan"),
                "positives_in_band": caught,
                "precision_in_band": round(precision, 4),
                "lift_in_band": round(precision / base_rate, 2)
                if base_rate and not np.isnan(precision)
                else float("nan"),
                "cumulative_recall_at_or_above": round(cumulative_caught / positives, 4)
                if positives
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def bands_to_config(
    band_table: pd.DataFrame,
    version: int,
    name: str,
) -> dict[str, object]:
    """Render a band table as the ``thresholds.yaml`` structure.

    ``calibrated: true`` is set here and only here — the shipped placeholder
    config carries ``false``, and a test asserts it stays false until this
    procedure has actually been run.
    """
    labels = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
    colors = {
        "low": "#2E7D32",
        "medium": "#F9A825",
        "high": "#EF6C00",
        "critical": "#C62828",
    }
    actions = {
        "low": "No outreach. Included in routine cohort monitoring.",
        "medium": "Passive monitoring. Flag if risk rises for two consecutive checkpoints.",
        "high": "Human review, then advisor outreach if the reviewer agrees.",
        "critical": "Priority human review and coordinated support offer.",
    }
    order = ["low", "medium", "high", "critical"]
    lookup = band_table.set_index("band")
    return {
        "version": version,
        "name": name,
        "calibrated": True,
        "bands": [
            {
                "key": key,
                "label": labels[key],
                "min_probability": float(lookup.loc[key, "min_probability"]),
                "color": colors[key],
                "action": actions[key],
            }
            for key in order
        ],
    }
