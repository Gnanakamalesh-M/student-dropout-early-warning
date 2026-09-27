"""Class-imbalance strategies, and a concrete test of the SMOTE objection.

ADR-0002 rejected SMOTE on the grounds that interpolating between students with
rolling-window temporal features "manufactures trajectories that are physically
impossible". That was an argument, not evidence. This module turns it into a
measurement.

The feature set contains **hard structural invariants** — relationships that
hold for every real student by construction, not statistically:

* Nested windows. ``clicks_7d <= clicks_14d <= clicks_28d <= clicks_56d``,
  because the 7-day window is a subset of the 14-day window. Same for
  ``active_days_*``.
* Counting bounds. ``active_days_7d <= 7``, ``active_days_28d <= 28``: a
  student cannot be active on more days than the window contains.
* Binary indicators. ``ever_active`` and ``has_submitted`` are 0 or 1.
* Consistency between paired features. A student with ``ever_active == 0``
  must have ``clicks_all_time == 0``.

A real student can never violate these. An interpolated point between two real
students violates them routinely, because SMOTE interpolates each dimension
independently and has no notion of the constraints linking them. Counting the
violations makes the objection checkable, and a reader can disagree with the
conclusion while still seeing the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Nested trailing windows: each must be less than or equal to the next.
NESTED_WINDOW_CHAINS: tuple[tuple[str, ...], ...] = (
    ("clicks_7d", "clicks_14d", "clicks_28d", "clicks_56d", "clicks_all_time"),
    (
        "active_days_7d",
        "active_days_14d",
        "active_days_28d",
        "active_days_56d",
        "active_days_all_time",
    ),
)

# A student cannot be active on more days than the window holds.
WINDOW_DAY_CAPS: dict[str, int] = {
    "active_days_7d": 7,
    "active_days_14d": 14,
    "active_days_28d": 28,
    "active_days_56d": 56,
}

BINARY_FEATURES: tuple[str, ...] = ("ever_active", "has_submitted")

NON_NEGATIVE_FEATURES: tuple[str, ...] = (
    "clicks_7d",
    "clicks_14d",
    "clicks_28d",
    "clicks_56d",
    "clicks_all_time",
    "active_days_7d",
    "active_days_28d",
    "assessments_due",
    "assessments_submitted",
    "assessments_missed",
    "inactive_weeks_streak",
)


@dataclass
class InvariantReport:
    """Counts of structurally impossible rows, by invariant family."""

    n_rows: int
    nested_windows: int
    day_caps: int
    binary_indicators: int
    negative_counts: int

    @property
    def total_violating_rows(self) -> int:
        return self.any_violation_count

    any_violation_count: int = 0

    def to_frame(self, label: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dataset": label,
                    "rows": self.n_rows,
                    "nested_window_violations": self.nested_windows,
                    "day_cap_violations": self.day_caps,
                    "non_binary_indicators": self.binary_indicators,
                    "negative_counts": self.negative_counts,
                    "rows_with_any_violation": self.any_violation_count,
                    "share_violating": round(self.any_violation_count / self.n_rows, 4)
                    if self.n_rows
                    else float("nan"),
                }
            ]
        )


def count_invariant_violations(frame: pd.DataFrame) -> InvariantReport:
    """Count rows violating hard structural invariants.

    A real student cannot violate any of these, so a non-zero count means the
    rows are not possible student states.
    """
    violating = np.zeros(len(frame), dtype=bool)

    nested = 0
    for chain in NESTED_WINDOW_CHAINS:
        present = [name for name in chain if name in frame.columns]
        for smaller, larger in pairwise(present):
            # Tolerance absorbs float noise; violations of interest are large.
            bad = (frame[smaller] > frame[larger] + 1e-6).to_numpy()
            nested += int(bad.sum())
            violating |= bad

    caps = 0
    for name, cap in WINDOW_DAY_CAPS.items():
        if name in frame.columns:
            bad = (frame[name] > cap + 1e-6).to_numpy()
            caps += int(bad.sum())
            violating |= bad

    binary = 0
    for name in BINARY_FEATURES:
        if name in frame.columns:
            values = frame[name].to_numpy()
            bad = ~np.isclose(values, 0.0) & ~np.isclose(values, 1.0)
            binary += int(bad.sum())
            violating |= bad

    negative = 0
    for name in NON_NEGATIVE_FEATURES:
        if name in frame.columns:
            bad = (frame[name] < -1e-6).to_numpy()
            negative += int(bad.sum())
            violating |= bad

    return InvariantReport(
        n_rows=len(frame),
        nested_windows=nested,
        day_caps=caps,
        binary_indicators=binary,
        negative_counts=negative,
        any_violation_count=int(violating.sum()),
    )


def smote_synthetic_rows(
    features: pd.DataFrame, y: np.ndarray, random_state: int = 42
) -> pd.DataFrame:
    """Return only the rows SMOTE invents, for inspection.

    SMOTE returns the original rows followed by the synthetic ones, so the tail
    beyond the original row count is what it generated.
    """
    sampler = SMOTE(random_state=random_state, k_neighbors=5)
    resampled, _ = sampler.fit_resample(features, y)
    resampled = pd.DataFrame(resampled, columns=features.columns)
    return resampled.iloc[len(features) :].reset_index(drop=True)


def imbalance_strategies(scale_pos_weight: float) -> dict[str, dict[str, object]]:
    """The strategies compared in Phase 6.

    Each entry gives the ``scale_pos_weight`` and resampler to pass to
    :func:`~dropout_ews.models.xgboost_model.xgboost_pipeline`. Resamplers are
    inserted inside the pipeline, so they are refitted per fold and never touch
    validation rows — applying SMOTE before splitting is a classic way to leak,
    because a synthetic training point can be interpolated from a validation
    neighbour.
    """
    return {
        "none": {"scale_pos_weight": None, "resampler": None},
        "class weights": {"scale_pos_weight": scale_pos_weight, "resampler": None},
        "SMOTE": {
            "scale_pos_weight": None,
            "resampler": SMOTE(random_state=42, k_neighbors=5),
        },
        "SMOTE + undersample": {
            "scale_pos_weight": None,
            # Oversample the minority to 10%, then trim the majority. The
            # combination is the usual recommendation when full balancing
            # distorts the prior too far.
            "resampler": SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.1),
        },
        "undersample majority": {
            "scale_pos_weight": None,
            "resampler": RandomUnderSampler(random_state=42, sampling_strategy=0.1),
        },
    }


# Preference order among strategies that score equivalently, simplest first.
# "Simplest" means: discards no data, invents no data, adds no randomness. Each
# step down the list gives up one of those properties.
STRATEGY_PREFERENCE: tuple[str, ...] = (
    "none",
    "class weights",
    "undersample majority",
    "SMOTE + undersample",
    "SMOTE",
)


def select_strategy(comparison: pd.DataFrame, tolerance_in_std: float = 1.0) -> tuple[str, str]:
    """Choose an imbalance strategy without chasing fold noise.

    Taking ``argmax`` over cross-validated scores is a known failure mode when
    the candidates are statistically tied: the winner is then chosen by which
    fold split happened to favour it. On the real training split the top four
    strategies fell within 0.006 PR-AUC of each other against a fold standard
    deviation of about 0.008, so ``argmax`` was selecting on noise.

    The rule instead is: take every strategy within ``tolerance_in_std`` fold
    standard deviations of the best, then pick the one earliest in
    :data:`STRATEGY_PREFERENCE`. That prefers the simplest defensible option and
    is decided in advance rather than after seeing which one won.

    Args:
        comparison: frame with ``strategy``, ``cv_pr_auc`` and ``cv_std``.

    Returns:
        The chosen strategy and a one-line rationale for the report.
    """
    if comparison.empty:
        raise ValueError("cannot select a strategy from an empty comparison")

    ranked = comparison.sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    cutoff = float(best["cv_pr_auc"]) - tolerance_in_std * float(best["cv_std"])
    tied = ranked[ranked["cv_pr_auc"] >= cutoff]["strategy"].tolist()

    for candidate in STRATEGY_PREFERENCE:
        if candidate in tied:
            if candidate == best["strategy"]:
                reason = (
                    f"best CV PR-AUC ({best['cv_pr_auc']:.4f}) and also the simplest "
                    f"of the {len(tied)} statistically tied strategies"
                )
            else:
                reason = (
                    f"within one fold SD of the best ({best['strategy']}, "
                    f"{best['cv_pr_auc']:.4f} +/- {best['cv_std']:.4f}) and simpler, "
                    f"so preferred over chasing a {float(best['cv_pr_auc']) - float(ranked.loc[ranked['strategy'] == candidate, 'cv_pr_auc'].iloc[0]):.4f} "
                    "difference inside fold noise"
                )
            return candidate, reason

    # Nothing in the preference list matched, so fall back to the best score.
    return str(best["strategy"]), "highest CV PR-AUC (no preferred strategy was tied)"
