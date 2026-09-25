"""EDA computations for the OULAD checkpoint population.

Every function here respects the as-of rule from docs/TASK_SPEC.md: an
exploratory statistic computed from post-checkpoint data would suggest a signal
the model can never actually use, which is a worse outcome than no analysis at
all. The trailing-window aggregation in :func:`engagement_windows` is
therefore written with the same strict ``course_day <= checkpoint_day``
boundary the production feature builder will use.

These aggregates are **exploratory**. Phase 4 builds the real
``FeatureBuilder`` with the as-of property test; nothing here is imported by
the training pipeline or the API.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from dropout_ews.data.clickstream import STUDENT_DAY_PARQUET
from dropout_ews.data.loaders import load_student_info

KEYS = ["code_module", "code_presentation", "id_student"]


# ---------------------------------------------------------------------------
# Trailing-window engagement, computed strictly as of the checkpoint
# ---------------------------------------------------------------------------

_WINDOW_SQL = """
WITH pop AS (
    SELECT code_module, code_presentation, id_student, checkpoint_day, label
    FROM population
),
joined AS (
    SELECT
        p.code_module, p.code_presentation, p.id_student,
        p.checkpoint_day, p.label,
        e.course_day, e.clicks, e.distinct_resources
    FROM pop p
    LEFT JOIN read_parquet($engagement) e
      ON  e.code_module       = p.code_module
      AND e.code_presentation = p.code_presentation
      AND e.id_student        = p.id_student
      -- THE AS-OF BOUNDARY. Nothing after the checkpoint may contribute.
      AND e.course_day       <= p.checkpoint_day
)
SELECT
    code_module, code_presentation, id_student, checkpoint_day, label,

    COALESCE(SUM(clicks), 0)                                   AS clicks_total,
    COALESCE(SUM(CASE WHEN course_day > checkpoint_day - 7  THEN clicks END), 0) AS clicks_7d,
    COALESCE(SUM(CASE WHEN course_day > checkpoint_day - 14 THEN clicks END), 0) AS clicks_14d,
    COALESCE(SUM(CASE WHEN course_day > checkpoint_day - 30 THEN clicks END), 0) AS clicks_30d,
    COALESCE(SUM(CASE WHEN course_day > checkpoint_day - 60
                       AND course_day <= checkpoint_day - 30 THEN clicks END), 0) AS clicks_prev_30d,

    COUNT(DISTINCT CASE WHEN course_day > checkpoint_day - 30 THEN course_day END) AS active_days_30d,
    COUNT(DISTINCT course_day)                                 AS active_days_total,
    COALESCE(MAX(CASE WHEN course_day > checkpoint_day - 30
                      THEN distinct_resources END), 0)         AS max_resources_30d,

    -- NULL when the student has never been active; handled by the caller
    -- rather than silently coerced, because "never active" is a real and
    -- substantial population (6,519 students) and not a zero.
    MAX(course_day)                                            AS last_active_day
FROM joined
GROUP BY 1, 2, 3, 4, 5
"""


def engagement_windows(
    population: pd.DataFrame, engagement_path: str | None = None
) -> pd.DataFrame:
    """Compute trailing-window engagement aggregates as of each checkpoint.

    Derived columns added on top of the SQL aggregates:

    * ``days_since_last_activity`` — from ``last_active_day``; ``NaN`` where
      the student has no activity at all.
    * ``ever_active`` — explicit flag for the never-engaged population.
    * ``clicks_delta_30d`` — recent window minus the preceding one, the
      simplest possible trend signal.
    * ``clicks_ratio_30d`` — the same as a ratio, which is scale-free across
      students with very different baseline activity.
    """
    path = str(engagement_path or STUDENT_DAY_PARQUET)
    connection = duckdb.connect()
    try:
        connection.register("population", population)
        frame = connection.execute(_WINDOW_SQL, {"engagement": path}).fetchdf()
    finally:
        connection.close()

    frame["ever_active"] = frame["last_active_day"].notna()
    frame["days_since_last_activity"] = (frame["checkpoint_day"] - frame["last_active_day"]).astype(
        float
    )
    frame["clicks_delta_30d"] = frame["clicks_30d"] - frame["clicks_prev_30d"]
    # +1 smoothing so a student with no activity in the previous window does
    # not produce an infinity.
    frame["clicks_ratio_30d"] = (frame["clicks_30d"] + 1) / (frame["clicks_prev_30d"] + 1)
    return frame


# ---------------------------------------------------------------------------
# Cohort composition
# ---------------------------------------------------------------------------


def outcome_by_module() -> pd.DataFrame:
    """Withdrawal rate per module.

    If modules differ substantially, an absolute engagement threshold means
    different things in different courses, and cohort-relative features
    (z-scores within module-presentation) become necessary rather than
    optional.
    """
    info = load_student_info()
    grouped = (
        info.assign(withdrawn=(info["final_result"] == "Withdrawn").astype(int))
        .groupby("code_module")
        .agg(students=("id_student", "size"), withdrawal_rate=("withdrawn", "mean"))
    )
    return grouped.sort_values("withdrawal_rate", ascending=False)


def positive_rate_by_group(windows: pd.DataFrame, column: str = "code_module") -> pd.DataFrame:
    """Checkpoint-level positive rate by group, with counts."""
    return (
        windows.groupby(column)
        .agg(rows=("label", "size"), positives=("label", "sum"), rate=("label", "mean"))
        .sort_values("rate", ascending=False)
    )


# ---------------------------------------------------------------------------
# Engagement contrast between withdrawers and completers
# ---------------------------------------------------------------------------


def engagement_by_label(windows: pd.DataFrame) -> pd.DataFrame:
    """Median engagement for positives vs negatives, per checkpoint.

    Medians rather than means: click distributions are heavily right-skewed,
    so a mean is dominated by a handful of very active students and hides the
    typical case.
    """
    columns = [
        "clicks_30d",
        "active_days_30d",
        "clicks_delta_30d",
        "days_since_last_activity",
    ]
    grouped = windows.groupby(["checkpoint_day", "label"])[columns].median()
    return grouped.round(2)


def zero_engagement_rate(windows: pd.DataFrame) -> pd.DataFrame:
    """Share of students with no activity in the trailing 30 days.

    This is the single most operationally useful simple signal, and it also
    bounds how much a model can achieve: if a large share of positives had
    zero recent activity, a trivial rule already finds them.
    """
    windows = windows.assign(zero_30d=(windows["clicks_30d"] == 0).astype(int))
    return (
        windows.groupby(["checkpoint_day", "label"])["zero_30d"]
        .agg(rows="size", zero_rate="mean")
        .round(4)
    )


def aligned_engagement_before_event(
    population: pd.DataFrame,
    engagement_path: str | None = None,
    weeks_before: int = 8,
) -> pd.DataFrame:
    """Weekly clicks aligned to weeks before withdrawal, for withdrawers.

    Aligning on the event rather than on calendar time is what reveals whether
    disengagement *precedes* withdrawal — the premise the entire early-warning
    system rests on. If engagement only collapses in the final days, the usable
    warning window is short and the project's value proposition weakens.

    Computed only for students with a known event time, and only from activity
    strictly before the event.
    """
    withdrawers = population.loc[population["date_unregistration"].notna(), :]
    withdrawers = withdrawers.drop_duplicates(subset=KEYS)[[*KEYS, "date_unregistration"]]

    path = str(engagement_path or STUDENT_DAY_PARQUET)
    connection = duckdb.connect()
    try:
        connection.register("withdrawers", withdrawers)
        frame = connection.execute(
            """
            SELECT
                CAST(FLOOR((w.date_unregistration - e.course_day - 1) / 7) AS INTEGER)
                    AS weeks_before_event,
                w.id_student,
                SUM(e.clicks) AS clicks
            FROM withdrawers w
            JOIN read_parquet($engagement) e
              ON  e.code_module       = w.code_module
              AND e.code_presentation = w.code_presentation
              AND e.id_student        = w.id_student
              AND e.course_day        < w.date_unregistration
            GROUP BY 1, 2
            """,
            {"engagement": path},
        ).fetchdf()
    finally:
        connection.close()

    frame = frame[frame["weeks_before_event"].between(0, weeks_before - 1)]
    return (
        frame.groupby("weeks_before_event")["clicks"]
        .agg(students="size", median_clicks="median", mean_clicks="mean")
        .round(2)
        .sort_index()
    )


# ---------------------------------------------------------------------------
# Discriminative power screening
# ---------------------------------------------------------------------------

CANDIDATE_FEATURES = [
    "clicks_7d",
    "clicks_14d",
    "clicks_30d",
    "clicks_prev_30d",
    "clicks_total",
    "clicks_delta_30d",
    "clicks_ratio_30d",
    "active_days_30d",
    "active_days_total",
    "max_resources_30d",
    "days_since_last_activity",
]


def discriminative_power(
    windows: pd.DataFrame,
    candidates: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank candidate features by univariate ROC-AUC and mutual information.

    ROC-AUC here is a *screening* statistic on a single column, not a model
    result. It answers "does this column carry signal at all", which is what
    should drive the Phase 4 allowlist. It says nothing about performance in
    combination, and a strong univariate score can still add nothing once
    correlated features are present.

    ``days_since_last_activity`` is NaN for never-active students; those rows
    are dropped for that feature only, and the retained count is reported so a
    high score on a small subset is visible rather than misleading.
    """
    candidates = candidates or CANDIDATE_FEATURES
    rows = []
    for feature in candidates:
        column = windows[feature]
        mask = column.notna()
        values = column[mask].to_numpy(dtype=float).reshape(-1, 1)
        labels = windows.loc[mask, "label"].to_numpy()

        if labels.min() == labels.max():
            continue

        auc = roc_auc_score(labels, values.ravel())
        mutual_information = mutual_info_classif(
            values, labels, discrete_features=False, random_state=random_state
        )[0]
        rows.append(
            {
                "feature": feature,
                # Reported as distance from 0.5 so that a strongly *negative*
                # association ranks as informative rather than as poor.
                "auc": round(float(auc), 4),
                "abs_auc_lift": round(abs(float(auc) - 0.5), 4),
                "mutual_information": round(float(mutual_information), 5),
                "coverage": round(float(mask.mean()), 4),
            }
        )
    columns = ["feature", "auc", "abs_auc_lift", "mutual_information", "coverage"]
    if not rows:
        # A single-class subset yields no scorable feature. Return the empty
        # frame with its schema intact so callers can still index columns.
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("abs_auc_lift", ascending=False)
        .reset_index(drop=True)
    )


def discriminative_power_by_checkpoint(windows: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Univariate AUC for one feature at each checkpoint separately.

    A feature that only works late is much less useful than the aggregate
    number suggests, so this breakdown decides whether aggregate screening
    results can be trusted.
    """
    rows = []
    for checkpoint, group in windows.groupby("checkpoint_day"):
        mask = group[feature].notna()
        labels = group.loc[mask, "label"]
        if labels.nunique() < 2:
            continue
        rows.append(
            {
                "checkpoint_day": checkpoint,
                "rows": int(mask.sum()),
                "positive_rate": round(float(labels.mean()), 4),
                "auc": round(float(roc_auc_score(labels, group.loc[mask, feature])), 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subgroup base rates, for the Phase 12 fairness audit
# ---------------------------------------------------------------------------

PROTECTED_ATTRIBUTES = ["gender", "age_band", "imd_band", "disability"]


def subgroup_base_rates(windows: pd.DataFrame, attributes: list[str] | None = None) -> pd.DataFrame:
    """Positive rate by protected attribute, with a Wilson interval.

    Establishing *base rates* before modelling matters: a subgroup with a
    genuinely higher withdrawal rate will receive more alerts from a
    well-calibrated model, and that is not by itself unfairness. Without these
    numbers recorded in advance, any later disparity is uninterpretable.

    These attributes are read here for auditing only. They are excluded from
    the feature matrix — see docs/ETHICS.md.
    """
    attributes = attributes or PROTECTED_ATTRIBUTES
    info = load_student_info()[[*KEYS, *attributes]]
    merged = windows.merge(info, on=KEYS, how="left", validate="many_to_one")

    rows = []
    for attribute in attributes:
        grouped = merged.groupby(attribute, dropna=False)["label"].agg(["size", "sum", "mean"])
        for value, record in grouped.iterrows():
            n = int(record["size"])
            successes = int(record["sum"])
            low, high = _wilson_interval(successes, n)
            rows.append(
                {
                    "attribute": attribute,
                    "value": "missing" if pd.isna(value) else value,
                    "rows": n,
                    "positives": successes,
                    "rate": round(float(record["mean"]), 4),
                    "ci_low": round(low, 4),
                    "ci_high": round(high, 4),
                }
            )
    return pd.DataFrame(rows)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because positive rates here are
    around 3%, where the normal interval is unreliable and can extend below
    zero.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denominator = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    spread = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return ((centre - spread) / denominator, (centre + spread) / denominator)


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------


def trivial_rule_baselines(windows: pd.DataFrame) -> pd.DataFrame:
    """Score simple single-rule heuristics.

    Any model must beat these to justify its complexity. Recording them before
    training removes the temptation to skip the comparison afterwards.
    """
    rules = {
        "no clicks in last 30d": windows["clicks_30d"] == 0,
        "no clicks in last 14d": windows["clicks_14d"] == 0,
        "no clicks in last 7d": windows["clicks_7d"] == 0,
        "never active at all": ~windows["ever_active"],
        "engagement halved vs prev 30d": windows["clicks_ratio_30d"] < 0.5,
    }
    labels = windows["label"].to_numpy()
    positives = labels.sum()

    rows = []
    for name, flagged in rules.items():
        predicted = flagged.to_numpy()
        true_positive = int((predicted & (labels == 1)).sum())
        flagged_total = int(predicted.sum())
        precision = true_positive / flagged_total if flagged_total else float("nan")
        recall = true_positive / positives if positives else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "rule": name,
                "flagged": flagged_total,
                "flagged_share": round(flagged_total / len(windows), 4),
                "precision": round(precision, 4) if flagged_total else None,
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
