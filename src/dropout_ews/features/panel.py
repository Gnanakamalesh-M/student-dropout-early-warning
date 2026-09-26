"""As-of weekly panels from the raw event tables.

This is the only place that touches raw event data with a time boundary, so it
is the only place a temporal leak can originate. Every query here carries an
explicit ``<= checkpoint_day`` predicate, and the property test in
``tests/ml/test_as_of_property.py`` verifies that deleting all post-checkpoint
rows changes nothing.

Two panels are produced:

* **engagement** — one row per (population row, week-back index), from the
  student-day clickstream. Week 0 is ``[t-6, t]``, week 1 is ``[t-13, t-7]``,
  and so on, so weeks are disjoint and align to the checkpoint rather than to
  the calendar.
* **assessment** — one row per population row, aggregating submissions and
  their due dates up to the checkpoint.

Week-aligned windows are used rather than the exploratory 30/60-day windows
from Phase 3 because 28 days is exactly four whole weeks, which keeps the
weekly panel and the window sums consistent with each other. The EDA figures
used 30 days; the difference is immaterial to those findings.
"""

from __future__ import annotations

import duckdb
import pandas as pd

KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]

# Trailing weeks retained in the panel. Eight covers the trend and volatility
# windows with room to spare; deeper history adds little, because Phase 3
# showed engagement is flat beyond roughly five weeks before an event.
PANEL_WEEKS = 8

_ENGAGEMENT_PANEL_SQL = f"""
WITH activity AS (
    SELECT
        p.row_id,
        -- Weeks back from the checkpoint. course_day = t falls in week 0 and
        -- t-7 falls in week 1, so weeks are disjoint 7-day blocks.
        CAST(FLOOR((p.checkpoint_day - e.course_day) / 7.0) AS INTEGER) AS weeks_back,
        e.clicks,
        e.distinct_resources,
        e.course_day
    FROM population p
    JOIN engagement e
      ON  e.code_module       = p.code_module
      AND e.code_presentation = p.code_presentation
      AND e.id_student        = p.id_student
      -- THE AS-OF BOUNDARY. Removing this predicate is the single change that
      -- would invalidate every metric this project reports.
      AND e.course_day       <= p.checkpoint_day
)
SELECT
    row_id,
    weeks_back,
    SUM(clicks)::BIGINT                 AS clicks,
    COUNT(DISTINCT course_day)::INTEGER AS active_days,
    SUM(distinct_resources)::BIGINT     AS resource_views
FROM activity
WHERE weeks_back BETWEEN 0 AND {PANEL_WEEKS - 1}
GROUP BY 1, 2
"""

_ENGAGEMENT_TOTALS_SQL = """
SELECT
    p.row_id,
    COALESCE(SUM(e.clicks), 0)::BIGINT      AS clicks_all_time,
    COUNT(DISTINCT e.course_day)::INTEGER   AS active_days_all_time,
    MAX(e.course_day)                       AS last_active_day,
    MIN(e.course_day)                       AS first_active_day,
    -- Baseline: the first four weeks of the course, capped at the checkpoint so
    -- it never reads ahead. Supports decline-against-own-baseline, the trend
    -- form Phase 3 flagged as worth testing.
    COALESCE(SUM(CASE WHEN e.course_day BETWEEN 0 AND 27 THEN e.clicks END), 0)::BIGINT
        AS clicks_baseline_28d
FROM population p
LEFT JOIN engagement e
  ON  e.code_module       = p.code_module
  AND e.code_presentation = p.code_presentation
  AND e.id_student        = p.id_student
  AND e.course_day       <= p.checkpoint_day
GROUP BY 1
"""

# Assessments. `assessments.date` is NULL for some final exams, where the due
# day is taken to be the end of the presentation (recorded in schemas.py).
# `is_banked = 1` marks a score carried over from a previous attempt, which is
# an administrative transfer rather than a submission in this presentation, so
# it is counted separately and excluded from timeliness signals.
_ASSESSMENT_SQL = """
WITH due AS (
    SELECT
        p.row_id,
        a.id_assessment,
        COALESCE(a.date, p.module_presentation_length) AS due_day,
        a.weight
    FROM population p
    JOIN assessments a
      ON  a.code_module       = p.code_module
      AND a.code_presentation = p.code_presentation
      -- Only assessments already due at the checkpoint.
      AND COALESCE(a.date, p.module_presentation_length) <= p.checkpoint_day
),
submitted AS (
    SELECT
        p.row_id,
        s.date_submitted,
        s.score,
        s.is_banked,
        COALESCE(a.date, p.module_presentation_length) AS due_day
    FROM population p
    JOIN student_assessment s
      ON  s.id_student      = p.id_student
      -- AS-OF BOUNDARY: only submissions made at or before the checkpoint.
      AND s.date_submitted <= p.checkpoint_day
    JOIN assessments a
      ON  a.id_assessment     = s.id_assessment
      AND a.code_module       = p.code_module
      AND a.code_presentation = p.code_presentation
),
due_agg AS (
    SELECT row_id,
           COUNT(*)::INTEGER   AS assessments_due,
           SUM(weight)::DOUBLE AS weight_due
    FROM due GROUP BY 1
),
sub_agg AS (
    SELECT
        row_id,
        COUNT(*)::INTEGER                                       AS assessments_submitted,
        SUM(CASE WHEN is_banked = 1 THEN 1 ELSE 0 END)::INTEGER AS assessments_banked,
        AVG(score)                                              AS mean_score,
        MIN(score)                                              AS min_score,
        SUM(CASE WHEN score < 40 THEN 1 ELSE 0 END)::INTEGER    AS assessments_failed,
        MAX(date_submitted)                                     AS last_submission_day,
        SUM(CASE WHEN is_banked = 0 AND date_submitted > due_day THEN 1 ELSE 0 END)::INTEGER
            AS assessments_late,
        AVG(CASE WHEN is_banked = 0 THEN date_submitted - due_day END)
            AS mean_submission_lag
    FROM submitted GROUP BY 1
)
SELECT
    p.row_id,
    COALESCE(d.assessments_due, 0)       AS assessments_due,
    COALESCE(d.weight_due, 0.0)          AS weight_due,
    COALESCE(s.assessments_submitted, 0) AS assessments_submitted,
    COALESCE(s.assessments_banked, 0)    AS assessments_banked,
    COALESCE(s.assessments_failed, 0)    AS assessments_failed,
    COALESCE(s.assessments_late, 0)      AS assessments_late,
    s.mean_score,
    s.min_score,
    s.last_submission_day,
    s.mean_submission_lag
FROM population p
LEFT JOIN due_agg d USING (row_id)
LEFT JOIN sub_agg s USING (row_id)
"""

_EMPTY_ENGAGEMENT = pd.DataFrame(
    {
        "code_module": pd.Series(dtype="object"),
        "code_presentation": pd.Series(dtype="object"),
        "id_student": pd.Series(dtype="int64"),
        "course_day": pd.Series(dtype="int64"),
        "clicks": pd.Series(dtype="int64"),
        "distinct_resources": pd.Series(dtype="int64"),
    }
)


def engagement_panel(
    population: pd.DataFrame, engagement: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the weekly engagement panel and the all-time totals."""
    connection = duckdb.connect()
    try:
        connection.register("population", population)
        connection.register("engagement", engagement)
        panel = connection.execute(_ENGAGEMENT_PANEL_SQL).fetchdf()
        totals = connection.execute(_ENGAGEMENT_TOTALS_SQL).fetchdf()
    finally:
        connection.close()
    return panel, totals


def assessment_aggregates(
    population: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-row assessment aggregates as of each checkpoint."""
    connection = duckdb.connect()
    try:
        connection.register("population", population)
        connection.register("engagement", _EMPTY_ENGAGEMENT)
        connection.register("assessments", assessments)
        connection.register("student_assessment", student_assessment)
        return connection.execute(_ASSESSMENT_SQL).fetchdf()
    finally:
        connection.close()
