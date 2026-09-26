"""The feature builder. Imported by both the training pipeline and the API.

Reimplementing this logic in the backend is the classic cause of train/serve
skew, and it fails silently: the service returns plausible probabilities
computed from subtly different inputs. So there is exactly one implementation,
and a test asserts the offline and online paths produce identical vectors
(ADR-0004).

Two properties are deliberate and load-bearing:

**Stateless.** ``build`` fits nothing. Given the same inputs it returns the
same output, which is what makes the as-of property test meaningful. Anything
requiring fitted state — cohort z-scores in particular — lives in
:mod:`dropout_ews.preprocessing.cohort` as a scikit-learn transformer fitted
per fold inside a ``Pipeline`` (ADR-0003 control #4).

**Batch and single-row use the same code path.** Scoring one student at serving
time is just a population frame with one row. There is no separate
"online" branch that could drift.

Feature groups follow the Phase 3 findings (docs/EDA_FINDINGS.md):

* ``level`` and ``recency`` first — these screened strongest and were stable
  across every checkpoint.
* ``trend`` is included but flagged as a **hypothesis**: the crude single-window
  deltas screened worst and were unstable, so Phase 5 must show the properly
  specified versions earn their place before they stay in the allowlist.
* ``assessment`` features are new in this phase and were never screened; the
  173,912 submission rows carry timing and score signals that are plausibly
  complementary to clickstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dropout_ews.data.clickstream import load_student_day_engagement
from dropout_ews.data.loaders import load_assessments, load_student_assessment
from dropout_ews.features.panel import PANEL_WEEKS, assessment_aggregates, engagement_panel

KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]

# Trailing windows, in whole weeks, derived by summing panel weeks.
WINDOW_WEEKS = {"7d": 1, "14d": 2, "28d": 4, "56d": 8}

# Weeks used for the recent-trend slope. Four weeks is the shortest span that
# supports a stable least-squares fit while staying inside the 30-day horizon
# the model predicts over.
TREND_WEEKS = 4

# Passthrough columns from studentInfo. These are enrolment facts known at
# registration, not protected attributes: prior attempts and credit load are
# legitimate context. Demographics are deliberately absent — see ETHICS.md.
STATIC_COLUMNS = ["num_of_prev_attempts", "studied_credits", "date_registration"]


@dataclass(frozen=True)
class FeatureSet:
    """Built features plus the identifier columns needed to join them back."""

    frame: pd.DataFrame
    feature_names: list[str]

    def matrix(self) -> pd.DataFrame:
        """Just the model input columns, in a stable order."""
        return self.frame[self.feature_names]


class FeatureBuilder:
    """Build as-of features for a (student, checkpoint) population.

    Args:
        panel_weeks: trailing weeks of engagement history to use.
        trend_weeks: weeks over which the recent slope is fitted.
    """

    def __init__(self, panel_weeks: int = PANEL_WEEKS, trend_weeks: int = TREND_WEEKS) -> None:
        if trend_weeks < 2:
            raise ValueError("trend_weeks must be at least 2 to define a slope")
        if trend_weeks > panel_weeks:
            raise ValueError("trend_weeks cannot exceed panel_weeks")
        self.panel_weeks = panel_weeks
        self.trend_weeks = trend_weeks

    # -- public API --------------------------------------------------------

    def build(
        self,
        population: pd.DataFrame,
        engagement: pd.DataFrame | None = None,
        assessments: pd.DataFrame | None = None,
        student_assessment: pd.DataFrame | None = None,
        static: pd.DataFrame | None = None,
    ) -> FeatureSet:
        """Build features for every row of ``population``.

        Args:
            population: must carry ``KEY_COLUMNS``, ``checkpoint_day`` and
                ``module_presentation_length``. Extra columns are preserved.
            engagement: student-day clickstream. Loaded from the interim
                Parquet when omitted.
            assessments, student_assessment: loaded from the raw tables when
                omitted.
            static: enrolment context. Loaded from studentInfo when omitted.

        Returns:
            A :class:`FeatureSet` whose frame has one row per input row, in the
            input order.
        """
        self._validate_population(population)

        engagement = engagement if engagement is not None else load_student_day_engagement()
        assessments = assessments if assessments is not None else load_assessments()
        student_assessment = (
            student_assessment if student_assessment is not None else load_student_assessment()
        )

        # A stable surrogate key. Positional, so the output order matches the
        # input order regardless of how DuckDB returns groups.
        work = population.reset_index(drop=True).copy()
        work["row_id"] = np.arange(len(work), dtype="int64")

        panel, totals = engagement_panel(
            work[[*KEY_COLUMNS, "row_id", "checkpoint_day"]], engagement
        )
        assessment = assessment_aggregates(
            work[[*KEY_COLUMNS, "row_id", "checkpoint_day", "module_presentation_length"]],
            assessments,
            student_assessment,
        )

        weekly = self._widen_panel(panel, work["row_id"])
        features = work[["row_id", "checkpoint_day"]].copy()
        features = features.merge(totals, on="row_id", how="left", validate="one_to_one")
        features = features.merge(weekly, on="row_id", how="left", validate="one_to_one")
        features = features.merge(assessment, on="row_id", how="left", validate="one_to_one")

        frame = self._derive(features)
        frame = self._attach_static(frame, work, static)

        feature_names = [
            column for column in frame.columns if column not in {"row_id", *KEY_COLUMNS}
        ]
        # Identifiers are carried alongside so callers can join predictions
        # back, but they are not part of the matrix.
        out = pd.concat([work[[*KEY_COLUMNS, "row_id"]], frame.drop(columns=["row_id"])], axis=1)
        return FeatureSet(frame=out, feature_names=feature_names)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _validate_population(population: pd.DataFrame) -> None:
        """Check the population carries what the builder needs.

        ``label`` and ``date_unregistration`` may travel on the population
        frame, because the trainer needs them. The builder simply never reads
        them, and the feature allowlist in ``features.yaml`` is what stops them
        reaching a model.
        """
        required = {*KEY_COLUMNS, "checkpoint_day", "module_presentation_length"}
        missing = required - set(population.columns)
        if missing:
            raise ValueError(f"population is missing required columns: {sorted(missing)}")

    def _widen_panel(self, panel: pd.DataFrame, row_ids: pd.Series) -> pd.DataFrame:
        """Pivot the long weekly panel to one column per week, zero-filled.

        Zero is the correct fill here: absence of a clickstream row genuinely
        means no activity that week. This differs from
        ``days_since_last_activity``, where absence means *never* active and a
        zero would be a lie.
        """
        weeks = list(range(self.panel_weeks))
        wide = (
            panel.pivot(index="row_id", columns="weeks_back", values=["clicks", "active_days"])
            if not panel.empty
            else None
        )
        frame = pd.DataFrame({"row_id": row_ids.to_numpy()})
        for measure in ("clicks", "active_days"):
            for week in weeks:
                name = f"{measure}_w{week}"
                if wide is not None and (measure, week) in wide.columns:
                    series = wide[(measure, week)]
                    frame[name] = frame["row_id"].map(series).fillna(0).astype("float64")
                else:
                    frame[name] = 0.0
        return frame

    def _derive(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute the final feature columns from the panel and aggregates."""
        out = pd.DataFrame({"row_id": features["row_id"].to_numpy()})
        clicks = features[[f"clicks_w{w}" for w in range(self.panel_weeks)]].to_numpy()
        active = features[[f"active_days_w{w}" for w in range(self.panel_weeks)]].to_numpy()

        # --- level ---
        out["checkpoint_day"] = features["checkpoint_day"].to_numpy()
        for label, weeks in WINDOW_WEEKS.items():
            weeks = min(weeks, self.panel_weeks)
            out[f"clicks_{label}"] = clicks[:, :weeks].sum(axis=1)
            out[f"active_days_{label}"] = active[:, :weeks].sum(axis=1)
        out["clicks_all_time"] = features["clicks_all_time"].fillna(0).to_numpy()
        out["active_days_all_time"] = features["active_days_all_time"].fillna(0).to_numpy()

        # Intensity: 200 clicks over 10 days is a different pattern from 200 in
        # a single burst. Zero active days yields 0 rather than a division error.
        with np.errstate(invalid="ignore", divide="ignore"):
            out["clicks_per_active_day_28d"] = np.where(
                out["active_days_28d"] > 0,
                out["clicks_28d"] / out["active_days_28d"].replace(0, np.nan),
                0.0,
            )
        out["clicks_per_active_day_28d"] = np.nan_to_num(out["clicks_per_active_day_28d"], nan=0.0)

        # --- recency ---
        # NaN, not a sentinel: 3,095 real rows have no activity ever, and a
        # sentinel would be indistinguishable from "active recently".
        out["days_since_last_activity"] = (
            features["checkpoint_day"] - features["last_active_day"]
        ).astype("float64")
        out["ever_active"] = features["last_active_day"].notna().astype("int8")
        out["inactive_weeks_streak"] = self._leading_zero_streak(clicks)

        # --- trend (hypothesis; see module docstring) ---
        out["clicks_slope_4w"] = self._slope(clicks[:, : self.trend_weeks])
        recent = clicks[:, : self.trend_weeks].sum(axis=1)
        previous = clicks[:, self.trend_weeks : 2 * self.trend_weeks].sum(axis=1)
        out["clicks_delta_4w"] = recent - previous
        # +1 smoothing keeps this finite when the earlier window is empty.
        out["clicks_ratio_4w"] = (recent + 1.0) / (previous + 1.0)
        baseline = features["clicks_baseline_28d"].fillna(0).to_numpy()
        out["clicks_vs_baseline_ratio"] = (recent + 1.0) / (baseline + 1.0)
        out["declining_weeks_streak"] = self._declining_streak(clicks)

        # --- volatility ---
        out["clicks_weekly_std"] = clicks.std(axis=1, ddof=0)

        # --- assessment ---
        out["assessments_due"] = features["assessments_due"].fillna(0).to_numpy()
        out["assessments_submitted"] = features["assessments_submitted"].fillna(0).to_numpy()
        out["assessments_banked"] = features["assessments_banked"].fillna(0).to_numpy()
        out["assessments_failed"] = features["assessments_failed"].fillna(0).to_numpy()
        out["assessments_late"] = features["assessments_late"].fillna(0).to_numpy()
        # Missed = due but never submitted by the checkpoint. This is the
        # backlog analogue for OULAD.
        out["assessments_missed"] = np.maximum(
            0.0, out["assessments_due"] - out["assessments_submitted"]
        )
        due = out["assessments_due"].to_numpy()
        out["submission_rate"] = np.where(
            due > 0, out["assessments_submitted"].to_numpy() / np.maximum(due, 1), np.nan
        )
        # NaN where nothing is due yet, which is a real state at early
        # checkpoints and must not be confused with a zero submission rate.
        out["mean_score"] = features["mean_score"].to_numpy()
        out["min_score"] = features["min_score"].to_numpy()
        out["mean_submission_lag"] = features["mean_submission_lag"].to_numpy()
        out["days_since_last_submission"] = (
            features["checkpoint_day"] - features["last_submission_day"]
        ).astype("float64")
        out["has_submitted"] = features["last_submission_day"].notna().astype("int8")
        return out

    @staticmethod
    def _slope(values: np.ndarray) -> np.ndarray:
        """Least-squares slope per row, oldest-to-newest.

        Panel columns run newest-first, so they are reversed: a positive slope
        means engagement is *increasing* over time, which is the intuitive
        reading for anyone inspecting a SHAP plot later.
        """
        ordered = values[:, ::-1]
        n = ordered.shape[1]
        x = np.arange(n, dtype="float64")
        x_centred = x - x.mean()
        denominator = (x_centred**2).sum()
        y_centred = ordered - ordered.mean(axis=1, keepdims=True)
        slope: np.ndarray = (y_centred * x_centred).sum(axis=1) / denominator
        return slope

    @staticmethod
    def _leading_zero_streak(clicks: np.ndarray) -> np.ndarray:
        """Consecutive most-recent weeks with zero activity.

        The engagement analogue of consecutive absences.
        """
        nonzero = clicks > 0
        # Index of the first active week; the panel width when never active.
        first_active = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), clicks.shape[1])
        return first_active.astype("float64")

    @staticmethod
    def _declining_streak(clicks: np.ndarray) -> np.ndarray:
        """Consecutive most-recent weeks that fell against the week before.

        Counts strict decreases walking backwards from the newest week. Panel
        columns are newest-first, so ``clicks[:, i] < clicks[:, i + 1]`` means
        week *i* is lower than the week preceding it in time.
        """
        declines = clicks[:, :-1] < clicks[:, 1:]
        streak = np.zeros(clicks.shape[0], dtype="float64")
        still_running = np.ones(clicks.shape[0], dtype=bool)
        for index in range(declines.shape[1]):
            still_running &= declines[:, index]
            streak += still_running
        return streak

    def _attach_static(
        self,
        frame: pd.DataFrame,
        work: pd.DataFrame,
        static: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Join enrolment context, loading it when not supplied.

        ``date_registration`` lives in studentRegistration rather than
        studentInfo, so both tables are joined. Registering early is a
        plausible commitment signal and dropping it silently because it sat in
        the other table would have been an easy mistake to miss.
        """
        if static is None:
            from dropout_ews.data.loaders import (
                load_student_info,
                load_student_registration,
            )

            static = load_student_info().merge(
                load_student_registration()[[*KEY_COLUMNS, "date_registration"]],
                on=KEY_COLUMNS,
                how="left",
                validate="one_to_one",
            )
        available = [column for column in STATIC_COLUMNS if column in static.columns]
        if not available:
            return frame
        joined = work[[*KEY_COLUMNS, "row_id"]].merge(
            static[[*KEY_COLUMNS, *available]].drop_duplicates(subset=KEY_COLUMNS),
            on=KEY_COLUMNS,
            how="left",
            validate="many_to_one",
        )
        return frame.merge(
            joined[["row_id", *available]], on="row_id", how="left", validate="one_to_one"
        )
