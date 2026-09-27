"""Data access, behind an interface Phase 9 will reimplement over PostgreSQL.

The service layer depends on :class:`StudentRepository` (a Protocol), not on this
implementation. That seam exists so Phase 9 swaps the storage without touching a
router or a service, and so the API tests do not need a database.

The current implementation reads the processed feature parquet and scores on
demand. Two things justify that rather than it being a stopgap:

* Explanation latency was measured at 28.8 ms for a single student and 1.55 ms
  per row in batch, so **on-demand scoring is fast enough** and persisting
  explanations for performance is unnecessary. My Phase 1 plan asserted the
  opposite; the measurement says otherwise.
* Persisting *predictions* is still needed, but for a different reason: a risk
  trajectory is only meaningful if historical scores are retained rather than
  recomputed under a newer model. That is a semantics requirement, not a latency
  one, and it lands in Phase 9.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from dropout_ews.config.settings import DATA_DIR, load_feature_config, load_threshold_config
from dropout_ews.data.checkpoints import build_checkpoint_rows
from dropout_ews.evaluation.splits import make_temporal_split

FEATURES_PARQUET = DATA_DIR / "processed" / "features.parquet"
KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
DROP_COLUMNS = ("label", "date_unregistration", "split", "row_id")

# Columns grouped for the student profile view.
ENGAGEMENT_FIELDS = (
    "clicks_7d",
    "clicks_28d",
    "active_days_7d",
    "active_days_28d",
    "days_since_last_activity",
    "inactive_weeks_streak",
    "clicks_vs_baseline_ratio",
)
ASSESSMENT_FIELDS = (
    "assessments_due",
    "assessments_submitted",
    "assessments_missed",
    "assessments_failed",
    "assessments_late",
    "submission_rate",
    "mean_score",
    "min_score",
    "days_since_last_submission",
)
CONTEXT_FIELDS = ("checkpoint_day", "num_of_prev_attempts", "studied_credits", "date_registration")


def student_code(module: str, presentation: str, student_id: int) -> str:
    """Opaque, stable identifier for a student-presentation.

    Derived from the key rather than being the raw ``id_student``, so a URL or a
    screenshot does not expose the source identifier. Deterministic, so the same
    student always resolves to the same code.
    """
    digest = hashlib.sha256(f"{module}|{presentation}|{student_id}".encode()).hexdigest()
    return f"S-{digest[:12].upper()}"


@dataclass
class StoredIntervention:
    id: int
    student_code: str
    intervention_key: str
    title: str
    status: str
    assigned_to: str | None
    assigned_by: str | None
    note: str | None
    created_at: datetime


@dataclass
class StoredAlert:
    id: int
    student_code: str
    checkpoint_day: int
    reason: str
    severity: str
    probability: float
    previous_probability: float | None
    band: str
    acknowledged: bool
    created_at: datetime


class StudentRepository(Protocol):
    """What the service layer needs from storage."""

    def list_students(
        self,
        band: str | None = None,
        direction: str | None = None,
        module: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[pd.DataFrame, int]: ...

    def latest_rows(self) -> pd.DataFrame: ...

    def student_rows(self, code: str) -> pd.DataFrame: ...

    def feature_row(self, code: str, checkpoint_day: int | None = None) -> pd.DataFrame: ...

    def alerts(
        self, acknowledged: bool | None = None, severity: str | None = None
    ) -> list[StoredAlert]: ...

    def acknowledge_alert(self, alert_id: int, username: str) -> StoredAlert: ...

    def interventions(self, code: str) -> list[StoredIntervention]: ...

    def add_intervention(
        self,
        code: str,
        intervention_key: str,
        title: str,
        assigned_to: str,
        assigned_by: str,
        note: str | None,
    ) -> StoredIntervention: ...


class ParquetRepository:
    """Repository over the processed feature parquet, scoring on demand.

    Scoped to the **test presentation** on purpose: it is the cohort the model has
    never trained on, so the demo shows the system behaving as it would in
    deployment rather than reciting training data.
    """

    def __init__(self, model: object, model_version: str) -> None:
        self._model = model
        self.model_version = model_version
        self._thresholds = load_threshold_config()
        self._alert_rules = self._thresholds.alerts
        self._interventions: list[StoredIntervention] = []
        self._next_intervention_id = 1
        self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> None:
        frame = pd.read_parquet(FEATURES_PARQUET)
        population, _ = build_checkpoint_rows()
        frame.index = population.index
        frame["code_presentation"] = population["code_presentation"].to_numpy()
        split = make_temporal_split(population)

        cohort = frame.loc[split.test].copy()
        features = cohort.drop(columns=[c for c in DROP_COLUMNS if c in cohort.columns])
        cohort["probability"] = self._model.predict_proba(features)[:, 1]  # type: ignore[attr-defined]
        cohort["band"] = [self._thresholds.band_for(p).key for p in cohort["probability"]]
        cohort["student_code"] = [
            student_code(m, p, s)
            for m, p, s in zip(
                cohort["code_module"],
                cohort["code_presentation"],
                cohort["id_student"],
                strict=True,
            )
        ]
        cohort = cohort.sort_values(["student_code", "checkpoint_day"]).reset_index(drop=True)

        # Risk direction from the two most recent checkpoints. `unknown` where a
        # student has only one, rather than defaulting to `stable`, which would
        # claim a trend that has not been observed.
        delta = cohort.groupby("student_code")["probability"].diff()
        cohort["probability_delta"] = delta
        cohort["risk_direction"] = np.select(
            [delta.isna(), delta > 0.005, delta < -0.005],
            ["unknown", "worsening", "improving"],
            default="stable",
        )
        self._cohort = cohort
        self._features = features
        self._alerts = self._build_alerts(cohort)

    def _build_alerts(self, cohort: pd.DataFrame) -> list[StoredAlert]:
        """Derive alerts from the configured rules.

        Phase 9 persists these; here they are recomputed, which is acceptable
        because the rules are deterministic given the scores.
        """
        rules = self._alert_rules
        escalation_bands = set(rules.band_escalation.into_bands)
        alerts: list[StoredAlert] = []
        alert_id = 1
        now = datetime.now(timezone.utc)

        for code, group in cohort.groupby("student_code", sort=True):
            group = group.sort_values("checkpoint_day")
            previous_band: str | None = None
            previous_probability: float | None = None
            rising_run = 0

            for _, row in group.iterrows():
                band = str(row["band"])
                probability = float(row["probability"])
                delta = (
                    probability - previous_probability if previous_probability is not None else None
                )
                if delta is not None and delta > 0:
                    rising_run += 1
                else:
                    rising_run = 0

                reason: str | None = None
                if (
                    rules.band_escalation.enabled
                    and band in escalation_bands
                    and previous_band is not None
                    and previous_band != band
                ):
                    reason = "band_escalated"
                elif (
                    rules.rapid_increase.enabled
                    and delta is not None
                    and delta >= rules.rapid_increase.min_delta
                ):
                    reason = "rapid_increase"
                elif (
                    rules.sustained_increase.enabled
                    and rising_run >= rules.sustained_increase.consecutive_checkpoints
                ):
                    reason = "sustained_increase"

                if reason:
                    alerts.append(
                        StoredAlert(
                            id=alert_id,
                            student_code=str(code),
                            checkpoint_day=int(row["checkpoint_day"]),
                            reason=reason,
                            severity="critical" if band == "critical" else "high",
                            probability=probability,
                            previous_probability=previous_probability,
                            band=band,
                            acknowledged=False,
                            created_at=now,
                        )
                    )
                    alert_id += 1

                previous_band = band
                previous_probability = probability
        return alerts

    # -- queries -----------------------------------------------------------

    def latest_rows(self) -> pd.DataFrame:
        """One row per student: their most recent checkpoint."""
        return (
            self._cohort.sort_values("checkpoint_day")
            .groupby("student_code", as_index=False)
            .last()
        )

    def list_students(
        self,
        band: str | None = None,
        direction: str | None = None,
        module: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[pd.DataFrame, int]:
        rows = self.latest_rows()
        if band:
            rows = rows[rows["band"] == band]
        if direction:
            rows = rows[rows["risk_direction"] == direction]
        if module:
            rows = rows[rows["code_module"] == module]
        if search:
            rows = rows[rows["student_code"].str.contains(search.upper(), na=False)]

        rows = rows.sort_values("probability", ascending=False)
        total = len(rows)
        start = (page - 1) * page_size
        return rows.iloc[start : start + page_size], total

    def student_rows(self, code: str) -> pd.DataFrame:
        rows = self._cohort[self._cohort["student_code"] == code]
        return rows.sort_values("checkpoint_day")

    def feature_row(self, code: str, checkpoint_day: int | None = None) -> pd.DataFrame:
        rows = self.student_rows(code)
        if rows.empty:
            return rows
        if checkpoint_day is not None:
            rows = rows[rows["checkpoint_day"] == checkpoint_day]
            if rows.empty:
                return rows
        else:
            rows = rows.tail(1)
        return rows.drop(
            columns=[
                c
                for c in (
                    *DROP_COLUMNS,
                    "probability",
                    "band",
                    "student_code",
                    "risk_direction",
                    "probability_delta",
                )
                if c in rows.columns
            ]
        )

    def alerts(
        self, acknowledged: bool | None = None, severity: str | None = None
    ) -> list[StoredAlert]:
        result = self._alerts
        if acknowledged is not None:
            result = [a for a in result if a.acknowledged == acknowledged]
        if severity:
            result = [a for a in result if a.severity == severity]
        return sorted(result, key=lambda a: (-a.probability, a.student_code))

    def acknowledge_alert(self, alert_id: int, username: str) -> StoredAlert:
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return alert
        raise KeyError(alert_id)

    def interventions(self, code: str) -> list[StoredIntervention]:
        return [item for item in self._interventions if item.student_code == code]

    def add_intervention(
        self,
        code: str,
        intervention_key: str,
        title: str,
        assigned_to: str,
        assigned_by: str,
        note: str | None,
    ) -> StoredIntervention:
        record = StoredIntervention(
            id=self._next_intervention_id,
            student_code=code,
            intervention_key=intervention_key,
            title=title,
            # Assignment is the human act that moves it beyond `recommended`.
            status="assigned",
            assigned_to=assigned_to,
            assigned_by=assigned_by,
            note=note,
            created_at=datetime.now(timezone.utc),
        )
        self._interventions.append(record)
        self._next_intervention_id += 1
        return record

    # -- aggregates --------------------------------------------------------

    @property
    def cohort(self) -> pd.DataFrame:
        return self._cohort

    def band_counts(self) -> list[tuple[str, int]]:
        rows = self.latest_rows()
        counts = rows["band"].value_counts().to_dict()
        return [(band.key, int(counts.get(band.key, 0))) for band in self._thresholds.bands]

    def worsening_count(self) -> int:
        return int((self.latest_rows()["risk_direction"] == "worsening").sum())

    @property
    def alert_budget(self) -> float:
        return load_feature_config().evaluation.alert_budget

    @property
    def is_calibrated(self) -> bool:
        return self._thresholds.calibrated


@dataclass
class RepositoryState:
    """Application-wide singletons, populated at startup."""

    repository: ParquetRepository | None = None
    explainer: object | None = None
    metadata: object | None = None
    load_error: str | None = None
    notes: list[str] = field(default_factory=list)
