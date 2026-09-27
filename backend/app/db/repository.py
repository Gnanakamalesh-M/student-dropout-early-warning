"""SQLAlchemy implementation of the repository interface.

This satisfies the same ``StudentRepository`` Protocol as the parquet-backed
version from Phase 8, so no router or service changes. That was the point of the
seam — if adopting the database had required editing endpoints, the seam was in
the wrong place.

The substantive difference is not the storage engine. It is that predictions are
now **read** rather than recomputed: a trajectory shows the scores that were
actually produced at the time, under the model version recorded against each row.
Recomputing history under a newer model would silently rewrite what staff saw.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Subquery

from backend.app.db import models
from backend.app.repository import StoredAlert, StoredIntervention

# Feature columns the profile view reads back off the prediction's stored row.
# The DB holds events, not built features, so profile numbers are rebuilt from
# events on demand by the feature builder rather than duplicated here.
PROFILE_SOURCE = "database"


class SqlRepository:
    """Repository over PostgreSQL (or SQLite in tests)."""

    def __init__(self, session: Session, model_version: str) -> None:
        self.session = session
        self.model_version = model_version
        self._model_version_row = self._active_model_version()

    # -- reference data ----------------------------------------------------

    def _active_model_version(self) -> models.ModelVersion:
        row = self.session.scalar(
            select(models.ModelVersion).where(models.ModelVersion.version == self.model_version)
        )
        if row is None:
            raise LookupError(
                f"model version {self.model_version!r} is not registered in the database; "
                "run scripts/seed_db.py"
            )
        return row

    @property
    def is_calibrated(self) -> bool:
        row = self.session.scalar(
            select(models.ThresholdConfig).where(models.ThresholdConfig.is_active.is_(True))
        )
        return bool(row.calibrated) if row else False

    @property
    def alert_budget(self) -> float:
        row = self.session.scalar(
            select(models.ThresholdConfig).where(models.ThresholdConfig.is_active.is_(True))
        )
        return float(row.alert_budget) if row and row.alert_budget else 0.05

    # -- core queries ------------------------------------------------------

    def _latest_prediction_subquery(self) -> Subquery:
        """Per student, the highest checkpoint scored under the active model."""
        return (
            select(
                models.Prediction.student_id.label("student_id"),
                func.max(models.Prediction.checkpoint_day).label("checkpoint_day"),
            )
            .where(models.Prediction.model_version_id == self._model_version_row.id)
            .group_by(models.Prediction.student_id)
            .subquery()
        )

    def latest_rows(self) -> pd.DataFrame:
        latest = self._latest_prediction_subquery()
        statement = (
            select(
                models.Student.student_code,
                models.Student.code_module,
                models.Student.code_presentation,
                models.Prediction.checkpoint_day,
                models.Prediction.probability,
                models.Prediction.band,
                models.Prediction.id.label("prediction_id"),
            )
            .join(models.Prediction, models.Prediction.student_id == models.Student.id)
            .join(
                latest,
                (latest.c.student_id == models.Prediction.student_id)
                & (latest.c.checkpoint_day == models.Prediction.checkpoint_day),
            )
            .where(models.Prediction.model_version_id == self._model_version_row.id)
        )
        frame = pd.DataFrame(self.session.execute(statement).mappings().all())
        if frame.empty:
            return frame
        return self._attach_direction(frame)

    def _attach_direction(self, latest: pd.DataFrame) -> pd.DataFrame:
        """Add risk direction by comparing each student's last two predictions.

        ``unknown`` where only one prediction exists, rather than defaulting to
        ``stable`` — that would claim a trend nobody has observed.
        """
        history = self.trajectories()
        if history.empty:
            latest["risk_direction"] = "unknown"
            return latest

        ordered = history.sort_values(["student_code", "checkpoint_day"])
        deltas = ordered.groupby("student_code")["probability"].diff()
        ordered = ordered.assign(delta=deltas)
        last = ordered.groupby("student_code").tail(1).set_index("student_code")["delta"]

        def classify(code: str) -> str:
            delta = last.get(code)
            if delta is None or pd.isna(delta):
                return "unknown"
            if delta > 0.005:
                return "worsening"
            if delta < -0.005:
                return "improving"
            return "stable"

        latest["risk_direction"] = [classify(code) for code in latest["student_code"]]
        return latest

    def trajectories(self) -> pd.DataFrame:
        """Every prediction under the active model, for trend computation."""
        statement = (
            select(
                models.Student.student_code,
                models.Prediction.checkpoint_day,
                models.Prediction.probability,
                models.Prediction.band,
            )
            .join(models.Prediction, models.Prediction.student_id == models.Student.id)
            .where(models.Prediction.model_version_id == self._model_version_row.id)
            .order_by(models.Student.student_code, models.Prediction.checkpoint_day)
        )
        return pd.DataFrame(self.session.execute(statement).mappings().all())

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
        if rows.empty:
            return rows, 0
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
        statement = (
            select(
                models.Student.student_code,
                models.Student.code_module,
                models.Student.code_presentation,
                models.Prediction.checkpoint_day,
                models.Prediction.probability,
                models.Prediction.band,
            )
            .join(models.Prediction, models.Prediction.student_id == models.Student.id)
            .where(
                models.Student.student_code == code,
                models.Prediction.model_version_id == self._model_version_row.id,
            )
            .order_by(models.Prediction.checkpoint_day)
        )
        frame = pd.DataFrame(self.session.execute(statement).mappings().all())
        if frame.empty:
            return frame
        return self._attach_direction(frame)

    def student_id(self, code: str) -> int | None:
        return self.session.scalar(
            select(models.Student.id).where(models.Student.student_code == code)
        )

    # -- feature building from stored events --------------------------------

    def feature_row(self, code: str, checkpoint_day: int | None = None) -> pd.DataFrame:
        """Rebuild the feature vector for one student from stored events.

        This uses the **same** ``FeatureBuilder`` as training, reading the event
        tables rather than the processed parquet. That is the ADR-0004 claim made
        concrete: there is one feature implementation, and it works off whatever
        holds the events.

        Returns an empty frame when the student or checkpoint is unknown, which
        the router turns into a 404.
        """
        from dropout_ews.features.builder import FeatureBuilder

        student = self.session.scalar(
            select(models.Student).where(models.Student.student_code == code)
        )
        if student is None:
            return pd.DataFrame()

        if checkpoint_day is None:
            checkpoint_day = self.session.scalar(
                select(func.max(models.Prediction.checkpoint_day)).where(
                    models.Prediction.student_id == student.id,
                    models.Prediction.model_version_id == self._model_version_row.id,
                )
            )
        if checkpoint_day is None:
            return pd.DataFrame()

        population = pd.DataFrame(
            [
                {
                    "code_module": student.code_module,
                    "code_presentation": student.code_presentation,
                    "id_student": student.source_student_id,
                    "checkpoint_day": int(checkpoint_day),
                    "module_presentation_length": student.module_presentation_length,
                }
            ]
        )

        engagement = pd.DataFrame(
            self.session.execute(
                select(
                    models.EngagementRecord.course_day,
                    models.EngagementRecord.clicks,
                    models.EngagementRecord.distinct_resources,
                ).where(models.EngagementRecord.student_id == student.id)
            )
            .mappings()
            .all()
        )
        if engagement.empty:
            engagement = pd.DataFrame(
                columns=["course_day", "clicks", "distinct_resources"]
            ).astype({"course_day": "int64", "clicks": "int64", "distinct_resources": "int64"})
        engagement = engagement.assign(
            code_module=student.code_module,
            code_presentation=student.code_presentation,
            id_student=student.source_student_id,
        )

        records = pd.DataFrame(
            self.session.execute(
                select(
                    models.AssessmentRecord.assessment_id,
                    models.AssessmentRecord.assessment_type,
                    models.AssessmentRecord.due_day,
                    models.AssessmentRecord.weight,
                    models.AssessmentRecord.date_submitted,
                    models.AssessmentRecord.score,
                    models.AssessmentRecord.is_banked,
                ).where(models.AssessmentRecord.student_id == student.id)
            )
            .mappings()
            .all()
        )
        if records.empty:
            assessments = pd.DataFrame(
                columns=[
                    "code_module",
                    "code_presentation",
                    "id_assessment",
                    "assessment_type",
                    "date",
                    "weight",
                ]
            )
            submissions = pd.DataFrame(
                columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]
            )
        else:
            assessments = pd.DataFrame(
                {
                    "code_module": student.code_module,
                    "code_presentation": student.code_presentation,
                    "id_assessment": records["assessment_id"],
                    "assessment_type": records["assessment_type"],
                    # Float, because the builder treats a NULL due date as the
                    # end of the presentation and needs a nullable dtype.
                    "date": records["due_day"].astype("float64"),
                    "weight": records["weight"],
                }
            )
            submitted = records.dropna(subset=["date_submitted"])
            submissions = pd.DataFrame(
                {
                    "id_assessment": submitted["assessment_id"],
                    "id_student": student.source_student_id,
                    "date_submitted": submitted["date_submitted"].astype("int64"),
                    "is_banked": submitted["is_banked"].astype("int64"),
                    "score": submitted["score"],
                }
            )

        static = pd.DataFrame(
            [
                {
                    "code_module": student.code_module,
                    "code_presentation": student.code_presentation,
                    "id_student": student.source_student_id,
                    # Enrolment context is not stored as events, so the values the
                    # model saw at scoring time are not reconstructable here. NaN is
                    # honest; the imputer handles it and the missingness indicator
                    # records it.
                    "num_of_prev_attempts": float("nan"),
                    "studied_credits": float("nan"),
                    "date_registration": float("nan"),
                }
            ]
        )

        built = FeatureBuilder().build(
            population,
            engagement=engagement,
            assessments=assessments,
            student_assessment=submissions,
            static=static,
        )
        frame = built.frame.copy()
        frame["code_presentation"] = student.code_presentation
        return frame.drop(columns=["row_id"], errors="ignore")

    # -- alerts ------------------------------------------------------------

    def alerts(
        self, acknowledged: bool | None = None, severity: str | None = None
    ) -> list[StoredAlert]:
        statement = (
            select(models.Alert, models.Student.student_code)
            .join(models.Student, models.Student.id == models.Alert.student_id)
            .order_by(models.Alert.probability.desc())
        )
        if acknowledged is not None:
            statement = statement.where(models.Alert.acknowledged.is_(acknowledged))
        if severity:
            statement = statement.where(models.Alert.severity == severity)

        return [
            StoredAlert(
                id=alert.id,
                student_code=code,
                checkpoint_day=alert.checkpoint_day,
                reason=alert.reason,
                severity=alert.severity,
                probability=alert.probability,
                previous_probability=alert.previous_probability,
                band=alert.band,
                acknowledged=alert.acknowledged,
                created_at=alert.created_at,
            )
            for alert, code in self.session.execute(statement).all()
        ]

    def acknowledge_alert(self, alert_id: int, username: str) -> StoredAlert:
        alert = self.session.get(models.Alert, alert_id)
        if alert is None:
            raise KeyError(alert_id)
        alert.acknowledged = True
        alert.acknowledged_by = username
        alert.acknowledged_at = datetime.now(timezone.utc)
        self.session.flush()
        code = self.session.scalar(
            select(models.Student.student_code).where(models.Student.id == alert.student_id)
        )
        return StoredAlert(
            id=alert.id,
            student_code=str(code),
            checkpoint_day=alert.checkpoint_day,
            reason=alert.reason,
            severity=alert.severity,
            probability=alert.probability,
            previous_probability=alert.previous_probability,
            band=alert.band,
            acknowledged=alert.acknowledged,
            created_at=alert.created_at,
        )

    # -- interventions -----------------------------------------------------

    def interventions(self, code: str) -> list[StoredIntervention]:
        statement = (
            select(models.Intervention)
            .join(models.Student, models.Student.id == models.Intervention.student_id)
            .where(models.Student.student_code == code)
            .order_by(models.Intervention.created_at)
        )
        return [
            StoredIntervention(
                id=row.id,
                student_code=code,
                intervention_key=row.intervention_key,
                title=row.title,
                status=row.status,
                assigned_to=row.assigned_to,
                assigned_by=row.assigned_by,
                note=row.note,
                created_at=row.created_at,
            )
            for row in self.session.scalars(statement).all()
        ]

    def add_intervention(
        self,
        code: str,
        intervention_key: str,
        title: str,
        assigned_to: str,
        assigned_by: str,
        note: str | None,
    ) -> StoredIntervention:
        student_id = self.student_id(code)
        if student_id is None:
            raise KeyError(code)
        row = models.Intervention(
            student_id=student_id,
            intervention_key=intervention_key,
            title=title,
            # Assignment is the human act that moves it beyond `recommended`.
            status="assigned",
            assigned_to=assigned_to,
            assigned_by=assigned_by,
            note=note,
        )
        self.session.add(row)
        self.session.flush()
        return StoredIntervention(
            id=row.id,
            student_code=code,
            intervention_key=row.intervention_key,
            title=row.title,
            status=row.status,
            assigned_to=row.assigned_to,
            assigned_by=row.assigned_by,
            note=row.note,
            created_at=row.created_at,
        )

    # -- audit -------------------------------------------------------------

    def record_audit(
        self, request_id: str, actor: str, action: str, resource: str, status_code: int
    ) -> None:
        """Append an audit entry. There is deliberately no update or delete."""
        self.session.add(
            models.AuditLogEntry(
                request_id=request_id,
                actor=actor,
                action=action,
                resource=resource,
                status_code=status_code,
            )
        )

    def audit_entries(
        self, actor: str | None = None, limit: int = 100
    ) -> list[models.AuditLogEntry]:
        statement = (
            select(models.AuditLogEntry).order_by(models.AuditLogEntry.id.desc()).limit(limit)
        )
        if actor:
            statement = statement.where(models.AuditLogEntry.actor == actor)
        return list(self.session.scalars(statement).all())

    # -- aggregates --------------------------------------------------------

    def band_counts(self) -> list[tuple[str, int]]:
        rows = self.latest_rows()
        counts = rows["band"].value_counts().to_dict() if not rows.empty else {}
        return [(band, int(counts.get(band, 0))) for band in models.BAND_KEYS]

    def worsening_count(self) -> int:
        rows = self.latest_rows()
        if rows.empty:
            return 0
        return int((rows["risk_direction"] == "worsening").sum())

    @property
    def cohort(self) -> pd.DataFrame:
        return self.trajectories()
