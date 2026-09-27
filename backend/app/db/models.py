"""SQLAlchemy models. Target is PostgreSQL 16; SQLite is supported for tests.

Three schema decisions are worth reading before the code.

**There is no ``prediction_explanations`` table.** The original plan had one, on
the grounds that request-time SHAP over a cohort would be too slow. Phase 8
measured it: 28.8 ms for one student, 1.55 ms per row in batch. The justification
was simply wrong, so the table is dropped rather than carried as complexity
nobody needs. Explanations are computed on demand. If a future requirement is
*audit reproducibility* — "what exactly did we show this counsellor in March" —
that is a different and legitimate reason to store them, and it would need its own
decision.

**``risk_history`` is a view, not a table.** ``predictions`` is already unique on
(student, checkpoint, model version), so every row a history table would hold is
already there. Duplicating it would create two places for the same fact and a way
for them to disagree. The view gives the trajectory query a stable name without
that risk.

**Demographics live in their own table.** ``student_demographics`` is read only by
the offline fairness audit and is never joined into the feature pipeline. Data
minimisation is structural here rather than a policy note: the separation is what
makes "the model cannot see this" a checkable claim (ETHICS.md).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# JSONB on PostgreSQL, plain JSON elsewhere. Declared once so every model gets
# the indexable Postgres type in production while tests can still run on SQLite.
JSONType = JSON().with_variant(postgresql.JSONB(), "postgresql")

BAND_KEYS = ("low", "medium", "high", "critical")
ALERT_REASONS = ("band_escalated", "rapid_increase", "sustained_increase")
INTERVENTION_STATUSES = ("recommended", "assigned", "in_progress", "completed")
ROLES = ("admin", "counsellor", "analyst")
DATA_SOURCES = ("real_oulad", "synthetic")


class Base(DeclarativeBase):
    pass


def _timestamp() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Student(Base):
    """One row per student-presentation.

    No name, no contact detail — none exists in the source data. ``student_code``
    is the opaque identifier the API exposes; ``source_student_id`` is the OULAD
    ``id_student`` and stays server-side.
    """

    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_code: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    code_module: Mapped[str] = mapped_column(String(16), nullable=False)
    code_presentation: Mapped[str] = mapped_column(String(16), nullable=False)
    source_student_id: Mapped[int] = mapped_column(Integer, nullable=False)
    data_source: Mapped[str] = mapped_column(String(16), nullable=False)
    module_presentation_length: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = _timestamp()

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )
    alerts: Mapped[list[Alert]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )
    interventions: Mapped[list[Intervention]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )
    demographics: Mapped[StudentDemographics | None] = relationship(
        back_populates="student", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        # The natural key. A student appears in several module-presentations, so
        # id_student alone is not unique (Phase 2 finding).
        UniqueConstraint(
            "code_module", "code_presentation", "source_student_id", name="uq_student_enrolment"
        ),
        CheckConstraint(f"data_source IN {DATA_SOURCES}", name="ck_students_data_source"),
        Index("ix_students_cohort", "code_module", "code_presentation"),
    )


class StudentDemographics(Base):
    """Protected attributes, deliberately isolated.

    Read **only** by the offline fairness audit. Never joined into the feature
    pipeline, and not exposed by any API endpoint. Keeping these in a separate
    table is what turns "the model does not see these" from an assurance into
    something a reviewer can verify by reading the queries.
    """

    __tablename__ = "student_demographics"

    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), primary_key=True
    )
    gender: Mapped[str | None] = mapped_column(String(16))
    age_band: Mapped[str | None] = mapped_column(String(16))
    imd_band: Mapped[str | None] = mapped_column(String(16))
    disability: Mapped[str | None] = mapped_column(String(8))
    region: Mapped[str | None] = mapped_column(String(64))

    student: Mapped[Student] = relationship(back_populates="demographics")


class EngagementRecord(Base):
    """Student-day clickstream, aggregated from the raw resource-level rows."""

    __tablename__ = "engagement_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    course_day: Mapped[int] = mapped_column(Integer, nullable=False)
    clicks: Mapped[int] = mapped_column(Integer, nullable=False)
    distinct_resources: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("student_id", "course_day", name="uq_engagement_student_day"),
        # Feature building always reads a trailing window for one student, so the
        # composite index matches the access pattern rather than being generic.
        Index("ix_engagement_student_day", "student_id", "course_day"),
    )


class AssessmentRecord(Base):
    """One row per assessment a student was due, submitted or not.

    ``date_submitted`` NULL means due but not submitted — the backlog signal, so
    the absence has to be representable rather than implied by a missing row.
    """

    __tablename__ = "assessment_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    assessment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    assessment_type: Mapped[str] = mapped_column(String(8), nullable=False)
    due_day: Mapped[int] = mapped_column(Integer, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    date_submitted: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    is_banked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("student_id", "assessment_id", name="uq_assessment_student"),
        Index("ix_assessment_student_due", "student_id", "due_day"),
    )


class ModelVersion(Base):
    """A registered model artifact.

    Predictions reference this, so "which model produced this student's score" is
    answerable for any stored row — the question that becomes unanswerable the
    moment a second version exists without it.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    trained_at: Mapped[str] = mapped_column(String(64), nullable=False)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    package_version: Mapped[str | None] = mapped_column(String(32))
    n_features: Mapped[int] = mapped_column(Integer, nullable=False)
    train_data_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    imbalance_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    calibration_method: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    feature_names: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = _timestamp()

    predictions: Mapped[list[Prediction]] = relationship(back_populates="model_version")


class ThresholdConfig(Base):
    """A versioned set of risk bands.

    Bands are an institutional capacity choice, so they change independently of
    the model and every prediction records which set was in force when it was
    banded. ``calibrated`` mirrors the flag in ``thresholds.yaml``: a UI showing
    bands derived from placeholders must be able to say so.
    """

    __tablename__ = "threshold_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alert_budget: Mapped[float | None] = mapped_column(Float)
    bands: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = _timestamp()


class Prediction(Base):
    """A risk estimate for one student at one checkpoint under one model.

    Retained rather than recomputed, because a trajectory only means something if
    past scores are the ones that were actually shown. Recomputing history under
    a newer model would silently rewrite what staff saw.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    threshold_config_id: Mapped[int] = mapped_column(
        ForeignKey("threshold_configs.id"), nullable=False
    )
    checkpoint_day: Mapped[int] = mapped_column(Integer, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    band: Mapped[str] = mapped_column(String(16), nullable=False)
    features_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[int | None] = mapped_column(Integer)
    """Populated only once the outcome is observed, so monitoring can score past
    predictions. NULL means the horizon has not yet closed."""

    created_at: Mapped[datetime] = _timestamp()

    student: Mapped[Student] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")
    alerts: Mapped[list[Alert]] = relationship(back_populates="prediction")

    __table_args__ = (
        # Re-scoring the same student and checkpoint under the same model is an
        # idempotent overwrite, not a second row.
        UniqueConstraint(
            "student_id", "checkpoint_day", "model_version_id", name="uq_prediction_scope"
        ),
        CheckConstraint("probability >= 0 AND probability <= 1", name="ck_prediction_probability"),
        CheckConstraint(f"band IN {BAND_KEYS}", name="ck_prediction_band"),
        # The trajectory query.
        Index("ix_predictions_student_checkpoint", "student_id", "checkpoint_day"),
        # The dashboard only ever lists the upper bands, so a partial index keeps
        # it small. PostgreSQL only; SQLite ignores the predicate.
        Index(
            "ix_predictions_upper_bands",
            "band",
            "probability",
            postgresql_where=text("band IN ('high', 'critical')"),
        ),
    )


class Alert(Base):
    """A change in risk worth a human looking at.

    Alerts are about *change* as much as level: a student steady at 0.30 needs
    less attention than one who moved 0.10 to 0.30.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False
    )
    checkpoint_day: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    previous_probability: Mapped[float | None] = mapped_column(Float)
    band: Mapped[str] = mapped_column(String(16), nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _timestamp()

    student: Mapped[Student] = relationship(back_populates="alerts")
    prediction: Mapped[Prediction] = relationship(back_populates="alerts")

    __table_args__ = (
        UniqueConstraint("prediction_id", "reason", name="uq_alert_prediction_reason"),
        CheckConstraint(f"reason IN {ALERT_REASONS}", name="ck_alert_reason"),
        # The alert queue is always "open, most severe first".
        Index("ix_alerts_queue", "acknowledged", "severity", "probability"),
    )


class Intervention(Base):
    """A suggested support action, and its human-assigned state.

    Rows are created as ``recommended``. Moving beyond that requires a person,
    which is the constraint ADR-0005 exists to protect; the check constraint keeps
    the vocabulary closed so a new status cannot be invented at the call site.
    """

    __tablename__ = "interventions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    intervention_key: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="recommended")
    matched_factor: Mapped[str | None] = mapped_column(String(64))
    assigned_to: Mapped[str | None] = mapped_column(String(64))
    assigned_by: Mapped[str | None] = mapped_column(String(64))
    note: Mapped[str | None] = mapped_column(Text)
    outcome: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _timestamp()
    updated_at: Mapped[datetime] = _timestamp()

    student: Mapped[Student] = relationship(back_populates="interventions")

    __table_args__ = (
        CheckConstraint(f"status IN {INTERVENTION_STATUSES}", name="ck_intervention_status"),
        Index("ix_interventions_student", "student_id", "status"),
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = _timestamp()

    __table_args__ = (CheckConstraint(f"role IN {ROLES}", name="ck_user_role"),)


class AuditLogEntry(Base):
    """Who read which student, and when.

    ETHICS.md commits to answering that question. Append-only by convention: no
    update or delete path exists in the repository, because an audit trail that
    can be edited is not an audit trail.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    occurred_at: Mapped[datetime] = _timestamp()
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource: Mapped[str] = mapped_column(String(128), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_audit_actor_time", "actor", "occurred_at"),
        Index("ix_audit_resource_time", "resource", "occurred_at"),
    )


# ---------------------------------------------------------------------------
# risk_history view
# ---------------------------------------------------------------------------

# Created by the migration rather than declared as a model, because it holds no
# facts of its own. See the module docstring for why this is a view.
RISK_HISTORY_VIEW_SQL = """
CREATE VIEW risk_history AS
SELECT
    p.student_id,
    s.student_code,
    p.checkpoint_day,
    p.probability,
    p.band,
    p.model_version_id,
    mv.version AS model_version,
    p.created_at
FROM predictions p
JOIN students s ON s.id = p.student_id
JOIN model_versions mv ON mv.id = p.model_version_id
"""

DROP_RISK_HISTORY_VIEW_SQL = "DROP VIEW IF EXISTS risk_history"
