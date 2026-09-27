"""Database tests: schema constraints, migrations, and the repository.

**What these verify and what they do not.** They run against SQLite by default,
because no PostgreSQL was available on the development machine. SQLAlchemy makes
the ORM behaviour identical, so constraint logic, query results, index
declarations and the migration round trip are all genuinely exercised. What SQLite
does *not* verify:

* JSONB behaviour and operators (SQLite gets plain JSON via ``with_variant``);
* the partial index predicate on ``predictions`` (SQLite ignores
  ``postgresql_where``);
* server-side default and timezone semantics;
* PostgreSQL's stricter type coercion — SQLite is permissive, so a type error
  could pass here and fail in production.

Setting ``TEST_DATABASE_URL`` to a PostgreSQL DSN reruns the whole module against
the real engine, which is what the CI service container does. Being explicit about
the gap is the point: claiming these are "PostgreSQL tests" would be false.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from backend.app.db import models
from backend.app.db.repository import SqlRepository
from backend.app.db.session import enforce_sqlite_foreign_keys
from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.integration

# A real PostgreSQL DSN here reruns everything against the production engine.
POSTGRES_URL = os.environ.get("TEST_DATABASE_URL")
IS_POSTGRES = bool(POSTGRES_URL and POSTGRES_URL.startswith("postgres"))


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    url = POSTGRES_URL or "sqlite://"
    created = create_engine(
        url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
    )
    # Without this, SQLite ignores every foreign key and ON DELETE CASCADE, and
    # the referential-integrity tests below would pass without testing anything.
    enforce_sqlite_foreign_keys(created)
    if IS_POSTGRES:
        with created.begin() as connection:
            connection.exec_driver_sql(models.DROP_RISK_HISTORY_VIEW_SQL)
        models.Base.metadata.drop_all(created)
    models.Base.metadata.create_all(created)
    with created.begin() as connection:
        connection.exec_driver_sql(models.DROP_RISK_HISTORY_VIEW_SQL)
        connection.exec_driver_sql(models.RISK_HISTORY_VIEW_SQL)
    yield created
    created.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session rolled back after each test, so tests do not see each other."""
    connection = engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, expire_on_commit=False)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


# ---------------------------------------------------------------------------
# Fixtures that build a small, complete world
# ---------------------------------------------------------------------------


def _model_version(session: Session, version: str = "xgboost-test") -> models.ModelVersion:
    row = models.ModelVersion(
        version=version,
        model_type="xgboost",
        trained_at="2026-09-27T00:00:00Z",
        git_commit="abc123",
        package_version="0.1.0",
        n_features=37,
        train_data_hash="deadbeef",
        imbalance_strategy="none",
        calibration_method="isotonic",
        metrics={"test_pr_auc": 0.0885},
        feature_names=["clicks_7d"],
        is_active=True,
    )
    session.add(row)
    session.flush()
    return row


def _threshold_config(session: Session) -> models.ThresholdConfig:
    row = models.ThresholdConfig(
        version=2,
        name="budget-calibrated-v1",
        calibrated=True,
        alert_budget=0.05,
        bands=[
            {"key": "low", "min_probability": 0.0},
            {"key": "medium", "min_probability": 0.03},
            {"key": "high", "min_probability": 0.06},
            {"key": "critical", "min_probability": 0.09},
        ],
        is_active=True,
    )
    session.add(row)
    session.flush()
    return row


def _student(session: Session, code: str, student_id: int, module: str = "AAA") -> models.Student:
    row = models.Student(
        student_code=code,
        code_module=module,
        code_presentation="2014J",
        source_student_id=student_id,
        data_source="real_oulad",
        module_presentation_length=269,
    )
    session.add(row)
    session.flush()
    return row


def _prediction(
    session: Session,
    student: models.Student,
    model_version: models.ModelVersion,
    thresholds: models.ThresholdConfig,
    checkpoint_day: int,
    probability: float,
    band: str,
) -> models.Prediction:
    row = models.Prediction(
        student_id=student.id,
        model_version_id=model_version.id,
        threshold_config_id=thresholds.id,
        checkpoint_day=checkpoint_day,
        horizon_days=30,
        probability=probability,
        band=band,
        features_hash="abc",
    )
    session.add(row)
    session.flush()
    return row


@pytest.fixture
def world(session: Session):
    """Two students with trajectories: one worsening, one improving."""
    model_version = _model_version(session)
    thresholds = _threshold_config(session)
    rising = _student(session, "S-RISING", 1001)
    falling = _student(session, "S-FALLING", 1002)

    for day, probability, band in ((30, 0.02, "low"), (60, 0.05, "medium"), (90, 0.11, "critical")):
        _prediction(session, rising, model_version, thresholds, day, probability, band)
    for day, probability, band in ((30, 0.10, "critical"), (60, 0.07, "high"), (90, 0.02, "low")):
        _prediction(session, falling, model_version, thresholds, day, probability, band)
    session.flush()
    return {
        "model_version": model_version,
        "thresholds": thresholds,
        "rising": rising,
        "falling": falling,
    }


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


def test_foreign_keys_are_enforced(engine: Engine) -> None:
    """Guard the guard.

    SQLite disables foreign keys by default, which made every cascade and
    referential-integrity test below vacuous until the pragma was set. If this
    regresses, those tests go quiet rather than failing, so it is asserted
    directly.
    """
    with engine.connect() as connection:
        if engine.dialect.name == "sqlite":
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        else:
            assert engine.dialect.name == "postgresql"


def test_inserting_a_prediction_for_a_missing_student_is_rejected(session: Session) -> None:
    """Referential integrity, which only means something with the pragma on."""
    model_version = _model_version(session)
    thresholds = _threshold_config(session)
    session.add(
        models.Prediction(
            student_id=999999,
            model_version_id=model_version.id,
            threshold_config_id=thresholds.id,
            checkpoint_day=30,
            horizon_days=30,
            probability=0.1,
            band="critical",
            features_hash="x",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_expected_tables_exist(engine: Engine) -> None:
    tables = set(inspect(engine).get_table_names())
    for name in (
        "students",
        "student_demographics",
        "engagement_records",
        "assessment_records",
        "model_versions",
        "threshold_configs",
        "predictions",
        "alerts",
        "interventions",
        "users",
        "audit_log",
    ):
        assert name in tables, name


def test_prediction_explanations_table_does_not_exist(engine: Engine) -> None:
    """Dropped deliberately. Phase 8 measured explanation latency at 28.8 ms per
    student, so the performance justification for storing them did not hold."""
    assert "prediction_explanations" not in set(inspect(engine).get_table_names())


def test_risk_history_is_a_view_not_a_table(engine: Engine) -> None:
    """A table would duplicate every fact already in `predictions` and create a
    way for the two to disagree."""
    inspector = inspect(engine)
    assert "risk_history" in set(inspector.get_view_names())
    assert "risk_history" not in set(inspector.get_table_names())


def test_risk_history_view_returns_the_trajectory(session: Session, world) -> None:
    rows = session.execute(
        text(
            "SELECT student_code, checkpoint_day, probability, model_version "
            "FROM risk_history WHERE student_code = :code ORDER BY checkpoint_day"
        ),
        {"code": "S-RISING"},
    ).all()
    assert [row.checkpoint_day for row in rows] == [30, 60, 90]
    assert rows[0].model_version == "xgboost-test"


def test_trajectory_index_exists_on_predictions(engine: Engine) -> None:
    """Declared because the trajectory query is the hottest read path."""
    names = {index["name"] for index in inspect(engine).get_indexes("predictions")}
    assert "ix_predictions_student_checkpoint" in names


def test_alert_queue_index_exists(engine: Engine) -> None:
    names = {index["name"] for index in inspect(engine).get_indexes("alerts")}
    assert "ix_alerts_queue" in names


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_duplicate_enrolment_is_rejected(session: Session) -> None:
    """The natural key is (module, presentation, student) — id_student alone is
    not unique, because students appear in several module-presentations."""
    _student(session, "S-ONE", 1001)
    with pytest.raises(IntegrityError):
        _student(session, "S-TWO", 1001)
        session.flush()


def test_same_student_id_in_a_different_module_is_allowed(session: Session) -> None:
    _student(session, "S-A", 1001, module="AAA")
    _student(session, "S-B", 1001, module="BBB")
    assert session.scalar(select(models.Student).where(models.Student.student_code == "S-B"))


def test_student_code_is_unique(session: Session) -> None:
    _student(session, "S-DUP", 1001)
    with pytest.raises(IntegrityError):
        _student(session, "S-DUP", 1002)
        session.flush()


def test_rescoring_the_same_checkpoint_is_rejected_as_a_duplicate(session: Session, world) -> None:
    """Unique on (student, checkpoint, model version), so a re-run updates in place
    rather than accumulating rows."""
    with pytest.raises(IntegrityError):
        _prediction(
            session,
            world["rising"],
            world["model_version"],
            world["thresholds"],
            30,
            0.5,
            "critical",
        )
        session.flush()


def test_probability_outside_zero_to_one_is_rejected(session: Session, world) -> None:
    with pytest.raises(IntegrityError):
        _prediction(
            session,
            world["rising"],
            world["model_version"],
            world["thresholds"],
            200,
            1.5,
            "critical",
        )
        session.flush()


def test_unknown_band_is_rejected(session: Session, world) -> None:
    with pytest.raises(IntegrityError):
        _prediction(
            session,
            world["rising"],
            world["model_version"],
            world["thresholds"],
            210,
            0.5,
            "apocalyptic",
        )
        session.flush()


def test_unknown_intervention_status_is_rejected(session: Session, world) -> None:
    """A closed vocabulary, so a new status cannot be invented at a call site."""
    session.add(
        models.Intervention(
            student_id=world["rising"].id,
            intervention_key="advisor_check_in",
            title="Advisor check-in",
            status="punished",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_unknown_user_role_is_rejected(session: Session) -> None:
    session.add(models.User(username="x", role="superuser", hashed_password="h"))
    with pytest.raises(IntegrityError):
        session.flush()


def test_unknown_data_source_is_rejected(session: Session) -> None:
    """Provenance is load-bearing for ADR-0002's honesty rule, so the vocabulary
    is closed rather than free text."""
    session.add(
        models.Student(
            student_code="S-X",
            code_module="AAA",
            code_presentation="2014J",
            source_student_id=9,
            data_source="made_up",
            module_presentation_length=269,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_deleting_a_student_cascades_to_their_records(session: Session, world) -> None:
    student = world["rising"]
    session.add(
        models.EngagementRecord(
            student_id=student.id, course_day=10, clicks=5, distinct_resources=2
        )
    )
    session.flush()
    session.delete(student)
    session.flush()
    assert (
        session.scalar(select(models.Prediction).where(models.Prediction.student_id == student.id))
        is None
    )
    assert (
        session.scalar(
            select(models.EngagementRecord).where(models.EngagementRecord.student_id == student.id)
        )
        is None
    )


def test_engagement_is_unique_per_student_day(session: Session, world) -> None:
    student = world["rising"]
    session.add(
        models.EngagementRecord(
            student_id=student.id, course_day=10, clicks=5, distinct_resources=1
        )
    )
    session.flush()
    session.add(
        models.EngagementRecord(
            student_id=student.id, course_day=10, clicks=9, distinct_resources=1
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_unsubmitted_assessment_is_representable(session: Session, world) -> None:
    """A due-but-unsubmitted assessment is the backlog signal, so the absence has
    to be a NULL rather than a missing row."""
    session.add(
        models.AssessmentRecord(
            student_id=world["rising"].id,
            assessment_id=1,
            assessment_type="TMA",
            due_day=20,
            weight=25.0,
            date_submitted=None,
            score=None,
        )
    )
    session.flush()
    row = session.scalar(select(models.AssessmentRecord))
    assert row is not None and row.date_submitted is None


# ---------------------------------------------------------------------------
# Repository behaviour
# ---------------------------------------------------------------------------


def test_repository_requires_a_registered_model_version(session: Session) -> None:
    with pytest.raises(LookupError, match="not registered"):
        SqlRepository(session, "xgboost-does-not-exist")


def test_latest_rows_returns_one_row_per_student(session: Session, world) -> None:
    repository = SqlRepository(session, "xgboost-test")
    rows = repository.latest_rows()
    assert len(rows) == 2
    assert set(rows["student_code"]) == {"S-RISING", "S-FALLING"}
    assert dict(zip(rows["student_code"], rows["checkpoint_day"], strict=True)) == {
        "S-RISING": 90,
        "S-FALLING": 90,
    }


def test_risk_direction_is_computed_from_the_last_two_predictions(session: Session, world) -> None:
    repository = SqlRepository(session, "xgboost-test")
    rows = repository.latest_rows().set_index("student_code")
    assert rows.loc["S-RISING", "risk_direction"] == "worsening"
    assert rows.loc["S-FALLING", "risk_direction"] == "improving"


def test_risk_direction_is_unknown_with_a_single_prediction(session: Session) -> None:
    """`unknown` rather than `stable`, which would claim a trend nobody has
    observed."""
    model_version = _model_version(session)
    thresholds = _threshold_config(session)
    student = _student(session, "S-NEW", 2001)
    _prediction(session, student, model_version, thresholds, 30, 0.04, "medium")

    rows = SqlRepository(session, "xgboost-test").latest_rows().set_index("student_code")
    assert rows.loc["S-NEW", "risk_direction"] == "unknown"


def test_student_rows_are_ordered_by_checkpoint(session: Session, world) -> None:
    rows = SqlRepository(session, "xgboost-test").student_rows("S-RISING")
    assert rows["checkpoint_day"].tolist() == [30, 60, 90]
    assert rows["probability"].is_monotonic_increasing


def test_unknown_student_returns_an_empty_frame(session: Session, world) -> None:
    assert SqlRepository(session, "xgboost-test").student_rows("S-NOPE").empty


def test_list_students_filters_and_sorts_by_risk(session: Session, world) -> None:
    repository = SqlRepository(session, "xgboost-test")
    rows, total = repository.list_students()
    assert total == 2
    assert rows["probability"].tolist() == sorted(rows["probability"], reverse=True)

    critical, count = repository.list_students(band="critical")
    assert count == 1
    assert critical["student_code"].tolist() == ["S-RISING"]


def test_list_students_filters_by_direction(session: Session, world) -> None:
    rows, total = SqlRepository(session, "xgboost-test").list_students(direction="improving")
    assert total == 1
    assert rows["student_code"].tolist() == ["S-FALLING"]


def test_band_counts_cover_every_band(session: Session, world) -> None:
    counts = dict(SqlRepository(session, "xgboost-test").band_counts())
    assert set(counts) == set(models.BAND_KEYS)
    assert sum(counts.values()) == 2


def test_worsening_count(session: Session, world) -> None:
    assert SqlRepository(session, "xgboost-test").worsening_count() == 1


def test_calibration_flag_comes_from_the_active_threshold_config(session: Session, world) -> None:
    repository = SqlRepository(session, "xgboost-test")
    assert repository.is_calibrated is True
    assert repository.alert_budget == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Interventions and alerts through the repository
# ---------------------------------------------------------------------------


def test_assigned_intervention_round_trips(session: Session, world) -> None:
    repository = SqlRepository(session, "xgboost-test")
    record = repository.add_intervention(
        "S-RISING", "advisor_check_in", "Advisor check-in", "counsellor", "admin", "note"
    )
    assert record.status == "assigned"
    assert record.assigned_by == "admin"
    listed = repository.interventions("S-RISING")
    assert [item.id for item in listed] == [record.id]


def test_assigning_to_an_unknown_student_raises(session: Session, world) -> None:
    with pytest.raises(KeyError):
        SqlRepository(session, "xgboost-test").add_intervention(
            "S-NOPE", "advisor_check_in", "t", "a", "b", None
        )


def test_alerts_are_returned_severity_ordered_and_acknowledgeable(session: Session, world) -> None:
    prediction = session.scalar(
        select(models.Prediction).where(
            models.Prediction.student_id == world["rising"].id,
            models.Prediction.checkpoint_day == 90,
        )
    )
    assert prediction is not None
    session.add(
        models.Alert(
            student_id=world["rising"].id,
            prediction_id=prediction.id,
            checkpoint_day=90,
            reason="band_escalated",
            severity="critical",
            probability=0.11,
            previous_probability=0.05,
            band="critical",
        )
    )
    session.flush()

    repository = SqlRepository(session, "xgboost-test")
    open_alerts = repository.alerts(acknowledged=False)
    assert len(open_alerts) == 1
    assert open_alerts[0].student_code == "S-RISING"

    acknowledged = repository.acknowledge_alert(open_alerts[0].id, "counsellor")
    assert acknowledged.acknowledged is True
    assert repository.alerts(acknowledged=False) == []


def test_duplicate_alert_for_the_same_prediction_and_reason_is_rejected(
    session: Session, world
) -> None:
    """Re-running the batch scorer must not produce a second copy of an alert."""
    prediction = session.scalar(
        select(models.Prediction).where(models.Prediction.student_id == world["rising"].id)
    )
    assert prediction is not None
    for _ in range(2):
        session.add(
            models.Alert(
                student_id=world["rising"].id,
                prediction_id=prediction.id,
                checkpoint_day=prediction.checkpoint_day,
                reason="rapid_increase",
                severity="high",
                probability=0.2,
                band="critical",
            )
        )
    with pytest.raises(IntegrityError):
        session.flush()


def test_acknowledging_an_unknown_alert_raises(session: Session, world) -> None:
    with pytest.raises(KeyError):
        SqlRepository(session, "xgboost-test").acknowledge_alert(999999, "counsellor")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_audit_entries_are_appended_and_queryable_by_actor(session: Session, world) -> None:
    repository = SqlRepository(session, "xgboost-test")
    repository.record_audit(
        "req-1", "counsellor", "GET /students/S-RISING", "student:S-RISING", 200
    )
    repository.record_audit("req-2", "admin", "GET /students/S-FALLING", "student:S-FALLING", 200)
    session.flush()

    assert len(repository.audit_entries()) == 2
    counsellor_entries = repository.audit_entries(actor="counsellor")
    assert len(counsellor_entries) == 1
    assert counsellor_entries[0].resource == "student:S-RISING"


def test_audit_log_has_no_update_or_delete_path() -> None:
    """An audit trail that can be edited is not an audit trail, so the repository
    exposes no mutation beyond append."""
    methods = {name for name in dir(SqlRepository) if "audit" in name}
    assert methods == {"record_audit", "audit_entries"}


# ---------------------------------------------------------------------------
# Demographic isolation
# ---------------------------------------------------------------------------


def test_demographics_live_in_their_own_table(session: Session, world) -> None:
    """Data minimisation is structural: the separation is what makes "the model
    cannot see this" verifiable rather than an assurance (ETHICS.md)."""
    student_columns = {column.name for column in models.Student.__table__.columns}
    for attribute in ("gender", "age_band", "imd_band", "disability", "region"):
        assert attribute not in student_columns


def test_repository_never_selects_a_protected_attribute(session: Session, world) -> None:
    """The stronger claim: no repository read returns one."""
    repository = SqlRepository(session, "xgboost-test")
    session.add(
        models.StudentDemographics(
            student_id=world["rising"].id, gender="F", age_band="0-35", imd_band="0-10%"
        )
    )
    session.flush()

    for frame in (
        repository.latest_rows(),
        repository.student_rows("S-RISING"),
        repository.trajectories(),
    ):
        for attribute in ("gender", "age_band", "imd_band", "disability", "region"):
            assert attribute not in frame.columns


# ---------------------------------------------------------------------------
# What this module does not cover
# ---------------------------------------------------------------------------


def test_reports_whether_it_ran_against_postgresql() -> None:
    """Not an assertion so much as a record in the test output.

    SQLite does not exercise JSONB, the partial index predicate, server-side
    default semantics, or PostgreSQL's stricter type coercion. Set
    TEST_DATABASE_URL to a PostgreSQL DSN to close that gap.
    """
    if IS_POSTGRES:
        assert POSTGRES_URL
    else:
        pytest.skip(
            "running against SQLite; set TEST_DATABASE_URL to a PostgreSQL DSN to "
            "verify JSONB, partial indexes and server-side defaults"
        )
