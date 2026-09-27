"""The API, served from the database instead of the parquet file.

This is the check on whether the Phase 8 repository seam was in the right place.
If switching storage had required editing a router or a service, it was not. The
only change is ``REPOSITORY_BACKEND=database``.

The endpoints asserted here are the ones whose behaviour depends on storage:
listing, profile, trajectory, alerts, dashboard. Auth and RBAC are covered
exhaustively in ``tests/api/test_api.py`` and are not repeated.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from backend.app.db import models
from backend.app.db.session import enforce_sqlite_foreign_keys
from backend.app.security import Role, create_access_token
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from dropout_ews.config.settings import MODELS_DIR

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (MODELS_DIR / "LATEST").is_file(),
        reason="no registered model; run scripts/train_model.py",
    ),
]

MODEL_VERSION = "xgboost-seam-test"


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> Iterator[Engine]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        # A file rather than :memory:, so the app's own session sees the same
        # database as the fixture that populated it.
        path = tmp_path_factory.mktemp("db") / "seam.db"
        url = f"sqlite:///{path.as_posix()}"
    created = create_engine(
        url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
    )
    enforce_sqlite_foreign_keys(created)
    with created.begin() as connection:
        connection.exec_driver_sql(models.DROP_RISK_HISTORY_VIEW_SQL)
    models.Base.metadata.drop_all(created)
    models.Base.metadata.create_all(created)
    with created.begin() as connection:
        connection.exec_driver_sql(models.RISK_HISTORY_VIEW_SQL)
    yield created
    created.dispose()


@pytest.fixture(scope="module")
def populated(engine: Engine) -> Iterator[dict[str, object]]:
    """A small cohort with real trajectories, committed so the app can read it."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session: Session = factory()

    model_version = models.ModelVersion(
        version=MODEL_VERSION,
        model_type="xgboost",
        trained_at="2026-09-27T00:00:00Z",
        n_features=37,
        train_data_hash="hash",
        imbalance_strategy="none",
        calibration_method="isotonic",
        metrics={"test_pr_auc": 0.0885},
        feature_names=["clicks_7d"],
        is_active=True,
    )
    thresholds = models.ThresholdConfig(
        version=2,
        name="budget-calibrated-v1",
        calibrated=True,
        alert_budget=0.05,
        bands=[
            {"key": "low", "label": "Low", "min_probability": 0.0, "color": "#0", "action": "a"},
            {
                "key": "medium",
                "label": "Medium",
                "min_probability": 0.03,
                "color": "#1",
                "action": "b",
            },
            {"key": "high", "label": "High", "min_probability": 0.06, "color": "#2", "action": "c"},
            {
                "key": "critical",
                "label": "Critical",
                "min_probability": 0.09,
                "color": "#3",
                "action": "d",
            },
        ],
        is_active=True,
    )
    session.add_all([model_version, thresholds])
    session.flush()

    trajectories = {
        "S-AAA0000000A": [(30, 0.02, "low"), (60, 0.05, "medium"), (90, 0.12, "critical")],
        "S-AAA0000000B": [(30, 0.11, "critical"), (60, 0.07, "high"), (90, 0.02, "low")],
        "S-AAA0000000C": [(30, 0.04, "medium")],
    }
    for index, (code, points) in enumerate(trajectories.items(), start=1):
        student = models.Student(
            student_code=code,
            code_module="AAA",
            code_presentation="2014J",
            source_student_id=5000 + index,
            data_source="real_oulad",
            module_presentation_length=269,
        )
        session.add(student)
        session.flush()
        for day, probability, band in points:
            prediction = models.Prediction(
                student_id=student.id,
                model_version_id=model_version.id,
                threshold_config_id=thresholds.id,
                checkpoint_day=day,
                horizon_days=30,
                probability=probability,
                band=band,
                features_hash="h",
            )
            session.add(prediction)
            session.flush()
            if band == "critical" and day > 30:
                session.add(
                    models.Alert(
                        student_id=student.id,
                        prediction_id=prediction.id,
                        checkpoint_day=day,
                        reason="band_escalated",
                        severity="critical",
                        probability=probability,
                        previous_probability=0.05,
                        band=band,
                    )
                )
    session.commit()
    yield {"session": session, "codes": list(trajectories)}
    session.close()


@pytest.fixture(scope="module")
def client(engine: Engine, populated) -> Iterator[TestClient]:
    """The app, configured to read from the database.

    ``load_state`` is patched only to inject this test's engine and model version;
    the repository it builds is the production ``SqlRepository``.
    """
    from backend.app import main as app_module
    from backend.app.repository import RepositoryState

    from dropout_ews.explainability.shap_explainer import RiskExplainer
    from dropout_ews.models.registry import load_background, load_model

    model, metadata = load_model()
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    # The app resolves the model version from artifact metadata; the fixture
    # registers its own, so point them at each other.
    metadata.model_version = MODEL_VERSION

    def fake_load_state() -> RepositoryState:
        fresh = RepositoryState()
        fresh.metadata = metadata
        fresh.explainer = RiskExplainer(model, background=load_background())
        # The app builds a SqlRepository per request from this factory, which is
        # what makes writes commit. Only the engine and version are injected.
        fresh.backend = "database"
        fresh.session_factory = factory
        return fresh

    original = app_module.load_state
    app_module.load_state = fake_load_state
    try:
        with TestClient(app_module.app) as test_client:
            yield test_client
    finally:
        app_module.load_state = original


def auth(role: Role = Role.COUNSELLOR) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(role.value, role)}"}


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def test_the_api_serves_from_the_database(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_student_list_comes_from_stored_predictions(client: TestClient, populated) -> None:
    body = client.get("/api/v1/students", headers=auth()).json()
    assert body["meta"]["total"] == 3
    assert {row["student_code"] for row in body["data"]} == set(populated["codes"])
    probabilities = [row["probability"] for row in body["data"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_trajectory_is_read_not_recomputed(client: TestClient) -> None:
    """The reason predictions are persisted: a trajectory must show the scores
    that were actually produced, not fresh ones under a newer model."""
    body = client.get("/api/v1/students/S-AAA0000000A/risk-history", headers=auth()).json()
    assert [point["checkpoint_day"] for point in body] == [30, 60, 90]
    assert [point["probability"] for point in body] == [0.02, 0.05, 0.12]
    assert [point["band"] for point in body] == ["low", "medium", "critical"]


def test_risk_direction_is_derived_from_stored_history(client: TestClient) -> None:
    rows = {
        row["student_code"]: row["risk_direction"]
        for row in client.get("/api/v1/students", headers=auth()).json()["data"]
    }
    assert rows["S-AAA0000000A"] == "worsening"
    assert rows["S-AAA0000000B"] == "improving"
    # One prediction only: `unknown`, not `stable`, which would claim a trend
    # nobody has observed.
    assert rows["S-AAA0000000C"] == "unknown"


def test_profile_carries_provenance_from_the_database(client: TestClient) -> None:
    body = client.get("/api/v1/students/S-AAA0000000A", headers=auth()).json()
    assert body["risk"]["model_version"]
    assert body["risk"]["is_calibrated"] is True
    assert "not the causes" in body["risk"]["disclaimer"]


def test_explanation_rebuilds_features_from_stored_events(client: TestClient) -> None:
    """The ADR-0004 claim, made concrete.

    ``SqlRepository.feature_row`` runs the **same** ``FeatureBuilder`` as training,
    reading the event tables instead of the processed parquet. So explanations work
    off the database with no explanations table — and an earlier version of this
    test caught that ``feature_row`` had never been implemented at all, which
    showed the Protocol was documentation rather than a check.
    """
    response = client.get("/api/v1/students/S-AAA0000000A/explanation", headers=auth())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["student_code"] == "S-AAA0000000A"
    assert "pre-calibration" in body["attribution_basis"]
    assert body["disclaimer"]


def test_alerts_come_from_the_database_and_can_be_acknowledged(client: TestClient) -> None:
    open_alerts = client.get(
        "/api/v1/alerts", params={"acknowledged": False}, headers=auth()
    ).json()
    assert open_alerts
    alert_id = open_alerts[0]["id"]
    assert client.post(f"/api/v1/alerts/{alert_id}/acknowledge", headers=auth()).status_code == 200
    remaining = client.get("/api/v1/alerts", params={"acknowledged": False}, headers=auth()).json()
    assert alert_id not in [alert["id"] for alert in remaining]


def test_dashboard_statistics_aggregate_stored_predictions(client: TestClient) -> None:
    body = client.get("/api/v1/dashboard/statistics", headers=auth()).json()
    assert body["total_students"] == 3
    assert body["scored_checkpoints"] == 7
    assert sum(item["count"] for item in body["band_counts"]) == 3
    assert body["worsening_count"] == 1
    assert body["is_calibrated"] is True


def test_assigning_an_intervention_persists_to_the_database(client: TestClient, populated) -> None:
    response = client.post(
        "/api/v1/students/S-AAA0000000A/interventions",
        json={"intervention_key": "advisor_check_in", "assigned_to": "counsellor"},
        headers=auth(),
    )
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "assigned"

    session: Session = populated["session"]  # type: ignore[assignment]
    session.expire_all()
    rows = session.scalars(select(models.Intervention)).all()
    assert any(row.intervention_key == "advisor_check_in" for row in rows)


def test_unknown_student_is_still_a_404(client: TestClient) -> None:
    assert client.get("/api/v1/students/S-NOPE", headers=auth()).status_code == 404


def test_analyst_is_still_denied_individual_access(client: TestClient) -> None:
    """RBAC is enforced in the router, so it cannot vary by storage backend."""
    response = client.get("/api/v1/students/S-AAA0000000A", headers=auth(Role.ANALYST))
    assert response.status_code == 403


def test_no_protected_attribute_is_returned_by_any_endpoint(client: TestClient) -> None:
    """Demographics sit in their own table and no repository query touches it."""
    body = client.get("/api/v1/students/S-AAA0000000A", headers=auth()).text.lower()
    for attribute in ("gender", "imd_band", "disability", '"region"', "age_band"):
        assert attribute not in body
