"""API tests: every endpoint, every auth outcome, and the RBAC matrix.

The RBAC matrix test is the important one. Access control written as scattered
per-endpoint checks drifts: an endpoint added later quietly omits a check and
nothing fails. Enumerating the matrix means a new endpoint has to be added to it
deliberately.

These run against the real model artifact, so they are skipped when none exists.
No database is needed — that is the point of the repository seam.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from dropout_ews.config.settings import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "LATEST").is_file(),
    reason="no registered model; run scripts/train_model.py",
)

CREDENTIALS = {
    "admin": "admin-demo-password",
    "counsellor": "counsellor-demo-password",
    "analyst": "analyst-demo-password",
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def tokens(client: TestClient) -> dict[str, str]:
    issued = {}
    for username, password in CREDENTIALS.items():
        response = client.post(
            "/api/v1/auth/token", data={"username": username, "password": password}
        )
        assert response.status_code == 200, response.text
        issued[username] = response.json()["access_token"]
    return issued


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def sample_code(client: TestClient, tokens: dict[str, str]) -> str:
    response = client.get(
        "/api/v1/students", params={"page_size": 1}, headers=auth(tokens["counsellor"])
    )
    assert response.status_code == 200, response.text
    return response.json()["data"][0]["student_code"]


# ---------------------------------------------------------------------------
# Health and startup
# ---------------------------------------------------------------------------


def test_health_is_unauthenticated(client: TestClient) -> None:
    """A load balancer cannot hold a token."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_health_reports_readiness_separately_from_liveness(client: TestClient) -> None:
    """`model_loaded` exists so a deployment shipping a broken artifact fails its
    readiness check instead of serving 500s."""
    body = client.get("/health").json()
    assert "model_loaded" in body
    assert "data_loaded" in body


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Student Dropout Early-Warning API"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_login_returns_a_usable_token(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/token", data={"username": "admin", "password": CREDENTIALS["admin"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["role"] == "admin", "role must serialise as its value, not 'Role.ADMIN'"


def test_login_rejects_a_bad_password(client: TestClient) -> None:
    response = client.post("/api/v1/auth/token", data={"username": "admin", "password": "wrong"})
    assert response.status_code == 401


def test_login_rejects_an_unknown_user_with_the_same_message(client: TestClient) -> None:
    """Identical responses, so the endpoint does not enumerate valid usernames."""
    unknown = client.post("/api/v1/auth/token", data={"username": "nobody", "password": "x"})
    bad_password = client.post("/api/v1/auth/token", data={"username": "admin", "password": "x"})
    assert unknown.status_code == bad_password.status_code == 401
    assert unknown.json()["detail"] == bad_password.json()["detail"]


def test_me_reports_capabilities(client: TestClient, tokens: dict[str, str]) -> None:
    body = client.get("/api/v1/auth/me", headers=auth(tokens["analyst"])).json()
    assert body["role"] == "analyst"
    assert body["may_view_individuals"] is False
    assert body["may_assign_interventions"] is False


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/students",
        "/api/v1/dashboard/statistics",
        "/api/v1/alerts",
        "/api/v1/model/info",
        "/metrics",
    ],
)
def test_protected_endpoints_reject_anonymous_requests(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


def test_malformed_token_is_rejected(client: TestClient) -> None:
    assert client.get("/api/v1/students", headers=auth("not-a-jwt")).status_code == 401


def test_expired_token_is_rejected(client: TestClient) -> None:
    from backend.app.security import Role, create_access_token

    expired = create_access_token("admin", Role.ADMIN, expires_minutes=-1)
    assert client.get("/api/v1/students", headers=auth(expired)).status_code == 401


# ---------------------------------------------------------------------------
# The RBAC matrix
# ---------------------------------------------------------------------------

# (method, path template, {role: expected status}). Enumerated so a new endpoint
# has to be added here deliberately rather than shipping unguarded.
RBAC_MATRIX = [
    ("GET", "/api/v1/students", {"admin": 200, "counsellor": 200, "analyst": 403}),
    ("GET", "/api/v1/students/{code}", {"admin": 200, "counsellor": 200, "analyst": 403}),
    (
        "GET",
        "/api/v1/students/{code}/risk-history",
        {"admin": 200, "counsellor": 200, "analyst": 403},
    ),
    (
        "GET",
        "/api/v1/students/{code}/explanation",
        {"admin": 200, "counsellor": 200, "analyst": 403},
    ),
    (
        "GET",
        "/api/v1/students/{code}/recommendations",
        {"admin": 200, "counsellor": 200, "analyst": 403},
    ),
    (
        "GET",
        "/api/v1/students/{code}/interventions",
        {"admin": 200, "counsellor": 200, "analyst": 403},
    ),
    ("GET", "/api/v1/dashboard/statistics", {"admin": 200, "counsellor": 200, "analyst": 200}),
    (
        "GET",
        "/api/v1/analytics/risk-distribution",
        {"admin": 200, "counsellor": 200, "analyst": 200},
    ),
    (
        "GET",
        "/api/v1/analytics/feature-importance",
        {"admin": 200, "counsellor": 200, "analyst": 200},
    ),
    ("GET", "/api/v1/alerts", {"admin": 200, "counsellor": 200, "analyst": 403}),
    ("GET", "/api/v1/model/info", {"admin": 200, "counsellor": 200, "analyst": 200}),
    ("GET", "/api/v1/admin/audit-log", {"admin": 200, "counsellor": 403, "analyst": 403}),
    ("GET", "/metrics", {"admin": 200, "counsellor": 200, "analyst": 200}),
]


@pytest.mark.parametrize(("method", "path", "expected"), RBAC_MATRIX)
def test_rbac_matrix(
    client: TestClient,
    tokens: dict[str, str],
    sample_code: str,
    method: str,
    path: str,
    expected: dict[str, int],
) -> None:
    url = path.format(code=sample_code)
    for role, status_code in expected.items():
        response = client.request(method, url, headers=auth(tokens[role]))
        assert response.status_code == status_code, (
            f"{method} {url} as {role}: expected {status_code}, got "
            f"{response.status_code} ({response.text[:200]})"
        )


def test_analyst_is_denied_every_individual_endpoint(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    """The substantive restriction: cohort analysis must not require access to an
    identifiable student's risk (ETHICS.md)."""
    individual_paths = [
        path for _, path, _ in RBAC_MATRIX if "{code}" in path or path.endswith("/alerts")
    ]
    for path in individual_paths:
        response = client.get(path.format(code=sample_code), headers=auth(tokens["analyst"]))
        assert response.status_code == 403, path


def test_analyst_can_still_use_every_aggregate_endpoint(
    client: TestClient, tokens: dict[str, str]
) -> None:
    """Denying individuals must not make the role useless."""
    for path in (
        "/api/v1/dashboard/statistics",
        "/api/v1/analytics/risk-distribution",
        "/api/v1/analytics/feature-importance",
        "/api/v1/model/info",
    ):
        assert client.get(path, headers=auth(tokens["analyst"])).status_code == 200, path


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------


def test_student_list_is_paginated_and_sorted_by_risk(
    client: TestClient, tokens: dict[str, str]
) -> None:
    body = client.get(
        "/api/v1/students", params={"page_size": 25}, headers=auth(tokens["counsellor"])
    ).json()
    assert len(body["data"]) == 25
    assert body["meta"]["total"] > 25
    probabilities = [row["probability"] for row in body["data"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_student_list_carries_no_personal_identifiers(
    client: TestClient, tokens: dict[str, str]
) -> None:
    """Data minimisation: `student_code` is opaque and derived, and the source
    `id_student` never leaves the service."""
    body = client.get(
        "/api/v1/students", params={"page_size": 5}, headers=auth(tokens["counsellor"])
    ).json()
    for row in body["data"]:
        assert row["student_code"].startswith("S-")
        assert "id_student" not in row
        assert "name" not in row
        assert "email" not in row


def test_student_list_filters_by_band(client: TestClient, tokens: dict[str, str]) -> None:
    body = client.get(
        "/api/v1/students",
        params={"band": "critical", "page_size": 20},
        headers=auth(tokens["counsellor"]),
    ).json()
    assert body["data"], "expected some critical-band students"
    assert {row["band"] for row in body["data"]} == {"critical"}


def test_invalid_band_filter_is_a_422(client: TestClient, tokens: dict[str, str]) -> None:
    response = client.get(
        "/api/v1/students", params={"band": "apocalyptic"}, headers=auth(tokens["counsellor"])
    )
    assert response.status_code == 422


def test_unknown_student_is_a_404(client: TestClient, tokens: dict[str, str]) -> None:
    response = client.get("/api/v1/students/S-NOPE", headers=auth(tokens["counsellor"]))
    assert response.status_code == 404


def test_student_profile_shape(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    body = client.get(f"/api/v1/students/{sample_code}", headers=auth(tokens["counsellor"])).json()
    assert body["student_code"] == sample_code
    assert body["trajectory"], "a profile should carry a risk trajectory"
    assert body["engagement"] and body["assessment"]
    assert body["risk_direction"] in {"improving", "stable", "worsening", "unknown"}


def test_risk_history_is_ordered_by_checkpoint(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    body = client.get(
        f"/api/v1/students/{sample_code}/risk-history", headers=auth(tokens["counsellor"])
    ).json()
    days = [point["checkpoint_day"] for point in body]
    assert days == sorted(days)


# ---------------------------------------------------------------------------
# Every risk figure carries its provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/students/{code}",
        "/api/v1/students/{code}/explanation",
    ],
)
def test_risk_responses_carry_model_version_and_disclaimer(
    client: TestClient, tokens: dict[str, str], sample_code: str, path: str
) -> None:
    """A probability detached from what produced it and from what it means is the
    thing this project most needs not to ship."""
    body = client.get(path.format(code=sample_code), headers=auth(tokens["counsellor"])).json()
    risk = body["risk"]
    assert risk["model_version"]
    assert risk["disclaimer"]
    assert "not the causes" in risk["disclaimer"]
    assert "is_calibrated" in risk


def test_explanation_separates_actionable_from_contextual_factors(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    body = client.get(
        f"/api/v1/students/{sample_code}/explanation", headers=auth(tokens["counsellor"])
    ).json()
    assert "risk_factors" in body and "context_factors" in body
    for factor in body["risk_factors"]:
        assert factor["actionable"] is True
    for factor in body["context_factors"]:
        assert factor["actionable"] is False
    assert "pre-calibration" in body["attribution_basis"]


def test_explanation_accepts_an_as_of_checkpoint(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    history = client.get(
        f"/api/v1/students/{sample_code}/risk-history", headers=auth(tokens["counsellor"])
    ).json()
    earliest = history[0]["checkpoint_day"]
    body = client.get(
        f"/api/v1/students/{sample_code}/explanation",
        params={"as_of": earliest},
        headers=auth(tokens["counsellor"]),
    ).json()
    assert body["checkpoint_day"] == earliest


def test_explanation_for_an_unscored_checkpoint_is_a_404(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    response = client.get(
        f"/api/v1/students/{sample_code}/explanation",
        params={"as_of": 999},
        headers=auth(tokens["counsellor"]),
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Recommendations and assignment
# ---------------------------------------------------------------------------


def test_recommendations_require_human_review(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    body = client.get(
        f"/api/v1/students/{sample_code}/recommendations", headers=auth(tokens["counsellor"])
    ).json()
    assert body["case_note"]
    assert body["case_note_source"] in {"llm", "template"}
    for item in body["recommendations"]:
        assert item["status"] == "recommended"
        assert item["requires_human_review"] is True


def test_assigning_an_intervention_is_the_human_act(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    """Assignment moves a recommendation past `recommended`, and only a human
    request can do it (ADR-0005)."""
    response = client.post(
        f"/api/v1/students/{sample_code}/interventions",
        json={"intervention_key": "advisor_check_in", "assigned_to": "counsellor"},
        headers=auth(tokens["counsellor"]),
    )
    assert response.status_code == 201, response.text
    record = response.json()
    assert record["status"] == "assigned"
    assert record["assigned_by"] == "counsellor"

    listed = client.get(
        f"/api/v1/students/{sample_code}/interventions", headers=auth(tokens["counsellor"])
    ).json()
    assert any(item["id"] == record["id"] for item in listed)


def test_assigning_an_unknown_intervention_is_a_422(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    response = client.post(
        f"/api/v1/students/{sample_code}/interventions",
        json={"intervention_key": "expel_student", "assigned_to": "x"},
        headers=auth(tokens["counsellor"]),
    )
    assert response.status_code == 422


def test_analyst_cannot_assign_an_intervention(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    response = client.post(
        f"/api/v1/students/{sample_code}/interventions",
        json={"intervention_key": "advisor_check_in", "assigned_to": "x"},
        headers=auth(tokens["analyst"]),
    )
    assert response.status_code == 403


def test_assignment_to_an_unknown_student_is_a_404(
    client: TestClient, tokens: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/students/S-NOPE/interventions",
        json={"intervention_key": "advisor_check_in", "assigned_to": "x"},
        headers=auth(tokens["counsellor"]),
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_returns_a_features_hash(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    """The hash ties a stored prediction to the exact inputs that produced it."""
    response = client.post(
        "/api/v1/predict",
        json={"student_code": sample_code},
        headers=auth(tokens["counsellor"]),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["features_hash"]
    assert body["risk"]["model_version"]


def test_predict_rejects_a_raw_feature_vector(client: TestClient, tokens: dict[str, str]) -> None:
    """Features must come from the shared builder. Accepting a vector would let a
    caller submit one no real student could produce, and the as-of guarantee would
    mean nothing."""
    response = client.post(
        "/api/v1/predict",
        json={"clicks_7d": 0, "mean_score": 10},
        headers=auth(tokens["counsellor"]),
    )
    assert response.status_code == 422


def test_predict_for_an_unknown_student_is_a_404(
    client: TestClient, tokens: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/predict", json={"student_code": "S-NOPE"}, headers=auth(tokens["counsellor"])
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Dashboard, analytics, alerts
# ---------------------------------------------------------------------------


def test_dashboard_statistics_shape(client: TestClient, tokens: dict[str, str]) -> None:
    body = client.get("/api/v1/dashboard/statistics", headers=auth(tokens["counsellor"])).json()
    assert body["total_students"] > 0
    assert {item["band"] for item in body["band_counts"]} == {
        "low",
        "medium",
        "high",
        "critical",
    }
    assert sum(item["count"] for item in body["band_counts"]) == body["total_students"]
    assert body["is_calibrated"] is True
    assert 0 < body["alert_budget"] <= 1


def test_risk_distribution_bins_sum_to_the_cohort(
    client: TestClient, tokens: dict[str, str]
) -> None:
    total = client.get("/api/v1/dashboard/statistics", headers=auth(tokens["analyst"])).json()[
        "total_students"
    ]
    bins = client.get("/api/v1/analytics/risk-distribution", headers=auth(tokens["analyst"])).json()
    assert sum(item["count"] for item in bins) == total


# Columns that travel alongside the features in the cohort frame but must never
# be servable. `date_unregistration` is the one that mattered: an earlier version
# validated `x` against column presence rather than an allowlist, so
# `?x=date_unregistration` returned HTTP 200 and served the withdrawal event time
# — the label — to any authenticated user, including an analyst who is denied all
# individual data. This test found that.
FORBIDDEN_SCATTER_FIELDS = [
    "date_unregistration",
    "label",
    "id_student",
    "final_result",
    "gender",
    "imd_band",
    "disability",
    "probability",
    "student_code",
]


@pytest.mark.parametrize("field", FORBIDDEN_SCATTER_FIELDS)
def test_scatter_refuses_to_serve_a_forbidden_column(
    client: TestClient, tokens: dict[str, str], field: str
) -> None:
    """Regression test for an arbitrary-column read that leaked the label."""
    response = client.get(
        "/api/v1/analytics/scatter", params={"x": field}, headers=auth(tokens["analyst"])
    )
    assert response.status_code == 422, f"scatter served '{field}', which must never be exposed"


def test_scatter_allowlist_contains_no_outcome_or_protected_column() -> None:
    """Check the allowlist itself, not only the endpoint, so a future edit to the
    field groups cannot reintroduce the leak."""
    from backend.app import services

    from dropout_ews.config.settings import load_feature_config

    allowed = set(services.known_scatter_fields())
    forbidden = set(load_feature_config().forbidden.exact)
    assert not (allowed & forbidden), f"allowlist exposes forbidden columns: {allowed & forbidden}"
    assert "label" not in allowed


def test_scatter_returns_points_for_a_known_field(
    client: TestClient, tokens: dict[str, str]
) -> None:
    body = client.get(
        "/api/v1/analytics/scatter", params={"x": "clicks_28d"}, headers=auth(tokens["analyst"])
    ).json()
    assert body and all("x" in point and "y" in point for point in body)


def test_feature_importance_uses_human_labels(client: TestClient, tokens: dict[str, str]) -> None:
    body = client.get(
        "/api/v1/analytics/feature-importance", headers=auth(tokens["analyst"])
    ).json()
    assert body
    for item in body:
        assert item["label"], f"{item['feature']} has no human label"


def test_alerts_can_be_filtered_and_acknowledged(
    client: TestClient, tokens: dict[str, str]
) -> None:
    open_alerts = client.get(
        "/api/v1/alerts", params={"acknowledged": False}, headers=auth(tokens["counsellor"])
    ).json()
    assert open_alerts, "expected some open alerts"
    alert_id = open_alerts[0]["id"]

    acknowledged = client.post(
        f"/api/v1/alerts/{alert_id}/acknowledge", headers=auth(tokens["counsellor"])
    )
    assert acknowledged.status_code == 200
    assert acknowledged.json()["acknowledged"] is True


def test_acknowledging_an_unknown_alert_is_a_404(
    client: TestClient, tokens: dict[str, str]
) -> None:
    response = client.post("/api/v1/alerts/999999/acknowledge", headers=auth(tokens["counsellor"]))
    assert response.status_code == 404


def test_model_info_surfaces_limitations(client: TestClient, tokens: dict[str, str]) -> None:
    """Limitations travel with the metrics, so a client cannot present one
    without the other."""
    body = client.get("/api/v1/model/info", headers=auth(tokens["analyst"])).json()
    assert body["metrics"]
    assert len(body["limitations"]) >= 5
    assert any("87%" in item or "not about to withdraw" in item for item in body["limitations"])


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_reading_a_student_profile_is_audited(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    """ETHICS.md commits to answering "who looked at this student, and when". A
    system that cannot answer that should not hold individual risk scores."""
    from backend.app.middleware import audit_log

    before = len(audit_log.for_student(sample_code))
    client.get(f"/api/v1/students/{sample_code}", headers=auth(tokens["counsellor"]))
    entries = audit_log.for_student(sample_code)
    assert len(entries) == before + 1
    latest = entries[-1]
    assert latest.actor == "counsellor"
    assert sample_code in latest.resource
    assert latest.request_id


def test_aggregate_requests_are_not_audited(client: TestClient, tokens: dict[str, str]) -> None:
    """Auditing every aggregate call would bury the entries that matter."""
    from backend.app.middleware import audit_log

    before = len(audit_log.entries)
    client.get("/api/v1/dashboard/statistics", headers=auth(tokens["analyst"]))
    assert len(audit_log.entries) == before


def test_denied_access_is_not_recorded_as_a_successful_read(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    from backend.app.middleware import audit_log

    before = len(audit_log.for_student(sample_code))
    response = client.get(f"/api/v1/students/{sample_code}", headers=auth(tokens["analyst"]))
    assert response.status_code == 403
    assert len(audit_log.for_student(sample_code)) == before


def test_admin_can_read_the_audit_log(
    client: TestClient, tokens: dict[str, str], sample_code: str
) -> None:
    client.get(f"/api/v1/students/{sample_code}", headers=auth(tokens["counsellor"]))
    body = client.get(
        "/api/v1/admin/audit-log", params={"actor": "counsellor"}, headers=auth(tokens["admin"])
    ).json()
    assert body
    assert all(entry["actor"] == "counsellor" for entry in body)


def test_responses_carry_a_request_id(client: TestClient, tokens: dict[str, str]) -> None:
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")


def test_supplied_request_id_is_echoed(client: TestClient) -> None:
    """So a client can correlate its own logs with the server's."""
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
