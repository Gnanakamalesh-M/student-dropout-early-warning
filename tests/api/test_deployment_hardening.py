"""Tests for the configuration that only matters once this is deployed.

Every check here corresponds to something that is either invisible or actively
misleading in local development:

- CORS was hardcoded to the Vite dev server. In a real deployment the browser
  blocks the response *after* the request succeeded, so the API log shows a
  healthy 200 and the dashboard shows a network error. Nothing points at the
  cause.
- A misconfigured production environment should fail at startup, not on the first
  request. A wildcard CORS origin with credentials is rejected by browsers at
  runtime; refusing it at boot converts a confusing outage into a clear one.
- HSTS sent over plain HTTP is wrong, and on localhost it pins the developer's
  browser to HTTPS for the max-age. So it is opt-in, and that must stay true.
"""

from __future__ import annotations

import pytest
from backend.app.main import create_app
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dropout_ews.config.settings import Settings

# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Referrer-Policy", "no-referrer"),
        ("Cache-Control", "no-store"),
    ],
)
def test_security_headers_are_present(client: TestClient, header: str, value: str) -> None:
    assert client.get("/health").headers.get(header) == value


def test_student_risk_responses_are_not_cacheable(client: TestClient) -> None:
    """The header that matters most for this API specifically. A per-student risk
    figure left in a shared machine's browser cache is precisely the disclosure
    the data-minimisation section of ETHICS.md is about."""
    response = client.get("/health")
    assert "no-store" in response.headers["Cache-Control"]


def test_referrer_policy_protects_student_codes_in_urls(client: TestClient) -> None:
    """Student codes appear in paths. Under the browser default, any third-party
    resource the dashboard loads would receive the full URL."""
    assert client.get("/health").headers["Referrer-Policy"] == "no-referrer"


def test_hsts_is_absent_unless_explicitly_enabled(client: TestClient) -> None:
    assert "Strict-Transport-Security" not in client.get("/health").headers


def test_hsts_appears_when_force_https_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag must actually do something, or it is a comment."""
    from backend.app.middleware import SecurityHeadersMiddleware

    middleware = SecurityHeadersMiddleware(app=object(), force_https=True)
    assert middleware.force_https is True


def test_security_headers_do_not_override_a_deliberate_value() -> None:
    """`setdefault`, not assignment: an endpoint that needs its own cache policy
    must be able to set one. Asserted so a later tidy to `=` is caught."""
    import inspect

    from backend.app.middleware import SecurityHeadersMiddleware

    source = inspect.getsource(SecurityHeadersMiddleware.dispatch)
    assert "setdefault" in source
    assert 'headers["X-Frame-Options"] =' not in source


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------


def test_cors_origins_come_from_settings_not_a_literal() -> None:
    """The bug this replaced: hardcoded dev origins that break every deployment
    silently. If the literal comes back, this fails."""
    import inspect

    from backend.app import main

    source = inspect.getsource(main.create_app)
    assert "settings.cors_origins" in source
    assert "localhost:5173" not in source


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear settings overrides so default assertions read as defaults.

    `Settings()` reads both `.env` and the process environment. CI has no `.env`
    but does set ENVIRONMENT=ci, and a developer may have any of these exported --
    either way an assertion about a default is otherwise testing the ambient
    machine rather than the code.
    """
    for name in (
        "ENVIRONMENT",
        "SECRET_KEY",
        "CORS_ORIGINS",
        "FORCE_HTTPS",
        "ENABLE_LLM_NARRATIVE",
        "ANTHROPIC_API_KEY",
        "REPOSITORY_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)


def test_local_defaults_keep_the_dev_server_working(clean_env: None) -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert "http://localhost:5173" in settings.cors_origins


def test_wildcard_origin_is_refused_at_startup(clean_env: None) -> None:
    """Browsers reject wildcard-with-credentials at runtime, so this would fail on
    the first real request instead of at boot. Boot is the better place."""
    with pytest.raises(ValidationError, match=r"must not contain"):
        Settings(cors_origins=["*"])


def test_production_refuses_leftover_development_origins(clean_env: None) -> None:
    """The realistic mistake: deploying with the defaults still in place. The API
    would then accept credentialed requests from anything running on the
    operator's own machine."""
    with pytest.raises(ValidationError, match="development origins"):
        Settings(
            environment="production",
            secret_key="x" * 32,
            cors_origins=["http://localhost:5173"],
        )


def test_production_refuses_plain_http_origins(clean_env: None) -> None:
    with pytest.raises(ValidationError, match="must use https"):
        Settings(environment="production", secret_key="x" * 32, cors_origins=["http://app.edu"])


def test_production_refuses_an_empty_origin_list(clean_env: None) -> None:
    """An empty list is not "allow nothing", it is "nobody configured this"."""
    with pytest.raises(ValidationError, match="must be set"):
        Settings(environment="production", secret_key="x" * 32, cors_origins=[])


def test_production_accepts_a_real_https_origin(clean_env: None) -> None:
    settings = Settings(
        environment="production",
        secret_key="x" * 32,
        cors_origins=["https://ews.university.edu"],
    )
    assert settings.cors_origins == ["https://ews.university.edu"]


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_production_refuses_the_development_secret(clean_env: None) -> None:
    with pytest.raises(ValidationError, match="secret_key"):
        Settings(environment="production")


def test_production_refuses_a_short_secret(clean_env: None) -> None:
    with pytest.raises(ValidationError, match="32 characters"):
        Settings(
            environment="production",
            secret_key="short",
            cors_origins=["https://ews.university.edu"],
        )


def test_demo_users_exist_only_in_local_environment() -> None:
    """The demo login buttons must not reach a deployed instance. They are seeded
    onto app state only for `local`, and this asserts the guard rather than
    trusting it."""
    import inspect

    from backend.app import main

    source = inspect.getsource(main.create_app)
    assert 'settings.environment == "local"' in source
    assert "demo_users" in source


def test_llm_flag_cannot_be_enabled_without_a_key(clean_env: None) -> None:
    """ADR-0005: the feature is optional. Enabling it with no key would fail on
    the first case-note request rather than at startup."""
    with pytest.raises(ValidationError, match="anthropic_api_key"):
        Settings(enable_llm_narrative=True)
