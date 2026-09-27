"""FastAPI application.

The model is loaded once at startup, not per request. ``/health`` reports
``model_loaded`` separately from process liveness, so a deployment that ships a
broken artifact fails its readiness check instead of serving 500s.

Startup is tolerant of a missing model: the app still boots and ``/health``
reports the failure. A container that refuses to start gives you no way to ask it
what went wrong.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from backend.app import schemas, services
from backend.app.middleware import RequestContextMiddleware, audit_log, configure_logging
from backend.app.repository import ParquetRepository, RepositoryState
from backend.app.security import (
    Role,
    User,
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    require_any_role,
    require_individual_access,
    verify_password,
)
from dropout_ews.config.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session

state = RepositoryState()

# Demo accounts. A real deployment reads users from the database (Phase 9); these
# exist so the API is usable and the RBAC matrix is testable without one. The
# passwords are obviously not secrets.
DEMO_USERS: dict[str, User] = {
    "admin": User("admin", Role.ADMIN, hash_password("admin-demo-password")),
    "counsellor": User("counsellor", Role.COUNSELLOR, hash_password("counsellor-demo-password")),
    "analyst": User("analyst", Role.ANALYST, hash_password("analyst-demo-password")),
}


def load_state() -> RepositoryState:
    """Load the model, explainer and repository. Records failure rather than raising."""
    fresh = RepositoryState()
    try:
        from dropout_ews.explainability.shap_explainer import RiskExplainer
        from dropout_ews.models.registry import load_background, load_model

        model, metadata = load_model()
        fresh.metadata = metadata
        fresh.explainer = RiskExplainer(model, background=load_background())

        settings = get_settings()
        if settings.repository_backend == "database":
            # Nothing is built here. A repository over a long-lived session would
            # share one transaction across concurrent requests, and — as an
            # integration test found — never commit, so a write returned HTTP 201
            # and was silently lost. The repository is built per request instead;
            # see `get_repository`.
            fresh.repository = None
            fresh.backend = "database"
        else:
            fresh.repository = ParquetRepository(model, metadata.model_version)
            fresh.backend = "parquet"
    except Exception as exc:
        fresh.load_error = f"{type(exc).__name__}: {exc}"
    return fresh


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings().log_level)
    loaded = load_state()
    state.repository = loaded.repository
    state.explainer = loaded.explainer
    state.metadata = loaded.metadata
    state.load_error = loaded.load_error
    state.backend = loaded.backend
    state.session_factory = loaded.session_factory
    yield


def get_repository() -> Iterator[Any]:
    """Yield a repository for this request.

    For the database backend this opens a session, commits on success and rolls
    back on failure. That is not boilerplate: the first version held one session
    open for the process lifetime and never committed, so assigning an
    intervention returned HTTP 201 and lost the row. A per-request transaction is
    also the only correct choice under concurrency.
    """
    if state.load_error is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model or data not loaded: {state.load_error}",
        )

    if state.backend == "database":
        from backend.app.db.repository import SqlRepository
        from backend.app.db.session import get_session_factory

        version = getattr(state.metadata, "model_version", None)
        if version is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
            )
        factory = (
            cast("Callable[[], Session]", state.session_factory)
            if state.session_factory
            else get_session_factory()
        )
        session = factory()
        try:
            yield SqlRepository(session, version)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return

    if state.repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model or data not loaded: {state.load_error or 'unknown reason'}",
        )
    yield state.repository


def get_explainer() -> Any:
    if state.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainer not loaded",
        )
    return state.explainer


async def record_actor(request: Request, user: User = Depends(get_current_user)) -> User:
    """Put the authenticated username on request state for the audit log."""
    request.state.actor = user.username
    return user


# Typed as Any because either repository implementation may be behind it; the
# Protocol in backend.app.repository is the contract.
RepositoryDep = Annotated[Any, Depends(get_repository)]
ExplainerDep = Annotated[Any, Depends(get_explainer)]

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@auth_router.post("/token", response_model=schemas.Token)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> schemas.Token:
    user = DEMO_USERS.get(form.username)
    # Verify even when the user is unknown, so response time does not reveal
    # which usernames exist.
    hashed = user.hashed_password if user else hash_password("no-such-user")
    if not verify_password(form.password, hashed) or user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    settings = get_settings()
    return schemas.Token(
        access_token=create_access_token(user.username, user.role),
        expires_in_minutes=settings.access_token_expire_minutes,
        role=user.role.value,
    )


@auth_router.get("/me", response_model=schemas.CurrentUser)
async def me(user: User = Depends(get_current_user)) -> schemas.CurrentUser:
    return schemas.CurrentUser(
        username=user.username,
        role=user.role.value,
        may_view_individuals=user.may_view_individuals,
        may_assign_interventions=user.may_assign_interventions,
    )


students_router = APIRouter(prefix="/api/v1/students", tags=["students"])


@students_router.get("", response_model=schemas.Page[schemas.StudentSummary])
async def list_students(
    repository: RepositoryDep,
    user: User = Depends(require_individual_access),
    band: schemas.BandKey | None = None,
    direction: schemas.RiskDirection | None = None,
    module: str | None = None,
    search: str | None = Query(default=None, max_length=64),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> schemas.Page[schemas.StudentSummary]:
    rows, total = repository.list_students(band, direction, module, search, page, page_size)
    return schemas.Page(
        data=services.student_summaries(rows),
        meta=schemas.Meta(
            total=total, page=page, page_size=page_size, model_version=repository.model_version
        ),
    )


@students_router.get("/{student_code}", response_model=schemas.StudentProfile)
async def get_student(
    student_code: str,
    repository: RepositoryDep,
    user: User = Depends(record_actor),
) -> schemas.StudentProfile:
    if not user.may_view_individuals:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{user.role}' may see aggregates only, not individual students.",
        )
    profile = services.student_profile(repository, student_code)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return profile


@students_router.get("/{student_code}/risk-history", response_model=list[schemas.TrajectoryPoint])
async def risk_history(
    student_code: str,
    repository: RepositoryDep,
    user: User = Depends(record_actor),
) -> list[schemas.TrajectoryPoint]:
    if not user.may_view_individuals:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Aggregates only")
    profile = services.student_profile(repository, student_code)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return profile.trajectory


@students_router.get("/{student_code}/explanation", response_model=schemas.ExplanationResponse)
async def get_explanation(
    student_code: str,
    repository: RepositoryDep,
    explainer: ExplainerDep,
    user: User = Depends(record_actor),
    as_of: int | None = Query(default=None, description="Checkpoint day; defaults to the latest."),
) -> schemas.ExplanationResponse:
    if not user.may_view_individuals:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Aggregates only")
    result = services.explanation(repository, explainer, student_code, as_of)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Student or checkpoint not found"
        )
    return result


@students_router.get(
    "/{student_code}/recommendations", response_model=schemas.RecommendationsResponse
)
async def get_recommendations(
    student_code: str,
    repository: RepositoryDep,
    explainer: ExplainerDep,
    user: User = Depends(record_actor),
) -> schemas.RecommendationsResponse:
    if not user.may_view_individuals:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Aggregates only")
    result = services.recommendations(repository, explainer, student_code)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return result


@students_router.get(
    "/{student_code}/interventions", response_model=list[schemas.InterventionRecord]
)
async def list_interventions(
    student_code: str,
    repository: RepositoryDep,
    user: User = Depends(record_actor),
) -> list[schemas.InterventionRecord]:
    if not user.may_view_individuals:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Aggregates only")
    return [
        schemas.InterventionRecord(**vars(item)) for item in repository.interventions(student_code)
    ]


@students_router.post(
    "/{student_code}/interventions",
    response_model=schemas.InterventionRecord,
    status_code=status.HTTP_201_CREATED,
)
async def assign_intervention(
    student_code: str,
    payload: schemas.InterventionAssignment,
    repository: RepositoryDep,
    user: User = Depends(record_actor),
) -> schemas.InterventionRecord:
    """Assign a recommended action. This is the human act ADR-0005 requires."""
    if not user.may_assign_interventions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{user.role}' may not assign interventions.",
        )
    if repository.student_rows(student_code).empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    from dropout_ews.interventions.engine import load_catalog

    catalog = load_catalog()
    matched = next(
        (item for item in catalog.interventions if item.key == payload.intervention_key), None
    )
    if matched is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown intervention '{payload.intervention_key}'",
        )
    record = repository.add_intervention(
        student_code,
        matched.key,
        matched.title,
        payload.assigned_to,
        user.username,
        payload.note,
    )
    return schemas.InterventionRecord(**vars(record))


predict_router = APIRouter(prefix="/api/v1", tags=["prediction"])


@predict_router.post("/predict", response_model=schemas.PredictResponse)
async def predict(
    payload: schemas.PredictRequest,
    repository: RepositoryDep,
    user: User = Depends(require_individual_access),
) -> schemas.PredictResponse:
    """Score a student on demand.

    Takes a student reference, never a raw feature vector: features must come from
    the shared builder, or a caller could submit a vector no real student could
    produce and the as-of guarantee would mean nothing.
    """
    features = repository.feature_row(payload.student_code, payload.checkpoint_day)
    if features.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Student or checkpoint not found"
        )
    rows = repository.student_rows(payload.student_code)
    row = (
        rows.iloc[-1]
        if payload.checkpoint_day is None
        else rows[rows["checkpoint_day"] == payload.checkpoint_day].iloc[0]
    )
    return schemas.PredictResponse(
        student_code=payload.student_code,
        risk=services.build_risk_assessment(
            row["probability"], row["band"], int(row["checkpoint_day"]), repository.model_version
        ),
        features_hash=services.features_hash(features),
    )


dashboard_router = APIRouter(prefix="/api/v1", tags=["dashboard"])


@dashboard_router.get("/dashboard/statistics", response_model=schemas.DashboardStatistics)
async def dashboard_statistics(
    repository: RepositoryDep, user: User = Depends(require_any_role)
) -> schemas.DashboardStatistics:
    return services.dashboard_statistics(repository)


@dashboard_router.get("/analytics/risk-distribution", response_model=list[schemas.DistributionBin])
async def risk_distribution(
    repository: RepositoryDep,
    user: User = Depends(require_any_role),
    bins: int = Query(default=20, ge=5, le=50),
) -> list[schemas.DistributionBin]:
    return services.risk_distribution(repository, bins)


@dashboard_router.get("/analytics/scatter", response_model=list[schemas.ScatterPoint])
async def scatter(
    repository: RepositoryDep,
    user: User = Depends(require_any_role),
    x: str = Query(default="clicks_28d"),
) -> list[schemas.ScatterPoint]:
    try:
        return services.scatter(repository, x)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown field '{x}'. Allowed: {services.known_scatter_fields()}",
        ) from exc


@dashboard_router.get(
    "/analytics/feature-importance", response_model=list[schemas.FeatureImportance]
)
async def feature_importance(
    repository: RepositoryDep,
    explainer: ExplainerDep,
    user: User = Depends(require_any_role),
    limit: int = Query(default=20, ge=5, le=60),
) -> list[schemas.FeatureImportance]:
    return services.feature_importance(repository, explainer, limit)


alerts_router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@alerts_router.get("", response_model=list[schemas.AlertOut])
async def list_alerts(
    repository: RepositoryDep,
    user: User = Depends(require_individual_access),
    acknowledged: bool | None = None,
    severity: str | None = None,
) -> list[schemas.AlertOut]:
    return [
        schemas.AlertOut(**vars(alert))
        for alert in repository.alerts(acknowledged=acknowledged, severity=severity)
    ]


@alerts_router.post("/{alert_id}/acknowledge", response_model=schemas.AlertOut)
async def acknowledge_alert(
    alert_id: int, repository: RepositoryDep, user: User = Depends(require_individual_access)
) -> schemas.AlertOut:
    try:
        return schemas.AlertOut(**vars(repository.acknowledge_alert(alert_id, user.username)))
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found"
        ) from exc


model_router = APIRouter(prefix="/api/v1/model", tags=["model"])


@model_router.get("/info", response_model=schemas.ModelInfo)
async def model_info(
    repository: RepositoryDep, user: User = Depends(require_any_role)
) -> schemas.ModelInfo:
    return services.model_info(repository, state.metadata)


admin_router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@admin_router.get("/audit-log")
async def read_audit_log(
    user: User = Depends(require_admin),
    actor: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, object]]:
    """Who looked at which student, and when (ETHICS.md). Admin only."""
    entries = audit_log.for_actor(actor) if actor else audit_log.entries
    return [
        {
            "timestamp": entry.timestamp.isoformat(),
            "request_id": entry.request_id,
            "actor": entry.actor,
            "action": entry.action,
            "resource": entry.resource,
            "status_code": entry.status_code,
        }
        for entry in entries[-limit:]
    ]


system_router = APIRouter(tags=["system"])


@system_router.get("/health", response_model=schemas.HealthResponse)
async def health() -> schemas.HealthResponse:
    """Liveness and readiness, reported separately.

    Unauthenticated on purpose: a load balancer cannot hold a token.
    """
    # With the database backend the repository is built per request, so its
    # absence from state is expected and must not read as "not ready".
    storage_ready = state.repository is not None or state.backend == "database"
    loaded = storage_ready and state.explainer is not None and state.load_error is None
    version = getattr(state.metadata, "model_version", None)
    return schemas.HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=version,
        data_loaded=storage_ready,
        environment=get_settings().environment,
    )


@system_router.get("/metrics")
async def metrics(user: User = Depends(require_any_role)) -> dict[str, object]:
    """Minimal operational counters.

    Authenticated, because band counts over a cohort are still information about
    that cohort.
    """
    repository = cast("Any", state.repository)
    if repository is None:
        # The database backend builds its repository per request, so /metrics
        # reports what it can rather than claiming the model is unloaded.
        return {"model_loaded": state.backend == "database", "backend": state.backend}
    return {
        "model_loaded": True,
        "model_version": repository.model_version,
        "scored_rows": len(repository.cohort),
        "students": len(repository.latest_rows()),
        "open_alerts": len(repository.alerts(acknowledged=False)),
        "audit_entries": len(audit_log.entries),
        "bands": dict(repository.band_counts()),
    }


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Student Dropout Early-Warning API",
        version="0.1.0",
        description=(
            "Estimates the probability that a student withdraws within 30 days of a "
            "checkpoint, with non-causal explanations and supportive intervention "
            "suggestions. Every risk figure is returned with the model version and a "
            "disclaimer. See docs/MODEL_CARD.md for limitations."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        # An allowlist rather than "*": credentials are sent on these requests.
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )
    for router in (
        auth_router,
        students_router,
        predict_router,
        dashboard_router,
        alerts_router,
        model_router,
        admin_router,
        system_router,
    ):
        app.include_router(router)
    if settings.environment == "local":
        app.state.demo_users = list(DEMO_USERS)
    return app


app = create_app()
