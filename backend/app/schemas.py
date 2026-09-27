"""Request and response schemas.

Two conventions are enforced by type rather than by review:

* **Every response containing a risk figure carries ``model_version`` and
  ``disclaimer``.** :class:`RiskAssessment` bundles them, so a probability cannot
  be serialised on its own. A number that looks like a probability, detached from
  what produced it and from what it means, is the thing this project most needs
  not to ship.
* **No response carries a student name or contact detail**, because none exists
  in the data. ``student_code`` is an opaque identifier (ETHICS.md).
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from dropout_ews.explainability.narratives import EXPLANATION_DISCLAIMER

T = TypeVar("T")

BandKey = Literal["low", "medium", "high", "critical"]
RiskDirection = Literal["improving", "stable", "worsening", "unknown"]


class Meta(BaseModel):
    """Pagination and provenance envelope."""

    total: int
    page: int
    page_size: int
    model_version: str | None = None


class Page(BaseModel, Generic[T]):
    data: list[T]
    meta: Meta


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in_minutes: int
    role: str


class CurrentUser(BaseModel):
    username: str
    role: str
    may_view_individuals: bool
    may_assign_interventions: bool


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class RiskBandInfo(BaseModel):
    key: BandKey
    label: str
    min_probability: float
    color: str
    action: str


class RiskAssessment(BaseModel):
    """A risk figure, and everything needed to read it responsibly.

    ``disclaimer`` and ``model_version`` are required fields with no default that
    can be omitted, so it is not possible to return a probability without them.
    """

    model_config = ConfigDict(protected_namespaces=())

    probability: float = Field(ge=0.0, le=1.0)
    band: RiskBandInfo
    checkpoint_day: int
    model_version: str
    is_calibrated: bool
    """Whether the risk bands were derived by the calibration procedure rather
    than being placeholders. The UI shows a warning when false."""

    disclaimer: str = EXPLANATION_DISCLAIMER


class StudentSummary(BaseModel):
    """List-view row. No names: `student_code` is opaque."""

    student_code: str
    code_module: str
    code_presentation: str
    checkpoint_day: int
    probability: float
    band: BandKey
    risk_direction: RiskDirection


class TrajectoryPoint(BaseModel):
    checkpoint_day: int
    probability: float
    band: BandKey


class StudentProfile(BaseModel):
    student_code: str
    code_module: str
    code_presentation: str
    risk: RiskAssessment
    risk_direction: RiskDirection
    trajectory: list[TrajectoryPoint]
    engagement: dict[str, float | None]
    assessment: dict[str, float | None]
    context: dict[str, float | None]


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


class Factor(BaseModel):
    feature: str
    label: str
    impact: Literal["High", "Medium", "Low"]
    direction: Literal["increases", "decreases"]
    actionable: bool
    sentence: str


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    student_code: str
    checkpoint_day: int
    risk: RiskAssessment
    risk_factors: list[Factor]
    protective_factors: list[Factor]
    context_factors: list[Factor]
    """Contributors nobody can act on, such as how far through the course the
    student is. Separated so they do not crowd out actionable factors."""

    notes: list[str] = []
    disclaimer: str = EXPLANATION_DISCLAIMER
    attribution_basis: str = (
        "Attributions describe the model's pre-calibration score. They are "
        "reported as direction and relative impact, not as an effect on the "
        "probability."
    )


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------


class RecommendationOut(BaseModel):
    key: str
    title: str
    description: str
    intensity: Literal["light", "moderate"]
    owner_role: str
    typical_effort_minutes: int
    matched_factor: str
    rationale: str
    status: Literal["recommended"] = "recommended"
    requires_human_review: Literal[True] = True
    """Typed as a literal ``True``: the schema itself makes it impossible to
    serialise a recommendation that does not require review (ADR-0005)."""


class RecommendationsResponse(BaseModel):
    student_code: str
    band: BandKey
    band_guidance: str
    recommendations: list[RecommendationOut]
    case_note: str
    case_note_source: Literal["llm", "template"]
    disclaimer: str = EXPLANATION_DISCLAIMER


class InterventionAssignment(BaseModel):
    """A human assigning a recommended action to themselves or a colleague."""

    intervention_key: str
    assigned_to: str
    note: str | None = Field(default=None, max_length=2000)


class InterventionRecord(BaseModel):
    id: int
    student_code: str
    intervention_key: str
    title: str
    status: Literal["recommended", "assigned", "in_progress", "completed"]
    assigned_to: str | None
    assigned_by: str | None
    note: str | None
    created_at: datetime


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Ad-hoc scoring for a student already in the system.

    Deliberately does not accept a raw feature vector. Features must be built by
    the shared :class:`~dropout_ews.features.builder.FeatureBuilder` from event
    data, or the caller could submit a vector that no real student could produce
    and the as-of guarantee would mean nothing.
    """

    student_code: str
    checkpoint_day: int | None = Field(
        default=None, description="Defaults to the student's latest scored checkpoint."
    )


class PredictResponse(BaseModel):
    student_code: str
    risk: RiskAssessment
    features_hash: str
    """Hash of the feature vector scored, so a stored prediction can be tied to
    the exact inputs that produced it."""


# ---------------------------------------------------------------------------
# Dashboard, analytics, alerts
# ---------------------------------------------------------------------------


class BandCount(BaseModel):
    band: BandKey
    label: str
    count: int
    share: float


class DashboardStatistics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_students: int
    scored_checkpoints: int
    band_counts: list[BandCount]
    worsening_count: int
    """Students whose risk rose between their two most recent checkpoints."""

    open_alerts: int
    model_version: str
    is_calibrated: bool
    alert_budget: float
    """Share of the cohort the configured capacity allows contacting, so the UI
    can show how the critical band was sized."""


class DistributionBin(BaseModel):
    lower: float
    upper: float
    count: int


class ScatterPoint(BaseModel):
    x: float
    y: float
    band: BandKey


class FeatureImportance(BaseModel):
    feature: str
    label: str
    mean_abs_shap: float


class AlertOut(BaseModel):
    id: int
    student_code: str
    checkpoint_day: int
    reason: Literal["band_escalated", "rapid_increase", "sustained_increase"]
    severity: Literal["high", "critical"]
    probability: float
    previous_probability: float | None
    band: BandKey
    acknowledged: bool
    created_at: datetime


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_type: str
    created_at: str
    git_commit: str | None
    n_features: int
    imbalance_strategy: str
    calibration_method: str
    train_presentations: list[str]
    test_presentations: list[str]
    metrics: dict[str, object]
    bands: list[RiskBandInfo]
    is_calibrated: bool
    limitations: list[str]
    """Surfaced through the API rather than living only in the model card, so a
    client cannot present the metrics without them."""


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    """Distinguished from liveness on purpose: a process that is up but has no
    model must fail its readiness check rather than serve errors."""

    model_version: str | None
    data_loaded: bool
    environment: str
