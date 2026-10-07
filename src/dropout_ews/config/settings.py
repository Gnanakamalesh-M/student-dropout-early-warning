"""Typed configuration for the training pipeline and the API.

Two kinds of configuration live here:

* **Task/model parameters** (``features.yaml``, ``thresholds.yaml``) — versioned
  in git, because changing them changes what the model means. Loaded via
  :func:`load_feature_config` and :func:`load_threshold_config`.
* **Environment settings** (secrets, connection strings, feature flags) — never
  in git, read from the environment via :class:`Settings`.
"""

from __future__ import annotations

import fnmatch
import functools
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Repository root. Derived from this file's location by default --
# src/dropout_ews/config/settings.py -> up four levels -- which is correct for a
# source checkout and for an editable install.
#
# It is *wrong* for a non-editable install, and that is not hypothetical: in the
# Docker image `pip install ".[api]"` copies the package into site-packages, so
# the same four levels up land on `/opt/venv/lib/python3.10` and MODELS_DIR
# becomes `/opt/venv/lib/python3.10/models`. The container then started cleanly,
# served /health, and reported `model_loaded: false` -- the model was mounted at
# /app/models the whole time.
#
# DROPOUT_EWS_ROOT makes the root explicit for that case. Deliberately an env var
# rather than a cwd fallback or a walk-up looking for pyproject.toml: both guess,
# and a path that is guessed wrong fails the same silent way.
def _resolve_project_root(env_root: str | None, module_file: Path) -> Path:
    """Project root from the override if set, else derived from this module.

    Split out as a pure function so it is testable without reloading the module.
    An earlier test did reload it, which replaced ``get_settings`` -- other
    modules kept a reference to the old function and its own ``functools.cache``,
    so an unrelated LLM test failed only when the two ran in the same session.
    A path helper is not worth that kind of spooky failure.
    """
    if env_root:
        return Path(env_root).resolve()
    return module_file.resolve().parents[3]


PROJECT_ROOT = _resolve_project_root(os.environ.get("DROPOUT_EWS_ROOT"), Path(__file__))

# Always next to this module: config travels with the package, unlike data.
CONFIG_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# features.yaml
# ---------------------------------------------------------------------------


class TaskConfig(BaseModel):
    """The prediction task itself. See docs/TASK_SPEC.md."""

    checkpoints: list[int]
    horizon_days: int = Field(gt=0)
    drop_censored_rows: bool = True
    require_minimum_history: bool = True

    @field_validator("checkpoints")
    @classmethod
    def _sorted_and_unique(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("at least one checkpoint is required")
        if sorted(set(v)) != v:
            raise ValueError("checkpoints must be sorted ascending and unique")
        if v[0] <= 0:
            raise ValueError("checkpoints must be positive course-relative days")
        return v


class EvaluationConfig(BaseModel):
    primary_metric: str = "average_precision"

    alert_budget: float = Field(default=0.05, gt=0.0, le=1.0)
    """Share of the cohort an institution can follow up in one cycle.

    The primary operating control. Phase 5 showed a recall target is the wrong
    knob: 80% recall means flagging 47% of the cohort at 4.9% precision, which
    no institution can staff. See features.yaml for the measured tradeoff.
    """

    target_recall: float = Field(ge=0.0, le=1.0)
    """Kept for the published recall/precision tradeoff table, not as the
    operating point."""

    report_per_checkpoint: bool = True
    bootstrap_iterations: int = Field(default=1000, ge=0)
    random_seed: int = 42


class FeatureGroups(BaseModel):
    """The feature allowlist, grouped by kind. See ADR-0003 control #2.

    ``extra="forbid"`` is essential rather than stylistic. With Pydantic
    defaults, a group present in ``features.yaml`` but absent here is silently
    discarded, so the allowlist would quietly shrink and the model would train
    on fewer features than the config appears to declare. That happened during
    Phase 4 with the ``assessment`` and ``static`` groups: 15 features vanished
    without any error, and only a test pinning the expected count caught it.
    """

    model_config = ConfigDict(extra="forbid")

    level: list[str] = []
    recency: list[str] = []
    trend: list[str] = []
    volatility: list[str] = []
    assessment: list[str] = []
    relative: list[str] = []
    static: list[str] = []
    composite: list[str] = []

    # Flattening order. Declared explicitly so the feature-matrix column order
    # is stable and reproducible across runs.
    _GROUP_ORDER = (
        "level",
        "recency",
        "trend",
        "volatility",
        "assessment",
        "relative",
        "static",
        "composite",
    )

    def all_features(self) -> list[str]:
        """Flatten the allowlist, preserving group order."""
        return [name for group in self._GROUP_ORDER for name in getattr(self, group)]


class ForbiddenColumns(BaseModel):
    exact: list[str] = []
    patterns: list[str] = []


class FeatureConfig(BaseModel):
    task: TaskConfig
    evaluation: EvaluationConfig
    features: FeatureGroups
    forbidden: ForbiddenColumns
    excluded_by_ablation: list[str] = []
    """Builder outputs the model deliberately does not use.

    Distinct from ``forbidden``: these are safe to compute and would not leak,
    they simply failed to earn their place. Recorded so that every builder
    output is accounted for as either allowlisted, forbidden, or explicitly
    excluded — an unlisted omission is an accident rather than a decision.
    """

    @model_validator(mode="after")
    def _allowlist_and_forbidden_are_disjoint(self) -> FeatureConfig:
        """A column cannot be both permitted and forbidden.

        Catches the case where a feature is added to the allowlist without
        noticing it is outcome-adjacent.
        """
        allowed = set(self.features.all_features())
        clashes = allowed & set(self.forbidden.exact)
        for pattern in self.forbidden.patterns:
            clashes |= {name for name in allowed if fnmatch.fnmatch(name, pattern)}
        if clashes:
            raise ValueError(
                f"features present in both the allowlist and the forbidden list: {sorted(clashes)}"
            )
        return self


# ---------------------------------------------------------------------------
# thresholds.yaml
# ---------------------------------------------------------------------------

BandKey = Literal["low", "medium", "high", "critical"]


class RiskBand(BaseModel):
    key: BandKey
    label: str
    min_probability: float = Field(ge=0.0, le=1.0)
    color: str
    action: str


class BandEscalationRule(BaseModel):
    enabled: bool = True
    into_bands: list[BandKey] = []


class RapidIncreaseRule(BaseModel):
    enabled: bool = True
    min_delta: float = Field(default=0.15, gt=0.0, le=1.0)


class SustainedIncreaseRule(BaseModel):
    enabled: bool = True
    consecutive_checkpoints: int = Field(default=3, ge=2)


class AlertRules(BaseModel):
    band_escalation: BandEscalationRule = BandEscalationRule()
    rapid_increase: RapidIncreaseRule = RapidIncreaseRule()
    sustained_increase: SustainedIncreaseRule = SustainedIncreaseRule()


class Governance(BaseModel):
    require_human_review_before_intervention: bool = True
    auto_punitive_actions_permitted: bool = False

    @field_validator("require_human_review_before_intervention")
    @classmethod
    def _human_review_is_mandatory(cls, v: bool) -> bool:
        # A hard constraint from ADR-0005, deliberately not configurable away:
        # the config file can express the intent, but it cannot disable it.
        if not v:
            raise ValueError(
                "human review before intervention cannot be disabled (ADR-0005, ETHICS.md)"
            )
        return v

    @field_validator("auto_punitive_actions_permitted")
    @classmethod
    def _no_punitive_automation(cls, v: bool) -> bool:
        if v:
            raise ValueError("automated punitive actions are not permitted (ETHICS.md)")
        return v


class ThresholdConfig(BaseModel):
    version: int
    name: str
    calibrated: bool = False
    bands: list[RiskBand]
    alerts: AlertRules = AlertRules()
    governance: Governance = Governance()

    @model_validator(mode="after")
    def _bands_are_ordered_and_cover_zero(self) -> ThresholdConfig:
        if len(self.bands) < 2:
            raise ValueError("at least two risk bands are required")
        mins = [b.min_probability for b in self.bands]
        if mins != sorted(mins) or len(set(mins)) != len(mins):
            raise ValueError("band min_probability values must be strictly ascending")
        if mins[0] != 0.0:
            raise ValueError("the lowest band must start at probability 0.0")
        return self

    def band_for(self, probability: float) -> RiskBand:
        """Return the band a calibrated probability falls into.

        Bands are half-open ``[min, next_min)``, with the top band unbounded.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        selected = self.bands[0]
        for band in self.bands:
            if probability >= band.min_probability:
                selected = band
            else:
                break
        return selected


# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Environment-driven settings. See .env.example."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["local", "ci", "staging", "production"] = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://ews:ews@localhost:5432/ews"

    # Auth. The default is unusable on purpose: a real value must be supplied
    # outside local development, and the validator below enforces that.
    secret_key: str = "dev-only-insecure-change-me"
    access_token_expire_minutes: int = 30

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    """Browser origins allowed to call the API with credentials.

    Was hardcoded to the Vite dev server, which silently breaks any real
    deployment: the browser blocks the response and the dashboard shows a network
    error with nothing in the API log, because the request never reached a route.
    A validator below refuses a wildcard and refuses to leave localhost in place
    outside development.
    """

    force_https: bool = False
    """Send HSTS. Off by default because sending it over plain HTTP is wrong, and
    on a developer machine it would pin localhost to HTTPS in the browser for the
    duration of the max-age -- a confusing, persistent, self-inflicted outage."""

    repository_backend: Literal["parquet", "database"] = "parquet"
    """Which storage the API reads from.

    ``parquet`` scores on demand from the processed feature file and needs no
    database; ``database`` reads persisted predictions. Both satisfy the same
    repository interface, so this is configuration rather than a code path —
    which is the check on whether the Phase 8 seam was in the right place.
    """

    model_dir: Path = MODELS_DIR
    active_model_version: str | None = None

    # ADR-0005: the LLM case-note feature is optional and off by default. With
    # it disabled, deterministic templates render and the product is complete.
    enable_llm_narrative: bool = False
    anthropic_api_key: str | None = None

    @model_validator(mode="after")
    def _production_requires_real_secret(self) -> Settings:
        if self.environment in {"staging", "production"}:
            if self.secret_key == "dev-only-insecure-change-me":
                raise ValueError("secret_key must be set outside local development")
            if len(self.secret_key) < 32:
                raise ValueError("secret_key must be at least 32 characters")
        return self

    @model_validator(mode="after")
    def _production_cors_must_be_explicit(self) -> Settings:
        """A wildcard origin with credentials is not a policy, it is the absence
        of one -- and browsers reject the combination anyway, so it would fail at
        runtime rather than at startup. Failing at startup is better.
        """
        if "*" in self.cors_origins:
            raise ValueError(
                "cors_origins must not contain '*': credentials are sent on these "
                "requests, and browsers reject wildcard-with-credentials"
            )
        if self.environment in {"staging", "production"}:
            if not self.cors_origins:
                raise ValueError("cors_origins must be set outside local development")
            local = [o for o in self.cors_origins if "localhost" in o or "127.0.0.1" in o]
            if local:
                raise ValueError(
                    f"cors_origins still contains development origins {local}; set the "
                    "deployed dashboard origin instead"
                )
            insecure = [o for o in self.cors_origins if o.startswith("http://")]
            if insecure:
                raise ValueError(f"cors_origins must use https outside development: {insecure}")
        return self

    @model_validator(mode="after")
    def _llm_flag_requires_key(self) -> Settings:
        if self.enable_llm_narrative and not self.anthropic_api_key:
            raise ValueError(
                "enable_llm_narrative is true but anthropic_api_key is unset; "
                "leave the flag off to use deterministic narrative templates"
            )
        return self


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


@functools.cache
def load_feature_config(path: Path | None = None) -> FeatureConfig:
    """Load and validate ``features.yaml``."""
    return FeatureConfig.model_validate(_read_yaml(path or CONFIG_DIR / "features.yaml"))


@functools.cache
def load_threshold_config(path: Path | None = None) -> ThresholdConfig:
    """Load and validate ``thresholds.yaml``."""
    return ThresholdConfig.model_validate(_read_yaml(path or CONFIG_DIR / "thresholds.yaml"))


@functools.cache
def get_settings() -> Settings:
    """Load environment settings (cached)."""
    return Settings()
