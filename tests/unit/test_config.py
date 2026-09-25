"""Tests for configuration loading and the invariants the config must uphold.

These are Phase 1's substantive tests. They are not smoke tests: several of
them encode governance constraints from ADR-0005 and docs/ETHICS.md that must
not be configurable away, and the band-ordering tests protect the risk
classification from silently producing nonsense.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from dropout_ews.config.settings import (
    CONFIG_DIR,
    PROJECT_ROOT,
    FeatureConfig,
    Settings,
    ThresholdConfig,
    load_feature_config,
    load_threshold_config,
)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_project_root_resolves_to_repository_root() -> None:
    """PROJECT_ROOT is derived by walking up from settings.py; if the package
    is ever moved, this catches it before every data path breaks."""
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src" / "dropout_ews").is_dir()


# ---------------------------------------------------------------------------
# features.yaml
# ---------------------------------------------------------------------------


def test_shipped_feature_config_is_valid() -> None:
    config = load_feature_config()
    assert config.task.horizon_days > 0
    assert config.task.checkpoints == sorted(config.task.checkpoints)


def test_feature_config_matches_task_spec() -> None:
    """The shipped config must agree with docs/TASK_SPEC.md.

    The spec is authoritative; drift between the two is a bug, and it is the
    kind of drift nobody notices until a reported metric means something
    different from what the documentation claims.
    """
    config = load_feature_config()
    assert config.task.checkpoints == [30, 60, 90, 120, 150, 180]
    assert config.task.horizon_days == 30
    assert config.task.drop_censored_rows is True
    assert config.evaluation.target_recall == pytest.approx(0.80)
    assert config.evaluation.primary_metric == "average_precision"


def test_outcome_columns_are_forbidden() -> None:
    """The columns that would leak the label must be denied (ADR-0003)."""
    forbidden = load_feature_config().forbidden
    for column in ("date_unregistration", "withdrawal_day", "final_result"):
        assert column in forbidden.exact
    assert "withdraw*" in forbidden.patterns


def test_checkpoints_must_be_ascending_and_unique() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        FeatureConfig.model_validate(
            {
                "task": {"checkpoints": [60, 30], "horizon_days": 30},
                "evaluation": {"target_recall": 0.8},
                "features": {},
                "forbidden": {},
            }
        )


def test_allowlist_may_not_contain_a_forbidden_column() -> None:
    """A feature that is also outcome-adjacent must fail loudly at load time,
    not silently train a leaking model."""
    with pytest.raises(ValidationError, match="allowlist and the forbidden list"):
        FeatureConfig.model_validate(
            {
                "task": {"checkpoints": [30], "horizon_days": 30},
                "evaluation": {"target_recall": 0.8},
                "features": {"level": ["attendance_pct", "final_result"]},
                "forbidden": {"exact": ["final_result"]},
            }
        )


def test_allowlist_forbidden_check_also_matches_patterns() -> None:
    with pytest.raises(ValidationError, match="allowlist and the forbidden list"):
        FeatureConfig.model_validate(
            {
                "task": {"checkpoints": [30], "horizon_days": 30},
                "evaluation": {"target_recall": 0.8},
                "features": {"trend": ["gpa_slope", "withdraw_risk_prior"]},
                "forbidden": {"patterns": ["withdraw*"]},
            }
        )


# ---------------------------------------------------------------------------
# thresholds.yaml
# ---------------------------------------------------------------------------


def test_shipped_threshold_config_is_valid() -> None:
    config = load_threshold_config()
    assert [b.key for b in config.bands] == ["low", "medium", "high", "critical"]


def test_shipped_thresholds_are_flagged_uncalibrated() -> None:
    """The default cutoffs are placeholders. Until the Phase 6 calibration
    procedure runs, the config must not claim otherwise — this guards against
    presenting arbitrary numbers as validated."""
    assert load_threshold_config().calibrated is False


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.00, "low"),
        (0.14, "low"),
        (0.15, "medium"),  # boundary belongs to the upper band
        (0.34, "medium"),
        (0.35, "high"),
        (0.59, "high"),
        (0.60, "critical"),
        (1.00, "critical"),
    ],
)
def test_band_for_assigns_expected_band(probability: float, expected: str) -> None:
    assert load_threshold_config().band_for(probability).key == expected


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_band_for_rejects_out_of_range_probability(probability: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        load_threshold_config().band_for(probability)


def _threshold_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "name": "test",
        "bands": [
            {"key": "low", "label": "Low", "min_probability": 0.0, "color": "#0", "action": "a"},
            {"key": "high", "label": "High", "min_probability": 0.5, "color": "#1", "action": "b"},
        ],
    }
    payload.update(overrides)
    return payload


def test_bands_must_start_at_zero() -> None:
    """Every probability must land in some band. If the lowest cutoff is above
    zero, low-risk students fall through and get no classification."""
    payload = _threshold_payload()
    payload["bands"][0]["min_probability"] = 0.1  # type: ignore[index]
    with pytest.raises(ValidationError, match=r"must start at probability 0\.0"):
        ThresholdConfig.model_validate(payload)


def test_bands_must_be_strictly_ascending() -> None:
    payload = _threshold_payload()
    payload["bands"][1]["min_probability"] = 0.0  # type: ignore[index]
    with pytest.raises(ValidationError, match="strictly ascending"):
        ThresholdConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Governance constraints (ADR-0005, docs/ETHICS.md)
# ---------------------------------------------------------------------------


def test_human_review_cannot_be_disabled_via_config() -> None:
    """Human-in-the-loop is a hard constraint. The config file can state the
    intent but must not be able to switch it off."""
    with pytest.raises(ValidationError, match="human review"):
        ThresholdConfig.model_validate(
            _threshold_payload(governance={"require_human_review_before_intervention": False})
        )


def test_automated_punitive_actions_cannot_be_enabled() -> None:
    with pytest.raises(ValidationError, match="punitive"):
        ThresholdConfig.model_validate(
            _threshold_payload(governance={"auto_punitive_actions_permitted": True})
        )


def test_shipped_config_enforces_human_review() -> None:
    governance = load_threshold_config().governance
    assert governance.require_human_review_before_intervention is True
    assert governance.auto_punitive_actions_permitted is False


# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------


def test_local_defaults_load() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.environment == "local"
    assert settings.enable_llm_narrative is False


def test_production_rejects_the_default_secret_key() -> None:
    with pytest.raises(ValidationError, match="secret_key must be set"):
        Settings(_env_file=None, environment="production")  # type: ignore[call-arg]


def test_production_rejects_a_short_secret_key() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(_env_file=None, environment="production", secret_key="short")  # type: ignore[call-arg]


def test_llm_flag_requires_an_api_key() -> None:
    """Fail at startup rather than at the first request, and point the operator
    at the working fallback."""
    with pytest.raises(ValidationError, match="deterministic narrative templates"):
        Settings(_env_file=None, enable_llm_narrative=True)  # type: ignore[call-arg]


def test_llm_flag_accepted_when_key_present() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, enable_llm_narrative=True, anthropic_api_key="test-key"
    )
    assert settings.enable_llm_narrative is True


# ---------------------------------------------------------------------------
# Documentation consistency
# ---------------------------------------------------------------------------


def test_every_adr_is_referenced_by_the_docs_index() -> None:
    """ADRs that nothing links to get forgotten and then contradicted."""
    index = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    for adr in sorted((PROJECT_ROOT / "docs" / "adr").glob("[0-9]*.md")):
        assert adr.name in index, f"{adr.name} is not listed in docs/README.md"


def test_config_yaml_files_are_parseable_and_shipped_as_package_data() -> None:
    for name in ("features.yaml", "thresholds.yaml"):
        path = CONFIG_DIR / name
        assert path.is_file()
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
