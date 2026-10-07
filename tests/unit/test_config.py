"""Tests for configuration loading and the invariants the config must uphold.

These are Phase 1's substantive tests. They are not smoke tests: several of
them encode governance constraints from ADR-0005 and docs/ETHICS.md that must
not be configurable away, and the band-ordering tests protect the risk
classification from silently producing nonsense.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

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


def test_shipped_thresholds_are_calibrated_and_say_how() -> None:
    """The shipped bands were derived by the Phase 6 procedure, so the config
    now claims calibration — and must carry a name that identifies the
    derivation rather than the Phase 1 placeholder.

    Until Phase 6 this asserted ``calibrated is False``. The invariant changed
    when the bands were genuinely derived, but it did not disappear: the pairing
    below is what stops arbitrary numbers being presented as validated.
    """
    config = load_threshold_config()
    assert config.calibrated is True
    assert "placeholder" not in config.name
    assert config.version >= 2


def test_uncalibrated_config_must_not_claim_a_derived_name() -> None:
    """The complement: a config flagged uncalibrated has to be labelled as a
    placeholder, so the two fields cannot drift apart."""
    from dropout_ews.config.settings import ThresholdConfig

    payload = {
        "version": 1,
        "name": "default-uncalibrated-placeholder",
        "calibrated": False,
        "bands": [
            {"key": "low", "label": "Low", "min_probability": 0.0, "color": "#0", "action": "a"},
            {"key": "high", "label": "High", "min_probability": 0.5, "color": "#1", "action": "b"},
        ],
    }
    config = ThresholdConfig.model_validate(payload)
    assert config.calibrated is False
    assert "placeholder" in config.name


def test_rapid_increase_threshold_suits_the_calibrated_probability_scale() -> None:
    """Calibrated probabilities reach only about 0.33 on test, so the Phase 1
    placeholder delta of 0.15 fired on 0.11% of checkpoint transitions. An alert
    rule that never fires is worse than none, because it reads as coverage."""
    rule = load_threshold_config().alerts.rapid_increase
    assert rule.enabled is True
    assert rule.min_delta <= 0.10


def _fixed_band_config() -> ThresholdConfig:
    """A config with known cutoffs, for testing ``band_for`` logic.

    Deliberately not the shipped config. Band cutoffs are recalibrated whenever
    the model is retrained, so asserting logic against live values couples the
    test to the one thing that is meant to change — and it did break when Phase 6
    replaced the Phase 1 placeholders.
    """
    return ThresholdConfig.model_validate(
        {
            "version": 99,
            "name": "fixture",
            "calibrated": True,
            "bands": [
                {"key": "low", "label": "L", "min_probability": 0.0, "color": "#0", "action": "a"},
                {
                    "key": "medium",
                    "label": "M",
                    "min_probability": 0.15,
                    "color": "#1",
                    "action": "b",
                },
                {
                    "key": "high",
                    "label": "H",
                    "min_probability": 0.35,
                    "color": "#2",
                    "action": "c",
                },
                {
                    "key": "critical",
                    "label": "C",
                    "min_probability": 0.60,
                    "color": "#3",
                    "action": "d",
                },
            ],
        }
    )


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
    assert _fixed_band_config().band_for(probability).key == expected


def test_shipped_band_boundaries_assign_to_the_upper_band() -> None:
    """The same boundary rule, checked against whatever cutoffs are live, so a
    recalibration cannot silently invert the convention."""
    config = load_threshold_config()
    for band in config.bands:
        assert config.band_for(band.min_probability).key == band.key


def test_shipped_bands_cover_the_calibrated_probability_range() -> None:
    """Calibrated probabilities on test span roughly 0 to 0.33. Every band must
    be reachable within that range, or a tier exists that no student can enter."""
    config = load_threshold_config()
    assert all(band.min_probability < 0.34 for band in config.bands)


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


# Every field below is also readable from the process environment, so a test that
# asserts a *default* has to clear the environment as well as disable .env.
# `_env_file=None` only does the latter. CI sets ENVIRONMENT=ci, which made
# test_local_defaults_load fail with "assert 'ci' == 'local'" -- the test had
# always been environment-dependent and nothing had exposed it, because locally
# the variable is unset and .env happens to say "local" anyway.
SETTINGS_ENV_VARS = (
    "ENVIRONMENT",
    "LOG_LEVEL",
    "DATABASE_URL",
    "SECRET_KEY",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "CORS_ORIGINS",
    "FORCE_HTTPS",
    "REPOSITORY_BACKEND",
    "MODEL_DIR",
    "ACTIVE_MODEL_VERSION",
    "ENABLE_LLM_NARRATIVE",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every settings override so a default actually reads as a default."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_local_defaults_load(clean_env: None) -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.environment == "local"
    assert settings.enable_llm_narrative is False


def test_production_rejects_the_default_secret_key(clean_env: None) -> None:
    with pytest.raises(ValidationError, match="secret_key must be set"):
        Settings(_env_file=None, environment="production")  # type: ignore[call-arg]


def test_production_rejects_a_short_secret_key(clean_env: None) -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(_env_file=None, environment="production", secret_key="short")  # type: ignore[call-arg]


def test_llm_flag_requires_an_api_key(clean_env: None) -> None:
    """Fail at startup rather than at the first request, and point the operator
    at the working fallback.

    Needs `clean_env`: an ANTHROPIC_API_KEY in the environment would satisfy the
    validator and the expected error would never be raised.
    """
    with pytest.raises(ValidationError, match="deterministic narrative templates"):
        Settings(_env_file=None, enable_llm_narrative=True)  # type: ignore[call-arg]


def test_llm_flag_accepted_when_key_present(clean_env: None) -> None:
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


# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------


def test_project_root_is_the_repository_root_in_a_source_checkout() -> None:
    """The default derivation. Correct for a checkout and an editable install."""
    from dropout_ews.config.settings import DATA_DIR, MODELS_DIR, PROJECT_ROOT

    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert MODELS_DIR == PROJECT_ROOT / "models"
    assert DATA_DIR == PROJECT_ROOT / "data"


def test_the_derivation_walks_four_levels_up_from_the_module() -> None:
    """src/dropout_ews/config/settings.py -> repository root. Pinned because the
    count is silently wrong if the module ever moves."""
    from dropout_ews.config.settings import _resolve_project_root

    module = Path("/repo/src/dropout_ews/config/settings.py")
    assert _resolve_project_root(None, module) == Path("/repo").resolve()


def test_the_derivation_is_wrong_for_a_non_editable_install() -> None:
    """The bug this override exists for, pinned as a fact rather than prose.

    `pip install ".[api]"` copies the package into site-packages, so the same four
    levels up land inside the virtualenv. In the Docker image MODELS_DIR became
    `/opt/venv/lib/python3.10/models`: the container started, passed its
    healthcheck, served /health, and reported `model_loaded: false` with the model
    mounted and readable at /app/models the whole time.
    """
    from dropout_ews.config.settings import _resolve_project_root

    installed = Path("/opt/venv/lib/python3.10/site-packages/dropout_ews/config/settings.py")
    derived = _resolve_project_root(None, installed)
    assert derived == Path("/opt/venv/lib/python3.10").resolve()
    assert derived.name != "site-packages"  # i.e. nowhere near the real root


def test_the_override_wins_over_the_derivation() -> None:
    from dropout_ews.config.settings import _resolve_project_root

    installed = Path("/opt/venv/lib/python3.10/site-packages/dropout_ews/config/settings.py")
    assert _resolve_project_root("/app", installed) == Path("/app").resolve()


def test_an_empty_override_falls_back_rather_than_resolving_to_cwd() -> None:
    """`DROPOUT_EWS_ROOT=` in a .env file yields an empty string. Treating that as
    a path would silently resolve to the working directory."""
    from dropout_ews.config.settings import _resolve_project_root

    module = Path("/repo/src/dropout_ews/config/settings.py")
    assert _resolve_project_root("", module) == Path("/repo").resolve()


def test_config_dir_follows_the_module_not_the_project_root() -> None:
    """`features.yaml` and `thresholds.yaml` are package data and ship inside the
    wheel, so CONFIG_DIR must track the module. Tying it to the overridable root
    would break the installed package in the one case the override exists to fix.
    """
    from dropout_ews.config.settings import CONFIG_DIR

    assert (CONFIG_DIR / "features.yaml").is_file()
    assert CONFIG_DIR.name == "config"
    assert CONFIG_DIR.parent.name == "dropout_ews"


def test_the_override_takes_effect_in_a_fresh_interpreter() -> None:
    """The pure function above cannot show that the env var is actually read at
    import time. A subprocess can, without reloading the module in this one --
    reloading replaces `get_settings`, and an earlier version of this test broke
    an unrelated LLM test that way.
    """
    import json
    import subprocess
    import sys

    env = {**os.environ, "DROPOUT_EWS_ROOT": str(Path(tempfile.gettempdir()) / "ews-root")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json;from dropout_ews.config.settings import PROJECT_ROOT,MODELS_DIR,DATA_DIR;"
            "print(json.dumps([str(PROJECT_ROOT),str(MODELS_DIR),str(DATA_DIR)]))",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    root, models, data = json.loads(result.stdout)
    expected = (Path(tempfile.gettempdir()) / "ews-root").resolve()
    assert Path(root) == expected
    assert Path(models) == expected / "models"
    assert Path(data) == expected / "data"
