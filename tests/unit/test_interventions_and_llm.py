"""Tests for the intervention engine and the optional LLM layer.

The governance constraints in ADR-0005 are only real if code enforces them, so
these tests attack them directly: a punitive catalog entry must be rejected at
load time, nothing may execute without human review, and the LLM prompt must
refuse to carry an identifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import yaml

from dropout_ews.explainability.llm_summary import (
    SYSTEM_PROMPT,
    CaseNote,
    PromptSafetyError,
    build_prompt,
    generate_case_note,
    template_case_note,
)
from dropout_ews.explainability.narratives import (
    EXPLANATION_DISCLAIMER,
    RenderedExplanation,
    RenderedFactor,
)
from dropout_ews.interventions.engine import (
    FALLBACK_FACTOR,
    PunitiveInterventionError,
    find_forbidden_language,
    load_catalog,
    recommend,
)


def _factor(
    feature: str = "clicks_7d",
    label: str = "Course activity in the last week",
    impact: str = "High",
    direction: str = "increases",
    factor: str | None = "low_engagement",
    actionable: bool = True,
) -> RenderedFactor:
    return RenderedFactor(
        feature=feature,
        label=label,
        impact=impact,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
        actionable=actionable,
        factor=factor,
        sentence=f"{label}: the model weighted this heavily and it contributed to a higher estimate for this student.",
        shap_value=0.5 if direction == "increases" else -0.5,
    )


def _explanation(*factors: RenderedFactor) -> RenderedExplanation:
    return RenderedExplanation(
        risk_factors=[f for f in factors if f.direction == "increases" and f.actionable],
        protective_factors=[f for f in factors if f.direction == "decreases" and f.actionable],
        context_factors=[f for f in factors if not f.actionable],
    )


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------


def test_shipped_catalog_loads_and_is_supportive() -> None:
    catalog = load_catalog()
    assert len(catalog.interventions) >= 10
    for intervention in catalog.interventions:
        assert not find_forbidden_language(f"{intervention.title} {intervention.description}")


def test_every_narrative_factor_maps_to_at_least_one_intervention() -> None:
    """A factor shown to staff with no available action is a dead end."""
    from dropout_ews.explainability.narratives import FEATURE_META

    catalog = load_catalog()
    referenced = {meta.factor for meta in FEATURE_META.values() if meta.factor}
    uncovered = sorted(referenced - catalog.covered_factors)
    assert not uncovered, f"risk factors with no intervention: {uncovered}"


def test_punitive_entry_is_rejected_at_load_time(tmp_path: Path) -> None:
    """The guard has to fail closed: a punitive entry must stop the process, not
    be quietly served."""
    payload = {
        "interventions": [
            {
                "key": "bad",
                "title": "Formal warning",
                "factors": ["low_engagement"],
                "intensity": "light",
                "owner_role": "admin",
                "description": "Issue a disciplinary warning to the student.",
                "typical_effort_minutes": 5,
            }
        ],
        "band_guidance": {"high": {"max_interventions": 1, "note": "n"}},
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PunitiveInterventionError, match="disciplinary"):
        load_catalog(path)


def test_surveillance_entry_is_rejected(tmp_path: Path) -> None:
    """Punishment is the obvious failure mode; increased surveillance is the
    subtler one and is denied too."""
    payload = {
        "interventions": [
            {
                "key": "bad",
                "title": "Enhanced tracking",
                "factors": ["low_engagement"],
                "intensity": "light",
                "owner_role": "admin",
                "description": "Monitor closely and inform parents of the pattern.",
                "typical_effort_minutes": 5,
            }
        ],
        "band_guidance": {"high": {"max_interventions": 1, "note": "n"}},
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PunitiveInterventionError):
        load_catalog(path)


def test_supportive_phrasing_containing_a_denied_word_is_permitted() -> None:
    """ "Reducing load without penalty" is the most supportive line in the catalog
    and must not be blocked. A naive substring check did block it."""
    assert find_forbidden_language("Reduce load without penalty where allowed") == []
    assert find_forbidden_language("Apply a penalty for non-submission") == ["penalty"]


def test_duplicate_intervention_keys_are_rejected(tmp_path: Path) -> None:
    entry = {
        "key": "same",
        "title": "Advisor check-in",
        "factors": ["low_engagement"],
        "intensity": "light",
        "owner_role": "counsellor",
        "description": "A friendly message.",
        "typical_effort_minutes": 5,
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump({"interventions": [entry, dict(entry)]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate intervention keys"):
        load_catalog(path)


# ---------------------------------------------------------------------------
# Recommendation behaviour
# ---------------------------------------------------------------------------


def test_low_band_recommends_no_outreach() -> None:
    """Flagging a low-risk student for contact would be disproportionate."""
    recommendations, note = recommend(["low_engagement"], "low")
    assert recommendations == []
    assert "no outreach" in note.lower()


def test_band_caps_the_number_of_recommendations() -> None:
    factors = ["low_engagement", "academic_difficulty", "missed_assessments", "workload"]
    for band, cap in (("medium", 1), ("high", 2), ("critical", 3)):
        recommendations, _ = recommend(factors, band)
        assert len(recommendations) <= cap, f"{band} exceeded its cap"


def test_recommendations_follow_factor_impact_order() -> None:
    """The action addressing the most heavily weighted factor should come first."""
    recommendations, _ = recommend(["academic_difficulty", "low_engagement"], "critical")
    assert recommendations[0].matched_factor == "academic_difficulty"


def test_lighter_touch_actions_come_first_within_a_factor() -> None:
    """Escalating straight to intensive support on a probabilistic signal is not
    proportionate."""
    recommendations, _ = recommend(["academic_difficulty"], "critical")
    intensities = [r.intervention.intensity for r in recommendations]
    assert intensities == sorted(intensities, key=lambda i: {"light": 0, "moderate": 1}[i])


def test_every_recommendation_requires_human_review_and_is_not_executed() -> None:
    """The hard constraint from ADR-0005."""
    recommendations, _ = recommend(["low_engagement"], "critical")
    assert recommendations
    for recommendation in recommendations:
        assert recommendation.status == "recommended"
        assert recommendation.requires_human_review is True
        assert recommendation.to_dict()["requires_human_review"] is True


def test_no_duplicate_interventions_across_factors() -> None:
    """Several factors map to the same action; a counsellor should not see it
    twice."""
    recommendations, _ = recommend(
        ["low_engagement", "disengaged", "declining_engagement"], "critical"
    )
    keys = [r.intervention.key for r in recommendations]
    assert len(keys) == len(set(keys))


def test_empty_factor_list_falls_back_rather_than_returning_nothing() -> None:
    """A flagged student with no standout factor still needs an offer of help."""
    recommendations, _ = recommend([], "critical")
    assert recommendations
    assert recommendations[0].matched_factor == FALLBACK_FACTOR


def test_unmatched_factors_fall_back() -> None:
    recommendations, _ = recommend(["some_factor_with_no_actions"], "high")
    assert recommendations
    assert recommendations[0].matched_factor == FALLBACK_FACTOR


def test_unknown_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown risk band"):
        recommend(["low_engagement"], "catastrophic")


def test_rationale_uses_contribution_language() -> None:
    recommendations, _ = recommend(["low_engagement"], "critical")
    rationale = recommendations[0].rationale.lower()
    assert "the model weighted" in rationale
    assert "caused" not in rationale


# ---------------------------------------------------------------------------
# LLM prompt safety
# ---------------------------------------------------------------------------


def test_prompt_contains_only_derived_data() -> None:
    explanation = _explanation(
        _factor(), _factor(direction="decreases", factor="academic_difficulty")
    )
    recommendations, _ = recommend(explanation.matched_factors, "high")
    prompt = build_prompt(explanation, recommendations, "high")

    assert "Course activity in the last week" in prompt
    assert "impact: High" in prompt
    # Nothing that identifies a person or a raw value.
    assert "id_student" not in prompt
    assert "clicks_7d" not in prompt


def test_prompt_rejects_a_student_identifier() -> None:
    """Fail closed: a prompt is an outbound transfer, and a silent leak there is
    not recoverable."""
    leaky = _explanation(_factor(label="Activity for id_student 123456"))
    with pytest.raises(PromptSafetyError, match="student id field"):
        build_prompt(leaky, [], "high")


def test_prompt_rejects_a_long_numeric_identifier() -> None:
    leaky = _explanation(_factor(label="Student 8472913 recent activity"))
    with pytest.raises(PromptSafetyError, match="numeric identifier"):
        build_prompt(leaky, [], "high")


def test_prompt_rejects_an_email_address() -> None:
    leaky = _explanation(_factor(label="Contact at student@example.edu"))
    with pytest.raises(PromptSafetyError, match="email address"):
        build_prompt(leaky, [], "high")


@pytest.mark.parametrize("term", ["gender", "imd_band", "disability", "region", "deprivation"])
def test_prompt_rejects_protected_attributes(term: str) -> None:
    """These are excluded from the feature matrix, so they must not reach a
    prompt either (ETHICS.md)."""
    leaky = _explanation(_factor(label=f"Pattern by {term}"))
    with pytest.raises(PromptSafetyError, match="protected term"):
        build_prompt(leaky, [], "high")


def test_prompt_includes_protective_factors_when_present() -> None:
    """An explanation of only negatives misrepresents a student doing several
    things well."""
    explanation = _explanation(
        _factor(),
        _factor(
            label="Average assessment score", direction="decreases", factor="academic_difficulty"
        ),
    )
    prompt = build_prompt(explanation, [], "high")
    assert "lower risk" in prompt
    assert "Average assessment score" in prompt


# ---------------------------------------------------------------------------
# Case notes
# ---------------------------------------------------------------------------


def test_template_case_note_is_always_available() -> None:
    explanation = _explanation(_factor())
    recommendations, _ = recommend(explanation.matched_factors, "high")
    note = template_case_note(explanation, recommendations, "high")
    assert note.source == "template"
    assert note.disclaimer == EXPLANATION_DISCLAIMER
    assert "course activity in the last week" in note.text.lower()


def test_template_case_note_handles_no_meaningful_factors() -> None:
    note = template_case_note(RenderedExplanation([], [], []), [], "medium")
    assert "limited" in note.text.lower()


def test_case_note_uses_the_template_when_the_flag_is_off(monkeypatch) -> None:
    """Off by default: with the flag disabled the product must be fully
    functional without any network call."""
    from dropout_ews.config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("ENABLE_LLM_NARRATIVE", "false")
    try:
        note = generate_case_note(_explanation(_factor()), [], "high")
        assert note.source == "template"
    finally:
        settings_module.get_settings.cache_clear()


def test_case_note_falls_back_when_the_llm_call_fails(monkeypatch) -> None:
    """A support tool must not show a counsellor an error where a sentence
    belongs."""
    from dropout_ews.config import settings as settings_module

    class BrokenClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**_: object) -> object:
                raise RuntimeError("api down")

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("ENABLE_LLM_NARRATIVE", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    try:
        note = generate_case_note(_explanation(_factor()), [], "high", client=BrokenClient())
        assert note.source == "template"
        assert note.text
    finally:
        settings_module.get_settings.cache_clear()


def test_case_note_uses_the_llm_when_enabled_and_healthy(monkeypatch) -> None:
    from dropout_ews.config import settings as settings_module

    class Block:
        type: ClassVar[str] = "text"
        text: ClassVar[str] = "The model gave most weight to recent course activity."

    class Response:
        content: ClassVar[list[Block]] = [Block()]

    class WorkingClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kwargs: object) -> Response:
                # The system prompt must be the governed one, not ad hoc text.
                assert kwargs["system"] == SYSTEM_PROMPT
                return Response()

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("ENABLE_LLM_NARRATIVE", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    try:
        note = generate_case_note(_explanation(_factor()), [], "high", client=WorkingClient())
        assert note.source == "llm"
        assert "recent course activity" in note.text
    finally:
        settings_module.get_settings.cache_clear()


def test_case_note_carries_the_disclaimer_whichever_path_ran() -> None:
    for note in (
        template_case_note(_explanation(_factor()), [], "high"),
        CaseNote(text="x", source="llm"),
    ):
        assert note.disclaimer == EXPLANATION_DISCLAIMER
