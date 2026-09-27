"""The language lint — ADR-0005's central enforcement mechanism.

Causal and deterministic phrasing is the failure mode that makes an
early-warning system harmful rather than merely imperfect. A counsellor who reads
"low attendance caused this student's risk" will act on a causal claim the system
cannot support; one who reads "this student will drop out" will treat a
probability as a verdict.

Prose guidance does not survive contact with a deadline, so this is a test. It
scans the narrative templates, the intervention catalog, and the LLM system
prompt — every string that can reach a human.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dropout_ews.explainability import narratives
from dropout_ews.explainability.llm_summary import SYSTEM_PROMPT
from dropout_ews.explainability.narratives import (
    EXPLANATION_DISCLAIMER,
    FEATURE_META,
    RenderedExplanation,
    describe_feature,
    impact_band,
)
from dropout_ews.interventions.engine import load_catalog

# Phrases that assert causation, or certainty about the future. Each is paired
# with what it wrongly implies, so a failure message explains itself.
FORBIDDEN_PHRASES: dict[str, str] = {
    "caused": "asserts causation the model cannot establish",
    "causes": "asserts causation the model cannot establish",
    "causing": "asserts causation the model cannot establish",
    "because of": "asserts causation the model cannot establish",
    "due to": "asserts causation the model cannot establish",
    "results in": "asserts causation the model cannot establish",
    "leads to": "asserts causation the model cannot establish",
    "will drop out": "states a certainty about the future",
    "will withdraw": "states a certainty about the future",
    "is going to drop": "states a certainty about the future",
    "will fail": "states a certainty about the future",
    "destined": "states a certainty about the future",
    "proves": "overstates evidential strength",
    "confirms": "overstates evidential strength",
    "guarantees": "overstates evidential strength",
    "definitely": "overstates evidential strength",
}

# Judgements about the person rather than the record.
FORBIDDEN_CHARACTER_TERMS: tuple[str, ...] = (
    "lazy",
    "unmotivated",
    "careless",
    "weak student",
    "poor student",
    "bad student",
    "incapable",
    "not committed",
    "does not care",
)


def _strings_from_feature_meta() -> dict[str, str]:
    """Every human-visible string in the feature vocabulary."""
    collected: dict[str, str] = {}
    for name, meta in FEATURE_META.items():
        collected[f"{name}.label"] = meta.label
        collected[f"{name}.higher_means"] = meta.higher_means
    return collected


def _generated_sentences() -> dict[str, str]:
    """Render every template branch: each feature, both directions, all bands.

    Testing the templates as strings is not enough — the sentence is assembled at
    runtime, so the assembled output is what a counsellor actually reads.
    """
    sentences: dict[str, str] = {}
    for name, meta in FEATURE_META.items():
        for raises_risk in (True, False):
            for band in ("High", "Medium", "Low"):
                key = f"{name}|{'risk' if raises_risk else 'protective'}|{band}"
                sentences[key] = narratives._sentence(meta, raises_risk, band)
    return sentences


def _catalog_strings() -> dict[str, str]:
    catalog = load_catalog()
    collected: dict[str, str] = {}
    for item in catalog.interventions:
        collected[f"{item.key}.title"] = item.title
        collected[f"{item.key}.description"] = item.description
    for band, guidance in catalog.band_guidance.items():
        collected[f"band_guidance.{band}"] = guidance.note
    return collected


def _rationale_strings() -> dict[str, str]:
    from dropout_ews.interventions.engine import _FACTOR_PHRASES, _factor_phrase

    return {
        f"rationale.{factor}": (
            f"Suggested because the model weighted {_factor_phrase(factor)} for this student."
        )
        for factor in _FACTOR_PHRASES
    }


ALL_HUMAN_VISIBLE_STRINGS: dict[str, str] = {
    **_strings_from_feature_meta(),
    **_generated_sentences(),
    **_catalog_strings(),
    **_rationale_strings(),
    "disclaimer": EXPLANATION_DISCLAIMER,
    "llm_system_prompt": SYSTEM_PROMPT,
}


# Negating contexts in which a denied word legitimately appears, because the text
# is *disclaiming* causation rather than asserting it. The disclaimer has to be
# able to say "not the causes of withdrawal", and the LLM prompt has to be able to
# forbid causal phrasing by naming it.
#
# These are stripped before scanning, the same pattern the intervention denylist
# uses for "without penalty". A keyword check with no negative-context handling is
# either too loose to be useful or too tight to permit correct wording.
PERMITTED_NEGATING_CONTEXTS: tuple[str, ...] = (
    "not the causes",
    "not the cause",
    "never caused",
    "never state or imply that these factors caused",
    "do not cause",
    "rather than the causes",
)


def _scannable(text: str) -> str:
    """Lowercase the text and remove legitimate negating contexts."""
    lowered = text.lower()
    for context in PERMITTED_NEGATING_CONTEXTS:
        lowered = lowered.replace(context, " ")
    return lowered


@pytest.mark.parametrize("phrase", sorted(FORBIDDEN_PHRASES))
def test_no_human_visible_string_makes_a_causal_or_certain_claim(phrase: str) -> None:
    """The load-bearing assertion of ADR-0005."""
    reason = FORBIDDEN_PHRASES[phrase]
    offenders = [
        name
        for name, text in ALL_HUMAN_VISIBLE_STRINGS.items()
        # The LLM prompt is instructions to a model about what not to write, so it
        # names prohibited phrasing throughout and is scanned separately.
        if phrase in _scannable(text) and name != "llm_system_prompt"
    ]
    assert not offenders, f"the phrase {phrase!r} {reason}, and appears in: {sorted(offenders)}"


def test_the_negating_context_allowance_does_not_swallow_a_real_claim() -> None:
    """Guard the guard: the allowance must not make the scan permissive.

    A string that asserts causation has to still be caught even though a
    disclaiming string containing the same word is permitted.
    """
    assert "caused" in _scannable("Low attendance caused this student to withdraw.")
    assert "caused" not in _scannable(
        "They describe the model's behaviour, not the causes of withdrawal."
    )


@pytest.mark.parametrize("term", FORBIDDEN_CHARACTER_TERMS)
def test_no_human_visible_string_judges_the_student(term: str) -> None:
    offenders = [
        name
        for name, text in ALL_HUMAN_VISIBLE_STRINGS.items()
        if term in text.lower() and name != "llm_system_prompt"
    ]
    assert not offenders, f"the term {term!r} judges the student; found in {sorted(offenders)}"


def test_llm_system_prompt_forbids_the_same_language() -> None:
    """The prompt is the only place forbidden phrasing may appear, and only
    because it is instructing the model not to use it."""
    lowered = SYSTEM_PROMPT.lower()
    assert "never" in lowered
    assert "caused" in lowered, "the prompt should explicitly forbid causal claims"
    assert "predict what the student will do" in lowered
    assert "percentages" in lowered, "the prompt should forbid quoting probabilities"


def test_every_generated_sentence_uses_contribution_language() -> None:
    """Positive requirement, not just absence of bad phrasing: every sentence has
    to actually frame itself as a contribution to an estimate."""
    for key, sentence in _generated_sentences().items():
        assert "estimate" in sentence.lower(), f"{key} does not frame itself as an estimate"
        assert "the model" in sentence.lower(), f"{key} does not attribute to the model"


def test_disclaimer_states_the_three_required_things() -> None:
    lowered = EXPLANATION_DISCLAIMER.lower()
    assert "not the causes" in lowered
    assert "not a judgement" in lowered
    assert "probability" in lowered


def test_no_sentence_quotes_a_numeric_effect() -> None:
    """Attributions are log-odds of the pre-calibration score, so any
    percentage-point claim would be false."""
    numeric = re.compile(r"\d+(\.\d+)?\s*(%|percentage point|pp\b)")
    for key, sentence in _generated_sentences().items():
        assert not numeric.search(sentence), f"{key} quotes a numeric effect"


# ---------------------------------------------------------------------------
# Feature vocabulary coverage
# ---------------------------------------------------------------------------


def test_every_allowlisted_feature_has_narrative_metadata() -> None:
    """An unmapped feature would be shown to staff as a raw column name such as
    `clicks_vs_baseline_ratio`."""
    from dropout_ews.config.settings import load_feature_config

    missing = [
        name for name in load_feature_config().features.all_features() if name not in FEATURE_META
    ]
    assert not missing, f"features with no narrative metadata: {missing}"


def test_missingness_indicators_get_a_readable_label() -> None:
    """The imputer appends `<feature>__missing` columns, which the model uses and
    SHAP therefore attributes to."""
    meta = describe_feature("mean_score__missing")
    assert "no recorded data" in meta.label.lower()
    assert meta.actionable is False


def test_unknown_feature_fails_loudly() -> None:
    with pytest.raises(KeyError, match="no narrative metadata"):
        describe_feature("some_new_feature_nobody_mapped")


def test_contextual_features_are_marked_non_actionable() -> None:
    """`checkpoint_day` is the largest single global contributor, but "it is day
    30" is not something a counsellor can act on. Presenting it as a top reason
    would waste their attention."""
    for name in ("checkpoint_day", "num_of_prev_attempts", "date_registration"):
        assert FEATURE_META[name].actionable is False


def test_engagement_and_assessment_features_are_actionable() -> None:
    for name in ("clicks_7d", "days_since_last_activity", "submission_rate", "mean_score"):
        assert FEATURE_META[name].actionable is True


# ---------------------------------------------------------------------------
# Impact bands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "largest", "expected"),
    [
        (1.0, 1.0, "High"),
        (0.6, 1.0, "High"),
        (0.59, 1.0, "Medium"),
        (0.25, 1.0, "Medium"),
        (0.24, 1.0, "Low"),
        (-1.0, 1.0, "High"),  # bands are about magnitude, not direction
    ],
)
def test_impact_band_thresholds(value: float, largest: float, expected: str) -> None:
    assert impact_band(value, largest) == expected


def test_impact_band_is_relative_not_absolute() -> None:
    """A student whose factors are all small should still see which mattered most
    for them; that is the question a reader is asking."""
    assert impact_band(0.01, 0.01) == "High"
    assert impact_band(0.01, 1.0) == "Low"


def test_impact_band_handles_a_zero_scale() -> None:
    assert impact_band(0.0, 0.0) == "Low"


def test_empty_explanation_carries_the_disclaimer_and_a_note() -> None:
    """A student with no meaningful factors must not get a blank panel that reads
    as an error."""
    empty = RenderedExplanation(risk_factors=[], protective_factors=[], context_factors=[])
    assert empty.disclaimer == EXPLANATION_DISCLAIMER
    assert empty.matched_factors == []


def test_narrative_module_has_no_stray_forbidden_text() -> None:
    """Belt and braces: scan the source files themselves, so a docstring example
    or a comment cannot introduce phrasing that later gets copied into a template.
    Only the prohibition lists and the prompt are exempt.
    """
    root = Path(narratives.__file__).parent
    for path in (root / "narratives.py",):
        text = path.read_text(encoding="utf-8").lower()
        # "never causal" style prose is fine; the check targets assertions about a
        # student, which always co-occur with "this student".
        for phrase in ("student will drop out", "student will withdraw"):
            assert phrase not in text, f"{path.name} contains {phrase!r}"
