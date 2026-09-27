"""Optional LLM case-note rendering. Off by default.

Scope, from ADR-0005, and deliberately narrow: the LLM converts an
already-computed factor list and intervention set into a short, warm case note. It
is a rendering layer downstream of the model, not part of the judgement.

Hard constraints, enforced in code below rather than by convention:

* **Derived data only.** The prompt carries factor labels, impact bands, the risk
  band and intervention titles. It never receives a student identifier, raw
  feature values, free-text notes, or any demographic attribute.
  :func:`build_prompt` scrubs and then asserts this, and a test feeds it a payload
  laced with identifiers to confirm they are rejected.
* **It cannot influence the score or the recommendations.** Both are inputs. No
  LLM output is persisted as a feature or fed back into the model.
* **Off by default.** With ``ENABLE_LLM_NARRATIVE=false`` or no API key, the
  deterministic template renders instead and the product is fully functional.
* **Same language rules.** The system prompt forbids causal and deterministic
  phrasing, and the narrative-language lint scans it alongside the templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dropout_ews.config.settings import get_settings
from dropout_ews.explainability.narratives import EXPLANATION_DISCLAIMER, RenderedExplanation
from dropout_ews.interventions.engine import Recommendation

# Anthropic's current small model: this is a short summarisation task where a
# larger model buys nothing.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 400

SYSTEM_PROMPT = """You write short internal case notes for student support staff.

You receive only a list of factors a statistical model weighted, an overall risk
band, and support actions that have already been selected. You never receive
student identities or personal details.

Write two to four sentences for a colleague who is about to offer this student
support. Requirements:

- Describe what the model weighted. Never state or imply that these factors
  caused anything, and never predict what the student will do.
- Say nothing about the student's ability, motivation, or character.
- Be warm and matter-of-fact. This note precedes an offer of help, not a
  judgement.
- Do not invent details. If the factors are thin, say the signal is limited.
- Do not quote probabilities or percentages.
- End with the suggested next step, framed as an offer.
"""

# Patterns that must never reach the prompt. Checked after scrubbing, so a new
# field added to the payload upstream cannot silently carry identifiers through.
_IDENTIFIER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bid_student\b", "student id field"),
    (r"\bstudent_code\b", "student code field"),
    (r"\b\d{5,}\b", "a long numeric identifier"),
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "an email address"),
)

# Demographic terms that are excluded from the feature matrix and must not appear
# in a prompt either (ETHICS.md).
_PROTECTED_TERMS: tuple[str, ...] = (
    "gender",
    "male",
    "female",
    "age_band",
    "imd_band",
    "deprivation",
    "disability",
    "disabled",
    "region",
    "ethnic",
)


class PromptSafetyError(ValueError):
    """Raised when a prompt would carry identifying or protected information."""


@dataclass(frozen=True)
class CaseNote:
    text: str
    source: str
    """``"llm"`` or ``"template"``, so the UI and the audit log can tell which
    produced the note."""

    disclaimer: str = EXPLANATION_DISCLAIMER


def build_prompt(
    explanation: RenderedExplanation,
    recommendations: list[Recommendation],
    risk_band: str,
) -> str:
    """Build the user prompt from derived data only.

    Raises:
        PromptSafetyError: if the assembled prompt contains anything that looks
            like an identifier or a protected attribute. Failing closed is the
            point: a prompt is an outbound transfer, and a silent leak there is
            not recoverable.
    """
    risk_lines = [
        f"- {factor.label} (impact: {factor.impact}, {factor.direction} the estimate)"
        for factor in explanation.risk_factors
    ] or ["- No single factor carried meaningful weight."]
    protective_lines = [
        f"- {factor.label} (impact: {factor.impact})" for factor in explanation.protective_factors
    ]
    action_lines = [
        f"- {r.intervention.title}: {r.intervention.description}" for r in recommendations
    ]

    parts = [
        f"Risk band: {risk_band}",
        "",
        "Factors the model weighted toward higher risk:",
        *risk_lines,
    ]
    if protective_lines:
        parts += ["", "Factors the model weighted toward lower risk:", *protective_lines]
    if action_lines:
        parts += ["", "Support actions already selected:", *action_lines]
    if explanation.notes:
        parts += ["", "Notes:", *(f"- {note}" for note in explanation.notes)]

    prompt = "\n".join(parts)
    _assert_prompt_is_safe(prompt)
    return prompt


def _assert_prompt_is_safe(prompt: str) -> None:
    for pattern, description in _IDENTIFIER_PATTERNS:
        if re.search(pattern, prompt, flags=re.IGNORECASE):
            raise PromptSafetyError(
                f"prompt appears to contain {description}; the LLM layer receives "
                "derived factors only (ADR-0005)"
            )
    lowered = prompt.lower()
    for term in _PROTECTED_TERMS:
        if term in lowered:
            raise PromptSafetyError(
                f"prompt contains the protected term {term!r}; these attributes are "
                "excluded from the model and must not reach a prompt (ETHICS.md)"
            )


def template_case_note(
    explanation: RenderedExplanation,
    recommendations: list[Recommendation],
    risk_band: str,
) -> CaseNote:
    """Deterministic fallback. Always available, never requires a network call."""
    if not explanation.risk_factors:
        body = (
            f"This student sits in the {risk_band} band, but no individual factor "
            "carried meaningful weight, so the signal here is limited."
        )
    else:
        top = explanation.risk_factors[0]
        others = [factor.label.lower() for factor in explanation.risk_factors[1:3]]
        body = (
            f"This student sits in the {risk_band} band. The model gave most weight "
            f"to {top.label.lower()}"
        )
        if others:
            body += ", alongside " + " and ".join(others)
        body += "."

    if explanation.protective_factors:
        body += (
            " It also weighted "
            + explanation.protective_factors[0].label.lower()
            + " toward a lower estimate."
        )
    if recommendations:
        body += f" Suggested next step: {recommendations[0].intervention.title.lower()}."
    else:
        body += " No outreach is suggested at this band."
    return CaseNote(text=body, source="template")


def generate_case_note(
    explanation: RenderedExplanation,
    recommendations: list[Recommendation],
    risk_band: str,
    client: Any | None = None,
) -> CaseNote:
    """Produce a case note, using the LLM only when it is enabled and healthy.

    Falls back to the deterministic template when the feature flag is off, no key
    is configured, or the call fails for any reason. A support tool should not
    show a counsellor an error where a sentence belongs.
    """
    settings = get_settings()
    if not settings.enable_llm_narrative:
        return template_case_note(explanation, recommendations, risk_band)

    prompt = build_prompt(explanation, recommendations, risk_band)
    try:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            raise ValueError("empty response")
        return CaseNote(text=text, source="llm")
    except Exception:
        return template_case_note(explanation, recommendations, risk_band)
