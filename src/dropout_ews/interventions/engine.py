"""Map explained risk factors to supportive interventions.

Deterministic rules over a reviewed catalog, not a learned recommender. A learned
model would need intervention-outcome data this project does not have, and would
be much harder to defend to the staff and students it affects.

Two constraints are structural rather than advisory:

* **Everything is an offer of support.** The catalog is checked against a
  punitive-keyword denylist at load time, so a punitive entry cannot be
  introduced by editing YAML.
* **Nothing executes.** Recommendations carry ``status="recommended"`` and a
  ``requires_human_review`` flag that is always true. Assignment is a separate,
  human action (ADR-0005, ETHICS.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.yaml"

Intensity = Literal["light", "moderate"]

# Language that must never appear in an intervention. Two categories: punishment
# and surveillance. Both are failure modes an early-warning system drifts toward
# unless something actively stops it.
#
# Matching is on WORD BOUNDARIES, not substrings, and a small allowlist of
# supportive phrases is excluded first. Naive substring matching is not adequate
# here and that was found the hard way: the most supportive line in the catalog,
# "reducing load without penalty", tripped a bare `penal` substring. A denylist
# with no negative-context handling is either too loose to be useful or too tight
# to allow correct phrasing.
FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    # punishment
    "penalty",
    "penalties",
    "penalise",
    "penalize",
    "penalised",
    "penalized",
    "punish",
    "punished",
    "punitive",
    "sanction",
    "sanctions",
    "discipline",
    "disciplinary",
    "reprimand",
    "expel",
    "expelled",
    "exclude",
    "excluded",
    "terminate",
    "revoke",
    "withhold",
    "suspend",
    # surveillance and disclosure
    "surveillance",
    "covertly",
    "monitor closely",
    "inform parents",
    "next of kin",
    "flag to employer",
    "report to management",
)

# Supportive phrases that legitimately contain a denied word. Removed before the
# scan so correct wording is not blocked. Kept deliberately short: every entry
# widens what the guard permits, so each needs to be obviously supportive.
PERMITTED_PHRASES: tuple[str, ...] = (
    "without penalty",
    "no penalty",
    "free of penalty",
)

# `unspecified` is the fallback factor, so a flagged student never receives an
# empty list of options.
FALLBACK_FACTOR = "unspecified"


@dataclass(frozen=True)
class Intervention:
    key: str
    title: str
    factors: tuple[str, ...]
    intensity: Intensity
    owner_role: str
    description: str
    typical_effort_minutes: int


@dataclass(frozen=True)
class BandGuidance:
    max_interventions: int
    note: str


@dataclass(frozen=True)
class Recommendation:
    """A suggested action. Not an action taken."""

    intervention: Intervention
    matched_factor: str
    rationale: str
    status: str = "recommended"
    requires_human_review: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.intervention.key,
            "title": self.intervention.title,
            "description": self.intervention.description,
            "intensity": self.intervention.intensity,
            "owner_role": self.intervention.owner_role,
            "typical_effort_minutes": self.intervention.typical_effort_minutes,
            "matched_factor": self.matched_factor,
            "rationale": self.rationale,
            "status": self.status,
            "requires_human_review": self.requires_human_review,
        }


@dataclass
class Catalog:
    interventions: list[Intervention]
    band_guidance: dict[str, BandGuidance] = field(default_factory=dict)

    def for_factor(self, factor: str) -> list[Intervention]:
        return [item for item in self.interventions if factor in item.factors]

    @property
    def covered_factors(self) -> set[str]:
        return {factor for item in self.interventions for factor in item.factors}


class PunitiveInterventionError(ValueError):
    """Raised when a catalog entry contains punitive or surveillance language."""


def find_forbidden_language(text: str) -> list[str]:
    """Return denied terms present in ``text``, ignoring permitted phrases.

    Word-boundary matching, so "penalty" is caught but "without penalty" is not
    after the allowlist pass. Exposed rather than private because the catalog test
    and the narrative language lint both scan with the same rule.
    """
    haystack = text.lower()
    for phrase in PERMITTED_PHRASES:
        haystack = haystack.replace(phrase, " ")
    return [
        keyword
        for keyword in FORBIDDEN_KEYWORDS
        if re.search(rf"\b{re.escape(keyword)}\b", haystack)
    ]


def _assert_supportive(intervention: Intervention) -> None:
    found = find_forbidden_language(f"{intervention.title} {intervention.description}")
    if found:
        raise PunitiveInterventionError(
            f"intervention {intervention.key!r} contains forbidden language {found!r}. "
            "Interventions must offer support; see docs/ETHICS.md."
        )


def load_catalog(path: Path | None = None) -> Catalog:
    """Load and validate the catalog.

    Validation happens at load time so a punitive entry cannot reach production
    by editing YAML: the process refuses to start rather than quietly serving it.
    """
    payload = yaml.safe_load((path or CATALOG_PATH).read_text(encoding="utf-8"))
    interventions = [
        Intervention(
            key=item["key"],
            title=item["title"],
            factors=tuple(item["factors"]),
            intensity=item["intensity"],
            owner_role=item["owner_role"],
            description=" ".join(item["description"].split()),
            typical_effort_minutes=int(item["typical_effort_minutes"]),
        )
        for item in payload["interventions"]
    ]
    keys = [item.key for item in interventions]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise ValueError(f"duplicate intervention keys: {sorted(duplicates)}")
    for intervention in interventions:
        _assert_supportive(intervention)

    guidance = {
        band: BandGuidance(
            max_interventions=int(spec["max_interventions"]),
            note=" ".join(spec["note"].split()),
        )
        for band, spec in (payload.get("band_guidance") or {}).items()
    }
    return Catalog(interventions=interventions, band_guidance=guidance)


def recommend(
    matched_factors: list[str],
    risk_band: str,
    catalog: Catalog | None = None,
) -> tuple[list[Recommendation], str]:
    """Recommend supportive actions for a student.

    Args:
        matched_factors: risk-factor keys from the rendered explanation, highest
            impact first.
        risk_band: ``low``, ``medium``, ``high`` or ``critical``.

    Returns:
        The recommendations and the band guidance note, so the UI can show *why*
        the list is the length it is.

    Ordering follows factor impact, so the action addressing the most heavily
    weighted factor appears first. Within a factor, lighter-touch actions come
    first: a 15-minute message is the right opening move, and escalating straight
    to intensive support on a probabilistic signal is not proportionate.
    """
    catalog = catalog or load_catalog()
    guidance = catalog.band_guidance.get(risk_band)
    if guidance is None:
        raise ValueError(f"unknown risk band {risk_band!r}")

    if guidance.max_interventions == 0:
        return [], guidance.note

    intensity_order = {"light": 0, "moderate": 1}
    recommendations: list[Recommendation] = []
    seen: set[str] = set()

    factors = matched_factors or [FALLBACK_FACTOR]
    for factor in factors:
        options = sorted(
            catalog.for_factor(factor), key=lambda item: intensity_order.get(item.intensity, 9)
        )
        for intervention in options:
            if intervention.key in seen:
                continue
            seen.add(intervention.key)
            recommendations.append(
                Recommendation(
                    intervention=intervention,
                    matched_factor=factor,
                    rationale=(
                        f"Suggested because the model weighted "
                        f"{_factor_phrase(factor)} for this student."
                    ),
                )
            )
            if len(recommendations) >= guidance.max_interventions:
                return recommendations, guidance.note

    if not recommendations:
        # Every listed factor was unmatched; fall back rather than return nothing.
        fallback = catalog.for_factor(FALLBACK_FACTOR)
        if fallback:
            recommendations.append(
                Recommendation(
                    intervention=fallback[0],
                    matched_factor=FALLBACK_FACTOR,
                    rationale=("Suggested because no single factor stood out for this student."),
                )
            )
    return recommendations, guidance.note


# Phrasing for rationales. Contribution language only, no causal claims: the
# same lint that guards narrative templates also scans these.
_FACTOR_PHRASES: dict[str, str] = {
    "low_engagement": "recent course activity",
    "disengaged": "the gap since this student was last active",
    "never_engaged": "the absence of any recorded course access",
    "declining_engagement": "activity compared with this student's earlier pattern",
    "missed_assessments": "assessments due but not submitted",
    "academic_difficulty": "assessment scores",
    "time_management": "submission timing against deadlines",
    "workload": "credit load this term",
    FALLBACK_FACTOR: "the overall pattern",
}


def _factor_phrase(factor: str) -> str:
    return _FACTOR_PHRASES.get(factor, factor.replace("_", " "))
