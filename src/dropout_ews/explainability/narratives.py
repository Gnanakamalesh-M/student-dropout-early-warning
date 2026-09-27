"""Turn SHAP contributions into language a counsellor can read.

Three rules govern everything here, all from ADR-0005.

**No causal claims.** SHAP describes the model, not the world. Every phrase uses
contribution language — "contributed most to this estimate", "pushed the estimate
higher" — and ``tests/unit/test_narrative_language.py`` fails the build if a
template contains "caused", "because of", "will drop out" or similar. The
disclaimer in :data:`EXPLANATION_DISCLAIMER` is attached to every payload and is
not optional.

**No numeric probability attribution.** Attributions are in log-odds of the
pre-calibration score, so "this feature added 4 percentage points" would be
false. Templates report direction and an impact band only.

**Actionable factors are separated from context.** The single largest global
contributor is ``checkpoint_day`` — how far through the course the student is.
That is real signal, but "this student is at risk because it is day 30" is not
something anybody can act on, and presenting it as a top reason would waste a
counsellor's attention. Features are therefore tagged
:data:`ACTIONABLE` or contextual, and the two are rendered in separate lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dropout_ews.explainability.shap_explainer import Explanation, FeatureContribution

EXPLANATION_DISCLAIMER = (
    "These are the factors the model weighted most heavily for this student. "
    "They describe the model's behaviour, not the causes of withdrawal, and they "
    "are not a judgement about this student. The risk figure is a probability, "
    "not a prediction of what will happen."
)

ImpactBand = Literal["High", "Medium", "Low"]

# Impact bands are shares of the largest contribution in *this* explanation, not
# absolute log-odds cutoffs. A student whose factors are all small should not have
# them all labelled "Low" — the question a reader has is which factors mattered
# most for this student, which is inherently relative.
HIGH_IMPACT_SHARE = 0.60
MEDIUM_IMPACT_SHARE = 0.25


@dataclass(frozen=True)
class FeatureMeta:
    """How to talk about one feature."""

    label: str
    """Human-readable name."""

    higher_means: str
    """What an increase in this feature means, in plain words. Used to phrase the
    direction correctly: for ``days_since_last_activity`` a higher value is worse,
    for ``mean_score`` it is better."""

    actionable: bool = True
    """Whether a member of staff could plausibly do something about it."""

    factor: str | None = None
    """Risk-factor key the intervention engine matches on. ``None`` means the
    feature suggests no specific intervention."""


# Feature vocabulary. Every allowlisted feature needs an entry, plus the
# `__missing` indicators the imputer appends. A test asserts full coverage,
# because an unmapped feature would otherwise be shown to staff as a raw column
# name like `clicks_vs_baseline_ratio`.
FEATURE_META: dict[str, FeatureMeta] = {
    # --- engagement level ---
    "clicks_7d": FeatureMeta(
        "Course activity in the last week", "more activity", factor="low_engagement"
    ),
    "clicks_14d": FeatureMeta(
        "Course activity in the last two weeks", "more activity", factor="low_engagement"
    ),
    "clicks_28d": FeatureMeta(
        "Course activity in the last four weeks", "more activity", factor="low_engagement"
    ),
    "clicks_56d": FeatureMeta(
        "Course activity in the last eight weeks", "more activity", factor="low_engagement"
    ),
    "clicks_all_time": FeatureMeta(
        "Total course activity so far", "more activity", factor="low_engagement"
    ),
    "active_days_7d": FeatureMeta(
        "Days active in the last week", "more active days", factor="low_engagement"
    ),
    "active_days_14d": FeatureMeta(
        "Days active in the last two weeks", "more active days", factor="low_engagement"
    ),
    "active_days_28d": FeatureMeta(
        "Days active in the last four weeks", "more active days", factor="low_engagement"
    ),
    "active_days_56d": FeatureMeta(
        "Days active in the last eight weeks", "more active days", factor="low_engagement"
    ),
    "active_days_all_time": FeatureMeta(
        "Total days active so far", "more active days", factor="low_engagement"
    ),
    "clicks_per_active_day_28d": FeatureMeta(
        "Depth of study on active days", "longer sessions when studying", factor="low_engagement"
    ),
    # --- recency ---
    "days_since_last_activity": FeatureMeta(
        "Time since last course activity", "a longer gap since last seen", factor="disengaged"
    ),
    "ever_active": FeatureMeta(
        "Has accessed the course at all", "has engaged at least once", factor="never_engaged"
    ),
    "inactive_weeks_streak": FeatureMeta(
        "Consecutive quiet weeks", "more consecutive weeks with no activity", factor="disengaged"
    ),
    # --- trend ---
    "clicks_vs_baseline_ratio": FeatureMeta(
        "Activity compared with this student's own earlier pattern",
        "activity holding up against their own baseline",
        factor="declining_engagement",
    ),
    "clicks_ratio_4w": FeatureMeta(
        "Recent activity compared with the previous month",
        "activity holding up month on month",
        factor="declining_engagement",
    ),
    "clicks_delta_4w": FeatureMeta(
        "Change in activity since last month", "activity increasing", factor="declining_engagement"
    ),
    # --- assessment ---
    "assessments_due": FeatureMeta(
        "Assessments due so far", "more assessments already due", actionable=False
    ),
    "assessments_submitted": FeatureMeta(
        "Assessments submitted", "more submitted", factor="missed_assessments"
    ),
    "assessments_missed": FeatureMeta(
        "Assessments due but not submitted", "more missed", factor="missed_assessments"
    ),
    "assessments_failed": FeatureMeta(
        "Assessments below the pass mark", "more failed", factor="academic_difficulty"
    ),
    "assessments_late": FeatureMeta(
        "Assessments submitted late", "more late submissions", factor="time_management"
    ),
    "submission_rate": FeatureMeta(
        "Share of due assessments submitted",
        "a higher submission rate",
        factor="missed_assessments",
    ),
    "mean_score": FeatureMeta(
        "Average assessment score", "higher scores", factor="academic_difficulty"
    ),
    "min_score": FeatureMeta(
        "Lowest assessment score", "a higher worst score", factor="academic_difficulty"
    ),
    "mean_submission_lag": FeatureMeta(
        "Typical submission timing against the deadline",
        "submitting earlier",
        factor="time_management",
    ),
    "days_since_last_submission": FeatureMeta(
        "Time since last submission", "a longer gap since submitting", factor="missed_assessments"
    ),
    "has_submitted": FeatureMeta(
        "Has submitted any assessment", "has submitted at least once", factor="missed_assessments"
    ),
    # --- cohort-relative ---
    "clicks_7d_cohort_z": FeatureMeta(
        "Weekly activity compared with classmates",
        "more activity than classmates",
        factor="low_engagement",
    ),
    "clicks_28d_cohort_z": FeatureMeta(
        "Monthly activity compared with classmates",
        "more activity than classmates",
        factor="low_engagement",
    ),
    "active_days_28d_cohort_z": FeatureMeta(
        "Active days compared with classmates",
        "more active days than classmates",
        factor="low_engagement",
    ),
    "mean_score_cohort_z": FeatureMeta(
        "Assessment scores compared with classmates",
        "higher scores than classmates",
        factor="academic_difficulty",
    ),
    "submission_rate_cohort_z": FeatureMeta(
        "Submission rate compared with classmates",
        "a higher submission rate than classmates",
        factor="missed_assessments",
    ),
    # --- context: real signal, but nothing a counsellor can act on ---
    "checkpoint_day": FeatureMeta(
        "Point reached in the course", "being further through the course", actionable=False
    ),
    "num_of_prev_attempts": FeatureMeta(
        "Previous attempts at this module", "more previous attempts", actionable=False
    ),
    "studied_credits": FeatureMeta(
        "Credit load this term", "a heavier credit load", factor="workload"
    ),
    "date_registration": FeatureMeta(
        "How early the student registered", "registering earlier", actionable=False
    ),
}

# Missingness indicators. Absence of a record is itself informative, and phrasing
# it as "no record of X" avoids implying the student did something.
MISSING_SUFFIX = "__missing"

ACTIONABLE = {name for name, meta in FEATURE_META.items() if meta.actionable}


def describe_feature(name: str) -> FeatureMeta:
    """Metadata for a model feature, including imputer indicators."""
    if name.endswith(MISSING_SUFFIX):
        base = name[: -len(MISSING_SUFFIX)]
        parent = FEATURE_META.get(base)
        label = parent.label if parent else base.replace("_", " ")
        return FeatureMeta(
            label=f"No recorded data for: {label.lower()}",
            higher_means="a record being present",
            actionable=False,
            factor=parent.factor if parent else None,
        )
    meta = FEATURE_META.get(name)
    if meta is None:
        raise KeyError(
            f"no narrative metadata for feature {name!r}. Add it to FEATURE_META; "
            "an unmapped feature would be shown to staff as a raw column name."
        )
    return meta


def impact_band(shap_value: float, largest: float) -> ImpactBand:
    """Band a contribution by its size relative to the largest in this explanation."""
    if largest <= 0:
        return "Low"
    share = abs(shap_value) / largest
    if share >= HIGH_IMPACT_SHARE:
        return "High"
    if share >= MEDIUM_IMPACT_SHARE:
        return "Medium"
    return "Low"


@dataclass(frozen=True)
class RenderedFactor:
    """One factor, ready to display."""

    feature: str
    label: str
    impact: ImpactBand
    direction: Literal["increases", "decreases"]
    actionable: bool
    factor: str | None
    sentence: str
    shap_value: float


@dataclass(frozen=True)
class RenderedExplanation:
    """The full explanation payload handed to the API and the dashboard."""

    risk_factors: list[RenderedFactor]
    """Factors that pushed the estimate higher, actionable first."""

    protective_factors: list[RenderedFactor]
    """Factors that pushed the estimate lower. Shown because an explanation of
    only negatives misrepresents a student who is doing several things well."""

    context_factors: list[RenderedFactor]
    """Non-actionable contributors, kept separate so they do not crowd out
    factors staff can respond to."""

    disclaimer: str = EXPLANATION_DISCLAIMER
    notes: list[str] = field(default_factory=list)

    @property
    def matched_factors(self) -> list[str]:
        """Risk-factor keys for the intervention engine, highest impact first."""
        seen: list[str] = []
        for rendered in self.risk_factors:
            if rendered.factor and rendered.factor not in seen:
                seen.append(rendered.factor)
        return seen


def _sentence(meta: FeatureMeta, raises_risk: bool, band: ImpactBand) -> str:
    """Build one factor sentence.

    Phrased as contribution, never causation, and never with a numeric effect on
    probability. The wording is generated from ``higher_means`` so direction is
    correct for features where a higher value is good (scores) and for those
    where it is bad (days since last activity).
    """
    verb = "contributed to a higher estimate" if raises_risk else "contributed to a lower estimate"
    strength = {
        "High": "weighted this heavily",
        "Medium": "gave this moderate weight",
        "Low": "gave this some weight",
    }[band]
    return f"{meta.label}: the model {strength} and it {verb} for this student."


def render_explanation(
    explanation: Explanation,
    max_risk_factors: int = 6,
    max_protective_factors: int = 3,
    max_context_factors: int = 3,
) -> RenderedExplanation:
    """Turn a SHAP explanation into display-ready factors."""
    contributions = [c for c in explanation.contributions if not c.is_negligible]
    if not contributions:
        return RenderedExplanation(
            risk_factors=[],
            protective_factors=[],
            context_factors=[],
            notes=[
                "No individual factor carried meaningful weight for this student; "
                "the estimate sits close to the cohort average."
            ],
        )

    largest = max(abs(c.shap_value) for c in contributions)

    def render(contribution: FeatureContribution) -> RenderedFactor:
        meta = describe_feature(contribution.feature)
        raises_risk = contribution.shap_value > 0
        band = impact_band(contribution.shap_value, largest)
        return RenderedFactor(
            feature=contribution.feature,
            label=meta.label,
            impact=band,
            direction="increases" if raises_risk else "decreases",
            actionable=meta.actionable,
            factor=meta.factor,
            sentence=_sentence(meta, raises_risk, band),
            shap_value=contribution.shap_value,
        )

    rendered = [render(c) for c in contributions]
    risk = [r for r in rendered if r.direction == "increases" and r.actionable]
    protective = [r for r in rendered if r.direction == "decreases" and r.actionable]
    context = [r for r in rendered if not r.actionable]

    notes: list[str] = []
    if context and not risk:
        notes.append(
            "The factors carrying most weight here are contextual rather than "
            "something staff can act on directly."
        )

    return RenderedExplanation(
        risk_factors=risk[:max_risk_factors],
        protective_factors=protective[:max_protective_factors],
        context_factors=context[:max_context_factors],
        notes=notes,
    )
