# ADR-0005 — Explainability language and the scope of LLM use

- **Status:** Accepted
- **Date:** 2026-09-25

## Context

The system shows a risk percentage and contributing factors to staff who will
act on them. Two failure modes are serious and both are primarily *linguistic*
rather than statistical:

1. **Causal overreach.** SHAP attributions describe a model's behaviour, not the
   world. "Low attendance caused this student's dropout risk" is unsupported by
   anything the system computes. Staff will reasonably read causal phrasing as
   a causal claim.
2. **Deterministic framing.** "This student will drop out" misrepresents a
   calibrated probability and invites treating a prediction as a verdict about
   a person.

## Decision

### Explainability method

`shap.TreeExplainer` on the tuned XGBoost model, with a **fixed, versioned
background sample** serialised alongside the model artifact so explanations are
reproducible across runs. Global views: beeswarm, mean-|SHAP| bar, dependence
plots. Local: top-6 signed contributions per prediction, bucketed into impact
bands (High / Medium / Low) rather than shown as raw SHAP values, which are not
meaningful to a non-technical reader.

### Mandated language rules

Every rendered explanation uses contribution phrasing:

- Allowed: "contributed most to this prediction", "pushed the estimate upward",
  "the model weighted this heavily".
- Forbidden: "caused", "because of", "due to", "will drop out", "is going to",
  "proves", "confirms".

Enforcement is a test, not a guideline: `tests/unit/test_narrative_language.py`
scans all narrative templates for the forbidden set and fails the build on a
match. This also covers any LLM output template.

Every explanation payload carries a mandatory disclaimer field, rendered
wherever the explanation appears:

> These are the factors the model weighted most heavily for this student. They
> describe the model's behaviour, not the causes of dropout, and they are not a
> judgement about this student.

### Intervention engine

Deterministic rules mapping `(factor, direction, impact_band)` to entries in
`interventions/catalog.yaml`. Rules are hand-written and reviewed, not learned:
a learned recommender would need outcome data the project does not have, and
would be far harder to defend.

Catalog constraints, enforced by test:

- Every intervention is **supportive** — tutoring, mentoring, advisor outreach,
  learning-support referral, financial-aid signposting.
- No punitive, surveillance-increasing, or stigmatising action may appear.
  `tests/unit/test_intervention_catalog.py` asserts a denylist of punitive
  keywords and asserts every risk factor maps to at least one supportive
  action.
- Recommendations are created with status `recommended` and require a human to
  assign them. Nothing auto-executes.

### LLM scope: narrow, optional, off by default

An LLM is used for exactly one feature: converting the already-computed factor
list and intervention set into a short, warm, non-stigmatising case note for a
counsellor.

Hard constraints:

- **Input is derived data only** — factor names, impact bands, risk band,
  intervention names. It never receives raw student records, the student
  identifier, free-text notes, or any demographic attribute.
- **It cannot influence the risk score or the recommendations.** It is a
  rendering layer downstream of both. No LLM output is persisted as a feature
  or fed back into the model.
- **Feature-flagged off by default** (`ENABLE_LLM_NARRATIVE=false`). With the
  flag off or the API key absent, the deterministic Jinja template renders
  instead, and the product is fully functional.
- The same forbidden-language test applies to its system prompt and to a
  recorded sample of its outputs.

Rationale: this uses an LLM for language, which it is good at, and keeps
judgement in the traditional model, which is auditable and calibrated. The core
prediction system contains no LLM. If a reviewer asks "is this just an LLM
wrapper?", the answer is demonstrably no.

## Consequences

- Explanations are defensible and the causal-claim risk is contained by CI.
- The LLM feature is cuttable without affecting the system, which is the right
  property for an optional enhancement.
- Slight verbosity in the UI from mandatory disclaimers. Accepted: this is an
  educational risk system and the disclaimer is load-bearing, not decorative.
