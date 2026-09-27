# ADR-0005 — Explainability language and the scope of LLM use

- **Status:** Accepted, **implemented 2026-09-27** — see
  [Implementation notes](#implementation-notes-2026-09-27).
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

## Implementation notes (2026-09-27)

Everything in this ADR is implemented and enforced by tests. Three things turned
out differently from the plan, and one assumption needed checking before it was
safe to build on.

**The `shap` package could not be used to compute the values.** `shap` 0.49
parses XGBoost's `base_score` as a float, and XGBoost 3.x serialises it as an
array string (`'[2.7229803E-2]'`), so `TreeExplainer` raises on construction.
Rather than pin XGBoost back and retrain, values come from XGBoost's own
`pred_contribs=True` — the same exact TreeSHAP algorithm, implemented inside
XGBoost. Additivity against the model margin is asserted in CI to 1e-4.

**Attributions explain the pre-calibration score, not the displayed
probability.** The artifact is a calibrated wrapper, and TreeSHAP cannot see
through it. This is sound because isotonic calibration is monotone: a feature
pushing the raw score up also pushes the calibrated probability up, so the
ranking the explanation describes is the ranking the displayed figure reflects.
What is lost is any numeric percentage-point attribution, so the templates never
make one and a test asserts no rendered sentence contains a percentage.

**Attribution stability was measured before any of this was wired to a UI.** The
risk was specific: at a 2.7% positive rate with 37 correlated features,
per-student attributions could have been small and unstable, and two
near-identical students receiving different "main reasons" would mislead staff
even with a well-calibrated probability. Measured by top-3 factor overlap against
nearest neighbours in the model's own feature space:

| Population | Mean overlap | Median | 10th pct |
|---|---:|---:|---:|
| All test rows | 0.761 | 0.733 | 0.600 |
| Top 5% by risk | **0.902** | **1.000** | 0.667 |

Attributions are *more* stable for high-risk students, which is the right
direction since those are the ones staff see. They are not perfectly stable,
which is exactly why the mandated wording is "factors the model weighted most
heavily" rather than "the reasons".

**A design change the data forced.** The single largest global contributor is
`checkpoint_day` — how far through the course a student is. That is real signal,
but "this student is at risk because it is day 30" is not something anyone can
act on. Features are therefore tagged actionable or contextual and rendered in
separate lists, so context does not crowd out factors staff can respond to. A
test asserts `checkpoint_day`, `num_of_prev_attempts` and `date_registration` stay
marked non-actionable.

**Both keyword guards needed negative-context handling.** The punitive denylist
initially blocked the most supportive line in the catalog — "reducing load
without penalty" — because it substring-matched `penal`. The language lint
similarly flagged the disclaimer's own phrase "not the causes of withdrawal".
Both now strip an explicit allowlist of permitted phrases and match on word
boundaries, and each has a test asserting the allowance did not make the guard
permissive. A keyword check with no negative-context handling is either too loose
to be useful or too tight to permit correct wording.
