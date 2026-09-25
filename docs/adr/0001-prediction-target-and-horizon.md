# ADR-0001 — Prediction target, time origin, and horizon

- **Status:** Accepted (horizon value provisional — see "Revisit" below)
- **Date:** 2026-09-25
- **Supersedes:** none

## Context

"Predict student dropout" is underspecified, and the ambiguity is the single
largest source of invalid results in published student-dropout work. Three
distinct framings get conflated:

1. **Retrospective classification.** One row per student, all features computed
   from the student's complete record, label = final outcome. This is what most
   public notebooks do. It is *not* an early-warning system: features computed
   from the full record include information from after the point at which an
   intervention would have to happen. Reported accuracies in the 90s are
   routinely an artefact of this.
2. **Survival analysis.** Model time-to-withdrawal with censoring. Statistically
   the most principled framing, and it handles censored students correctly.
3. **Point-in-time binary classification with a fixed horizon.** At a defined
   checkpoint, predict whether the event occurs within the next H days.

We need a framing that (a) is honest about what information is available when,
(b) produces a probability a counsellor can act on this month, and (c) supports
a *trajectory* — risk re-estimated repeatedly over time.

## Decision

We adopt framing 3.

**Target.** For a student *s* and checkpoint *t*:

```
y(s, t) = 1  if s withdraws in the window (t, t + H]
          0  otherwise
```

**Time origin.** Days are measured relative to the start of the student's
course presentation, not calendar date. This makes checkpoints comparable
across cohorts that start at different times.

**Checkpoints.** Fixed course-relative days: **30, 60, 90, 120, 150, 180**.
One row per (student, checkpoint) pair.

**Horizon.** `H = 30` days, as the initial value.

**Inclusion rules.** A (student, checkpoint) row is included only if:

- the student is still actively enrolled at day `t` (no rows are emitted at or
  after a student's withdrawal — post-event rows would be trivially separable
  and would leak the outcome), and
- the course presentation has at least `t + H` days of observation remaining,
  so the label is fully determined. Rows whose label window extends past the
  end of observation are **dropped, not assumed negative**. Treating
  administratively censored students as non-withdrawers is a labelling error
  that biases the model toward under-predicting risk.

**Features.** Every feature for row `(s, t)` is computed from records with
timestamp `<= t` only. This is enforced by a property test, not by convention
(see ADR-0003).

## Consequences

Positive:

- "Early warning" becomes a verifiable property of the system rather than a
  claim in a README: the model demonstrably has no access to post-`t` data.
- Risk trajectory falls out naturally — six predictions per student over a
  presentation.
- Recall at *early* checkpoints becomes measurable and reportable separately,
  which is the metric that actually matters operationally. A model that only
  identifies at-risk students at day 180 has little intervention value even
  with excellent aggregate metrics.

Negative / costs:

- Rows are not independent: one student contributes up to six rows. All
  cross-validation must group on `student_id` (ADR-0003), and standard
  `train_test_split` is invalid here.
- The positive class becomes much rarer than the raw withdrawal rate suggests.
  A cohort with ~30% eventual withdrawal spread over six 30-day windows may
  yield a per-row positive rate in the low single digits. This drives the
  imbalance strategy and makes PR-AUC, not ROC-AUC, the headline metric.
- Censoring-based row exclusion discards data near the end of each
  presentation.

## Rejected alternatives

- **Retrospective classification** — rejected: leaks by construction, and
  cannot express a trajectory.
- **Survival analysis (Cox / discrete-time hazard)** — rejected as the
  *primary* framing, not on statistical grounds but on interpretability and
  scope: SHAP over a calibrated probability is far easier to present honestly
  to a non-technical counsellor than a hazard ratio, and the project's stated
  model set is LR / RF / XGBoost. Noted in `docs/FUTURE_WORK.md` as the most
  defensible extension. A discrete-time hazard model is in fact closely
  related to what we are fitting, which is worth stating in the README.

## Revisit

`H = 30` is provisional and **must be re-evaluated in Phase 3 (EDA)** against
the observed positive rate per checkpoint. Decision rule agreed in advance, so
the choice is not made post-hoc to flatter the metrics:

- If the day-30 checkpoint positive rate is `>= 1.5%`, keep `H = 30`.
- If it falls below that, report `H = 30` **and** `H = 60` side by side and
  designate the larger horizon as primary, documenting the tradeoff in
  intervention lead time.

`H` is a pipeline parameter (`config/features.yaml`), not a constant, so this
is a config change rather than a rewrite.
