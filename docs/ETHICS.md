# Ethics, privacy, and governance

## What this system is

An **early-warning support tool**. It estimates the probability that a student
withdraws in the next 30 days, so that an institution can offer help sooner.

## What this system is not

It does not determine, predict, or describe a student's future, ability,
effort, or worth. A risk score is a probabilistic signal computed from
attendance, academic, and engagement records — nothing more.

**It must never be used for:**

- admissions, selection, or scholarship decisions
- disciplinary action of any kind
- withdrawing support, funding, or opportunity from a student
- any automated decision affecting a student without human review
- ranking, publishing, or comparing students
- performance management of staff

A model with meaningful false-positive and false-negative rates — which this
one has, like all such models — is not fit for any of the above. Its errors are
acceptable only because the downstream action is *an offer of support*, where a
false positive costs an unnecessary conversation rather than harm to a student.

## Data minimization

The feature matrix contains **no** name, contact detail, ethnicity, caste,
religion, disability status, or any other protected attribute.

Where the source data carries such attributes (OULAD has `imd_band` area
deprivation and `age_band`), they are routed to a separate
`student_demographics` table read **only** by the offline fairness audit. They
are never joined into the feature pipeline. This is structural: the fairness
audit needs them, the model must not have them, and separate tables make that
distinction enforceable rather than aspirational.

Student identity in the application is an opaque `student_code`, not a name.

## Fairness

Phase 12 reports per-subgroup recall, precision, and false-positive rate with
confidence intervals.

Two honesty commitments:

1. **Excluding protected attributes from features does not make a model fair.**
   Proxies exist — engagement patterns correlate with employment, caring
   responsibilities, and connectivity. Measuring subgroup performance is
   therefore necessary regardless of what the model can see.
2. **Where subgroup samples are too small to support a conclusion, we say so**
   rather than reporting a reassuring point estimate. A wide confidence
   interval is a finding, not a failure to report.

Note the direction that matters here: a *lower* false-negative rate for a group
is good (more students found), while a higher false-positive rate mainly costs
staff time. This asymmetry should shape how any disparity is interpreted.

## Governance — human in the loop

Hard constraints, enforced in code rather than policy:

- Interventions are created with status `recommended` and require explicit
  assignment by a member of staff. Nothing auto-executes.
- `require_human_review_before_intervention` cannot be set to false — the
  config validator rejects it (`config/settings.py`).
- `auto_punitive_actions_permitted` cannot be set to true.
- The intervention catalog is tested against a punitive-keyword denylist; only
  supportive actions may appear.
- Every rendered explanation carries a disclaimer, and narrative templates are
  tested against causal language ("caused", "because of", "will drop out").
  See [ADR-0005](adr/0005-explainability-and-llm-scope.md).

## Access control and audit

- Roles: `admin`, `counsellor`, `analyst`. `analyst` sees aggregates only and
  is denied individual student profiles.
- Every read of an individual student record is written to `audit_log` with
  actor, action, resource, and timestamp. Being able to answer "who looked at
  this student's risk profile, and when" is a baseline requirement for a system
  like this.
- JWTs with short expiry; passwords hashed with argon2.
- TLS in transit; database encryption at rest documented in
  `docs/DEPLOYMENT.md`.

## Transparency to students

This project is a portfolio system without real students, but a real deployment
should: tell students the system exists, explain what data feeds it in plain
language, offer a route to contest or query a risk assessment, and never
present a score to a student as a prediction about their future.

## Risk-band thresholds are policy, not science

The band cutoffs are an institutional choice about support capacity, not a
validated scale. `config/thresholds.yaml` documents the calibration procedure
and the shipped defaults are explicitly flagged `calibrated: false`. Any claim
that these thresholds are scientifically validated would be false.

## Known ethical limitations

- Trained on UK distance-learning data from 2013–2014; applying it elsewhere
  without re-validation would be inappropriate.
- The label is administrative de-registration, which under-counts students who
  disengage without formally withdrawing — likely not at random across groups.
- Engagement features reward students who can be online frequently, which is
  not evenly distributed. The model may partly measure circumstance rather than
  risk, and outreach driven by it should be framed as an offer of help, never
  as a judgement about commitment.
