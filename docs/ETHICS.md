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

`make fairness` regenerates `reports/fairness_audit.md` and the CSVs under
`reports/fairness/`. Five attributes are audited — gender, age band, deprivation
band, declared disability, region — read from the isolated
`student_demographics` source. `evaluation/fairness.py` is their only consumer;
none is a feature, and a test asserts none appears in the allowlist.

### What the audit tests, and what it refuses to test

**Alert-rate differences are expected and are not the finding.** A calibrated
model flags higher-base-rate groups more often because those students really do
withdraw more often. Phase 3 recorded those base rates before any model existed,
precisely so this audit could not mistake them for model behaviour. Demographic
parity is the wrong test here, and chasing it would mean withholding support
from the group that needs it most.

**The question is whether errors differ**, and specifically the false-negative
rate: an at-risk student the model misses receives no offer of help. A higher
false-positive rate mostly costs staff time — it matters, but it is not the same
kind of harm.

**A gap whose confidence intervals overlap is not a finding.** Ranking groups and
writing up the largest difference is how a fairness audit manufactures
conclusions; at a 3% positive rate almost any ranking produces a plausible gap.
Overlap is checked, and the column is named `distinguishable_from_noise` rather
than `conclusive` so it cannot be confused with "this group had enough data".

### Result at the 5% alert budget

**No false-negative gap survives the overlap check.** Every one of the five
attributes shows overlapping intervals:

| Attribute | Worst-served | FNR | Best-served | FNR | Gap | Distinguishable |
|---|---|---:|---|---:|---:|---|
| region | Ireland | 85.2% | North Western | 67.8% | 17.4pp | no |
| imd_band | 60–70% | 82.4% | 0–10% | 68.0% | 14.4pp | no |
| disability | N | 74.7% | Y | 69.1% | 5.7pp | no |
| age_band | 35–55 | 75.3% | 0–35 | 73.3% | 2.0pp | no |
| gender | F | 74.7% | M | 73.4% | 1.4pp | no |

Three things to be clear about:

1. **This is an absence of evidence of disparity, not evidence of fairness.**
   The distinction is the whole point. A 17-point regional gap that fails the
   overlap check is not a gap that has been ruled out — it is a gap this sample
   cannot resolve either way. A larger cohort might well resolve it.
2. **The overall false-negative rate is ~74% in every group.** That is the
   context for the table above: the model already misses roughly three of every
   four withdrawing students everywhere. Subgroup gaps here are differences in
   how a large failure is distributed, not the difference between working and
   not working.
3. **`disability Y` reads the way a calibrated model should.** Its base rate is
   higher (4.65% against 2.74%), so its alert rate is higher (7.70% against
   6.06%) — the expected consequence of calibration, not a disparity — and its
   false-negative rate is slightly *lower*, meaning marginally more of those
   students are found.

### The one directional finding worth naming

`disability Y` carries the largest calibration gap in the audit, and it points
the wrong way: mean predicted risk 3.32% against an observed 4.65%. The model
**understates** risk for the group with the highest base rate. It is small
(-1.3pp), it is not tested for significance, and it does not currently change
who gets flagged at a 5% budget. It is recorded here because it is the only
result in the audit with a direction that would deny support if it grew, and
because suppressing a small inconvenient number is how the next audit ends up
unable to see a large one.

### Groups the audit could not assess

`age_band 55<=` — 390 rows, 13 positives. Reported as inconclusive rather than
omitted. Its point-estimate recall is high and completely uninformative: the
interval spans most of [0, 1]. An audit that silently drops what it could not
measure reads as though it measured everything.

### What this audit does not establish

- **Excluding protected attributes from features does not make a model fair.**
  Engagement correlates with employment, caring responsibilities and
  connectivity, so proxies exist whatever the feature list says. That is why
  error rates are measured rather than assumed.
- **Intersections are not tested.** Each attribute is assessed marginally, and
  at a 3% positive rate this cohort cannot support intersectional cells. A
  disparity affecting a combination of attributes would not appear here.
- **The label under-counts disengagement.** It records formal de-registration;
  students who stop engaging without withdrawing are labelled negative. That
  under-counting is unlikely to be uniform across groups, which would bias every
  rate in this audit in a direction it cannot measure.
- **One institution, one cohort, 2013–2014.** These results do not transfer.

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
