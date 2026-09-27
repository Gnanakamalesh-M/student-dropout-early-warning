# Monitoring and the retraining policy

`make drift` regenerates `reports/drift_report.md` and the CSVs under
`reports/drift/`. The module is `src/dropout_ews/monitoring/drift.py`.

The report is run against a real pair of cohorts — train presentations
(`2013B`, `2013J`, 65,443 rows) against the held-out test presentation (`2014J`,
43,407 rows) — rather than against a synthetic shift. That makes it a self-test
as well as a measurement: test-set performance is already published in the model
card, so a monitor that panics about a cohort the model handled fine is a monitor
that is miscalibrated.

---

## Three constraints this problem imposes

### 1. Drift must be measured within checkpoint

Engagement decays across a presentation for everyone, so a cohort at day 150
looks nothing like the same cohort at day 30. Pooling checkpoints makes the
number uninterpretable **in both directions**, and I only established the second
by running it:

- **Pooling can invent drift.** Two cohorts with identical per-checkpoint
  distributions still differ once pooled if their checkpoint composition differs
  — and a mid-presentation cohort always has a different composition.
- **Pooling can hide drift.** This is what actually happened.
  `days_since_last_submission` shifted **+6.3 days at checkpoint 90** and
  **−7.3 days at checkpoint 150**: opposite directions, similar magnitudes.
  Pooled, the means are 23.6 against 24.6, PSI reads 0.19 — a "minor" band. Within
  checkpoint, **all six checkpoints are major**, PSI 0.71 to 1.15. Across the
  whole feature set, pooling reported **1** major feature where the
  within-checkpoint comparison found **7**.

I designed the module around the first effect and wrote a test for it. The real
data showed the second, which is the dangerous one: a monitor that averages a
drifting cohort into looking healthy will not fire when it should. Both
directions are now pinned by tests, and `by_checkpoint=True` is the default.

### 2. Labels arrive 30 days late, so anything label-based is stale

The target is "withdraws within 30 days of the checkpoint". A row scored today
cannot be scored *against* for a month, and a presentation's last checkpoints
resolve only after it ends. So:

| Signal | Needs labels? | Available when? |
|---|---|---|
| Feature drift | no | immediately |
| Prediction drift | no | immediately |
| Coverage change | no | immediately |
| Calibration drift | **yes** | ≥30 days late |
| Recall / precision | **yes** | ≥30 days late |

This is the shape of the problem, not a gap in the implementation. It is why
`prediction_drift` and `calibration_drift` are separate functions, and why
`CalibrationDrift` carries a `labels_complete` flag: a calibration gap computed
over unresolved rows counts every pending student as a negative, which biases the
observed rate downward and makes the model look over-confident when it may not
be.

**If only one signal is watched, watch prediction drift** — specifically
`alert_rate_change`. It needs no labels and it is the operationally meaningful
number: the cutoff was chosen for a 5% staffing capacity, so a cohort that pushes
the flagged share up has silently increased the workload the operating point
promised, whether or not the model is still accurate.

### 3. PSI thresholds are convention, and PSI is biased upward on small samples

The 0.10 / 0.25 bands come from credit-scoring practice. They are **not**
validated for this task, feature set, or cohort size, and nothing here
establishes that a PSI of 0.26 means something different from 0.24.

Worse, at small *n* the bands are actively misleading. Measured — 300 trials per
row, two samples drawn from the **same** normal distribution, so the true PSI is
zero — with a fixed 10 bins against a 2,000-row reference:

| current n | median PSI | 90th pct |
|---:|---:|---:|
| 25 | 1.3115 | 2.5697 |
| 50 | 0.1793 | 0.3677 |
| 100 | 0.0910 | 0.1641 |
| 200 | 0.0449 | 0.0777 |
| 400 | 0.0256 | 0.0445 |
| 5000 | 0.0059 | 0.0100 |

At n=50 the *median* noise reading already sits in "minor drift" and the 90th
percentile clears "major" — from nothing at all. A monitor using the conventional
thresholds on a small checkpoint would fire every term on cohorts that had not
changed.

So `population_stability_index` adapts the bin count to the smaller sample
(~40 rows per bin) and refuses outright below 50 rows. Re-measured with adaptive
bins the 90th-percentile noise floor is **0.059 at n=50** and 0.043 at n=400,
while a genuine one-SD shift still reads **0.56 at n=50** — roughly a 6x
reduction in the false-positive floor at no measurable cost in sensitivity. Two
tests assert both halves of that property, because restoring a fixed bin count is
an easy thing to do while tidying.

---

## What the real comparison found

`requires_review: True`. 32 features × 6 checkpoints.

| Band | Count | Features |
|---|---:|---|
| major | 7 | `assessments_due`, `assessments_late`, `assessments_submitted`, `days_since_last_submission`, `mean_score`, `min_score`, `submission_rate` |
| minor | 6 | `active_days_14d`, `clicks_14d`, `clicks_delta_4w`, `clicks_ratio_4w`, `mean_submission_lag`, + overlap |
| not measurable | 5 | `assessments_failed`, `assessments_missed`, `checkpoint_day`, `ever_active`, `has_submitted` |
| coverage change | 1 | `submission_rate` (+5.5pp missing at checkpoint 30) |

### Every major-drift feature is assessment-derived

That is the whole finding, and it has a cause in the source data. The assessment
schedule changed between presentations:

| Module | 2013 assessments | 2014J assessments |
|---|---:|---:|
| BBB | 12 | 6 |
| DDD | 14 (2013B) | 7 |

`assessments_due` has the largest PSI in the report (5.74 at checkpoint 180, mean
6.41 → 4.36), and every feature derived from the assessment calendar moved with
it. **No clickstream feature exceeds the minor band.** So the drift is a
curriculum change, not a change in student behaviour — and a monitor *should*
scream about it, because the feature means genuinely moved.

The test cohort also contains a module the training cohort does not: `CCC`
appears only in `2014J`. That is population drift of a different kind — a course
the model never trained on — and it contributes to the assessment-feature shift.

### And yet test performance held

This is the part that determines the policy. Those seven majorly-drifted features
coexisted with the test-set performance reported in the model card. PR-AUC 0.0885
against a 2.9% base rate is what it is *on this drifted cohort*.

Calibration also did not degrade — it slightly improved:

| | Predicted | Observed | Gap |
|---|---:|---:|---:|
| Reference | 0.0311 | 0.0272 | +0.0039 |
| Current | 0.0303 | 0.0290 | +0.0013 |

Change in gap **−0.0026**. Negative means less over-statement, same sign
convention as the fairness audit (negative = understates risk) so the two reports
cannot be misread against each other.

Prediction drift is mild and points *downward* — the newer cohort scores lower,
so fewer students are flagged:

| Checkpoint | PSI | Band | Ref alert rate | Current alert rate | Change |
|---:|---:|---|---:|---:|---:|
| 30 | 0.049 | stable | 9.61% | 7.12% | −2.48pp |
| 60 | 0.018 | stable | 8.15% | 6.91% | −1.24pp |
| 90 | 0.165 | minor | 9.88% | 7.46% | −2.42pp |
| 120 | 0.021 | stable | 7.88% | 6.74% | −1.14pp |
| 150 | 0.042 | stable | 8.79% | 5.56% | −3.23pp |
| 180 | 0.138 | minor | 2.53% | 3.03% | +0.50pp |

**An operational finding worth naming separately:** the reference alert rate is
9.6% at checkpoint 30 but 2.5% at checkpoint 180, against a nominal 5% budget.
One global cutoff produces very uneven per-checkpoint workload, because late-term
rows score systematically lower. If staffing is allocated per checkpoint rather
than per term, the threshold should be derived per checkpoint too. That is a
change to the operating point, not to the model, and it is not made here.

---

## Retraining policy

**Nothing in this repository triggers an automatic retrain.** `DriftSummary`
exposes `requires_review` and deliberately has no `requires_retraining` — a test
asserts the absence.

The reason is the result above: a whole presentation's worth of major feature
drift coexisted with stable, calibrated performance. Drift in an input
distribution is **not** evidence that the model got worse. Retraining on a drift
signal alone would replace a model known to work with one whose behaviour nobody
has measured, and would do it on a schedule set by the curriculum committee.

### What escalates

`requires_review` is True when **any** feature shows major drift at **any**
checkpoint, or when any feature's missing-rate changes by more than 5 points.

Escalation is per checkpoint rather than averaged across them, deliberately: a
severe shift at day 30 — where an early-warning system is actually useful — must
not be cancelled out by stability at day 180, where a withdrawal warning arrives
too late to act on.

### What a human should then check, in order

1. **Coverage first.** A feature that stopped being populated shows a *stable*
   PSI on the rows that remain, because those rows really are unchanged. This is
   the failure an upstream pipeline change actually produces, and PSI alone cannot
   see it. Coverage is tracked separately for exactly this reason; there is a test
   where PSI reads "stable" while the monitor still escalates.
2. **Is the drift structural or behavioural?** A changed assessment calendar
   (structural) is different from students engaging less (behavioural). The first
   may need feature definitions revisited; the second may be the signal the system
   exists to detect. The report cannot tell these apart — that needs someone who
   knows what changed in the courses.
3. **Has the alert rate moved?** This is the capacity question and it is
   answerable immediately, without labels.
4. **Only once labels resolve**, has calibration or recall actually degraded?
   This is the only evidence that would justify retraining, and it is at least 30
   days behind.

### Retraining is justified by measured degradation, not by drift

Concretely: a sustained calibration gap or a drop in recall at the operating
point, on resolved labels, with the same cluster-bootstrap intervals used in
Phase 6 — so the comparison is against sampling noise rather than against the
previous point estimate. A new model then goes through the whole Phase 5–6 route
again: grouped CV, a threshold chosen on validation, calibration assessed, and
the fairness audit re-run before anything is deployed.

---

## What this monitoring does not do

- **No live monitoring loop.** These are batch reports run against stored
  cohorts. Scheduling, alerting, and dashboarding are deployment concerns and
  Docker is not available on the machine this was built on, so nothing that
  cannot be verified here has been written as though it works.
- **No drift on protected attributes.** Deliberate: they are not features, and
  `evaluation/fairness.py` is their only consumer. Subgroup error rates are the
  fairness question, and they belong in `docs/ETHICS.md`, not here.
- **No concept-drift test.** Whether the *relationship* between features and
  withdrawal has changed — as opposed to the feature distributions — needs
  resolved labels and a second fitted model to compare against. That is a
  retraining experiment, not a monitor.
- **`checkpoint_day` is reported as not measurable**, which is correct but
  trivial: it is constant within each checkpoint group by construction. It stays
  in the feature list because the model uses it, and the monitor reports honestly
  that it cannot assess it.
