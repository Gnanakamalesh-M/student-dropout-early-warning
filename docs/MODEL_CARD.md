# Model card — Student Dropout Early-Warning System

Primary model: **XGBoost, isotonically calibrated.** Generated figures live in
`reports/model_comparison.md`; regenerate with `python scripts/train_model.py`.

---

## What the model does

Given only information available about a student as of course-relative day *t*,
it estimates the probability that the student withdraws within the following
**30 days**. Students are scored at six checkpoints (days 30, 60, 90, 120, 150,
180), producing a risk trajectory rather than a single verdict.

Authoritative task definition: [TASK_SPEC.md](TASK_SPEC.md).

## What it does not do

It does not predict whether a student will *ever* drop out, degree completion,
academic ability, or anything about a student's worth or effort. It must never be
used for admissions, funding, or disciplinary decisions. See
[ETHICS.md](ETHICS.md).

---

## Performance

Held-out test set: the chronologically final course presentation (2014J), 43,407
rows, 7,528 students, **2.90% base rate**. The operating threshold was chosen on
the validation presentation and applied unchanged to test.

| Metric | Value |
|---|---|
| PR-AUC (headline) | **0.089** [0.079, 0.099] |
| ROC-AUC | 0.747 |
| Brier score | 0.0273 |
| Lift over base rate | ~3.1× |

Intervals are 95% percentile bootstrap resampling students.

### The numbers that matter operationally

PR-AUC integrates over every threshold, including operating points nobody would
use. The decision-relevant view is what the model delivers within an alert
budget an institution can actually staff:

| Alert budget | Recall | Precision | Lift | Ties at cutoff |
|---:|---:|---:|---:|---:|
| 1% of cohort | 6.8% | **19.6%** | 6.8× | 83 |
| 2% | 11.4% | 16.5% | 5.7× | 2,124 |
| **5%** | 21.8% | **12.6%** | 4.4× | 2,124 |
| 10% | 35.2% | 10.2% | 3.5× | 1,632 |

At a 5% budget, roughly **one in eight** students contacted is genuinely about to
withdraw, against a base rate of one in 34.

The ties column is not decoration. At the 2% and 5% cutoffs, 2,124 students share
the boundary probability, so which of them lands inside the budget is decided by
an arbitrary tie-break rather than by the model. Staff should not read a band
boundary as meaningful separation between adjacent students.

### Per-checkpoint

Reported separately because aggregate metrics hide the failure that matters most:
a model that only works late has little intervention value. Signal is present
from the first checkpoint, which is the property an early-warning system needs.
See `reports/model/per_checkpoint.csv`.

---

## Honest comparison against the baselines

| Model | PR-AUC | Brier | Precision @1% | @5% | @10% |
|---|---:|---:|---:|---:|---:|
| dummy (base rate) | 0.029 | — | 2.9% | 2.9% | 2.9% |
| best single rule ("no clicks in 14d") | 0.048 | n/a | — | — | — |
| logistic regression (untuned) | 0.072 | — | — | — | — |
| random forest (untuned) | **0.095** | 0.0627 | **20.5%** | 12.6% | 9.3% |
| **XGBoost (tuned + calibrated)** | 0.089 | **0.0273** | 19.6% | 12.6% | **10.2%** |

**XGBoost does not outperform the untuned random forest on predictive accuracy.**
This needs saying directly rather than being buried:

- The random forest scores *higher* on PR-AUC (0.095 vs 0.089), though the
  confidence intervals overlap almost entirely, so the two are statistically
  indistinguishable.
- At the operating points that matter they are effectively tied: identical
  precision at a 5% budget, the random forest marginally ahead at 1–2%, XGBoost
  marginally ahead at 10%.
- Forty Optuna trials bought very little over sensible defaults.

XGBoost is retained as the primary model for two reasons, neither of which is
accuracy:

1. **Calibration.** Brier 0.0273 against 0.0627 — a 2.3× improvement. The
   dashboard shows a counsellor a percentage, so the number has to mean what it
   says. Two models that rank equally well are not equally useful when one of
   them is systematically overconfident.
2. **Fast exact SHAP.** `TreeExplainer` on XGBoost gives exact attributions
   quickly, which Phase 7 depends on.

**A reasonable person could choose the random forest instead**, calibrate it the
same way, and lose nothing measurable. That option is not foreclosed, and the
baseline pipeline remains in the repository.

A prediction made before this phase was that tuning would reach PR-AUC 0.10–0.12.
It reached 0.089, below even the untuned random forest. That estimate was
optimistic and is recorded here rather than quietly dropped.

## Class imbalance: what was tested

Grouped CV on the training split only, with resamplers inside the pipeline so
they are refitted per fold — applying SMOTE before splitting leaks, because a
synthetic training point can be interpolated from a validation neighbour.

| Strategy | CV PR-AUC |
|---|---:|
| undersample majority | 0.1060 ± 0.0084 |
| none | 0.1040 ± 0.0078 |
| SMOTE + undersample | 0.1031 ± 0.0071 |
| class weights | 0.0996 ± 0.0124 |
| SMOTE | **0.0860** ± 0.0097 |

**Selection was not `argmax`.** The top four fall within fold noise of each
other, so taking the highest score would select on which fold split happened to
favour it. The rule is: keep everything within one fold standard deviation of the
best, then prefer the simplest — one that discards no data, invents no data, and
adds no randomness.

That yields **none**: no imbalance handling at all. The recorded rationale is
"within one fold SD of the best (undersample majority, 0.1060 ± 0.0084) and
simpler, so preferred over chasing a 0.0020 difference inside fold noise."

The result is worth stating plainly: at a 2.7% positive rate, with a *ranking*
metric and calibration applied afterwards, reweighting and resampling did not
help.

There is corroborating evidence for *why*. With class weights, the raw model
predicted a mean risk of 8.6% against a 3.1% observed rate and isotonic
calibration had to pull it down hard (Brier 0.0350 → 0.0290). With no
reweighting, the raw model is already nearly calibrated (mean predicted 2.7% vs
3.1% observed) and calibration barely changes anything (0.02913 → 0.02901). The
reweighting was creating the miscalibration that the calibration step then
repaired; doing neither arrives at the same place more simply.

### The SMOTE objection, tested rather than asserted

ADR-0002 rejected SMOTE on the argument that interpolating between students
manufactures impossible trajectories. That argument was converted into a
measurement. The feature set carries hard structural invariants — nested windows
(`clicks_7d ≤ clicks_14d ≤ clicks_28d`), day caps (`active_days_28d ≤ 28`), and
binary indicators — which no real student can violate.

| Dataset | Rows | Rows with an impossible value |
|---|---:|---:|
| Real students | 65,443 | **0** (0.0%) |
| SMOTE synthetic | 61,879 | **13,667** (22.1%) |

All 24,741 violations are **nested-window** violations. That is the precise
finding: SMOTE preserves each feature's marginal bounds but breaks the
*relationships between* features, because it interpolates every dimension
independently and has no notion of the constraints linking them. 22% of the
students it invents are arithmetically impossible, not merely unusual.

SMOTE was also the worst-performing strategy, which is consistent.

---

## Calibration

Isotonic regression fitted on the held-out validation presentation. Isotonic
rather than Platt: Platt assumes a sigmoid link between score and outcome, a
strong assumption at a 3% base rate, and validation has 993 positives — enough
for a nonparametric fit.

| | Before | After |
|---|---:|---:|
| Brier score | 0.0350 | **0.0290** |
| Mean predicted probability | 0.0855 | **0.0310** |
| Observed rate | 0.0310 | 0.0310 |

The raw model was overconfident by roughly 2.8×, predicting an 8.6% mean risk
against a 3.1% observed rate. After calibration the mean prediction matches the
observed rate to three decimals. Reliability by quantile bin is in
`reports/model/reliability.csv`; quantile rather than equal-width bins, because
at a 3% base rate nearly every prediction falls in the lowest equal-width bucket.

---

## Risk bands

Bands are **derived from support capacity, not chosen**, which is the procedure
`thresholds.yaml` specified before any model existed. The CRITICAL cutoff is set
so the band size matches what staff can take on.

See `reports/model/risk_bands.csv` for cutoffs with realised precision, recall
and lift per band.

Two caveats that belong here rather than in a footnote:

- **The bands are not a validated risk scale.** They are an institutional policy
  choice about capacity. A different institution with different staffing should
  re-derive them.
- **Ties at the cutoff are large.** Isotonic calibration is piecewise-constant,
  so students share identical probabilities at band boundaries — 2,124 of them at
  the 5% cutoff. `ties_at_cut` is reported alongside every budget for exactly this
  reason: where the tie is large, the model cannot distinguish the students at
  the boundary, and the choice between them is an arbitrary tie-break rather than
  a modelling decision. Staff should not read a boundary as meaningful
  separation.

---

## Training data

- **Source:** OULAD (CC-BY 4.0), 7 modules across 4 presentation codes.
- **Train:** 2013B + 2013J — 65,443 rows, 11,643 students, 2.72% positive.
- **Validation:** 2014B — 32,076 rows (calibration, bands, thresholds).
- **Test:** 2014J — 43,407 rows, evaluated once.
- **Features:** 37, all computed strictly from data at or before the checkpoint.
  See [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md).

No synthetic data is used in this model. The synthetic cohort exists to exercise
GPA/backlog/attendance features that OULAD lacks, and its metrics measure
pipeline correctness only ([DATA_CARD.md](DATA_CARD.md)).

---

## Limitations

Stated plainly, because most matter more than the headline number.

1. **The ceiling on this problem is low.** The best single heuristic reaches
   PR-AUC 0.048 and the model reaches 0.089. This is a genuine improvement, not
   a solved problem: at a 5% budget, **87% of the students contacted were not
   about to withdraw, and 78% of those who did withdraw were not contacted**.
2. **A recall target is not achievable at usable precision.** Reaching 80%
   recall means flagging 47% of the cohort at 4.9% precision. The system finds
   *some* at-risk students earlier; it does not find most of them.
3. **Evaluation under-represents repeat students.** Resolving cross-presentation
   student overlap removed 13.2% of test rows, all of them students who also
   appear in an earlier presentation (ADR-0003 amendment). Test metrics describe
   a slightly different population than the full cohort.
4. **Warning time is weeks, not months.** Engagement before withdrawal is flat
   from 9 to 5 weeks out and declines mostly in the final fortnight (Phase 3).
5. **Pre-course attrition is out of scope.** 5,127 students withdraw at or
   before the first checkpoint, 3,089 before day 0. No engagement-based model
   can flag them.
6. **One institution, one context, 2013–2014.** UK distance learning. Distance
   learners' engagement differs substantially from on-campus students; the model
   should not be applied elsewhere without re-validation.
7. **The label is administrative.** Formal de-registration under-counts students
   who disengage without withdrawing, and that under-counting is unlikely to be
   random across groups.
8. **Engagement features partly measure circumstance.** Connectivity, work and
   caring commitments all shape clickstream activity. A deprivation gradient is
   present in the base rates (Phase 3). Outreach driven by this model should be
   framed as an offer of help, never as a judgement about commitment.
9. **Tuning bought almost nothing, and the primary model is not the most
   accurate one.** 40 Optuna trials moved CV PR-AUC very little over sensible
   defaults, and the tuned XGBoost scores *below* an untuned random forest on
   PR-AUC. XGBoost is preferred for calibration and SHAP speed, not accuracy. Do
   not expect headroom from further tuning.
10. **A fairness audit has not yet been run.** Base rates by subgroup are
    recorded (Phase 3); *error*-rate parity is Phase 12 work. Until then, no
    fairness claim is made.

---

## Intended use

- **Users:** institutional support staff, with role-based access.
- **Intended use:** prioritising a limited amount of proactive outreach.
- **Human review is mandatory** before any contact. Recommendations are created
  with status `recommended` and require explicit assignment; the config
  validator rejects any attempt to disable this.
- **Explanations describe the model, not causes.** SHAP attributions are rendered
  with contribution phrasing, enforced by a test that fails the build on causal
  language.

## Reproducing this model

```bash
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/build_features.py
python scripts/train_model.py
```

Each artifact ships `metadata.json` recording metrics, hyperparameters, the
feature list, a hash of the training matrix, the package version, and the git
commit, so any stored prediction can be traced to the model that produced it.
