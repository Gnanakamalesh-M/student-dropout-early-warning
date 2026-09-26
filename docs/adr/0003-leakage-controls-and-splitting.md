# ADR-0003 — Leakage controls and evaluation splitting

- **Status:** Accepted, **amended 2026-09-26** — see
  [Amendment: chronology alone was not sufficient](#amendment-2026-09-26--chronology-alone-was-not-sufficient).
- **Date:** 2026-09-25

## Context

ADR-0001's framing creates three concrete leakage risks, each of which is
individually sufficient to invalidate all reported metrics:

1. **Temporal leakage** — a feature for checkpoint `t` incorporates data from
   after `t`. Sources: rolling windows that centre rather than trail,
   full-dataset aggregates, and outcome-adjacent columns
   (`date_unregistration`, `final_result`).
2. **Group leakage** — the same student appears in both train and test. Because
   one student contributes up to six rows with heavily autocorrelated features,
   a random split lets the model memorise students rather than learn patterns,
   inflating scores substantially.
3. **Preprocessing leakage** — imputation values, scalers, encoders, or cohort
   statistics fitted on data that includes the test set.

## Decision

### Four enforced controls

**1. The as-of property test (primary guarantee).**

For a set of fixture student histories and every checkpoint `t`:

```python
full = FeatureBuilder().build(history, as_of=t)
trimmed = FeatureBuilder().build(history.filter(timestamp <= t), as_of=t)
assert full.equals(trimmed)  # exact, not approximate
```

If any feature reads post-`t` data, this fails. It runs in CI on every commit
and is the single most important test in the repository. Implemented with
Hypothesis over generated histories in addition to hand-built fixtures.

**2. Forbidden-column allowlist.** The feature matrix is built from an explicit
allowlist in `config/features.yaml`, never by dropping columns from a wide
frame. A test asserts that `date_unregistration`, `final_result`, and any
column matching `*_final`, `*_outcome`, `withdraw*` are absent from the trained
model's feature names. Denylists silently fail when a new column appears;
allowlists fail loudly.

**3. Split design — temporal outer, grouped inner.**

OULAD presentations are ordered in time (`2013B < 2013J < 2014B < 2014J`).

- **Test set:** the chronologically **last** presentation, held out entirely.
  This is a forward-in-time evaluation — training data precedes test data in
  wall-clock time, matching deployment, where a model trained on past cohorts
  scores a future one.
- **Validation set:** the second-to-last presentation, used for
  hyperparameter selection, probability calibration, and threshold/band
  selection.
- **Training set:** all earlier presentations.
- **Inner CV:** `StratifiedGroupKFold(groups=student_id)` within training data.
  Grouping is mandatory; stratification keeps the rare positive class present
  in every fold.

A test asserts zero `student_id` intersection across all splits and that
`max(train.date) <= min(test.date)`.

**4. Everything fitted lives inside a `Pipeline`.** Imputers, scalers,
encoders, and resamplers (via `imblearn.Pipeline`) are fitted per fold. Nothing
is fitted on the full frame. Cohort-relative z-scores (ADR: feature design) use
**training-period cohort statistics only**, stored as part of the fitted
transformer, not recomputed at inference.

### Reporting requirement

Because these controls typically *lower* headline numbers relative to naive
published baselines, the README includes a short "Why our numbers look lower"
section explaining each control. Under-reporting with correct methodology is
the intended outcome, and it should be presented as a feature of the work, not
apologised for.

## Consequences

- Metrics will be materially lower than typical public student-dropout
  notebooks. This is correct and expected.
- Only one presentation is available as a test set, so test-set confidence
  intervals will be wide. All reported metrics carry bootstrap CIs; point
  estimates alone are not reported.
- Development is slower: every new feature needs a fixture demonstrating its
  as-of behaviour.

## Rejected alternatives

- **Random `train_test_split`** — invalid under both group and temporal
  dependence.
- **Plain `StratifiedKFold`** — ignores student grouping; inflates scores.
- **`TimeSeriesSplit` on rows** — rows are not a single ordered series; there
  are many students in parallel, so this neither groups nor respects
  presentation boundaries.

## Amendment (2026-09-26) — chronology alone was not sufficient

Control #3 as originally written assumed that splitting on presentation code
would separate students, and treated a student straddling the boundary as a
theoretical possibility to guard against. Implementing it in Phase 4 showed the
assumption was wrong on real data.

**What was found.** `id_student` is reused across presentations as well as
across modules: **1,954 of 24,710 students (7.9%)** appear in more than one
presentation, having retaken a module or studied several. A purely chronological
split leaves **664 students in both train and validation** and **758 in both
train and test**. The guard fired on the first real run, which is the only
reason this was caught rather than silently inflating every Phase 5 metric.

**Resolution adopted.** Students appearing in an earlier split are removed from
the **later** split. Training keeps them; validation and test give them up. The
direction is deliberate — contaminating the evaluation sets is the error that
inflates reported metrics, whereas a slightly smaller training set only costs a
little signal.

**Cost, recorded rather than buried.**

| | Rows dropped | Share | Positive rate before | After |
|---|---:|---:|---:|---:|
| Validation | 3,398 | 9.6% | 3.35% | 3.10% |
| Test | 6,614 | 13.2% | 3.18% | 2.90% |

Resulting split: train 65,443 rows / 11,643 students; validation 32,076 /
5,539; test 43,407 / 7,528; 10,012 rows set aside as `dropped_overlap`, which
is retained and reported so the exclusion is auditable.

**New limitation this creates.** The evaluation sets now **under-represent
repeat students**, who plausibly differ in withdrawal risk from first-time
students. Test metrics therefore describe a slightly different population than
the full cohort. This belongs in the model card and is not fixable within a
chronological design — the alternative would be assigning whole students to
splits, which would sacrifice the forward-in-time property that makes the
evaluation honest in the first place.

**Also confirmed in this phase.** Control #1, the as-of property test, is
implemented in `tests/ml/test_as_of_property.py` using Hypothesis over generated
histories plus hand-built adversarial fixtures, and runs in CI. Control #2, the
allowlist, is bound to the code by `tests/ml/test_feature_allowlist.py`; that
test immediately caught a silent config bug in which 15 features were being
discarded because the schema lacked the matching group fields.
