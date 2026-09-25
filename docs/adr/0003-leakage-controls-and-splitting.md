# ADR-0003 — Leakage controls and evaluation splitting

- **Status:** Accepted
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
