# Prediction task specification

The authoritative definition of what this system predicts. Any code, metric, or
claim that contradicts this document is a bug. Rationale lives in
[ADR-0001](adr/0001-prediction-target-and-horizon.md) and
[ADR-0003](adr/0003-leakage-controls-and-splitting.md).

## One-sentence statement

> Given only the information available about a student as of course-relative
> day `t`, estimate the probability that the student withdraws from their
> course within the following `H = 30` days.

## Formal definition

| Element | Definition |
|---|---|
| **Unit of prediction** | `(student_id, checkpoint_day)` — a student is scored repeatedly |
| **Time origin** | Day 0 = first day of the student's course presentation |
| **Checkpoints `t`** | 30, 60, 90, 120, 150, 180 (course-relative days) |
| **Horizon `H`** | 30 days (provisional — see Revisit in ADR-0001) |
| **Label `y(s,t)`** | 1 if `t < withdrawal_day <= t + H`, else 0 |
| **Feature cutoff** | All features from records with `timestamp <= t`, strictly |
| **Positive class** | Withdrawal within the horizon (the rare, costly-to-miss class) |

## Row inclusion rules

A row `(s, t)` enters the dataset if and only if all hold:

1. **Still enrolled at `t`.** `withdrawal_day` is null or `> t`. No rows are
   emitted at or after withdrawal.
2. **Label fully observed.** The presentation has observation through `t + H`.
   Rows whose horizon extends beyond available observation are **dropped**.
3. **Minimum history.** At least one engagement or assessment record exists at
   or before `t`. Students with an entirely empty record are excluded and
   counted separately in the data card — they are a real operational
   population (never-engaged registrants) but they are a degenerate modelling
   case and would dominate early checkpoints.

### Why censored rows are dropped rather than labelled 0

Labelling an unobserved horizon as "did not withdraw" asserts a negative that
the data does not support. At the final checkpoints this systematically
under-states risk and biases the model toward optimism — the opposite of the
error this system should make. Dropping costs sample size and is the correct
tradeoff.

## What is explicitly NOT predicted

- Whether a student will *ever* drop out.
- Degree completion, final grade, or academic ability.
- Anything about a student's worth, effort, or potential.

## Forbidden features

Never permitted in the feature matrix, enforced by allowlist plus test:

- `date_unregistration` / `withdrawal_day` and anything derived from them
- `final_result` or any end-of-course outcome
- Any record with `timestamp > t`
- Protected attributes (see [ETHICS.md](ETHICS.md)) — these live in an isolated
  table used only by the offline fairness audit

## Evaluation contract

- **Headline metric:** PR-AUC on the held-out final presentation. ROC-AUC is
  reported but is not the headline — it is optimistic under severe class
  imbalance.
- **Operating point:** chosen on the validation presentation to hit a target
  positive-class recall (initial target **0.80**), with the resulting
  precision and absolute alert volume reported alongside. Never the default
  0.5 threshold.
- **Mandatory breakdown:** metrics reported **per checkpoint**. Aggregate
  metrics hide the failure mode that matters — a model that only detects risk
  at day 180 has little intervention value.
- **Calibration:** reliability curve and Brier score reported. A displayed
  percentage must mean what it says.
- **Uncertainty:** all test metrics carry bootstrap confidence intervals. The
  test set is a single presentation, so point estimates alone would overstate
  precision.
- **Baselines for context:** majority-class, and a single-rule heuristic
  ("attendance/engagement below threshold"). A model that cannot beat the
  one-rule heuristic does not justify its complexity.

## Configurability

`H`, the checkpoint list, the recall target, and the risk-band cutoffs are all
config values (`src/dropout_ews/config/features.yaml` and `thresholds.yaml`),
not constants. Risk bands are an **institutional policy choice** calibrated on
validation data, not a scientifically validated scale — see
`thresholds.yaml` for the calibration procedure.
