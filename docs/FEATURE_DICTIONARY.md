# Feature dictionary

37 features, all computed strictly from data with `timestamp <= checkpoint_day`.
Group membership and exclusions were decided by the ablation study below, not by
intuition. Regenerate with `python scripts/build_features.py`.

Provenance: **all features are derived**, not observed measurements. The raw
observations are the OULAD clickstream, assessment submissions, and enrolment
records; everything here is computed from them.

---

## The ablation study that set this allowlist

4-fold grouped CV on the **training split only** (2013B + 2013J), RandomForest,
PR-AUC. Cost = how much removing the group from the full 39-feature model
reduced PR-AUC. Fold standard deviation was ±0.0048, which is the bar a group
has to clear to count as more than noise.

| Group | Cost of removal | Verdict |
|---|---:|---|
| static | **+0.0078** | matters |
| assessment | **+0.0065** | matters |
| relative (cohort z) | **+0.0056** | matters |
| trend (all 5) | +0.0020 | within noise |
| volatility | +0.0002 | nothing |

Full allowlist: **PR-AUC 0.0986 ± 0.0044** against a 2.72% training base rate.

### This contradicts the project plan, and the plan was wrong

The plan called longitudinal trend features "one of the most important parts"
and "the part that carries the project". Removing **every** trend feature costs
+0.0020 PR-AUC — inside fold noise. What actually carries the model is
assessment behaviour, cohort-relative normalisation, and enrolment context.

The trend group is retained in reduced form (3 of 5 features) for two honest
reasons: the direction of the effect is consistently positive, and declining
engagement is a product requirement for the dashboard's risk trajectory. It is
**not** retained because it drives performance, and no claim to that effect is
made anywhere.

### Excluded features

Still computed by the builder, deliberately not modelled:

| Feature | Univariate lift | Why excluded |
|---|---:|---|
| `clicks_slope_4w` | 0.020 | Weakest trend form. The plan's centrepiece. |
| `declining_weeks_streak` | 0.026 | Near-noise; redundant with the ratio features. |
| `clicks_weekly_std` | 0.082 | Removal cost +0.0002, i.e. nothing. |

Dropping all three is free: 36 features score 0.0968 against 0.0972 for 39.

---

## level (11) — current engagement volume and breadth

Stable across every checkpoint (Phase 3 finding 2). Absence of a clickstream row
genuinely means no activity, so these are zero-filled.

| Feature | Definition |
|---|---|
| `clicks_7d` / `_14d` / `_28d` / `_56d` | Total clicks in the trailing 1/2/4/8 whole weeks |
| `clicks_all_time` | Total clicks from day 0 to the checkpoint |
| `active_days_7d` / `_14d` / `_28d` / `_56d` | Distinct days with any activity in each window |
| `active_days_all_time` | Distinct active days to date |
| `clicks_per_active_day_28d` | Intensity. 200 clicks over 10 days is a different pattern from 200 in one burst. 0 when there are no active days. |

Windows are whole weeks (7/14/28/56 days) rather than Phase 3's 30/60 days, so
the panel and the window sums stay consistent with each other.

## recency (3) — how long since the student was last seen

| Feature | Definition |
|---|---|
| `days_since_last_activity` | Checkpoint day minus last active day. **NaN** when never active. |
| `ever_active` | 0/1 indicator pairing with the NaN above |
| `inactive_weeks_streak` | Consecutive most-recent weeks with zero activity; 8 (the panel width) when never active |

`days_since_last_activity` is NaN rather than a sentinel on purpose. 3,095
checkpoint rows have no activity ever, and a sentinel such as 999 or 0 would be
indistinguishable from a real value.

## trend (3) — direction of change

Retained in ratio and baseline-relative form only; see the ablation note above.

| Feature | Definition | Univariate lift |
|---|---|---:|
| `clicks_vs_baseline_ratio` | Recent 4 weeks vs the student's own first 4 course weeks, +1 smoothed | 0.102 |
| `clicks_ratio_4w` | Recent 4 weeks vs the preceding 4 weeks, +1 smoothed | 0.070 |
| `clicks_delta_4w` | Recent 4 weeks minus the preceding 4 weeks | 0.049 |

`clicks_vs_baseline_ratio` is the strongest and is exactly the form Phase 3
predicted would work, because it separates a genuine faller from a consistently
low performer. All ratios use +1 smoothing so an empty earlier window yields a
finite value rather than an infinity.

## assessment (11) — submission behaviour and marks

New in Phase 4 and never previously screened. Second most valuable group, which
vindicates the Phase 3 note that the 173,912 submission rows were unexploited.

| Feature | Definition |
|---|---|
| `assessments_due` | Assessments whose due day has passed at the checkpoint |
| `assessments_submitted` | Submissions made at or before the checkpoint |
| `assessments_missed` | `due - submitted`. The OULAD analogue of a backlog. |
| `assessments_late` | Submitted after the due day, both before the checkpoint. Excludes banked. |
| `assessments_banked` | Scores transferred from a previous attempt (administrative, not behavioural) |
| `assessments_failed` | Submissions scoring below 40 |
| `submission_rate` | `submitted / due`. **NaN** when nothing is due yet. |
| `mean_score` / `min_score` | Over submissions to date. NaN before any submission. |
| `mean_submission_lag` | Mean days between due day and submission. Negative means early. |
| `days_since_last_submission` | Checkpoint minus last submission day. NaN if none. |
| `has_submitted` | 0/1 indicator pairing with the NaNs above |

A NULL `assessments.date` (some final exams) is treated as due at the end of the
presentation.

## relative (5) — cohort z-scores

Added by `preprocessing.cohort.CohortZScorer`, **not** by `FeatureBuilder`,
because they are fitted state and must use training-period statistics only
(ADR-0003 control #4). Cohort is keyed on
`(code_module, code_presentation, checkpoint_day)`.

| Feature | Univariate lift |
|---|---:|
| `submission_rate_cohort_z` | **0.193** — strongest single feature in the whole set |
| `active_days_28d_cohort_z` | 0.179 |
| `clicks_28d_cohort_z` | 0.176 |
| `clicks_7d_cohort_z` | 0.176 |
| `mean_score_cohort_z` | 0.154 |

Every one of these outranks its raw counterpart, which is the clearest possible
confirmation of Phase 3 finding 3: the checkpoint positive rate varies 4.2×
across modules, so absolute engagement means different things in different
courses.

Cohorts smaller than 30 rows fall back to global statistics — a standard
deviation from a handful of rows is noise, and dividing by it manufactures
extreme values. Unseen cohorts also fall back, so a deployed model can score a
presentation it was not trained on.

## static (4) — enrolment context

Most valuable group by ablation, which was not anticipated.

| Feature | Definition |
|---|---|
| `checkpoint_day` | Which checkpoint. Positive rate varies 2.5× across them (4.04% at day 30, 1.59% at day 180). |
| `num_of_prev_attempts` | Previous attempts at this module |
| `studied_credits` | Credit load this presentation |
| `date_registration` | Days before course start that the student registered; negative. Plausible commitment signal. |

`checkpoint_day` was added after the group ablation, so it was tested
separately: removing it costs +0.0018 PR-AUC against ±0.0044 fold noise.
Marginal, kept because the direction is positive and the model needs to know
where in the course it is.

These are enrolment facts known at registration, **not** protected attributes.
Gender, age band, deprivation band, disability and region are in the forbidden
list and never reach the matrix — see [ETHICS.md](ETHICS.md).

---

## Guarantees enforced by tests

| Guarantee | Test |
|---|---|
| No feature reads post-checkpoint data | `tests/ml/test_as_of_property.py` — Hypothesis over generated histories plus adversarial fixtures |
| Batch and single-row builds agree | `test_batch_and_single_row_paths_agree` |
| Every allowlisted feature is actually produced | `tests/ml/test_feature_allowlist.py` |
| No forbidden column or protected attribute is modelled | same file |
| No infinities in the matrix | same file |
| An unknown feature group fails loudly | `test_unknown_feature_group_is_rejected` |

The last one exists because of a real bug found in this phase: `FeatureGroups`
originally had no `assessment` or `static` fields, so Pydantic silently
discarded 15 features and the allowlist reported 22 instead of 37. `extra="forbid"`
now makes that impossible.
