# EDA findings

All numbers reproducible via `python scripts/run_eda.py`; tables in
`reports/eda/`, figures in `reports/figures/`. Computed on the real OULAD
population only (150,938 rows, 3.02% positive).

Every statistic here respects the as-of boundary from
[TASK_SPEC.md](TASK_SPEC.md). An exploratory finding computed from
post-checkpoint data would suggest signal the model can never use, which is
worse than no analysis.

---

## Finding 1 — Disengagement does precede withdrawal, but the signal is weaker than the plan assumed

This is the premise the whole system rests on, so it was tested directly:
weekly clicks aligned to **weeks before the withdrawal event** (not calendar
time), for the 10,063 students with a known event time.

| Weeks before withdrawal | Median weekly clicks |
|---:|---:|
| 9 | 24 |
| 8 | 24 |
| 7 | 24 |
| 6 | 25 |
| 5 | 24 |
| 4 | 23 |
| 3 | 21 |
| 2 | 18 |
| 1 | 16 |
| 0 (final week) | 11 |

**Read this honestly.** Engagement is essentially flat from 9 to 5 weeks out,
then declines gradually, with the sharpest drop inside the final two weeks.
Median activity roughly halves (24 → 11), but most of that movement is late.

Implications, stated plainly:

- There **is** a real pre-event signal, so the premise holds.
- It is **gradual, not dramatic**. Claims that this system catches students
  "long before" they leave are not supported by this data. A realistic framing
  is weeks of warning, not months.
- Because the horizon is 30 days, the model is predicting into roughly the
  window where the decline becomes visible — which is a point in H=30's
  favour, arrived at independently of the ADR-0001 decision rule.

---

## Finding 2 — Simple trend features underperform level features, contradicting the plan's emphasis

The project plan called longitudinal trend features "one of the most important
parts" and "the part that carries the project". Univariate screening across all
checkpoints does not support that for the *simple* versions of those features:

| Feature | AUC | \|lift\| | Mutual info | Coverage |
|---|---:|---:|---:|---:|
| `clicks_7d` | 0.364 | **0.136** | 0.0039 | 1.00 |
| `active_days_30d` | 0.367 | **0.133** | 0.0037 | 1.00 |
| `days_since_last_activity` | 0.629 | **0.129** | 0.0033 | 0.98 |
| `clicks_14d` | 0.372 | 0.128 | 0.0029 | 1.00 |
| `clicks_30d` | 0.380 | 0.121 | 0.0029 | 1.00 |
| `active_days_total` | 0.403 | 0.097 | 0.0013 | 1.00 |
| `clicks_total` | 0.408 | 0.092 | 0.0013 | 1.00 |
| `clicks_prev_30d` | 0.411 | 0.089 | 0.0007 | 1.00 |
| `max_resources_30d` | 0.417 | 0.083 | 0.0035 | 1.00 |
| `clicks_ratio_30d` | 0.459 | 0.041 | 0.0027 | 1.00 |
| `clicks_delta_30d` | 0.475 | **0.025** | 0.0008 | 1.00 |

AUC below 0.5 is the expected direction: more activity means less risk. Lift is
reported as distance from 0.5 so a strong negative association ranks as
informative.

The two explicit trend features rank **last**. Worse, `clicks_delta_30d` is
unstable across checkpoints and changes direction:

| Checkpoint | `clicks_delta_30d` AUC | `days_since_last_activity` AUC | `clicks_7d` AUC |
|---:|---:|---:|---:|
| 30 | 0.418 | 0.618 | 0.360 |
| 60 | 0.437 | 0.649 | 0.362 |
| 90 | **0.509** | 0.630 | 0.359 |
| 120 | 0.389 | 0.652 | 0.341 |
| 150 | 0.410 | 0.672 | 0.326 |
| 180 | **0.512** | 0.670 | 0.340 |

At checkpoints 90 and 180 the crude delta is indistinguishable from noise and
its sign flips. Level and recency features are stable everywhere, and
`days_since_last_activity` is in fact *stronger at later checkpoints*.

**What this does and does not mean.** It does not mean temporal features are
useless — these are deliberately crude single-window deltas. Properly
specified trend features (OLS slope across several windows, decline against a
student's own baseline, consecutive-decline streaks) may well do better, and
the synthetic cohort shows baseline-relative decline carries real signal by
construction.

What it does mean is that **the plan's confident framing was not evidence-based**,
and Phase 4 must treat trend features as a hypothesis to test rather than a
foregone conclusion. Concretely: build level and recency features first,
establish a baseline with them, then add trend features and require them to
demonstrate incremental value before they enter the allowlist.

---

## Finding 3 — Module variation is large, confirming the need for cohort-relative features

| Module | Students | Withdrawal rate | Checkpoint positive rate |
|---|---:|---:|---:|
| CCC | 4,434 | 44.5% | 5.30% |
| DDD | 6,272 | 35.9% | 3.87% |
| FFF | 7,762 | 31.0% | 3.10% |
| BBB | 7,909 | 30.2% | 2.30% |
| EEE | 2,934 | 24.6% | 2.17% |
| AAA | 748 | 16.8% | 1.90% |
| GGG | 2,534 | 11.5% | 1.25% |

A **4.2× spread** in checkpoint positive rate (1.25% to 5.30%). This confirms
the plan's `relative` feature group: an absolute engagement threshold means
different things in different courses, so z-scores within module-presentation
are necessary rather than decorative. It also means module-presentation should
be available to the model as context.

---

## Finding 4 — Zero recent engagement is informative but catches only a minority

Share of students with **no clicks in the trailing 30 days**:

| Checkpoint | Negatives | Positives | Ratio |
|---:|---:|---:|---:|
| 30 | 4.1% | 7.4% | 1.8× |
| 60 | 7.9% | 15.3% | 1.9× |
| 90 | 14.3% | 24.6% | 1.7× |
| 120 | 14.7% | 27.6% | 1.9× |
| 150 | 17.1% | 29.7% | 1.7× |
| 180 | 19.8% | 37.2% | 1.9× |

Positives are consistently ~1.8× more likely to have gone silent — but even at
checkpoint 180, **63% of students who withdraw within 30 days were still
active** in the prior month. A "has the student gone quiet" rule cannot carry
this problem.

Median engagement contrast at checkpoint 180: negatives 71 clicks / 6 active
days / 4 days since last activity; positives 6 clicks / 1 active day / 17 days
since last activity.

---

## Finding 5 — Trivial baselines are weak, so a model has real work to do

Scored before any modelling, so the comparison cannot be skipped later:

| Rule | Flagged | Share | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| No clicks in last 14d | 33,485 | 22.2% | 5.1% | 37.6% | **0.090** |
| No clicks in last 7d | 50,527 | 33.5% | 4.9% | 53.7% | 0.089 |
| No clicks in last 30d | 19,603 | 13.0% | 4.9% | 21.2% | 0.080 |
| Engagement halved vs prev 30d | 37,604 | 24.9% | 4.2% | 34.7% | 0.075 |
| Never active at all | 3,095 | 2.1% | 3.5% | 2.4% | 0.029 |

Against a 3.02% base rate, the best rule reaches only **5.1% precision** —
about 1.7× lift while flagging 22% of the cohort. Operationally that is
unusable: an institution cannot run intensive outreach on a fifth of its
students at those odds.

This sets the bar. **Any model must beat F1 ≈ 0.09 and, more importantly,
deliver materially better precision at a usable alert volume.** It also
tempers expectations: the ceiling on this problem is not high, and a modest
PR-AUC will still be a genuine improvement over what an institution could do
with a spreadsheet rule.

---

## Finding 6 — Subgroup base rates, recorded before modelling

Positive rates with Wilson 95% intervals (used rather than the normal
approximation, which is unreliable at ~3%). These attributes are read for
auditing only and are **excluded from the feature matrix**
([ETHICS.md](ETHICS.md)).

| Attribute | Value | Rows | Rate | 95% CI |
|---|---|---:|---:|---|
| disability | Y | 13,750 | **4.50%** | 4.17–4.86% |
| disability | N | 137,188 | **2.87%** | 2.78–2.96% |
| gender | M | 82,634 | 3.20% | 3.08–3.32% |
| gender | F | 68,304 | 2.81% | 2.68–2.93% |
| imd_band | 0-10% (most deprived) | 14,316 | 3.54% | 3.25–3.86% |
| imd_band | 90-100% (least deprived) | 12,426 | 2.58% | 2.32–2.88% |
| age_band | 0-35 | 104,902 | 3.08% | 2.97–3.18% |
| age_band | 35-55 | 44,962 | 2.91% | 2.76–3.07% |
| age_band | 55<= | 1,074 | 2.42% | 1.66–3.52% |

Three points, and the framing matters:

1. **Students declaring a disability withdraw at 1.57× the rate** of those who
   do not, with non-overlapping intervals. This is the largest disparity in the
   data.
2. **These are base rates, not model errors.** A well-calibrated model *will*
   flag higher-base-rate groups more often, and that is not by itself
   unfairness — it reflects who actually withdraws. The fairness question for
   Phase 12 is whether **error rates** (false negatives especially) differ
   across groups, not whether alert rates do. Recording base rates now is what
   makes that later distinction interpretable.
3. **The 55+ group has only 1,074 rows** and a CI spanning 1.66–3.52%. No
   conclusion is available for it, and the fairness audit must say so rather
   than report a reassuring point estimate.

There is a monotone deprivation gradient (3.54% down to 2.58%), consistent with
engagement features partly measuring circumstance rather than risk — the
concern already flagged in ETHICS.md.

---

## Consequences for Phase 4

1. **Build level and recency features first.** `clicks_7d`,
   `active_days_30d`, and `days_since_last_activity` are the strongest and
   most stable signals.
2. **Treat trend features as a hypothesis.** Implement OLS slopes,
   baseline-relative decline, and decline streaks, but require each to show
   incremental value over the level baseline before entering the allowlist.
   Do not assume the plan's framing was right.
3. **Include cohort-relative z-scores within module-presentation.** The 4.2×
   module spread makes this necessary.
4. **Handle "never active" explicitly.** 3,095 checkpoint rows (2.1%) have no
   activity ever; `days_since_last_activity` is undefined there. Use an
   explicit indicator rather than imputing a sentinel value.
5. **Add assessment-submission features.** Not yet screened — the 173,912
   `studentAssessment` rows carry submission timing and scores, which are
   plausibly complementary to clickstream and remain unexploited.
6. **Expect modest metrics.** With a 3.02% base rate and a best trivial rule at
   5.1% precision, a PR-AUC in the 0.10–0.25 range would be a real result. Any
   number far above that should be treated as a leakage symptom and
   investigated before it is believed.
