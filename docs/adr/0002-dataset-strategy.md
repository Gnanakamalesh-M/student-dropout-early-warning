# ADR-0002 — Dataset strategy: OULAD primary, synthetic supplementary

- **Status:** Accepted. Source **verified** 2026-09-25 — see
  [Verification outcome](#verification-outcome-2026-09-25).
- **Date:** 2026-09-25

## Context

The system specification calls for longitudinal academic, attendance, and
engagement data with dropout labels. No single public dataset provides all
three of:

- **real dropout labels**,
- **true longitudinal structure** (repeated observations with timestamps), and
- **the academic schema described** (semester GPA, backlogs, subject-wise
  attendance).

Candidates assessed:

| Dataset | Labels | Longitudinal | Academic schema | License |
|---|---|---|---|---|
| **OULAD** (Open University Learning Analytics Dataset) | Yes — `final_result` incl. `Withdrawn`, plus `date_unregistration` | **Yes** — daily VLE clickstream, timestamped assessment submissions | Partial — assessment scores, no GPA/backlogs/attendance | CC-BY 4.0 |
| **UCI "Predict Students' Dropout and Academic Success"** (Realinho et al., Polytechnic Institute of Portalegre) | Yes — 3-class | **No** — one row per student, cross-sectional | Good — curricular units approved/failed by semester | CC-BY 4.0 |
| xAPI-Edu-Data (Kalboard 360) | Proxy only (grade band) | No | Weak | Open |
| KDD Cup 2010 / EdNet | No dropout label | Yes | No | Varies |

## Decision

A two-track strategy with explicit provenance labelling.

**Track A — OULAD as the primary dataset.** All headline metrics reported in
the README and model card come from OULAD alone. It is the only candidate that
supports ADR-0001's point-in-time framing with real labels: the clickstream
gives daily engagement, `date_unregistration` gives a timestamped event, and
the seven course presentations are ordered in time, which enables a genuine
forward-in-time test split (ADR-0003).

**Track B — a documented synthetic longitudinal generator.** For the academic
dimensions OULAD lacks (semester GPA, backlog counts, subject-wise attendance),
we generate a synthetic cohort from an explicit, published causal DAG:

```
latent engagement propensity ──┬──> attendance ──┬──> assessment marks ──> backlogs ──┐
                               └──> study effort ─┘                                    │
                     prior attainment ────────────────────────────────────────────────>├──> re-enrolment decision
                     external shock (financial / health) ────────────────────────────> ┘
```

with configurable noise, realistic missingness, and a tunable dropout base
rate. Seeded for reproducibility.

**Track C — UCI, demoted.** Kept as a single exploratory notebook only
(`notebooks/07_uci_static_benchmark.ipynb`), to sanity-check that a static
model on an independent real dataset behaves sensibly. It shares no feature
space with OULAD, so it cannot support a transfer or generalisation claim, and
none will be made. It is **not** part of the pipeline, the API, or the
reported results.

## The honesty rule

This is a hard requirement, enforced in code and not only in prose:

1. Every persisted record carries `data_source ∈ {real_oulad, synthetic}`.
2. README, model card, and any dashboard view showing synthetic-derived figures
   display a provenance banner.
3. Synthetic metrics are reported in a **separate table** from real metrics,
   under this exact caveat:

   > Metrics computed on synthetic data measure pipeline correctness, not
   > real-world predictive performance. They must not be read as evidence that
   > the model would perform at this level on real students.

4. The generator's DAG and parameters are published, so the circularity is
   visible rather than hidden. A generator whose data-generating process
   encodes the same relationships the features are designed to detect will
   yield flattering, near-meaningless metrics. Stating this openly is the only
   defensible way to use synthetic data here.
5. Derived/engineered features are documented as such in the feature
   dictionary, distinct from raw observed columns.

## Consequences

- The project has real longitudinal signal, which is what makes the temporal
  feature engineering and leakage controls meaningful rather than theatrical.
- The dashboard can demonstrate the full GPA/backlog/attendance UI without
  fabricating those fields onto real students — the two cohorts stay separate.
- Cost: two loaders, two schemas, and provenance plumbing throughout the stack.
- Risk: OULAD's clickstream is tens of millions of rows. Pre-aggregation to
  student-week via DuckDB/PyArrow is required; naive per-checkpoint pandas
  groupby will not scale. Tracked as a Phase 2 task.

## Verification outcome (2026-09-25)

Row counts, column names, and license terms have been verified against the
actual extraction. Full detail in `docs/DATA_CARD.md`; the figures are asserted
by `tests/integration/test_oulad_real_data.py`.

**Confirmed:** 32,593 students; 10,655,280 clickstream rows; CC-BY 4.0; all
seven CSVs present with the expected column names; `date_unregistration` gives
a usable timestamped event for 10,063 withdrawals; presentation codes order
chronologically as assumed.

**Corrections to this ADR:**

1. **"~32,593 students / 7 course presentations" was wrong on the second
   count.** OULAD has **7 modules** across **4 presentation codes**, giving
   **22 module-presentations**. The forward-in-time split orders on the 4
   presentation codes. ADR-0003's ordering (`2013B < 2013J < 2014B < 2014J`)
   was already correct.

2. **The canonical download is dead.** The OU's own
   `anonymisedData.zip` link returns HTTP 404 and the older
   `analyse.kmi.open.ac.uk` endpoints redirect to a marketing page. The
   mirror-list approach in `scripts/download_data.py` was added in response;
   the dataset was obtained from the UCI mirror. This was an unanticipated
   single-point-of-failure risk in the original plan.

3. **The scale risk was real but is resolved.** DuckDB aggregation to
   student-day granularity takes the clickstream from 10,655,280 rows /
   453.8 MB CSV to 1,808,119 rows / 4.5 MB Parquet in about 3 seconds, out of
   core. No pandas step ever materialises the full clickstream.

**Newly discovered, not in the original assessment:**

4. **Missing values are the literal string `"?"`.** Read without
   `na_values=["?"]`, `date_unregistration` parses as `object` and all label
   logic silently inverts. This is the single most dangerous property of the
   dataset and is now covered by an integration test.

5. **6,519 of 32,593 students have no clickstream at all**, and **5,127
   withdraw at or before the first checkpoint** (3,089 before day 0). Neither
   population is modellable from in-course behaviour. Both are documented as
   out of scope rather than imputed or silently dropped.

6. **Two label-source disagreements** (93 withdrawn with no event time; 9 with
   an event time but `final_result = "Fail"`) required explicit, documented
   handling rules.
