# API reference

FastAPI service over the registered model. Machine-readable spec:
[`openapi.json`](openapi.json), or `/docs` when running.

```bash
uvicorn backend.app.main:app --reload
```

---

## Design rules

**Every risk figure carries its provenance.** A probability is only ever
serialised inside a `RiskAssessment`, which requires `model_version`,
`is_calibrated` and `disclaimer`. `services.build_risk_assessment` is the only
function that constructs one. A number that looks like a probability, detached
from what produced it and what it means, is the thing this project most needs not
to ship.

**No personal identifiers exist to leak.** `student_code` is an opaque
SHA-256-derived value (`S-XXXXXXXXXXXX`); the source `id_student` never leaves the
service, and there are no names or contact details in the data at all.

**`/predict` takes a student reference, never a feature vector.** Accepting a raw
vector would let a caller submit one no real student could produce, and the as-of
guarantee ([ADR-0003](adr/0003-leakage-controls-and-splitting.md)) would mean
nothing.

**Limitations travel with the metrics.** `/model/info` returns a `limitations`
array, so a client cannot present performance figures without them.

---

## Roles

| Role | Individual students | Aggregates | Assign interventions | Audit log |
|---|:--:|:--:|:--:|:--:|
| `admin` | yes | yes | yes | yes |
| `counsellor` | yes | yes | yes | no |
| `analyst` | **no** | yes | no | no |

The analyst restriction is the substantive one: cohort analysis must not require
access to an identifiable student's risk ([ETHICS.md](ETHICS.md)). It is enforced
by an enumerated RBAC matrix test, so an endpoint added later has to be added to
the matrix rather than shipping unguarded.

---

## Endpoints

### Auth

| Method | Path | Role | Notes |
|---|---|---|---|
| POST | `/api/v1/auth/token` | — | OAuth2 password form. Unknown users and wrong passwords return an identical 401, so the endpoint does not enumerate usernames. |
| GET | `/api/v1/auth/me` | any | Username, role, and capability flags for the UI to gate on. |

### Students

| Method | Path | Role | Notes |
|---|---|---|---|
| GET | `/api/v1/students` | admin, counsellor | Paginated, sorted by risk descending. Filters: `band`, `direction`, `module`, `search`, `page`, `page_size`. |
| GET | `/api/v1/students/{code}` | admin, counsellor | Profile with risk, trajectory, engagement, assessment, context. **Audited.** |
| GET | `/api/v1/students/{code}/risk-history` | admin, counsellor | Trajectory points, ascending by checkpoint. **Audited.** |
| GET | `/api/v1/students/{code}/explanation` | admin, counsellor | SHAP factors, split actionable / protective / contextual. `?as_of=<day>`. **Audited.** |
| GET | `/api/v1/students/{code}/recommendations` | admin, counsellor | Supportive actions plus a case note. **Audited.** |
| GET | `/api/v1/students/{code}/interventions` | admin, counsellor | Assigned interventions. **Audited.** |
| POST | `/api/v1/students/{code}/interventions` | admin, counsellor | Assign an action. This is the human act ADR-0005 requires; nothing executes without it. |

### Prediction

| Method | Path | Role | Notes |
|---|---|---|---|
| POST | `/api/v1/predict` | admin, counsellor | `{student_code, checkpoint_day?}`. Returns risk plus a `features_hash` tying the prediction to its exact inputs. |

### Dashboard and analytics

| Method | Path | Role | Notes |
|---|---|---|---|
| GET | `/api/v1/dashboard/statistics` | any | Totals, band counts, worsening count, open alerts, `alert_budget`. |
| GET | `/api/v1/analytics/risk-distribution` | any | Histogram; bins sum to the cohort. |
| GET | `/api/v1/analytics/scatter` | any | `?x=<field>`. **Allowlisted** — see the security note below. |
| GET | `/api/v1/analytics/feature-importance` | any | Mean \|SHAP\| with human labels. |

### Alerts, model, admin, system

| Method | Path | Role | Notes |
|---|---|---|---|
| GET | `/api/v1/alerts` | admin, counsellor | Filters: `acknowledged`, `severity`. |
| POST | `/api/v1/alerts/{id}/acknowledge` | admin, counsellor | |
| GET | `/api/v1/model/info` | any | Version, metrics, bands, and `limitations`. |
| GET | `/api/v1/admin/audit-log` | **admin** | Who read which student, and when. |
| GET | `/health` | — | Unauthenticated: a load balancer cannot hold a token. |
| GET | `/metrics` | any | Authenticated: band counts over a cohort are still information about it. |

---

## A security bug the tests caught

`/api/v1/analytics/scatter` originally validated its `x` parameter against
**column presence** rather than an allowlist. The cohort frame carries
`date_unregistration` alongside the features, so:

```
GET /api/v1/analytics/scatter?x=date_unregistration
```

returned **HTTP 200** and served the withdrawal event time — the label itself — to
any authenticated user, including an `analyst` who is otherwise denied all
individual data.

The allowlist function existed; the endpoint simply never consulted it. It now
validates against an intersection of profile fields and modelled features, and
there is a parametrised regression test over nine columns that must never be
servable, plus a second test asserting the allowlist itself contains nothing from
the forbidden-column list in `features.yaml`.

The lesson is narrow and worth keeping: a guard that is written but not wired in
is indistinguishable from no guard, and only a test that attacks the endpoint
tells you which you have.

---

## Health and readiness

`/health` reports `model_loaded` separately from process liveness:

```json
{"status": "ok", "model_loaded": true, "model_version": "xgboost-...", "data_loaded": true, "environment": "local"}
```

Startup is deliberately tolerant of a missing model — the app boots and reports
`degraded` rather than crash-looping, because a container that refuses to start
gives you no way to ask it what went wrong. Requests that need the model then
return 503 with the load error.

## Observability

Every response carries `X-Request-ID`, echoing a client-supplied one when
present. Logs are structured JSON with the request ID bound, so a request can be
traced without grepping.

Reads of an **individual** student are written to the audit log; aggregate
requests are not, because auditing everything buries the entries that matter.
Denied requests are not recorded as successful reads.

## Storage

The current repository reads the processed feature parquet and scores on demand,
behind a `StudentRepository` Protocol that Phase 9 reimplements over PostgreSQL.
The seam means API tests need no database.

On-demand scoring was a measured decision, not a shortcut: explanation latency is
**28.8 ms** for one student and **1.55 ms per row** in batch. My Phase 1 plan
justified persisting explanations on the grounds that request-time SHAP would be
too slow; that justification was wrong. Persisting *predictions* is still needed
in Phase 9, but for a different reason — a risk trajectory is only meaningful if
historical scores are retained rather than recomputed under a newer model.

The cohort served is the **held-out test presentation**, the one the model never
trained on, so the demo shows deployment behaviour rather than reciting training
data.
