# Database

PostgreSQL 16. Models in `backend/app/db/models.py`, migrations in
`backend/alembic/`.

```bash
alembic -c backend/alembic.ini upgrade head
python scripts/seed_db.py
python scripts/score_cohort.py
```

---

## ER diagram

```mermaid
erDiagram
    students ||--o| student_demographics : "isolated 1:1"
    students ||--o{ engagement_records : has
    students ||--o{ assessment_records : has
    students ||--o{ predictions : scored_in
    students ||--o{ alerts : raises
    students ||--o{ interventions : receives
    model_versions ||--o{ predictions : produced
    threshold_configs ||--o{ predictions : banded_by
    predictions ||--o{ alerts : triggers
    predictions }|..|| risk_history : "VIEW over"

    students {
        int id PK
        string student_code UK "opaque, exposed by the API"
        string code_module
        string code_presentation
        int source_student_id "OULAD id_student, server-side only"
        string data_source "real_oulad | synthetic"
        int module_presentation_length
    }
    student_demographics {
        int student_id PK_FK
        string gender "fairness audit only"
        string age_band "fairness audit only"
        string imd_band "fairness audit only"
        string disability "fairness audit only"
        string region "fairness audit only"
    }
    engagement_records {
        int id PK
        int student_id FK
        int course_day
        int clicks
        int distinct_resources
    }
    assessment_records {
        int id PK
        int student_id FK
        int assessment_id
        int due_day
        float weight
        int date_submitted "NULL = due but not submitted"
        float score
        bool is_banked
    }
    model_versions {
        int id PK
        string version UK
        string train_data_hash
        string imbalance_strategy
        string calibration_method
        json metrics
        bool is_active
    }
    threshold_configs {
        int id PK
        string name UK
        bool calibrated
        float alert_budget
        json bands
        bool is_active
    }
    predictions {
        int id PK
        int student_id FK
        int model_version_id FK
        int threshold_config_id FK
        int checkpoint_day
        int horizon_days
        float probability
        string band
        string features_hash
        int label "NULL until the horizon closes"
    }
    alerts {
        int id PK
        int student_id FK
        int prediction_id FK
        string reason
        string severity
        float probability
        float previous_probability
        bool acknowledged
        string acknowledged_by
    }
    interventions {
        int id PK
        int student_id FK
        string intervention_key
        string status "recommended|assigned|in_progress|completed"
        string assigned_to
        string assigned_by
    }
```

`users` and `audit_log` have no foreign keys into this graph and are omitted from
the diagram for legibility.

---

## Four decisions worth explaining

### There is no `prediction_explanations` table

The original plan had one, justified on the grounds that request-time SHAP over a
cohort would be too slow. Phase 8 measured it: **28.8 ms** for one student,
**1.55 ms per row** in batch. The justification did not hold, so the table is
dropped rather than carried as complexity nobody needs.

If a future requirement is *audit reproducibility* — "what exactly did we show
this counsellor in March" — that is a different and legitimate reason to store
them, and it would need its own decision rather than inheriting this one.

### `risk_history` is a view, not a table

`predictions` is already unique on `(student_id, checkpoint_day, model_version_id)`,
so every row a history table would hold is already there. Duplicating it would
create two places for the same fact and a way for them to disagree. The view gives
the trajectory query a stable name without that risk.

### Predictions are persisted; explanations are not

These look inconsistent and are not. A trajectory only means something if past
scores are **the ones that were actually produced**, under the model version
recorded against each row — recomputing history under a newer model would
silently rewrite what staff saw. That is a semantics requirement.

Explanations have no such property: an explanation of the current score,
recomputed from the same model and inputs, is the same explanation. So one is
stored and the other is not, for reasons rather than symmetry.

### Demographics are physically separate

`student_demographics` is read **only** by the offline fairness audit. No endpoint
exposes it and no repository query joins it. Data minimisation is structural here
rather than a policy note: the separation is what makes "the model cannot see
this" verifiable by reading the queries, and there is a test asserting no
repository read returns a protected attribute.

---

## Indexes, and why each exists

| Index | Table | Rationale |
|---|---|---|
| `ix_predictions_student_checkpoint` | predictions | The trajectory query, the hottest read path |
| `ix_predictions_upper_bands` (partial) | predictions | The dashboard only ever lists `high`/`critical`, so the index stays small. PostgreSQL only |
| `ix_engagement_student_day` | engagement_records | Feature building reads a trailing window for one student |
| `ix_assessment_student_due` | assessment_records | Same access shape for assessments |
| `ix_alerts_queue` | alerts | The queue is always "open, most severe first" |
| `ix_audit_actor_time`, `ix_audit_resource_time` | audit_log | The two audit questions: what did this person read, who read this student |
| `ix_students_cohort` | students | Cohort filters on the student list |

## Constraints that encode rules

- `uq_student_enrolment` on `(code_module, code_presentation, source_student_id)` —
  `id_student` alone is **not** unique, because students appear in several
  module-presentations (Phase 2 finding).
- `uq_prediction_scope` — re-scoring is an idempotent overwrite, not a second row.
- `uq_alert_prediction_reason` — re-running the batch scorer cannot duplicate an
  alert.
- `ck_prediction_probability` — `0 <= probability <= 1`.
- `ck_intervention_status` — a closed vocabulary, so a new status cannot be
  invented at a call site.
- `ck_students_data_source` — provenance is load-bearing for ADR-0002's honesty
  rule, so it is enumerated rather than free text.

The audit log has **no update or delete path** in the repository. An audit trail
that can be edited is not an audit trail.

---

## Testing, and what it does not cover

No Docker or PostgreSQL was available on the development machine, so the tests run
against **SQLite** by default. SQLAlchemy makes the ORM behaviour identical, so
constraints, cascades, query results and the migration round trip are genuinely
exercised. What SQLite does **not** verify:

- JSONB behaviour and operators (SQLite gets plain `JSON` via `with_variant`);
- the partial index predicate (SQLite ignores `postgresql_where`);
- server-side default and timezone semantics;
- PostgreSQL's stricter type coercion — SQLite is permissive, so a type error
  could pass locally and fail in production.

Set `TEST_DATABASE_URL` to a PostgreSQL DSN to rerun everything against the real
engine, which is what the CI service container does:

```bash
docker run -d --name ews-pg -e POSTGRES_USER=ews -e POSTGRES_PASSWORD=ews   -e POSTGRES_DB=ews_test -p 5432:5432 postgres:16
TEST_DATABASE_URL=postgresql+psycopg://ews:ews@localhost:5432/ews_test   pytest tests/integration -m integration
```

Measured counts, so the numbers are attributable: `tests/integration` holds **65**
tests — `test_database.py` 41, `test_api_over_database.py` 12,
`test_oulad_real_data.py` 12. On SQLite, **64 pass and 1 skips**, and that skip
names itself rather than passing quietly:

```
SKIPPED tests/integration/test_database.py:681: running against SQLite;
set TEST_DATABASE_URL to a PostgreSQL DSN to verify JSONB, partial indexes
and server-side defaults
```

Measured locally against `postgres:16-alpine`: **65 passed, 0 skipped** — the one
SQLite skip disappears, so JSONB behaviour, the partial-index predicate,
server-side defaults and PostgreSQL's stricter type coercion are all actually
checked. The migration round trip applies cleanly too, and PostgreSQL reports
transactional DDL where SQLite reports non-transactional, which means the
downgrade path is exercised rather than assumed. CI's `database` job runs the
same thing.

### A SQLite gap that mattered

SQLite ships with **foreign keys disabled**. Until `PRAGMA foreign_keys=ON` was
set, every FK constraint and every `ON DELETE CASCADE` was silently inert — so the
referential-integrity tests passed while testing nothing. A cascade test caught
it. The pragma is now set for every SQLite connection and asserted by its own
test, because if it regresses those tests go quiet rather than failing.

---

## Two bugs the seam test found

`tests/integration/test_api_over_database.py` runs the API against the database
with no router or service changes — the check on whether the Phase 8 repository
seam was in the right place. It found two real defects:

**The API never committed.** The first version held one session open for the
process lifetime. Assigning an intervention returned HTTP 201 and the row was
silently lost. The repository is now built **per request**, committing on success
and rolling back on failure — which is also the only correct choice under
concurrency.

**`SqlRepository.feature_row` was never implemented.** It is declared in the
`StudentRepository` Protocol, but a Protocol with no runtime or static check is
documentation. It now rebuilds features from the stored event tables using the
**same** `FeatureBuilder` as training, which is the ADR-0004 claim made concrete:
one feature implementation, working off whatever holds the events.

One honest limitation of that path: enrolment context (`num_of_prev_attempts`,
`studied_credits`, `date_registration`) is not stored as events, so the values the
model saw at scoring time are not reconstructable from the event tables. They come
back as `NaN`, the imputer handles them, and the missingness indicator records it —
rather than inventing a plausible number.

---

## Seeding scope

`scripts/seed_db.py` defaults to the **test split**, not the test presentation.
Those differ: the split drops 1,588 students from 2014J because they also appear
in an earlier presentation (ADR-0003 amendment). Seeding by presentation alone put
students the model trained on into the demo cohort and quietly undermined the claim
that it shows deployment behaviour. An early run did exactly that — 9,116 students
instead of 7,848.

Seeded scope: 7,848 enrolments, 542,767 engagement rows, 44,877 assessment rows,
43,407 predictions, 3,027 alerts.

Note "students" here means **enrolments**: 7,848 student-module rows covering 7,528
distinct people, because a student can take two modules in the same presentation.
The natural key is the enrolment, which is why the unique constraint spans all
three columns.

`--all-cohorts` loads every presentation and reports the time rather than
pretending it is free.

## Migrations

```bash
alembic -c backend/alembic.ini upgrade head     # apply
alembic -c backend/alembic.ini downgrade base   # reverse
alembic -c backend/alembic.ini revision --autogenerate -m "..."
```

The URL comes from `DATABASE_URL` via `env.py`, never from `alembic.ini`, so a
connection string is not committed. The upgrade/downgrade round trip is tested: 11
tables plus the view up, everything down.

`sync_database_url` normalises an `asyncpg` URL to `psycopg`. `.env` carries the
async form for forward compatibility, but this layer is synchronous — the API is
bound on model inference, which is CPU work that blocks the event loop regardless
of how the query is issued, so async would add colour to every signature for no
measured gain.
