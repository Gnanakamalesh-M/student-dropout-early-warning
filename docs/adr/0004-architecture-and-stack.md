# ADR-0004 — System architecture and technology stack

- **Status:** Accepted
- **Date:** 2026-09-25

## Decisions

### Shared feature code between training and serving

`src/dropout_ews/features/builder.py` is imported by **both** the training
pipeline and the FastAPI service. Reimplementing feature logic in the backend
is the most common cause of train/serve skew, and it fails silently: the model
returns plausible-looking probabilities computed from subtly different inputs.

Enforcement: the API stores a `features_hash` with every prediction, and a test
asserts that the offline pipeline and the online service produce byte-identical
feature vectors for the same fixture student.

Consequence: `src/dropout_ews` is an installable package (`pip install -e .`),
not a loose directory reached via `sys.path` manipulation.

### Predictions are precomputed and persisted, not computed on page load

A batch scorer (`scripts/score_cohort.py`) runs per checkpoint, writing to
`predictions`, `prediction_explanations`, `risk_history`, and `alerts`. The
dashboard reads stored rows.

Rationale: SHAP over a tree ensemble for a whole cohort is far too slow for
request-time aggregate queries, and a risk *trajectory* is only meaningful if
historical predictions are retained rather than recomputed. `POST /predict`
remains available for ad-hoc/what-if scoring.

### Frontend: React, not Streamlit

Streamlit would deliver the analytics views for roughly 15% of the effort and
is genuinely adequate for charts. Rejected because:

- it eliminates the resume-relevant React + TypeScript + REST integration work,
  which is a stated project goal; and
- a Streamlit app calling the model in-process makes the FastAPI layer
  decorative, collapsing the client/server separation the architecture exists
  to demonstrate.

Cost mitigation: shadcn/ui for components and Plotly.js for charts, so the work
is layout and data-fetching rather than component or chart internals. If scope
must be cut, the LLM feature (ADR-0005) goes first; the frontend does not.

### Stack

| Layer | Choice | Note |
|---|---|---|
| ML | scikit-learn, XGBoost, imbalanced-learn, SHAP, Optuna | Optuna over GridSearchCV for pruning and sample efficiency |
| Data validation | Pandera (frames) + Pydantic v2 (API) | Pandera over Great Expectations: lighter, no separate project scaffolding |
| Big-data step | DuckDB | Clickstream pre-aggregation (ADR-0002 scale risk) |
| Backend | FastAPI, SQLAlchemy 2.0 async, Alembic | |
| Database | PostgreSQL 16 | |
| Frontend | React 18, TypeScript, Vite, TanStack Query, Plotly.js, Tailwind, shadcn/ui | |
| Tracking | MLflow, local file backend | Optional; no server required |
| Ops | Docker Compose, GitHub Actions, structlog | |

### Python 3.10

The development machine has 3.14 first on PATH, but SHAP and XGBoost wheel
availability on 3.14 is unreliable and building from source on Windows is not a
reasonable project dependency. `requires-python = ">=3.10,<3.13"` is pinned and
the venv is created explicitly with `py -3.10`. Revisit when SHAP publishes
3.13+ wheels.

### Model artifacts are build outputs

`models/` is gitignored. Each artifact ships with `metadata.json` recording
metrics, hyperparameters, training data hash, feature list, and git SHA. The
artifact is baked into the backend Docker image at build time; an object-store
registry is noted as the production path but adds a paid dependency for no
portfolio benefit.

## Consequences

- One feature codebase, one source of truth, testable skew.
- Dashboard reads are fast and trajectories are real history.
- Higher frontend effort than a Streamlit build, accepted deliberately.
