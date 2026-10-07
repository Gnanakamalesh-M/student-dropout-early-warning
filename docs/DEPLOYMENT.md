# Deployment

## What is verified, and how

Docker Desktop is now installed on the development machine, so the container
work has been **built and run locally**, end to end. This section previously said
the opposite; it was accurate when written and is superseded.

Versions used: Docker 29.8.2, Compose v5.5.1, linux/x86_64 engine.

| Component | Verified |
|---|---|
| API image builds | **locally** (`docker compose build api`) |
| Dashboard image builds | **locally** — built first try, no changes needed |
| Runs as unprivileged `ews`, not root | **locally** (`id -un` → `ews`) |
| No compiler in the runtime image | **locally** (`command -v gcc` absent) |
| `xgboost` + `shap` import inside the image | **locally** (xgboost 3.2.0, shap 0.49.1) |
| Full stack boots, all three healthy | **locally** (`api`, `dashboard`, `db` all `healthy`) |
| `/health` reports `status: ok`, `model_loaded: true` | **locally** |
| Security headers present, HSTS absent over HTTP | **locally** |
| **A real scoring request inside a container** | **locally** — see below |
| Clean `--no-cache` rebuild of both images | **locally** |
| Dashboard SPA fallback (`/students/ABC-123` → 200) | **locally** |
| Migration round trip (SQLite) | locally |
| Migration round trip (PostgreSQL 16) | **locally** (`postgres:16-alpine`) and CI `database` job |
| Full integration suite on PostgreSQL 16 | **locally — 65 passed, 0 skipped** |
| Frontend production build | locally (255 kB app + 1,097 kB Plotly) and CI |
| Config validation & security headers | locally, 20 tests in `tests/api/test_deployment_hardening.py` |

### The gap that is now closed

The CI `containers` job runs **without a trained model**, because `models/` is not
committed: the API boots and `/health` honestly reports `degraded`. That proved
the image ran and the stack wired together, but never that the system could
actually *score* in a container. `FUTURE_WORK.md` listed closing that as item 7.

It is closed. With `models/` and `data/` mounted, the containerised API returns a
full SHAP-explained prediction — probability 0.3333, `critical` band, 6 risk
factors, 3 protective, 3 contextual, `is_calibrated: true`, model version and
disclaimer attached. The cohort figures served from the container match the
non-container run exactly: 7,848 students, 43,407 scored checkpoints, identical
band counts.

CI remains the model-free smoke test. It is still useful — it catches image and
wiring regressions on a machine with no artifacts — but it is no longer the only
verification.

Docker also unblocked the one test that had always skipped. Against a throwaway
`postgres:16-alpine`, `pytest tests/integration -m integration` gives **65
passed, 0 skipped**, so JSONB behaviour, the partial-index predicate, server-side
defaults and PostgreSQL's stricter type coercion are now genuinely checked rather
than deferred to CI. The migration round trip also reports *transactional* DDL on
PostgreSQL where SQLite reports non-transactional, which means the downgrade path
is exercised rather than assumed.

Both images were finally rebuilt with `--no-cache` from a torn-down state, so
what is committed builds from scratch rather than from a warm layer cache.

---

## Four bugs that only a real build could find

Every one of these produced a *healthy-looking* failure. That is the argument for
having built it rather than reviewed it.

### 1. The builder stage omitted a declared package

```
error: package directory 'backend' does not exist
ERROR: Failed to build 'file:///build' when getting requirements to build wheel
```

`pyproject.toml` sets `package-dir = { "dropout_ews" = "src/dropout_ews",
"backend" = "backend" }`. The builder copied only `pyproject.toml`, `README.md`
and `src/` — an optimisation so the pip layer cached against source edits. Every
directory `pyproject.toml` declares must be present during
`prepare_metadata_for_build_wheel`, so setuptools failed, and pip reported it as
the far less informative "Failed to build … when getting requirements to build
wheel".

The optimisation was not merely suboptimal, it could not build at all. `backend/`
is now copied too; editing it invalidates the install layer, which is the correct
trade, because a cached layer that never builds is worth nothing.

### 2. Paths resolved inside the virtualenv, silently

The worst of the four. `config/settings.py` derived the project root from the
module location:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]
```

Correct for a source checkout and an editable install. In the image,
`pip install ".[api]"` copies the package into site-packages, so the same four
levels up land on `/opt/venv/lib/python3.10` — and `MODELS_DIR` became
`/opt/venv/lib/python3.10/models`.

The result was not a crash. The container started, passed its healthcheck,
answered `/health` with HTTP 200, and reported `model_loaded: false` while the
model sat mounted and readable at `/app/models`. Nothing was logged, because
`load_state()` captures the `FileNotFoundError` into `load_error` and no code
path prints it.

`DROPOUT_EWS_ROOT` now makes the root explicit; the image sets it to `/app`.
Deliberately an environment variable rather than a cwd fallback or a walk-up
looking for `pyproject.toml` — both guess, and a guessed path fails the same
silent way. Three tests in `tests/unit/test_config.py` cover it, including that
`CONFIG_DIR` must keep following the *module*, since `features.yaml` is package
data that ships inside the wheel.

### 3. Only `data/processed` was mounted

With the paths fixed, `/health` still reported `data_loaded: false`. The
`parquet` backend reads student and module metadata straight from the raw OULAD
CSVs, and only `./data/processed` was mounted — so
`data/raw/oulad/studentRegistration.csv` was missing, and the error was swallowed
the same way.

Compose now mounts `./data:/app/data:ro`. Still a read-only mount rather than an
image layer: the OULAD licence is the dataset's, so it must not travel inside a
distributable image.

### 4. The dashboard healthcheck used an IPv6-resolving name

The dashboard reported `unhealthy` for ten minutes while serving **HTTP 200 to
the host**. The healthcheck was:

```
wget -q --spider http://localhost:8080/
```

The container's `/etc/hosts` maps `localhost` to both `127.0.0.1` and `::1`,
busybox `wget` tries `::1` first, and `listen 8080` binds IPv4 only. Measured
inside the container: `localhost` → connection refused, `127.0.0.1` → OK.

Both healthchecks now address `127.0.0.1` explicitly. The alternative fix is
`listen [::]:8080` in `nginx.conf`; addressing the loopback is narrower than
changing what the server binds. The API's check had the same latent flaw and
passed only because `curl` falls back from `::1` where busybox `wget` does not.

### What this says about the CI job

CI would have caught #1 (the build fails outright). It would **not** reliably
have caught #2 or #3 — it runs with no model and no data and *expects*
`degraded`, so a path bug is indistinguishable from the intended state. And it
would have missed #4 entirely, because it curls the dashboard from the host,
where IPv4 resolution succeeds. Three of the four needed a real run with real
artifacts.

---

## Running the stack

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste as SECRET_KEY
docker compose up --build
docker compose exec api alembic -c backend/alembic.ini upgrade head
docker compose exec api python scripts/seed_db.py
docker compose exec api python scripts/score_cohort.py
```

Dashboard on `:8080`, API on `:8000`.

A trained model must exist in `models/` first — `make data && make features &&
make model` — because it is mounted, not baked in.

---

## Decisions worth explaining

### The model and the data are mounted, not baked into the image

A model baked into a layer cannot be rotated without rebuilding and
redeploying, which turns "roll back to last week's model" into a release. And
`data/raw` holds the OULAD extract, whose redistribution terms belong to the
dataset rather than to this project; copying it into a distributable image would
assume a right this repository does not have. Both are read-only bind mounts.

`.dockerignore` excludes `data/` and `models/` so a stray `COPY . .` cannot
reintroduce either by accident.

### Two-stage build, and the dependency that is easy to lose

XGBoost and `psycopg` need a compiler to install. The builder stage has
`build-essential`; the runtime stage does not, and CI asserts `gcc` is absent.

`libgomp1` is the trap. It is a **runtime** dependency of XGBoost, not only a
build one, and dropping it from the final stage produces
`libgomp.so.1: cannot open shared object file` on import — which reads like a
missing Python package and is not one. CI imports `xgboost` inside the image
specifically to catch this.

### Workers default to 1

The model is held in memory per worker, so worker count is a memory decision that
depends on instance size. A default of 4 inherited from a tutorial would quietly
quadruple the memory footprint of an image whose whole point is that it holds a
model. Set `--workers` deliberately.

### PostgreSQL is not published to the host

`expose`, not `ports`. Nothing outside the compose network needs the database, and
an exposed Postgres with a default password is how a demo deployment becomes an
incident. `POSTGRES_PASSWORD` is overridable and the default is only for local
use.

### The database URL says `asyncpg` but the engine is synchronous

Deliberate, and documented in `backend/app/db/session.py`:
`sync_database_url()` rewrites `+asyncpg` to `+psycopg`. The URL is written for
forward compatibility with an async engine; the rewrite means one environment
variable serves both. It looks like a mistake, which is why it is named here.

---

## Configuration that only matters once deployed

Two settings existed as hardcoded values until this phase, and both fail in ways
that are hard to diagnose.

### CORS origins

`cors_origins` was hardcoded to the Vite dev server. In a real deployment the
browser blocks the response **after** the request succeeded, so the API log shows
a healthy `200` and the dashboard shows a network error with nothing pointing at
the cause.

It is now configuration, and validated at startup. Outside `local`, the app
refuses to boot if the list is empty, still contains `localhost`/`127.0.0.1`, uses
plain `http://`, or contains `*`. The wildcard case is rejected even in
development: browsers refuse wildcard-with-credentials at runtime, so failing at
boot converts a confusing outage into a clear one.

### HSTS is opt-in

`force_https` defaults to `false`. Sending HSTS over plain HTTP is wrong, and on
`localhost` it pins the developer's browser to HTTPS for the max-age — a
persistent, self-inflicted outage that survives restarts and is genuinely
confusing to diagnose. Turn it on only behind real TLS.

### Security headers

Set by `SecurityHeadersMiddleware` on every response:

| Header | Why, for this API specifically |
|---|---|
| `X-Content-Type-Options: nosniff` | stops a JSON body being guessed as HTML and executed |
| `X-Frame-Options: DENY` | framing the dashboard is the setup for clickjacking a counsellor into assigning an intervention |
| `Referrer-Policy: no-referrer` | student codes appear in URLs; the browser default would leak them to third-party resources |
| `Cache-Control: no-store` | a per-student risk figure in a shared machine's cache is the disclosure `ETHICS.md` is about |

They use `setdefault`, so an endpoint can still set its own policy. A test
asserts that, because changing it to assignment would silently break any such
endpoint.

### Secrets

`SECRET_KEY` must be ≥32 characters and must not be the development default
outside `local`; startup fails otherwise. `ENABLE_LLM_NARRATIVE=true` without
`ANTHROPIC_API_KEY` also fails at startup rather than on the first case-note
request.

---

## Operational notes

### Readiness vs liveness

`/health` reports them separately and is unauthenticated, because a load balancer
cannot hold a token. `status: degraded` with `model_loaded: false` means the
process is alive but cannot score — the correct signal for "do not send traffic
yet" rather than "restart me".

### Logs are structured

`structlog` JSON, with a request ID on every line and echoed back in
`X-Request-ID`. Individual-record reads are audited separately; see
`docs/ETHICS.md` for what is recorded and why the in-memory audit list is a
memory guard rather than a retention policy.

### What is deliberately not here

- **No orchestration manifests.** Kubernetes, Helm, or a cloud-specific service
  definition would be unverifiable here *and* unverifiable in CI, which puts them
  in a different category from the compose file. Writing them would be
  presenting guesswork as deployment.
- **No TLS termination.** That belongs to whatever sits in front — a load
  balancer or ingress. `force_https` exists to cooperate with it.
- **No rate limiting configured.** `slowapi` is a dependency and the hooks exist,
  but no limits are set, because sensible limits depend on cohort size and staff
  count. An unset limit is more honest than an arbitrary one.
- **No secret manager integration.** Secrets arrive as environment variables.
  Which manager supplies them is a site decision.
- **No backup or retention policy.** This holds student records; retention is a
  legal and institutional question, not a technical default for a portfolio
  project to invent. `docs/ETHICS.md` says the same about the audit log.
