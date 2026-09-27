# Deployment

## What is verified, and by whom

This section is first because it is the most important thing on the page.

Docker is **not installed** on the machine this project was built on. The
`Dockerfile`s, `nginx.conf` and `docker-compose.yml` have therefore never been
built or run by their author. They are reviewed code that has not executed.

What verifies them is the `containers` job in `.github/workflows/ci.yml`, which
runs on every push and:

- builds both images;
- asserts the API image runs as the unprivileged `ews` user, not root;
- imports `xgboost` and `shap` **inside** the image, which is the check that
  `libgomp1` survived the two-stage build;
- asserts no compiler is left in the runtime image;
- boots the full stack with real PostgreSQL;
- applies migrations `upgrade → downgrade → upgrade`;
- checks `/health` answers and carries the security headers, and that HSTS is
  *absent* over plain HTTP;
- checks the dashboard serves `index.html` for a client-side route, which is the
  SPA-fallback bug users hit first.

| Component | Verified where |
|---|---|
| Migration round trip (SQLite) | locally, and in CI |
| Migration round trip (PostgreSQL 16) | CI `database` job |
| Frontend production build | locally (255 kB app + 1,097 kB Plotly), and in CI |
| Config validation & security headers | locally, 20 tests in `tests/api/test_deployment_hardening.py` |
| Image builds, non-root, stack boot | **CI only — never run by the author** |
| Scoring a request inside a container | **not verified anywhere** (see below) |

The gap worth naming: the CI smoke test deliberately runs **without a trained
model**. `load_state()` records a load error rather than raising, so the API boots
and `/health` honestly reports `degraded`. That proves the image runs and the
stack wires together. It does **not** prove that a scoring request works in a
container, because `models/` is not committed. Do not read a green CI badge as
more than it is.

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
