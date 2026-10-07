# Backend API image.
#
# Built and run locally (Docker 29.8.2, Compose v5.5.1) as well as by the
# `containers` job in .github/workflows/ci.yml. Doing so found two bugs that both
# produced a healthy-looking container: a declared package missing from the
# builder stage, and project paths resolving inside the virtualenv. See
# docs/DEPLOYMENT.md.
#
# Python 3.10 matches the pin in pyproject.toml (ADR-0004: SHAP/XGBoost wheels
# on 3.13+ are unreliable). The digest is not pinned here because Dependabot
# cannot update a digest it cannot parse; the minor version is pinned, which is
# the tradeoff this project takes elsewhere too.

FROM python:3.10-slim-bookworm AS builder

# Build wheels in a stage that is thrown away, so the runtime image carries no
# compiler. XGBoost and psycopg need one; the final image must not ship it.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Every directory `pyproject.toml` declares as a package must be present, or
# setuptools fails during `prepare_metadata_for_build_wheel` -- not at import
# time. `package-dir` maps both `dropout_ews` -> src/dropout_ews and `backend`
# -> backend, so copying only src/ here failed with "package directory
# 'backend' does not exist", surfaced by pip as the far less helpful
# "Failed to build 'file:///build' when getting requirements to build wheel".
#
# An earlier version of this file copied only pyproject.toml, README.md and src/
# to keep the pip layer cached against source edits. That was wrong rather than
# merely suboptimal: it could not build at all. The cache benefit is narrower
# now -- editing backend/ invalidates the install layer -- and that is the
# correct trade, because a cached layer that never builds is worth nothing.
# README.md is required too: pyproject sets `readme = "README.md"`.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY backend/ ./backend/
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".[api]"


FROM python:3.10-slim-bookworm AS runtime

# libgomp1 is a runtime dependency of XGBoost, not just a build one: without it
# the import fails with a bare "libgomp.so.1: cannot open shared object file",
# which reads like a missing Python package and is not one.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 ews

COPY --from=builder /opt/venv /opt/venv
# DROPOUT_EWS_ROOT is required, not cosmetic. The package is installed
# non-editable into site-packages, so the repo-relative derivation of
# DATA_DIR and MODELS_DIR in config/settings.py resolves inside /opt/venv:
# the API then starts healthy and reports `model_loaded: false` with the
# model mounted and readable the whole time.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DROPOUT_EWS_ROOT=/app

WORKDIR /app
COPY --chown=ews:ews src/ ./src/
COPY --chown=ews:ews backend/ ./backend/
COPY --chown=ews:ews pyproject.toml README.md ./

# Non-root. The API reads student records; a container escape should not also be
# a root shell.
USER ews

EXPOSE 8000

# `/health` reports readiness including whether the configured repository backend
# actually resolved, so this is a real check rather than a liveness ping.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    # 127.0.0.1 for the same reason as the dashboard image: uvicorn --host
    # 0.0.0.0 binds IPv4 only. curl happens to fall back from ::1, so this one
    # passed by luck rather than by design.
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# No --reload, and workers left at 1 by default: the model is held in memory per
# worker, so worker count is a memory decision the operator should make with
# their instance size in front of them, not a default inherited from a tutorial.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
