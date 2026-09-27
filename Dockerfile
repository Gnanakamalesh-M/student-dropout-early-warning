# Backend API image.
#
# NOT BUILT OR RUN BY ITS AUTHOR. Docker is unavailable on the machine this
# project was developed on, so this file is verified by the `containers` job in
# .github/workflows/ci.yml — which builds it and boots the stack — and nowhere
# else. Treat it as reviewed-but-unexercised until that job has passed.
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
# Copy only what defines the dependency set first, so the expensive layer is
# cached against source edits rather than rebuilt on every commit.
COPY pyproject.toml README.md ./
COPY src/ ./src/
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
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

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
    CMD curl -fsS http://localhost:8000/health || exit 1

# No --reload, and workers left at 1 by default: the model is held in memory per
# worker, so worker count is a memory decision the operator should make with
# their instance size in front of them, not a default inherited from a tutorial.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
