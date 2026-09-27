"""Request IDs, structured logging, and the audit log.

The audit log is a requirement rather than an operational nicety. ETHICS.md
commits to being able to answer "who looked at this student's risk profile, and
when", and a system that cannot answer that should not hold individual risk
scores at all.

Audit entries are written for reads of an **individual** record only. Logging
every aggregate request would bury the entries that matter in noise, and an audit
trail nobody can read is not an audit trail.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger("dropout_ews.api")

REQUEST_ID_HEADER = "X-Request-ID"

# Paths whose access is auditable: anything addressing one student.
_INDIVIDUAL_PATH = re.compile(r"/api/v1/students/(?P<code>[^/]+)")


@dataclass
class AuditEntry:
    timestamp: datetime
    request_id: str
    actor: str
    action: str
    resource: str
    status_code: int


@dataclass
class AuditLog:
    """In-memory audit sink.

    Phase 9 replaces this with the ``audit_log`` table. The interface is
    deliberately narrow so that swap touches nothing else.
    """

    entries: list[AuditEntry] = field(default_factory=list)
    max_entries: int = 10_000

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            # Drop the oldest. A production sink appends durably; bounding an
            # in-memory list is a memory guard, not a retention policy.
            del self.entries[: len(self.entries) - self.max_entries]

    def for_student(self, code: str) -> list[AuditEntry]:
        return [entry for entry in self.entries if entry.resource.endswith(code)]

    def for_actor(self, actor: str) -> list[AuditEntry]:
        return [entry for entry in self.entries if entry.actor == actor]


audit_log = AuditLog()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID, log the request, and audit individual-record reads."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed", method=request.method, path=request.url.path)
            structlog.contextvars.clear_contextvars()
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id

        actor = getattr(request.state, "actor", "anonymous")
        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
            actor=actor,
        )

        match = _INDIVIDUAL_PATH.match(request.url.path)
        if match and response.status_code < 400:
            audit_log.record(
                AuditEntry(
                    timestamp=datetime.now(timezone.utc),
                    request_id=request_id,
                    actor=actor,
                    action=f"{request.method} {request.url.path}",
                    resource=f"student:{match.group('code')}",
                    status_code=response.status_code,
                )
            )

        structlog.contextvars.clear_contextvars()
        return response


def configure_logging(level: str = "INFO") -> None:
    """Structured JSON logs, so request IDs are queryable rather than grepped."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}.get(level.upper(), 20)
        ),
        cache_logger_on_first_use=True,
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Response headers that limit what a browser will do with our responses.

    This API returns individual students' risk figures, so the headers worth
    setting are the ones that stop those responses being reinterpreted or
    embedded somewhere they were not intended:

    - ``X-Content-Type-Options: nosniff`` stops a browser guessing that a JSON
      body is HTML and executing it.
    - ``X-Frame-Options: DENY`` stops the dashboard being framed by another site,
      which is the setup for clickjacking a counsellor into assigning an
      intervention.
    - ``Referrer-Policy: no-referrer`` matters more than usual here because
      student codes appear in URLs; the default policy would leak them to any
      third-party resource the page loads.
    - ``Cache-Control: no-store`` on API responses, because a per-student risk
      figure cached by a shared-machine browser is exactly the disclosure this
      project's data-minimisation section is about.

    HSTS is opt-in via ``force_https``. Sending it over plain HTTP is wrong, and
    on a developer machine it pins localhost to HTTPS in the browser for the
    max-age — a persistent, confusing, self-inflicted outage.
    """

    def __init__(self, app: object, force_https: bool = False) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.force_https = force_https

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        if self.force_https:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response
