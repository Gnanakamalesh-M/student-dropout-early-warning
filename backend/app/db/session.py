"""Engine and session management.

Sync SQLAlchemy rather than async. The API is not I/O-bound on the database --
it is bound on model inference, which is CPU work that blocks the event loop
regardless of how the query is issued. Async here would add colour to every
function signature for no measured gain, and FastAPI runs sync endpoints in a
threadpool anyway. If profiling later shows query wait dominating, this is a
contained change.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from dropout_ews.config.settings import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def sync_database_url(url: str | None = None) -> str:
    """Normalise a configured URL to a sync driver.

    ``.env`` carries an asyncpg URL for forward compatibility, but this module is
    sync, so the driver suffix is stripped rather than failing confusingly at
    connect time.
    """
    resolved = url or get_settings().database_url
    return resolved.replace("+asyncpg", "+psycopg").replace("+aiosqlite", "")


def enforce_sqlite_foreign_keys(engine: Engine) -> None:
    """Turn on foreign-key enforcement for SQLite connections.

    SQLite ships with foreign keys **disabled**, so without this every FK
    constraint and every ON DELETE CASCADE is silently inert. That is not a
    cosmetic difference when SQLite is standing in for PostgreSQL in tests: it
    makes every cascade and referential-integrity test vacuous while appearing to
    pass. A cascade test caught exactly that.

    PostgreSQL enforces these unconditionally, so this only levels SQLite up.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    global _engine
    if _engine is None or url is not None:
        resolved = sync_database_url(url)
        connect_args = {}
        if resolved.startswith("sqlite"):
            # SQLite in tests: allow cross-thread use, since FastAPI's
            # threadpool runs sync endpoints off the main thread.
            connect_args["check_same_thread"] = False
        engine = create_engine(resolved, echo=echo, pool_pre_ping=True, connect_args=connect_args)
        enforce_sqlite_foreign_keys(engine)
        if url is not None:
            return engine
        _engine = engine
    return _engine


def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None or url is not None:
        factory = sessionmaker(bind=get_engine(url), expire_on_commit=False)
        if url is not None:
            return factory
        _session_factory = factory
    return _session_factory


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = get_session_factory(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session


def reset_engine() -> None:
    """Drop cached engine and factory. Used by tests between databases."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
