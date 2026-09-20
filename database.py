"""
database.py – SQLAlchemy engine, session factory, and declarative base.

All models must inherit from :data:`Base`.  The :func:`get_db` generator is
injected into FastAPI route handlers via ``Depends``.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import get_settings

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────

connect_args: dict = {}
if settings.database_url.startswith("sqlite"):
    # Required for SQLite to allow multi-threaded access inside FastAPI
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug,
    pool_pre_ping=True,
)

# Enable WAL mode for SQLite – better concurrent read performance
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

# ── Session Factory ───────────────────────────────────────────────────────────

SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ── Declarative Base ──────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Shared SQLAlchemy declarative base for all ORM models."""


# ── Dependency ────────────────────────────────────────────────────────────────


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a transactional DB session.

    Ensures the session is always closed, even on exceptions.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_context() -> Generator[Session, None, None]:
    """Context-manager variant of :func:`get_db` for use outside FastAPI.

    Example::

        with db_context() as db:
            db.add(some_model)
            db.commit()
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_tables() -> None:
    """Create all database tables that have not yet been created."""
    Base.metadata.create_all(bind=engine)
