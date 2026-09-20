"""
tests/conftest.py – Shared pytest fixtures for all test modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, get_db
from app import app

# ── In-memory SQLite ──────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSession  = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def create_test_tables():
    import models.student    # noqa: F401
    import models.attendance # noqa: F401
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    connection  = test_engine.connect()
    transaction = connection.begin()
    session     = TestSession(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def test_client(db_session):
    def _override():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = _override
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    app.dependency_overrides.clear()


# ── Reusable fake encodings ───────────────────────────────────────────────────

@pytest.fixture
def fake_encoding_1404():
    """A normalised 1404-D encoding (MediaPipe format)."""
    v = np.random.default_rng(42).random(1404).astype(np.float64)
    return v / np.linalg.norm(v)


@pytest.fixture
def different_encoding_1404():
    """A different normalised 1404-D encoding."""
    v = np.random.default_rng(99).random(1404).astype(np.float64)
    return v / np.linalg.norm(v)


# ── Mock camera ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_camera():
    cam = MagicMock()
    cam.open.return_value = True
    cam.is_open.return_value = True
    cam.check_connection.return_value = True
    # Return a blank 480x640 BGR frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cam.read_frame.return_value = (True, blank)
    cam.read_frame_with_retry.return_value = blank
    return cam
