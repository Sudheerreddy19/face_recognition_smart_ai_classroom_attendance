"""
tests/test_springboot.py – Tests for SpringBootService (Phase 8).

Tests cover:
  - health_check returns True / False correctly
  - sync_attendance returns True on 2xx, False on error
  - verify_student returns dict on success, None on 404
  - Connection errors are handled gracefully (no exceptions raised)
"""

from __future__ import annotations

from datetime import date, time
from unittest.mock import MagicMock, patch

import pytest
import requests

from services.springboot_service import SpringBootService


def _make_svc():
    return SpringBootService(base_url="http://localhost:8080", timeout_s=5)


class TestHealthCheck:

    def test_returns_true_on_200(self):
        svc = _make_svc()
        mock_resp = MagicMock()
        mock_resp.ok = True
        with patch.object(svc._session, "get", return_value=mock_resp):
            assert svc.health_check() is True

    def test_returns_false_on_500(self):
        svc = _make_svc()
        mock_resp = MagicMock()
        mock_resp.ok = False
        with patch.object(svc._session, "get", return_value=mock_resp):
            assert svc.health_check() is False

    def test_returns_false_on_connection_error(self):
        svc = _make_svc()
        with patch.object(svc._session, "get", side_effect=requests.ConnectionError):
            assert svc.health_check() is False


class TestVerifyStudent:

    def test_returns_dict_on_success(self):
        svc = _make_svc()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"registerNumber": "CSE001", "name": "Sudheer"}
        with patch.object(svc._session, "get", return_value=mock_resp):
            result = svc.verify_student("CSE001")
        assert result is not None
        assert result["registerNumber"] == "CSE001"

    def test_returns_none_on_404(self):
        svc = _make_svc()
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        with patch.object(svc._session, "get", return_value=mock_resp):
            result = svc.verify_student("NOBODY")
        assert result is None

    def test_returns_none_on_connection_error(self):
        svc = _make_svc()
        with patch.object(svc._session, "get", side_effect=requests.ConnectionError):
            result = svc.verify_student("CSE001")
        assert result is None


class TestSyncAttendance:

    def test_returns_true_on_201(self):
        svc = _make_svc()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 201
        with patch.object(svc._session, "post", return_value=mock_resp):
            result = svc.sync_attendance(
                student_id="CSE001",
                student_name="Sudheer",
                attendance_date=date.today(),
                attendance_time=time(9, 0),
                subject="Java",
                teacher_id="T001",
                session="Period 1",
            )
        assert result is True

    def test_returns_false_on_500(self):
        svc = _make_svc()
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch.object(svc._session, "post", return_value=mock_resp):
            result = svc.sync_attendance(
                student_id="CSE001",
                student_name="Sudheer",
                attendance_date=date.today(),
                attendance_time=time(9, 0),
                subject="Java",
                teacher_id="T001",
                session="Period 1",
            )
        assert result is False

    def test_returns_false_on_timeout(self):
        svc = _make_svc()
        with patch.object(svc._session, "post", side_effect=requests.ReadTimeout):
            result = svc.sync_attendance(
                student_id="CSE001",
                student_name="Sudheer",
                attendance_date=date.today(),
                attendance_time=time(9, 0),
                subject="Java",
                teacher_id="T001",
                session="Period 1",
            )
        assert result is False

    def test_returns_false_on_connection_error(self):
        svc = _make_svc()
        with patch.object(svc._session, "post", side_effect=requests.ConnectionError):
            result = svc.sync_attendance(
                student_id="CSE001",
                student_name="Sudheer",
                attendance_date=date.today(),
                attendance_time=time(9, 0),
                subject="Java",
                teacher_id="T001",
                session="Period 1",
            )
        assert result is False

    def test_payload_contains_register_number(self):
        """Ensure the payload key matches what Spring Boot expects."""
        svc = _make_svc()
        captured = {}

        def mock_post(url, json=None, timeout=None):
            captured.update(json or {})
            r = MagicMock()
            r.ok = True
            return r

        with patch.object(svc._session, "post", side_effect=mock_post):
            svc.sync_attendance(
                student_id="CSE001",
                student_name="Sudheer",
                attendance_date=date.today(),
                attendance_time=time(9, 0),
                subject="Java",
                teacher_id="T001",
                session="Period 1",
            )

        assert "registerNumber" in captured
        assert captured["registerNumber"] == "CSE001"
        assert captured["source"] == "PYTHON_FACE_SERVICE"
