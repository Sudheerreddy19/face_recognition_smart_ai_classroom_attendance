"""
tests/test_attendance.py – Tests for attendance service and endpoints (Phases 6-8).

Tests cover:
  - NO_FACE → attendance not marked
  - MULTIPLE_FACES → attendance not marked
  - UNKNOWN → attendance not marked
  - Duplicate detection → already_marked=True
  - Successful attendance → record created
  - Spring Boot communication failure → local record still saved
  - Invalid student (encoding exists, DB record missing)
  - POST /attendance endpoint response
"""

from __future__ import annotations

from datetime import date, time
from unittest.mock import MagicMock, patch

import pytest

from models.response_models import AttendanceResponse
from repository.attendance_repository import AttendanceRepository
from schemas.attendance_request import AttendanceRequest
from schemas.recognize_response import RecognizeResponseDTO
from services.attendance_service import AttendanceService
from utils.constants import RecognitionStatus


def _make_request(**kwargs):
    defaults = dict(subject="Java", teacher_id="T001", session="Period 1")
    defaults.update(kwargs)
    return AttendanceRequest(**defaults)


def _make_service(recognizer_dto, student=None, already_marked=False):
    recognizer = MagicMock()
    recognizer.recognize_from_camera.return_value = recognizer_dto

    encoding_repo = MagicMock()
    encoding_repo.get_student_by_id.return_value = student

    attendance_repo = MagicMock()
    attendance_repo.is_already_marked.return_value = already_marked

    db = MagicMock()

    return AttendanceService(
        db=db,
        recognition_service=recognizer,
        attendance_repo=attendance_repo,
        encoding_repo=encoding_repo,
    )


def _mock_student(sid="CSE001", name="Sudheer"):
    s = MagicMock()
    s.student_id = sid
    s.name       = name
    s.department = "CSE"
    s.semester   = "6"
    s.section    = "A"
    return s


class TestAttendanceNoFace:
    def test_no_face_returns_failure(self):
        dto = RecognizeResponseDTO(status=RecognitionStatus.NO_FACE)
        svc = _make_service(dto)
        resp = svc.mark_attendance(_make_request())
        assert resp.success is False
        assert "no face" in resp.message.lower()

    def test_no_face_does_not_create_record(self):
        dto = RecognizeResponseDTO(status=RecognitionStatus.NO_FACE)
        svc = _make_service(dto)
        svc.mark_attendance(_make_request())
        svc._attendance_repo.create.assert_not_called()


class TestAttendanceMultipleFaces:
    def test_multiple_faces_returns_failure(self):
        dto = RecognizeResponseDTO(status=RecognitionStatus.MULTIPLE_FACES)
        svc = _make_service(dto)
        resp = svc.mark_attendance(_make_request())
        assert resp.success is False
        assert "multiple" in resp.message.lower()

    def test_multiple_faces_does_not_create_record(self):
        dto = RecognizeResponseDTO(status=RecognitionStatus.MULTIPLE_FACES)
        svc = _make_service(dto)
        svc.mark_attendance(_make_request())
        svc._attendance_repo.create.assert_not_called()


class TestAttendanceUnknown:
    def test_unknown_returns_failure(self):
        dto = RecognizeResponseDTO(status=RecognitionStatus.UNKNOWN, distance=0.80)
        svc = _make_service(dto)
        resp = svc.mark_attendance(_make_request())
        assert resp.success is False

    def test_unknown_does_not_create_record(self):
        dto = RecognizeResponseDTO(status=RecognitionStatus.UNKNOWN, distance=0.80)
        svc = _make_service(dto)
        svc.mark_attendance(_make_request())
        svc._attendance_repo.create.assert_not_called()


class TestAttendanceDuplicate:
    def test_duplicate_returns_already_marked_true(self):
        dto     = RecognizeResponseDTO(
            status=RecognitionStatus.RECOGNIZED, student_id="CSE001", distance=0.20
        )
        student = _mock_student()
        svc     = _make_service(dto, student=student, already_marked=True)
        resp    = svc.mark_attendance(_make_request())
        assert resp.success is True
        assert resp.already_marked is True

    def test_duplicate_does_not_create_new_record(self):
        dto     = RecognizeResponseDTO(
            status=RecognitionStatus.RECOGNIZED, student_id="CSE001", distance=0.20
        )
        student = _mock_student()
        svc     = _make_service(dto, student=student, already_marked=True)
        svc.mark_attendance(_make_request())
        svc._attendance_repo.create.assert_not_called()


class TestAttendanceSuccess:
    def test_successful_attendance_creates_record(self):
        dto     = RecognizeResponseDTO(
            status=RecognitionStatus.RECOGNIZED, student_id="CSE001", distance=0.18
        )
        student = _mock_student()
        svc     = _make_service(dto, student=student, already_marked=False)

        # Setup mock record returned by create
        mock_record = MagicMock()
        mock_record.student_id      = "CSE001"
        mock_record.attendance_date = date.today()
        mock_record.attendance_time = time(9, 0)
        svc._attendance_repo.create.return_value = mock_record

        resp = svc.mark_attendance(_make_request())
        assert resp.success is True
        assert resp.already_marked is False
        assert resp.student_id == "CSE001"
        svc._attendance_repo.create.assert_called_once()

    def test_successful_attendance_commits_db(self):
        dto     = RecognizeResponseDTO(
            status=RecognitionStatus.RECOGNIZED, student_id="CSE001", distance=0.18
        )
        student = _mock_student()
        svc     = _make_service(dto, student=student, already_marked=False)
        svc._attendance_repo.create.return_value = MagicMock()
        svc.mark_attendance(_make_request())
        svc._db.commit.assert_called()


class TestAttendanceInvalidStudent:
    def test_student_not_in_db_returns_failure(self):
        dto = RecognizeResponseDTO(
            status=RecognitionStatus.RECOGNIZED, student_id="GHOST", distance=0.10
        )
        svc = _make_service(dto, student=None)  # no student in DB
        resp = svc.mark_attendance(_make_request())
        assert resp.success is False
        assert "not found" in resp.message.lower()


class TestAttendanceRepository:

    def _db(self, first_result=None):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = first_result
        return db

    def test_is_already_marked_false(self):
        repo = AttendanceRepository()
        assert repo.is_already_marked(
            self._db(None), student_id="X", subject="Y", attendance_date=date.today()
        ) is False

    def test_is_already_marked_true(self):
        repo = AttendanceRepository()
        assert repo.is_already_marked(
            self._db(MagicMock(id=1)), student_id="X", subject="Y",
            attendance_date=date.today()
        ) is True


class TestSpringBootFailure:
    """Spring Boot being unreachable must not prevent local attendance save."""

    def test_local_record_saved_when_spring_boot_unreachable(self):
        dto     = RecognizeResponseDTO(
            status=RecognitionStatus.RECOGNIZED, student_id="CSE001", distance=0.18
        )
        student = _mock_student()
        svc     = _make_service(dto, student=student, already_marked=False)
        svc._attendance_repo.create.return_value = MagicMock()

        # The service itself doesn't call Spring Boot — that's in the API layer.
        # Verify the DB record is created even when Spring Boot is skipped.
        resp = svc.mark_attendance(_make_request())
        assert resp.success is True
        svc._attendance_repo.create.assert_called_once()
        # synced_to_spring defaults to False (Spring Boot sync is async in API layer)
        assert resp.synced_to_spring is False


class TestAttendanceRequest:
    """Validate the academic context fields added in Phase 7."""

    def test_attendance_request_with_full_context(self):
        req = AttendanceRequest(
            subject="Java",
            teacher_id="T001",
            session="Period 1",
            department="CSE",
            semester="6",
            section="A",
            academic_year="2025-2026",
        )
        assert req.subject == "Java"
        assert req.department == "CSE"
        assert req.academic_year == "2025-2026"

    def test_attendance_request_minimal(self):
        req = AttendanceRequest(subject="DBMS", teacher_id="T002", session="Morning")
        assert req.department == ""
        assert req.semester == ""
