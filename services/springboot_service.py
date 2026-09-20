"""
services/springboot_service.py – HTTP client for Spring Boot REST API.

RESPONSIBILITIES (PHASE 8)
--------------------------
Python calls Spring Boot for:
  1. Verifying a student exists (GET /api/students/{registerNumber})
  2. Sending a recognised register number + context to create attendance
     (POST /api/attendance/mark)
  3. Health check (GET /actuator/health)

Spring Boot is responsible for:
  - Persistent attendance records in PostgreSQL
  - Timetable/session validation
  - Teacher/subject/period verification
  - Student status checks (active/inactive)

Python does NOT duplicate the full student database.  The local SQLite
database stores only the data needed to:
  - Associate a face encoding with a register number
  - Display the student name during live recognition
  - Cache attendance records while Spring Boot is unreachable

FAILURE HANDLING
----------------
If Spring Boot is unreachable, attendance is still saved locally in SQLite
and marked synced_to_spring=False.  A background retry mechanism can be
added later.  The face service never blocks attendance marking for a Spring
Boot outage.

SECURITY (PHASE 13)
--------------------
All URLs and credentials are loaded from environment variables / .env.
No secrets are hardcoded.
"""

from __future__ import annotations

from datetime import date, time
from typing import Any, Optional

import requests
from requests.exceptions import ConnectionError, ReadTimeout, RequestException

from config import get_settings
from utils.constants import ATTENDANCE_STATUS_PRESENT
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class SpringBootService:
    """HTTP client for the Spring Boot backend.

    All methods are synchronous — call via ``asyncio.to_thread`` from async
    route handlers to avoid blocking the event loop.

    Args:
        base_url:  Spring Boot base URL (defaults to settings).
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: Optional[int] = None,
    ) -> None:
        self._base_url = (base_url or settings.spring_boot_base_url).rstrip("/")
        self._timeout  = timeout_s or settings.spring_boot_timeout
        self._session  = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        logger.debug(
            "SpringBootService init – base=%s timeout=%ds",
            self._base_url, self._timeout,
        )

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Ping Spring Boot actuator health endpoint.

        Returns:
            True if Spring Boot responds with 2xx.
        """
        url = f"{self._base_url}/actuator/health"
        try:
            resp = self._session.get(url, timeout=self._timeout)
            ok = resp.ok
            logger.debug("Spring Boot health check: %s", "OK" if ok else f"HTTP {resp.status_code}")
            return ok
        except RequestException as exc:
            logger.warning("Spring Boot health check failed: %s", exc)
            return False

    # ── Student verification ──────────────────────────────────────────────────

    def verify_student(self, register_number: str) -> Optional[dict]:
        """Fetch student details from Spring Boot by register number.

        Used during enrollment to confirm the student exists in PostgreSQL
        before opening the camera.

        Args:
            register_number: Student register number.

        Returns:
            Student dict from Spring Boot, or None on failure / not found.
        """
        url = f"{self._base_url}{settings.spring_boot_student_endpoint}/{register_number}"
        try:
            resp = self._session.get(url, timeout=self._timeout)
            if resp.status_code == 404:
                logger.warning("Student %s not found in Spring Boot", register_number)
                return None
            if resp.ok:
                data = resp.json()
                logger.info("Verified student %s from Spring Boot", register_number)
                return data
            logger.error(
                "Spring Boot student lookup HTTP %d for %s",
                resp.status_code, register_number,
            )
            return None
        except ConnectionError:
            logger.warning("Cannot connect to Spring Boot – is it running?")
            return None
        except RequestException as exc:
            logger.exception("Student verify error: %s", exc)
            return None

    # ── Attendance sync ───────────────────────────────────────────────────────

    def sync_attendance(
        self,
        *,
        student_id: str,
        student_name: str,
        attendance_date: date,
        attendance_time: time,
        subject: str,
        teacher_id: str,
        session: str,
        confidence: float = 0.0,
        status: str = ATTENDANCE_STATUS_PRESENT,
    ) -> bool:
        """POST a recognised attendance record to Spring Boot.

        Spring Boot is responsible for:
          - Verifying the timetable/session is active
          - Verifying the teacher/subject/period combination
          - Creating the persistent PostgreSQL record

        Args:
            student_id:       Register number (matches Spring Boot student).
            student_name:     Display name.
            attendance_date:  Date of attendance.
            attendance_time:  Time of recognition.
            subject:          Subject name.
            teacher_id:       Teacher identifier.
            session:          Session/period label.
            confidence:       Raw L2 distance stored as metadata.
            status:           Attendance status string (default PRESENT).

        Returns:
            True if Spring Boot accepted (2xx), False otherwise.
        """
        url = f"{self._base_url}{settings.spring_boot_attendance_endpoint}"
        payload = {
            "registerNumber":  student_id,
            "studentName":     student_name,
            "date":            attendance_date.isoformat(),
            "time":            attendance_time.strftime("%H:%M:%S"),
            "subject":         subject,
            "teacherId":       teacher_id,
            "session":         session,
            "recognitionDistance": round(confidence, 6),
            "status":          status,
            "source":          "PYTHON_FACE_SERVICE",
        }

        logger.info(
            "Syncing attendance to Spring Boot: student=%s subject=%s",
            student_id, subject,
        )

        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            if resp.ok:
                logger.info(
                    "Spring Boot sync OK (HTTP %d) student=%s",
                    resp.status_code, student_id,
                )
                return True
            logger.error(
                "Spring Boot sync HTTP %d for student=%s: %s",
                resp.status_code, student_id, resp.text[:300],
            )
            return False
        except ConnectionError:
            logger.warning(
                "Spring Boot unreachable at %s – attendance saved locally only", url
            )
            return False
        except ReadTimeout:
            logger.warning(
                "Spring Boot timed out after %ds for student=%s",
                self._timeout, student_id,
            )
            return False
        except RequestException as exc:
            logger.exception("Spring Boot sync unexpected error: %s", exc)
            return False

    # ── Unsynced records retry ────────────────────────────────────────────────

    def retry_unsynced(self, db_session: Any) -> int:
        """Push all locally-saved but unsynced records to Spring Boot.

        Called on startup or periodically to flush the local queue.

        Args:
            db_session: SQLAlchemy Session.

        Returns:
            Number of records successfully synced.
        """
        from repository.attendance_repository import AttendanceRepository
        from repository.encoding_repository import EncodingRepository

        att_repo     = AttendanceRepository()
        enc_repo     = EncodingRepository()
        unsynced     = att_repo.get_unsynced_records(db_session)
        synced_count = 0

        if not unsynced:
            logger.debug("No unsynced attendance records to retry")
            return 0

        logger.info("Retrying %d unsynced attendance record(s)", len(unsynced))

        for record in unsynced:
            student = enc_repo.get_student_by_id(db_session, record.student_id)
            if student is None:
                logger.warning("Skipping unsynced record – student %s not in DB", record.student_id)
                continue

            ok = self.sync_attendance(
                student_id=record.student_id,
                student_name=student.name,   # type: ignore[arg-type]
                attendance_date=record.attendance_date,
                attendance_time=record.attendance_time,
                subject=record.subject,
                teacher_id=record.teacher_id,
                session=record.session,
                confidence=record.confidence,
                status=record.status,
            )
            if ok:
                att_repo.mark_synced(db_session, record.id)
                synced_count += 1

        if synced_count:
            db_session.commit()

        logger.info("Retry complete – synced %d/%d records", synced_count, len(unsynced))
        return synced_count
