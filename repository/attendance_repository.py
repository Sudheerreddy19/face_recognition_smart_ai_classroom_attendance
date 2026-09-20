"""
repository/attendance_repository.py – Attendance record CRUD (database layer).
"""

from __future__ import annotations

from datetime import date, time

from sqlalchemy.orm import Session

from models.attendance import AttendanceRecord
from utils.logger import get_logger

logger = get_logger(__name__)


class AttendanceRepository:
    """Data access layer for :class:`~models.attendance.AttendanceRecord`."""

    # ── Read ──────────────────────────────────────────────────────────────────

    def is_already_marked(
        self,
        db: Session,
        *,
        student_id: str,
        subject: str,
        attendance_date: date,
    ) -> bool:
        """Check whether attendance was already recorded today.

        Args:
            db:              Active SQLAlchemy session.
            student_id:      Student identifier.
            subject:         Subject / course name.
            attendance_date: Date to check.

        Returns:
            ``True`` if a record already exists.
        """
        return (
            db.query(AttendanceRecord.id)
            .filter(
                AttendanceRecord.student_id == student_id,
                AttendanceRecord.subject == subject,
                AttendanceRecord.attendance_date == attendance_date,
            )
            .first()
            is not None
        )

    def get_records_by_student(
        self, db: Session, student_id: str
    ) -> list[AttendanceRecord]:
        """Return all attendance records for a student.

        Args:
            db:         Active SQLAlchemy session.
            student_id: Student identifier.

        Returns:
            List of :class:`~models.attendance.AttendanceRecord` objects.
        """
        return (
            db.query(AttendanceRecord)
            .filter(AttendanceRecord.student_id == student_id)
            .order_by(AttendanceRecord.attendance_date.desc())
            .all()
        )

    def get_unsynced_records(self, db: Session) -> list[AttendanceRecord]:
        """Return records not yet synced to Spring Boot.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            List of un-synced :class:`~models.attendance.AttendanceRecord` objects.
        """
        return (
            db.query(AttendanceRecord)
            .filter(AttendanceRecord.synced_to_spring == False)  # noqa: E712
            .order_by(AttendanceRecord.created_at.asc())
            .all()
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def create(
        self,
        db: Session,
        *,
        student_id: str,
        subject: str,
        teacher_id: str,
        session: str,
        attendance_date: date,
        attendance_time: time,
        confidence: float,
        status: str = "PRESENT",
    ) -> AttendanceRecord:
        """Insert a new attendance record.

        Args:
            db:              Active SQLAlchemy session.
            student_id:      Student identifier.
            subject:         Subject / course name.
            teacher_id:      Teacher identifier.
            session:         Session label.
            attendance_date: Date of attendance.
            attendance_time: Time of recognition.
            confidence:      Recognition confidence score.
            status:          Attendance status (default ``PRESENT``).

        Returns:
            Newly created :class:`~models.attendance.AttendanceRecord`.
        """
        record = AttendanceRecord(
            student_id=student_id,
            subject=subject,
            teacher_id=teacher_id,
            session=session,
            attendance_date=attendance_date,
            attendance_time=attendance_time,
            confidence=confidence,
            status=status,
            synced_to_spring=False,
        )
        db.add(record)
        db.flush()
        logger.info(
            "Created attendance record: student=%s subject=%s date=%s",
            student_id,
            subject,
            attendance_date,
        )
        return record

    def mark_synced(self, db: Session, record_id: int) -> None:
        """Mark a record as synced to Spring Boot.

        Args:
            db:        Active SQLAlchemy session.
            record_id: Primary key of the record to update.
        """
        record = db.query(AttendanceRecord).filter_by(id=record_id).first()
        if record:
            record.synced_to_spring = True
            db.flush()
            logger.debug("Marked attendance record %d as synced", record_id)
