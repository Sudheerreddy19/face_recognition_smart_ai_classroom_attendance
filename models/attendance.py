"""
models/attendance.py – SQLAlchemy ORM model for Attendance Records.
"""

from __future__ import annotations

from datetime import date, datetime, time

from sqlalchemy import Date, DateTime, Float, ForeignKey, String, Time, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class AttendanceRecord(Base):
    """Represents a single attendance entry for a student in a session."""

    __tablename__ = "attendance_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Foreign key to Student
    student_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("students.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Session metadata
    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    teacher_id: Mapped[str] = mapped_column(String(50), nullable=False)
    session: Mapped[str] = mapped_column(String(50), nullable=False)

    # Temporal data
    attendance_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    attendance_time: Mapped[time] = mapped_column(Time, nullable=False)

    # Recognition metadata
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PRESENT")

    # Spring Boot sync flag
    synced_to_spring: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship back to Student
    student: Mapped["Student"] = relationship(  # noqa: F821
        back_populates="attendance_records"
    )

    def __repr__(self) -> str:
        return (
            f"<AttendanceRecord student={self.student_id!r} "
            f"date={self.attendance_date} subject={self.subject!r}>"
        )
