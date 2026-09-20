"""
schemas/attendance_request.py â€“ Request schemas for attendance endpoints.

ACADEMIC CONTEXT MODEL
----------------------
Attendance is associated with:
    Student + Date + Subject + Period/Session + Teacher

This means one student can have multiple attendance records per day
(one per subject/period), and duplicate detection is keyed on
(student_id, subject, date) â€” NOT just (student_id, date).

Example timetable:
    09:00  Java     â†’ Present
    10:00  DBMS     â†’ Present
    11:00  AI       â†’ Absent
    12:00  Networks â†’ Present
"""

from pydantic import BaseModel, Field, field_validator


class AttendanceRequest(BaseModel):
    """Validated request body for POST /attendance.

    All fields are required.  The Python service does NOT verify whether
    the timetable/session is currently active â€” that validation is Spring
    Boot's responsibility (Phase 8).
    """

    subject: str = Field(
        default="Unknown",
        max_length=100,
        description="Subject / course name for this period",
        examples=["Java Programming", "DBMS", "Artificial Intelligence"],
    )
    teacher_id: str = Field(
        default="unknown",
        max_length=50,
        description="Teacher identifier (must exist in Spring Boot)",
        examples=["TEACH001"],
    )
    session: str = Field(
        default="session-1",
        max_length=50,
        description="Period or session label.",
        examples=["Period 1", "09:00", "Morning"],
    )

    department: str = Field(
        default="",
        max_length=100,
        description="Department filter (optional, used for reporting)",
        examples=["CSE", "ECE"],
    )
    semester: str = Field(
        default="",
        max_length=20,
        description="Semester filter (optional)",
        examples=["6", "4"],
    )
    section: str = Field(
        default="",
        max_length=20,
        description="Section filter (optional)",
        examples=["A", "B"],
    )
    academic_year: str = Field(
        default="",
        max_length=20,
        description="Academic year (optional, e.g. '2025-2026')",
        examples=["2025-2026"],
    )

    image_base64: str = Field(
        default="",
        description="Optional base64-encoded JPEG frame from browser webcam (data:image/jpeg;base64,...). If provided, skips camera capture.",
    )

    @field_validator("teacher_id", mode="before")
    @classmethod
    def strip_teacher_id(cls, v: str) -> str:
        return v.strip()

    @field_validator("subject", "session", "department", "semester", "section", mode="before")
    @classmethod
    def strip_fields(cls, v: str) -> str:
        return v.strip()
