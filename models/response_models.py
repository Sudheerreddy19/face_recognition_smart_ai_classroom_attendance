"""
models/response_models.py – Shared Pydantic response models.

These are returned directly from FastAPI route handlers.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Optional

from pydantic import BaseModel, Field

from utils.constants import CameraStatus, RecognitionStatus, ServiceStatus


# ── Camera ────────────────────────────────────────────────────────────────────


class CameraResponse(BaseModel):
    """Response schema for GET /camera."""

    status: CameraStatus
    message: str
    camera_index: int


# ── Registration ──────────────────────────────────────────────────────────────


class RegistrationResponse(BaseModel):
    """Response schema for POST /register-face."""

    success: bool
    message: str
    student_id: str
    images_captured: int
    encoding_generated: bool
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Recognition ───────────────────────────────────────────────────────────────


class RecognitionResult(BaseModel):
    """Single face recognition result (internal, used by attendance service)."""

    student_id: Optional[str] = None
    name: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[str] = None
    section: Optional[str] = None
    distance: Optional[float] = None
    status: RecognitionStatus
    recognition_time_ms: float


class RecognitionResponse(BaseModel):
    """Response schema for POST /recognize (internal nested form)."""

    success: bool
    result: RecognitionResult
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Flat recognition responses (per API spec, Phase 9) ───────────────────────
# These are the exact shapes returned by POST /recognize to callers.


class RecognizeSuccessResponse(BaseModel):
    """Returned when exactly one face is detected and matched.

    Example::

        {
            "success": true,
            "recognized": true,
            "registerNumber": "CSE001",
            "name": "Sudheer",
            "department": "CSE",
            "semester": "6",
            "section": "A",
            "distance": 0.38,
            "recognition_time_ms": 142.5
        }

    Note on ``distance``:
        Raw L2 distance between the query encoding and the closest enrolled
        encoding.  Lower = more similar.  This is NOT a probability.
        Typical same-person range: 0.05 – 0.35.
    """

    success: bool = True
    recognized: bool = True
    registerNumber: str
    name: Optional[str] = None
    department: Optional[str] = None
    semester: Optional[str] = None
    section: Optional[str] = None
    distance: float
    recognition_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


class RecognizeFailResponse(BaseModel):
    """Returned when recognition fails for any reason.

    Example (multiple faces)::

        {
            "success": false,
            "recognized": false,
            "code": "MULTIPLE_FACES",
            "message": "Multiple faces detected. Attendance was not marked."
        }

    Example (no face)::

        {
            "success": false,
            "recognized": false,
            "code": "NO_FACE",
            "message": "No face detected."
        }

    Example (unknown person)::

        {
            "success": false,
            "recognized": false,
            "code": "UNKNOWN",
            "message": "Face not recognised. Distance 0.72 exceeds threshold.",
            "distance": 0.72
        }
    """

    success: bool = False
    recognized: bool = False
    code: str                      # "NO_FACE" | "MULTIPLE_FACES" | "UNKNOWN"
    message: str
    distance: Optional[float] = None   # present for UNKNOWN, not for NO_FACE/MULTIPLE
    recognition_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Attendance ────────────────────────────────────────────────────────────────


class AttendanceResponse(BaseModel):
    """Response schema for POST /attendance.

    ``code`` is present on failure and matches the RecognitionStatus value
    so the React frontend can distinguish NO_FACE / MULTIPLE_FACES / UNKNOWN
    without parsing the message string.
    """

    success: bool
    message: str
    code: Optional[str] = None          # "NO_FACE" | "MULTIPLE_FACES" | "UNKNOWN" | "DUPLICATE" | "MARKED"
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    attendance_date: Optional[date] = None
    attendance_time: Optional[time] = None
    subject: Optional[str] = None
    already_marked: bool = False
    synced_to_spring: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Students List ─────────────────────────────────────────────────────────────


class StudentInfo(BaseModel):
    """Compact student information for list endpoints."""

    student_id: str
    name: str
    department: str
    semester: str
    section: str
    teacher_id: str
    image_count: int
    is_active: bool
    registered_at: datetime

    model_config = {"from_attributes": True}


class StudentsListResponse(BaseModel):
    """Response schema for GET /students."""

    total: int
    students: list[StudentInfo]


# ── Encodings ─────────────────────────────────────────────────────────────────


class EncodingInfo(BaseModel):
    """Information about a stored face encoding file."""

    student_id: str
    encoding_file: str
    face_count: int
    file_size_bytes: int


class EncodingsListResponse(BaseModel):
    """Response schema for GET /encodings."""

    total: int
    encodings: list[EncodingInfo]


# ── Delete ────────────────────────────────────────────────────────────────────


class DeleteResponse(BaseModel):
    """Response schema for DELETE /student/{id}."""

    success: bool
    message: str
    student_id: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Generic Error ─────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error envelope returned on all non-2xx responses."""

    success: bool = False
    error: str
    detail: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)
