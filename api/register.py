"""
api/register.py – Student face registration endpoints.

ENDPOINTS
---------
POST /register-face
    Body: RegisterFaceRequest JSON
    Opens webcam, captures validated face frames, generates encodings,
    persists everything, and returns a RegistrationResponse.

DELETE /student/{student_id}
    Soft-deletes the student record and removes the encoding file.

GET /students
    Returns all registered students with metadata.

GET /encodings
    Returns metadata for every encoding file on disk.

DESIGN
------
- Camera capture and quality validation live in EnrollmentService.
- Database operations live in EncodingRepository.
- This file only orchestrates: validate → enroll → persist → respond.
- Spring Boot is NOT called from here (that is the attendance flow).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db
from models.response_models import (
    DeleteResponse,
    EncodingInfo,
    EncodingsListResponse,
    ErrorResponse,
    RegistrationResponse,
    StudentInfo,
    StudentsListResponse,
)
from repository.encoding_repository import EncodingRepository
from schemas.register_request import RegisterFaceRequest
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.encoding_service import EncodingService
from services.enrollment_service import EnrollmentService
from services.springboot_service import SpringBootService
from utils.constants import MSG_REGISTRATION_SUCCESS
from utils.logger import get_logger

router = APIRouter(tags=["Registration"])
logger = get_logger(__name__)
settings = get_settings()

# ── Shared service singletons ─────────────────────────────────────────────────
_camera_svc = CameraService()
_detector = DetectionService()
_encoder = EncodingService()
_student_repo = EncodingRepository()
_spring_svc = SpringBootService()


def _get_enrollment_service() -> EnrollmentService:
    """Build an EnrollmentService with the shared singletons."""
    return EnrollmentService(
        camera_service=_camera_svc,
        detection_service=_detector,
        encoding_service=_encoder,
        show_preview=True,
    )


# ── POST /face/capture/{student_db_id} ───────────────────────────────────────
# This is the endpoint Spring Boot calls.
# Spring Boot sends student details in the body; Python captures the face,
# saves the encoding, and returns success/failure.

from typing import Any
from pydantic import BaseModel

class FaceCaptureRequest(BaseModel):
    """Flexible body accepted from Spring Boot face capture call.

    Supports both camelCase (Spring Boot default) and snake_case field names.
    All fields except registerNumber / student_id are optional so the endpoint
    works regardless of which fields Spring Boot chooses to send.
    """
    # Register number — accept both naming conventions
    registerNumber: str | None = None
    student_id:     str | None = None

    # Name — accept both naming conventions
    studentName: str | None = None
    student_name: str | None = None

    # Optional metadata (stored locally; not required to start capture)
    department:  str | None = None
    semester:    str | None = None
    section:     str | None = None
    teacher_id:  str | None = None

    def resolved_register_number(self) -> str | None:
        return (self.registerNumber or self.student_id or "").strip() or None

    def resolved_name(self) -> str:
        return (self.studentName or self.student_name or "").strip()


@router.post(
    "/face/capture/{student_db_id}",
    summary="Capture face for a student (called by Spring Boot)",
    description=(
        "Spring Boot calls this endpoint when the teacher clicks 'Capture Face' "
        "in the React UI. The path parameter is the student's Spring Boot DB ID. "
        "The body contains student details (registerNumber, studentName, etc.). "
        "Python opens the webcam, captures and encodes the face, saves it locally, "
        "and returns a JSON result to Spring Boot."
    ),
)
async def face_capture(
    student_db_id: int,
    body: FaceCaptureRequest | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """Handle face capture request from Spring Boot.

    Args:
        student_db_id: Spring Boot student primary key (path param).
        body:          Student details sent by Spring Boot (optional).
        db:            Injected SQLAlchemy session.

    Returns:
        JSON with ``success``, ``message``, and ``registerNumber``.
    """
    # Extract register number from body (prefer camelCase)
    register_number = None
    student_name = "Unknown"

    if body:
        register_number = body.resolved_register_number()
        student_name    = body.resolved_name() or student_name

    logger.info(
        "Face capture request – db_id=%d register=%s name=%s",
        student_db_id, register_number, student_name,
    )

    if not register_number:
        logger.warning("No registerNumber in body for db_id=%d", student_db_id)
        return {
            "success": False,
            "message": "registerNumber is required in the request body.",
            "studentDbId": student_db_id,
        }

    # Check for existing encoding (re-enrollment allowed — overwrites old data)
    already_enrolled = _student_repo.student_exists(db, register_number)
    if already_enrolled:
        logger.info("Re-enrollment: overwriting existing encoding for %s", register_number)

    # Run camera capture + encoding in a background thread (blocking I/O)
    enrollment_svc = _get_enrollment_service()
    try:
        result = await asyncio.to_thread(enrollment_svc.enroll, register_number)
    except Exception as exc:
        logger.exception("Enrollment thread error for %s: %s", register_number, exc)
        return {
            "success": False,
            "message": f"Face capture error: {exc}",
            "registerNumber": register_number,
            "studentDbId": student_db_id,
        }

    if not result.success:
        logger.warning(
            "Enrollment failed for %s: %s", register_number, result.error_message
        )
        return {
            "success": False,
            "message": result.error_message or "Face capture failed.",
            "errorCode": result.error_code,
            "registerNumber": register_number,
            "studentDbId": student_db_id,
        }

    # Persist student metadata to local SQLite (only if not already there)
    if not already_enrolled:
        try:
            _student_repo.create_student(
                db,
                student_id=register_number,
                name=student_name,
                department=body.department or "" if body else "",
                semester=body.semester or "" if body else "",
                section=body.section or "" if body else "",
                teacher_id=body.teacher_id or "" if body else "",
            )
            db.commit()
        except Exception as exc:
            logger.warning(
                "Could not save student metadata for %s: %s", register_number, exc
            )


    logger.info(
        "Face capture SUCCESS – register=%s encodings=%d",
        register_number, len(result.encodings),
    )

    # Save encodings to disk and get path
    try:
        encoding_path = _encoder.save_encodings(register_number, result.encodings)
        result.encoding_path = encoding_path
    except Exception as exc:
        logger.warning("Could not save encoding file for %s: %s", register_number, exc)
        encoding_path = None

    return {
        "success": True,
        "message": f"Face captured successfully for {register_number}.",
        "registerNumber": register_number,
        "studentDbId": student_db_id,
        "images_captured": result.images_captured,
        "encoding_path": str(result.encoding_path) if result.encoding_path else None,
        "image_paths": [str(p) for p in result.saved_paths],
        "encodingCount": len(result.encodings),
        "alreadyEnrolled": already_enrolled,
    }


# ── POST /register-face/verify ────────────────────────────────────────────────

@router.post(
    "/register-face/verify",
    summary="Verify a student exists in Spring Boot before enrollment",
    description=(
        "Calls the Spring Boot backend to confirm the student register number "
        "exists and is active **before** opening the camera for enrollment.\n\n"
        "This keeps Python as a face-only service — student identity is always "
        "verified against the Spring Boot + PostgreSQL source of truth.\n\n"
        "If Spring Boot is unreachable the response will indicate that and "
        "enrollment can still proceed at the teacher's discretion."
    ),
)
async def verify_student_before_enrollment(
    register_number: str,
    db: Session = Depends(get_db),
) -> dict:
    """Check that a student register number exists in Spring Boot.

    Args:
        register_number: Student register number from the URL query param.
        db:              Injected SQLAlchemy session (for local duplicate check).

    Returns:
        JSON with ``found``, ``student`` details if found, and ``already_enrolled``
        flag indicating whether a face encoding already exists locally.
    """
    logger.info("Verify student before enrollment: %s", register_number)

    # Local duplicate check
    already_enrolled = _student_repo.student_exists(db, register_number)

    # Spring Boot lookup
    student_data = await asyncio.to_thread(
        _spring_svc.verify_student, register_number
    )

    if student_data is None:
        spring_reachable = await asyncio.to_thread(_spring_svc.health_check)
        if not spring_reachable:
            return {
                "found": None,              # unknown — Spring Boot unreachable
                "spring_boot_reachable": False,
                "already_enrolled": already_enrolled,
                "message": (
                    "Spring Boot is not reachable. Cannot verify student. "
                    "Proceed with caution or wait for Spring Boot to start."
                ),
            }
        return {
            "found": False,
            "spring_boot_reachable": True,
            "already_enrolled": already_enrolled,
            "message": f"Student '{register_number}' not found in Spring Boot.",
        }

    return {
        "found": True,
        "spring_boot_reachable": True,
        "already_enrolled": already_enrolled,
        "student": student_data,
        "message": (
            f"Student '{register_number}' verified. "
            + ("Already enrolled — re-enrollment will overwrite the existing face data."
               if already_enrolled else "Ready for face enrollment.")
        ),
    }


# ── POST /register-face ───────────────────────────────────────────────────────

@router.post(
    "/register-face",
    response_model=RegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a student's face",
    description=(
        "Opens the webcam and captures multiple validated face images for a "
        "new student. Applies quality checks (blur, face size, single-face "
        "requirement). Generates face encodings and stores them to disk and "
        "the SQLite database."
    ),
    responses={
        409: {"model": ErrorResponse, "description": "Student already registered"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Camera / encoding failure"},
    },
)
async def register_face(
    request: RegisterFaceRequest,
    db: Session = Depends(get_db),
) -> RegistrationResponse:
    """Register a student's face.

    Args:
        request: Validated JSON body with student details.
        db:      Injected SQLAlchemy session.

    Returns:
        :class:`~models.response_models.RegistrationResponse` on success.

    Raises:
        HTTPException 409: Student already registered.
        HTTPException 500: Camera, encoding, or DB failure.
    """
    logger.info(
        "Registration request – student_id=%s name=%s dept=%s",
        request.student_id,
        request.student_name,
        request.department,
    )

    # ── 1. Duplicate check ────────────────────────────────────────────────────
    if _student_repo.student_exists(db, request.student_id):
        logger.warning(
            "Duplicate registration attempt for student %s", request.student_id
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Student '{request.student_id}' is already registered. "
                "Delete the existing record first if you want to re-enroll."
            ),
        )

    # ── 2. Run enrollment in a thread (blocking camera I/O) ───────────────────
    enrollment_svc = _get_enrollment_service()
    try:
        result = await asyncio.to_thread(enrollment_svc.enroll, request.student_id)
    except Exception as exc:
        logger.exception("Unexpected error in enrollment thread: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment error: {exc}",
        ) from exc

    # ── 3. Handle enrollment failures ─────────────────────────────────────────
    if not result.success:
        error_code = result.error_code or "ENROLLMENT_FAILED"
        detail = result.error_message or "Enrollment failed."

        # Map specific codes to appropriate HTTP status
        if error_code == "CAMERA_ERROR":
            http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif error_code in ("NO_FACE", "MULTIPLE_FACES"):
            http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
        else:
            http_status = status.HTTP_500_INTERNAL_SERVER_ERROR

        logger.error(
            "Enrollment failed for student %s – code=%s detail=%s",
            request.student_id,
            error_code,
            detail,
        )
        raise HTTPException(status_code=http_status, detail=detail)

    # ── 4. Persist encodings to disk ──────────────────────────────────────────
    encoding_path = _encoder.save_encodings(request.student_id, result.encodings)
    result.encoding_path = encoding_path

    # ── 5. Save student record to SQLite ──────────────────────────────────────
    try:
        _student_repo.create_student(
            db,
            student_id=request.student_id,
            name=request.student_name,
            department=request.department,
            semester=request.semester,
            section=request.section,
            teacher_id=request.teacher_id,
            encoding_path=str(encoding_path),
            image_count=result.images_captured,
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception(
            "DB write failed for student %s: %s", request.student_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while saving student: {exc}",
        ) from exc

    logger.info(
        "Registration complete – student=%s images=%d encodings=%d skipped=%s",
        request.student_id,
        result.images_captured,
        len(result.encodings),
        result.skipped_frames,
    )

    return RegistrationResponse(
        success=True,
        message=MSG_REGISTRATION_SUCCESS,
        student_id=request.student_id,
        images_captured=result.images_captured,
        encoding_generated=True,
    )


# ── DELETE /student/{student_id} ──────────────────────────────────────────────

@router.delete(
    "/student/{student_id}",
    response_model=DeleteResponse,
    summary="Delete a student record",
    description=(
        "Soft-deletes the student in the database and removes the encoding "
        "pickle file from disk. Does not delete captured images."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Student not found"},
    },
)
async def delete_student(
    student_id: str,
    db: Session = Depends(get_db),
) -> DeleteResponse:
    """Soft-delete a student and remove their encoding file.

    Args:
        student_id: Student identifier from the URL path.
        db:         Injected SQLAlchemy session.

    Returns:
        :class:`~models.response_models.DeleteResponse`.

    Raises:
        HTTPException 404: Student not found.
    """
    if not _student_repo.student_exists(db, student_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{student_id}' not found.",
        )

    _student_repo.soft_delete_student(db, student_id)
    db.commit()

    # Remove encoding file from disk (best-effort — don't fail if missing)
    deleted_encoding = _encoder.delete_encoding(student_id)
    logger.info(
        "Student %s deleted (encoding_file_removed=%s)", student_id, deleted_encoding
    )

    return DeleteResponse(
        success=True,
        message=f"Student '{student_id}' has been deactivated.",
        student_id=student_id,
    )


# ── GET /students ─────────────────────────────────────────────────────────────

@router.get(
    "/students",
    response_model=StudentsListResponse,
    summary="List all registered students",
)
async def list_students(
    db: Session = Depends(get_db),
) -> StudentsListResponse:
    """Return all active student records.

    Args:
        db: Injected SQLAlchemy session.

    Returns:
        :class:`~models.response_models.StudentsListResponse`.
    """
    students = _student_repo.get_all_students(db)
    return StudentsListResponse(
        total=len(students),
        students=[StudentInfo.model_validate(s) for s in students],
    )


# ── GET /encodings ────────────────────────────────────────────────────────────

@router.get(
    "/encodings",
    response_model=EncodingsListResponse,
    summary="List all encoding files on disk",
)
async def list_encodings() -> EncodingsListResponse:
    """Return metadata for every encoding pickle on disk.

    Returns:
        :class:`~models.response_models.EncodingsListResponse`.
    """
    infos = _encoder.list_encoding_infos()
    return EncodingsListResponse(
        total=len(infos),
        encodings=[EncodingInfo(**info) for info in infos],
    )
