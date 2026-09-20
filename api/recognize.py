"""
api/recognize.py – Face recognition endpoints.

ENDPOINTS
---------
POST /recognize
    Opens the webcam, scans up to 5 frames, identifies the person.
    Returns flat JSON matching the spec (Phase 9):

    Recognized:
        {"success": true, "recognized": true, "registerNumber": "CSE001",
         "name": "Sudheer", "distance": 0.38, ...}

    Multiple faces:
        {"success": false, "recognized": false, "code": "MULTIPLE_FACES",
         "message": "Multiple faces detected. Attendance was not marked."}

    No face:
        {"success": false, "recognized": false, "code": "NO_FACE",
         "message": "No face detected."}

    Unknown person:
        {"success": false, "recognized": false, "code": "UNKNOWN",
         "message": "Face not recognised. Distance 0.72 exceeds threshold.",
         "distance": 0.72}

POST /recognize/reload
    Refreshes the in-memory encoding cache without restarting the server.
    Call this after enrolling a new student.
"""

from __future__ import annotations

import asyncio
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.response_models import (
    ErrorResponse,
    RecognizeFailResponse,
    RecognizeSuccessResponse,
)
from repository.encoding_repository import EncodingRepository
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.encoding_service import EncodingService
from services.recognition_service import RecognitionService
from utils.constants import RecognitionStatus
from utils.logger import get_logger

router = APIRouter(tags=["Recognition"])
logger = get_logger(__name__)

# ── Shared service singletons ─────────────────────────────────────────────────

_camera_svc = CameraService()
_detector = DetectionService()
_encoder = EncodingService()
_student_repo = EncodingRepository()

_recognizer = RecognitionService(
    camera_service=_camera_svc,
    detection_service=_detector,
    encoding_service=_encoder,
)


# ── POST /recognize ───────────────────────────────────────────────────────────

@router.post(
    "/recognize",
    summary="Identify a student via webcam",
    description=(
        "Opens the webcam, scans multiple frames for stability, detects the "
        "face, and compares the encoding against all enrolled students.\n\n"
        "**Multiple-face safety rule:** if more than one face is visible in "
        "any frame, recognition is aborted and MULTIPLE_FACES is returned — "
        "attendance is never marked in this case.\n\n"
        "**Distance note:** the `distance` field is a raw L2 distance between "
        "the query face encoding and the closest enrolled encoding. It is NOT "
        "a probability. Lower = more similar (0.00 = identical)."
    ),
    response_model=Union[RecognizeSuccessResponse, RecognizeFailResponse],
    responses={
        200: {"description": "Recognition result (success or known failure code)"},
        500: {"model": ErrorResponse, "description": "Camera or internal error"},
    },
)
async def recognize_face(
    db: Session = Depends(get_db),
) -> Union[RecognizeSuccessResponse, RecognizeFailResponse]:
    """Identify a student from the webcam.

    Returns a flat JSON response matching the API spec (Phase 9).

    Args:
        db: Injected SQLAlchemy session for student lookup.
    """
    logger.info("POST /recognize — starting recognition")

    # Run blocking camera + recognition in a thread
    try:
        dto = await asyncio.to_thread(_recognizer.recognize_from_camera)
    except Exception as exc:
        logger.exception("Unexpected error during recognition: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recognition error: {exc}",
        ) from exc

    # ── NO FACE ───────────────────────────────────────────────────────────────
    if dto.status == RecognitionStatus.NO_FACE:
        logger.info("Result: NO_FACE")
        return RecognizeFailResponse(
            code="NO_FACE",
            message="No face detected.",
            recognition_time_ms=dto.recognition_time_ms,
        )

    # ── MULTIPLE FACES ────────────────────────────────────────────────────────
    if dto.status == RecognitionStatus.MULTIPLE_FACES:
        logger.warning("Result: MULTIPLE_FACES")
        return RecognizeFailResponse(
            code="MULTIPLE_FACES",
            message="Multiple faces detected. Attendance was not marked.",
            recognition_time_ms=dto.recognition_time_ms,
        )

    # ── UNKNOWN ───────────────────────────────────────────────────────────────
    if dto.status == RecognitionStatus.UNKNOWN:
        dist_str = f"{dto.distance:.4f}" if dto.distance is not None else "N/A"
        threshold = _recognizer._threshold
        logger.info("Result: UNKNOWN  distance=%s", dist_str)
        return RecognizeFailResponse(
            code="UNKNOWN",
            message=(
                f"Face not recognised. "
                f"Distance {dist_str} exceeds threshold {threshold:.2f}."
            ),
            distance=dto.distance,
            recognition_time_ms=dto.recognition_time_ms,
        )

    # ── RECOGNIZED ───────────────────────────────────────────────────────────
    if dto.status == RecognitionStatus.RECOGNIZED and dto.student_id:
        # Fetch student details from local SQLite
        # (Python DB is a cache/mirror — Spring Boot is the source of truth)
        student = _student_repo.get_student_by_id(db, dto.student_id)

        if student is None:
            # Encoding file exists but student was deleted from DB
            logger.error(
                "Encoding found for %s but student record missing in DB",
                dto.student_id,
            )
            return RecognizeFailResponse(
                code="UNKNOWN",
                message=(
                    f"Encoding matched student '{dto.student_id}' but "
                    "the student record was not found in the local database."
                ),
                distance=dto.distance,
                recognition_time_ms=dto.recognition_time_ms,
            )

        logger.info(
            "Result: RECOGNIZED  student=%s  name=%s  distance=%.4f",
            dto.student_id,
            student.name,
            dto.distance or 0.0,
        )
        return RecognizeSuccessResponse(
            registerNumber=dto.student_id,
            name=student.name,
            department=student.department,
            semester=student.semester,
            section=student.section,
            distance=dto.distance or 0.0,
            recognition_time_ms=dto.recognition_time_ms,
        )

    # Should never reach here — guard against unexpected status values
    logger.error("Unhandled recognition status: %s", dto.status)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Unhandled recognition status: {dto.status}",
    )


# ── POST /recognize/reload ────────────────────────────────────────────────────

@router.post(
    "/recognize/reload",
    summary="Refresh in-memory encoding cache",
    description=(
        "Forces the recognition service to reload all face encodings from "
        "disk into memory. Call this after enrolling a new student so the "
        "new encoding is immediately available for recognition without "
        "restarting the server."
    ),
)
async def reload_encodings() -> dict:
    """Reload encodings from disk into the recognition cache.

    Returns:
        JSON with the number of students loaded.
    """
    count = await asyncio.to_thread(_recognizer.reload_encodings)
    logger.info("Encoding cache reloaded — %d student(s)", count)
    return {
        "success": True,
        "message": f"Encoding cache refreshed. {count} student(s) loaded.",
        "students_loaded": count,
    }
