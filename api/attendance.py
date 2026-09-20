"""
api/attendance.py – Attendance endpoints.

ENDPOINTS
---------
POST /attendance
    Mark attendance for one student via face recognition (single shot).

POST /attendance/session
    Start a live classroom session — camera stays open, marks automatically.
    Press Q in the camera window to end.

GET /attendance/today
    All records for today, optionally filtered by subject / teacher / section.

GET /attendance/history
    Records for a date range (default: last 7 days).
    Query params: student_id, subject, date_from, date_to

GET /attendance/student/{student_id}
    Full history for one student, enriched with subject / session / teacher.

ACADEMIC CONTEXT (PHASE 7)
---------------------------
Each record is keyed on: student + date + subject + session + teacher.
A student can have multiple records per day (one per subject).
Duplicate check: (student_id, subject, attendance_date) — never (student_id, date) only.

MULTIPLE-FACE RULE (PHASE 6)
-----------------------------
response.code == "MULTIPLE_FACES"  → attendance NOT marked
response.code == "NO_FACE"         → no face detected
response.code == "UNKNOWN"         → face detected but not recognised
response.code == "DUPLICATE"       → already marked for this subject today
response.code == "MARKED"          → successfully marked
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db
from models.attendance import AttendanceRecord
from models.response_models import AttendanceResponse, ErrorResponse
from repository.attendance_repository import AttendanceRepository
from repository.encoding_repository import EncodingRepository
from schemas.attendance_request import AttendanceRequest
from services.attendance_service import AttendanceService
from services.camera_service import CameraService
from services.classroom_service import ClassroomService
from services.detection_service import DetectionService
from services.encoding_service import EncodingService
from services.recognition_service import RecognitionService
from services.springboot_service import SpringBootService
from utils.logger import get_logger

router = APIRouter(tags=["Attendance"])
logger = get_logger(__name__)
settings = get_settings()

# ── Shared service singletons ─────────────────────────────────────────────────

_camera_svc = CameraService()
_detector   = DetectionService()
_encoder    = EncodingService()
_recognizer = RecognitionService(
    camera_service=_camera_svc,
    detection_service=_detector,
    encoding_service=_encoder,
)
_spring_svc = SpringBootService()


# ── POST /attendance ──────────────────────────────────────────────────────────

@router.post(
    "/attendance",
    response_model=AttendanceResponse,
    summary="Mark attendance for one student via face recognition",
    description=(
        "Opens the webcam once, scans up to 5 frames for the best face match, "
        "checks duplicate attendance for the same subject today, saves the "
        "record to SQLite, and asynchronously syncs to Spring Boot.\n\n"
        "**Response codes:**\n"
        "- `MARKED` — attendance successfully recorded\n"
        "- `DUPLICATE` — already marked for this subject today\n"
        "- `NO_FACE` — no face detected\n"
        "- `MULTIPLE_FACES` — more than one face visible, not marked\n"
        "- `UNKNOWN` — face detected but not matched to any enrolled student"
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Camera or recognition error"},
    },
)
async def mark_attendance(
    request: AttendanceRequest,
    db: Session = Depends(get_db),
) -> AttendanceResponse:
    logger.info(
        "POST /attendance – subject=%s teacher=%s session=%s",
        request.subject, request.teacher_id, request.session,
    )

    svc = AttendanceService(
        db=db,
        recognition_service=_recognizer,
        attendance_repo=AttendanceRepository(),
        encoding_repo=EncodingRepository(),
    )

    try:
        response = await asyncio.to_thread(svc.mark_attendance, request)
    except Exception as exc:
        logger.exception("Attendance marking error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    if response.success and not response.already_marked:
        asyncio.create_task(
            _sync_to_spring(response, request.teacher_id, request.session)
        )

    return response


# ── POST /attendance/multi ────────────────────────────────────────────────────

@router.post(
    "/attendance/multi",
    response_model=list[AttendanceResponse],
    summary="Mark attendance for ALL faces detected in a single browser frame",
    description=(
        "Receives a base64-encoded JPEG frame from the browser webcam, detects "
        "every face in the image, recognises each one against enrolled students "
        "in parallel, and marks attendance for all matches simultaneously.\n\n"
        "**Response:** A list of AttendanceResponse — one per detected face.\n"
        "Empty list = no faces detected in the frame."
    ),
)
async def mark_attendance_multi(
    request: AttendanceRequest,
    db: Session = Depends(get_db),
) -> list[AttendanceResponse]:
    logger.info(
        "POST /attendance/multi – subject=%s session=%s",
        request.subject, request.session,
    )

    svc = AttendanceService(
        db=db,
        recognition_service=_recognizer,
        attendance_repo=AttendanceRepository(),
        encoding_repo=EncodingRepository(),
    )

    try:
        results = await asyncio.to_thread(svc.mark_attendance_multi, request)
    except Exception as exc:
        logger.exception("Multi-face attendance error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # Async sync to Spring Boot for MARKED results (fire-and-forget)
    marked = [r for r in results if getattr(r, "code", None) == "MARKED"]
    if marked:
        for r in marked:
            asyncio.create_task(
                _sync_to_spring(r, request.teacher_id, request.session)
            )

    return results




# ── POST /attendance/session ──────────────────────────────────────────────────

@router.post(
    "/attendance/session",
    summary="Start a live classroom attendance session",
    description=(
        "Opens the camera and runs a continuous loop. Recognition is triggered "
        "automatically when one stable face is detected for 8 consecutive frames. "
        "Multiple faces block recognition. Press Q to end the session.\n\n"
        "Returns a session summary: students marked, multiple-face events, totals."
    ),
)
async def start_session(
    request: AttendanceRequest,
    db: Session = Depends(get_db),
) -> dict:
    logger.info(
        "POST /attendance/session – subject=%s session=%s",
        request.subject, request.session,
    )

    attendance_repo = AttendanceRepository()
    encoding_repo   = EncodingRepository()

    def _on_recognised(student_id: str, subject: str, session_label: str) -> None:
        """Callback fired by ClassroomService for each recognised student."""
        from schemas.attendance_request import AttendanceRequest as AR
        inner_req = AR(
            subject=subject,
            teacher_id=request.teacher_id,
            session=session_label,
        )
        svc = AttendanceService(
            db=db,
            recognition_service=_recognizer,
            attendance_repo=attendance_repo,
            encoding_repo=encoding_repo,
        )
        resp = svc._mark_already_recognised(student_id, inner_req)
        if resp.success and not resp.already_marked:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(
                    _sync_to_spring(resp, request.teacher_id, session_label)
                )
            except RuntimeError:
                pass   # no running loop in thread context — sync will retry on startup

    classroom_svc = ClassroomService(
        camera_service=_camera_svc,
        recognition_service=_recognizer,
        show_preview=True,
    )

    try:
        result = await asyncio.to_thread(
            classroom_svc.run_session,
            request.subject,
            request.teacher_id,
            request.session,
            _on_recognised,
        )
    except Exception as exc:
        logger.exception("Session error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return {
        "success": True,
        "subject": result.subject,
        "teacher_id": result.teacher_id,
        "session": result.session_label,
        "date": str(result.date),
        "students_marked": result.recognised,
        "students_marked_count": len(result.recognised),
        "duplicate_attempts": result.duplicates,
        "multiple_face_events": result.multiple_face_events,
        "no_face_frames": result.no_face_frames,
        "total_frames_processed": result.total_frames,
    }


# ── GET /attendance/today ─────────────────────────────────────────────────────

@router.get(
    "/attendance/today",
    summary="All attendance records for today",
    description=(
        "Returns every attendance record for today enriched with student name. "
        "Optional filters: subject, teacher_id, section, semester."
    ),
)
async def get_today_attendance(
    subject:    Optional[str] = Query(None, description="Filter by subject"),
    teacher_id: Optional[str] = Query(None, description="Filter by teacher ID"),
    section:    Optional[str] = Query(None, description="Filter by section"),
    semester:   Optional[str] = Query(None, description="Filter by semester"),
    db: Session = Depends(get_db),
) -> dict:
    today = date.today()

    query = db.query(AttendanceRecord).filter(
        AttendanceRecord.attendance_date == today
    )
    if subject:
        query = query.filter(AttendanceRecord.subject == subject)
    if teacher_id:
        query = query.filter(AttendanceRecord.teacher_id == teacher_id)

    records = query.order_by(
        AttendanceRecord.subject,
        AttendanceRecord.attendance_time,
    ).all()

    enc_repo = EncodingRepository()
    enriched = []
    for r in records:
        student = enc_repo.get_student_by_id(db, r.student_id)
        # Apply section / semester filters (stored on student, not record)
        if section and (student is None or student.section != section):
            continue
        if semester and (student is None or student.semester != semester):
            continue
        enriched.append({
            "student_id":   r.student_id,
            "student_name": student.name if student else None,
            "department":   student.department if student else None,
            "semester":     student.semester if student else None,
            "section":      student.section if student else None,
            "subject":      r.subject,
            "session":      r.session,
            "teacher_id":   r.teacher_id,
            "time":         str(r.attendance_time),
            "status":       r.status,
            "synced":       r.synced_to_spring,
        })

    return {
        "date": str(today),
        "filters": {
            "subject": subject,
            "teacher_id": teacher_id,
            "section": section,
            "semester": semester,
        },
        "total": len(enriched),
        "records": enriched,
    }


# ── GET /attendance/history ───────────────────────────────────────────────────

@router.get(
    "/attendance/history",
    summary="Attendance records for a date range",
    description=(
        "Returns attendance records between date_from and date_to "
        "(defaults to last 7 days). "
        "Optional filters: student_id, subject, teacher_id."
    ),
)
async def get_attendance_history(
    student_id:  Optional[str]  = Query(None, description="Filter by student register number"),
    subject:     Optional[str]  = Query(None, description="Filter by subject"),
    teacher_id:  Optional[str]  = Query(None, description="Filter by teacher ID"),
    date_from:   Optional[date] = Query(None, description="Start date (YYYY-MM-DD), default 7 days ago"),
    date_to:     Optional[date] = Query(None, description="End date (YYYY-MM-DD), default today"),
    db: Session = Depends(get_db),
) -> dict:
    if date_to is None:
        date_to = date.today()
    if date_from is None:
        date_from = date_to - timedelta(days=6)

    if date_from > date_to:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="date_from must be on or before date_to",
        )

    filters = [
        AttendanceRecord.attendance_date >= date_from,
        AttendanceRecord.attendance_date <= date_to,
    ]
    if student_id:
        filters.append(AttendanceRecord.student_id == student_id)
    if subject:
        filters.append(AttendanceRecord.subject == subject)
    if teacher_id:
        filters.append(AttendanceRecord.teacher_id == teacher_id)

    records = (
        db.query(AttendanceRecord)
        .filter(and_(*filters))
        .order_by(
            AttendanceRecord.attendance_date.desc(),
            AttendanceRecord.subject,
            AttendanceRecord.attendance_time,
        )
        .all()
    )

    enc_repo = EncodingRepository()
    student_cache: dict[str, object] = {}

    def _get_student(sid: str):
        if sid not in student_cache:
            student_cache[sid] = enc_repo.get_student_by_id(db, sid)
        return student_cache[sid]

    return {
        "date_from": str(date_from),
        "date_to":   str(date_to),
        "filters": {
            "student_id": student_id,
            "subject":    subject,
            "teacher_id": teacher_id,
        },
        "total": len(records),
        "records": [
            {
                "student_id":   r.student_id,
                "student_name": (s := _get_student(r.student_id)) and s.name,
                "department":   s.department if s else None,
                "semester":     s.semester   if s else None,
                "section":      s.section    if s else None,
                "date":         str(r.attendance_date),
                "subject":      r.subject,
                "session":      r.session,
                "teacher_id":   r.teacher_id,
                "time":         str(r.attendance_time),
                "status":       r.status,
                "synced":       r.synced_to_spring,
            }
            for r in records
        ],
    }


# ── GET /attendance/student/{student_id} ─────────────────────────────────────

@router.get(
    "/attendance/student/{student_id}",
    summary="Full attendance history for one student",
)
async def get_student_attendance(
    student_id: str,
    subject:    Optional[str]  = Query(None, description="Filter by subject"),
    date_from:  Optional[date] = Query(None, description="Start date YYYY-MM-DD"),
    date_to:    Optional[date] = Query(None, description="End date YYYY-MM-DD"),
    db: Session = Depends(get_db),
) -> dict:
    enc_repo = EncodingRepository()
    student = enc_repo.get_student_by_id(db, student_id)
    if student is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{student_id}' not found.",
        )

    att_repo = AttendanceRepository()
    records  = att_repo.get_records_by_student(db, student_id)

    # Optional filters
    if subject:
        records = [r for r in records if r.subject == subject]
    if date_from:
        records = [r for r in records if r.attendance_date >= date_from]
    if date_to:
        records = [r for r in records if r.attendance_date <= date_to]

    # Build per-subject summary
    subject_totals: dict[str, int] = {}
    for r in records:
        subject_totals[r.subject] = subject_totals.get(r.subject, 0) + 1

    return {
        "student_id":   student_id,
        "student_name": student.name,
        "department":   student.department,
        "semester":     student.semester,
        "section":      student.section,
        "total_records": len(records),
        "by_subject": subject_totals,
        "records": [
            {
                "date":       str(r.attendance_date),
                "subject":    r.subject,
                "session":    r.session,
                "teacher_id": r.teacher_id,
                "time":       str(r.attendance_time),
                "status":     r.status,
                "synced":     r.synced_to_spring,
            }
            for r in records
        ],
    }


# ── Spring Boot sync helper ───────────────────────────────────────────────────

async def _sync_to_spring(
    response: AttendanceResponse,
    teacher_id: str,
    session_label: str,
) -> None:
    """Fire-and-forget: push one attendance record to Spring Boot."""
    if not response.student_id or not response.attendance_date or not response.attendance_time:
        return
    try:
        synced = await asyncio.to_thread(
            _spring_svc.sync_attendance,
            student_id=response.student_id,
            student_name=response.student_name or "",
            attendance_date=response.attendance_date,
            attendance_time=response.attendance_time,
            subject=response.subject or "",
            teacher_id=teacher_id,
            session=session_label,
            confidence=0.0,
            status="PRESENT",
        )
        level = logger.info if synced else logger.warning
        level(  # type: ignore[operator]
            "Spring Boot sync %s – student=%s",
            "OK" if synced else "FAILED",
            response.student_id,
        )
    except Exception as exc:
        logger.exception(
            "Spring Boot sync error – student=%s: %s", response.student_id, exc
        )
