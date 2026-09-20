"""
services/attendance_service.py – Business logic for marking attendance.

PIPELINE
--------
  1. Recognise face via RecognitionService (camera or pre-captured frame).
  2. Handle NO_FACE / MULTIPLE_FACES / UNKNOWN — return early without marking.
  3. Fetch student details from local SQLite.
  4. Check for duplicate attendance for the same subject on the same date.
  5. Persist a new AttendanceRecord to SQLite.
  6. Return AttendanceResponse (Spring Boot sync is triggered by the API layer).

DISTANCE FIELD
--------------
The recognition DTO carries ``distance`` (raw L2), not a confidence score.
Attendance records store this as ``confidence`` in the DB column for backward
compatibility, but it is documented as distance everywhere else.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy.orm import Session

from models.response_models import AttendanceResponse
from repository.attendance_repository import AttendanceRepository
from repository.encoding_repository import EncodingRepository
from schemas.attendance_request import AttendanceRequest
from services.recognition_service import RecognitionService
from utils.constants import (
    ATTENDANCE_STATUS_PRESENT,
    MSG_ATTENDANCE_DUPLICATE,
    MSG_ATTENDANCE_MARKED,
    MSG_NO_FACE_FOUND,
    MSG_MULTIPLE_FACES,
    MSG_STUDENT_NOT_FOUND,
    RecognitionStatus,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Message used when a face is detected but does not match any enrolled student
_MSG_NOT_RECOGNISED = (
    "Face detected but not recognised — distance exceeds threshold."
)


class AttendanceService:
    """Orchestrates the full attendance marking pipeline.

    Args:
        db:                  SQLAlchemy database session.
        recognition_service: Performs face recognition.
        attendance_repo:     Attendance CRUD.
        encoding_repo:       Student lookup.
    """

    def __init__(
        self,
        db: Session,
        recognition_service: RecognitionService,
        attendance_repo: AttendanceRepository,
        encoding_repo: EncodingRepository,
    ) -> None:
        self._db = db
        self._recognizer = recognition_service
        self._attendance_repo = attendance_repo
        self._encoding_repo = encoding_repo

    # ── Public API ────────────────────────────────────────────────────────────

    def mark_attendance(self, request: AttendanceRequest) -> AttendanceResponse:
        """Recognise a face from the camera and mark attendance if matched.

        Args:
            request: Validated attendance context (subject, teacher, session).

        Returns:
            :class:`~models.response_models.AttendanceResponse`
        """
        logger.info(
            "Attendance mark – subject=%s teacher=%s session=%s",
            request.subject,
            request.teacher_id,
            request.session,
        )

        # ── Step 1: Recognise ─────────────────────────────────────────────────
        if request.image_base64:
            # Browser sent a webcam frame — decode and use it directly
            import base64
            import numpy as np
            import cv2
            try:
                # Strip data URI prefix if present (data:image/jpeg;base64,...)
                b64_data = request.image_base64
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr_frame is None:
                    raise ValueError("Could not decode image")
                # Upscale if too small — face_recognition needs at least 480px height
                h, w = bgr_frame.shape[:2]
                if h < 480:
                    scale = 480.0 / h
                    bgr_frame = cv2.resize(bgr_frame, (int(w * scale), 480),
                                           interpolation=cv2.INTER_LINEAR)
                    logger.debug("Upscaled browser frame from %dx%d to %dx480", w, h, int(w * scale))
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                logger.info("Browser frame decoded: %dx%d px", rgb_frame.shape[1], rgb_frame.shape[0])
                dto = self._recognizer.recognize_from_frame(rgb_frame)
                logger.info("recognize_from_frame result: status=%s", dto.status)
            except Exception as e:
                logger.error("Failed to decode/process base64 image: %s", e)
                return AttendanceResponse(
                    success=False,
                    code="NO_FACE",
                    message=f"Image decode failed: {e}",
                )
        else:
            # Fallback: open physical camera on the server
            dto = self._recognizer.recognize_from_camera()

        if dto.status == RecognitionStatus.NO_FACE:
            return AttendanceResponse(
                success=False,
                code="NO_FACE",
                message=MSG_NO_FACE_FOUND,
            )

        if dto.status == RecognitionStatus.MULTIPLE_FACES:
            logger.warning("Attendance blocked — multiple faces detected")
            return AttendanceResponse(
                success=False,
                code="MULTIPLE_FACES",
                message=MSG_MULTIPLE_FACES,
            )

        if dto.status == RecognitionStatus.UNKNOWN:
            logger.info(
                "Face not recognised — distance=%s",
                f"{dto.distance:.4f}" if dto.distance is not None else "N/A",
            )
            return AttendanceResponse(
                success=False,
                code="UNKNOWN",
                message=_MSG_NOT_RECOGNISED,
            )

        # ── Step 2: Fetch student from local DB (optional) ────────────────────
        student_id = dto.student_id
        assert student_id is not None  # guaranteed when status == RECOGNIZED

        student = self._encoding_repo.get_student_by_id(self._db, student_id)
        if student is None:
            # Student has a pickle encoding but no row in Python's SQLite DB.
            # This is normal when enrollment was done via Spring Boot API.
            # We can still mark attendance using the recognised roll number.
            logger.warning(
                "Recognised %s — not in Python DB, proceeding with roll number only",
                student_id,
            )
            student_name = student_id  # use roll number as display name fallback

        else:
            student_name = student.name

        # ── Step 3: Duplicate check ───────────────────────────────────────────
        today = date.today()
        already_marked = self._attendance_repo.is_already_marked(
            self._db,
            student_id=student_id,
            subject=request.subject,
            attendance_date=today,
        )

        if already_marked:
            logger.info(
                "Duplicate — student=%s subject=%s date=%s",
                student_id, request.subject, today,
            )
            return AttendanceResponse(
                success=True,
                code="DUPLICATE",
                message=MSG_ATTENDANCE_DUPLICATE,
                student_id=student_id,
                student_name=student_name,
                attendance_date=today,
                subject=request.subject,
                already_marked=True,
            )

        # ── Step 4: Persist attendance record ─────────────────────────────────
        now = datetime.now()
        # distance stored in the ``confidence`` DB column for schema compat.
        distance_val = dto.distance if dto.distance is not None else 0.0

        record = self._attendance_repo.create(
            db=self._db,
            student_id=student_id,
            subject=request.subject,
            teacher_id=request.teacher_id,
            session=request.session,
            attendance_date=today,
            attendance_time=now.time(),
            confidence=distance_val,       # stored as-is; documented as distance
            status=ATTENDANCE_STATUS_PRESENT,
        )
        self._db.commit()
        self._db.refresh(record)

        logger.info(
            "Attendance marked — student=%s subject=%s date=%s time=%s distance=%.4f",
            student_id, request.subject, today, now.time(), distance_val,
        )

        return AttendanceResponse(
            success=True,
            code="MARKED",
            message=MSG_ATTENDANCE_MARKED,
            student_id=student_id,
            student_name=student_name,
            attendance_date=today,
            attendance_time=now.time(),
            subject=request.subject,
            already_marked=False,
            synced_to_spring=False,   # Spring Boot sync triggered by API layer
        )

    def _mark_already_recognised(
        self, student_id: str, request: AttendanceRequest
    ) -> AttendanceResponse:
        """Mark attendance for a student already identified by ClassroomService.

        Skips the camera/recognition step — the student_id is already known.
        Used by the live session loop so camera is not re-opened per student.

        Args:
            student_id: Already-recognised student register number.
            request:    Attendance context.

        Returns:
            :class:`~models.response_models.AttendanceResponse`
        """
        student = self._encoding_repo.get_student_by_id(self._db, student_id)
        if student is None:
            return AttendanceResponse(
                success=False,
                message=MSG_STUDENT_NOT_FOUND,
                student_id=student_id,
            )

        student_name: str = student.name  # type: ignore[assignment]
        today = date.today()

        if self._attendance_repo.is_already_marked(
            self._db,
            student_id=student_id,
            subject=request.subject,
            attendance_date=today,
        ):
            return AttendanceResponse(
                success=True,
                message=MSG_ATTENDANCE_DUPLICATE,
                student_id=student_id,
                student_name=student_name,
                attendance_date=today,
                subject=request.subject,
                already_marked=True,
            )

        now = datetime.now()
        record = self._attendance_repo.create(
            db=self._db,
            student_id=student_id,
            subject=request.subject,
            teacher_id=request.teacher_id,
            session=request.session,
            attendance_date=today,
            attendance_time=now.time(),
            confidence=0.0,
            status=ATTENDANCE_STATUS_PRESENT,
        )
        self._db.commit()
        self._db.refresh(record)

        logger.info(
            "Session attendance marked – student=%s subject=%s",
            student_id, request.subject,
        )
        return AttendanceResponse(
            success=True,
            message=MSG_ATTENDANCE_MARKED,
            student_id=student_id,
            student_name=student_name,
            attendance_date=today,
            attendance_time=now.time(),
            subject=request.subject,
            already_marked=False,
            synced_to_spring=False,
        )

    # ── Multi-face ────────────────────────────────────────────────────────────

    def mark_attendance_multi(
        self, request: "AttendanceRequest"
    ) -> list[AttendanceResponse]:
        """Detect ALL faces in a single frame and mark attendance for each.

        Steps:
          1. Decode base64 JPEG frame (from browser webcam).
          2. Detect all face locations using dlib HOG detector.
          3. For each face: recognise → duplicate-check → persist in parallel.
          4. Return list of AttendanceResponse (one per face).

        Args:
            request: Attendance context + ``image_base64`` field.

        Returns:
            List of :class:`~models.response_models.AttendanceResponse`.
            Empty list if no faces detected.
        """
        import base64
        import cv2
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.constants import RecognitionStatus

        logger.info(
            "mark_attendance_multi – subject=%s session=%s",
            request.subject, request.session,
        )

        # ── Decode frame ──────────────────────────────────────────────────────
        if not request.image_base64:
            logger.warning("mark_attendance_multi: no image_base64 provided")
            return []

        try:
            b64 = request.image_base64
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode returned None")
            # Upscale if small for better face detection
            h, w = bgr.shape[:2]
            if h < 480:
                scale = 480.0 / h
                bgr = cv2.resize(bgr, (int(w * scale), 480), interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            logger.info("multi: frame decoded %dx%d", rgb_frame.shape[1], rgb_frame.shape[0])
        except Exception as e:
            logger.error("multi: frame decode failed – %s", e)
            return []

        # ── Recognise all faces ───────────────────────────────────────────────
        dtos = self._recognizer.recognize_multiple_from_frame(rgb_frame)
        if not dtos:
            logger.info("multi: no faces found in frame")
            return []

        logger.info("multi: %d face(s) detected", len(dtos))

        # ── Mark attendance for each recognised face (sequential, same DB session) ──
        results: list[AttendanceResponse] = []
        today = date.today()

        for i, dto in enumerate(dtos):
            if dto.status == RecognitionStatus.NO_FACE:
                continue
            if dto.status == RecognitionStatus.MULTIPLE_FACES:
                continue
            if dto.status == RecognitionStatus.UNKNOWN:
                results.append(AttendanceResponse(
                    success=False,
                    code="UNKNOWN",
                    message="Face detected but not recognised",
                ))
                continue

            # RECOGNIZED ──────────────────────────────────────────────────────
            student_id = dto.student_id
            student    = self._encoding_repo.get_student_by_id(self._db, student_id)
            student_name = student.name if student else student_id

            # Duplicate check (same subject + same day)
            if self._attendance_repo.is_already_marked(
                self._db,
                student_id=student_id,
                subject=request.subject,
                attendance_date=today,
            ):
                logger.info("multi: duplicate – student=%s", student_id)
                results.append(AttendanceResponse(
                    success=True,
                    code="DUPLICATE",
                    message="Already marked for today",
                    student_id=student_id,
                    student_name=student_name,
                    attendance_date=today,
                    subject=request.subject,
                    already_marked=True,
                ))
                continue

            # Persist ─────────────────────────────────────────────────────────
            now = datetime.now()
            dist = dto.distance if dto.distance is not None else 0.0
            self._attendance_repo.create(
                db=self._db,
                student_id=student_id,
                subject=request.subject,
                teacher_id=request.teacher_id,
                session=request.session,
                attendance_date=today,
                attendance_time=now.time(),
                confidence=dist,
                status=ATTENDANCE_STATUS_PRESENT,
            )
            self._db.commit()

            logger.info("multi: MARKED – student=%s distance=%.4f", student_id, dist)
            results.append(AttendanceResponse(
                success=True,
                code="MARKED",
                message="Attendance marked",
                student_id=student_id,
                student_name=student_name,
                attendance_date=today,
                attendance_time=now.time(),
                subject=request.subject,
                already_marked=False,
                synced_to_spring=False,
            ))

        return results

