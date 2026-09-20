"""
services/classroom_service.py – Classroom live-view attendance session.

PURPOSE
-------
Provides a continuous camera loop suitable for a classroom scenario:

  - Show a live feed with face detection overlay.
  - Enforce the multiple-face safety rule: if >1 face is visible,
    do NOT attempt recognition and display a clear warning.
  - When exactly 1 face is detected for STABLE_FRAME_COUNT consecutive
    frames, trigger recognition automatically.
  - After a successful recognition, mark attendance by calling
    AttendanceService, then continue the loop (next student).
  - Display recognised student name and status on screen.
  - Press Q to end the session.

MULTIPLE-FACE RULE (PHASE 6)
-----------------------------
  0 faces  → "NO FACE — waiting"
  1 face   → attempt recognition after stability threshold
  2+ faces → "MULTIPLE FACES — cannot mark attendance"
             Recognition is NEVER attempted.
             The overlay is shown in RED.

This class is used by POST /recognize/session and is also importable
for standalone test scripts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import cv2
import numpy as np

from config import get_settings
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.encoding_service import EncodingService
from services.recognition_service import RecognitionService
from utils.constants import (
    RECT_COLOR_KNOWN,
    RECT_COLOR_UNKNOWN,
    RECT_COLOR_DETECTING,
    RecognitionStatus,
)
from utils.face_compat import face_distance
from utils.image_utils import bgr_to_rgb
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Frames with a single stable face required before triggering recognition
STABLE_FRAME_COUNT: int = 8

# Seconds to wait before the same student can be recognised again
# (prevents flooding the endpoint if the person stays in frame)
RERECOGNITION_COOLDOWN_S: float = 10.0

# Colours
_RED    = (0,   0, 255)
_GREEN  = (0, 255,   0)
_ORANGE = (0, 165, 255)
_WHITE  = (255, 255, 255)
_YELLOW = (0, 255, 255)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class SessionResult:
    """Summary of a completed classroom session."""
    subject: str
    teacher_id: str
    session_label: str
    date: date = field(default_factory=date.today)
    recognised: list[str] = field(default_factory=list)    # student IDs marked
    duplicates: list[str] = field(default_factory=list)    # already-marked
    multiple_face_events: int = 0
    no_face_frames: int = 0
    total_frames: int = 0


class ClassroomService:
    """Runs a continuous classroom attendance session with live camera feed.

    Args:
        camera_service:      Camera lifecycle manager.
        recognition_service: Face recognition engine.
        show_preview:        Whether to display the OpenCV window.
    """

    def __init__(
        self,
        camera_service: CameraService,
        recognition_service: RecognitionService,
        show_preview: bool = True,
    ) -> None:
        self._camera = camera_service
        self._recognizer = recognition_service
        self._show_preview = show_preview

    # ── Public API ────────────────────────────────────────────────────────────

    def run_session(
        self,
        subject: str,
        teacher_id: str,
        session_label: str,
        on_recognised,   # Callable[[str, str, str], None]
    ) -> SessionResult:
        """Run a live classroom attendance session.

        Opens the camera and loops until Q is pressed.
        Calls ``on_recognised(student_id, subject, session_label)`` when a
        student is successfully identified.

        Args:
            subject:        Subject name for this session.
            teacher_id:     Teacher identifier.
            session_label:  Session label (e.g. "Period 1").
            on_recognised:  Callback invoked on each successful recognition.

        Returns:
            :class:`SessionResult` summary.
        """
        result = SessionResult(
            subject=subject,
            teacher_id=teacher_id,
            session_label=session_label,
        )

        if not self._camera.open():
            logger.error("ClassroomService: cannot open camera")
            return result

        window_title = f"Classroom  |  {subject}  |  {session_label}  |  Q=quit"
        stable_count: int = 0
        last_recognised: dict[str, float] = {}   # student_id → last seen time

        logger.info(
            "Classroom session started – subject=%s teacher=%s", subject, teacher_id
        )

        try:
            while True:
                frame = self._camera.read_frame_with_retry(max_retries=2)
                if frame is None:
                    continue

                result.total_frames += 1
                h, w = frame.shape[:2]

                # Detect on half-resolution for speed
                small_rgb = bgr_to_rgb(cv2.resize(frame, (w // 2, h // 2)))
                locations_small = self._recognizer._detector.detect_faces(small_rgb)
                locations = self._recognizer._detector.scale_face_locations(
                    locations_small, scale=0.5
                )
                face_count = len(locations)

                display = frame.copy()
                self._draw_header(display, subject, session_label, len(result.recognised))

                # ── 0 faces ────────────────────────────────────────────────
                if face_count == 0:
                    result.no_face_frames += 1
                    stable_count = 0
                    self._overlay(display, "Waiting for student...", _WHITE, bottom=True)

                # ── 2+ faces  →  HARD BLOCK ────────────────────────────────
                elif face_count > 1:
                    result.multiple_face_events += 1
                    stable_count = 0
                    for top, right, bottom, left in locations:
                        cv2.rectangle(display, (left, top), (right, bottom), _RED, 2)
                    self._overlay(
                        display,
                        f"MULTIPLE FACES ({face_count}) — cannot mark attendance",
                        _RED, bottom=True,
                    )

                # ── 1 face  →  stability check + recognition ───────────────
                else:
                    stable_count += 1
                    top, right, bottom, left = locations[0]
                    box_color = _GREEN if stable_count >= STABLE_FRAME_COUNT else _ORANGE
                    cv2.rectangle(display, (left, top), (right, bottom), box_color, 2)

                    # Stability bar
                    bar_pct = min(stable_count / STABLE_FRAME_COUNT, 1.0)
                    bar_w = int((right - left) * bar_pct)
                    cv2.rectangle(display, (left, bottom + 4),
                                  (left + bar_w, bottom + 12), box_color, -1)

                    if stable_count >= STABLE_FRAME_COUNT:
                        stable_count = 0   # reset after trigger
                        self._overlay(display, "Recognising...", _YELLOW, bottom=True)
                        if self._show_preview:
                            cv2.imshow(window_title, display)
                            cv2.waitKey(1)

                        # ── Recognition ────────────────────────────────────
                        rgb = bgr_to_rgb(frame)
                        dto = self._recognizer.recognize_from_frame(rgb)

                        if dto.status == RecognitionStatus.MULTIPLE_FACES:
                            result.multiple_face_events += 1
                            self._overlay(
                                display,
                                "MULTIPLE FACES — blocked",
                                _RED, bottom=True,
                            )

                        elif dto.status == RecognitionStatus.RECOGNIZED and dto.student_id:
                            sid = dto.student_id
                            now = time.time()
                            last = last_recognised.get(sid, 0)

                            if (now - last) < RERECOGNITION_COOLDOWN_S:
                                self._overlay(
                                    display,
                                    f"Already seen: {sid}",
                                    _YELLOW, bottom=True,
                                )
                            else:
                                last_recognised[sid] = now
                                on_recognised(sid, subject, session_label)
                                result.recognised.append(sid)
                                self._overlay(
                                    display,
                                    f"Recognised: {sid}  dist={dto.distance:.3f}",
                                    _GREEN, bottom=True,
                                )
                                cv2.putText(display, f"PRESENT: {sid}",
                                            (left, top - 10), _FONT, 0.7, _GREEN, 2)

                        elif dto.status == RecognitionStatus.UNKNOWN:
                            self._overlay(
                                display,
                                f"Unknown face  dist={dto.distance:.3f}" if dto.distance else "Unknown face",
                                _RED, bottom=True,
                            )
                        else:
                            self._overlay(display, "No face in frame", _WHITE, bottom=True)
                    else:
                        self._overlay(
                            display,
                            f"Hold still... {stable_count}/{STABLE_FRAME_COUNT}",
                            _ORANGE, bottom=True,
                        )

                if self._show_preview:
                    cv2.imshow(window_title, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        logger.info("Classroom session ended by user")
                        break

        finally:
            self._camera.release()
            if self._show_preview:
                cv2.destroyAllWindows()

        logger.info(
            "Session done – recognised=%d duplicates=%d multi_face_events=%d",
            len(result.recognised),
            len(result.duplicates),
            result.multiple_face_events,
        )
        return result

    # ── Display helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _draw_header(frame: np.ndarray, subject: str, session: str, count: int) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(frame, f"{subject}  |  {session}  |  Marked: {count}",
                    (10, 35), _FONT, 0.7, _WHITE, 2)

    @staticmethod
    def _overlay(
        frame: np.ndarray,
        text: str,
        color: tuple[int, int, int],
        bottom: bool = False,
    ) -> None:
        h, w = frame.shape[:2]
        y = h - 15 if bottom else 80
        cv2.putText(frame, text, (10, y), _FONT, 0.65, color, 2)
