"""
services/enrollment_service.py – Student face enrollment pipeline.

RESPONSIBILITIES
----------------
This service owns the entire enrollment capture loop:

  1. Open the camera.
  2. Read frames continuously.
  3. For each frame, run face detection (MediaPipe via DetectionService).
  4. Validate the frame:
       - Exactly 1 face detected  (NO_FACE / MULTIPLE_FACES guard)
       - Face region is large enough (face too small)
       - Frame is not excessively blurred (Laplacian variance check)
  5. Every CAPTURE_FRAME_GAP frames that pass validation, capture the frame.
  6. Accumulate up to `capture_count` validated frames.
  7. Generate one face encoding per validated frame.
  8. Return the list of encodings + metadata.

DESIGN DECISIONS
----------------
- This service does NOT touch the database. That is the API layer's job.
- It returns a structured EnrollmentResult so the API can decide what to store.
- Spring Boot is NOT called here. It is called by the API layer after DB save.
- Camera is always released in a finally block — never leaked.
- All quality thresholds are configurable via config / constructor args.

PHASE 10 PREP
-------------
The camera is opened through CameraService which already accepts a camera_index.
Later, CameraService will be extended to support ESP32 streams transparently.
This service will not need to change when that happens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import get_settings
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.encoding_service import EncodingService
from utils.constants import (
    CAPTURE_FRAME_GAP,
    IMAGE_EXTENSION,
    JPEG_QUALITY,
    MIN_FACE_AREA_FRACTION,
    POSE_LABELS,
    RECT_COLOR_KNOWN,
    RECT_COLOR_DETECTING,
    RECT_COLOR_UNKNOWN,
)
from utils.file_utils import ensure_directories, get_student_image_dir
from utils.image_utils import bgr_to_rgb, draw_capture_progress, draw_fps, save_frame
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Quality thresholds (tunable) ──────────────────────────────────────────────

# Laplacian variance below this → frame is too blurry
_BLUR_THRESHOLD: float = 80.0

# Face bounding box must be at least this fraction of frame area
_MIN_FACE_FRACTION: float = MIN_FACE_AREA_FRACTION   # from constants (0.02)

# Face bounding box must NOT exceed this fraction (too close to camera)
_MAX_FACE_FRACTION: float = 0.85

# Minimum face height in pixels (absolute guard regardless of resolution)
_MIN_FACE_HEIGHT_PX: int = 60


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EnrollmentResult:
    """Outcome of one complete enrollment session.

    Attributes:
        success:          True if at least `min_encodings` were generated.
        student_id:       Student identifier used for this session.
        encodings:        List of face encoding vectors (1 per captured frame).
        images_captured:  Number of validated frames captured.
        saved_paths:      Paths of saved JPEG images on disk.
        encoding_path:    Path where encodings were persisted (set by caller).
        error_code:       Machine-readable error code on failure.
        error_message:    Human-readable error detail.
        skipped_frames:   Frames that failed quality checks (diagnostic info).
    """
    success: bool = False
    student_id: str = ""
    encodings: list[np.ndarray] = field(default_factory=list)
    images_captured: int = 0
    saved_paths: list[Path] = field(default_factory=list)
    encoding_path: Optional[Path] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    skipped_frames: dict[str, int] = field(default_factory=dict)


# ── Service ───────────────────────────────────────────────────────────────────

class EnrollmentService:
    """Manages the multi-frame face capture and encoding pipeline for enrollment.

    Args:
        camera_service:    Manages webcam lifecycle.
        detection_service: Detects face bounding boxes per frame.
        encoding_service:  Generates and persists face encodings.
        capture_count:     Target number of validated frames. Defaults to
                           ``settings.capture_count``.
        blur_threshold:    Laplacian variance threshold for blur rejection.
        show_preview:      Whether to display an OpenCV preview window.
    """

    def __init__(
        self,
        camera_service: CameraService,
        detection_service: DetectionService,
        encoding_service: EncodingService,
        capture_count: Optional[int] = None,
        blur_threshold: float = _BLUR_THRESHOLD,
        show_preview: bool = True,
    ) -> None:
        self._camera = camera_service
        self._detector = detection_service
        self._encoder = encoding_service
        self._capture_count: int = capture_count or settings.capture_count
        self._blur_threshold = blur_threshold
        self._show_preview = show_preview
        logger.debug(
            "EnrollmentService init (capture_count=%d, blur_threshold=%.1f)",
            self._capture_count,
            self._blur_threshold,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def enroll(self, student_id: str) -> EnrollmentResult:
        """Run the full enrollment capture loop for *student_id*.

        Opens the camera, collects validated frames, generates encodings,
        saves images to disk, and returns the result.

        Args:
            student_id: Student identifier — used for directory naming.

        Returns:
            :class:`EnrollmentResult` with encodings or error details.
        """
        result = EnrollmentResult(student_id=student_id)

        if not self._camera.open():
            result.error_code = "CAMERA_ERROR"
            result.error_message = (
                f"Cannot open camera at index {settings.camera_index}"
            )
            logger.error(result.error_message)
            return result

        image_dir = get_student_image_dir(student_id, settings.upload_path)
        ensure_directories(image_dir)

        logger.info(
            "Enrollment started – student=%s target_frames=%d",
            student_id,
            self._capture_count,
        )

        try:
            self._run_capture_loop(student_id, image_dir, result)
        except Exception as exc:
            logger.exception("Unexpected error during enrollment: %s", exc)
            result.error_code = "INTERNAL_ERROR"
            result.error_message = str(exc)
        finally:
            self._camera.release()
            if self._show_preview:
                cv2.destroyAllWindows()
            logger.info(
                "Enrollment camera released – student=%s captured=%d",
                student_id,
                result.images_captured,
            )

        # ── Post-capture: generate encodings ──────────────────────────────────
        if result.images_captured > 0:
            self._generate_encodings(result)

        # ── Decide success ─────────────────────────────────────────────────────
        min_required = max(1, self._capture_count // 2)  # need at least 50%
        if len(result.encodings) >= min_required:
            result.success = True
            logger.info(
                "Enrollment successful – student=%s encodings=%d",
                student_id,
                len(result.encodings),
            )
        else:
            result.success = False
            if not result.error_code:
                result.error_code = "INSUFFICIENT_ENCODINGS"
                result.error_message = (
                    f"Only {len(result.encodings)} valid encoding(s) generated "
                    f"(minimum required: {min_required}). "
                    "Ensure the face is clearly visible and well-lit."
                )
            logger.warning(
                "Enrollment failed – student=%s encodings=%d required=%d",
                student_id,
                len(result.encodings),
                min_required,
            )

        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_capture_loop(
        self,
        student_id: str,
        image_dir: Path,
        result: EnrollmentResult,
    ) -> None:
        """Main frame-reading loop. Modifies *result* in-place.

        Args:
            student_id: Used for window title and file naming.
            image_dir:  Directory to save captured images.
            result:     EnrollmentResult to populate.
        """
        frame_counter = 0
        fps_prev_time = time.time()
        fps = 0.0
        window_title = f"Enrolling: {student_id}  (Q to cancel)"

        # Skip counters for diagnostics
        skipped: dict[str, int] = {
            "no_face": 0,
            "multiple_faces": 0,
            "too_small": 0,
            "blurry": 0,
            "frame_gap": 0,
        }

        while result.images_captured < self._capture_count:
            frame = self._camera.read_frame_with_retry(max_retries=3)
            if frame is None:
                logger.warning("Failed to read frame — skipping")
                continue

            frame_counter += 1

            # FPS calculation
            now = time.time()
            elapsed = now - fps_prev_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            fps_prev_time = now

            h, w = frame.shape[:2]
            rgb = bgr_to_rgb(frame)

            # ── Detection on half-resolution for speed ─────────────────────
            small_rgb = cv2.resize(rgb, (w // 2, h // 2))
            locations_small = self._detector.detect_faces(small_rgb)
            # Scale back to full resolution
            locations = self._detector.scale_face_locations(
                locations_small, scale=0.5
            )

            display = frame.copy()
            draw_fps(display, fps)
            draw_capture_progress(display, result.images_captured, self._capture_count)

            # ── Guard: exactly one face ────────────────────────────────────
            face_count = len(locations)

            if face_count == 0:
                skipped["no_face"] += 1
                self._overlay_status(display, "NO FACE DETECTED", RECT_COLOR_UNKNOWN)
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.warning("Enrollment cancelled by user")
                    break
                continue

            if face_count > 1:
                skipped["multiple_faces"] += 1
                self._overlay_status(
                    display,
                    f"MULTIPLE FACES ({face_count}) — only one person please",
                    RECT_COLOR_UNKNOWN,
                )
                # Draw all rectangles in red
                for top, right, bottom, left in locations:
                    cv2.rectangle(display, (left, top), (right, bottom),
                                  RECT_COLOR_UNKNOWN, 2)
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Exactly one face
            top, right, bottom, left = locations[0]
            face_h = bottom - top
            face_w = right - left
            face_area = face_h * face_w
            frame_area = h * w

            # ── Guard: face too small ──────────────────────────────────────
            if (face_area / frame_area) < _MIN_FACE_FRACTION:
                skipped["too_small"] += 1
                cv2.rectangle(display, (left, top), (right, bottom),
                              RECT_COLOR_DETECTING, 2)
                self._overlay_status(
                    display, "MOVE CLOSER — face too small", RECT_COLOR_DETECTING
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if face_h < _MIN_FACE_HEIGHT_PX:
                skipped["too_small"] += 1
                cv2.rectangle(display, (left, top), (right, bottom),
                              RECT_COLOR_DETECTING, 2)
                self._overlay_status(
                    display, "MOVE CLOSER — face height too small",
                    RECT_COLOR_DETECTING
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── Guard: blur check ──────────────────────────────────────────
            face_roi_gray = cv2.cvtColor(
                frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY
            )
            blur_score = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()

            if blur_score < self._blur_threshold:
                skipped["blurry"] += 1
                cv2.rectangle(display, (left, top), (right, bottom),
                              RECT_COLOR_DETECTING, 2)
                self._overlay_status(
                    display,
                    f"BLURRY (score={blur_score:.0f}) — hold still",
                    RECT_COLOR_DETECTING,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── Guard: frame gap (don't capture every single frame) ────────
            if frame_counter % CAPTURE_FRAME_GAP != 0:
                skipped["frame_gap"] += 1
                cv2.rectangle(display, (left, top), (right, bottom),
                              RECT_COLOR_KNOWN, 2)
                self._overlay_status(
                    display,
                    f"GOOD — capturing {result.images_captured}/{self._capture_count}",
                    RECT_COLOR_KNOWN,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── CAPTURE ────────────────────────────────────────────────────
            pose_label = POSE_LABELS[result.images_captured % len(POSE_LABELS)]
            filename = (
                f"{student_id}_{result.images_captured:03d}"
                f"_{pose_label}{IMAGE_EXTENSION}"
            )
            save_path = image_dir / filename

            try:
                save_frame(frame, save_path, quality=JPEG_QUALITY)
                result.saved_paths.append(save_path)
                result.images_captured += 1
                logger.debug(
                    "Captured frame %d/%d for student %s (blur=%.1f)",
                    result.images_captured,
                    self._capture_count,
                    student_id,
                    blur_score,
                )
            except Exception as exc:
                logger.warning("Failed to save frame: %s", exc)
                continue

            # Flash the bounding box green on capture
            cv2.rectangle(display, (left, top), (right, bottom),
                          RECT_COLOR_KNOWN, 3)
            self._overlay_status(
                display,
                f"CAPTURED {result.images_captured}/{self._capture_count}  "
                f"blur={blur_score:.0f}",
                RECT_COLOR_KNOWN,
            )
            self._show_frame(window_title, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.warning("Enrollment cancelled by user after capture")
                break

        result.skipped_frames = skipped
        logger.info(
            "Capture loop complete – student=%s frames_skipped=%s",
            student_id,
            skipped,
        )

    def _generate_encodings(self, result: EnrollmentResult) -> None:
        """Generate face encodings from all captured images in *result*.

        Reads saved JPEG files, generates encodings via EncodingService,
        and populates ``result.encodings``.

        Args:
            result: EnrollmentResult with populated ``saved_paths``.
        """
        from utils.image_utils import load_image_as_rgb

        encodings: list[np.ndarray] = []
        failed = 0

        for path in result.saved_paths:
            try:
                rgb = load_image_as_rgb(path)
                enc = self._encoder.encode_face(rgb)
                if enc is not None:
                    encodings.append(enc)
                else:
                    failed += 1
                    logger.debug("No encoding from %s", path.name)
            except Exception as exc:
                failed += 1
                logger.warning("Error encoding %s: %s", path.name, exc)

        result.encodings = encodings
        logger.info(
            "Encoding generation complete – student=%s encodings=%d failed=%d",
            result.student_id,
            len(encodings),
            failed,
        )

    # ── Display helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _overlay_status(
        frame: np.ndarray,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a status message in the bottom-left of *frame* in-place."""
        h = frame.shape[0]
        cv2.putText(
            frame,
            text,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    def _show_frame(self, window_title: str, frame: np.ndarray) -> None:
        """Display *frame* if preview is enabled."""
        if self._show_preview:
            cv2.imshow(window_title, frame)
