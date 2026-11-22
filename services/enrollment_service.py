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
_BLUR_THRESHOLD: float = 30.0

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
        """Main frame-reading loop with Face Alignment HUD. Modifies *result* in-place."""
        frame_counter = 0
        fps_prev_time = time.time()
        fps = 0.0
        window_title = f"Class AI Biometrics — Enrolling: {student_id}"

        if self._show_preview:
            try:
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_title, 640, 480)
                cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
                if hwnd:
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                    ctypes.windll.user32.ShowWindow(hwnd, 5)
            except Exception as win_err:
                logger.debug("Window setup notice: %s", win_err)

        skipped: dict[str, int] = {
            "no_face": 0,
            "multiple_faces": 0,
            "too_small": 0,
            "blurry": 0,
            "frame_gap": 0,
            "not_aligned": 0,
        }

        loop_start_time = time.time()
        max_duration = getattr(settings, "capture_job_timeout", 90) - 15  # 75s

        while result.images_captured < self._capture_count:
            if time.time() - loop_start_time > max_duration:
                logger.warning(
                    "Enrollment capture timed out after %.1fs (captured %d/%d frames)",
                    time.time() - loop_start_time,
                    result.images_captured,
                    self._capture_count,
                )
                result.error_code = "CAPTURE_TIMEOUT"
                result.error_message = (
                    f"Face capture timed out after {int(max_duration)}s. "
                    "Ensure student looks directly into the camera inside the green guide."
                )
                break

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

            # Detection on half-resolution for speed
            small_rgb = cv2.resize(rgb, (w // 2, h // 2))
            locations_small = self._detector.detect_faces(small_rgb)
            locations = self._detector.scale_face_locations(locations_small, scale=0.5)

            display = frame.copy()
            face_count = len(locations)

            # ── Guard: No face detected ─────────────────────────────────────
            if face_count == 0:
                skipped["no_face"] += 1
                self._draw_alignment_hud(
                    display, student_id, locations,
                    status_text="POSITION FACE INSIDE OVAL GUIDE",
                    status_color=(200, 200, 200),
                    oval_color=(180, 180, 180),
                    captured=result.images_captured,
                    target=self._capture_count,
                    fps=fps,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.warning("Enrollment cancelled by user")
                    break
                continue

            # ── Guard: Multiple faces ───────────────────────────────────────
            if face_count > 1:
                skipped["multiple_faces"] += 1
                self._draw_alignment_hud(
                    display, student_id, locations,
                    status_text=f"MULTIPLE FACES ({face_count}) — ONLY 1 PERSON",
                    status_color=(0, 0, 255),
                    oval_color=(0, 0, 255),
                    captured=result.images_captured,
                    target=self._capture_count,
                    fps=fps,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── Exactly one face — verify alignment ─────────────────────────
            top, right, bottom, left = locations[0]
            face_cx = (left + right) // 2
            face_cy = (top + bottom) // 2
            face_h = bottom - top
            guide_cx, guide_cy = w // 2, int(h * 0.48)

            dx = face_cx - guide_cx
            dy = face_cy - guide_cy

            # Check horizontal centering
            if abs(dx) > 65:
                skipped["not_aligned"] += 1
                msg = "MOVE LEFT ←" if dx > 0 else "MOVE RIGHT →"
                self._draw_alignment_hud(
                    display, student_id, locations,
                    status_text=f"ALIGNMENT: {msg}",
                    status_color=(0, 165, 255),
                    oval_color=(0, 165, 255),
                    captured=result.images_captured,
                    target=self._capture_count,
                    fps=fps,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Check vertical centering
            if abs(dy) > 65:
                skipped["not_aligned"] += 1
                msg = "MOVE UP ↑" if dy > 0 else "MOVE DOWN ↓"
                self._draw_alignment_hud(
                    display, student_id, locations,
                    status_text=f"ALIGNMENT: {msg}",
                    status_color=(0, 165, 255),
                    oval_color=(0, 165, 255),
                    captured=result.images_captured,
                    target=self._capture_count,
                    fps=fps,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Check distance (face size)
            if face_h < 120:
                skipped["too_small"] += 1
                self._draw_alignment_hud(
                    display, student_id, locations,
                    status_text="ALIGNMENT: MOVE CLOSER TO CAMERA",
                    status_color=(0, 215, 255),
                    oval_color=(0, 215, 255),
                    captured=result.images_captured,
                    target=self._capture_count,
                    fps=fps,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if face_h > 360:
                self._draw_alignment_hud(
                    display, student_id, locations,
                    status_text="ALIGNMENT: MOVE BACK SLIGHTLY",
                    status_color=(0, 215, 255),
                    oval_color=(0, 215, 255),
                    captured=result.images_captured,
                    target=self._capture_count,
                    fps=fps,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Check blur
            face_roi_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()
            if blur_score < self._blur_threshold:
                skipped["blurry"] += 1
                self._draw_alignment_hud(
                    display, student_id, locations,
                    status_text=f"HOLD STILL (Blur={blur_score:.0f})",
                    status_color=(0, 215, 255),
                    oval_color=(0, 215, 255),
                    captured=result.images_captured,
                    target=self._capture_count,
                    fps=fps,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Face is ALIGNED and SHARP — apply frame gap
            if frame_counter % CAPTURE_FRAME_GAP != 0:
                skipped["frame_gap"] += 1
                self._draw_alignment_hud(
                    display, student_id, locations,
                    status_text=f"FACE ALIGNED ✓ CAPTURING ({result.images_captured}/{self._capture_count})",
                    status_color=(34, 197, 94),
                    oval_color=(34, 197, 94),
                    captured=result.images_captured,
                    target=self._capture_count,
                    fps=fps,
                )
                self._show_frame(window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── CAPTURE FRAME ───────────────────────────────────────────────
            pose_label = POSE_LABELS[result.images_captured % len(POSE_LABELS)]
            filename = f"{student_id}_{result.images_captured:03d}_{pose_label}{IMAGE_EXTENSION}"
            save_path = image_dir / filename
            try:
                save_frame(frame, save_path, quality=JPEG_QUALITY)
                result.saved_paths.append(save_path)
                result.images_captured += 1
                logger.debug(
                    "Captured frame %d/%d for student %s (blur=%.1f)",
                    result.images_captured, self._capture_count, student_id, blur_score,
                )
            except Exception as exc:
                logger.warning("Failed to save frame: %s", exc)
                continue

            self._draw_alignment_hud(
                display, student_id, locations,
                status_text=f"CAPTURED {result.images_captured}/{self._capture_count} ✓ HOLD STILL",
                status_color=(34, 197, 94),
                oval_color=(34, 197, 94),
                captured=result.images_captured,
                target=self._capture_count,
                fps=fps,
            )
            self._show_frame(window_title, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        result.skipped_frames = skipped
        logger.info(
            "Capture loop complete — student=%s frames_skipped=%s",
            student_id, skipped,
        )

    def _draw_alignment_hud(
        self,
        frame: np.ndarray,
        student_id: str,
        locations: list[tuple[int, int, int, int]],
        status_text: str,
        status_color: tuple[int, int, int],
        oval_color: tuple[int, int, int],
        captured: int,
        target: int,
        fps: float = 0.0,
    ) -> None:
        """Draw biometric face alignment HUD overlay on *frame*."""
        h, w = frame.shape[:2]
        guide_center = (w // 2, int(h * 0.48))  # (320, 230)
        guide_axes   = (115, 155)

        # 1. Top dark banner
        cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"CLASS AI BIOMETRICS | Student: {student_id}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        if fps > 0:
            cv2.putText(
                frame,
                f"{int(fps)} FPS",
                (w - 85, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (160, 160, 160),
                1,
            )

        # 2. Draw face box with corner accents if 1 face detected
        if len(locations) == 1:
            top, right, bottom, left = locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), oval_color, 2)
            corner_len = 16
            cv2.line(frame, (left, top), (left + corner_len, top), (255, 255, 255), 3)
            cv2.line(frame, (left, top), (left, top + corner_len), (255, 255, 255), 3)
            cv2.line(frame, (right, top), (right - corner_len, top), (255, 255, 255), 3)
            cv2.line(frame, (right, top), (right, top + corner_len), (255, 255, 255), 3)
            cv2.line(frame, (left, bottom), (left + corner_len, bottom), (255, 255, 255), 3)
            cv2.line(frame, (left, bottom), (left, bottom - corner_len), (255, 255, 255), 3)
            cv2.line(frame, (right, bottom), (right - corner_len, bottom), (255, 255, 255), 3)
            cv2.line(frame, (right, bottom), (right, bottom - corner_len), (255, 255, 255), 3)

        # 3. Biometric guide oval
        cv2.ellipse(frame, guide_center, guide_axes, 0, 0, 360, oval_color, 3)
        cx, cy = guide_center
        ax, ay = guide_axes
        cv2.line(frame, (cx - 12, cy - ay), (cx + 12, cy - ay), (255, 255, 255), 2)
        cv2.line(frame, (cx - 12, cy + ay), (cx + 12, cy + ay), (255, 255, 255), 2)
        cv2.line(frame, (cx - ax, cy - 12), (cx - ax, cy + 12), (255, 255, 255), 2)
        cv2.line(frame, (cx + ax, cy - 12), (cx + ax, cy + 12), (255, 255, 255), 2)

        # 4. Bottom status bar
        cv2.rectangle(frame, (0, h - 50), (w, h), (18, 18, 18), -1)
        cv2.putText(
            frame,
            status_text,
            (20, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            status_color,
            2,
        )

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = w - 210, h - 34, 190, 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        progress = min(1.0, captured / max(1, target))
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (34, 197, 94), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
        cv2.putText(
            frame,
            f"{captured}/{target}",
            (bar_x + 72, bar_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )

    def _overlay_status(self, frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        """Deprecated compatibility overlay."""
        pass

    def _show_frame(self, window_title: str, frame: np.ndarray) -> None:
        """Display *frame* if preview is enabled."""
        if self._show_preview:
            try:
                cv2.imshow(window_title, frame)
            except Exception:
                pass

    def _generate_encodings(self, result: EnrollmentResult) -> None:
        """Generate face encodings in parallel from captured images.

        Uses a ThreadPoolExecutor to encode frames concurrently across CPU cores
        for fast completion within 5 to 10 seconds.
        """
        from concurrent.futures import ThreadPoolExecutor
        from utils.image_utils import load_image_as_rgb

        def _encode_single(path):
            try:
                rgb = load_image_as_rgb(path)
                return self._encoder.encode_face(rgb)
            except Exception as exc:
                logger.warning("Error encoding %s: %s", path.name, exc)
                return None

        paths = list(result.saved_paths)
        if not paths:
            result.encodings = []
            return

        with ThreadPoolExecutor(max_workers=min(len(paths), 4)) as pool:
            raw_results = list(pool.map(_encode_single, paths))

        encodings = [enc for enc in raw_results if enc is not None]
        failed = len(paths) - len(encodings)

        result.encodings = encodings
        logger.info(
            "Encoding generation complete (parallel) – student=%s encodings=%d failed=%d",
            result.student_id,
            len(encodings),
            failed,
        )
