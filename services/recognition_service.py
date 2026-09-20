"""
services/recognition_service.py – Identify a person from a camera frame.

PIPELINE
--------
  Camera frame
    │
    ├── Detect all faces
    │     ├── 0 faces   → NO_FACE
    │     ├── >1 faces  → MULTIPLE_FACES  (do NOT attempt recognition)
    │     └── 1 face    → continue
    │
    ├── Generate encoding for the single detected face
    │
    ├── Compare against all enrolled encodings (in-memory cache)
    │
    ├── Find closest match (lowest L2 distance)
    │
    ├── distance > threshold → UNKNOWN
    │
    └── distance ≤ threshold → RECOGNIZED + register_number

IMPORTANT DESIGN NOTES
-----------------------
distance vs confidence
    The raw L2 distance between MediaPipe landmark vectors is returned
    as ``distance`` in the DTO.  It is NOT converted to a probability or
    labelled "confidence".  Lower distance = more similar face.
    Range: approximately [0, 2] for unit-normalised vectors.
    Typical same-person distance: 0.05 – 0.30.
    Typical different-person distance: 0.40 – 1.20.

threshold
    Loaded once from settings.recognition_threshold (default 0.50).
    Do NOT hardcode this anywhere.  Change it in .env:
        RECOGNITION_THRESHOLD=0.45

multiple faces
    If more than one face is detected in a frame, recognition is
    NEVER attempted.  MULTIPLE_FACES is returned immediately.
    This is a hard safety rule for the classroom attendance system.

encoding cache
    All enrolled encodings are loaded into memory on the first call to
    recognise() and cached.  Call reload_encodings() to refresh after
    new enrollments without restarting the service.

stability pass
    Instead of using the very first detected frame, the service grabs
    up to MAX_SCAN_FRAMES frames and returns the result with the LOWEST
    distance (best match).  This prevents a single blurry transitional
    frame from causing a false result.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from config import get_settings
from schemas.recognize_response import RecognizeResponseDTO
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.encoding_service import EncodingService
from utils.constants import RecognitionStatus
from utils.face_compat import face_distance
from utils.image_utils import bgr_to_rgb
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# How many frames to scan before committing to the best result.
# Higher = more stable but slower.  5 is a good classroom trade-off.
MAX_SCAN_FRAMES: int = 5

# Maximum consecutive empty-frame attempts before giving up.
MAX_EMPTY_ATTEMPTS: int = 60


class RecognitionService:
    """Identifies a student from a live camera frame.

    Args:
        camera_service:    Manages webcam lifecycle.
        detection_service: Locates face bounding boxes.
        encoding_service:  Loads / persists face encodings.
    """

    def __init__(
        self,
        camera_service: CameraService,
        detection_service: DetectionService,
        encoding_service: EncodingService,
    ) -> None:
        self._camera = camera_service
        self._detector = detection_service
        self._encoder = encoding_service
        self._threshold: float = settings.recognition_threshold

        # In-memory encoding cache: {student_id: [encoding, ...]}
        # None = not loaded yet.  {} = loaded but empty.
        self._encoding_cache: Optional[dict[str, list[np.ndarray]]] = None

        logger.debug(
            "RecognitionService init (threshold=%.2f)", self._threshold
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def reload_encodings(self) -> int:
        """Force-reload all encodings from disk into the in-memory cache.

        Call this after a new student is enrolled without restarting the server.

        Returns:
            Number of students loaded into cache.
        """
        self._encoding_cache = self._encoder.load_all_encodings()
        count = len(self._encoding_cache)
        logger.info("Encoding cache refreshed – %d student(s) loaded", count)
        return count

    def recognize_from_camera(self) -> RecognizeResponseDTO:
        """Open the camera, scan up to MAX_SCAN_FRAMES, return best result.

        Returns:
            :class:`~schemas.recognize_response.RecognizeResponseDTO`
        """
        t_start = time.perf_counter()
        logger.info("Recognition started (threshold=%.2f)", self._threshold)

        if not self._camera.open():
            logger.error("Cannot open camera for recognition")
            return RecognizeResponseDTO(
                status=RecognitionStatus.NO_FACE,
                recognition_time_ms=_elapsed_ms(t_start),
            )

        try:
            return self._scan_frames(t_start)
        finally:
            self._camera.release()
            logger.debug("Camera released after recognition")

    def recognize_from_frame(self, rgb_frame: np.ndarray) -> RecognizeResponseDTO:
        """Run recognition on an already-captured RGB frame (single best face).

        Args:
            rgb_frame: RGB numpy array (H × W × 3).

        Returns:
            :class:`~schemas.recognize_response.RecognizeResponseDTO`
        """
        t_start = time.perf_counter()
        return self._process_frame(rgb_frame, t_start)

    def recognize_multiple_from_frame(
        self, rgb_frame: np.ndarray
    ) -> list[RecognizeResponseDTO]:
        """Detect ALL faces in *rgb_frame* and recognise each one.

        Each face is processed independently.  Results are returned in the
        same order as the detected face locations.

        Args:
            rgb_frame: RGB numpy array (H × W × 3).

        Returns:
            List of :class:`~schemas.recognize_response.RecognizeResponseDTO`
            (one per detected face, empty list if no faces found).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.face_compat import face_locations as _face_locations

        t_start = time.perf_counter()

        # ── Detect every face in the frame ────────────────────────────────────
        locations = _face_locations(rgb_frame, number_of_times_to_upsample=1)
        if not locations:
            logger.info("recognize_multiple: no faces detected in frame")
            return []

        logger.info("recognize_multiple: detected %d face(s) — processing each", len(locations))

        # Ensure encoding cache is warm before parallel work
        if self._encoding_cache is None:
            self.reload_encodings()

        # ── Recognise each face (parallel for speed) ─────────────────────────
        results: list[RecognizeResponseDTO] = [None] * len(locations)  # type: ignore[list-item]

        def _process(idx: int, loc: tuple) -> tuple[int, RecognizeResponseDTO]:
            dto = self._process_frame(rgb_frame, time.perf_counter(), face_location=loc)
            return idx, dto

        with ThreadPoolExecutor(max_workers=min(len(locations), 4)) as pool:
            futures = {pool.submit(_process, i, loc): i for i, loc in enumerate(locations)}
            for fut in as_completed(futures):
                try:
                    idx, dto = fut.result()
                    results[idx] = dto
                except Exception as exc:
                    i = futures[fut]
                    logger.error("recognize_multiple face[%d] error: %s", i, exc)
                    results[i] = RecognizeResponseDTO(
                        status=RecognitionStatus.NO_FACE,
                        recognition_time_ms=_elapsed_ms(t_start),
                    )

        logger.info(
            "recognize_multiple: done in %.0f ms — statuses: %s",
            _elapsed_ms(t_start),
            [r.status.value if r else "?" for r in results],
        )
        return results



    # ── Internal ──────────────────────────────────────────────────────────────

    def _scan_frames(self, t_start: float) -> RecognizeResponseDTO:
        """Read up to MAX_SCAN_FRAMES valid frames and return the best result.

        "Best" = the frame whose recognised encoding has the lowest distance
        to any enrolled student.

        Args:
            t_start: perf_counter timestamp at recognition start.

        Returns:
            RecognizeResponseDTO with the best outcome found.
        """
        good_frames: int = 0
        empty_attempts: int = 0

        # Collect candidates: (distance, dto) — we keep the lowest distance.
        best_dto: Optional[RecognizeResponseDTO] = None
        best_distance: float = float("inf")

        while good_frames < MAX_SCAN_FRAMES:
            frame = self._camera.read_frame_with_retry(max_retries=2)
            if frame is None:
                empty_attempts += 1
                if empty_attempts >= MAX_EMPTY_ATTEMPTS:
                    logger.error("Camera read failed %d times — aborting", empty_attempts)
                    break
                continue

            # Downscale BGR for detection speed, keep original for encoding
            h, w = frame.shape[:2]
            small_bgr = cv2.resize(frame, (w // 2, h // 2))
            small_rgb = bgr_to_rgb(small_bgr)
            locations_small = self._detector.detect_faces(small_rgb)
            face_count = len(locations_small)

            # ── MULTIPLE FACES: hard stop, return immediately ──────────────
            if face_count > 1:
                logger.warning(
                    "Multiple faces detected (%d) — recognition aborted", face_count
                )
                return RecognizeResponseDTO(
                    status=RecognitionStatus.MULTIPLE_FACES,
                    recognition_time_ms=_elapsed_ms(t_start),
                )

            # ── NO FACE: keep scanning ─────────────────────────────────────
            if face_count == 0:
                empty_attempts += 1
                continue

            # ── Exactly one face ───────────────────────────────────────────
            good_frames += 1
            locations_full = self._detector.scale_face_locations(
                locations_small, scale=0.5
            )
            rgb_full = bgr_to_rgb(frame)
            dto = self._process_frame(
                rgb_full, t_start, face_location=locations_full[0]
            )

            # Track the candidate with the lowest distance
            if dto.distance is not None and dto.distance < best_distance:
                best_distance = dto.distance
                best_dto = dto

            # Early exit: if we already have a very confident match, stop
            if best_distance < (self._threshold * 0.7):
                logger.debug(
                    "Early exit at frame %d — distance=%.4f", good_frames, best_distance
                )
                break

        # ── Resolve final result ───────────────────────────────────────────
        if best_dto is not None:
            logger.info(
                "Recognition result: status=%s student_id=%s distance=%s",
                best_dto.status,
                best_dto.student_id,
                f"{best_dto.distance:.4f}" if best_dto.distance is not None else "N/A",
            )
            return best_dto

        # All frames had no face
        logger.warning("No face detected across all scanned frames")
        return RecognizeResponseDTO(
            status=RecognitionStatus.NO_FACE,
            recognition_time_ms=_elapsed_ms(t_start),
        )

    def _process_frame(
        self,
        rgb_frame: np.ndarray,
        t_start: float,
        face_location: Optional[tuple[int, int, int, int]] = None,
    ) -> RecognizeResponseDTO:
        """Core matching logic for a single RGB frame.

        Args:
            rgb_frame:     Full-resolution RGB frame.
            t_start:       perf_counter timestamp at recognition start.
            face_location: Pre-computed (top, right, bottom, left) or None.

        Returns:
            RecognizeResponseDTO with distance (not a fake confidence).
        """
        # Generate encoding for the face
        encoding = self._encoder.encode_face(rgb_frame, face_location)
        if encoding is None:
            logger.warning("Could not generate encoding from frame")
            return RecognizeResponseDTO(
                status=RecognitionStatus.NO_FACE,
                recognition_time_ms=_elapsed_ms(t_start),
            )

        # Ensure cache is populated
        if self._encoding_cache is None:
            self.reload_encodings()

        known_data = self._encoding_cache  # type: ignore[assignment]
        if not known_data:
            logger.warning("No enrolled encodings in cache")
            return RecognizeResponseDTO(
                status=RecognitionStatus.UNKNOWN,
                recognition_time_ms=_elapsed_ms(t_start),
            )

        # Flatten for vectorised distance computation
        flat_encodings: list[np.ndarray] = []
        flat_ids: list[str] = []
        for sid, encs in known_data.items():
            for enc in encs:
                flat_encodings.append(enc)
                flat_ids.append(sid)

        distances = face_distance(flat_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        best_student_id = flat_ids[best_idx]

        logger.debug(
            "Best match: student=%s  distance=%.4f  threshold=%.4f",
            best_student_id,
            best_distance,
            self._threshold,
        )

        # Threshold check
        if best_distance > self._threshold:
            return RecognizeResponseDTO(
                status=RecognitionStatus.UNKNOWN,
                distance=best_distance,
                recognition_time_ms=_elapsed_ms(t_start),
            )

        return RecognizeResponseDTO(
            student_id=best_student_id,
            distance=best_distance,
            status=RecognitionStatus.RECOGNIZED,
            recognition_time_ms=_elapsed_ms(t_start),
        )


# ── Module-level helper ───────────────────────────────────────────────────────

def _elapsed_ms(t_start: float) -> float:
    """Return milliseconds elapsed since *t_start*."""
    return (time.perf_counter() - t_start) * 1_000
