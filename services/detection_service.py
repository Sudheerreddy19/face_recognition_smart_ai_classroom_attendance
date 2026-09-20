"""
services/detection_service.py – Face detection using face_recognition library.

Responsibilities:
  • Locate face bounding boxes in an RGB frame.
  • Validate detection results (count, size, etc.).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import get_settings
from utils.constants import (
    MAX_FACES_REGISTRATION,
    MSG_MULTIPLE_FACES,
    MSG_NO_FACE_FOUND,
)
from utils.face_compat import face_locations as _face_locations
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Type alias for a face location tuple (top, right, bottom, left)
FaceLocation = tuple[int, int, int, int]


class DetectionService:
    """Detects face bounding boxes in RGB image frames.

    Uses the face_recognition library which wraps dlib's HOG / CNN detector.
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialise the detection service.

        Args:
            model: Detection model – ``"hog"`` (CPU, fast) or ``"cnn"``
                   (GPU, accurate).  Defaults to ``settings.detection_model``.
        """
        self.model: str = model or settings.detection_model
        logger.debug("DetectionService initialised (model=%s)", self.model)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_faces(self, rgb_frame: np.ndarray) -> list[FaceLocation]:
        """Return bounding boxes for all faces in *rgb_frame*.

        Args:
            rgb_frame: RGB numpy array.

        Returns:
            List of ``(top, right, bottom, left)`` tuples – one per face.
        """
        try:
            locations: list[FaceLocation] = _face_locations(
                rgb_frame, model=self.model
            )
            logger.debug("Detected %d face(s)", len(locations))
            return locations
        except Exception as exc:
            logger.exception("Error during face detection: %s", exc)
            return []

    def detect_single_face(
        self, rgb_frame: np.ndarray
    ) -> tuple[Optional[FaceLocation], Optional[str]]:
        """Detect exactly one face; return an error message if not exactly one.

        Args:
            rgb_frame: RGB numpy array.

        Returns:
            Tuple of ``(face_location, error_message)``.
            On success: ``(location, None)``.
            On failure: ``(None, error_message)``.
        """
        locations = self.detect_faces(rgb_frame)

        if len(locations) == 0:
            logger.warning("No face detected in frame")
            return None, MSG_NO_FACE_FOUND

        if len(locations) > MAX_FACES_REGISTRATION:
            logger.warning("Multiple faces detected (%d)", len(locations))
            return None, MSG_MULTIPLE_FACES

        return locations[0], None

    def has_face(self, rgb_frame: np.ndarray) -> bool:
        """Return ``True`` if at least one face is found in *rgb_frame*.

        Args:
            rgb_frame: RGB numpy array.
        """
        return len(self.detect_faces(rgb_frame)) > 0

    def scale_face_locations(
        self, locations: list[FaceLocation], scale: float
    ) -> list[FaceLocation]:
        """Scale face bounding boxes back to full resolution.

        When frames are downscaled for faster recognition, detected locations
        must be scaled back before drawing on the original frame.

        Args:
            locations: Bounding boxes from a down-scaled frame.
            scale:     Scale factor used for downscaling (e.g. 0.5 for half).

        Returns:
            Scaled bounding boxes corresponding to the original frame.
        """
        inv = 1.0 / scale
        scaled: list[FaceLocation] = []
        for top, right, bottom, left in locations:
            scaled.append(
                (
                    int(top * inv),
                    int(right * inv),
                    int(bottom * inv),
                    int(left * inv),
                )
            )
        return scaled
