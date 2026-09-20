"""
utils/face_compat.py – Face detection and encoding using face_recognition (dlib).

WHY THIS EXISTS
---------------
Provides a single place to call face_recognition (dlib) with numpy 2.x
and dlib 20.x safety. All callers (detection_service, encoding_service,
recognition_service) import from here.

PUBLIC API
----------
  face_locations(img, ...)        -> list[tuple[top, right, bottom, left]]
  face_encodings(img, ...)        -> list[np.ndarray]   (128-D dlib vectors)
  face_distance(encodings, query) -> np.ndarray          (L2 distances)
  compare_faces(known, query, tolerance) -> list[bool]

NUMPY 2.x + DLIB 20.x NOTES
-----------------------------
- Always pass np.ascontiguousarray(img, dtype=np.uint8) to dlib.
- dlib 20.x encodings have norm ~1.0-1.5 (not exactly 1.0 like dlib 19.x).
- face_distance uses raw L2 norm — works correctly regardless of encoding norm.

THRESHOLD
---------
Default recognition threshold in config.py is 0.50 (L2 distance).
- Distance < 0.50 → same person (RECOGNIZED)
- Distance >= 0.50 → different person (UNKNOWN)
Lower threshold = stricter matching. Start at 0.50 and tune as needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ── face_recognition import guard ─────────────────────────────────────────────

try:
    import face_recognition as _fr
    _FR_AVAILABLE = True
    logger.debug("face_recognition (dlib) loaded successfully")
except ImportError:
    _FR_AVAILABLE = False
    logger.error(
        "face_recognition is not installed. "
        "Activate .venv and run: pip install face-recognition"
    )

# ── Type alias ────────────────────────────────────────────────────────────────

# (top, right, bottom, left) — standard face_recognition convention
FaceLocation = tuple[int, int, int, int]


# ── Public API ────────────────────────────────────────────────────────────────


def face_locations(
    img: np.ndarray,
    number_of_times_to_upsample: int = 1,
    model: str = "hog",
) -> list[FaceLocation]:
    """Detect face bounding boxes in *img* (RGB).

    Args:
        img:                         RGB numpy array (H x W x 3, uint8).
        number_of_times_to_upsample: Upsampling passes for small faces.
        model:                       ``"hog"`` (fast, CPU) or ``"cnn"`` (accurate, GPU).

    Returns:
        List of ``(top, right, bottom, left)`` tuples.
        Empty list if no faces found or face_recognition is unavailable.
    """
    if not _FR_AVAILABLE:
        logger.error("face_locations: face_recognition not available")
        return []

    # dlib 20.x requires C-contiguous uint8 arrays
    img = np.ascontiguousarray(img, dtype=np.uint8)

    try:
        locations = _fr.face_locations(
            img,
            number_of_times_to_upsample=number_of_times_to_upsample,
            model=model,
        )
        logger.debug("face_locations: detected %d face(s)", len(locations))
        return locations
    except Exception as exc:
        logger.exception("face_locations error: %s", exc)
        return []


def face_encodings(
    face_image: np.ndarray,
    known_face_locations: Optional[list[FaceLocation]] = None,
    num_jitters: int = 1,
    model: str = "small",
) -> list[np.ndarray]:
    """Generate 128-D face encoding(s) from *face_image* (RGB).

    Args:
        face_image:           RGB numpy array.
        known_face_locations: Optional pre-computed bounding boxes.
                              Skips detection step if provided.
        num_jitters:          Re-sampling count for accuracy (1 = fast).
        model:                ``"small"`` (fast) or ``"large"`` (accurate).

    Returns:
        List of 128-D numpy float64 arrays (one per detected face).
    """
    if not _FR_AVAILABLE:
        logger.error("face_encodings: face_recognition not available")
        return []

    face_image = np.ascontiguousarray(face_image, dtype=np.uint8)

    try:
        encodings = _fr.face_encodings(
            face_image,
            known_face_locations=known_face_locations,
            num_jitters=num_jitters,
            model=model,
        )
        logger.debug("face_encodings: generated %d encoding(s)", len(encodings))
        return encodings
    except Exception as exc:
        logger.exception("face_encodings error: %s", exc)
        return []


def face_distance(
    face_encodings_list: list[np.ndarray],
    face_to_compare: np.ndarray,
) -> np.ndarray:
    """Compute L2 distance between *face_to_compare* and each known encoding.

    Args:
        face_encodings_list: List of known 128-D face encodings.
        face_to_compare:     128-D encoding to compare against.

    Returns:
        NumPy array of L2 distances (one per known encoding).
        Lower distance = more similar faces.
    """
    if not face_encodings_list:
        return np.array([])

    try:
        known = np.array(face_encodings_list, dtype=np.float64)
        query = np.array(face_to_compare, dtype=np.float64)
        return np.linalg.norm(known - query, axis=1)
    except Exception as exc:
        logger.exception("face_distance error: %s", exc)
        return np.array([float("inf")] * len(face_encodings_list))


def compare_faces(
    known_face_encodings: list[np.ndarray],
    face_encoding_to_check: np.ndarray,
    tolerance: float = 0.6,
) -> list[bool]:
    """Return True for each known encoding within *tolerance* L2 distance.

    Args:
        known_face_encodings:   List of 128-D dlib encodings.
        face_encoding_to_check: 128-D encoding to test.
        tolerance:              Maximum L2 distance for a match (default 0.6).

    Returns:
        List of booleans -- True where distance <= tolerance.
    """
    distances = face_distance(known_face_encodings, face_encoding_to_check)
    return list(distances <= tolerance)
