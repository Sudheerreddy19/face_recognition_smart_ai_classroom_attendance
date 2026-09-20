"""
services/encoding_service.py – Generate and persist face encodings.

Responsibilities:
  • Produce 128-D face encodings from RGB image data.
  • Aggregate encodings from multiple training images into a single pickle.
  • Load encodings from disk for recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config import get_settings
from utils.face_compat import face_encodings
from utils.file_utils import get_encoding_path, load_pickle, save_pickle
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EncodingService:
    """Generates and manages 128-dimensional face encodings.

    Each encoding is produced by :func:`face_recognition.face_encodings` and
    stored in a pickle file keyed by *student_id*.
    """

    def __init__(self) -> None:
        self._encoding_dir: Path = settings.encoding_path
        self._encoding_dir.mkdir(parents=True, exist_ok=True)
        self._jitters: int = settings.encoding_jitters
        logger.debug(
            "EncodingService initialised (dir=%s, jitters=%d)",
            self._encoding_dir,
            self._jitters,
        )

    # ── Encoding Generation ───────────────────────────────────────────────────

    def encode_face(
        self,
        rgb_image: np.ndarray,
        face_location: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        """Generate a 128-D face encoding from *rgb_image*.

        Args:
            rgb_image:     RGB numpy array containing exactly one face.
            face_location: Pre-computed ``(top, right, bottom, left)`` tuple.
                           Passing it avoids a redundant detection step.

        Returns:
            128-D numpy array encoding, or ``None`` if no face found.
        """
        known_locations = [face_location] if face_location else None
        try:
            encodings = face_encodings(
                rgb_image,
                known_face_locations=known_locations,
                num_jitters=self._jitters,
            )
            if not encodings:
                logger.warning("face_encodings returned empty list")
                return None
            return encodings[0]
        except Exception as exc:
            logger.exception("Error generating face encoding: %s", exc)
            return None

    def encode_all_faces(
        self, rgb_image: np.ndarray
    ) -> list[np.ndarray]:
        """Generate encodings for every face found in *rgb_image*.

        Args:
            rgb_image: RGB numpy array.

        Returns:
            List of 128-D encoding arrays (one per detected face).
        """
        try:
            return face_encodings(rgb_image, num_jitters=self._jitters)
        except Exception as exc:
            logger.exception("Error generating batch face encodings: %s", exc)
            return []

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_encodings(
        self, student_id: str, encodings: list[np.ndarray]
    ) -> Path:
        """Persist *encodings* for *student_id* to a pickle file.

        If a pickle already exists for this student, it is overwritten.

        Args:
            student_id: Student identifier used as the file stem.
            encodings:  List of 128-D encoding arrays.

        Returns:
            Path to the saved pickle file.
        """
        path = get_encoding_path(student_id, self._encoding_dir)
        save_pickle(encodings, path)
        logger.info(
            "Saved %d encoding(s) for student %s → %s",
            len(encodings),
            student_id,
            path,
        )
        return path

    def load_encodings(self, student_id: str) -> list[np.ndarray]:
        """Load persisted encodings for *student_id*.

        Args:
            student_id: Student identifier.

        Returns:
            List of 128-D numpy arrays, or an empty list if none found.
        """
        path = get_encoding_path(student_id, self._encoding_dir)
        if not path.exists():
            logger.warning("No encoding file found for student %s", student_id)
            return []
        try:
            encodings: list[np.ndarray] = load_pickle(path)
            logger.debug(
                "Loaded %d encoding(s) for student %s", len(encodings), student_id
            )
            return encodings
        except Exception as exc:
            logger.exception(
                "Failed to load encodings for student %s: %s", student_id, exc
            )
            return []

    def load_all_encodings(
        self,
    ) -> dict[str, list[np.ndarray]]:
        """Load encodings for every student that has a pickle file.

        Returns:
            Mapping of ``student_id → [encoding, …]``.
        """
        result: dict[str, list[np.ndarray]] = {}
        for pickle_path in self._encoding_dir.glob("*.pickle"):
            student_id = pickle_path.stem
            encodings = self.load_encodings(student_id)
            if encodings:
                result[student_id] = encodings
        logger.debug(
            "Loaded encodings for %d student(s) from disk", len(result)
        )
        return result

    def has_encoding(self, student_id: str) -> bool:
        """Return ``True`` if a pickle file exists for *student_id*.

        Args:
            student_id: Student identifier.
        """
        path = get_encoding_path(student_id, self._encoding_dir)
        return path.exists()

    def delete_encoding(self, student_id: str) -> bool:
        """Delete the pickle file for *student_id*.

        Args:
            student_id: Student identifier.

        Returns:
            ``True`` if deleted, ``False`` if file did not exist.
        """
        path = get_encoding_path(student_id, self._encoding_dir)
        if path.exists():
            path.unlink()
            logger.info("Deleted encoding for student %s", student_id)
            return True
        return False

    def list_encoding_infos(self) -> list[dict]:
        """Return metadata for every encoding file on disk.

        Returns:
            List of dicts with keys: ``student_id``, ``encoding_file``,
            ``face_count``, ``file_size_bytes``.
        """
        infos: list[dict] = []
        for pickle_path in sorted(self._encoding_dir.glob("*.pickle")):
            student_id = pickle_path.stem
            try:
                encodings: list[np.ndarray] = load_pickle(pickle_path)
                count = len(encodings)
            except Exception:
                count = 0
            infos.append(
                {
                    "student_id": student_id,
                    "encoding_file": pickle_path.name,
                    "face_count": count,
                    "file_size_bytes": pickle_path.stat().st_size,
                }
            )
        return infos
