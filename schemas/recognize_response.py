"""
schemas/recognize_response.py – Internal DTO produced by RecognitionService.

IMPORTANT: distance vs confidence
----------------------------------
``distance`` is the raw L2 distance between the query face encoding and the
closest enrolled encoding.  It is NOT a probability or confidence score.

  distance ≈ 0.00 – 0.20  →  very strong match (same person, good lighting)
  distance ≈ 0.20 – 0.40  →  likely match
  distance ≈ 0.40 – 0.55  →  borderline (threshold region)
  distance ≈ 0.55 – 2.00  →  no match / different person

The threshold in config (RECOGNITION_THRESHOLD, default 0.50) is compared
against this distance value directly.  Tune it in .env after first enrollment.

The public-facing API response (api/recognize.py) passes ``distance`` through
to the JSON response as-is, clearly documented as a distance, not a probability.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from utils.constants import RecognitionStatus


class RecognizeResponseDTO(BaseModel):
    """Internal DTO produced by RecognitionService.

    Attributes:
        student_id:          Matched student register number, or None.
        distance:            Raw L2 distance to the closest enrolled encoding.
                             Lower is more similar.  None if no comparison made.
        status:              RecognitionStatus enum value.
        recognition_time_ms: Wall-clock time for the full recognition pass (ms).
    """

    student_id: Optional[str] = None
    distance: Optional[float] = None
    status: RecognitionStatus = RecognitionStatus.UNKNOWN
    recognition_time_ms: float = 0.0
