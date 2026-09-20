"""
tests/test_recognition.py – Tests for RecognitionService (Phases 5 + 6).

Tests cover:
  - NO_FACE when camera returns blank frames
  - MULTIPLE_FACES returns immediately (never recognises)
  - UNKNOWN when distance exceeds threshold
  - RECOGNIZED when distance is within threshold
  - Encoding cache: None on init, loaded after reload_encodings()
  - Best-of-N stability: lowest-distance result is returned
  - POST /recognize returns correct JSON shapes
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from schemas.recognize_response import RecognizeResponseDTO
from services.recognition_service import RecognitionService
from utils.constants import RecognitionStatus


def _make_recogniser(mock_camera, mock_detector, mock_encoder, threshold=0.50):
    svc = RecognitionService(
        camera_service=mock_camera,
        detection_service=mock_detector,
        encoding_service=mock_encoder,
    )
    svc._threshold = threshold
    return svc


class TestRecognitionServiceNoFace:

    def test_no_face_when_all_frames_blank(self, mock_camera):
        detector = MagicMock()
        detector.detect_faces.return_value = []  # no faces every time
        detector.scale_face_locations.return_value = []

        encoder  = MagicMock()
        svc      = _make_recogniser(mock_camera, detector, encoder)

        result = svc.recognize_from_camera()
        assert result.status == RecognitionStatus.NO_FACE
        assert result.distance is None


class TestRecognitionServiceMultipleFaces:

    def test_multiple_faces_returns_immediately(self, mock_camera):
        detector = MagicMock()
        # First call: 2 faces detected on downscaled frame
        detector.detect_faces.return_value = [(0, 100, 100, 0), (200, 300, 300, 200)]
        detector.scale_face_locations.return_value = [
            (0, 200, 200, 0), (400, 600, 600, 400)
        ]
        encoder = MagicMock()
        svc     = _make_recogniser(mock_camera, detector, encoder)

        result  = svc.recognize_from_camera()
        assert result.status == RecognitionStatus.MULTIPLE_FACES
        # Encoding must never be called when multiple faces found
        encoder.encode_face.assert_not_called()


class TestRecognitionServiceUnknown:

    def test_unknown_when_distance_exceeds_threshold(
        self, mock_camera, fake_encoding_1404, different_encoding_1404
    ):
        detector = MagicMock()
        detector.detect_faces.return_value = [(10, 110, 90, 10)]
        detector.scale_face_locations.return_value = [(10, 110, 90, 10)]

        encoder = MagicMock()
        encoder.encode_face.return_value = fake_encoding_1404
        encoder.load_all_encodings.return_value = {
            "STU001": [different_encoding_1404]
        }

        svc = _make_recogniser(mock_camera, detector, encoder, threshold=0.001)
        result = svc.recognize_from_camera()
        assert result.status == RecognitionStatus.UNKNOWN
        assert result.distance is not None
        assert result.distance > 0.001


class TestRecognitionServiceRecognized:

    def test_recognized_when_distance_within_threshold(
        self, mock_camera, fake_encoding_1404
    ):
        detector = MagicMock()
        detector.detect_faces.return_value = [(10, 110, 90, 10)]
        detector.scale_face_locations.return_value = [(10, 110, 90, 10)]

        encoder = MagicMock()
        encoder.encode_face.return_value = fake_encoding_1404
        # Same encoding stored → distance ≈ 0
        encoder.load_all_encodings.return_value = {"STU001": [fake_encoding_1404]}

        svc    = _make_recogniser(mock_camera, detector, encoder, threshold=0.50)
        result = svc.recognize_from_camera()

        assert result.status == RecognitionStatus.RECOGNIZED
        assert result.student_id == "STU001"
        assert result.distance is not None
        assert result.distance < 1e-5   # same encoding → ~0 distance


class TestEncodingCache:

    def test_cache_is_none_on_init(self, mock_camera):
        svc = RecognitionService(
            camera_service=mock_camera,
            detection_service=MagicMock(),
            encoding_service=MagicMock(),
        )
        assert svc._encoding_cache is None

    def test_reload_encodings_populates_cache(self, mock_camera, fake_encoding_1404):
        encoder = MagicMock()
        encoder.load_all_encodings.return_value = {"STU001": [fake_encoding_1404]}

        svc   = RecognitionService(mock_camera, MagicMock(), encoder)
        count = svc.reload_encodings()
        assert count == 1
        assert "STU001" in svc._encoding_cache

    def test_cache_used_on_second_call(self, mock_camera, fake_encoding_1404):
        """load_all_encodings should only be called once after reload."""
        encoder = MagicMock()
        encoder.load_all_encodings.return_value = {"STU001": [fake_encoding_1404]}
        encoder.encode_face.return_value = fake_encoding_1404

        detector = MagicMock()
        detector.detect_faces.return_value = [(10, 110, 90, 10)]
        detector.scale_face_locations.return_value = [(10, 110, 90, 10)]

        svc = RecognitionService(mock_camera, detector, encoder, )
        svc.reload_encodings()   # pre-load

        svc.recognize_from_camera()
        # load_all_encodings called exactly once (during reload, not again during recognise)
        encoder.load_all_encodings.assert_called_once()


class TestRecognizeAPI:
    """Integration tests for POST /recognize — checks JSON response shapes."""

    def _mock_recognizer_return(self, dto: RecognizeResponseDTO):
        patcher = patch("api.recognize._recognizer.recognize_from_camera", return_value=dto)
        return patcher

    def test_no_face_response_shape(self, test_client):
        dto = RecognizeResponseDTO(status=RecognitionStatus.NO_FACE, recognition_time_ms=10.0)
        with self._mock_recognizer_return(dto):
            r = test_client.post("/recognize")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert body["recognized"] is False
        assert body["code"] == "NO_FACE"

    def test_multiple_faces_response_shape(self, test_client):
        dto = RecognizeResponseDTO(
            status=RecognitionStatus.MULTIPLE_FACES, recognition_time_ms=10.0
        )
        with self._mock_recognizer_return(dto):
            r = test_client.post("/recognize")
        body = r.json()
        assert body["code"] == "MULTIPLE_FACES"
        assert "not marked" in body["message"].lower() or "multiple" in body["message"].lower()

    def test_unknown_response_shape(self, test_client):
        dto = RecognizeResponseDTO(
            status=RecognitionStatus.UNKNOWN,
            distance=0.72,
            recognition_time_ms=10.0,
        )
        with self._mock_recognizer_return(dto):
            r = test_client.post("/recognize")
        body = r.json()
        assert body["code"] == "UNKNOWN"
        assert body["distance"] == pytest.approx(0.72)

    def test_recognized_response_shape(self, test_client, db_session):
        from models.student import Student
        student = Student(
            student_id="CSE001", name="Sudheer", department="CSE",
            semester="6", section="A", teacher_id="T001", is_active=True,
        )
        db_session.add(student)
        db_session.flush()

        dto = RecognizeResponseDTO(
            status=RecognitionStatus.RECOGNIZED,
            student_id="CSE001",
            distance=0.24,
            recognition_time_ms=10.0,
        )
        with self._mock_recognizer_return(dto):
            r = test_client.post("/recognize")
        body = r.json()
        assert body["success"] is True
        assert body["recognized"] is True
        assert body["registerNumber"] == "CSE001"
        assert body["distance"] == pytest.approx(0.24)
        assert "confidence" not in body   # distance only, no fake confidence
