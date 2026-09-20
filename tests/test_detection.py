"""
tests/test_detection.py – Tests for face detection (Phase 1 + Phase 6 rules).

Tests cover:
  - Camera availability check (mocked)
  - No face in blank frame
  - Scale-back helper
  - MULTIPLE_FACES guard constant
  - DetectionService.detect_single_face returns correct error codes
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.detection_service import DetectionService
from services.camera_service import CameraService
from utils.constants import MAX_FACES_REGISTRATION, RecognitionStatus


class TestCameraAvailability:

    def test_check_connection_returns_true_when_camera_opens(self):
        svc = CameraService()
        with patch("services.camera_service.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_vc.return_value = mock_cap
            assert svc.check_connection() is True

    def test_check_connection_returns_false_when_no_camera(self):
        svc = CameraService()
        with patch("services.camera_service.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap
            assert svc.check_connection() is False

    def test_camera_source_laptop_from_settings(self):
        """CameraService reads camera_source from settings (set via .env)."""
        from config import get_settings
        svc = CameraService()
        # Default from .env / settings is "laptop"
        assert svc._source in ("laptop", "esp32")   # valid values

    def test_camera_source_resolves_laptop_index(self):
        """Laptop source resolves to an int (camera index)."""
        from unittest.mock import patch
        svc = CameraService()
        with patch.object(svc, "_source", "laptop"):
            resolved = svc._resolve_source()
        assert isinstance(resolved, int)

    def test_camera_source_resolves_esp32_url(self):
        """ESP32 source resolves to a string URL."""
        from unittest.mock import patch
        svc = CameraService()
        with patch.object(svc, "_source", "esp32"):
            resolved = svc._resolve_source()
        assert isinstance(resolved, str)
        assert resolved.startswith("http")


class TestFaceDetection:

    def test_detect_faces_blank_image_returns_list(self):
        """Blank image should return a list (empty or with false positives)."""
        svc   = DetectionService()
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 200
        result = svc.detect_faces(blank)
        assert isinstance(result, list)

    def test_detect_single_face_no_face_error_message(self):
        svc   = DetectionService()
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 200

        with patch("services.detection_service._face_locations", return_value=[]):
            loc, err = svc.detect_single_face(blank)
            assert loc is None
            assert err is not None
            assert "No face" in err or "no face" in err.lower()

    def test_detect_single_face_multiple_faces_error(self):
        svc   = DetectionService()
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 200
        two_faces = [(10, 100, 90, 10), (200, 300, 280, 200)]

        with patch("services.detection_service._face_locations", return_value=two_faces):
            loc, err = svc.detect_single_face(blank)
            assert loc is None
            assert err is not None
            assert "Multiple" in err or "multiple" in err.lower()

    def test_scale_face_locations_doubles_at_half_scale(self):
        svc = DetectionService()
        locations = [(50, 100, 150, 10)]
        scaled    = svc.scale_face_locations(locations, scale=0.5)
        assert scaled == [(100, 200, 300, 20)]

    def test_scale_face_locations_empty(self):
        svc    = DetectionService()
        result = svc.scale_face_locations([], scale=0.5)
        assert result == []

    def test_max_faces_registration_is_one(self):
        """The enrollment multiple-face guard must be exactly 1."""
        assert MAX_FACES_REGISTRATION == 1

    def test_multiple_faces_status_exists(self):
        assert RecognitionStatus.MULTIPLE_FACES.value == "MULTIPLE_FACES"
