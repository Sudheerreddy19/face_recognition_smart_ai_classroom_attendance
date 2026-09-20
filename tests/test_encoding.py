"""
tests/test_encoding.py – Tests for face encoding (Phase 2 / Phase 3).

Tests cover:
  - face_compat functions return correct types
  - face_distance returns ndarray with correct length
  - compare_faces respects tolerance
  - EncodingService save/load/delete
  - EncodingService list_encoding_infos
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.face_compat import face_distance, compare_faces


class TestFaceCompat:

    def test_face_distance_returns_ndarray(self, fake_encoding_1404, different_encoding_1404):
        distances = face_distance([fake_encoding_1404], different_encoding_1404)
        assert isinstance(distances, np.ndarray)
        assert len(distances) == 1

    def test_face_distance_same_encoding_is_near_zero(self, fake_encoding_1404):
        distances = face_distance([fake_encoding_1404], fake_encoding_1404)
        assert float(distances[0]) < 1e-6

    def test_face_distance_different_encodings_is_positive(
        self, fake_encoding_1404, different_encoding_1404
    ):
        distances = face_distance([fake_encoding_1404], different_encoding_1404)
        assert float(distances[0]) > 0.01

    def test_face_distance_empty_known_returns_empty(self, fake_encoding_1404):
        result = face_distance([], fake_encoding_1404)
        assert len(result) == 0

    def test_compare_faces_same_encoding_matches(self, fake_encoding_1404):
        result = compare_faces([fake_encoding_1404], fake_encoding_1404, tolerance=0.5)
        assert result == [True]

    def test_compare_faces_different_encoding_no_match(
        self, fake_encoding_1404, different_encoding_1404
    ):
        # Use a tight tolerance so different encodings don't match
        result = compare_faces(
            [fake_encoding_1404], different_encoding_1404, tolerance=0.001
        )
        assert result == [False]

    def test_face_distance_multiple_known(self, fake_encoding_1404, different_encoding_1404):
        distances = face_distance(
            [fake_encoding_1404, different_encoding_1404], fake_encoding_1404
        )
        assert len(distances) == 2
        # First should be ~0, second should be larger
        assert distances[0] < distances[1]


class TestEncodingService:

    def _make_svc(self, tmp_path: Path):
        from services.encoding_service import EncodingService
        svc = EncodingService.__new__(EncodingService)
        svc._encoding_dir = tmp_path
        svc._jitters      = 1
        return svc

    def test_has_encoding_false_for_unknown(self, tmp_path):
        svc = self._make_svc(tmp_path)
        assert svc.has_encoding("NOBODY") is False

    def test_save_and_load_round_trip(self, tmp_path, fake_encoding_1404):
        svc = self._make_svc(tmp_path)
        svc.save_encodings("STU001", [fake_encoding_1404])
        loaded = svc.load_encodings("STU001")
        assert len(loaded) == 1
        np.testing.assert_array_almost_equal(loaded[0], fake_encoding_1404)

    def test_has_encoding_true_after_save(self, tmp_path, fake_encoding_1404):
        svc = self._make_svc(tmp_path)
        svc.save_encodings("STU002", [fake_encoding_1404])
        assert svc.has_encoding("STU002") is True

    def test_delete_encoding_removes_file(self, tmp_path, fake_encoding_1404):
        svc = self._make_svc(tmp_path)
        svc.save_encodings("STU003", [fake_encoding_1404])
        assert svc.delete_encoding("STU003") is True
        assert svc.has_encoding("STU003") is False

    def test_delete_encoding_nonexistent_returns_false(self, tmp_path):
        svc = self._make_svc(tmp_path)
        assert svc.delete_encoding("NOBODY") is False

    def test_load_encodings_missing_returns_empty(self, tmp_path):
        svc = self._make_svc(tmp_path)
        result = svc.load_encodings("MISSING")
        assert result == []

    def test_load_all_encodings_multiple_students(
        self, tmp_path, fake_encoding_1404, different_encoding_1404
    ):
        svc = self._make_svc(tmp_path)
        svc.save_encodings("A", [fake_encoding_1404])
        svc.save_encodings("B", [different_encoding_1404])
        all_enc = svc.load_all_encodings()
        assert set(all_enc.keys()) == {"A", "B"}

    def test_list_encoding_infos(self, tmp_path, fake_encoding_1404):
        svc = self._make_svc(tmp_path)
        svc.save_encodings("INFO_TEST", [fake_encoding_1404])
        infos = svc.list_encoding_infos()
        assert len(infos) == 1
        assert infos[0]["student_id"] == "INFO_TEST"
        assert infos[0]["face_count"] == 1
        assert infos[0]["file_size_bytes"] > 0
