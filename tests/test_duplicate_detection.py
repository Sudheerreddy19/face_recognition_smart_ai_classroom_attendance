"""
tests/test_duplicate_detection.py - Comprehensive Unit & Scenario Tests for Duplicate Face Detection
Covers:
  1. Register a completely new face (NO_DUPLICATE)
  2. Attempt to register the same face for another student (DUPLICATE_FOUND)
  3. Attempt to register the same face again for the same student (re-registration allowed -> NO_DUPLICATE)
  4. Attempt registration with a slightly different camera angle (matches within tolerance -> DUPLICATE_FOUND)
  5. Attempt registration with poor lighting / low similarity (UNCERTAIN_MATCH)
  6. Test an uncertain match zone
  7. Test two possible matching students (conflicting candidates handled, top match selected)
  8. Confirm that duplicate attempts do not modify existing records
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from services.recognition_service import RecognitionService
from utils.constants import DuplicateCheckOutcome


@pytest.fixture
def base_encoding():
    # Unit normalized random vector of 128 dimensions
    vec = np.ones(128, dtype=np.float64)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def mock_services():
    camera_svc = MagicMock()
    detection_svc = MagicMock()
    encoding_svc = MagicMock()
    rec_svc = RecognitionService(
        camera_service=camera_svc,
        detection_service=detection_svc,
        encoding_service=encoding_svc,
    )
    return rec_svc, encoding_svc


class TestDuplicateFaceDetection:

    def test_scenario_1_register_completely_new_face(self, mock_services, base_encoding):
        rec_svc, encoder = mock_services
        # Existing database has student A
        encoder.load_all_encodings.return_value = {
            "22CS101": [base_encoding]
        }

        # New face is orthogonal (distance ~ 1.414, completely different person)
        new_vec = -base_encoding
        new_vec[0] = 0.5
        new_vec = new_vec / np.linalg.norm(new_vec)
        new_encodings = [new_vec] * 10

        res = rec_svc.check_duplicate_faces(new_encodings, target_student_id="22CS102")
        assert res["outcome"] == DuplicateCheckOutcome.NO_DUPLICATE
        assert "No duplicate" in res["message"]

    def test_scenario_2_register_same_face_for_another_student(self, mock_services, base_encoding):
        rec_svc, encoder = mock_services
        # Student A is already registered
        encoder.load_all_encodings.return_value = {
            "22CS101": [base_encoding]
        }

        # Another student (22CS102) tries to register with the identical face
        new_encodings = [base_encoding] * 10
        res = rec_svc.check_duplicate_faces(new_encodings, target_student_id="22CS102")

        assert res["outcome"] == DuplicateCheckOutcome.DUPLICATE_FOUND
        assert res["matched_student_id"] == "22CS101"
        assert res["min_distance"] < 1e-4
        assert res["matched_count"] == 10

    def test_scenario_3_register_same_face_for_same_student(self, mock_services, base_encoding):
        rec_svc, encoder = mock_services
        # Student A already registered
        encoder.load_all_encodings.return_value = {
            "22CS101": [base_encoding]
        }

        # Student A re-captures face under their own record (re-registration)
        new_encodings = [base_encoding] * 10
        res = rec_svc.check_duplicate_faces(new_encodings, target_student_id="22CS101")

        # Must NOT block their own record
        assert res["outcome"] == DuplicateCheckOutcome.NO_DUPLICATE

    def test_scenario_4_different_camera_angle_duplicate(self, mock_services, base_encoding):
        rec_svc, encoder = mock_services
        encoder.load_all_encodings.return_value = {
            "22CS101": [base_encoding]
        }

        # Slight perturbation simulating head tilt/angle (distance ~ 0.25 <= match_threshold 0.50)
        noise = np.random.normal(0, 0.02, 128)
        tilted_vec = base_encoding + noise
        tilted_vec = tilted_vec / np.linalg.norm(tilted_vec)
        new_encodings = [tilted_vec] * 10

        res = rec_svc.check_duplicate_faces(new_encodings, target_student_id="22CS105")
        assert res["outcome"] == DuplicateCheckOutcome.DUPLICATE_FOUND
        assert res["matched_student_id"] == "22CS101"

    def test_scenario_5_and_6_poor_lighting_uncertain_match(self, mock_services, base_encoding):
        rec_svc, encoder = mock_services
        encoder.load_all_encodings.return_value = {
            "22CS101": [base_encoding]
        }

        # Moderate perturbation in uncertain zone [0.50, 0.60] (e.g., distance 0.55)
        # Create a vector with angle giving distance ~ 0.55
        target_dist = 0.55
        # d^2 = 2 - 2*cos(theta) => cos(theta) = 1 - d^2 / 2
        cos_theta = 1.0 - (target_dist ** 2) / 2.0
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)

        # Orthogonal unit vector
        ortho = np.zeros(128)
        ortho[1] = 1.0  # perpendicular
        uncertain_vec = cos_theta * base_encoding + sin_theta * ortho
        uncertain_vec = uncertain_vec / np.linalg.norm(uncertain_vec)

        actual_dist = np.linalg.norm(uncertain_vec - base_encoding)
        assert 0.50 < actual_dist < 0.60

        new_encodings = [uncertain_vec] * 10
        res = rec_svc.check_duplicate_faces(new_encodings, target_student_id="22CS109")

        assert res["outcome"] == DuplicateCheckOutcome.UNCERTAIN_MATCH
        assert "improve lighting" in res["message"].lower()

    def test_scenario_7_two_possible_matching_students(self, mock_services, base_encoding):
        rec_svc, encoder = mock_services
        # Two registered students in DB
        s1 = base_encoding
        # s2 is close to s1 but slightly different
        noise2 = np.random.normal(0, 0.01, 128)
        s2 = (base_encoding + noise2) / np.linalg.norm(base_encoding + noise2)

        encoder.load_all_encodings.return_value = {
            "22CS101": [s1],
            "22CS102": [s2]
        }

        # Incoming face matches both, but closest to 22CS101
        new_encodings = [s1] * 10
        res = rec_svc.check_duplicate_faces(new_encodings, target_student_id="22CS999")

        assert res["outcome"] == DuplicateCheckOutcome.DUPLICATE_FOUND
        assert res["matched_student_id"] == "22CS101"
        assert "all_candidates" in res
        assert "22CS101" in res["all_candidates"]
        assert "22CS102" in res["all_candidates"]

    def test_scenario_8_duplicate_attempts_do_not_modify_existing_records(self, mock_services, base_encoding):
        rec_svc, encoder = mock_services
        original_db = {
            "22CS101": [base_encoding]
        }
        encoder.load_all_encodings.return_value = original_db

        # Duplicate attempt
        new_encodings = [base_encoding] * 10
        res = rec_svc.check_duplicate_faces(new_encodings, target_student_id="22CS999")
        assert res["outcome"] == DuplicateCheckOutcome.DUPLICATE_FOUND

        # Verify save_encodings was NEVER called
        encoder.save_encodings.assert_not_called()
        encoder.delete_encoding.assert_not_called()
