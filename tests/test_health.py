"""
tests/test_health.py – Tests for /health and /camera endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:

    @patch("api.health.CameraService")
    @patch("api.health.SpringBootService")
    @patch("api.health.EncodingService")
    def test_health_camera_connected(self, mock_enc, mock_spring, mock_cam):
        mock_cam.return_value.check_connection.return_value = True
        mock_spring.return_value.health_check.return_value = True
        mock_enc.return_value.list_encoding_infos.return_value = [{}, {}]

        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "UP"
        assert body["camera"] == "CONNECTED"
        assert body["spring_boot"] == "REACHABLE"
        assert body["encodings_loaded"] == 2
        assert "timestamp" in body

    @patch("api.health.CameraService")
    @patch("api.health.SpringBootService")
    @patch("api.health.EncodingService")
    def test_health_camera_disconnected(self, mock_enc, mock_spring, mock_cam):
        mock_cam.return_value.check_connection.return_value = False
        mock_spring.return_value.health_check.return_value = False
        mock_enc.return_value.list_encoding_infos.return_value = []

        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["camera"] == "DISCONNECTED"
        assert body["spring_boot"] == "UNREACHABLE"
        assert body["encodings_loaded"] == 0


class TestCameraEndpoint:

    @patch("api.camera.CameraService")
    def test_camera_connected(self, mock_cam):
        mock_cam.return_value.check_connection.return_value = True
        r = client.get("/camera")
        assert r.status_code == 200
        assert r.json()["status"] == "CONNECTED"

    @patch("api.camera.CameraService")
    def test_camera_not_found(self, mock_cam):
        mock_cam.return_value.check_connection.return_value = False
        r = client.get("/camera")
        assert r.status_code == 200
        assert r.json()["status"] == "NOT_FOUND"
