"""
services/camera_service.py – Camera abstraction layer.

PHASE 10 IMPLEMENTATION
-----------------------
Supports two camera sources configured via .env:

    CAMERA_SOURCE=laptop    → cv2.VideoCapture(CAMERA_INDEX)
    CAMERA_SOURCE=esp32     → MJPEG stream from ESP32_CAMERA_URL

The recognition, enrollment, and classroom services call ONLY:
    camera.open()
    camera.read_frame()
    camera.release()
    camera.is_open()

They do NOT know or care whether the frame came from a laptop webcam
or an ESP32-CAM over Wi-Fi.  Swapping sources is a one-line .env change.

ESP32-CAM NOTES (Phase 11)
--------------------------
The AI Thinker ESP32-CAM streams MJPEG over HTTP.  OpenCV's VideoCapture
can read MJPEG streams directly:
    cv2.VideoCapture("http://<IP>/stream")  or
    cv2.VideoCapture("http://<IP>:81/stream")

Do NOT enable ESP32 until the laptop pipeline is confirmed working.
Set ESP32_CAMERA_URL in .env, then change CAMERA_SOURCE=esp32.

THREAD SAFETY
-------------
All public methods are protected by a reentrant lock.
The same instance can be safely shared across threads (FastAPI thread pool).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CameraService:
    """Abstract camera source — laptop webcam or ESP32-CAM MJPEG stream.

    Usage::

        with CameraService() as cam:
            frame = cam.read_frame_with_retry()

    Or manually::

        cam = CameraService()
        cam.open()
        ret, frame = cam.read_frame()
        cam.release()
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._source = settings.camera_source.lower()   # "laptop" or "esp32"
        logger.debug("CameraService init (source=%s)", self._source)

    # ── Source resolution ─────────────────────────────────────────────────────

    def _resolve_source(self) -> int | str:
        """Return the OpenCV-compatible source identifier.

        Returns:
            int (camera index) for laptop, str (URL) for ESP32.
        """
        if self._source == "esp32":
            url = settings.esp32_camera_url.rstrip("/")
            # Append /stream if the URL doesn't already include a path
            if "/" not in url.split("://", 1)[-1]:
                url = f"{url}/stream"
            logger.info("ESP32-CAM source: %s", url)
            return url
        # Default: laptop / USB webcam
        logger.info("Laptop camera source: index=%d", settings.camera_index)
        return settings.camera_index

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Open the camera source.

        Returns:
            True if successfully opened.
        """
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                logger.debug("Camera already open — reusing")
                return True

            source = self._resolve_source()
            logger.info("Opening camera source: %r", source)
            self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                logger.error("Failed to open camera source: %r", source)
                self._cap = None
                return False

            # Resolution only applies to laptop/USB cameras
            # For ESP32 streams, the stream resolution is set on the ESP32 side
            if self._source != "esp32":
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)

            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                "Camera opened OK (source=%s, %dx%d)", self._source, w, h
            )
            return True

    def release(self) -> None:
        """Release the camera and free resources."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                logger.info("Camera released (source=%s)", self._source)

    def is_open(self) -> bool:
        """Return True if the camera is currently open."""
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    # ── Frame acquisition ─────────────────────────────────────────────────────

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one frame from the camera.

        Returns:
            (success, frame) where frame is BGR or None on failure.
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                logger.warning("read_frame called on closed camera")
                return False, None
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("read_frame: cap.read() returned False")
            return ret, frame if ret else None

    def read_frame_with_retry(
        self,
        max_retries: int = 3,
        delay_s: float = 0.1,
    ) -> Optional[np.ndarray]:
        """Read a frame with automatic retries on transient failure.

        Args:
            max_retries: Maximum read attempts before giving up.
            delay_s:     Seconds to wait between retries.

        Returns:
            BGR frame array, or None if all retries fail.
        """
        for attempt in range(1, max_retries + 1):
            ret, frame = self.read_frame()
            if ret and frame is not None:
                return frame
            if attempt < max_retries:
                logger.debug("Frame retry %d/%d", attempt, max_retries)
                time.sleep(delay_s)
        logger.error("All %d frame read attempts failed", max_retries)
        return None

    # ── Connectivity check ────────────────────────────────────────────────────

    def check_connection(self) -> bool:
        """Check whether the configured camera source is accessible.

        Opens a temporary capture, checks availability, then releases.
        Used by GET /health and GET /camera.

        Returns:
            True if the camera/stream is accessible.
        """
        source = self._resolve_source()
        try:
            tmp = cv2.VideoCapture(source)
            connected = tmp.isOpened()
            # For network streams, try reading one frame to confirm data flow
            if connected and self._source == "esp32":
                ret, _ = tmp.read()
                connected = ret
            tmp.release()
            logger.info(
                "Camera connection check: %s (source=%r)",
                "CONNECTED" if connected else "NOT FOUND",
                source,
            )
            return connected
        except Exception as exc:
            logger.exception("Camera connection check error: %s", exc)
            return False

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "CameraService":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
