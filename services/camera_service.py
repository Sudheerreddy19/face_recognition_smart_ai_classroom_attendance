from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_selected_camera_index: Optional[int] = None

class CameraService:
    def __init__(self, camera_index: Optional[int] = None) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._source = settings.camera_source.lower()
        self._camera_index = camera_index
        logger.debug('CameraService init (source=%s, custom_index=%s)', self._source, camera_index)

    @classmethod
    def set_active_camera_index(cls, index: int) -> None:
        global _selected_camera_index
        _selected_camera_index = index
        logger.info('Global active camera index set to %d', index)

    @classmethod
    def get_active_camera_index(cls) -> int:
        global _selected_camera_index
        if _selected_camera_index is not None:
            return _selected_camera_index
        preferred = cls.detect_preferred_camera_index()
        return preferred if preferred is not None else settings.camera_index

    @classmethod
    def get_pnp_camera_names(cls) -> List[Dict[str, Any]]:
        cmd = ['powershell', '-NoProfile', '-Command', 'Get-PnpDevice -Class Camera | Select-Object FriendlyName, InstanceId, Status, Present | ConvertTo-Json']
        try:
            out = subprocess.check_output(cmd, timeout=5).decode('utf-8', errors='ignore').strip()
            if not out:
                return []
            data = json.loads(out)
            if isinstance(data, dict):
                return [data]
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.debug('Could not query PnP cameras via PowerShell: %s', exc)
            return []

    @classmethod
    def list_available_cameras(cls, max_index: int = 4) -> List[Dict[str, Any]]:
        pnp_cams = [c for c in cls.get_pnp_camera_names() if c.get('Present') is not False]
        active_idx = cls.get_active_camera_index()
        cameras: List[Dict[str, Any]] = []

        for idx in range(max_index):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()

                    name = f'Camera {idx}'
                    if idx < len(pnp_cams):
                        pnp_name = pnp_cams[idx].get('FriendlyName')
                        if pnp_name:
                            name = pnp_name

                    lower_name = name.lower()
                    is_external = (
                        'external' in lower_name
                        or 'usb' in lower_name
                        or 'iball' in lower_name
                        or 'logitech' in lower_name
                        or 'creative' in lower_name
                        or idx > 0
                    )

                    cameras.append({
                        'index': idx,
                        'name': name,
                        'isExternal': is_external,
                        'active': (idx == active_idx),
                        'status': 'READY' if ret else 'CONNECTED',
                    })
            except Exception as exc:
                logger.debug('Error probing camera index %d: %s', idx, exc)

        return cameras

    @classmethod
    def detect_preferred_camera_index(cls) -> int:
        pnp_cams = cls.get_pnp_camera_names()
        for i, c in enumerate(pnp_cams):
            name = (c.get('FriendlyName') or '').lower()
            if 'external' in name or 'usb' in name or 'iball' in name:
                logger.info('Auto-detected external camera at index %d: %s', i, c.get('FriendlyName'))
                return i

        try:
            cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if cap1.isOpened():
                cap1.release()
                logger.info('Detected secondary camera at index 1, prioritizing external')
                return 1
        except Exception:
            pass

        return settings.camera_index

    def _resolve_source(self) -> int | str:
        if self._source == 'esp32':
            url = settings.esp32_camera_url.rstrip('/')
            if '/' not in url.split('://', 1)[-1]:
                url = f'{url}/stream'
            logger.info('ESP32-CAM source: %s', url)
            return url

        if self._camera_index is not None:
            idx = self._camera_index
        elif _selected_camera_index is not None:
            idx = _selected_camera_index
        else:
            idx = self.detect_preferred_camera_index()

        logger.info('Resolved camera source index: %d (configured default: %d)', idx, settings.camera_index)
        return idx

    def open(self) -> bool:
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                logger.debug('Camera already open - reusing')
                return True

            source = self._resolve_source()
            logger.info('Opening camera source: %r', source)

            if isinstance(source, int):
                self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    logger.warning('DirectShow open failed for index %d, trying default backend', source)
                    self._cap = cv2.VideoCapture(source)
            else:
                self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                logger.error('Failed to open camera source: %r', source)
                self._cap = None
                return False

            if self._source != 'esp32':
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)

            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info('Camera opened OK (source=%s, %dx%d)', source, w, h)
            return True

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                logger.info('Camera released (source=%s)', self._source)

    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                logger.warning('read_frame called on closed camera')
                return False, None
            ret, frame = self._cap.read()
            if not ret:
                logger.warning('read_frame: cap.read() returned False')
            return ret, frame if ret else None

    def read_frame_with_retry(
        self,
        max_retries: int = 3,
        delay_s: float = 0.1,
    ) -> Optional[np.ndarray]:
        for attempt in range(1, max_retries + 1):
            ret, frame = self.read_frame()
            if ret and frame is not None:
                return frame
            if attempt < max_retries:
                logger.debug('Frame retry %d/%d', attempt, max_retries)
                time.sleep(delay_s)
        logger.error('All %d frame read attempts failed', max_retries)
        return None

    def check_connection(self) -> bool:
        source = self._resolve_source()
        try:
            if isinstance(source, int):
                tmp = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                tmp = cv2.VideoCapture(source)
            connected = tmp.isOpened()
            if connected and self._source == 'esp32':
                ret, _ = tmp.read()
                connected = ret
            tmp.release()
            logger.info(
                'Camera connection check: %s (source=%r)',
                'CONNECTED' if connected else 'NOT FOUND',
                source,
            )
            return connected
        except Exception as exc:
            logger.exception('Camera connection check error: %s', exc)
            return False

    def __enter__(self) -> 'CameraService':
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
