"""
api/camera.py – Camera connectivity and live detection endpoints.

GET /camera   – Check if the camera is reachable.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import get_settings
from models.response_models import CameraResponse
from services.camera_service import CameraService
from utils.constants import (
    CameraStatus,
    MSG_CAMERA_CONNECTED,
    MSG_CAMERA_NOT_FOUND,
)
from utils.logger import get_logger

router = APIRouter(tags=["Camera"])
logger = get_logger(__name__)
settings = get_settings()


@router.get(
    "/camera",
    response_model=CameraResponse,
    summary="Camera connectivity test",
    description=(
        "Attempts to open the configured camera device and returns its "
        "connection status."
    ),
)
async def camera_test() -> CameraResponse:
    """Test webcam connectivity.

    Opens the camera at the configured index, checks availability, then
    immediately releases it.

    Returns:
        :class:`~models.response_models.CameraResponse` with connection status.
    """
    logger.info("Camera test requested (index=%d)", settings.camera_index)

    camera_svc = CameraService()
    is_connected = await asyncio.to_thread(camera_svc.check_connection)

    status = CameraStatus.CONNECTED if is_connected else CameraStatus.NOT_FOUND
    message = MSG_CAMERA_CONNECTED if is_connected else MSG_CAMERA_NOT_FOUND

    logger.info("Camera test result: %s", status)

    return CameraResponse(
        status=status,
        message=message,
        camera_index=settings.camera_index,
    )
