from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import get_settings
from models.response_models import CameraResponse
from services.camera_service import CameraService
from services.hardware_service import HardwareService
from utils.constants import (
    CameraStatus,
    MSG_CAMERA_CONNECTED,
    MSG_CAMERA_NOT_FOUND,
)
from utils.logger import get_logger

router = APIRouter(tags=['Camera & Hardware'])
logger = get_logger(__name__)
settings = get_settings()


class CameraSelectRequest(BaseModel):
    index: int


@router.get(
    '/camera',
    response_model=CameraResponse,
    summary='Camera connectivity test',
)
async def camera_test() -> CameraResponse:
    logger.info('Camera test requested (active index=%d)', CameraService.get_active_camera_index())
    camera_svc = CameraService()
    is_connected = await asyncio.to_thread(camera_svc.check_connection)

    status = CameraStatus.CONNECTED if is_connected else CameraStatus.NOT_FOUND
    message = MSG_CAMERA_CONNECTED if is_connected else MSG_CAMERA_NOT_FOUND

    return CameraResponse(
        status=status,
        message=message,
        camera_index=CameraService.get_active_camera_index(),
    )


@router.get('/camera/devices', summary='List all detected cameras')
async def list_cameras():
    cameras = await asyncio.to_thread(CameraService.list_available_cameras)
    active_idx = CameraService.get_active_camera_index()
    return {
        'cameras': cameras,
        'activeCameraIndex': active_idx,
        'preferredIndex': CameraService.detect_preferred_camera_index(),
    }


@router.post('/camera/select', summary='Select active camera index')
async def select_camera(req: CameraSelectRequest):
    CameraService.set_active_camera_index(req.index)
    return {
        'success': True,
        'selectedCameraIndex': req.index,
        'message': f'Active camera set to index {req.index}',
    }


@router.get('/devices/hardware', summary='Get all hardware devices for Device Manager')
async def get_hardware_devices():
    devices = await asyncio.to_thread(HardwareService.get_all_devices)
    return devices
