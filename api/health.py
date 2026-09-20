"""
api/health.py – Health check endpoint.

GET /health
    Returns service status, camera state, Spring Boot connectivity,
    loaded encodings count, and timestamp.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from pydantic import BaseModel, Field
from datetime import datetime

from services.camera_service import CameraService
from services.springboot_service import SpringBootService
from services.encoding_service import EncodingService
from utils.constants import CameraStatus, ServiceStatus
from utils.logger import get_logger

router = APIRouter(tags=["Health"])
logger = get_logger(__name__)


class HealthResponse(BaseModel):
    status: ServiceStatus
    camera: CameraStatus
    spring_boot: str          # "REACHABLE" | "UNREACHABLE"
    encodings_loaded: int
    service: str = "FACE RECOGNITION"
    timestamp: datetime = Field(default_factory=datetime.now)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns operational status of the Face Recognition service, "
        "camera connectivity, Spring Boot reachability, and the number "
        "of student face encodings currently on disk."
    ),
)
async def health_check() -> HealthResponse:
    """Return comprehensive service health.

    Runs camera check and Spring Boot ping concurrently.
    """
    logger.info("Health check requested")

    cam_svc    = CameraService()
    spring_svc = SpringBootService()
    enc_svc    = EncodingService()

    # Run camera + Spring Boot checks concurrently
    cam_ok, spring_ok = await asyncio.gather(
        asyncio.to_thread(cam_svc.check_connection),
        asyncio.to_thread(spring_svc.health_check),
    )

    # Encoding count is fast (just glob)
    enc_infos = enc_svc.list_encoding_infos()

    response = HealthResponse(
        status=ServiceStatus.UP,
        camera=CameraStatus.CONNECTED if cam_ok else CameraStatus.DISCONNECTED,
        spring_boot="REACHABLE" if spring_ok else "UNREACHABLE",
        encodings_loaded=len(enc_infos),
    )

    logger.info(
        "Health: camera=%s spring_boot=%s encodings=%d",
        response.camera, response.spring_boot, response.encodings_loaded,
    )
    return response
