"""
app.py – FastAPI application factory and entry point.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Or via the startup helper:
    python app.py
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import attendance
from api import camera
from api import health
from api import recognize
from api import register
from config import get_settings
from models.response_models import ErrorResponse
from startup import run_shutdown, run_startup
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown tasks."""
    run_startup()

    # ── Start WebSocket agent (outbound to Spring Boot on Render) ─────────────
    # The agent connects outbound so Spring Boot can dispatch face-capture jobs
    # to this local Python service without exposing port 8000 publicly.
    # If CLASSAI_AGENT_WS_URL / CLASSAI_AGENT_TOKEN are not set, the agent
    # logs a warning and exits immediately — all other features keep working.
    from agent.ws_agent import run_agent
    _agent_task = asyncio.create_task(run_agent(), name="ws-agent")
    logger.info("WebSocket agent task started")

    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    _agent_task.cancel()
    try:
        await _agent_task
    except asyncio.CancelledError:
        pass
    run_shutdown()


# ── Application Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        Configured :class:`fastapi.FastAPI` instance.
    """
    app = FastAPI(
        title="AI Smart Classroom – Face Recognition Microservice",
        description=(
            "Production-ready face recognition service for automated student "
            "attendance management. Provides REST APIs for face registration, "
            "recognition, and attendance marking with Spring Boot integration."
        ),
        version="1.0.0",
        contact={
            "name": "AI Smart Classroom Team",
            "email": "support@smartclassroom.ai",
            
        },
        license_info={
            "name": "MIT",
        },
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global Exception Handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unhandled server-side exceptions."""
        logger.exception(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                error="Internal Server Error",
                detail=str(exc),
            ).model_dump(mode="json"),
        )

    # ── Request / Response Logging Middleware ─────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Log every incoming request and outgoing response."""
        logger.info(
            "→ %s %s (client=%s)",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        response = await call_next(request)
        logger.info(
            "← %s %s HTTP %d",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(camera.router)
    app.include_router(register.router)
    app.include_router(recognize.router)
    app.include_router(attendance.router)

    return app


app = create_app()


# ── Dev Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
