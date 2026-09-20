"""
startup.py – Application startup and shutdown tasks.

Startup tasks:
  1. Create required directories.
  2. Create database tables.
  3. Warm up face recognition encoding cache.
  4. Retry unsynced attendance records against Spring Boot.
  5. Log configuration summary.

Shutdown tasks:
  1. Log shutdown message.
"""

from __future__ import annotations

from pathlib import Path

from config import get_settings
from database import create_tables, db_context
from utils.file_utils import ensure_directories
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def run_startup() -> None:
    """Execute all startup tasks."""
    logger.info("=" * 60)
    logger.info("  %s – Starting up", settings.app_name.upper())
    logger.info("  Environment   : %s", settings.app_env)
    logger.info("  Host          : %s:%d", settings.app_host, settings.app_port)
    logger.info("  Camera source : %s (index=%d)", settings.camera_source, settings.camera_index)
    logger.info("  DB URL        : %s", settings.database_url)
    logger.info("  Spring Boot   : %s", settings.spring_boot_base_url)
    logger.info("  Threshold     : %.2f", settings.recognition_threshold)
    logger.info("=" * 60)

    # ── 1. Directories ────────────────────────────────────────────────────────
    dirs: list[Path] = [
        settings.upload_path,
        settings.temp_path,
        settings.encoding_path,
        settings.log_path,
    ]
    ensure_directories(*dirs)
    logger.info("Directories verified")

    # ── 2. Database tables ────────────────────────────────────────────────────
    import models.student      # noqa: F401
    import models.attendance   # noqa: F401
    create_tables()
    logger.info("Database tables verified")

    # ── 3. Warm encoding cache ─────────────────────────────────────────────────
    try:
        from services.encoding_service import EncodingService
        enc_svc = EncodingService()
        all_encodings = enc_svc.load_all_encodings()
        logger.info(
            "Encoding cache warmed – %d student(s) loaded", len(all_encodings)
        )
    except Exception as exc:
        logger.warning("Could not pre-load encodings on startup: %s", exc)

    # ── 4. Retry unsynced attendance records ──────────────────────────────────
    try:
        from services.springboot_service import SpringBootService
        spring_svc = SpringBootService()
        if spring_svc.health_check():
            with db_context() as db:
                synced = spring_svc.retry_unsynced(db)
            if synced:
                logger.info("Startup sync: pushed %d unsynced record(s) to Spring Boot", synced)
        else:
            logger.info("Spring Boot not reachable on startup — unsynced records will retry later")
    except Exception as exc:
        logger.warning("Startup Spring Boot retry failed (non-fatal): %s", exc)

    logger.info("Startup complete — service ready ✓")


def run_shutdown() -> None:
    """Execute graceful shutdown tasks."""
    logger.info("=" * 60)
    logger.info("  %s – Shutting down", settings.app_name.upper())
    logger.info("=" * 60)
