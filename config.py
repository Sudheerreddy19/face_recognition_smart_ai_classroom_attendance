"""
config.py – Centralised application configuration using Pydantic Settings.

All configurable values are loaded from environment variables / .env file.
Override any value by setting the corresponding env var before launching the app.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "python-face-service"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # ── Camera ────────────────────────────────────────────────────────────────
    # camera_source: "laptop" uses cv2.VideoCapture(camera_index)
    # camera_source: "esp32"  uses MJPEG stream from esp32_camera_url
    camera_source: str = "laptop"
    camera_index: int = 0
    esp32_camera_url: str = "http://192.168.1.100"   # override via .env
    capture_count: int = 20
    frame_width: int = 640
    frame_height: int = 480

    # ── Face Recognition ──────────────────────────────────────────────────────
    recognition_threshold: float = 0.50
    detection_model: str = "hog"       # "hog" (CPU) or "cnn" (GPU)
    encoding_jitters: int = 1

    # ── File System Paths ─────────────────────────────────────────────────────
    upload_dir: str = "uploads/students"
    temp_dir: str = "uploads/temp"
    encoding_dir: str = "encodings"
    log_dir: str = "logs"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./face_service.db"

    # ── Spring Boot Integration ───────────────────────────────────────────────
    spring_boot_base_url: str = "http://localhost:8080"
    spring_boot_attendance_endpoint: str = "/api/attendance/save"
    spring_boot_student_endpoint: str = "/api/students"
    spring_boot_timeout: int = 10

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_max_bytes: int = 10_485_760   # 10 MB
    log_backup_count: int = 5

    # ── Derived Properties ────────────────────────────────────────────────────
    @property
    def spring_boot_attendance_url(self) -> str:
        """Full URL for the Spring Boot attendance endpoint."""
        return f"{self.spring_boot_base_url}{self.spring_boot_attendance_endpoint}"

    @property
    def upload_path(self) -> Path:
        """Absolute Path object for the student upload directory."""
        return Path(self.upload_dir)

    @property
    def temp_path(self) -> Path:
        """Absolute Path object for the temp directory."""
        return Path(self.temp_dir)

    @property
    def encoding_path(self) -> Path:
        """Absolute Path object for the encodings directory."""
        return Path(self.encoding_dir)

    @property
    def log_path(self) -> Path:
        """Absolute Path object for the logs directory."""
        return Path(self.log_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` instance.

    Using ``@lru_cache`` ensures the ``.env`` file is read only once.
    """
    return Settings()
