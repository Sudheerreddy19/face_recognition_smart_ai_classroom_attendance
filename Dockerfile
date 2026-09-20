# ─────────────────────────────────────────────────────────────────────────────
#  Dockerfile – python-face-service
#
#  IMPORTANT: Windows Docker cannot access the Windows laptop webcam through
#  /dev/video0.  This Docker image is built for DEPLOYMENT on Linux/server
#  where the camera source is an ESP32-CAM MJPEG stream (CAMERA_SOURCE=esp32).
#
#  Development (Windows): use .venv311 directly, not Docker.
#  Deployment (Linux):    docker-compose up — uses ESP32 stream.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="AI Smart Classroom"
LABEL description="Face recognition microservice — ESP32-CAM deployment"

# ── System dependencies for OpenCV headless ──────────────────────────────────
# libgl1 and libglib2.0 are required by cv2 on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Create persistent directories
RUN mkdir -p encodings uploads/students uploads/temp logs

# ── Runtime configuration ─────────────────────────────────────────────────────
# All sensitive values MUST be provided via environment variables or .env
# Do NOT hardcode any secrets in this file.
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV CAMERA_SOURCE=esp32
ENV LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
