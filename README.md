# 🎓 AI Smart Classroom – Face Recognition Microservice

> **Production-ready Python microservice** for automated student attendance management using computer vision and face recognition.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-orange.svg)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Virtual Environment Setup](#-virtual-environment-setup)
- [Environment Configuration](#-environment-configuration)
- [Running the Service](#-running-the-service)
- [Docker Setup](#-docker-setup)
- [API Reference](#-api-reference)
- [Spring Boot Integration](#-spring-boot-integration)
- [Troubleshooting](#-troubleshooting)

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                   FastAPI Application                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐   │
│  │  Health  │ │  Camera  │ │ Register │ │ Recognize │   │
│  │    API   │ │   API    │ │   API    │ │    API    │   │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │            Attendance API + Students + Encodings    │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌────────────────────┐  ┌────────────────────────────┐  │
│  │    Services Layer  │  │    Repository Layer        │  │
│  │ • CameraService    │  │ • EncodingRepository       │  │
│  │ • DetectionService │  │ • AttendanceRepository     │  │
│  │ • EncodingService  │  └────────────────────────────┘  │
│  │ • RecognitionSvc   │  ┌────────────────────────────┐  │
│  │ • AttendanceSvc    │  │    Database (SQLite)        │  │
│  │ • SpringBootSvc    │  │ • students                 │  │
│  └────────────────────┘  │ • attendance_records       │  │
└──────────────────────────┴────────────────────────────┘  │
                           └────────────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   Spring Boot API   │
                              │ POST /api/attendance│
                              │         /save       │
                              └─────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer          | Technology              | Version    |
|----------------|-------------------------|------------|
| Framework      | FastAPI                 | 0.111.0    |
| Server         | Uvicorn                 | 0.29.0     |
| CV             | OpenCV                  | 4.9.0      |
| Face AI        | face_recognition (dlib) | 1.3.0      |
| Numerics       | NumPy                   | 1.26.4     |
| Image          | Pillow                  | 10.3.0     |
| ORM            | SQLAlchemy              | 2.0.30     |
| Validation     | Pydantic v2             | 2.7.1      |
| HTTP Client    | Requests                | 2.31.0     |
| Python         | CPython                 | 3.11+      |

---

## 📁 Project Structure

```
python-face-service/
│── app.py                    # FastAPI app factory + entry point
│── config.py                 # Pydantic Settings (all config values)
│── database.py               # SQLAlchemy engine, session, Base
│── startup.py                # Startup & shutdown lifecycle tasks
│── requirements.txt          # Python dependencies
│── .env                      # Environment variables (do not commit)
│── Dockerfile                # Multi-stage Docker build
│── docker-compose.yml        # Docker Compose orchestration
│── .gitignore
│── README.md
│
├── api/                      # FastAPI route handlers (Controllers)
│   ├── health.py             # GET  /health
│   ├── camera.py             # GET  /camera
│   ├── register.py           # POST /register-face
│   ├── recognize.py          # POST /recognize
│   └── attendance.py         # POST /attendance, GET /students, etc.
│
├── services/                 # Business logic layer
│   ├── camera_service.py     # Webcam lifecycle management
│   ├── detection_service.py  # Face bounding box detection
│   ├── encoding_service.py   # 128-D face encoding generation & storage
│   ├── recognition_service.py# Face-to-encoding comparison
│   ├── attendance_service.py # Attendance orchestration pipeline
│   └── springboot_service.py # HTTP client for Spring Boot sync
│
├── models/                   # SQLAlchemy ORM models
│   ├── student.py
│   ├── attendance.py
│   └── response_models.py    # Pydantic response schemas
│
├── schemas/                  # Pydantic request schemas
│   ├── register_request.py
│   ├── attendance_request.py
│   └── recognize_response.py
│
├── repository/               # Data access layer
│   ├── encoding_repository.py # Student CRUD
│   └── attendance_repository.py # Attendance CRUD
│
├── utils/                    # Cross-cutting utilities
│   ├── logger.py             # Rotating file + console logger
│   ├── constants.py          # Magic values & enumerations
│   ├── image_utils.py        # Frame processing helpers
│   └── file_utils.py         # Disk I/O helpers
│
├── uploads/students/         # Captured student images
├── uploads/temp/             # Temporary images
├── encodings/                # face_recognition pickle files
├── logs/                     # Rotating log files
└── tests/                    # pytest test suite
```

---

## 🚀 Installation

### Prerequisites

| Requirement       | Minimum Version | Notes                              |
|-------------------|-----------------|------------------------------------|
| Python            | 3.11            | Use Python 3.11 only for this project |
| Webcam            | Any USB/built-in| For registration & recognition     |

#### Windows
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev
```

#### macOS
```bash
brew install cmake
xcode-select --install
```

---

## 🐍 Virtual Environment Setup

```powershell
# 1. Navigate to project directory
cd python-face-service

# 2. Create a Python 3.11 virtual environment
py -3.11 -m venv .venv311

# 3. Activate it (Windows PowerShell)
.\.venv311\Scripts\Activate.ps1

# 4. Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# 5. Install dependencies
pip install -r requirements.txt
```

> ✅ The service is configured to run on Python 3.11 without a manual CMake setup for the default path.

---

## ⚙️ Environment Configuration

Copy the sample `.env` file and adjust values:

```bash
cp .env .env.local   # optional: keep a local override
```

| Variable                        | Default                         | Description                           |
|---------------------------------|---------------------------------|---------------------------------------|
| `CAMERA_INDEX`                  | `0`                             | OpenCV camera device index            |
| `CAPTURE_COUNT`                 | `20`                            | Number of images per registration     |
| `RECOGNITION_THRESHOLD`         | `0.50`                          | Face distance threshold (lower=strict)|
| `DETECTION_MODEL`               | `hog`                           | `hog` (CPU) or `cnn` (GPU/CUDA)       |
| `DATABASE_URL`                  | `sqlite:///./face_service.db`   | SQLAlchemy connection string          |
| `SPRING_BOOT_BASE_URL`          | `http://localhost:8080`         | Spring Boot server URL                |
| `SPRING_BOOT_ATTENDANCE_ENDPOINT`| `/api/attendance/save`         | Attendance save endpoint path         |
| `LOG_LEVEL`                     | `INFO`                          | Logging verbosity                     |

---

## ▶️ Running the Service

### Development (with auto-reload)
```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Run via uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# OR run directly
python app.py
```

### Production
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

> 💡 Use `--workers 1` because OpenCV camera access is not process-safe across multiple workers.

### Access the API
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🧪 Running Tests

```bash
# Install test dependencies (included in requirements.txt)
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

---

## 🐳 Docker Setup

### Build and Run

```bash
# Build and start all services
docker compose up --build

# Run in background
docker compose up -d --build

# View logs
docker compose logs -f face-service

# Stop services
docker compose down

# Remove volumes (deletes all data!)
docker compose down -v
```

### Docker on Windows (Camera Access)

> ⚠️ **Camera access from Docker on Windows is not supported** out of the box.
> 
> **Recommended approach for Windows**: Run the service natively with the virtual environment and use Docker only for production Linux deployments.

---

## 📡 API Reference

### `GET /health`
Health check returning service and camera status.

```json
{
  "status": "UP",
  "camera": "CONNECTED",
  "service": "FACE RECOGNITION",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

### `GET /camera`
Test webcam connectivity.

```json
{
  "status": "CONNECTED",
  "message": "Camera Connected",
  "camera_index": 0
}
```

---

### `POST /register-face`
Register a student's face by capturing images from the webcam.

**Request Body:**
```json
{
  "student_id": "STU001",
  "student_name": "John Doe",
  "department": "Computer Science",
  "semester": "3",
  "section": "A",
  "teacher_id": "TEACH001"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Registration Successful",
  "student_id": "STU001",
  "images_captured": 20,
  "encoding_generated": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

> 📷 **Process**: Opens webcam → captures 20 images automatically → generates encodings → saves to DB.

---

### `POST /recognize`
Identify a person from the webcam.

**Response:**
```json
{
  "success": true,
  "result": {
    "student_id": "STU001",
    "name": "John Doe",
    "department": "Computer Science",
    "semester": "3",
    "section": "A",
    "confidence": 0.94,
    "status": "RECOGNIZED",
    "recognition_time_ms": 342.7
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

---

### `POST /attendance`
Mark attendance by recognising the face in front of the webcam.

**Request Body:**
```json
{
  "subject": "Data Structures",
  "teacher_id": "TEACH001",
  "session": "Morning"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Attendance Marked",
  "student_id": "STU001",
  "student_name": "John Doe",
  "attendance_date": "2024-01-15",
  "attendance_time": "10:30:00",
  "subject": "Data Structures",
  "confidence": 0.94,
  "already_marked": false,
  "synced_to_spring": false
}
```

---

### `GET /students`
List all registered students.

```json
{
  "total": 3,
  "students": [
    {
      "student_id": "STU001",
      "name": "John Doe",
      "department": "Computer Science",
      "semester": "3",
      "section": "A",
      "teacher_id": "TEACH001",
      "image_count": 20,
      "is_active": true,
      "registered_at": "2024-01-15T10:00:00"
    }
  ]
}
```

---

### `GET /encodings`
List all stored face encoding files.

```json
{
  "total": 1,
  "encodings": [
    {
      "student_id": "STU001",
      "encoding_file": "STU001.pickle",
      "face_count": 18,
      "file_size_bytes": 21432
    }
  ]
}
```

---

### `DELETE /student/{id}`
Remove a student and all their face data.

```json
{
  "success": true,
  "message": "Student STU001 successfully removed from the system.",
  "student_id": "STU001",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🔗 Spring Boot Integration

After attendance is successfully marked, the service automatically calls:

```
POST http://localhost:8080/api/attendance/save
```

**Payload:**
```json
{
  "studentId": "STU001",
  "studentName": "John Doe",
  "date": "2024-01-15",
  "time": "10:30:00",
  "subject": "Data Structures",
  "teacherId": "TEACH001",
  "session": "Morning",
  "confidence": 0.9412,
  "status": "PRESENT"
}
```

> 🔄 The sync is **fire-and-forget** (async background task). Attendance is saved locally even if Spring Boot is unreachable.

---

## 🔧 Troubleshooting

### `dlib` installation fails on Windows
```powershell
# Install Visual Studio C++ Build Tools
# Then install dlib with pre-built wheel:
pip install dlib==19.24.4
```

### Camera not detected
```bash
# Test camera index
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"

# Try different indices
CAMERA_INDEX=1   # in .env
CAMERA_INDEX=2
```

### Low recognition accuracy
- Increase `CAPTURE_COUNT` to 30 or more.
- Ensure good lighting during registration.
- Lower `RECOGNITION_THRESHOLD` (e.g., `0.45`) for stricter matching.
- Use `DETECTION_MODEL=cnn` if you have a GPU.

### Database reset
```bash
rm face_service.db       # Linux/macOS
del face_service.db      # Windows
# Restart the service – tables are recreated automatically
```

### Port already in use
```bash
# Kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
# Linux:
lsof -ti:8000 | xargs kill -9
```

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the AI Smart Classroom Management System.*
