"""
utils/constants.py – Application-wide constants.

Centralise every magic value here to keep the rest of the codebase clean.
"""

from enum import Enum


# ── Face Detection / Recognition ──────────────────────────────────────────────

FACE_DETECTION_SCALE_FACTOR: float = 1.1
FACE_DETECTION_MIN_NEIGHBORS: int = 5
FACE_DETECTION_MIN_SIZE: tuple[int, int] = (30, 30)

# Maximum number of faces allowed during registration (exactly 1)
MAX_FACES_REGISTRATION: int = 1

# Minimum and maximum face area relative to frame (fraction)
MIN_FACE_AREA_FRACTION: float = 0.02
MAX_FACE_AREA_FRACTION: float = 0.90

# Live-view display
FONT_SCALE: float = 0.7
FONT_THICKNESS: int = 2
RECT_THICKNESS: int = 2
RECT_COLOR_KNOWN: tuple[int, int, int] = (0, 255, 0)    # Green – BGR
RECT_COLOR_UNKNOWN: tuple[int, int, int] = (0, 0, 255)  # Red   – BGR
RECT_COLOR_DETECTING: tuple[int, int, int] = (255, 165, 0)  # Orange

# ── Image Capture ─────────────────────────────────────────────────────────────

IMAGE_EXTENSION: str = ".jpg"
JPEG_QUALITY: int = 95

# Pose labels for auto-capture rotation
POSE_LABELS: list[str] = ["front", "left", "right", "up", "down"]

# Gap between automatic captures (frames to skip)
CAPTURE_FRAME_GAP: int = 10

# ── Encoding Files ────────────────────────────────────────────────────────────

ENCODING_EXTENSION: str = ".pickle"

# ── HTTP Status Messages ──────────────────────────────────────────────────────

MSG_CAMERA_CONNECTED: str = "Camera Connected"
MSG_CAMERA_NOT_FOUND: str = "Camera Not Found"
MSG_REGISTRATION_SUCCESS: str = "Registration Successful"
MSG_ATTENDANCE_MARKED: str = "Attendance Marked"
MSG_ATTENDANCE_DUPLICATE: str = "Attendance Already Marked Today"
MSG_NO_FACE_FOUND: str = "No face detected in frame"
MSG_MULTIPLE_FACES: str = "Multiple faces detected – please ensure only one person is in frame"
MSG_LOW_CONFIDENCE: str = "Face recognized but confidence is below threshold"
MSG_STUDENT_NOT_FOUND: str = "Student not found in the system"
MSG_ENCODING_NOT_FOUND: str = "No face encodings found for this student"

# ── Spring Boot ───────────────────────────────────────────────────────────────

ATTENDANCE_STATUS_PRESENT: str = "PRESENT"
ATTENDANCE_STATUS_ABSENT: str = "ABSENT"


# ── Enumerations ──────────────────────────────────────────────────────────────

class CameraStatus(str, Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    NOT_FOUND = "NOT_FOUND"


class ServiceStatus(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class RecognitionStatus(str, Enum):
    RECOGNIZED = "RECOGNIZED"
    UNKNOWN = "UNKNOWN"
    NO_FACE = "NO_FACE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
