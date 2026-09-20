"""
utils/image_utils.py – Image pre-processing and manipulation helpers.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB.

    Args:
        frame: NumPy array in BGR colour order.

    Returns:
        NumPy array in RGB colour order.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert RGB array to OpenCV BGR.

    Args:
        frame: NumPy array in RGB colour order.

    Returns:
        NumPy array in BGR colour order.
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def resize_frame(
    frame: np.ndarray, width: int = 640, height: int = 480
) -> np.ndarray:
    """Resize *frame* to the target *width* × *height*.

    Args:
        frame:  Source frame.
        width:  Target width in pixels.
        height: Target height in pixels.

    Returns:
        Resized frame.
    """
    return cv2.resize(frame, (width, height))


def resize_for_recognition(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Shrink *frame* by *scale* factor for faster recognition.

    Args:
        frame: Full-resolution frame.
        scale: Downscale factor (default 0.5 = half-resolution).

    Returns:
        Down-scaled frame.
    """
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def draw_face_rectangle(
    frame: np.ndarray,
    top: int,
    right: int,
    bottom: int,
    left: int,
    label: str,
    color: tuple[int, int, int],
    confidence: float | None = None,
) -> np.ndarray:
    """Draw a labelled bounding box around a detected face.

    Args:
        frame:      Frame to draw on (modified in-place).
        top:        Top pixel coordinate of the face.
        right:      Right pixel coordinate.
        bottom:     Bottom pixel coordinate.
        left:       Left pixel coordinate.
        label:      Text label (e.g. student name).
        color:      BGR colour tuple for the rectangle.
        confidence: Optional confidence score to display alongside the label.

    Returns:
        Frame with the rectangle and label drawn.
    """
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    display = label
    if confidence is not None:
        display = f"{label} ({confidence:.1%})"

    # Background rectangle for text legibility
    (text_w, text_h), baseline = cv2.getTextSize(
        display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        frame,
        (left, bottom - text_h - baseline - 4),
        (left + text_w, bottom),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        display,
        (left, bottom - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS counter in the top-left corner of *frame*.

    Args:
        frame: Frame to draw on (modified in-place).
        fps:   Current frames-per-second value.

    Returns:
        Frame with FPS overlay.
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return frame


def draw_capture_progress(frame: np.ndarray, captured: int, total: int) -> np.ndarray:
    """Overlay capture progress indicator on *frame*.

    Args:
        frame:    Frame to draw on (modified in-place).
        captured: Number of images already captured.
        total:    Total number of images to capture.

    Returns:
        Frame with progress overlay.
    """
    text = f"Captured: {captured}/{total}"
    cv2.putText(
        frame,
        text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return frame


def frame_to_bytes(frame: np.ndarray, extension: str = ".jpg") -> bytes:
    """Encode *frame* to JPEG bytes.

    Args:
        frame:     BGR frame.
        extension: OpenCV image format extension (default ``.jpg``).

    Returns:
        Encoded image bytes.
    """
    success, buffer = cv2.imencode(extension, frame)
    if not success:
        raise ValueError("Failed to encode frame to bytes")
    return buffer.tobytes()


def bytes_to_frame(data: bytes) -> np.ndarray:
    """Decode raw image *data* bytes into an OpenCV BGR frame.

    Args:
        data: Raw image bytes (JPEG, PNG, etc.).

    Returns:
        Decoded BGR frame.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image bytes to frame")
    return frame


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a :class:`PIL.Image` to an OpenCV BGR numpy array.

    Args:
        pil_image: PIL Image object.

    Returns:
        OpenCV-compatible BGR numpy array.
    """
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to a :class:`PIL.Image`.

    Args:
        frame: OpenCV BGR numpy array.

    Returns:
        PIL Image in RGB mode.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_frame(frame: np.ndarray, path: Path, quality: int = 95) -> None:
    """Save *frame* to *path* as a JPEG with specified *quality*.

    Args:
        frame:   BGR frame to save.
        path:    Destination file path (parent dirs must exist).
        quality: JPEG quality 0–100 (default 95).
    """
    success = cv2.imwrite(
        str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise IOError(f"Failed to save frame to {path}")
    logger.debug("Saved frame to %s", path)


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert a frame to a Base64-encoded JPEG string.

    Args:
        frame: BGR frame.

    Returns:
        Base64-encoded string.
    """
    img_bytes = frame_to_bytes(frame)
    return base64.b64encode(img_bytes).decode("utf-8")


def load_image_as_rgb(path: Path) -> np.ndarray:
    """Load an image from *path* and return it as an RGB numpy array.

    Args:
        path: Path to the image file.

    Returns:
        RGB numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the image cannot be decoded.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Could not decode image: {path}")
    return bgr_to_rgb(frame)
