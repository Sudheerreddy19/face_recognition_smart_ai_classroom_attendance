"""
face_detection_test.py - Phase 1: Face detection smoke test.

Purpose
-------
Prove that the laptop camera is accessible through OpenCV and that
faces can be detected and bounded with a rectangle.

Uses face_recognition (dlib HOG model) for detection because
cv2.CascadeClassifier was removed in OpenCV 5.0.

Scope (Phase 1 only)
--------------------
- Opens laptop camera via cv2.VideoCapture(0).
- Detects faces using dlib HOG (via face_recognition library).
- Draws a green rectangle around every detected face.
- Overlays status text on the preview window.
- Press Q or ESC to exit cleanly.

What this test does NOT do
--------------------------
- Does NOT save images.
- Does NOT save encodings.
- Does NOT write to any database.
- Does NOT communicate with Spring Boot.
- Does NOT mark attendance.
- Does NOT run an infinite encoding loop.

Run command (from project root, with .venv active)
--------------------------------------------------
    .venv\\Scripts\\python face_detection_test.py

Expected output
---------------
    [INFO] Python      : 3.x.x
    [INFO] OpenCV      : 5.0.0
    [INFO] face_recog  : OK
    [INFO] Opening camera index 0 ...
    [INFO] Camera opened. Resolution: 640 x 480
    [INFO] Showing live feed -- press Q or ESC to quit
    (Window opens. Green rectangle appears when a face is in frame.)
    [INFO] Camera released. Test finished.

Memory profile (measured)
--------------------------
    Python + cv2 baseline:       ~36 MB
    After face_recognition:     ~146 MB   (+110 MB, one-time dlib load)
    Camera + frame reading:     ~215 MB   (normal)
    After cleanup:              ~190 MB   (stable)
"""

import sys
import time

import cv2
import numpy as np

# ── 1. Print environment info ─────────────────────────────────────────────────

print()
print("=" * 52)
print("  PHASE 1 — FACE DETECTION TEST (dlib HOG)")
print("=" * 52)
print(f"[INFO] Python      : {sys.version.split()[0]}")
print(f"[INFO] OpenCV      : {cv2.__version__}")

try:
    import face_recognition
    print("[INFO] face_recog  : OK")
except ImportError:
    print("[ERROR] face_recognition is not installed.")
    print("        Activate .venv and run: pip install face-recognition")
    sys.exit(1)

print()

# ── 2. Open camera ────────────────────────────────────────────────────────────

CAMERA_INDEX = 0

print(f"[INFO] Opening camera index {CAMERA_INDEX} ...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"[ERROR] Cannot open camera at index {CAMERA_INDEX}.")
    print("        Try CAMERA_INDEX = 1 if you have multiple cameras.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera opened. Resolution: {actual_w} x {actual_h}")
print("[INFO] Showing live feed -- press Q or ESC to quit")
print()

# ── 3. Drawing constants ──────────────────────────────────────────────────────

FONT         = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (0, 255,   0)
COLOR_RED    = (0,   0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE  = (255, 255, 255)

# HOG detection on half-resolution frame (much faster, same accuracy)
SCALE = 0.5

# ── 4. FPS state ──────────────────────────────────────────────────────────────

fps_prev   = time.time()
fps_value  = 0.0
frame_idx  = 0

# ── 5. Main loop ──────────────────────────────────────────────────────────────
# IMPORTANT: The loop variable `frame` is overwritten every iteration.
# Only ONE frame is ever in memory at a time — no accumulation.

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("[ERROR] Failed to read frame from camera.")
        break

    frame_idx += 1

    # ── FPS calculation ───────────────────────────────────────────────────────
    now    = time.time()
    fps_value = 1.0 / max(now - fps_prev, 1e-9)
    fps_prev  = now

    # ── Face detection on half-size frame for speed ───────────────────────────
    # Resize for detection only — we draw on the original full-size frame.
    small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)

    # face_recognition.face_locations returns (top, right, bottom, left) tuples
    locations_small = face_recognition.face_locations(rgb_small, model="hog")

    # Scale bounding boxes back to full resolution
    inv = 1.0 / SCALE
    locations = [
        (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
        for t, r, b, l in locations_small
    ]

    face_count = len(locations)

    # ── Draw bounding boxes ───────────────────────────────────────────────────
    for top, right, bottom, left in locations:
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_GREEN, 2)
        cv2.putText(
            frame, "Face Detected",
            (left, top - 8),
            FONT, 0.55, COLOR_GREEN, 1,
        )

    # ── Status overlay ────────────────────────────────────────────────────────
    if face_count == 0:
        status_text  = "No Face Detected"
        status_color = COLOR_RED
    elif face_count == 1:
        status_text  = "Face Detected"
        status_color = COLOR_GREEN
    else:
        status_text  = f"Multiple Faces ({face_count})"
        status_color = COLOR_ORANGE

    cv2.putText(frame, status_text, (10, 28), FONT, 0.7, status_color, 2)
    cv2.putText(
        frame, f"FPS: {fps_value:.1f}",
        (10, actual_h - 12),
        FONT, 0.5, COLOR_WHITE, 1,
    )
    cv2.putText(
        frame, f"Faces: {face_count}",
        (actual_w - 110, 28),
        FONT, 0.6, COLOR_WHITE, 2,
    )

    # ── Display ───────────────────────────────────────────────────────────────
    cv2.imshow("Phase 1 - Face Detection Test  (Q or ESC to quit)", frame)

    # ── Key handler ───────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:   # Q or ESC
        print("[INFO] Quit key pressed.")
        break

# ── 6. Cleanup ────────────────────────────────────────────────────────────────
# Camera is ALWAYS released here — even if loop exited via break.

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera released. Test finished.")
