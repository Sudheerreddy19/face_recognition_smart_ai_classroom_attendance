"""
face_recognition_engine_test.py – Phase 2: MediaPipe face recognition engine test.

Purpose
-------
Verify that the face recognition engine (MediaPipe via face_compat.py) works
end-to-end on the laptop camera:

  Camera frame → Detect faces → Generate encoding → Compare encodings

What this test does
-------------------
1. Prints environment summary (Python, OpenCV, MediaPipe, NumPy).
2. Opens the laptop camera.
3. Detects faces using face_compat.face_locations() (MediaPipe under the hood).
4. Generates a face encoding using face_compat.face_encodings().
5. On first enrollment (press E), stores the encoding as "reference".
6. On subsequent frames, compares live encoding against the reference.
7. Displays distance value and MATCH / NO MATCH on screen.
8. Press Q to quit.

What this test does NOT do
--------------------------
- Does NOT save encodings to disk.
- Does NOT write to any database.
- Does NOT communicate with Spring Boot.
- Does NOT mark attendance.

Run command (from project root, with .venv311 active)
------------------------------------------------------
    .venv311\\Scripts\\python face_recognition_engine_test.py

Expected output
---------------
    Environment summary printed.
    Camera window opens.
    Green rectangle + encoding distance shown when a face is detected.
    After pressing E: distance drops towards 0.00 for the same face.
"""

import sys
import time

import cv2
import numpy as np

# ── 1. Environment summary ────────────────────────────────────────────────────

print()
print("=" * 58)
print("  PHASE 2 — FACE RECOGNITION ENGINE TEST (MediaPipe)")
print("=" * 58)
print(f"[INFO] Python    : {sys.version.split()[0]}")
print(f"[INFO] OpenCV    : {cv2.__version__}")

try:
    import mediapipe as mp
    print(f"[INFO] MediaPipe : {mp.__version__}")
except ImportError:
    print("[ERROR] MediaPipe NOT installed.")
    print("        Run: .venv311\\Scripts\\pip install mediapipe==0.10.14")
    sys.exit(1)

try:
    import numpy as np
    print(f"[INFO] NumPy     : {np.__version__}")
except ImportError:
    print("[ERROR] NumPy NOT installed.")
    sys.exit(1)

# ── 2. Import face_compat (our MediaPipe wrapper) ─────────────────────────────

try:
    from utils.face_compat import (
        face_locations,
        face_encodings,
        face_distance,
    )
    print("[INFO] face_compat : OK (MediaPipe backend)")
except Exception as e:
    print(f"[ERROR] Could not import face_compat: {e}")
    sys.exit(1)

print()

# ── 3. Open camera ────────────────────────────────────────────────────────────

CAMERA_INDEX = 0
print(f"[INFO] Opening camera (index {CAMERA_INDEX}) ...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"[ERROR] Cannot open camera at index {CAMERA_INDEX}.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"[INFO] Camera opened — {int(cap.get(3))}x{int(cap.get(4))}")
print()
print("[INFO] Controls:")
print("         E — enroll current frame as reference")
print("         Q — quit")
print()

# ── 4. State ──────────────────────────────────────────────────────────────────

reference_encoding: np.ndarray | None = None
MATCH_THRESHOLD = 0.50   # L2 distance; tune after enrollment

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (0, 255,   0)
COLOR_RED    = (0,   0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

fps_timer = time.time()
fps = 0.0
frame_count = 0

# ── 5. Main loop ──────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Failed to read frame.")
        break

    frame_count += 1
    elapsed = time.time() - fps_timer
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_timer = time.time()

    # Convert BGR → RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    locations = face_locations(rgb)
    face_count = len(locations)

    current_encoding: np.ndarray | None = None

    if face_count == 1:
        top, right, bottom, left = locations[0]

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_GREEN, 2)

        # Generate encoding
        encs = face_encodings(rgb, known_face_locations=locations)
        if encs:
            current_encoding = encs[0]

            if reference_encoding is not None:
                dist_arr = face_distance([reference_encoding], current_encoding)
                dist = float(dist_arr[0])
                match = dist <= MATCH_THRESHOLD
                label_color = COLOR_GREEN if match else COLOR_RED
                match_text = f"{'MATCH' if match else 'NO MATCH'}  dist={dist:.3f}"
                cv2.putText(frame, match_text, (left, top - 10),
                            FONT, 0.65, label_color, 2)
            else:
                cv2.putText(frame, "Press E to enroll", (left, top - 10),
                            FONT, 0.65, COLOR_YELLOW, 2)

    # Status overlay
    if face_count == 0:
        status = "No Face"
        sc = COLOR_RED
    elif face_count == 1:
        status = "Face Detected"
        sc = COLOR_GREEN
    else:
        status = f"Multiple Faces ({face_count}) — move others away"
        sc = COLOR_ORANGE

    cv2.putText(frame, status, (10, 30), FONT, 0.7, sc, 2)

    # Reference enrolled indicator
    if reference_encoding is not None:
        cv2.putText(frame, "Reference: ENROLLED", (10, 60),
                    FONT, 0.55, COLOR_GREEN, 1)
    else:
        cv2.putText(frame, "Reference: NOT YET (press E)", (10, 60),
                    FONT, 0.55, COLOR_YELLOW, 1)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 460),
                FONT, 0.5, COLOR_WHITE, 1)

    cv2.imshow("Phase 2 — Face Recognition Engine Test (E=enroll, Q=quit)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    if key == ord("e"):
        if face_count == 1 and current_encoding is not None:
            reference_encoding = current_encoding
            print("[INFO] Reference encoding enrolled for this session.")
            print(f"       Encoding shape : {reference_encoding.shape}")
            print(f"       Encoding norm  : {np.linalg.norm(reference_encoding):.4f}  (should be ~1.0)")
            print(f"       Threshold      : {MATCH_THRESHOLD}")
        elif face_count == 0:
            print("[WARN] No face detected — cannot enroll.")
        else:
            print("[WARN] Multiple faces detected — move others out of frame first.")

# ── 6. Cleanup ────────────────────────────────────────────────────────────────

cap.release()
cv2.destroyAllWindows()
print()
print("[INFO] Camera released. Phase 2 test finished.")
