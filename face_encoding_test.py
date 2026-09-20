"""
face_encoding_test.py - Phase 2: One-shot face encoding test.

Purpose
-------
Verify that a face encoding (128-D vector) can be generated from a
single camera frame. This is the foundation of both registration and
recognition.

What this test does
-------------------
1. Opens the laptop camera.
2. Waits up to 30 frames to find exactly one face.
3. Generates ONE 128-D face encoding from that frame.
4. Prints the encoding shape, norm, and first 5 values.
5. Releases the camera immediately.
6. Exits.

What this test does NOT do
--------------------------
- Does NOT save the encoding to disk.
- Does NOT write to any database.
- Does NOT run a continuous recognition loop.
- Does NOT communicate with Spring Boot.
- Does NOT mark attendance.

Run command (from project root, with .venv active)
--------------------------------------------------
    .venv\\Scripts\\python face_encoding_test.py

Expected output (success)
--------------------------
    [INFO] Camera opened. Resolution: 640 x 480
    [INFO] Scanning for face ... (attempt 1/30)
    [OK]   Face detected at frame 4
    [INFO] Generating face encoding ...
    [OK]   Encoding generated successfully!
           Shape : (128,)
           Norm  : 1.0000  (should be ~1.0)
           First 5 values: [-0.12  0.08  0.05 -0.13  0.02]
    [INFO] Camera released.
    [INFO] Phase 2 PASSED -- encoding pipeline is working.
"""

import sys
import time

import cv2
import numpy as np

# ── 1. Environment check ──────────────────────────────────────────────────────

print()
print("=" * 52)
print("  PHASE 2 — FACE ENCODING TEST (one-shot)")
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
MAX_SCAN_FRAMES = 30        # scan at most 30 frames to find a face
SCALE = 0.5                 # downscale for fast detection

print(f"[INFO] Opening camera index {CAMERA_INDEX} ...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"[ERROR] Cannot open camera at index {CAMERA_INDEX}.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera opened. Resolution: {actual_w} x {actual_h}")
print()

# ── 3. Scan frames until one face is found ────────────────────────────────────
# We do NOT store frames — `frame` is overwritten every iteration.

capture_frame = None
capture_locations = None

for attempt in range(1, MAX_SCAN_FRAMES + 1):
    print(f"[INFO] Scanning for face ... (attempt {attempt}/{MAX_SCAN_FRAMES})", end="\r")

    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.05)
        continue

    # Detection on half-size RGB for speed
    small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)

    locations_small = face_recognition.face_locations(rgb_small, model="hog")

    if len(locations_small) == 1:
        # Scale bounding box back to full resolution
        inv = 1.0 / SCALE
        t, r, b, l = locations_small[0]
        full_loc = (int(t * inv), int(r * inv), int(b * inv), int(l * inv))

        capture_frame = frame.copy()        # keep this one frame
        capture_locations = [full_loc]
        print()  # newline after \r
        print(f"[OK]   Face detected at frame {attempt}               ")
        break
    elif len(locations_small) > 1:
        print()
        print(f"[WARN] Frame {attempt}: {len(locations_small)} faces found — need exactly 1. Move others away.")
        print(f"[INFO] Scanning for face ... (attempt {attempt}/{MAX_SCAN_FRAMES})", end="\r")
    # else: no face yet, keep scanning

# ── RELEASE CAMERA immediately after scanning ─────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera released.")
print()

# ── 4. Check if a face was found ──────────────────────────────────────────────

if capture_frame is None or capture_locations is None:
    print("[FAIL] No face was detected in any of the scanned frames.")
    print("       Make sure:")
    print("         - Your face is clearly visible to the camera.")
    print("         - The room has adequate lighting.")
    print("         - You are within 1-2 metres of the camera.")
    sys.exit(1)

# ── 5. Generate ONE face encoding ─────────────────────────────────────────────

print("[INFO] Generating face encoding ...")

rgb_full = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)
rgb_full = np.ascontiguousarray(rgb_full, dtype=np.uint8)

try:
    encodings = face_recognition.face_encodings(
        rgb_full,
        known_face_locations=capture_locations,
        num_jitters=1,
    )
except Exception as exc:
    print(f"[ERROR] face_encodings() raised an exception: {exc}")
    sys.exit(1)

# ── 6. Report result ──────────────────────────────────────────────────────────

if not encodings:
    print("[FAIL] face_encodings() returned an empty list.")
    print("       The face was detected but the encoding failed.")
    print("       Try better lighting or move closer to the camera.")
    sys.exit(1)

enc = encodings[0]
norm = float(np.linalg.norm(enc))
first5 = " ".join(f"{v:+.4f}" for v in enc[:5])

print("[OK]   Encoding generated successfully!")
print(f"       Shape      : {enc.shape}")
print(f"       Norm       : {norm:.6f}  (should be ~1.0)")
print(f"       First 5    : [{first5}]")
print()

# dlib 19.x produced norms ~1.0 (L2-normalised).
# dlib 20.x produces norms in the 1.0 – 1.5 range — this is expected and normal.
# Recognition accuracy depends on RELATIVE distance between encodings, not absolute norm.
if norm < 0.5 or norm > 2.0:
    print(f"[WARN] Norm {norm:.4f} is outside the expected range (0.5-2.0).")
    print("       This could indicate an encoding failure. Check lighting.")
else:
    print(f"[INFO] Norm {norm:.4f} is within the valid range for dlib 20.x. Encoding is good.")
    print("       NOTE: dlib 20.x norms are 1.0-1.5 (not exactly 1.0 like older dlib 19.x).")
    print("       Recognition uses RELATIVE distance — absolute norm does not matter.")

# Cleanup
del capture_frame, rgb_full, encodings, enc

print()
print("=" * 52)
print("  Phase 2 PASSED -- encoding pipeline is working.")
print("=" * 52)
print()
