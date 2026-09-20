"""
memory_test.py – Safe memory diagnostic for the face recognition service.

PURPOSE
-------
Find EXACTLY which operation causes the large memory increase.
This script does NOT start FastAPI, does NOT run uvicorn, does NOT
run any camera loops.  It measures memory at each step, releases
everything cleanly, and reports the delta.

USAGE
-----
    python memory_test.py

REQUIREMENTS
------------
    pip install psutil   (already in requirements.txt)
"""

from __future__ import annotations

import gc
import os
import sys
import time

# ── psutil ────────────────────────────────────────────────────────────────────

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed.  Run:  pip install psutil")
    sys.exit(1)

_PROC = psutil.Process(os.getpid())


def ram_mb() -> float:
    """Return current process RSS (Resident Set Size) in MB."""
    gc.collect()
    return _PROC.memory_info().rss / (1024 * 1024)


def report(label: str, before: float) -> float:
    """Print RAM usage and return current value."""
    now = ram_mb()
    delta = now - before
    sign = "+" if delta >= 0 else ""
    print(f"  {label:<45}  {now:7.1f} MB  ({sign}{delta:.1f} MB)")
    return now


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 70)
    print("  PYTHON FACE SERVICE – MEMORY DIAGNOSTIC")
    print("=" * 70)

    # ── BASELINE ──────────────────────────────────────────────────────────────
    cur = ram_mb()
    print(f"\n  BASELINE (Python interpreter only):  {cur:.1f} MB\n")

    # ── Step 1: numpy ─────────────────────────────────────────────────────────
    print("── Step 1: numpy ────────────────────────────────────────────────────")
    prev = cur
    import numpy as np
    cur = report("import numpy", prev)
    time.sleep(0.2)

    # ── Step 2: OpenCV ────────────────────────────────────────────────────────
    print("\n── Step 2: OpenCV ───────────────────────────────────────────────────")
    prev = cur
    import cv2
    cur = report(f"import cv2  (version {cv2.__version__})", prev)
    time.sleep(0.2)

    # ── Step 3: face_recognition (loads dlib + all .dat models) ──────────────
    print("\n── Step 3: face_recognition + dlib models ───────────────────────────")
    print("  NOTE: dlib loads 3 large .dat model files into RAM here.")
    prev = cur
    import face_recognition
    cur = report("import face_recognition (dlib + models)", prev)
    time.sleep(0.2)

    # ── Step 4: Open camera (NO loop, just open) ──────────────────────────────
    print("\n── Step 4: Open camera ──────────────────────────────────────────────")
    prev = cur
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cur = report("cv2.VideoCapture(0) opened", prev)
    else:
        print("  WARNING: Camera index 0 could not be opened. Check connection.")
        cur = report("cv2.VideoCapture(0) FAILED", prev)

    # ── Step 5: Read ONE frame ────────────────────────────────────────────────
    print("\n── Step 5: Read ONE frame ───────────────────────────────────────────")
    frame = None
    prev = cur
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            size_kb = frame.nbytes / 1024
            cur = report(f"cap.read() OK  ({w}x{h}, {size_kb:.0f} KB)", prev)
        else:
            cur = report("cap.read() FAILED", prev)

    # ── RELEASE CAMERA immediately ─────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    prev = cur
    gc.collect()
    cur = report("cap.release() + gc.collect()", prev)

    # ── Step 6: Face detection on the ONE frame ────────────────────────────────
    print("\n── Step 6: Face detection on single frame ───────────────────────────")
    if frame is not None:
        prev = cur
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        locations = face_recognition.face_locations(rgb, model="hog")
        cur = report(
            f"face_locations() → {len(locations)} face(s) found",
            prev,
        )
        del rgb
    else:
        print("  SKIPPED — no frame was captured.")

    # ── Step 7: Face encoding on the ONE frame ─────────────────────────────────
    print("\n── Step 7: Face encoding on single frame ────────────────────────────")
    if frame is not None and locations:
        prev = cur
        rgb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb2 = np.ascontiguousarray(rgb2, dtype=np.uint8)
        encodings = face_recognition.face_encodings(rgb2, known_face_locations=locations, num_jitters=1)
        cur = report(
            f"face_encodings() → {len(encodings)} encoding(s) generated",
            prev,
        )
        del rgb2, encodings
    elif frame is not None:
        print("  SKIPPED — no face was detected in Step 6.")
    else:
        print("  SKIPPED — no frame was captured.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n── Cleanup ──────────────────────────────────────────────────────────")
    prev = cur
    del frame
    gc.collect()
    time.sleep(0.3)
    cur = report("del frame + gc.collect()", prev)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    baseline = ram_mb()
    print()
    print("=" * 70)
    print(f"  FINAL RAM (after cleanup):  {cur:.1f} MB")
    print()
    print("  INTERPRETATION:")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  • Baseline (Python):          ~20–30 MB   — normal")
    print("  • After numpy:                +10–20 MB   — normal")
    print("  • After cv2:                  +30–60 MB   — normal")
    print("  • After face_recognition:     +600–900 MB — dlib loads 3 model files")
    print("                                              This is THE LARGEST jump.")
    print("  • After open camera:          +5–20 MB    — normal")
    print("  • After read frame:           +1–5 MB     — normal")
    print("  • After face detection:       +1–10 MB    — normal")
    print("  • After face encoding:        +1–10 MB    — normal")
    print()
    print("  IF 'import face_recognition' JUMPED BY >1 GB:")
    print("  → dlib is loading models — this is expected and unavoidable.")
    print("  → But it should only load ONCE per process.")
    print("  → With --reload, uvicorn restarts the process on file changes,")
    print("    causing dlib to reload.  Use --no-reload to prevent this.")
    print()
    print("  KNOWN BUG IN THIS PROJECT:")
    print("  → 3 separate modules (register.py, recognize.py, attendance.py)")
    print("    each create their OWN DetectionService/EncodingService instances.")
    print("  → Each DetectionService call loads face_recognition into memory.")
    print("  → With uvicorn --reload, this can multiply dlib memory usage.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
