"""
tests/multi_face_tracking_test.py
──────────────────────────────────
SMART CLASSROOM — Phase 2: Multi-Face Detection + Tracking Test

PURPOSE
-------
Detect ALL faces in a classroom simultaneously and assign stable
temporary tracking IDs (Face #1, Face #2, Face #3 …).

TWO MODES
---------
MODE 1 — LIVE CAMERA (real classroom use)
    python tests/multi_face_tracking_test.py

MODE 2 — STATIC IMAGE (test alone with a classroom photo)
    python tests/multi_face_tracking_test.py --image path/to/classroom.jpg

    Example:
    python tests/multi_face_tracking_test.py --image uploads/classroom_test.jpg

    Press any key to close the image window.

IMPORTANT
---------
- NO student recognition in this phase
- NO face encodings
- NO database
- NO Spring Boot
- NO FastAPI
- Face #1, #2, #3 are TEMPORARY tracking IDs — NOT student register numbers

USAGE — CAMERA
--------------
    python tests/multi_face_tracking_test.py

USAGE — IMAGE
-------------
    python tests/multi_face_tracking_test.py --image <path_to_image>

KEYBOARD CONTROLS
-----------------
    Q / ESC  — quit (camera mode)
    Any key  — close (image mode)

CONFIGURATION
-------------
    CAMERA_INDEX    Camera device index
    DETECT_SCALE    Fraction of frame for detection (0.5 = half-size, faster)
    UPSAMPLE        1=fast, 2=catches smaller/farther faces (slower)
    MAX_DISAPPEARED Frames before a lost track is dropped
    IOU_THRESHOLD   Minimum IoU to consider two boxes the same face
    MIN_FACE_PX     Face width/height below this → labelled LOW QUALITY / FAR

NOTE ON DETECTION RANGE
-----------------------
dlib HOG detector works best on real photographic faces.
At DETECT_SCALE=0.5 and UPSAMPLE=1 it reliably detects faces roughly
≥ 80 px wide in the full frame (students within ~2-3 metres of the camera).
For students farther away, increase UPSAMPLE to 2 (slower but more sensitive).
Cartoon / illustrated images may not be detected — use real classroom photos.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

# ── psutil for RAM monitoring ─────────────────────────────────────────────────
try:
    import psutil as _psutil
    _PROC = _psutil.Process(os.getpid())
    def _ram_mb() -> float:
        return _PROC.memory_info().rss / (1024 * 1024)
except ImportError:
    def _ram_mb() -> float:
        return 0.0

# ── face_recognition (dlib HOG) ───────────────────────────────────────────────
try:
    import face_recognition as _fr
    _FR_OK = True
except ImportError:
    print("[ERROR] face_recognition is not installed.")
    print("        Activate .venv:  .venv\\Scripts\\activate")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION — edit these values to tune the detector
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX    = 0
FRAME_W         = 640
FRAME_H         = 480

# Detection scale.
# 1.0 = full resolution → detects small/far faces (students at back of class).
# 0.5 = half resolution → faster but misses small faces.
# For classroom use: keep at 1.0 so background students are detected.
DETECT_SCALE    = 1.0

# HOG upsampling.
# 1 = fast, detects faces roughly ≥ 80 px wide.
# 2 = slower (~3× CPU), detects faces roughly ≥ 40 px wide.
#     Use 2 for classrooms where students sit far from the camera.
UPSAMPLE        = 2

# Detect every N frames — tracker runs every frame for smooth display.
# Detection is the slow step (dlib HOG). Tracking is fast (pure Python).
# 1 = detect every frame (most accurate, lowest FPS)
# 3 = detect every 3rd frame (recommended — good accuracy + smooth display)
# 5 = detect every 5th frame (fastest, may miss brief appearances)
DETECT_EVERY_N  = 3

# Tracker parameters
MAX_DISAPPEARED = 30     # frames before a lost track is removed
IOU_THRESHOLD   = 0.20   # minimum IoU to match detection to existing track

# Face quality label threshold (full-resolution pixels)
MIN_FACE_PX     = 60     # faces smaller than this → "LOW QUALITY / FAR"

# Drawing colours (BGR)
COLOR_TRACKED   = (0,  220,  0)    # green  — tracked face
COLOR_NEW       = (0,  180, 255)   # orange — newly appeared this frame
COLOR_FAR       = (0,  165, 255)   # amber  — small / far face
COLOR_TEXT      = (255, 255, 255)  # white  — overlay text
COLOR_BG        = (30,  30,  30)   # dark   — status bar background

FONT     = cv2.FONT_HERSHEY_SIMPLEX
WIN_LIVE = "Smart Classroom — Multi-Face Tracking  (Q/ESC to quit)"
WIN_IMG  = "Smart Classroom — Multi-Face Image Test  (any key to close)"


# ─────────────────────────────────────────────────────────────────────────────
#  IoU TRACKER  (pure Python, zero extra dependencies)
# ─────────────────────────────────────────────────────────────────────────────

class IouTracker:
    """Lightweight IoU-based tracker.

    Assigns stable integer IDs to bounding boxes across frames using
    greedy IoU matching. Works well for ≤ 50 faces (classroom scale).
    """

    def __init__(self, max_disappeared: int = MAX_DISAPPEARED,
                 iou_threshold: float = IOU_THRESHOLD) -> None:
        self._next_id  = 1
        self._tracks: dict[int, dict] = {}   # id → {box, disappeared, is_new}
        self._max_dis  = max_disappeared
        self._iou_thr  = iou_threshold

    def update(self, detections: list[tuple]) -> dict[int, tuple]:
        """Match detections to existing tracks; return id→box mapping."""

        for t in self._tracks.values():
            t["is_new"] = False

        if not detections:
            lost = [tid for tid, t in self._tracks.items()
                    if self._tick(t)]
            for tid in lost:
                del self._tracks[tid]
            return {tid: t["box"] for tid, t in self._tracks.items()}

        if not self._tracks:
            for box in detections:
                self._register(box)
            return {tid: t["box"] for tid, t in self._tracks.items()}

        # Build IoU matrix
        ids    = list(self._tracks.keys())
        tboxes = [self._tracks[i]["box"] for i in ids]

        iou_mat = np.zeros((len(tboxes), len(detections)), dtype=np.float32)
        for ti, tb in enumerate(tboxes):
            for di, db in enumerate(detections):
                iou_mat[ti, di] = self._iou(tb, db)

        # Greedy matching (highest IoU first)
        matched_t, matched_d = set(), set()
        pairs = sorted(
            [(iou_mat[ti, di], ti, di)
             for ti in range(len(ids))
             for di in range(len(detections))],
            key=lambda x: -x[0],
        )
        for iou_val, ti, di in pairs:
            if iou_val < self._iou_thr:
                break
            if ti in matched_t or di in matched_d:
                continue
            tid = ids[ti]
            self._tracks[tid]["box"]         = detections[di]
            self._tracks[tid]["disappeared"] = 0
            matched_t.add(ti); matched_d.add(di)

        # Handle unmatched tracks
        lost = []
        for ti, tid in enumerate(ids):
            if ti not in matched_t and self._tick(self._tracks[tid]):
                lost.append(tid)
        for tid in lost:
            del self._tracks[tid]

        # Register new detections
        for di, box in enumerate(detections):
            if di not in matched_d:
                self._register(box)

        return {tid: t["box"] for tid, t in self._tracks.items()}

    def is_new(self, tid: int) -> bool:
        t = self._tracks.get(tid)
        return t["is_new"] if t else False

    def _register(self, box: tuple) -> None:
        self._tracks[self._next_id] = {"box": box, "disappeared": 0, "is_new": True}
        self._next_id += 1

    def _tick(self, track: dict) -> bool:
        track["disappeared"] += 1
        return track["disappeared"] > self._max_dis

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        at, ar, ab_, al = a
        bt, br, bb_, bl = b
        iw = max(0, min(ar, br) - max(al, bl))
        ih = max(0, min(ab_, bb_) - max(at, bt))
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = (ar-al)*(ab_-at) + (br-bl)*(bb_-bt) - inter
        return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_faces_in_frame(
    bgr_frame: np.ndarray,
    scale: float = DETECT_SCALE,
    upsample: int = UPSAMPLE,
) -> list[tuple[int, int, int, int]]:
    """Detect ALL faces in *bgr_frame*; return full-resolution bounding boxes.

    Args:
        bgr_frame: OpenCV BGR frame.
        scale:     Downscale factor for detection speed.
        upsample:  HOG upsample passes (1=fast, 2=catches smaller faces).

    Returns:
        List of (top, right, bottom, left) in full-resolution pixels.
    """
    h, w = bgr_frame.shape[:2]
    det_w, det_h = int(w * scale), int(h * scale)

    small    = cv2.resize(bgr_frame, (det_w, det_h))
    rgb_small = np.ascontiguousarray(
        cv2.cvtColor(small, cv2.COLOR_BGR2RGB), dtype=np.uint8
    )

    locs_small = _fr.face_locations(rgb_small, number_of_times_to_upsample=upsample, model="hog")

    del small, rgb_small  # free immediately

    inv = 1.0 / scale
    return [(int(t*inv), int(r*inv), int(b*inv), int(l*inv))
            for t, r, b, l in locs_small]


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_face(frame: np.ndarray, tid: int, box: tuple, is_new: bool) -> None:
    """Draw bounding box + label for one tracked face."""
    top, right, bottom, left = box
    fw = right  - left
    fh = bottom - top

    is_far = (fw < MIN_FACE_PX or fh < MIN_FACE_PX)
    color  = COLOR_FAR if is_far else (COLOR_NEW if is_new else COLOR_TRACKED)

    # Box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Label chip above box
    label = f"Face #{tid}"
    (lw, lh), base = cv2.getTextSize(label, FONT, 0.55, 1)
    ly = max(top - 6, lh + 4)
    cv2.rectangle(frame, (left, ly - lh - 4), (left + lw + 4, ly + base), color, -1)
    cv2.putText(frame, label, (left+2, ly-2), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Size info
    cv2.putText(frame, f"{fw}x{fh}px",
                (left, bottom + 16), FONT, 0.42, color, 1, cv2.LINE_AA)
    if is_far:
        cv2.putText(frame, "LOW QUALITY / FAR",
                    (left, bottom + 32), FONT, 0.38, COLOR_FAR, 1, cv2.LINE_AA)


def draw_status_bar(frame: np.ndarray, face_count: int,
                    fps: float, ram: float, mode: str) -> None:
    """Draw dark status bar at the top of *frame*."""
    fh, fw = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (fw, 38), COLOR_BG, -1)
    cv2.putText(frame, "SMART CLASSROOM  Multi-Face Detection",
                (8, 26), FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    stats = (f"Faces: {face_count}   FPS: {fps:4.1f}   RAM: {ram:.0f} MB   [{mode}]"
             if fps > 0 else
             f"Faces: {face_count}   RAM: {ram:.0f} MB   [{mode}]")
    (sw, _), _ = cv2.getTextSize(stats, FONT, 0.48, 1)
    cv2.putText(frame, stats, (fw - sw - 8, 26),
                FONT, 0.48, (0, 230, 180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 2 — STATIC IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def run_image_mode(image_path: str) -> None:
    """Detect all faces in a classroom photo and display the result."""

    print()
    print("=" * 60)
    print("  SMART CLASSROOM — IMAGE DETECTION TEST")
    print("=" * 60)
    print(f"  Image     : {image_path}")
    print(f"  Detector  : dlib HOG  (upsample={UPSAMPLE})")
    print(f"  RAM start : {_ram_mb():.1f} MB")
    print()

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load image: {image_path}")
        print("        Check the path and try again.")
        sys.exit(1)

    orig_h, orig_w = frame.shape[:2]
    print(f"[INFO]  Image size   : {orig_w} x {orig_h} px")

    # Detect faces (try with current UPSAMPLE first)
    print(f"[INFO]  Detecting faces (scale={DETECT_SCALE}, upsample={UPSAMPLE}) ...")
    t0 = time.perf_counter()
    locations = detect_faces_in_frame(frame, scale=DETECT_SCALE, upsample=UPSAMPLE)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"[INFO]  Detection time: {elapsed_ms:.0f} ms")
    print(f"[INFO]  Faces found   : {len(locations)}")

    # If no faces found, retry with upsample=2
    if not locations and UPSAMPLE < 2:
        print("[INFO]  Retrying with upsample=2 (detects smaller faces) ...")
        t0 = time.perf_counter()
        locations = detect_faces_in_frame(frame, scale=DETECT_SCALE, upsample=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[INFO]  Detection time: {elapsed_ms:.0f} ms")
        print(f"[INFO]  Faces found   : {len(locations)}")

    if not locations:
        print()
        print("[WARN]  No faces detected in this image.")
        print("        Possible reasons:")
        print("        1. Image is a cartoon/illustration (HOG is trained on real faces)")
        print("        2. Faces are too small — try with a higher-resolution photo")
        print("        3. Poor lighting or unusual angles in the photo")
        print("        4. Try a real classroom photograph instead")
        print()
        print("        For LIVE CAMERA test: run without --image flag")
    else:
        # Draw all detected faces with IDs
        tracker = IouTracker()
        active  = tracker.update(locations)

        print()
        print(f"  {'ID':<8} {'Top':>5} {'Right':>5} {'Bottom':>5} {'Left':>5} {'W':>5} {'H':>5}  Quality")
        print(f"  {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}  -------")
        for tid, box in sorted(active.items()):
            top, right, bottom, left = box
            fw = right  - left
            fh = bottom - top
            quality = "LOW QUALITY / FAR" if (fw < MIN_FACE_PX or fh < MIN_FACE_PX) else "OK"
            print(f"  Face #{tid:<3} {top:>5} {right:>5} {bottom:>5} {left:>5} {fw:>5} {fh:>5}  {quality}")
            draw_face(frame, tid, box, is_new=True)

        draw_status_bar(frame, len(active), fps=0, ram=_ram_mb(), mode="IMAGE")

    print()
    print(f"  RAM after detection: {_ram_mb():.1f} MB")
    print()
    print("[INFO]  Showing result window — press any key to close ...")

    # Show result
    # Scale down if image is larger than screen
    disp = frame.copy()
    max_disp = 900
    if orig_w > max_disp or orig_h > max_disp:
        scale_d = min(max_disp / orig_w, max_disp / orig_h)
        disp    = cv2.resize(disp, (int(orig_w * scale_d), int(orig_h * scale_d)))

    cv2.imshow(WIN_IMG, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("[INFO]  Window closed.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 1 — LIVE CAMERA
# ─────────────────────────────────────────────────────────────────────────────

def run_camera_mode() -> None:
    """Continuously detect and track all faces from the laptop camera."""

    print()
    print("=" * 60)
    print("  SMART CLASSROOM — LIVE MULTI-FACE TRACKING")
    print("=" * 60)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  OpenCV      : {cv2.__version__}")
    print(f"  Detector    : dlib HOG  (upsample={UPSAMPLE})")
    print(f"  Camera      : index {CAMERA_INDEX}  ({FRAME_W}x{FRAME_H})")
    print(f"  Scale       : {DETECT_SCALE}x  →  detect on {int(FRAME_W*DETECT_SCALE)}x{int(FRAME_H*DETECT_SCALE)}")
    print(f"  Min face px : {MIN_FACE_PX}  (smaller → LOW QUALITY / FAR)")
    print(f"  RAM start   : {_ram_mb():.1f} MB")
    print()

    print(f"[INFO] Opening camera index {CAMERA_INDEX} ...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {actual_w}x{actual_h}")
    print("[INFO] Press Q or ESC to quit.")
    print()

    tracker      = IouTracker()
    fps_prev     = time.perf_counter()
    fps_value    = 0.0
    frame_count  = 0
    max_faces    = 0
    ram_cached   = _ram_mb()
    last_locations: list = []   # reuse detections between detection frames

    print(f"[INFO] Detection every {DETECT_EVERY_N} frame(s), tracking every frame.")
    print(f"[INFO] Scale={DETECT_SCALE}  Upsample={UPSAMPLE}  MinFacePx={MIN_FACE_PX}")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1

            # FPS
            now       = time.perf_counter()
            fps_value = 1.0 / max(now - fps_prev, 1e-6)
            fps_prev  = now

            # RAM (sampled every 30 frames)
            if frame_count % 30 == 0:
                ram_cached = _ram_mb()

            # ── Detection (runs every DETECT_EVERY_N frames) ──────────────────
            # Between detection frames we reuse last_locations so tracking
            # stays smooth and FPS remains high.
            if frame_count % DETECT_EVERY_N == 0:
                last_locations = detect_faces_in_frame(
                    frame, scale=DETECT_SCALE, upsample=UPSAMPLE
                )

            locations = last_locations

            # Track
            active    = tracker.update(locations)
            n_faces   = len(active)
            max_faces = max(max_faces, n_faces)

            # Draw
            for tid, box in active.items():
                draw_face(frame, tid, box, is_new=tracker.is_new(tid))

            draw_status_bar(frame, n_faces, fps_value, ram_cached, mode="LIVE")

            cv2.imshow(WIN_LIVE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[INFO] Quit key pressed.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Summary
    print()
    print("=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Frames processed      : {frame_count}")
    print(f"  Max simultaneous faces: {max_faces}")
    print(f"  Unique faces tracked  : {tracker._next_id - 1}")
    print(f"  Final RAM             : {_ram_mb():.1f} MB")
    print(f"  Camera released       : YES")
    print("=" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Classroom — Multi-Face Detection + Tracking Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Live camera:
    python tests/multi_face_tracking_test.py

  Static classroom image:
    python tests/multi_face_tracking_test.py --image uploads/classroom.jpg
        """,
    )
    parser.add_argument(
        "--image", "-i",
        metavar="PATH",
        default=None,
        help="Path to a classroom photo. Omit to use live camera.",
    )
    args = parser.parse_args()

    if args.image:
        run_image_mode(args.image)
    else:
        run_camera_mode()


if __name__ == "__main__":
    main()
