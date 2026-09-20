"""
tests/multi_face_recognition_test.py
──────────────────────────────────────
SMART CLASSROOM — Phase 3: Multi-Face Recognition Test

PURPOSE
-------
Detect ALL faces in the classroom, track them with stable IDs, and
attempt to identify each face against the registered student database.

PIPELINE
--------
Camera Frame
    ↓
HOG Face Detection  (every DETECT_EVERY_N frames for speed)
    ↓
IoU Tracker         (every frame — smooth display)
    ↓
Per tracked face:
    ├── Too small?  → label LOW QUALITY / FAR, skip recognition
    ├── Too blurry? → label BLURRY, skip recognition
    ├── Recently recognised? → show cached result
    └── Due for recognition?
            ↓
        Crop face ROI → generate 128-D encoding (dlib)
            ↓
        Compare vs every registered student encoding
            ├── Best distance < threshold → RECOGNISED: Name (Reg#)
            └── All distances ≥ threshold → UNKNOWN

WHAT THIS TEST DOES NOT DO
--------------------------
- Does NOT mark attendance (that is a later phase)
- Does NOT modify the database
- Does NOT connect to Spring Boot
- Does NOT connect to the FastAPI service

USAGE
-----
Make sure students are registered first via the FastAPI service:
    POST http://127.0.0.1:8000/register-face

Then run (from project root, with .venv active):
    python tests/multi_face_recognition_test.py

KEYBOARD CONTROLS
-----------------
    Q / ESC   — quit cleanly

CONFIGURATION (edit constants below)
--------------------------------------
    DETECT_SCALE      Fraction of frame used for HOG detection
    UPSAMPLE          HOG upsampling passes (1=fast, 2=catches smaller faces)
    DETECT_EVERY_N    Run detection every N frames (track every frame)
    RECOG_INTERVAL    Seconds between recognition attempts per tracked face
    RECOG_THRESHOLD   Max L2 distance to count as a match (lower = stricter)
    QUALITY_MIN_PX    Skip recognition if face is smaller than this
    BLUR_THRESHOLD    Skip recognition if Laplacian variance below this
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Ensure project root is on sys.path ───────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

# ── face_recognition (dlib) ───────────────────────────────────────────────────
try:
    import face_recognition as _fr
    _FR_OK = True
except ImportError:
    print("[ERROR] face_recognition not installed. Activate .venv.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX   = 0
FRAME_W        = 640
FRAME_H        = 480

DETECT_SCALE   = 1.0     # 1.0 = full resolution (detects all students including far ones)
UPSAMPLE       = 1       # 1 = fast, fewer false positives. Use 2 only if students are very far.
DETECT_EVERY_N = 3       # run detection every 3rd frame; track every frame

RECOG_INTERVAL  = 3.0    # seconds between recognition attempts per tracked face
RECOG_THRESHOLD = 0.50   # L2 distance — lower is stricter

QUALITY_MIN_PX   = 60    # skip RECOGNITION if face is below this size
BLUR_THRESHOLD   = 40.0  # Laplacian variance — below this = too blurry

# Tracking size limits — applied BEFORE the tracker sees a box
# Boxes outside this range are almost certainly false positives
MIN_TRACK_PX  = 40       # discard boxes smaller than 40x40 (background noise)
MAX_TRACK_PX  = 380      # discard boxes larger than 380x380 (poster / background)

MAX_DISAPPEARED = 20     # frames before a lost track is dropped (~0.7 sec at 30fps)
IOU_THRESHOLD   = 0.30   # min IoU overlap to match detection → existing track

COLOR_RECOGNISED  = (0,   220,   0)    # green  — confirmed this session (first time)
COLOR_PRESENT     = (200, 180,   0)    # teal   — already confirmed earlier
COLOR_UNKNOWN     = (0,     0, 220)    # red    — unrecognised
COLOR_LOW_QUAL    = (0,   165, 255)    # amber  — too small / blurry
COLOR_PENDING     = (180, 180,   0)    # yellow — recognition pending
COLOR_TEXT       = (255, 255, 255)
COLOR_BG         = (30,   30,  30)

FONT     = cv2.FONT_HERSHEY_SIMPLEX
WIN_LIVE = "Smart Classroom — Face Recognition  (Q/ESC to quit)"


# ─────────────────────────────────────────────────────────────────────────────
#  IoU TRACKER  (copied from Phase 2 — pure Python, no extra deps)
# ─────────────────────────────────────────────────────────────────────────────

class IouTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED,
                 iou_threshold=IOU_THRESHOLD):
        self._next_id = 1
        self._tracks: dict[int, dict] = {}
        self._max_dis = max_disappeared
        self._iou_thr = iou_threshold

    def update(self, detections: list[tuple]) -> dict[int, tuple]:
        for t in self._tracks.values():
            t["is_new"] = False

        if not detections:
            lost = [tid for tid, t in self._tracks.items() if self._tick(t)]
            for tid in lost:
                del self._tracks[tid]
            return {tid: t["box"] for tid, t in self._tracks.items()}

        if not self._tracks:
            for box in detections:
                self._register(box)
            return {tid: t["box"] for tid, t in self._tracks.items()}

        ids    = list(self._tracks.keys())
        tboxes = [self._tracks[i]["box"] for i in ids]
        iou_mat = np.zeros((len(tboxes), len(detections)), dtype=np.float32)
        for ti, tb in enumerate(tboxes):
            for di, db in enumerate(detections):
                iou_mat[ti, di] = self._iou(tb, db)

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

        lost = []
        for ti, tid in enumerate(ids):
            if ti not in matched_t and self._tick(self._tracks[tid]):
                lost.append(tid)
        for tid in lost:
            del self._tracks[tid]

        for di, box in enumerate(detections):
            if di not in matched_d:
                self._register(box)

        return {tid: t["box"] for tid, t in self._tracks.items()}

    def is_new(self, tid: int) -> bool:
        t = self._tracks.get(tid)
        return t["is_new"] if t else False

    def _register(self, box):
        self._tracks[self._next_id] = {"box": box, "disappeared": 0, "is_new": True}
        self._next_id += 1

    def _tick(self, track: dict) -> bool:
        track["disappeared"] += 1
        return track["disappeared"] > self._max_dis

    @staticmethod
    def _iou(a, b) -> float:
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
#  RECOGNITION CACHE — one entry per tracked face
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackResult:
    """Cached recognition result for one tracked face."""
    status: str          = "PENDING"    # PENDING | LOW_QUALITY | BLURRY | RECOGNISED | UNKNOWN
    student_id: str      = ""
    student_name: str    = ""
    department: str      = ""
    semester: str        = ""
    distance: float      = 1.0
    last_recog_time: float = 0.0        # time.perf_counter() of last recognition run


# ─────────────────────────────────────────────────────────────────────────────
#  STUDENT DATABASE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_student_database() -> tuple[dict[str, list[np.ndarray]], dict[str, dict]]:
    """Load all student encodings and metadata from the local database.

    Returns:
        encodings_db  : {student_id → [enc1, enc2, ...]}
        student_info  : {student_id → {name, department, semester, section}}
    """
    # ── Load encodings from pickle files ─────────────────────────────────────
    encodings_db: dict[str, list[np.ndarray]] = {}
    encoding_dir = "encodings"

    if not os.path.isdir(encoding_dir):
        print(f"[WARN] Encoding directory not found: {encoding_dir}")
        return {}, {}

    import pickle
    for fname in os.listdir(encoding_dir):
        if not fname.endswith(".pickle"):
            continue
        student_id = fname.replace(".pickle", "")
        fpath = os.path.join(encoding_dir, fname)
        try:
            with open(fpath, "rb") as f:
                encodings = pickle.load(f)
            if encodings:
                encodings_db[student_id] = encodings
                print(f"  [DB] Loaded {len(encodings):2d} encoding(s) for: {student_id}")
        except Exception as exc:
            print(f"  [WARN] Could not load {fname}: {exc}")

    # ── Load student metadata from SQLite ─────────────────────────────────────
    student_info: dict[str, dict] = {}
    try:
        from database import SessionLocal
        from models.student import Student
        db = SessionLocal()
        try:
            students = db.query(Student).filter(Student.is_active == True).all()  # noqa: E712
            for s in students:
                student_info[s.student_id] = {
                    "name":       s.name,
                    "department": s.department,
                    "semester":   s.semester,
                    "section":    s.section,
                }
        finally:
            db.close()
        print(f"  [DB] Loaded metadata for {len(student_info)} student(s) from SQLite.")
    except Exception as exc:
        print(f"  [WARN] Could not load student metadata from DB: {exc}")
        print("         Names will show as student_id only.")

    return encodings_db, student_info


# ─────────────────────────────────────────────────────────────────────────────
#  NON-MAXIMUM SUPPRESSION  (remove duplicate detections of the same face)
# ─────────────────────────────────────────────────────────────────────────────

def nms(locations: list[tuple], iou_threshold: float = 0.30) -> list[tuple]:
    """Remove overlapping duplicate detections, keeping the largest box.

    dlib HOG with upsample=2 can detect the same face multiple times at
    slightly different bounding-box sizes. NMS merges those into one box.

    Args:
        locations:     List of (top, right, bottom, left) face boxes.
        iou_threshold: Boxes with IoU above this are considered duplicates.
                       0.30 is a good default for face detection.

    Returns:
        Filtered list with duplicate boxes removed.
    """
    if len(locations) <= 1:
        return locations

    def _area(b):
        t, r, bo, l = b
        return max(0, r - l) * max(0, bo - t)

    def _iou(a, b):
        at, ar, ab_, al = a
        bt, br, bb_, bl = b
        iw = max(0, min(ar, br) - max(al, bl))
        ih = max(0, min(ab_, bb_) - max(at, bt))
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = _area(a) + _area(b) - inter
        return inter / union if union > 0 else 0.0

    # Sort largest area first so we always keep the biggest box
    sorted_locs = sorted(locations, key=_area, reverse=True)

    kept       = []
    suppressed = set()

    for i, box_a in enumerate(sorted_locs):
        if i in suppressed:
            continue
        kept.append(box_a)
        for j in range(i + 1, len(sorted_locs)):
            if j in suppressed:
                continue
            if _iou(box_a, sorted_locs[j]) > iou_threshold:
                suppressed.add(j)

    return kept


# ─────────────────────────────────────────────────────────────────────────────
#  FACE QUALITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_face_quality(
    bgr_frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[bool, str]:
    """Check if a detected face is good enough for recognition.

    Args:
        bgr_frame: Full BGR camera frame.
        box:       (top, right, bottom, left) bounding box.

    Returns:
        (is_ok, reason) — is_ok=True means face is suitable for recognition.
        reason is empty string on success, or description on failure.
    """
    top, right, bottom, left = box
    fw = right  - left
    fh = bottom - top

    # Size check
    if fw < QUALITY_MIN_PX or fh < QUALITY_MIN_PX:
        return False, f"LOW QUALITY / FAR ({fw}x{fh}px)"

    # Crop face region safely
    h, w = bgr_frame.shape[:2]
    top_c    = max(0, top)
    bottom_c = min(h, bottom)
    left_c   = max(0, left)
    right_c  = min(w, right)

    if bottom_c <= top_c or right_c <= left_c:
        return False, "INVALID BOX"

    face_crop = bgr_frame[top_c:bottom_c, left_c:right_c]
    if face_crop.size == 0:
        return False, "EMPTY CROP"

    # Blur check (Laplacian variance)
    gray       = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < BLUR_THRESHOLD:
        return False, f"BLURRY (score={blur_score:.0f})"

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
#  RECOGNITION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def recognise_face(
    bgr_frame: np.ndarray,
    box: tuple[int, int, int, int],
    encodings_db: dict[str, list[np.ndarray]],
    student_info: dict[str, dict],
) -> tuple[str, str, str, str, float]:
    """Attempt to recognise one face against the student database.

    Args:
        bgr_frame:   Full BGR frame from the camera.
        box:         (top, right, bottom, left) bounding box.
        encodings_db: {student_id → [encoding, ...]}
        student_info: {student_id → {name, department, ...}}

    Returns:
        (status, student_id, student_name, department, distance)
        status ∈ {"RECOGNISED", "UNKNOWN", "ENCODE_FAIL"}
    """
    top, right, bottom, left = box

    # Convert full frame to RGB (dlib requirement)
    rgb_frame = np.ascontiguousarray(
        cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), dtype=np.uint8
    )

    # Generate encoding — pass the known location to skip re-detection
    try:
        encodings = _fr.face_encodings(
            rgb_frame,
            known_face_locations=[(top, right, bottom, left)],
            num_jitters=1,
            model="small",
        )
    except Exception as exc:
        return "ENCODE_FAIL", "", "", "", 1.0

    if not encodings:
        return "ENCODE_FAIL", "", "", "", 1.0

    live_enc = encodings[0]

    # Compare against all known students
    best_id   = ""
    best_dist = float("inf")

    for sid, known_encs in encodings_db.items():
        if not known_encs:
            continue
        known_arr = np.array(known_encs, dtype=np.float64)
        query_arr = np.array(live_enc,   dtype=np.float64)
        distances = np.linalg.norm(known_arr - query_arr, axis=1)
        min_dist  = float(np.min(distances))
        if min_dist < best_dist:
            best_dist = min_dist
            best_id   = sid

    if best_dist <= RECOG_THRESHOLD:
        info = student_info.get(best_id, {})
        return (
            "RECOGNISED",
            best_id,
            info.get("name", best_id),
            info.get("department", ""),
            best_dist,
        )
    else:
        return "UNKNOWN", "", "", "", best_dist


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_tracked_face(
    frame: np.ndarray,
    tid: int,
    box: tuple,
    result: TrackResult,
) -> None:
    """Draw bounding box and recognition label for one tracked face."""
    top, right, bottom, left = box
    fw = right - left
    fh = bottom - top

    status = result.status

    if status == "RECOGNISED":
        color   = COLOR_RECOGNISED
        line1   = (result.student_name or result.student_id) + "  ✓ COMPLETED"
        line2   = f"Reg: {result.student_id}  d={result.distance:.3f}"
        line3   = result.department
    elif status == "ALREADY_PRESENT":
        color   = COLOR_PRESENT
        line1   = (result.student_name or result.student_id) + "  ✓ Already Present"
        line2   = f"Reg: {result.student_id}"
        line3   = ""
    elif status == "UNKNOWN":
        color   = COLOR_UNKNOWN
        line1   = "UNKNOWN"
        line2   = f"d={result.distance:.3f}"
        line3   = ""
    elif status in ("LOW_QUALITY", "BLURRY", "INVALID_BOX"):
        color   = COLOR_LOW_QUAL
        line1   = f"Face #{tid}"
        line2   = f"{fw}x{fh}px  LOW QUALITY"
        line3   = ""
    else:                       # PENDING
        color   = COLOR_PENDING
        line1   = f"Face #{tid}"
        line2   = "Analysing..."
        line3   = ""

    # Bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Label chip above box
    label = f"Face #{tid}"
    (lw, lh), base = cv2.getTextSize(label, FONT, 0.50, 1)
    ly = max(top - 6, lh + 4)
    cv2.rectangle(frame, (left, ly - lh - 4), (left + lw + 4, ly + base), color, -1)
    cv2.putText(frame, label, (left+2, ly-2), FONT, 0.50, (0,0,0), 1, cv2.LINE_AA)

    # Info lines below box
    y_off = bottom + 16
    for text in [line1, line2, line3]:
        if not text:
            continue
        cv2.putText(frame, text, (left, y_off), FONT, 0.42, color, 1, cv2.LINE_AA)
        y_off += 16


def draw_status_bar(frame, n_faces, session_confirmed_count, fps, ram) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), COLOR_BG, -1)
    cv2.putText(frame, "SMART CLASSROOM  Face Recognition",
                (8, 28), FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    stats = (f"Faces: {n_faces}  Session Attendance: {session_confirmed_count}  "
             f"FPS: {fps:4.1f}  RAM: {ram:.0f} MB")
    (sw, _), _ = cv2.getTextSize(stats, FONT, 0.48, 1)
    cv2.putText(frame, stats, (w - sw - 8, 28),
                FONT, 0.48, (0, 230, 180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 64)
    print("  SMART CLASSROOM — MULTI-FACE RECOGNITION TEST  (Phase 3)")
    print("=" * 64)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  OpenCV      : {cv2.__version__}")
    print(f"  Detector    : dlib HOG  (scale={DETECT_SCALE}, upsample={UPSAMPLE})")
    print(f"  Detect every: {DETECT_EVERY_N} frame(s)")
    print(f"  Recog interval: {RECOG_INTERVAL}s per face")
    print(f"  Threshold   : {RECOG_THRESHOLD}  (L2 distance)")
    print(f"  Min face px : {QUALITY_MIN_PX}")
    print(f"  Blur check  : Laplacian var >= {BLUR_THRESHOLD}")
    print(f"  RAM start   : {_ram_mb():.1f} MB")
    print()

    # ── Load student database ─────────────────────────────────────────────────
    print("[INFO] Loading student database ...")
    encodings_db, student_info = load_student_database()

    if not encodings_db:
        print()
        print("[WARN] No student encodings found in the 'encodings/' directory.")
        print("       Register students first:")
        print("         1. Start the FastAPI service:  .venv\\Scripts\\python -m uvicorn app:app --port 8000")
        print("         2. POST http://127.0.0.1:8000/register-face")
        print("       Then re-run this test.")
        print()
        print("[INFO] Continuing in DETECTION ONLY mode (no recognition).")
        print()
    else:
        total_encs = sum(len(v) for v in encodings_db.values())
        print(f"[INFO] {len(encodings_db)} student(s) loaded, {total_encs} encoding(s) total.")
        print()

    # ── Open camera ───────────────────────────────────────────────────────────
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

    # ── State ──────────────────────────────────────────────────────────────────
    tracker       = IouTracker()
    track_results: dict[int, TrackResult] = {}   # tid → TrackResult
    session_confirmed: set[str] = set()          # student_ids confirmed this session

    fps_prev      = time.perf_counter()
    fps_value     = 0.0
    frame_count   = 0
    max_faces     = 0
    ram_cached    = _ram_mb()
    last_locations: list = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            now         = time.perf_counter()
            fps_value   = 1.0 / max(now - fps_prev, 1e-6)
            fps_prev    = now

            if frame_count % 30 == 0:
                ram_cached = _ram_mb()

            # ── Detection (every DETECT_EVERY_N frames) ───────────────────────
            if frame_count % DETECT_EVERY_N == 0:
                det_w = int(actual_w * DETECT_SCALE)
                det_h = int(actual_h * DETECT_SCALE)
                small    = cv2.resize(frame, (det_w, det_h)) if DETECT_SCALE < 1.0 else frame
                rgb_s    = np.ascontiguousarray(
                    cv2.cvtColor(small, cv2.COLOR_BGR2RGB), dtype=np.uint8
                )
                locs_s   = _fr.face_locations(rgb_s, number_of_times_to_upsample=UPSAMPLE, model="hog")
                if DETECT_SCALE < 1.0:
                    inv = 1.0 / DETECT_SCALE
                    last_locations = [(int(t*inv), int(r*inv), int(b*inv), int(l*inv))
                                      for t, r, b, l in locs_s]
                    del small
                else:
                    last_locations = locs_s
                del rgb_s

                # Apply NMS to remove duplicate detections of the same face
                last_locations = nms(last_locations, iou_threshold=0.30)

                # ── Size filter: discard before tracker sees them ──────────────
                # Removes tiny false positives (background noise) and
                # oversized false positives (posters, text on walls).
                last_locations = [
                    box for box in last_locations
                    if MIN_TRACK_PX <= (box[1]-box[3]) <= MAX_TRACK_PX
                    and MIN_TRACK_PX <= (box[2]-box[0]) <= MAX_TRACK_PX
                ]

            # ── Tracker update ────────────────────────────────────────────────
            active    = tracker.update(last_locations)
            n_faces   = len(active)
            max_faces = max(max_faces, n_faces)

            # Remove stale TrackResults for disappeared tracks
            for tid in list(track_results.keys()):
                if tid not in active:
                    del track_results[tid]

            # ── Recognition per tracked face ──────────────────────────────────
            n_recognised = 0
            for tid, box in active.items():

                # Initialise result for new track
                if tid not in track_results:
                    track_results[tid] = TrackResult()

                result = track_results[tid]

                # Count recognised faces for status bar
                if result.status == "RECOGNISED":
                    n_recognised += 1
                    session_confirmed.add(result.student_id)

                # Skip recognition if no students registered
                if not encodings_db:
                    result.status = "PENDING"
                    continue

                # ── RECOGNISED / ALREADY_PRESENT faces are LOCKED ─────────────
                if result.status in ("RECOGNISED", "ALREADY_PRESENT"):
                    continue

                # Check cooldown for UNKNOWN / LOW_QUALITY / PENDING faces
                elapsed = now - result.last_recog_time
                if elapsed < RECOG_INTERVAL and result.status not in ("PENDING",):
                    continue

                # Quality check
                ok, reason = check_face_quality(frame, box)
                if not ok:
                    result.status = "LOW_QUALITY"
                    continue

                # Run recognition
                status, sid, name, dept, dist = recognise_face(
                    frame, box, encodings_db, student_info
                )

                if status == "RECOGNISED":
                    if sid in session_confirmed:
                        # Student already confirmed earlier this session
                        result.status       = "ALREADY_PRESENT"
                        result.student_id   = sid
                        result.student_name = name
                        result.department   = dept
                        result.distance     = dist
                    else:
                        # First time confirmed this session
                        result.status       = "RECOGNISED"
                        result.student_id   = sid
                        result.student_name = name
                        result.department   = dept
                        result.distance     = dist
                        session_confirmed.add(sid)
                else:
                    result.status      = status
                    result.student_id  = sid
                    result.student_name = name
                    result.department  = dept
                    result.distance    = dist

                result.last_recog_time = now

            # ── Draw ──────────────────────────────────────────────────────────
            for tid, box in active.items():
                result = track_results.get(tid, TrackResult())
                draw_tracked_face(frame, tid, box, result)

            draw_status_bar(frame, n_faces, len(session_confirmed), fps_value, ram_cached)
            cv2.imshow(WIN_LIVE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[INFO] Quit key pressed.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  SESSION ATTENDANCE SUMMARY")
    print("=" * 64)
    print(f"  Frames processed          : {frame_count}")
    print(f"  Max simultaneous faces    : {max_faces}")
    print(f"  Total students confirmed  : {len(session_confirmed)}")
    print()
    if session_confirmed:
        print("  Confirmed students this session:")
        for sid in sorted(session_confirmed):
            info = student_info.get(sid, {})
            name = info.get('name', sid)
            dept = info.get('department', '')
            sem  = info.get('semester', '')
            print(f"    ✓  {sid:20s}  {name:25s}  {dept} Sem-{sem}")
    else:
        print("  No students confirmed this session.")
    print()
    print(f"  Final RAM                 : {_ram_mb():.1f} MB")
    print(f"  Camera released           : YES")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
